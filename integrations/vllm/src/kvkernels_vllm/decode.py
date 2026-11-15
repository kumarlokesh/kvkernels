"""Decode attention on vLLM's paged KV cache through kvkernels' cascade.

Nothing here imports vLLM, so it runs under the Triton interpreter in tests.

vLLM keeps one tensor per layer with K and V of a head side by side:

    kv_cache   [blocks, kv_heads, block_size, 2 · head_size]
                                                └─ K values ─┘└─ V values ─┘
    transpose(1, 2), split the last axis in two:
    k, v       [blocks, block_size, kv_heads, head_size] each, views of the same memory

which is the view kvkernels' attention reads, so no copy is made.

The batch's single-token decodes come first. Planning runs once per step from
their rows of vLLM's block table and sequence lengths; every layer then reuses
the plan.
"""

from __future__ import annotations

import math

import torch

from kvkernels.ops.cascade import CascadePlan, cascade_decode_attention
from kvkernels.ops.cascade_plan import MAX_PLAN_REQUESTS
from kvkernels.reference.attention import RequestKV, attention_ref32, attention_ref_low
from kvkernels.tolerance import assert_close

HEAD_SIZES = (64, 128)
MAX_GQA_GROUP = 16
KV_DTYPES = ("auto", "float16", "bfloat16")
NUM_BLOCKS_UNCHECKED = 2**31 - 1  # the host plan's block-id check is never used here


def kv_views(kv_cache: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V of one layer as [blocks, block_size, kv_heads, head_size] views."""
    kv = kv_cache.transpose(1, 2)
    return kv[..., :head_size], kv[..., head_size : 2 * head_size]


def unsupported(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    kv_cache_dtype: str,
    sliding_window: bool,
    alibi: bool,
    soft_cap: bool,
    sinks: bool,
    decoder: bool,
) -> list[str]:
    """Reasons the cascade cannot serve a layer; empty when it can."""
    reasons = []
    if head_size not in HEAD_SIZES:
        reasons.append(f"head size {head_size} (supported: {HEAD_SIZES})")
    if num_heads % num_kv_heads or num_heads // num_kv_heads > MAX_GQA_GROUP:
        reasons.append(f"{num_heads} query heads over {num_kv_heads} KV heads")
    if kv_cache_dtype not in KV_DTYPES:
        reasons.append(f"KV cache dtype {kv_cache_dtype}")
    for flag, name in (
        (sliding_window, "sliding window"),
        (alibi, "ALiBi"),
        (soft_cap, "logit soft cap"),
        (sinks, "attention sinks"),
        (not decoder, "attention other than decoder self-attention"),
    ):
        if flag:
            reasons.append(name)
    return reasons


def new_plan(
    *,
    max_requests: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    batch_invariant: bool = False,
    min_shared_tokens: int | None = None,
) -> CascadePlan:
    """A plan sized for vLLM's maximum number of sequences, filled each step on the GPU."""
    if max_requests > MAX_PLAN_REQUESTS:
        raise ValueError(f"max_num_seqs {max_requests} exceeds {MAX_PLAN_REQUESTS}")
    return CascadePlan(
        max_requests=max_requests,
        max_groups=max(max_requests // 2, 1),
        max_blocks=1,  # holds the host plan's block lists, which this plan never uses
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=NUM_BLOCKS_UNCHECKED,
        device=device,
        min_shared_tokens=min_shared_tokens,
        batch_invariant=batch_invariant,
    )


def decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    plan: CascadePlan,
    out: torch.Tensor,
    *,
    head_size: int,
    scale: float,
    lse_out: torch.Tensor | None = None,
) -> None:
    """Attention for the decode rows the plan was filled for.

    query, out: [rows, heads, head_size]; lse_out, if given: [rows, heads], fp32.
    """
    k, v = kv_views(kv_cache, head_size)
    cascade_decode_attention(query, k, v, plan, out, lse_out, sm_scale=scale)


def check_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    engine_out: torch.Tensor,
    *,
    head_size: int,
    scale: float,
) -> None:
    """Fail unless the cascade is as close to fp32 as vLLM's own kernel, within the tolerance rule.

    engine_out is vLLM's Triton kernel on the same rows; it serves as the
    low-precision reference for out. The LSE is checked against the fp32 and
    low-precision references of kvkernels.reference. Copies the batch's block
    table to the host, so it runs only in eager mode.
    """
    k, v = kv_views(kv_cache, head_size)
    block_size = k.shape[1]
    lens = seq_lens.tolist()
    rows = block_table.tolist()
    live = [r for r, n in enumerate(lens) if n > 0]
    if not live:
        return
    requests = [RequestKV([], 0, rows[r][: math.ceil(lens[r] / block_size)], lens[r]) for r in live]
    q = query[live]
    out32, lse32 = attention_ref32(q, k, v, requests, scale)
    _, lse_low = attention_ref_low(q, k, v, requests, scale)
    assert_close(out[live], engine_out[live], out32)
    assert_close(lse[live], lse_low, lse32)
