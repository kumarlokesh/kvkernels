"""Run vLLM's Triton attention kernel for one decode step, outside vLLM.

The copied kernel file imports vLLM's settings, logger, platform, Triton
re-export, KV quantisation modes and attention helpers; stand_ins.py supplies
them. This adapter builds the per-step inputs the way vLLM's Triton backend does
for a decode batch (vllm/v1/attention/backends/triton_attn.py at the same
commit):

    block table      [requests, most KV blocks of any request], int32; unused entries 0
    sequence lengths [requests], int32: each request's prefix and suffix tokens
    query starts     0, 1, …, requests: one query token per request
    scratch          per-segment output, maximum and sum for the split-softmax path

vLLM picks between two launches of the same kernel. With at most
128 // kv_heads requests it splits each sequence into 16 segments (a third grid
dimension) and reduces them in a second launch; with more it runs one program
per (request, KV head). The constants are vLLM's MIN_LAUNCH_GRID_SIZE_2D and
NUM_PAR_SOFTMAX_SEGMENTS. vLLM moves the threshold to the nearest CUDA graph
capture size when graphs are on; for 8 KV heads that is 16 either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton

from bench.baselines.vllm import stand_ins

MIN_LAUNCH_GRID_SIZE_2D = 128
NUM_PAR_SOFTMAX_SEGMENTS = 16


@dataclass(frozen=True)
class DecodeStep:
    """What vLLM's backend prepares on the host once per step, as device tensors."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_starts: torch.Tensor
    max_seq_len: int
    k_descale: torch.Tensor
    v_descale: torch.Tensor
    seq_threshold_3d: int
    segm_output: torch.Tensor
    segm_max: torch.Tensor
    segm_expsum: torch.Tensor


def plan_decode(
    block_lists: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> DecodeStep:
    """The step's inputs for requests whose tokens live in block_lists, in order."""
    requests = len(block_lists)
    width = max((len(b) for b in block_lists), default=1) or 1
    table = torch.zeros((requests, width), dtype=torch.int32)
    for r, blocks in enumerate(block_lists):
        table[r, : len(blocks)] = torch.tensor(list(blocks), dtype=torch.int32)
    threshold = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
    rows = max(threshold, 1)
    padded = int(triton.next_power_of_2(head_dim))
    segments = NUM_PAR_SOFTMAX_SEGMENTS
    # vLLM's layer scales are 1.0 for an unquantised cache, expanded per (request, KV head).
    one = torch.ones(1, dtype=torch.float32, device=device)
    return DecodeStep(
        block_table=table.to(device),
        seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32, device=device),
        query_starts=torch.arange(requests + 1, dtype=torch.int32, device=device),
        max_seq_len=max(seq_lens, default=0),
        k_descale=one.expand(requests, num_kv_heads),
        v_descale=one.expand(requests, num_kv_heads),
        seq_threshold_3d=threshold,
        segm_output=torch.empty(
            (rows, num_q_heads, segments, padded), dtype=torch.float32, device=device
        ),
        segm_max=torch.empty((rows, num_q_heads, segments), dtype=torch.float32, device=device),
        segm_expsum=torch.empty((rows, num_q_heads, segments), dtype=torch.float32, device=device),
    )


def decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    step: DecodeStep,
    sm_scale: float,
) -> None:
    """Decode attention with vLLM's kernel. k and v: [blocks, slots, kv_heads, head_dim] views."""
    stand_ins.install()
    from bench.baselines.vllm.triton_unified_attention import unified_attention

    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=step.query_starts,
        max_seqlen_q=1,
        seqused_k=step.seq_lens,
        max_seqlen_k=step.max_seq_len,
        softmax_scale=sm_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=step.block_table,
        softcap=0,
        q_descale=None,
        k_descale=step.k_descale,
        v_descale=step.v_descale,
        seq_threshold_3D=step.seq_threshold_3d,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=step.segm_output,
        softmax_segm_max=step.segm_max,
        softmax_segm_expsum=step.segm_expsum,
    )
