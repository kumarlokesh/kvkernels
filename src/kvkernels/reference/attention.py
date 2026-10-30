"""Pure-PyTorch references for decode attention over a paged cache, and for merging.

Decode: one query token per request. A request's keys are its shared prefix's
tokens followed by its own suffix's tokens, each a list of KV blocks and a
length. K and V are the per-layer views of kvkernels.layouts: [blocks, slots,
kv_heads, head_dim]. Query head h reads KV head h // group.

    attention_ref32     the answer to compare against: fp64 arithmetic, returned as fp32
    attention_ref_low   out from PyTorch's SDPA in the input dtype; lse from scores
                        rounded to the input dtype
    merge_states_ref    combine partial results (o, lse) of disjoint key sets

LSE is the natural log of the sum of exponentiated scores; an empty key set has
lse = -inf and output 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RequestKV:
    """Where one request's keys are: its prefix's blocks and length, then its suffix's."""

    prefix_blocks: Sequence[int]
    prefix_len: int
    suffix_blocks: Sequence[int]
    suffix_len: int


def gather_tokens(view: torch.Tensor, blocks: Sequence[int], length: int) -> torch.Tensor:
    """The first `length` tokens of a block list: [length, kv_heads, head_dim]."""
    if length == 0:
        return view.new_empty((0, *view.shape[2:]))
    ids = torch.tensor(list(blocks), dtype=torch.long, device=view.device)
    return view[ids].flatten(0, 1)[:length]


def request_kv(
    k: torch.Tensor, v: torch.Tensor, req: RequestKV
) -> tuple[torch.Tensor, torch.Tensor]:
    """The request's keys and values, prefix then suffix: each [tokens, kv_heads, head_dim]."""
    keys = torch.cat(
        [
            gather_tokens(k, req.prefix_blocks, req.prefix_len),
            gather_tokens(k, req.suffix_blocks, req.suffix_len),
        ]
    )
    values = torch.cat(
        [
            gather_tokens(v, req.prefix_blocks, req.prefix_len),
            gather_tokens(v, req.suffix_blocks, req.suffix_len),
        ]
    )
    return keys, values


def _scale(head_dim: int, sm_scale: float | None) -> float:
    return sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)


def attention_ref32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    requests: Sequence[RequestKV],
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, lse) per request and query head, computed in fp64 and returned as fp32.

    q: [requests, q_heads, head_dim]. out: like q in fp32; lse: [requests, q_heads].
    """
    num_req, q_heads, head_dim = q.shape
    group = q_heads // k.shape[2]
    scale = _scale(head_dim, sm_scale)
    out = torch.zeros((num_req, q_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.full((num_req, q_heads), float("-inf"), dtype=torch.float32, device=q.device)
    for r, req in enumerate(requests):
        keys, values = request_kv(k, v, req)
        if keys.shape[0] == 0:
            continue
        kd = keys.double().repeat_interleave(group, dim=1)  # [tokens, q_heads, d]
        vd = values.double().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", q[r].double(), kd) * scale
        row_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - row_lse[:, None])
        out[r] = torch.einsum("ht,thd->hd", probs, vd).float()
        lse[r] = row_lse.float()
    return out, lse


def attention_ref_low(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    requests: Sequence[RequestKV],
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(out in q's dtype from SDPA, lse in fp32 from scores rounded to q's dtype)."""
    num_req, q_heads, head_dim = q.shape
    group = q_heads // k.shape[2]
    scale = _scale(head_dim, sm_scale)
    out = torch.zeros_like(q)
    lse = torch.full((num_req, q_heads), float("-inf"), dtype=torch.float32, device=q.device)
    for r, req in enumerate(requests):
        keys, values = request_kv(k, v, req)
        if keys.shape[0] == 0:
            continue
        kt = keys.transpose(0, 1)[None]  # [1, kv_heads, tokens, d]
        vt = values.transpose(0, 1)[None]
        qr = q[r][None, :, None, :]  # [1, q_heads, 1, d]
        out[r] = F.scaled_dot_product_attention(qr, kt, vt, scale=scale, enable_gqa=group > 1)[
            0, :, 0
        ]
        kr = keys.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", q[r], kr).float() * scale
        lse[r] = torch.logsumexp(scores, dim=-1)
    return out, lse


def merge_states_ref(o: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge partial results over their first dimension, in fp32.

    o: [splits, ..., head_dim]; lse: [splits, ...]. Returns (o, lse) without the
    split dimension. Splits with lse = -inf contribute nothing; if every split
    is empty, the output is 0 and lse is -inf.
    """
    lse32 = lse.float()
    top = lse32.max(dim=0).values
    empty = torch.isneginf(top)
    safe_top = torch.where(empty, torch.zeros_like(top), top)
    weights = torch.exp(lse32 - safe_top[None])  # exp(-inf) = 0 for empty splits
    total = weights.sum(dim=0)
    merged = (weights[..., None] * o.float()).sum(dim=0) / torch.where(
        empty, torch.ones_like(total), total
    )[..., None]
    merged = torch.where(empty[..., None], torch.zeros_like(merged), merged)
    merged_lse = torch.where(empty, top, safe_top + torch.log(total))
    return merged, merged_lse
