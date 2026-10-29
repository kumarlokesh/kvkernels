"""Pure-PyTorch references for attention forward with an LSE output.

Shapes: q [batch, q_heads, seq_q, head_dim]; k, v [batch, kv_heads, seq_k, head_dim];
q_heads is a multiple of kv_heads, and query head h reads KV head h // group.
Outputs: out like q; lse [batch, q_heads, seq_q] in fp32, lse = log Σ exp(scores).

    attention_ref32      the answer to compare against: fp64 arithmetic, returned as fp32
    attention_ref_low    out from PyTorch's SDPA in the input dtype; lse from scores
                         rounded to the input dtype

Causal attention needs seq_q == seq_k: query row i sees keys 0 … i.
Both references work through query rows in chunks, so the score matrix of a long
sequence never has to fit in memory at once.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Largest score chunk materialized at once, in elements.
CHUNK_ELEMENTS = 1 << 26


def repeat_kv(t: torch.Tensor, group: int) -> torch.Tensor:
    """[batch, kv_heads, seq, d] → [batch, kv_heads · group, seq, d].

    Each KV head serves `group` consecutive query heads.
    """
    return t.repeat_interleave(group, dim=1)


def _group(q: torch.Tensor, k: torch.Tensor) -> int:
    if q.shape[1] % k.shape[1]:
        raise ValueError(f"q_heads {q.shape[1]} is not a multiple of kv_heads {k.shape[1]}")
    return q.shape[1] // k.shape[1]


def _chunk_rows(q: torch.Tensor, seq_k: int) -> int:
    batch, heads = q.shape[0], q.shape[1]
    return max(1, CHUNK_ELEMENTS // (batch * heads * seq_k))


def _causal_mask(start: int, stop: int, seq_k: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(start, stop, device=device)[:, None]
    cols = torch.arange(seq_k, device=device)[None, :]
    return cols > rows  # True where the key comes after the query: hidden


def _scale(q: torch.Tensor, sm_scale: float | None) -> float:
    return sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def attention_ref32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, lse), computed in fp64 and returned as fp32."""
    group = _group(q, k)
    scale = _scale(q, sm_scale)
    seq_q, seq_k = q.shape[2], k.shape[2]
    if causal and seq_q != seq_k:
        raise ValueError("causal attention needs seq_q == seq_k")
    kd = repeat_kv(k, group).double()
    vd = repeat_kv(v, group).double()
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    step = _chunk_rows(q, seq_k)
    for start in range(0, seq_q, step):
        stop = min(start + step, seq_q)
        scores = (q[:, :, start:stop].double() @ kd.transpose(-1, -2)) * scale
        if causal:
            scores.masked_fill_(_causal_mask(start, stop, seq_k, q.device), float("-inf"))
        chunk_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - chunk_lse[..., None])
        out[:, :, start:stop] = (probs @ vd).float()
        lse[:, :, start:stop] = chunk_lse.float()
    return out, lse


def attention_ref_low(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(out in the input dtype from SDPA, lse in fp32 from scores rounded to the input dtype)."""
    group = _group(q, k)
    scale = _scale(q, sm_scale)
    seq_q, seq_k = q.shape[2], k.shape[2]
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=scale, enable_gqa=group > 1
    )
    kr = repeat_kv(k, group)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    step = _chunk_rows(q, seq_k)
    for start in range(0, seq_q, step):
        stop = min(start + step, seq_q)
        scores = (q[:, :, start:stop] @ kr.transpose(-1, -2)).float() * scale
        if causal:
            scores.masked_fill_(_causal_mask(start, stop, seq_k, q.device), float("-inf"))
        lse[:, :, start:stop] = torch.logsumexp(scores, dim=-1)
    return out, lse
