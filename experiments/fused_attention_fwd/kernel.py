"""Fused attention forward with an LSE output and GQA.

Grid: (blocks of BLOCK_M query rows, batch · query heads). Query head h reads KV
head h // group, where group = q_heads / kv_heads.

One program keeps its query tile in registers and streams its KV head's keys and
values, one tile of BLOCK_N keys at a time:

    resident                                  per key tile
    q    BLOCK_M · head_dim                   s = q · kᵀ · scale       scores, in base-2 units
    m    running max of s, per row            m_new = max(m, rowmax(s))
    l    running sum of 2^(s - m), per row    l = l · 2^(m - m_new) + rowsum(2^(s - m_new))
    acc  running Σ 2^(s - m) · v, fp32        acc = acc · 2^(m - m_new) + 2^(s - m_new) · v

    end:  out = acc / l;   lse = ln Σ e^score = m · ln 2 + ln l

Scores are kept in base 2 (scale multiplied by log2 e) because exp2 is a single
hardware instruction. With causal attention (seq_q == seq_k), row i sees keys
0 … i, so a program stops after the key tile that holds the last row's diagonal.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import triton
import triton.language as tl

from scripts.dump_kernel_ir import Example

# Searched by autotuning. Each entry's comment says the case it is for.
CONFIGS: list[dict[str, int]] = [
    # head_dim 64 on GPUs with large shared memory: tall query tiles, more reuse of each key tile.
    {"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 3, "num_warps": 8},
    # Tall query tiles with a narrower key tile, for less shared memory per stage.
    {"BLOCK_M": 128, "BLOCK_N": 32, "num_stages": 3, "num_warps": 4},
    # head_dim 128: the query tile and accumulator take twice the registers.
    {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2, "num_warps": 4},
    # GPUs with small shared memory (the T4: 64 KiB per SM).
    {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 2, "num_warps": 4},
]

# The interpreter cannot time configurations; it runs the smallest tiles.
_INTERPRETER_CONFIG = CONFIGS[3]

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
HEAD_DIMS = (64, 128)


@triton.jit
def _attention_fwd_kernel(  # type: ignore[no-untyped-def]
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    scale_log2e,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    q_heads,
    group,
    seq_q,
    seq_k,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // q_heads
    head = batch_head % q_heads
    kv_head = head // group

    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    cols = tl.arange(0, BLOCK_N)
    row_ok = rows < seq_q

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    k_base = k_ptr + batch * stride_kb + kv_head * stride_kh
    v_base = v_ptr + batch * stride_vb + kv_head * stride_vh
    q = tl.load(
        q_base + rows[:, None] * stride_qm + dims[None, :] * stride_qd,
        mask=row_ok[:, None],
        other=0.0,
    )

    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)  # noqa: E741
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal: the block's last row is block_m · BLOCK_M + BLOCK_M - 1, so no key past it is needed.
    end = tl.minimum(seq_k, (block_m + 1) * BLOCK_M) if CAUSAL else seq_k
    for start in range(0, end, BLOCK_N):
        keys = start + cols
        key_ok = keys < seq_k
        k = tl.load(
            k_base + keys[None, :] * stride_kn + dims[:, None] * stride_kd,
            mask=key_ok[None, :],
            other=0.0,
        )
        s = tl.dot(q, k) * scale_log2e
        visible = key_ok[None, :]
        if CAUSAL:
            visible = visible & (keys[None, :] <= rows[:, None])
        s = tl.where(visible, s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        # A row that has seen no visible key yet keeps m = -inf; subtracting 0 there
        # keeps 2^(-inf - 0) = 0 instead of the NaN from -inf - (-inf).
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l = l * alpha + tl.sum(p, axis=1)  # noqa: E741
        v = tl.load(
            v_base + keys[:, None] * stride_vn + dims[None, :] * stride_vd,
            mask=key_ok[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    out = acc / l[:, None]
    out_base = out_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        out_base + rows[:, None] * stride_om + dims[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None],
    )
    lse = m * 0.6931471805599453 + tl.log(l)  # m is in base-2 units: ln 2^m = m · ln 2
    tl.store(lse_ptr + batch * stride_lb + head * stride_lh + rows * stride_lm, lse, mask=row_ok)


_attention_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": c["BLOCK_M"], "BLOCK_N": c["BLOCK_N"]},
            num_stages=c["num_stages"],
            num_warps=c["num_warps"],
        )
        for c in CONFIGS
    ],
    key=["seq_q", "seq_k", "HEAD_DIM", "CAUSAL"],
)(_attention_fwd_kernel)


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    causal: bool,
) -> None:
    if any(t.dim() != 4 for t in (q, k, v, out)):
        raise ValueError("q, k, v and out must be 4-D: [batch, heads, seq, head_dim]")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype for t in (k, v, out)
    ):
        raise ValueError(f"q, k, v and out need one dtype, fp16 or bf16; got {q.dtype}")
    batch, q_heads, seq_q, head_dim = q.shape
    if head_dim not in HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
    if k.shape != v.shape or k.shape[0] != batch or k.shape[3] != head_dim:
        raise ValueError(f"k and v must be [{batch}, kv_heads, seq_k, {head_dim}]")
    kv_heads, seq_k = k.shape[1], k.shape[2]
    if q_heads % kv_heads:
        raise ValueError(f"q_heads {q_heads} is not a multiple of kv_heads {kv_heads}")
    if seq_k == 0:
        raise ValueError("seq_k must be at least 1")
    if causal and seq_q != seq_k:
        raise ValueError(f"causal attention needs seq_q == seq_k, got {seq_q} and {seq_k}")
    if out.shape != q.shape:
        raise ValueError(f"out must have q's shape {tuple(q.shape)}")
    if lse.dtype != torch.float32 or tuple(lse.shape) != (batch, q_heads, seq_q):
        raise ValueError(f"lse must be fp32 [{batch}, {q_heads}, {seq_q}]")
    if not _interpreting() and not all(t.is_cuda for t in (q, k, v, out, lse)):
        raise ValueError("tensors must be on a CUDA device")


def attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float | None = None,
    config: dict[str, int] | None = None,
) -> None:
    """out = softmax(q · kᵀ · sm_scale) · v and lse = log Σ exp(q · kᵀ · sm_scale), per row.

    q: [batch, q_heads, seq_q, head_dim]; k, v: [batch, kv_heads, seq_k, head_dim];
    q_heads a multiple of kv_heads; head_dim 64 or 128; fp16 or bf16; any strides.
    out: like q. lse: [batch, q_heads, seq_q], fp32. causal needs seq_q == seq_k.
    sm_scale defaults to 1 / sqrt(head_dim). config, one entry of CONFIGS,
    replaces autotuning.
    """
    _check(q, k, v, out, lse, causal)
    batch, q_heads, seq_q, head_dim = q.shape
    kv_heads, seq_k = k.shape[1], k.shape[2]
    if seq_q == 0 or batch == 0:
        return
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)
    args = (
        q,
        k,
        v,
        out,
        lse,
        scale * _LOG2E,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *lse.stride(),
        q_heads,
        q_heads // kv_heads,
        seq_q,
        seq_k,
    )
    if config is None and _interpreting():
        config = _INTERPRETER_CONFIG
    if config is not None:
        grid = (triton.cdiv(seq_q, config["BLOCK_M"]), batch * q_heads)
        _attention_fwd_kernel[grid](
            *args,
            HEAD_DIM=head_dim,
            CAUSAL=causal,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            num_stages=config["num_stages"],
            num_warps=config["num_warps"],
        )
        return

    def autotuned_grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(seq_q, meta["BLOCK_M"]), batch * q_heads)

    _attention_autotuned[autotuned_grid](*args, HEAD_DIM=head_dim, CAUSAL=causal)


def dump_examples() -> list[Example]:
    """For scripts/dump_kernel_ir.py.

    Batch 4, 32 query heads, 8 KV heads, sequence 4,096, head_dim 128, every configuration.
    """
    batch, q_heads, kv_heads, seq, head_dim = 4, 32, 8, 4096, 128
    q = torch.empty((batch, q_heads, seq, head_dim), dtype=torch.float16)
    kv = torch.empty((batch, kv_heads, seq, head_dim), dtype=torch.float16)
    lse = torch.empty((batch, q_heads, seq), dtype=torch.float32)
    names = ("b", "h", "m", "d")
    args: dict[str, Any] = {
        "q_ptr": q,
        "k_ptr": kv,
        "v_ptr": kv,
        "out_ptr": q,
        "lse_ptr": lse,
        "scale_log2e": _LOG2E / math.sqrt(head_dim),
        **{f"stride_q{n}": s for n, s in zip(names, q.stride(), strict=True)},
        **{f"stride_k{n}": s for n, s in zip(("b", "h", "n", "d"), kv.stride(), strict=True)},
        **{f"stride_v{n}": s for n, s in zip(("b", "h", "n", "d"), kv.stride(), strict=True)},
        **{f"stride_o{n}": s for n, s in zip(names, q.stride(), strict=True)},
        **{f"stride_l{n}": s for n, s in zip(("b", "h", "m"), lse.stride(), strict=True)},
        "q_heads": q_heads,
        "group": q_heads // kv_heads,
        "seq_q": seq,
        "seq_k": seq,
    }
    return [
        Example(
            name=f"attention_fwd_{'causal' if causal else 'full'}_cfg{i}",
            kernel=_attention_fwd_kernel,
            args=args,
            constexprs={
                "HEAD_DIM": head_dim,
                "CAUSAL": causal,
                "BLOCK_M": c["BLOCK_M"],
                "BLOCK_N": c["BLOCK_N"],
            },
            num_warps=c["num_warps"],
            num_stages=c["num_stages"],
        )
        for causal in (False, True)
        for i, c in enumerate(CONFIGS)
    ]
