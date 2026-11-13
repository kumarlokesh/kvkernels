"""Merging partial attention results.

Attention over a key set split into parts equals a merge of the attention over
each part, given each part's output o_j and LSE (log of the sum of its
exponentiated scores):

    lse = log Σ_j exp(lse_j)          out = Σ_j exp(lse_j - lse) · o_j

One program merges one (token, head) row. It runs through the parts with a
running maximum m, a running sum s of exp(lse_j - m) and a running weighted sum
of outputs, rescaling both when m grows, so no exponential overflows. A part
with lse = -inf is an empty key set: its weight is exactly 0 and its output is
ignored whatever it holds. If every part is empty the row's output is 0 and its
LSE is -inf.

Three entry points:

    merge_attention_states     two parts, given as separate tensors
    merge_attention_states_n   N parts stacked along a first dimension
    merge_ragged (internal)    each row merges its own list of parts, for the cascade
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from kvkernels.ops._registry import registered_op

_OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_HEAD_DIM = 256


@triton.jit
def _merge_step(m, s, acc, part_lse, part_o):  # type: ignore[no-untyped-def]
    """Add one part to the running maximum m, sum s and weighted output sum acc."""
    m_new = tl.maximum(m, part_lse)
    # With every part so far empty, m_new is -inf; subtracting 0 keeps exp(-inf) = 0.
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    keep = tl.exp(m - m_safe)
    w = tl.exp(part_lse - m_safe)
    part_o = tl.where(w > 0.0, part_o, 0.0)  # an empty part's output may hold anything
    return m_new, s * keep + w, acc * keep + w * part_o


@triton.jit
def _finish(m, s, acc):  # type: ignore[no-untyped-def]
    """The merged output and LSE; 0 and -inf when every part was empty."""
    nonempty = s > 0.0
    out = tl.where(nonempty, acc / tl.where(nonempty, s, 1.0), 0.0)
    lse = tl.where(nonempty, m + tl.log(tl.where(nonempty, s, 1.0)), float("-inf"))
    return out, lse


@triton.jit
def _merge_n_kernel(  # type: ignore[no-untyped-def]
    o_ptr,
    lse_ptr,
    out_ptr,
    lse_out_ptr,
    stride_os,
    stride_ot,
    stride_oh,
    stride_od,
    stride_ls,
    stride_lt,
    stride_lh,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_lo_t,
    stride_lo_h,
    num_parts,
    heads,
    head_dim,
    WRITE_LSE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    t = row // heads
    h = row % heads
    d = tl.arange(0, BLOCK_D)
    d_ok = d < head_dim
    m = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    s = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in range(num_parts):
        part_lse = tl.load(lse_ptr + j * stride_ls + t * stride_lt + h * stride_lh)
        part_o = tl.load(
            o_ptr + j * stride_os + t * stride_ot + h * stride_oh + d * stride_od,
            mask=d_ok,
            other=0.0,
        ).to(tl.float32)
        m, s, acc = _merge_step(m, s, acc, part_lse, part_o)
    out, lse = _finish(m, s, acc)
    tl.store(
        out_ptr + t * stride_out_t + h * stride_out_h + d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_ok,
    )
    if WRITE_LSE:
        tl.store(lse_out_ptr + t * stride_lo_t + h * stride_lo_h + d * 0, lse, mask=d == 0)


@triton.jit
def _merge_two_kernel(  # type: ignore[no-untyped-def]
    oa_ptr,
    la_ptr,
    ob_ptr,
    lb_ptr,
    out_ptr,
    lse_out_ptr,
    stride_oa_t,
    stride_oa_h,
    stride_oa_d,
    stride_la_t,
    stride_la_h,
    stride_ob_t,
    stride_ob_h,
    stride_ob_d,
    stride_lb_t,
    stride_lb_h,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_lo_t,
    stride_lo_h,
    heads,
    head_dim,
    WRITE_LSE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    t = row // heads
    h = row % heads
    d = tl.arange(0, BLOCK_D)
    d_ok = d < head_dim
    m = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    s = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    lse_a = tl.load(la_ptr + t * stride_la_t + h * stride_la_h)
    o_a = tl.load(
        oa_ptr + t * stride_oa_t + h * stride_oa_h + d * stride_oa_d, mask=d_ok, other=0.0
    ).to(tl.float32)
    m, s, acc = _merge_step(m, s, acc, lse_a, o_a)
    lse_b = tl.load(lb_ptr + t * stride_lb_t + h * stride_lb_h)
    o_b = tl.load(
        ob_ptr + t * stride_ob_t + h * stride_ob_h + d * stride_ob_d, mask=d_ok, other=0.0
    ).to(tl.float32)
    m, s, acc = _merge_step(m, s, acc, lse_b, o_b)
    out, lse = _finish(m, s, acc)
    tl.store(
        out_ptr + t * stride_out_t + h * stride_out_h + d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_ok,
    )
    if WRITE_LSE:
        tl.store(lse_out_ptr + t * stride_lo_t + h * stride_lo_h + d * 0, lse, mask=d == 0)


@triton.jit
def _merge_ragged_kernel(  # type: ignore[no-untyped-def]
    part_o_ptr,
    part_lse_ptr,
    start_ptr,
    end_ptr,
    indices_ptr,
    num_rows_ptr,
    out_ptr,
    lse_out_ptr,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_lo_t,
    stride_lo_h,
    heads,
    head_dim,
    WRITE_LSE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Row r = token · heads + head merges parts indices[start[r] : end[r]].

    Parts are rows of part_o ([parts, head_dim], fp32) and part_lse ([parts]).
    The grid may be larger than the batch; rows at or past num_rows exit.
    """
    row = tl.program_id(0)
    if row >= tl.load(num_rows_ptr):
        return
    t = row // heads
    h = row % heads
    d = tl.arange(0, BLOCK_D)
    d_ok = d < head_dim
    m = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    s = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    first = tl.load(start_ptr + row)
    last = tl.load(end_ptr + row)
    for i in range(first, last):
        part = tl.load(indices_ptr + i).to(tl.int64)
        part_lse = tl.load(part_lse_ptr + part)
        part_o = tl.load(part_o_ptr + part * head_dim + d, mask=d_ok, other=0.0)
        m, s, acc = _merge_step(m, s, acc, part_lse, part_o)
    out, lse = _finish(m, s, acc)
    tl.store(
        out_ptr + t * stride_out_t + h * stride_out_h + d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_ok,
    )
    if WRITE_LSE:
        tl.store(lse_out_ptr + t * stride_lo_t + h * stride_lo_h + d * 0, lse, mask=d == 0)


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _block_d(head_dim: int) -> int:
    return max(16, int(triton.next_power_of_2(head_dim)))


def _check_out(
    out: torch.Tensor, lse_out: torch.Tensor | None, rows_shape: tuple[int, ...]
) -> None:
    if tuple(out.shape) != rows_shape or out.dtype not in _OUT_DTYPES:
        raise ValueError(
            f"out must be {rows_shape} in fp16, bf16 or fp32, got {tuple(out.shape)} {out.dtype}"
        )
    if lse_out is not None and (
        tuple(lse_out.shape) != rows_shape[:-1] or lse_out.dtype != torch.float32
    ):
        raise ValueError(f"lse_out must be fp32 {rows_shape[:-1]}")
    if rows_shape[-1] > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim {rows_shape[-1]} is above {_MAX_HEAD_DIM}")
    if not _interpreting() and not out.is_cuda:
        raise ValueError("tensors must be on a CUDA device")


def _lse_args(lse_out: torch.Tensor | None, out: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if lse_out is None:
        return out, 0, 0  # never written: WRITE_LSE is off
    return lse_out, lse_out.stride(0), lse_out.stride(1)


@registered_op("kvkernels::merge_attention_states_n", mutates_args=("out", "lse_out"))
def _merge_n_op(
    o: torch.Tensor, lse: torch.Tensor, out: torch.Tensor, lse_out: torch.Tensor | None
) -> None:
    if o.dim() != 4 or o.shape[0] == 0:
        raise ValueError(
            f"o must be [splits, tokens, heads, head_dim] with splits ≥ 1, got {tuple(o.shape)}"
        )
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must be fp32, got {lse.dtype}")
    if tuple(lse.shape) != tuple(o.shape[:-1]):
        raise ValueError(f"lse shape {tuple(lse.shape)} does not match o shape {tuple(o.shape)}")
    _check_out(out, lse_out, tuple(o.shape[1:]))
    splits, tokens, heads, head_dim = o.shape
    if tokens * heads == 0:
        return
    lse_target, lo_t, lo_h = _lse_args(lse_out, out)
    _merge_n_kernel[(tokens * heads,)](
        o,
        lse,
        out,
        lse_target,
        *o.stride(),
        *lse.stride(),
        *out.stride(),
        lo_t,
        lo_h,
        splits,
        heads,
        head_dim,
        WRITE_LSE=lse_out is not None,
        BLOCK_D=_block_d(head_dim),
    )


@_merge_n_op.register_fake
def _merge_n_fake(
    o: torch.Tensor, lse: torch.Tensor, out: torch.Tensor, lse_out: torch.Tensor | None
) -> None:
    return None


@registered_op("kvkernels::merge_attention_states", mutates_args=("out", "lse_out"))
def _merge_two_op(
    o_a: torch.Tensor,
    lse_a: torch.Tensor,
    o_b: torch.Tensor,
    lse_b: torch.Tensor,
    out: torch.Tensor,
    lse_out: torch.Tensor | None,
) -> None:
    for name, o, lse in (("a", o_a, lse_a), ("b", o_b, lse_b)):
        if o.dim() != 3:
            raise ValueError(f"o_{name} must be [tokens, heads, head_dim]")
        if lse.dtype != torch.float32:
            raise ValueError(f"lse_{name} must be fp32, got {lse.dtype}")
        if tuple(lse.shape) != tuple(o.shape[:-1]) or o.shape != o_a.shape:
            raise ValueError(f"part {name}: shape does not match")
    _check_out(out, lse_out, tuple(o_a.shape))
    tokens, heads, head_dim = o_a.shape
    if tokens * heads == 0:
        return
    lse_target, lo_t, lo_h = _lse_args(lse_out, out)
    _merge_two_kernel[(tokens * heads,)](
        o_a,
        lse_a,
        o_b,
        lse_b,
        out,
        lse_target,
        *o_a.stride(),
        *lse_a.stride(),
        *o_b.stride(),
        *lse_b.stride(),
        *out.stride(),
        lo_t,
        lo_h,
        heads,
        head_dim,
        WRITE_LSE=lse_out is not None,
        BLOCK_D=_block_d(head_dim),
    )


@_merge_two_op.register_fake
def _merge_two_fake(
    o_a: torch.Tensor,
    lse_a: torch.Tensor,
    o_b: torch.Tensor,
    lse_b: torch.Tensor,
    out: torch.Tensor,
    lse_out: torch.Tensor | None,
) -> None:
    return None


def merge_attention_states(
    o_a: torch.Tensor,
    lse_a: torch.Tensor,
    o_b: torch.Tensor,
    lse_b: torch.Tensor,
    out: torch.Tensor,
    lse_out: torch.Tensor | None = None,
) -> None:
    """Merge two parts. o_*: [tokens, heads, head_dim]; lse_*: [tokens, heads], fp32, natural log.

    out: [tokens, heads, head_dim] in fp16, bf16 or fp32. lse_out, if given:
    [tokens, heads], fp32. Runs as the custom op kvkernels::merge_attention_states.
    """
    _merge_two_op(o_a, lse_a, o_b, lse_b, out, lse_out)


def merge_attention_states_n(
    o: torch.Tensor, lse: torch.Tensor, out: torch.Tensor, lse_out: torch.Tensor | None = None
) -> None:
    """Merge N parts stacked on the first dimension. o: [splits, tokens, heads, head_dim].

    lse: [splits, tokens, heads], fp32, natural log. Outputs as merge_attention_states.
    Runs as the custom op kvkernels::merge_attention_states_n.
    """
    _merge_n_op(o, lse, out, lse_out)


def merge_ragged(
    part_o: torch.Tensor,
    part_lse: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    indices: torch.Tensor,
    num_rows: torch.Tensor,
    out: torch.Tensor,
    lse_out: torch.Tensor | None,
    *,
    max_rows: int,
) -> None:
    """Row r of out ([tokens, heads, head_dim]) merges parts indices[start[r] : end[r]].

    part_o: [parts, head_dim] fp32; part_lse: [parts] fp32. num_rows: a one-element
    device tensor holding tokens · heads; the grid is max_rows programs, so a CUDA
    graph captured once serves any batch up to max_rows rows.
    """
    heads, head_dim = out.shape[1], out.shape[2]
    lse_target, lo_t, lo_h = _lse_args(lse_out, out)
    _merge_ragged_kernel[(max_rows,)](
        part_o,
        part_lse,
        start,
        end,
        indices,
        num_rows,
        out,
        lse_target,
        *out.stride(),
        lo_t,
        lo_h,
        heads,
        head_dim,
        WRITE_LSE=lse_out is not None,
        BLOCK_D=_block_d(head_dim),
    )
