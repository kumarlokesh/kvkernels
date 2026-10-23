"""Roofline model and closed-form cost models for attention.

A kernel that performs F floating-point operations and moves B bytes between HBM
and the SMs has arithmetic intensity AI = F / B. On a GPU with memory bandwidth
W and compute peak P, it can reach at most min(P, AI * W). The ridge point
P / W is the intensity at which the two limits meet: below it a kernel is
memory-bound, above it compute-bound.

The cost models count only the dominant terms: K and V reads, and the two matrix
products (Q·Kᵀ and P·V) at 2 FLOPs per multiply-add. Reading Q, writing the output
and the softmax arithmetic are left out; at long context they are a small share.
"""

from __future__ import annotations

from typing import Literal

from kvkernels.specs import GPUSpec, peak_tensor_tflops


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    """FLOPs per byte moved."""
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def peak_tflops(spec: GPUSpec, dtype: str, acc_dtype: str = "fp32") -> float:
    peak = peak_tensor_tflops(spec, dtype, acc_dtype)
    if peak is None:
        raise ValueError(
            f"{spec.name} has no tensor-core peak for {dtype} with {acc_dtype} accumulation"
        )
    return peak


def ridge_point(spec: GPUSpec, dtype: str, acc_dtype: str = "fp32") -> float:
    """Arithmetic intensity, in FLOPs per byte, where the memory and compute limits meet."""
    return peak_tflops(spec, dtype, acc_dtype) * 1e12 / (spec.peak_hbm_gbps * 1e9)


def attainable_tflops(ai: float, spec: GPUSpec, dtype: str, acc_dtype: str = "fp32") -> float:
    """The roofline: min(compute peak, AI x memory bandwidth), in TFLOP/s."""
    if ai < 0:
        raise ValueError("arithmetic intensity must be non-negative")
    memory_limit = ai * spec.peak_hbm_gbps * 1e9 / 1e12
    return min(peak_tflops(spec, dtype, acc_dtype), memory_limit)


def bound(
    ai: float, spec: GPUSpec, dtype: str, acc_dtype: str = "fp32"
) -> Literal["memory", "compute"]:
    return "memory" if ai < ridge_point(spec, dtype, acc_dtype) else "compute"


def _check_heads(q_heads: int, kv_heads: int) -> None:
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"q_heads ({q_heads}) must be a positive multiple of kv_heads ({kv_heads})"
        )


def decode_attention_cost(
    batch: int, q_heads: int, kv_heads: int, ctx_len: int, head_dim: int, dtype_bytes: int
) -> tuple[int, int]:
    """(FLOPs, bytes) for one decode step of `batch` requests, each with its own context.

    Each request reads its K and V once:  bytes = batch · 2 · ctx · kv_heads · d · b
    Each query head does Q·Kᵀ and P·V:    flops = batch · 4 · ctx · q_heads · d
    So AI = 2 · (q_heads / kv_heads) / b, independent of batch and context length.
    """
    _check_heads(q_heads, kv_heads)
    nbytes = batch * 2 * ctx_len * kv_heads * head_dim * dtype_bytes
    flops = batch * 4 * ctx_len * q_heads * head_dim
    return flops, nbytes


def shared_prefix_attention_cost(
    batch: int, q_heads: int, kv_heads: int, prefix_len: int, head_dim: int, dtype_bytes: int
) -> tuple[int, int]:
    """(FLOPs, bytes) for `batch` decode queries attending to one shared prefix, read once.

    The prefix K and V are read once for the batch:  bytes = 2 · P · kv_heads · d · b
    Every query still does its own matrix products:  flops = batch · 4 · P · q_heads · d
    So AI = 2 · batch · (q_heads / kv_heads) / b, growing linearly with batch.
    """
    _check_heads(q_heads, kv_heads)
    nbytes = 2 * prefix_len * kv_heads * head_dim * dtype_bytes
    flops = batch * 4 * prefix_len * q_heads * head_dim
    return flops, nbytes
