"""Correctness rules shared by tests and the benchmark harness.

Floating-point outputs pass when their error against an fp32 reference is at most
twice the error of the same reference computed in the low-precision dtype, plus a
small floor. The allowed error grows with the problem, because a longer reduction
in low precision also increases the error of the low-precision reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Floors for outputs whose low-precision reference happens to be almost exact.
FLOORS: dict[torch.dtype, float] = {
    torch.float16: 1e-3,
    torch.bfloat16: 8e-3,
    torch.float32: 1e-5,
}


@dataclass(frozen=True)
class CloseResult:
    ok: bool
    err: float
    allowed: float


def max_abs_error(x: torch.Tensor, ref32: torch.Tensor) -> float:
    """Largest |x - ref32| over all elements, in fp32.

    A NaN in x where ref32 has none counts as an infinite error. Positions where
    both are NaN, or both are the same infinity, count as zero.
    """
    if x.shape != ref32.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(ref32.shape)}")
    if x.numel() == 0:
        return 0.0
    a = x.float()
    b = ref32.float()
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    diff = torch.where(same, torch.zeros_like(a), (a - b).abs())
    diff = torch.nan_to_num(diff, nan=float("inf"))
    return float(diff.max().item())


def check_close(out: torch.Tensor, ref_low: torch.Tensor, ref32: torch.Tensor) -> CloseResult:
    """Pass when err(out) <= 2 * err(ref_low) + floor[out.dtype]."""
    if out.dtype not in FLOORS:
        raise ValueError(f"no tolerance floor for {out.dtype}")
    err = max_abs_error(out, ref32)
    allowed = 2.0 * max_abs_error(ref_low, ref32) + FLOORS[out.dtype]
    return CloseResult(ok=err <= allowed, err=err, allowed=allowed)


def assert_close(out: torch.Tensor, ref_low: torch.Tensor, ref32: torch.Tensor) -> None:
    result = check_close(out, ref_low, ref32)
    if not result.ok:
        raise AssertionError(
            f"max abs error {result.err:.3e} exceeds allowed {result.allowed:.3e} ({out.dtype})"
        )


def assert_exact(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Data-movement ops: same dtype, shape and bits."""
    if out.dtype != ref.dtype or out.shape != ref.shape:
        raise AssertionError(
            f"dtype/shape mismatch: {out.dtype} {tuple(out.shape)}"
            f" vs {ref.dtype} {tuple(ref.shape)}"
        )
    out_bytes = out.reshape(-1).contiguous().view(torch.uint8)
    ref_bytes = ref.reshape(-1).contiguous().view(torch.uint8)
    if not torch.equal(out_bytes, ref_bytes):
        raise AssertionError("outputs differ from the reference")
