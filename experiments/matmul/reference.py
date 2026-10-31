"""Pure-PyTorch references for matmul.

Three references, used by the tolerance rule err(kernel) ≤ 2 · err(ref_low) + floor:

    matmul_ref32          the exact answer to compare against, computed in fp64 and
                          returned as fp32, so no TF32 rounding can enter it
    matmul_ref_low        PyTorch's own matmul in the input dtype (fp32 accumulation)
    matmul_fp16acc_ref_low
                          a matmul whose running sum is kept in fp16, for the kernel
                          variant that accumulates in fp16
"""

from __future__ import annotations

import torch

# The fp16-accumulating reference adds one partial product per k_step slice of K
# into its fp16 sum, so k_step is the number of K values the kernel sums before
# each fp16 rounding. Triton 3.8.0 compiles the kernel's fp16-accumulating dot
#   sm_80 and later   to mma m16n8k16 with an fp16 accumulator: 16 values per rounding
#   sm_75             to one fp16 FMA per product, with no tensor cores: 1 per rounding
# (read from the compiled SASS: HMMA on sm_80, HFMA2 on sm_75). On a Tesla T4 the
# kernel's max error equals the k_step = 1 reference's at M = N = K = 1,024, 4,096
# and 8,192 (fp16acc_rounding.py).
K_STEP = 16


def kernel_k_step(device: torch.device) -> int:
    """The reference's k_step for the kernel compiled for this device."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        return 1
    return K_STEP


def matmul_ref32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b in fp64, returned as fp32."""
    return (a.double() @ b.double()).float()


def matmul_ref_low(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b in the input dtype."""
    return a @ b


def matmul_fp16acc_ref_low(
    a: torch.Tensor, b: torch.Tensor, *, k_step: int = K_STEP, toward_zero: bool = False
) -> torch.Tensor:
    """a @ b with the running sum rounded to fp16 after every k_step slice of K.

    Each slice's partial product is computed in fp32; adding it to the sum and
    rounding the result to fp16 is the error an fp16 accumulator adds. The
    rounding is to nearest, or toward zero when toward_zero.
    """
    m, k = a.shape
    acc = torch.zeros((m, b.shape[1]), dtype=torch.float16, device=a.device)
    for start in range(0, k, k_step):
        stop = min(start + k_step, k)
        partial = a[:, start:stop].float() @ b[start:stop].float()
        acc = to_fp16(acc.float() + partial, toward_zero=toward_zero)
    return acc


def to_fp16(x: torch.Tensor, *, toward_zero: bool = False) -> torch.Tensor:
    """x rounded to fp16, to nearest or toward zero."""
    y = x.half()
    if not toward_zero:
        return y
    # Where rounding to nearest went up in magnitude, take the next fp16 value
    # toward zero. Lowering the bit pattern by one does that for either sign,
    # and turns an overflow to inf into the largest finite value.
    lower = (y.view(torch.int16) - 1).view(torch.float16)
    return torch.where(y.float().abs() > x.abs(), lower, y)
