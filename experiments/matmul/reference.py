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

# The fp16-accumulating reference adds one partial product per K_STEP slice of K
# into its fp16 sum. A tensor-core instruction with fp16 inputs covers 16 values
# of K and rounds its fp16 sum once, so 16 gives the reference as many roundings
# as the kernel makes (inference from the instruction shape, m16n8k16).
K_STEP = 16


def matmul_ref32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b in fp64, returned as fp32."""
    return (a.double() @ b.double()).float()


def matmul_ref_low(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b in the input dtype."""
    return a @ b


def matmul_fp16acc_ref_low(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with the running sum rounded to fp16 after every K_STEP slice of K.

    Each slice's partial product is computed in fp32; adding it to the sum and
    rounding the result to fp16 is the error an fp16 accumulator adds.
    """
    m, k = a.shape
    acc = torch.zeros((m, b.shape[1]), dtype=torch.float16, device=a.device)
    for start in range(0, k, K_STEP):
        stop = min(start + K_STEP, k)
        partial = a[:, start:stop].float() @ b[start:stop].float()
        acc = (acc.float() + partial).half()
    return acc
