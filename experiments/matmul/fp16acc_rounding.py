"""Which fp16 accumulation model the GPU's fp16-accumulating matmul follows.

Run on a GPU as ``python -m experiments.matmul.fp16acc_rounding``.

For square fp16 matmuls, prints the max error against the fp64 answer of: the
kernel accumulating in fp16; cuBLAS accumulating in fp16 on the same GPU; and the
fp16-accumulating reference rounding its running sum to nearest every 1, 2, 4, 8
or 16 values of K, and toward zero every 8. The second line of each size is the
max difference between the kernel and each of the others. The kernel's test needs
a reference whose error is at least half the kernel's.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from experiments.matmul import kernel
from experiments.matmul.reference import matmul_fp16acc_ref_low, matmul_ref32
from kvkernels.tolerance import max_abs_error

SIZES = (1024, 4096, 8192)
Matmul = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MODELS = [(1, False), (2, False), (4, False), (8, False), (16, False), (8, True)]


def cublas_fp16acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    try:
        return a @ b
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = False


def model(k_step: int, toward_zero: bool) -> Matmul:
    return lambda a, b: matmul_fp16acc_ref_low(a, b, k_step=k_step, toward_zero=toward_zero)


def row(cells: list[str]) -> str:
    return "  ".join(f"{c:>14}" for c in cells)


def main() -> None:
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()} sm_{major}{minor}")
    others: dict[str, Matmul] = {"cublas-fp16acc": cublas_fp16acc}
    for k, zero in MODELS:
        others[f"k{k}-{'zero' if zero else 'nearest'}"] = model(k, zero)
    print(row(["size", "measure", "kernel", *others]))
    for size in SIZES:
        gen = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn((size, size), generator=gen, device="cuda").half()
        b = torch.randn((size, size), generator=gen, device="cuda").half()
        out = torch.empty((size, size), dtype=torch.float16, device="cuda")
        kernel.matmul(a, b, out, fp16_accumulate=True)
        ref32 = matmul_ref32(a, b)
        results = [fn(a, b) for fn in others.values()]
        errors = [f"{max_abs_error(r, ref32):.4f}" for r in results]
        diffs = [f"{max_abs_error(out, r.float()):.4f}" for r in results]
        print(row([str(size), "error", f"{max_abs_error(out, ref32):.4f}", *errors]))
        print(row(["", "kernel - this", "", *diffs]))


if __name__ == "__main__":
    main()
