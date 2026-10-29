"""Matmul kernel against its references.

The kernel module (kernel.py) provides:

    CONFIGS: list[dict[str, int]]
        the autotuning list; keys BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps
    matmul(a, b, out, *, fp16_accumulate=False, config=None) -> None
        out = a @ b. a: [M, K], b: [K, N], out: [M, N], all fp16 or all bf16, any
        strides. Accumulates in fp32, or in fp16 when fp16_accumulate (fp16 only).
        config, one entry of CONFIGS, replaces autotuning.

Tolerance: err(kernel) ≤ 2 · err(ref_low) + floor, with ref32 computed in fp64.
Tests without the gpu mark keep every tensor at or below 65,536 elements.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

pytest.importorskip("torch")

import torch

from experiments.matmul.reference import (
    K_STEP,
    matmul_fp16acc_ref_low,
    matmul_ref32,
    matmul_ref_low,
)
from kvkernels.tolerance import assert_close, max_abs_error

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING or not torch.cuda.is_available() else "cuda"

SMALL_SHAPES = [(16, 16, 16), (33, 1000, 65), (100, 72, 40), (256, 256, 256)]
SQUARE = [512, 1024, 2048, 4096, 8192]
# Llama 3.1 8B projections as (N, K); M is the number of tokens.
PROJECTIONS = {
    "qkv": (6144, 4096),
    "o": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
}
TOKENS = [16, 128, 2048]


def kernel_module() -> ModuleType:
    """kernel.py, or a skip while it has not been written; other import errors fail."""
    try:
        return importlib.import_module("experiments.matmul.kernel")
    except ModuleNotFoundError as exc:
        if exc.name == "experiments.matmul.kernel":
            pytest.skip("experiments/matmul/kernel.py is not written yet")
        raise


def can_run_kernels() -> bool:
    return INTERPRETING or torch.cuda.is_available()


needs_kernel_runtime = pytest.mark.skipif(
    not can_run_kernels(), reason="needs the Triton interpreter or a CUDA GPU"
)


def operands(m: int, n: int, k: int, dtype: torch.dtype, seed: int = 0) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((m, k), generator=gen).to(dtype=dtype, device=DEVICE)
    b = torch.randn((k, n), generator=gen).to(dtype=dtype, device=DEVICE)
    out = torch.empty((m, n), dtype=dtype, device=DEVICE)
    return a, b, out


def check(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    assert_close(out, matmul_ref_low(a, b), matmul_ref32(a, b))


# --- the references themselves (no kernel needed) ---------------------------


def test_references_agree_exactly_when_nothing_rounds() -> None:
    # Small integers and short sums are exact in fp16, fp32 and fp64.
    gen = torch.Generator().manual_seed(1)
    a = torch.randint(-3, 4, (8, 16), generator=gen).half()
    b = torch.randint(-3, 4, (16, 8), generator=gen).half()
    expected = matmul_ref32(a, b)
    assert torch.equal(matmul_ref_low(a, b).float(), expected)
    assert torch.equal(matmul_fp16acc_ref_low(a, b).float(), expected)


def test_fp16acc_reference_with_one_slice_is_one_rounding() -> None:
    a, b, _ = operands(16, 24, K_STEP, torch.float16)
    expected = (a.float() @ b.float()).half()
    assert torch.equal(matmul_fp16acc_ref_low(a, b), expected)


def test_fp16acc_reference_loses_more_than_fp32_accumulation() -> None:
    # Many fp16 roundings of a growing sum cost more than one rounding at the end.
    a, b, _ = operands(64, 64, 2048, torch.float16)
    ref32 = matmul_ref32(a, b)
    assert max_abs_error(matmul_fp16acc_ref_low(a, b), ref32) > max_abs_error(
        matmul_ref_low(a, b), ref32
    )


# --- the kernel ----------------------------------------------------------------


@needs_kernel_runtime
@pytest.mark.parametrize("shape", SMALL_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_reference(shape: tuple[int, int, int]) -> None:
    kernel = kernel_module()
    a, b, out = operands(*shape, torch.float16)
    kernel.matmul(a, b, out)
    check(a, b, out)


@needs_kernel_runtime
def test_transposed_operands() -> None:
    kernel = kernel_module()
    a, b, out = operands(100, 72, 40, torch.float16)
    a_t = a.t().contiguous().t()  # same values, column-major
    b_t = b.t().contiguous().t()
    kernel.matmul(a_t, b_t, out)
    check(a, b, out)


@needs_kernel_runtime
@pytest.mark.parametrize(
    "shape", [(16, 16, 16), (100, 72, 40)], ids=lambda s: "x".join(map(str, s))
)
def test_fp16_accumulation_matches_its_reference(shape: tuple[int, int, int]) -> None:
    kernel = kernel_module()
    a, b, out = operands(*shape, torch.float16)
    kernel.matmul(a, b, out, fp16_accumulate=True)
    assert_close(out, matmul_fp16acc_ref_low(a, b), matmul_ref32(a, b))


@needs_kernel_runtime
def test_every_configuration_is_correct() -> None:
    kernel = kernel_module()
    a, b, out = operands(100, 72, 40, torch.float16)
    for config in kernel.CONFIGS:
        out.zero_()
        kernel.matmul(a, b, out, config=config)
        check(a, b, out)


@needs_kernel_runtime
def test_rejects_bad_arguments() -> None:
    kernel = kernel_module()
    a, b, out = operands(16, 16, 16, torch.float16)
    with pytest.raises(ValueError, match="inner"):
        kernel.matmul(a, b[:8], out)
    with pytest.raises(ValueError, match="dtype"):
        kernel.matmul(a, b.float(), out)
    with pytest.raises(ValueError, match="out"):
        kernel.matmul(a, b, out[:8])
    with pytest.raises(ValueError, match="fp16"):
        kernel.matmul(a.bfloat16(), b.bfloat16(), out.bfloat16(), fp16_accumulate=True)


# --- GPU tier ---------------------------------------------------------------------


def gpu_dtypes() -> list[torch.dtype]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return [torch.float16, torch.bfloat16]
    return [torch.float16]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", gpu_dtypes(), ids=str)
@pytest.mark.parametrize("size", SQUARE)
def test_square(size: int, dtype: torch.dtype) -> None:
    kernel = kernel_module()
    a, b, out = operands(size, size, size, dtype)
    kernel.matmul(a, b, out)
    check(a, b, out)


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("projection", list(PROJECTIONS))
def test_llama_projections(projection: str, tokens: int) -> None:
    kernel = kernel_module()
    n, k = PROJECTIONS[projection]
    a, b, out = operands(tokens, n, k, torch.float16)
    kernel.matmul(a, b, out)
    check(a, b, out)


@pytest.mark.gpu
@pytest.mark.parametrize("size", [4096, 8192])
def test_fp16_accumulation_full_size(size: int) -> None:
    kernel = kernel_module()
    a, b, out = operands(size, size, size, torch.float16)
    kernel.matmul(a, b, out, fp16_accumulate=True)
    assert_close(out, matmul_fp16acc_ref_low(a, b), matmul_ref32(a, b))


# --- the benchmark suite (no kernel needed) --------------------------------------


def test_suite_grid() -> None:
    from experiments.matmul import bench_suite

    params = list(bench_suite.grid())
    cases = [str(p["case"]) for p in params]
    assert cases.count("square") == len(bench_suite.SQUARE) + 1  # plus one bf16
    assert cases.count("fp16acc") == len(bench_suite.FP16ACC_SIZES)
    assert sum(c.startswith("projection_") for c in cases) == 12
    shapes = {(int(p["M"]), int(p["N"]), int(p["K"])) for p in params}
    assert shapes >= bench_suite.SWEEP_SHAPES
    assert bench_suite.flops({"M": 2, "N": 3, "K": 4}) == 48
