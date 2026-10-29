"""Matmul: out = a @ b with tl.dot, tiled over the output and over K.

Grid: one program per output tile of BLOCK_M · BLOCK_N. Program ids are assigned
in groups of GROUP_M tile rows: consecutive programs walk down a column of tiles
before moving right, so programs running at the same time read the same columns
of b from L2.

    output tiles, GROUP_M = 2; the number is the program id
     0  2  4  6
     1  3  5  7
     8 10 12 14
     9 11 13 15

One program loops over K, one BLOCK_K slice at a time: it loads a BLOCK_M · BLOCK_K
slice of a and a BLOCK_K · BLOCK_N slice of b, multiplies them with tl.dot, and adds
the product into an accumulator held in registers. Edge tiles mask the rows,
columns and K values that fall outside the matrices, loading zeros there.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl

from scripts.dump_kernel_ir import Example

# Searched by autotuning. Each entry's comment says the shape it is for.
CONFIGS: list[dict[str, int]] = [
    # Large shapes on GPUs with large shared memory (A100, H100).
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
    # Large shapes elsewhere: smaller tiles fit 100 KiB of shared memory per SM.
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    # Medium shapes: more programs to fill the GPU.
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    # M below 64: a taller tile would be mostly padding.
    {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
    # M = 16, a decode-sized batch: one tile row, so grouping does nothing.
    {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 1, "num_stages": 4, "num_warps": 4},
]

# The interpreter cannot time configurations; it runs the M < 64 entry, whose
# small tiles make interpreter-sized matrices span several tiles in every direction.
_INTERPRETER_CONFIG = CONFIGS[5]

_META_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")


@triton.jit
def _matmul_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    FP16_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped order: within a group of GROUP_M tile rows, the tile row changes fastest.
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * tiles_n
    first_tile_m = (pid // programs_per_group) * GROUP_M
    rows_in_group = tl.minimum(tiles_m - first_tile_m, GROUP_M)
    within = pid % programs_per_group
    tile_m = first_tile_m + within % rows_in_group
    tile_n = within // rows_in_group

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    if FP16_ACC:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k_tile * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        # FP16_ACC is a compile-time constant: Triton compiles only the branch taken.
        if FP16_ACC:  # noqa: SIM108
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(rm[:, None] < M) & (rn[None, :] < N))


_matmul_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {k: c[k] for k in _META_KEYS}, num_stages=c["num_stages"], num_warps=c["num_warps"]
        )
        for c in CONFIGS
    ],
    key=["M", "N", "K", "FP16_ACC"],
)(_matmul_kernel)


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _check(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, fp16_accumulate: bool) -> None:
    if a.dim() != 2 or b.dim() != 2 or out.dim() != 2:
        raise ValueError("a, b and out must be 2-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions differ: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError(f"a and b need the same dtype, fp16 or bf16; got {a.dtype}, {b.dtype}")
    if tuple(out.shape) != (a.shape[0], b.shape[1]) or out.dtype != a.dtype:
        raise ValueError(
            f"out must be {(a.shape[0], b.shape[1])} {a.dtype}, got {tuple(out.shape)} {out.dtype}"
        )
    if fp16_accumulate and a.dtype != torch.float16:
        raise ValueError("fp16 accumulation needs fp16 inputs")
    if not _interpreting() and not (a.is_cuda and b.is_cuda and out.is_cuda):
        raise ValueError("tensors must be on a CUDA device")
    for name, t in (("a", a), ("b", b), ("out", out)):
        # Offsets are 32-bit inside the kernel.
        if t.numel() >= 2**31:
            raise ValueError(f"{name} has {t.numel()} elements; at most 2**31 - 1 supported")


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    fp16_accumulate: bool = False,
    config: dict[str, int] | None = None,
) -> None:
    """out = a @ b. a: [M, K], b: [K, N], out: [M, N]; fp16 or bf16; any strides.

    Accumulates in fp32, or in fp16 when fp16_accumulate (fp16 inputs only).
    config, one entry of CONFIGS, replaces autotuning.
    """
    _check(a, b, out, fp16_accumulate)
    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0:
        return
    if k == 0:
        out.zero_()
        return
    args = (a, b, out, m, n, k, *a.stride(), *b.stride(), *out.stride())
    if config is None and _interpreting():
        config = _INTERPRETER_CONFIG
    if config is not None:
        grid = (triton.cdiv(m, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]),)
        _matmul_kernel[grid](
            *args,
            FP16_ACC=fp16_accumulate,
            **{key: config[key] for key in _META_KEYS},
            num_stages=config["num_stages"],
            num_warps=config["num_warps"],
        )
        return

    def autotuned_grid(meta: dict[str, Any]) -> tuple[int]:
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _matmul_autotuned[autotuned_grid](*args, FP16_ACC=fp16_accumulate)


def dump_examples() -> list[Example]:
    """For scripts/dump_kernel_ir.py: a 4,096-square fp16 matmul under each configuration."""
    size = 4096
    a = torch.empty((size, size), dtype=torch.float16)
    args = {
        "a_ptr": a,
        "b_ptr": a,
        "out_ptr": a,
        "M": size,
        "N": size,
        "K": size,
        "stride_am": size,
        "stride_ak": 1,
        "stride_bk": size,
        "stride_bn": 1,
        "stride_om": size,
        "stride_on": 1,
    }
    return [
        Example(
            name=f"matmul_cfg{i}",
            kernel=_matmul_kernel,
            args=args,
            constexprs={"FP16_ACC": False, **{key: c[key] for key in _META_KEYS}},
            num_warps=c["num_warps"],
            num_stages=c["num_stages"],
        )
        for i, c in enumerate(CONFIGS)
    ]
