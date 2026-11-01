"""Gather and scatter of whole KV blocks between a paged KV cache and a contiguous buffer.

A **plane** is a tensor whose first dimension indexes KV blocks and in which each
block is one contiguous run of bytes. Consecutive blocks may be further apart than
one block (a block stride), as in caches that store all layers of a block together.
An engine's cache tensors for one layer reduce to one plane (K and V stored
together) or two (separate K and V tensors).

    plane (8 blocks)                             out (3 rows)
    │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │ ──▶ │ b5 │ b1 │ b6 │     block_ids = [5, 1, 6]

Gather copies the named blocks into consecutive rows of ``out``; scatter is the
same copy in the other direction. Both copy raw bytes, viewed as the widest
integer word that divides the block size, the block stride and every base
address, so the result is bit-exact for every dtype, including fp8 and any scale
bytes stored inside a block.

The kernel grid is (number of ids, tiles per block). Program (i, j) reads
``block_ids[i]`` and copies tile j, a run of ``TILE`` words, of that block:

                     tile 0   tile 1   tile 2
    i = 0 → block 5  (0,0)    (0,1)    (0,2)
    i = 1 → block 1  (1,0)    (1,1)    (1,2)      the last tile is masked when the
    i = 2 → block 6  (2,0)    (2,1)    (2,2)      block is not a whole number of tiles

Each plane is one launch. The ops allocate nothing and do not synchronize with
the host, so they can be captured in a CUDA graph. The first call for a new block
size times several tile sizes (autotuning); make that call before capturing.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from kvkernels.ops._registry import registered_op

# The integer type the kernel copies in, by word width in bytes.
_WORD_TYPES = {8: tl.int64, 4: tl.int32, 2: tl.int16, 1: tl.uint8}

# Element types the kernel can take as they are: it reinterprets their pointers as
# words. Others (fp8) are passed as a uint8 view, since Triton rejects some fp8
# types as kernel arguments on GPUs without fp8 support.
_DIRECT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

# CUDA allows at most 65,535 programs on grid axes 1 and 2.
_MAX_TILES_PER_BLOCK = 65_535

# TILE counts words; bytes per program = TILE · word size. The sizes below are
# for 8-byte words, the width every block of a standard layout gets.
_CONFIGS = [
    # 2 KiB: one SGLang page at page size 1 (8 KV heads · head_dim 128, fp16)
    # is exactly one program, with no masked lanes.
    triton.Config({"TILE": 256}, num_warps=4),
    # 4 KiB: one vLLM slot (8 KV heads · K and V · head_dim 128, fp16).
    triton.Config({"TILE": 512}, num_warps=4),
    # 8 KiB: a 64 KiB vLLM block becomes 8 programs, enough to fill the GPU
    # from a few hundred ids.
    triton.Config({"TILE": 1024}, num_warps=8),
    # 16 KiB: fewer, longer programs, for id lists long enough to fill the GPU
    # without splitting blocks finely.
    triton.Config({"TILE": 2048}, num_warps=8),
]
_SMALLEST_TILE = min(int(c.kwargs["TILE"]) for c in _CONFIGS)

# The Triton interpreter cannot time configurations, so it runs one fixed tile.
# It is small so that interpreter-sized blocks still span several tiles.
_INTERPRETER_TILE = 64


@triton.jit
def _copy_blocks_kernel(  # type: ignore[no-untyped-def]
    src_ptr,
    dst_ptr,
    ids_ptr,
    src_stride,
    dst_stride,
    block_words,
    IDS_INDEX_SRC: tl.constexpr,
    ALIGN: tl.constexpr,
    WORD: tl.constexpr,
    TILE: tl.constexpr,
):
    """Copy tile `program_id(1)` of one block. Strides and sizes are in words of type WORD."""
    src_ptr = src_ptr.to(tl.pointer_type(WORD), bitcast=True)
    dst_ptr = dst_ptr.to(tl.pointer_type(WORD), bitcast=True)
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    # 64-bit, because block id · block stride can pass 2**31 words.
    block = tl.load(ids_ptr + row).to(tl.int64)
    if ALIGN > 1:
        # Every block start is 16-byte aligned, so loads and stores can be 16 bytes wide.
        src_stride = tl.multiple_of(src_stride, ALIGN)
        dst_stride = tl.multiple_of(dst_stride, ALIGN)
    if IDS_INDEX_SRC:  # gather: block ids pick the source rows
        src_row = block
        dst_row = row
    else:  # scatter: block ids pick the destination rows
        src_row = row
        dst_row = block
    offs = tile * TILE + tl.arange(0, TILE)
    mask = offs < block_words
    words = tl.load(src_ptr + src_row * src_stride + offs, mask=mask)
    tl.store(dst_ptr + dst_row * dst_stride + offs, words, mask=mask)


_copy_blocks_autotuned = triton.autotune(configs=_CONFIGS, key=["block_words"])(_copy_blocks_kernel)


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _block_numel(t: torch.Tensor) -> int:
    return math.prod(t.shape[1:])


def _block_is_contiguous(t: torch.Tensor) -> bool:
    """True when each t[i] is one contiguous run and blocks do not overlap."""
    expected = 1
    for size, stride in zip(reversed(t.shape[1:]), reversed(t.stride()[1:]), strict=True):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return t.shape[0] <= 1 or t.stride(0) >= expected


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """[start, end) of the bytes a tensor can touch."""
    start = t.data_ptr()
    if t.numel() == 0:
        return start, start
    last = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride(), strict=True))
    return start, start + (last + 1) * t.element_size()


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    a0, a1 = _byte_span(a)
    b0, b1 = _byte_span(b)
    return a0 < b1 and b0 < a1


def _word_bytes(*sizes_and_addresses: int) -> int:
    """Widest word (8, 4, 2 or 1 bytes) that divides every size and address given."""
    for width in (8, 4, 2):
        if all(x % width == 0 for x in sizes_and_addresses):
            return width
    return 1


def _kernel_arg(t: torch.Tensor) -> torch.Tensor:
    """t itself, or for an element type the kernel cannot take, a uint8 view of t."""
    if t.dtype in _DIRECT_DTYPES:
        return t
    return t.view(t.shape[0], -1).view(torch.uint8)


def _num_ids(block_ids: torch.Tensor) -> int:
    if block_ids.dtype != torch.int32:
        raise ValueError(f"block_ids must be int32, got {block_ids.dtype}")
    if block_ids.dim() != 1 or not block_ids.is_contiguous():
        raise ValueError("block_ids must be a contiguous 1-D tensor")
    return int(block_ids.shape[0])


def _check_pairs(
    planes: Sequence[torch.Tensor], rows: Sequence[torch.Tensor], num_ids: int, rows_name: str
) -> torch.device:
    """Check planes against the contiguous buffers with one row per id; return the device."""
    if len(planes) != len(rows):
        raise ValueError(f"got {len(planes)} planes and {len(rows)} {rows_name}")
    if not planes:
        raise ValueError("need at least one plane")
    device = planes[0].device
    if device.type != "cuda" and not _interpreting():
        raise ValueError(f"tensors must be on a CUDA device, got {device}")
    for p, (plane, row) in enumerate(zip(planes, rows, strict=True)):
        where = f"plane {p}"
        if plane.dim() < 1:
            raise ValueError(f"{where}: needs a first dimension that indexes blocks")
        if plane.device != device or row.device != device:
            raise ValueError(f"{where}: every tensor must be on {device}")
        if row.dtype != plane.dtype:
            raise ValueError(f"{where}: {rows_name} dtype {row.dtype} != plane {plane.dtype}")
        expected = (num_ids, *plane.shape[1:])
        if tuple(row.shape) != expected:
            raise ValueError(f"{where}: {rows_name} shape {tuple(row.shape)} != {expected}")
        if not row.is_contiguous():
            raise ValueError(f"{where}: {rows_name} must be contiguous")
        if not _block_is_contiguous(plane):
            raise ValueError(f"{where}: each block must be one contiguous run of bytes")
        if num_ids and plane.shape[0] == 0:
            raise ValueError(f"{where}: has no blocks to copy")
    for p, plane in enumerate(planes):
        for q, row in enumerate(rows):
            if _overlaps(plane, row):
                raise ValueError(f"{rows_name} {q} overlaps plane {p}")
    return device


def _validate_ids(planes: Sequence[torch.Tensor], block_ids: torch.Tensor, unique: bool) -> None:
    """Range and uniqueness checks. They read block_ids on the host, so they synchronize."""
    lo = int(block_ids.min())
    hi = int(block_ids.max())
    for p, plane in enumerate(planes):
        if lo < 0 or hi >= plane.shape[0]:
            raise IndexError(f"block_ids span [{lo}, {hi}], plane {p} has {plane.shape[0]} blocks")
    if unique and torch.unique(block_ids).numel() != block_ids.numel():
        raise ValueError("block_ids has a repeated id; scatter needs distinct ids")


# The tile the autotuner chose, by (words per block, word width, source dtype,
# destination dtype). Later calls launch the kernel directly with it, which skips
# the autotuner's own per-call work.
_TUNED: dict[tuple[int, int, torch.dtype, torch.dtype], triton.Config] = {}


def _copy_plane(
    plane: torch.Tensor, rows: torch.Tensor, block_ids: torch.Tensor, gather: bool
) -> None:
    element = plane.element_size()
    block_bytes = _block_numel(plane) * element
    stride_bytes = plane.stride(0) * element
    layout = (block_bytes, stride_bytes, plane.data_ptr(), rows.data_ptr())
    width = _word_bytes(*layout)
    block_words = block_bytes // width
    # rows is contiguous, so its blocks are block_bytes apart.
    plane_arg, plane_stride = _kernel_arg(plane), stride_bytes // width
    rows_arg, rows_stride = _kernel_arg(rows), block_bytes // width
    # When every block start is 16-byte aligned (which implies 8-byte words),
    # tell the compiler that strides are multiples of two words.
    align = 16 // width if all(x % 16 == 0 for x in layout) else 1
    if gather:
        args = (plane_arg, rows_arg, block_ids, plane_stride, rows_stride, block_words)
    else:
        args = (rows_arg, plane_arg, block_ids, rows_stride, plane_stride, block_words)
    meta = {"IDS_INDEX_SRC": gather, "ALIGN": align, "WORD": _WORD_TYPES[width]}
    num_ids = block_ids.shape[0]
    if _interpreting():
        fixed_grid = (num_ids, triton.cdiv(block_words, _INTERPRETER_TILE))
        _copy_blocks_kernel[fixed_grid](*args, **meta, TILE=_INTERPRETER_TILE)
        return

    key = (block_words, width, args[0].dtype, args[1].dtype)
    config = _TUNED.get(key)
    if config is None:

        def grid(meta: dict[str, int]) -> tuple[int, int]:
            return (num_ids, triton.cdiv(block_words, meta["TILE"]))

        _copy_blocks_autotuned[grid](*args, **meta)
        _TUNED[key] = _copy_blocks_autotuned.best_config
        return
    tile = int(config.kwargs["TILE"])
    _copy_blocks_kernel[(num_ids, triton.cdiv(block_words, tile))](
        *args, **meta, TILE=tile, num_warps=config.num_warps, num_stages=config.num_stages
    )


def _check_tiles(planes: Sequence[torch.Tensor]) -> None:
    tile = _INTERPRETER_TILE if _interpreting() else _SMALLEST_TILE
    for p, plane in enumerate(planes):
        block_bytes = _block_numel(plane) * plane.element_size()
        # The narrowest possible word is 1 byte, so this bound holds for every width.
        if triton.cdiv(block_bytes, tile) > _MAX_TILES_PER_BLOCK:
            raise ValueError(f"plane {p}: a block of {block_bytes} bytes is too large")


# Registered as PyTorch custom ops; eager calls skip the dispatch (kvkernels.ops._registry).
@registered_op("kvkernels::gather_kv_blocks", mutates_args=("outs",))
def _gather_op(
    planes: list[torch.Tensor], block_ids: torch.Tensor, outs: list[torch.Tensor], validate: bool
) -> None:
    device = _check_pairs(planes, outs, _num_ids(block_ids), "outs")
    if block_ids.device != device:
        raise ValueError(f"block_ids is on {block_ids.device}, the planes on {device}")
    _check_tiles(planes)
    if block_ids.numel() == 0:
        return
    if validate:
        _validate_ids(planes, block_ids, unique=False)
    for plane, out in zip(planes, outs, strict=True):
        _copy_plane(plane, out, block_ids, gather=True)


@_gather_op.register_fake
def _gather_fake(
    planes: list[torch.Tensor], block_ids: torch.Tensor, outs: list[torch.Tensor], validate: bool
) -> None:
    return None


@registered_op("kvkernels::scatter_kv_blocks", mutates_args=("planes",))
def _scatter_op(
    srcs: list[torch.Tensor], block_ids: torch.Tensor, planes: list[torch.Tensor], validate: bool
) -> None:
    device = _check_pairs(planes, srcs, _num_ids(block_ids), "srcs")
    if block_ids.device != device:
        raise ValueError(f"block_ids is on {block_ids.device}, the planes on {device}")
    _check_tiles(planes)
    if block_ids.numel() == 0:
        return
    if validate:
        _validate_ids(planes, block_ids, unique=True)
    for src, plane in zip(srcs, planes, strict=True):
        _copy_plane(plane, src, block_ids, gather=False)


@_scatter_op.register_fake
def _scatter_fake(
    srcs: list[torch.Tensor], block_ids: torch.Tensor, planes: list[torch.Tensor], validate: bool
) -> None:
    return None


def gather_kv_blocks(
    planes: Sequence[torch.Tensor],
    block_ids: torch.Tensor,
    outs: Sequence[torch.Tensor],
    *,
    validate: bool = False,
) -> None:
    """outs[p][i] = planes[p][block_ids[i]], copied as raw bytes.

    planes[p]: [num_blocks, *block_shape]; each block one contiguous run of bytes.
    block_ids: int32, 1-D, contiguous; the same ids apply to every plane. An id
        may repeat. Every id must be in [0, num_blocks); pass validate=True to
        check that (it synchronizes with the host).
    outs[p]: [len(block_ids), *block_shape], contiguous, same dtype as planes[p],
        not overlapping any plane.
    Runs as the custom op kvkernels::gather_kv_blocks.
    """
    _gather_op(list(planes), block_ids, list(outs), validate)


def scatter_kv_blocks(
    srcs: Sequence[torch.Tensor],
    block_ids: torch.Tensor,
    planes: Sequence[torch.Tensor],
    *,
    validate: bool = False,
) -> None:
    """planes[p][block_ids[i]] = srcs[p][i], copied as raw bytes.

    Arguments as for gather_kv_blocks, with srcs in place of outs. Ids must be
    distinct: with a repeated id, the block's final contents are undefined.
    validate=True checks range and distinctness (it synchronizes with the host).
    Runs as the custom op kvkernels::scatter_kv_blocks.
    """
    _scatter_op(list(srcs), block_ids, list(planes), validate)
