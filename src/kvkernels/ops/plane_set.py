"""Gather and scatter of KV blocks across every plane of a cache, in one launch.

gather_kv_blocks copies one plane per launch. An offload copies every layer, 32
to 64 planes, and outside a CUDA graph the launches then cost more CPU time than
the copies take on the GPU (71 us of CPU against 156 us of GPU per plane, 512
ids, on an L4). A plane set is built once from the planes and the row buffers
they copy to and from: it checks them once and writes a table of their
addresses to the GPU. Each call then checks only the ids and makes one launch:

    plane table (int64, one row per plane)
    │ plane address │ rows address │ plane stride │ rows stride │ words per block │

    grid (ids, tiles, planes): program (i, j, p) copies tile j of block_ids[i] in plane p;
    the tile axis covers the largest block, and a program past its plane's
    last tile copies nothing

Blocks are copied as raw words, as in gather_kv_blocks; one word width, the
widest that divides every block size, stride and address of the set, serves
the whole launch, so a plane with an odd block size slows every plane in its
set; such planes go in a set of their own. The set keeps references to its
tensors so the addresses in
the table stay valid; a tensor whose storage changes later (resize_, set_) is
not detected, so build a new set. The first call times the tile sizes; later
calls allocate nothing and copy nothing from the host, so they can be captured
in a CUDA graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from kvkernels.ops._registry import registered_op
from kvkernels.ops.kv_gather import (
    _CONFIGS,
    _INTERPRETER_TILE,
    _WORD_TYPES,
    _block_is_contiguous,
    _block_numel,
    _byte_span,
    _check_tiles,
    _interpreting,
    _num_ids,
    _validate_ids,
    _word_bytes,
)

# The plane table's columns: plane address, rows address, plane stride, rows
# stride, words per block. The kernel reads them in this order.
_BLOCK_WORDS = 4
# The plane axis is grid axis 2, which CUDA limits to 65,535 programs.
_MAX_PLANES = 65_535


@triton.jit
def _copy_plane_set_kernel(  # type: ignore[no-untyped-def]
    table_ptr,
    ids_ptr,
    max_words,
    IDS_INDEX_PLANE: tl.constexpr,
    ALIGN: tl.constexpr,
    WORD: tl.constexpr,
    TILE: tl.constexpr,
):
    """Copy tile `program_id(1)` of one block of plane `program_id(2)`; sizes are in words."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    entry = table_ptr + tl.program_id(2).to(tl.int64) * 5
    plane_address = tl.load(entry + 0)
    rows_address = tl.load(entry + 1)
    plane_stride = tl.load(entry + 2)
    rows_stride = tl.load(entry + 3)
    block_words = tl.load(entry + 4)
    if ALIGN > 1:
        # Every block start in the set is 16-byte aligned.
        plane_address = tl.multiple_of(plane_address, 16)
        rows_address = tl.multiple_of(rows_address, 16)
        plane_stride = tl.multiple_of(plane_stride, ALIGN)
        rows_stride = tl.multiple_of(rows_stride, ALIGN)
    plane_ptr = plane_address.to(tl.pointer_type(WORD))
    rows_ptr = rows_address.to(tl.pointer_type(WORD))
    # 64-bit, because block id · block stride can pass 2**31 words.
    block = tl.load(ids_ptr + row).to(tl.int64)
    offs = tile * TILE + tl.arange(0, TILE)
    mask = offs < block_words
    in_plane = plane_ptr + block * plane_stride + offs
    in_rows = rows_ptr + row * rows_stride + offs
    if IDS_INDEX_PLANE:  # gather: plane blocks into rows
        tl.store(in_rows, tl.load(in_plane, mask=mask), mask=mask)
    else:  # scatter: rows into plane blocks
        tl.store(in_plane, tl.load(in_rows, mask=mask), mask=mask)


_copy_plane_set_autotuned = triton.autotune(configs=_CONFIGS, key=["max_words"])(
    _copy_plane_set_kernel
)

# The tile the autotuner chose, by (largest block in words, word width, direction).
_TUNED: dict[tuple[int, int, bool], triton.Config] = {}


def _check_no_overlaps(planes: Sequence[torch.Tensor], rows: Sequence[torch.Tensor]) -> None:
    """No row buffer may overlap a plane or another row buffer; planes may overlap each other.

    Sorted by start address, a buffer overlaps an earlier one exactly when it
    starts before the furthest end seen so far, so one pass finds every pair.
    """
    spans = [(*_byte_span(t), "plane", p) for p, t in enumerate(planes)]
    spans += [(*_byte_span(t), "rows", p) for p, t in enumerate(rows)]
    plane_end = rows_end = -1
    last_plane = last_rows = -1
    for start, end, kind, p in sorted(s for s in spans if s[0] < s[1]):
        if kind == "rows":
            if start < plane_end:
                raise ValueError(f"rows {p} overlaps plane {last_plane}")
            if start < rows_end:
                raise ValueError(f"rows {p} overlaps rows {last_rows}")
            if end > rows_end:
                rows_end, last_rows = end, p
        else:
            if start < rows_end:
                raise ValueError(f"rows {last_rows} overlaps plane {p}")
            if end > plane_end:
                plane_end, last_plane = end, p


def _launch(
    table: torch.Tensor,
    block_ids: torch.Tensor,
    num_planes: int,
    max_words: int,
    width: int,
    align: int,
    gather: bool,
) -> None:
    num_ids = block_ids.shape[0]
    args = (table, block_ids, max_words)
    meta = {"IDS_INDEX_PLANE": gather, "ALIGN": align, "WORD": _WORD_TYPES[width]}
    if _interpreting():
        grid = (num_ids, triton.cdiv(max_words, _INTERPRETER_TILE), num_planes)
        _copy_plane_set_kernel[grid](*args, **meta, TILE=_INTERPRETER_TILE)
        return
    key = (max_words, width, gather)
    config = _TUNED.get(key)
    if config is None:

        def tuned_grid(meta: dict[str, int]) -> tuple[int, int, int]:
            return (num_ids, triton.cdiv(max_words, meta["TILE"]), num_planes)

        _copy_plane_set_autotuned[tuned_grid](*args, **meta)
        _TUNED[key] = _copy_plane_set_autotuned.best_config
        return
    tile = int(config.kwargs["TILE"])
    _copy_plane_set_kernel[(num_ids, triton.cdiv(max_words, tile), num_planes)](
        *args, **meta, TILE=tile, num_warps=config.num_warps, num_stages=config.num_stages
    )


def _check_call(table: torch.Tensor, rows: list[torch.Tensor], block_ids: torch.Tensor) -> int:
    num_ids = _num_ids(block_ids)
    if block_ids.device != table.device:
        raise ValueError(f"block_ids is on {block_ids.device}, the planes on {table.device}")
    capacity = rows[0].shape[0]
    if num_ids > capacity:
        raise ValueError(f"{num_ids} block ids exceed the plane set's capacity of {capacity}")
    return num_ids


# Registered as PyTorch custom ops; eager calls skip the dispatch (kvkernels.ops._registry).
@registered_op("kvkernels::gather_plane_set", mutates_args=("rows",))
def _gather_set_op(
    table: torch.Tensor,
    planes: list[torch.Tensor],
    rows: list[torch.Tensor],
    block_ids: torch.Tensor,
    max_words: int,
    width: int,
    align: int,
    validate: bool,
) -> None:
    if _check_call(table, rows, block_ids) == 0:
        return
    if validate:
        _validate_ids(planes, block_ids, unique=False)
    _launch(table, block_ids, len(planes), max_words, width, align, gather=True)


@_gather_set_op.register_fake
def _gather_set_fake(
    table: torch.Tensor,
    planes: list[torch.Tensor],
    rows: list[torch.Tensor],
    block_ids: torch.Tensor,
    max_words: int,
    width: int,
    align: int,
    validate: bool,
) -> None:
    return None


@registered_op("kvkernels::scatter_plane_set", mutates_args=("planes",))
def _scatter_set_op(
    table: torch.Tensor,
    planes: list[torch.Tensor],
    rows: list[torch.Tensor],
    block_ids: torch.Tensor,
    max_words: int,
    width: int,
    align: int,
    validate: bool,
) -> None:
    if _check_call(table, rows, block_ids) == 0:
        return
    if validate:
        _validate_ids(planes, block_ids, unique=True)
    _launch(table, block_ids, len(planes), max_words, width, align, gather=False)


@_scatter_set_op.register_fake
def _scatter_set_fake(
    table: torch.Tensor,
    planes: list[torch.Tensor],
    rows: list[torch.Tensor],
    block_ids: torch.Tensor,
    max_words: int,
    width: int,
    align: int,
    validate: bool,
) -> None:
    return None


class PlaneSet:
    """Planes and the row buffers they copy to and from, checked once, copied in one launch.

    planes[p]: [num_blocks_p, *block_shape_p]; each block one contiguous run of
        bytes. Planes may be views of one buffer and may interleave.
    rows[p]: [capacity, *block_shape_p], contiguous, same dtype as planes[p];
        the same capacity for every p. No row buffer may overlap a plane or
        another row buffer.
    A scatter into two planes that share a KV block leaves that block undefined,
    as a repeated id does.
    """

    def __init__(self, planes: Sequence[torch.Tensor], rows: Sequence[torch.Tensor]) -> None:
        planes, rows = list(planes), list(rows)
        if len(planes) != len(rows):
            raise ValueError(f"got {len(planes)} planes and {len(rows)} row buffers")
        if not planes:
            raise ValueError("need at least one plane")
        if len(planes) > _MAX_PLANES:
            raise ValueError(f"{len(planes)} planes; one launch takes at most {_MAX_PLANES}")
        device = planes[0].device
        if device.type != "cuda" and not _interpreting():
            raise ValueError(f"tensors must be on a CUDA device, got {device}")
        capacity = rows[0].shape[0] if rows[0].dim() else 0
        layout: list[int] = []  # every size and address the word width must divide
        entries: list[list[int]] = []
        for p, (plane, row) in enumerate(zip(planes, rows, strict=True)):
            where = f"plane {p}"
            if plane.dim() < 1:
                raise ValueError(f"{where}: needs a first dimension that indexes blocks")
            if plane.device != device or row.device != device:
                raise ValueError(f"{where}: every tensor must be on {device}")
            if row.dtype != plane.dtype:
                raise ValueError(f"{where}: rows dtype {row.dtype} != plane {plane.dtype}")
            if row.dim() < 1 or row.shape[0] != capacity:
                raise ValueError(f"{where}: rows capacity {row.shape[:1]} != {capacity}")
            expected = (capacity, *plane.shape[1:])
            if tuple(row.shape) != expected:
                raise ValueError(f"{where}: rows shape {tuple(row.shape)} != {expected}")
            if not row.is_contiguous():
                raise ValueError(f"{where}: rows must be contiguous")
            if not _block_is_contiguous(plane):
                raise ValueError(f"{where}: each block must be one contiguous run of bytes")
            if plane.shape[0] == 0:
                raise ValueError(f"{where}: has no blocks")
            element = plane.element_size()
            block_bytes = _block_numel(plane) * element
            stride_bytes = plane.stride(0) * element
            layout += [block_bytes, stride_bytes, plane.data_ptr(), row.data_ptr()]
            entries.append([plane.data_ptr(), row.data_ptr(), stride_bytes, block_bytes])
        _check_no_overlaps(planes, rows)
        _check_tiles(planes)
        width = _word_bytes(*layout)
        table = [
            [plane_address, rows_address, stride // width, block // width, block // width]
            for plane_address, rows_address, stride, block in entries
        ]
        self.planes = planes
        self.rows = rows
        self.capacity = capacity
        self._table = torch.tensor(table, dtype=torch.int64).to(device)
        self._width = width
        # 16-byte aligned starts imply 8-byte words: strides are then multiples of two words.
        self._align = 16 // width if all(x % 16 == 0 for x in layout) else 1
        self._max_words = max(row[_BLOCK_WORDS] for row in table)

    def gather(self, block_ids: torch.Tensor, *, validate: bool = False) -> None:
        """rows[p][i] = planes[p][block_ids[i]] for every plane p and i < len(block_ids).

        block_ids: int32, 1-D, contiguous, on the planes' device, at most
        capacity ids; repeats allowed. validate=True checks the range (it
        synchronizes with the host). Runs as kvkernels::gather_plane_set.
        """
        _gather_set_op(
            self._table,
            self.planes,
            self.rows,
            block_ids,
            self._max_words,
            self._width,
            self._align,
            validate,
        )

    def scatter(self, block_ids: torch.Tensor, *, validate: bool = False) -> None:
        """planes[p][block_ids[i]] = rows[p][i] for every plane p and i < len(block_ids).

        Ids must be distinct; validate=True checks range and distinctness (it
        synchronizes with the host). Runs as kvkernels::scatter_plane_set.
        """
        _scatter_set_op(
            self._table,
            self.planes,
            self.rows,
            block_ids,
            self._max_words,
            self._width,
            self._align,
            validate,
        )
