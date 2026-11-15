"""Paged KV-cache layouts of vLLM and SGLang, and the views kvkernels takes of them.

A layout is a memory order (the order in which layer, block, head, slot and
content appear in memory) and a block stride (how far apart consecutive blocks
of one layer are). For each layout this module builds a cache the way the engine
allocates it and returns three views of the same memory, never a copy:

    engine-shaped cache ──▶ planes      each KV block one contiguous run of bytes (gather, scatter)
                       ├──▶ kv_views    K and V as [blocks, slots, kv_heads, head_dim], every layout
                       └──▶ make_index  the block table or slot list the engine's attention reads

vLLM keeps one buffer whose per-layer views have axes (block, head, slot,
content), where content is a head's K values followed by its V values; the
memory order decides the strides. SGLang keeps one K and one V buffer per layer,
indexed by slot (default) or by page then head (heads first), and reserves page 0
for writes from padded tokens.

Block 0 is never given to a request: vLLM keeps it as its null block and SGLang
reserves it. Unused block-table entries are filled with it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "LAYOUTS",
    "CacheSpec",
    "PagedKVLayout",
    "kv_views",
    "make_cache",
    "make_index",
    "planes",
]

VLLM_REPO = "vllm-project/vllm"
VLLM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"  # v0.32.0
SGLANG_REPO = "sgl-project/sglang"
SGLANG_COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"  # v0.5.22

RESERVED_BLOCK = 0

# K and V views: an fp8 cache is viewed as float8_e4m3fn (engines store its
# bytes as uint8; view the cache as float8_e4m3fn first).
_VIEW_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn)


@dataclass(frozen=True)
class Source:
    """The upstream code a layout follows."""

    repo: str
    commit: str
    path: str
    symbol: str


@dataclass(frozen=True)
class CacheSpec:
    num_layers: int
    num_blocks: int  # KV blocks (vLLM) or pages (SGLang), including reserved block 0
    block_size: int  # tokens per KV block or page
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype


@dataclass(frozen=True)
class PagedKVLayout:
    name: str
    engine: Literal["vllm", "sglang"]
    kv_split: Literal["joint", "separate"]  # K and V in one buffer, or in two
    index_kind: Literal["block_table", "slot_list"]
    memory_order: str  # outermost first; vLLM: letters L B H N C; SGLang: "nhd" or "hnd"
    sources: tuple[Source, ...]


def _vllm(path: str, symbol: str) -> Source:
    return Source(VLLM_REPO, VLLM_COMMIT, path, symbol)


def _sglang(path: str, symbol: str) -> Source:
    return Source(SGLANG_REPO, SGLANG_COMMIT, path, symbol)


_VLLM_SOURCES = (
    _vllm("vllm/v1/kv_cache_layout.py", "KVCacheLayout"),
    _vllm("vllm/v1/kv_cache_interface.py", "create_kv_cache_views"),
    _vllm("vllm/v1/kv_cache_interface.py", "compute_layout_strides"),
    _vllm("vllm/v1/attention/backends/triton_attn.py", "TritonAttentionImpl.forward"),
    _vllm("vllm/v1/worker/gpu/block_table.py", "BlockTables"),
    _vllm("vllm/v1/core/block_pool.py", "null_block"),
)
_SGLANG_SOURCES = (
    _sglang("python/sglang/srt/mem_cache/memory_pool.py", "MHATokenToKVPool._kv_buffer_shapes"),
    _sglang("python/sglang/srt/layers/attention/triton_backend.py", "kv_indptr, kv_indices"),
    _sglang("python/sglang/srt/mem_cache/kv_index_translator.py", "pack_read_stream"),
)


def _vllm_layout(order: str) -> PagedKVLayout:
    return PagedKVLayout(
        name=f"vllm_{order}",
        engine="vllm",
        kv_split="joint",
        index_kind="block_table",
        memory_order=order,
        sources=_VLLM_SOURCES,
    )


LAYOUTS: dict[str, PagedKVLayout] = {
    layout.name: layout
    for layout in (
        # Orders whose KV blocks are one contiguous run per layer. In vLLM's
        # letters: L layer, B block, H head, N slot, C content.
        _vllm_layout("LBNHC"),  # vLLM's default
        _vllm_layout("LBHNC"),
        _vllm_layout("BLNHC"),  # all layers of a block together: block stride spans every layer
        _vllm_layout("BLHNC"),
        PagedKVLayout("sglang_nhd", "sglang", "separate", "slot_list", "nhd", _SGLANG_SOURCES),
        PagedKVLayout("sglang_hnd", "sglang", "separate", "slot_list", "hnd", _SGLANG_SOURCES),
    )
}


def _check_spec(spec: CacheSpec) -> None:
    for name in ("num_layers", "num_blocks", "block_size", "num_kv_heads", "head_dim"):
        if getattr(spec, name) < 1:
            raise ValueError(f"{name} must be at least 1, got {getattr(spec, name)}")
    if spec.num_blocks < 2:
        raise ValueError("num_blocks must leave at least one block after reserved block 0")


def _vllm_strides(spec: CacheSpec, order: str) -> tuple[dict[str, int], dict[str, int]]:
    """Sizes and element strides of the five axes, walking the memory order from innermost."""
    sizes = {
        "L": spec.num_layers,
        "B": spec.num_blocks,
        "H": spec.num_kv_heads,
        "N": spec.block_size,
        "C": 2 * spec.head_dim,
    }
    strides: dict[str, int] = {}
    step = 1
    for axis in reversed(order):
        strides[axis] = step
        step *= sizes[axis]
    return sizes, strides


def make_cache(
    layout: PagedKVLayout, spec: CacheSpec, device: str | torch.device = "cpu"
) -> list[tuple[torch.Tensor, ...]]:
    """One entry per layer, holding the tensors the engine gives its kernels for that layer.

    vLLM: one view per layer, axes (block, head, slot, content), over one buffer
    shared by all layers. SGLang: (K buffer, V buffer); default order [slots,
    heads, head_dim] with slots = num_blocks · block_size, heads first [pages,
    heads, page_size, head_dim]. The contents are uninitialized.
    """
    _check_spec(spec)
    if layout.engine == "vllm":
        sizes, strides = _vllm_strides(spec, layout.memory_order)
        buffer = torch.empty(math.prod(sizes.values()), dtype=spec.dtype, device=device)
        logical = "LBHNC"
        everything = buffer.as_strided([sizes[a] for a in logical], [strides[a] for a in logical])
        return [(everything[layer],) for layer in range(spec.num_layers)]
    if layout.memory_order == "nhd":
        shape: tuple[int, ...] = (
            spec.num_blocks * spec.block_size,
            spec.num_kv_heads,
            spec.head_dim,
        )
    else:
        shape = (spec.num_blocks, spec.num_kv_heads, spec.block_size, spec.head_dim)
    return [
        (
            torch.empty(shape, dtype=spec.dtype, device=device),
            torch.empty(shape, dtype=spec.dtype, device=device),
        )
        for _ in range(spec.num_layers)
    ]


def _slot_before_head(layout: PagedKVLayout) -> bool:
    order = layout.memory_order
    return order.index("N") < order.index("H")


def planes(
    layout: PagedKVLayout, spec: CacheSpec, layer_tensors: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """The planes of one layer: first dimension indexes KV blocks, each block one contiguous run.

    vLLM: one plane, K and V together. SGLang: two planes, K then V.
    """
    if layout.engine == "vllm":
        (view,) = layer_tensors
        # The view's axes are (block, head, slot, content); reorder them to memory order.
        return [view.permute(0, 2, 1, 3) if _slot_before_head(layout) else view]
    if layout.memory_order == "nhd":
        return [
            t.view(spec.num_blocks, spec.block_size, spec.num_kv_heads, spec.head_dim)
            for t in layer_tensors
        ]
    return list(layer_tensors)


def kv_views(
    layout: PagedKVLayout, spec: CacheSpec, layer_tensors: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V of one layer, each [blocks, slots, kv_heads, head_dim]; views of the cache."""
    if spec.dtype not in _VIEW_DTYPES:
        raise ValueError(f"K and V views support {_VIEW_DTYPES}, got {spec.dtype}")
    d = spec.head_dim
    if layout.engine == "vllm":
        (plane,) = planes(layout, spec, layer_tensors)
        k, v = plane[..., :d], plane[..., d:]
        if not _slot_before_head(layout):  # plane is (block, head, slot, content)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
        return k, v
    k_plane, v_plane = planes(layout, spec, layer_tensors)
    if layout.memory_order == "hnd":  # (page, head, slot, head_dim)
        return k_plane.transpose(1, 2), v_plane.transpose(1, 2)
    return k_plane, v_plane


def make_index(
    layout: PagedKVLayout,
    spec: CacheSpec,
    request_blocks: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """The index the engine's attention kernel reads, for requests owning the given blocks.

    vLLM ("block_table", "seq_lens"): one int32 row per request; entries past a
    request's blocks hold no defined value in the engine and are filled here
    with block 0, the null block. "seq_lens" is int32.

    SGLang ("kv_indptr", "kv_indices"): one int64 slot per token, at every page
    size; kv_indptr (int32) is where each request's slots begin.
    """
    if len(request_blocks) != len(seq_lens):
        raise ValueError(f"{len(request_blocks)} block lists for {len(seq_lens)} lengths")
    for r, (blocks, length) in enumerate(zip(request_blocks, seq_lens, strict=True)):
        needed = math.ceil(length / spec.block_size)
        if length < 1 or len(blocks) < needed:
            raise ValueError(
                f"request {r}: {length} tokens need {needed} blocks, got {len(blocks)}"
            )
        for b in blocks:
            if not RESERVED_BLOCK < b < spec.num_blocks:
                raise ValueError(f"request {r}: block {b} is outside 1 … {spec.num_blocks - 1}")
    if layout.index_kind == "block_table":
        width = max(len(b) for b in request_blocks) if request_blocks else 0
        table = torch.full((len(request_blocks), width), RESERVED_BLOCK, dtype=torch.int32)
        for r, blocks in enumerate(request_blocks):
            table[r, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        return {
            "block_table": table.to(device),
            "seq_lens": torch.tensor(list(seq_lens), dtype=torch.int32, device=device),
        }
    slots: list[int] = []
    starts = [0]
    for blocks, length in zip(request_blocks, seq_lens, strict=True):
        slots.extend(
            blocks[t // spec.block_size] * spec.block_size + t % spec.block_size
            for t in range(length)
        )
        starts.append(len(slots))
    return {
        "kv_indptr": torch.tensor(starts, dtype=torch.int32, device=device),
        "kv_indices": torch.tensor(slots, dtype=torch.int64, device=device),
    }
