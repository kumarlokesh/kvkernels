"""Gather and scatter of KV blocks, compared byte for byte with the PyTorch reference.

Planes come from caches built by kvkernels.layouts the way each engine
allocates them: vLLM's four orders whose blocks are one run per layer (two of
them store all layers of a block together, so a layer's plane is strided) and
SGLang's default and heads-first buffers (separate K and V planes).

Tests without the gpu mark keep every tensor at or below 65,536 elements so they
run under the Triton interpreter.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache
from kvkernels.layouts import planes as layout_planes_of
from kvkernels.ops.kv_gather import (
    _word_bytes,
    gather_kv_blocks,
    scatter_kv_blocks,
)
from kvkernels.reference.kv_gather import (
    gather_kv_blocks_ref,
    scatter_kv_blocks_ref,
)
from kvkernels.tolerance import assert_exact

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING else "cuda"

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)

Planes = list["torch.Tensor"]


def random_tensor(shape: tuple[int, ...], dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Random bytes viewed as dtype, so every bit pattern (NaNs included) appears."""
    gen = torch.Generator().manual_seed(seed)
    numel = 1
    for size in shape:
        numel *= size
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=gen)
    return raw.view(dtype).view(shape).to(DEVICE)


def layout_planes(
    name: str,
    blocks: int,
    block_size: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    layers: int = 1,
    layer: int = 0,
    seed: int = 0,
) -> Planes:
    """The planes of one layer of an engine-shaped cache, filled with random bytes."""
    layout = LAYOUTS[name]
    spec = CacheSpec(layers, blocks, block_size, heads, head_dim, dtype)
    cache = make_cache(layout, spec, DEVICE)
    gen = torch.Generator().manual_seed(seed)
    for tensors in cache:
        for plane in layout_planes_of(layout, spec, tensors):
            raw = plane.view(torch.uint8)
            raw.copy_(torch.randint(0, 256, tuple(raw.shape), dtype=torch.uint8, generator=gen))
    return layout_planes_of(layout, spec, cache[layer])


def vllm_joint(
    blocks: int, block_size: int, heads: int, head_dim: int, dtype: torch.dtype
) -> Planes:
    return layout_planes("vllm_LBNHC", blocks, block_size, heads, head_dim, dtype)


# (blocks, block or page size, kv heads, head_dim, layers, layer); every plane ≤ 32 blocks ·
# 2,048 values, every tensor ≤ 65,536 elements.
SMALL_SHAPES = {
    "vllm_LBNHC": (32, 4, 2, 64, 1, 0),
    "vllm_LBHNC": (32, 4, 2, 64, 1, 0),
    "vllm_BLNHC": (16, 4, 2, 64, 3, 1),
    "vllm_BLHNC": (16, 4, 2, 64, 3, 1),
    "sglang_nhd": (32, 1, 4, 64, 1, 0),
    "sglang_hnd": (32, 4, 2, 64, 1, 0),
}


def make_planes(layout: str, dtype: torch.dtype = torch.float16) -> Planes:
    blocks, block_size, heads, head_dim, layers, layer = SMALL_SHAPES[layout]
    return layout_planes(layout, blocks, block_size, heads, head_dim, dtype, layers, layer)


def ids_tensor(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device=DEVICE)


def empty_rows(planes: Planes, n: int) -> Planes:
    return [torch.empty((n, *p.shape[1:]), dtype=p.dtype, device=DEVICE) for p in planes]


def check_gather(planes: Planes, ids: list[int]) -> None:
    block_ids = ids_tensor(ids)
    outs = empty_rows(planes, len(ids))
    expected = empty_rows(planes, len(ids))
    gather_kv_blocks(planes, block_ids, outs, validate=True)
    gather_kv_blocks_ref(planes, block_ids, expected)
    for out, ref in zip(outs, expected, strict=True):
        assert_exact(out, ref)


def check_scatter(planes: Planes, ids: list[int]) -> None:
    block_ids = ids_tensor(ids)
    srcs = [random_tensor((len(ids), *p.shape[1:]), p.dtype, seed=9) for p in planes]
    expected = [p.clone() for p in planes]
    scatter_kv_blocks(srcs, block_ids, planes, validate=True)
    scatter_kv_blocks_ref(srcs, block_ids, expected)
    for plane, ref in zip(planes, expected, strict=True):
        assert_exact(plane, ref)


def id_patterns(blocks: int) -> dict[str, list[int]]:
    perm = torch.randperm(blocks, generator=torch.Generator().manual_seed(3)).tolist()
    return {
        "one": [blocks // 2],
        "permutation": perm,
        "descending": list(range(blocks - 1, -1, -1)),
        "largest": [blocks - 1, 0],
        "subset": perm[: blocks // 3],
    }


@pytest.mark.parametrize("pattern", ["one", "permutation", "descending", "largest", "subset"])
@pytest.mark.parametrize("layout", list(SMALL_SHAPES))
def test_gather_matches_reference(layout: str, pattern: str) -> None:
    planes = make_planes(layout)
    check_gather(planes, id_patterns(planes[0].shape[0])[pattern])


@pytest.mark.parametrize("pattern", ["one", "permutation", "descending", "largest", "subset"])
@pytest.mark.parametrize("layout", list(SMALL_SHAPES))
def test_scatter_matches_reference(layout: str, pattern: str) -> None:
    planes = make_planes(layout)
    check_scatter(planes, id_patterns(planes[0].shape[0])[pattern])


def test_gather_copies_a_repeated_id_to_every_row() -> None:
    check_gather(make_planes("vllm_LBNHC"), [3, 7, 3, 3])


def test_empty_block_ids_leave_everything_untouched() -> None:
    planes = make_planes("sglang_nhd")
    before = [p.clone() for p in planes]
    outs = empty_rows(planes, 0)
    gather_kv_blocks(planes, ids_tensor([]), outs)
    scatter_kv_blocks(outs, ids_tensor([]), planes)
    for plane, ref in zip(planes, before, strict=True):
        assert_exact(plane, ref)


def test_gather_then_scatter_into_other_blocks_round_trips() -> None:
    planes = make_planes("vllm_LBNHC")
    sent = [12, 5, 30, 0]
    received = [1, 31, 7, 20]  # the receiver's own free blocks
    staging = empty_rows(planes, len(sent))
    gather_kv_blocks(planes, ids_tensor(sent), staging)
    target = [torch.zeros_like(p) for p in planes]
    scatter_kv_blocks(staging, ids_tensor(received), target)
    back = empty_rows(planes, len(sent))
    gather_kv_blocks(target, ids_tensor(received), back)
    original = empty_rows(planes, len(sent))
    gather_kv_blocks_ref(planes, ids_tensor(sent), original)
    for got, ref in zip(back, original, strict=True):
        assert_exact(got, ref)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        # 2·40 values · 2 heads · 4 slots · 2 bytes = 1,280 bytes = 160 words: 2½ tiles.
        ((8, 4, 2, 40), torch.float16),
        # 80 values · 1 head · 1 slot · 2 bytes = 160 bytes = 20 words: one partial tile.
        ((8, 1, 1, 80), torch.float16),
    ],
)
def test_block_that_is_not_a_whole_number_of_tiles(
    shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    planes = vllm_joint(*shape, dtype)
    check_gather(planes, [6, 0, 3])
    check_scatter(planes, [6, 0, 3])


@pytest.mark.parametrize(
    ("shape", "dtype", "width"),
    [
        ((16, 1, 1, 3), torch.float16, 4),  # 2·3 values · 2 bytes = 12 bytes
        ((16, 1, 1, 3), torch.float32, 8),  # 24 bytes
        ((16, 3, 1, 1), torch.float16, 4),  # 3 slots · 2 values · 2 bytes = 12 bytes
        ((16, 1, 1, 1), torch.float16, 4),  # 4 bytes
    ],
)
def test_narrow_word_widths(
    shape: tuple[int, int, int, int], dtype: torch.dtype, width: int
) -> None:
    planes = vllm_joint(*shape, dtype)
    block_bytes = planes[0][0].numel() * planes[0].element_size()
    assert _word_bytes(block_bytes, block_bytes) == width
    check_gather(planes, [15, 2, 9])
    check_scatter(planes, [15, 2, 9])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16])
def test_odd_block_sizes_copy_with_one_or_two_byte_words(dtype: torch.dtype) -> None:
    plane = random_tensor((10, 7), dtype)  # 7 or 14 bytes per block
    check_gather([plane], [9, 1, 4])
    check_scatter([plane], [9, 1, 4])


def test_word_width_divides_sizes_and_addresses() -> None:
    assert _word_bytes(64, 128, 4096, 8192) == 8
    assert _word_bytes(64, 128, 4100) == 4
    assert _word_bytes(64, 128, 4098) == 2
    assert _word_bytes(63) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_interpreter_dtypes(dtype: torch.dtype) -> None:
    planes = layout_planes("sglang_hnd", 16, 2, 2, 64, dtype)
    check_gather(planes, [4, 15, 0])
    check_scatter(planes, [4, 15, 0])


def test_validate_rejects_an_out_of_range_id() -> None:
    planes = make_planes("vllm_LBNHC")
    blocks = planes[0].shape[0]
    with pytest.raises(IndexError):
        gather_kv_blocks(planes, ids_tensor([0, blocks]), empty_rows(planes, 2), validate=True)
    with pytest.raises(IndexError):
        scatter_kv_blocks(empty_rows(planes, 1), ids_tensor([-1]), planes, validate=True)


def test_validate_rejects_a_repeated_id_in_scatter() -> None:
    planes = make_planes("vllm_LBNHC")
    with pytest.raises(ValueError, match="repeated"):
        scatter_kv_blocks(empty_rows(planes, 2), ids_tensor([4, 4]), planes, validate=True)


def test_rejects_bad_arguments() -> None:
    planes = make_planes("vllm_LBNHC")
    ids = ids_tensor([1, 2])
    outs = empty_rows(planes, 2)
    with pytest.raises(ValueError, match="int32"):
        gather_kv_blocks(planes, ids.long(), outs)
    with pytest.raises(ValueError, match="1-D"):
        gather_kv_blocks(planes, ids.view(1, 2), outs)
    with pytest.raises(ValueError, match="planes and"):
        gather_kv_blocks(planes, ids, outs + outs)
    with pytest.raises(ValueError, match="shape"):
        gather_kv_blocks(planes, ids, empty_rows(planes, 3))
    with pytest.raises(ValueError, match="dtype"):
        gather_kv_blocks(planes, ids, [outs[0].view(torch.int16)])
    with pytest.raises(ValueError, match="contiguous"):
        wide = torch.empty((2, *planes[0].shape[1:-1], 2 * planes[0].shape[-1]), device=DEVICE)
        gather_kv_blocks(planes, ids, [wide.half()[..., ::2]])
    with pytest.raises(ValueError, match="one contiguous run"):
        gather_kv_blocks([planes[0].transpose(1, 2)], ids, [outs[0].transpose(1, 2).contiguous()])


def test_rejects_an_output_that_overlaps_its_plane() -> None:
    planes = make_planes("vllm_LBNHC")
    with pytest.raises(ValueError, match="overlaps"):
        gather_kv_blocks(planes, ids_tensor([5, 6]), [planes[0][:2]])
    with pytest.raises(ValueError, match="overlaps"):
        scatter_kv_blocks([planes[0][:2]], ids_tensor([5, 6]), planes)


# ---------------------------------------------------------------------------
# GPU tier: full shapes, every dtype, CUDA graph capture, 64-bit offsets.
# ---------------------------------------------------------------------------

FULL_SHAPES = [
    ("vllm_LBNHC", (2048, 16, 8, 128)),
    ("vllm_LBNHC", (2048, 32, 8, 64)),
    ("vllm_LBHNC", (2048, 16, 8, 128)),
    ("vllm_BLNHC", (256, 16, 8, 128)),
    ("sglang_nhd", (8192, 1, 8, 128)),
    ("sglang_nhd", (2048, 16, 1, 128)),
    ("sglang_hnd", (2048, 16, 8, 128)),
]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(("layout", "shape"), FULL_SHAPES)
def test_full_shapes(layout: str, shape: tuple[int, int, int, int], dtype: torch.dtype) -> None:
    layers, layer = (3, 1) if layout.startswith("vllm_BL") else (1, 0)
    planes = layout_planes(layout, *shape, dtype, layers, layer)
    blocks = planes[0].shape[0]
    ids = torch.randperm(blocks, generator=torch.Generator().manual_seed(5))[:512].tolist()
    check_gather(planes, ids)
    check_scatter(planes, ids)


@pytest.mark.gpu
def test_capture_in_a_cuda_graph() -> None:
    planes = vllm_joint(256, 16, 8, 128, torch.float16)
    ids = ids_tensor([200, 3, 77, 255])
    outs = empty_rows(planes, 4)
    srcs = [random_tensor(tuple(o.shape), o.dtype, seed=4) for o in outs]
    # The first call picks the tile size; capture only replays launches.
    gather_kv_blocks(planes, ids, outs)
    scatter_kv_blocks(srcs, ids, planes)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter_kv_blocks(srcs, ids, planes)
        gather_kv_blocks(planes, ids, outs)
    for out in outs:
        out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for out, src in zip(outs, srcs, strict=True):
        assert_exact(out, src)


@pytest.mark.gpu
def test_offsets_past_two_to_the_31_words() -> None:
    # 4,097-byte blocks force 1-byte words, so the last block starts past word 2**31.
    blocks = (2**31) // 4097 + 8
    plane = torch.empty((blocks, 4097), dtype=torch.uint8, device="cuda")
    plane[-1].random_(0, 256)
    plane[0].random_(0, 256)
    check_gather([plane], [blocks - 1, 0])
    src = random_tensor((1, 4097), torch.uint8, seed=6)
    scatter_kv_blocks([src], ids_tensor([blocks - 2]), [plane])
    assert_exact(plane[blocks - 2 : blocks - 1], src)
