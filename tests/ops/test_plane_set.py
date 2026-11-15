"""A plane set: every layer's planes of a cache, gathered or scattered in one launch.

Each case builds a cache of three layers with kvkernels.layouts, takes the
planes of every layer, and compares the plane set's copies with the per-plane
PyTorch reference, byte for byte.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache
from kvkernels.layouts import planes as layout_planes_of
from kvkernels.ops.plane_set import PlaneSet
from kvkernels.reference.kv_gather import gather_kv_blocks_ref, scatter_kv_blocks_ref
from kvkernels.tolerance import assert_exact
from tests.ops.test_kv_gather import (
    DEVICE,
    SMALL_SHAPES,
    Planes,
    empty_rows,
    id_patterns,
    ids_tensor,
    random_tensor,
)

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)

LAYERS = 3


def all_layer_planes(
    name: str,
    blocks: int,
    block_size: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    layers: int = LAYERS,
    seed: int = 0,
) -> Planes:
    """Every layer's planes of an engine-shaped cache, in layer order, filled with random bytes."""
    layout = LAYOUTS[name]
    spec = CacheSpec(layers, blocks, block_size, heads, head_dim, dtype)
    cache = make_cache(layout, spec, DEVICE)
    gen = torch.Generator().manual_seed(seed)
    planes: Planes = []
    for tensors in cache:
        for plane in layout_planes_of(layout, spec, tensors):
            raw = plane.view(torch.uint8)
            raw.copy_(torch.randint(0, 256, tuple(raw.shape), dtype=torch.uint8, generator=gen))
            planes.append(plane)
    return planes


def small_planes(name: str, dtype: torch.dtype = torch.float16) -> Planes:
    blocks, block_size, heads, head_dim, _, _ = SMALL_SHAPES[name]
    return all_layer_planes(name, blocks, block_size, heads, head_dim, dtype)


def check_gather(planes: Planes, ids: list[int], capacity: int | None = None) -> None:
    capacity = len(ids) if capacity is None else capacity
    rows = [random_tensor((capacity, *p.shape[1:]), p.dtype, seed=7) for p in planes]
    before = [r.clone() for r in rows]
    PlaneSet(planes, rows).gather(ids_tensor(ids), validate=True)
    expected = empty_rows(planes, len(ids))
    gather_kv_blocks_ref(planes, ids_tensor(ids), expected)
    for row, ref, old in zip(rows, expected, before, strict=True):
        assert_exact(row[: len(ids)], ref)
        assert_exact(row[len(ids) :], old[len(ids) :])


def check_scatter(planes: Planes, ids: list[int]) -> None:
    rows = [random_tensor((len(ids), *p.shape[1:]), p.dtype, seed=9) for p in planes]
    expected = [p.clone() for p in planes]
    PlaneSet(planes, rows).scatter(ids_tensor(ids), validate=True)
    scatter_kv_blocks_ref(rows, ids_tensor(ids), expected)
    for plane, ref in zip(planes, expected, strict=True):
        assert_exact(plane, ref)


@pytest.mark.parametrize("pattern", ["one", "permutation", "descending", "largest", "subset"])
@pytest.mark.parametrize("layout", list(SMALL_SHAPES))
def test_gather_of_every_layer_matches_reference(layout: str, pattern: str) -> None:
    planes = small_planes(layout)
    check_gather(planes, id_patterns(planes[0].shape[0])[pattern])


@pytest.mark.parametrize("pattern", ["one", "permutation", "descending", "largest", "subset"])
@pytest.mark.parametrize("layout", list(SMALL_SHAPES))
def test_scatter_into_every_layer_matches_reference(layout: str, pattern: str) -> None:
    planes = small_planes(layout)
    check_scatter(planes, id_patterns(planes[0].shape[0])[pattern])


def test_interleaved_layers_of_one_buffer() -> None:
    # vLLM's BLNHC stores all layers of a block together: the layers' planes
    # cover the same byte range, block by block, without sharing a block.
    planes = small_planes("vllm_BLNHC")
    assert planes[0].data_ptr() < planes[1].data_ptr() < planes[0][1].data_ptr()
    ids = [15, 2, 9]
    check_gather(planes, ids)
    check_scatter(planes, ids)


def test_planes_of_different_block_sizes_and_dtypes_in_one_set() -> None:
    planes = [
        *all_layer_planes("vllm_LBNHC", 32, 4, 2, 64, torch.float16, layers=2),
        *all_layer_planes("sglang_nhd", 32, 1, 4, 64, torch.float8_e4m3fn, layers=1, seed=1),
        random_tensor((32, 3, 5), torch.uint8, seed=2),  # 15-byte blocks: 1-byte words
    ]
    ids = [31, 0, 7, 7]
    check_gather(planes, ids)
    check_scatter(planes, [31, 0, 7])


def test_fewer_ids_than_capacity_leave_the_other_rows_untouched() -> None:
    check_gather(small_planes("sglang_hnd"), [5, 1], capacity=6)


def test_no_ids_leave_everything_untouched() -> None:
    planes = small_planes("vllm_LBNHC")
    rows = [random_tensor((4, *p.shape[1:]), p.dtype, seed=3) for p in planes]
    planes_before = [p.clone() for p in planes]
    rows_before = [r.clone() for r in rows]
    plane_set = PlaneSet(planes, rows)
    plane_set.gather(ids_tensor([]))
    plane_set.scatter(ids_tensor([]))
    for t, old in zip([*planes, *rows], [*planes_before, *rows_before], strict=True):
        assert_exact(t, old)


def test_gather_then_scatter_into_other_blocks_round_trips() -> None:
    planes = small_planes("sglang_nhd")
    rows = empty_rows(planes, 4)
    plane_set = PlaneSet(planes, rows)
    plane_set.gather(ids_tensor([0, 1, 2, 3]))
    plane_set.scatter(ids_tensor([28, 29, 30, 31]))
    for plane in planes:
        assert_exact(plane[28:32], plane[0:4])


def test_capacity_is_the_rows_first_dimension() -> None:
    planes = small_planes("vllm_LBNHC")
    assert PlaneSet(planes, empty_rows(planes, 5)).capacity == 5


def test_rejects_more_ids_than_capacity() -> None:
    planes = small_planes("vllm_LBNHC")
    plane_set = PlaneSet(planes, empty_rows(planes, 2))
    with pytest.raises(ValueError, match="capacity"):
        plane_set.gather(ids_tensor([0, 1, 2]))


def test_validate_rejects_an_out_of_range_id_and_a_repeat_in_scatter() -> None:
    planes = small_planes("vllm_LBNHC")
    plane_set = PlaneSet(planes, empty_rows(planes, 2))
    with pytest.raises(IndexError):
        plane_set.gather(ids_tensor([0, planes[0].shape[0]]), validate=True)
    with pytest.raises(ValueError, match="repeated"):
        plane_set.scatter(ids_tensor([3, 3]), validate=True)


def test_rejects_bad_sets() -> None:
    planes = small_planes("vllm_LBNHC")
    rows = empty_rows(planes, 2)
    with pytest.raises(ValueError, match="planes"):
        PlaneSet(planes, rows[:-1])
    with pytest.raises(ValueError, match="at least one"):
        PlaneSet([], [])
    with pytest.raises(ValueError, match="capacity"):
        PlaneSet(planes, [*rows[:-1], empty_rows(planes, 3)[-1]])
    with pytest.raises(ValueError, match="dtype"):
        PlaneSet(planes[:1], [rows[0].view(torch.int16)])
    with pytest.raises(ValueError, match="shape"):
        PlaneSet(planes[:1], [rows[0][:, :1]])
    with pytest.raises(ValueError, match="int32"):
        PlaneSet(planes, rows).gather(ids_tensor([0]).long())


def test_rejects_rows_that_overlap_a_plane_or_each_other() -> None:
    planes = small_planes("vllm_LBNHC")
    with pytest.raises(ValueError, match="overlaps"):
        PlaneSet(planes[:1], [planes[0][:2]])
    buffer = torch.empty((4, *planes[0].shape[1:]), dtype=planes[0].dtype, device=DEVICE)
    with pytest.raises(ValueError, match="overlaps"):
        PlaneSet(planes[:2], [buffer[:2], buffer[1:3]])


def test_ops_are_registered_and_declare_what_they_write() -> None:
    gather = torch.ops.kvkernels.gather_plane_set.default._schema
    scatter = torch.ops.kvkernels.scatter_plane_set.default._schema
    assert [a.name for a in gather.arguments if a.alias_info and a.alias_info.is_write] == ["rows"]
    assert [a.name for a in scatter.arguments if a.alias_info and a.alias_info.is_write] == [
        "planes"
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("layout", ["vllm_LBNHC", "vllm_BLNHC", "sglang_nhd"])
def test_full_shapes_of_32_layers(layout: str, dtype: torch.dtype) -> None:
    block_size = 1 if layout == "sglang_nhd" else 16
    blocks = 4096 if block_size == 1 else 256
    planes = all_layer_planes(layout, blocks, block_size, 8, 128, dtype, layers=32)
    ids = torch.randperm(blocks, generator=torch.Generator().manual_seed(5))[:128].tolist()
    check_gather(planes, ids, capacity=160)
    check_scatter(planes, ids)


@pytest.mark.gpu
def test_capture_in_a_cuda_graph_and_replay_with_new_ids() -> None:
    planes = all_layer_planes("vllm_LBNHC", 64, 16, 8, 128, layers=4)
    rows = empty_rows(planes, 8)
    plane_set = PlaneSet(planes, rows)
    ids = ids_tensor([60, 3, 17, 40])
    plane_set.gather(ids)  # the first call picks the tile size
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plane_set.gather(ids)
    ids.copy_(ids_tensor([1, 2, 63, 0]))
    graph.replay()
    torch.cuda.synchronize()
    expected = empty_rows(planes, 4)
    gather_kv_blocks_ref(planes, ids, expected)
    for row, ref in zip(rows, expected, strict=True):
        assert_exact(row[:4], ref)


@pytest.mark.gpu
def test_runs_under_torch_compile() -> None:
    planes = all_layer_planes("sglang_nhd", 256, 1, 8, 128, layers=2)
    rows = empty_rows(planes, 3)
    plane_set = PlaneSet(planes, rows)
    ids = ids_tensor([255, 5, 17])

    def step(ids: torch.Tensor) -> None:
        plane_set.gather(ids)

    torch.compile(step, fullgraph=True)(ids)
    expected = empty_rows(planes, 3)
    gather_kv_blocks_ref(planes, ids, expected)
    for row, ref in zip(rows, expected, strict=True):
        assert_exact(row, ref)
