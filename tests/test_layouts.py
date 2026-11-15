"""Checks that each layout matches the engine it follows, without running the engine.

Hand-worked strides. vLLM derives a cache's strides by walking its memory order
from the innermost axis outward, each stride the product of the sizes inside it
(compute_layout_strides at the pinned commit). For 2 layers, 3 blocks, 2 KV
heads, 4 slots per block and head_dim 8 (content = 16 values: K then V):

    order   C   N (slot)  H (head)  B (block)  L (layer)   per-layer view strides (B, H, N, C)
    LBNHC   1   32        16        128        384         (128, 16, 32, 1)
    LBHNC   1   16        64        128        384         (128, 64, 16, 1)
    BLNHC   1   32        16        256        128         (256, 16, 32, 1)
    BLHNC   1   16        64        256        128         (256, 64, 16, 1)

LBNHC, innermost first: C = 1; H = 16 (C's 16 values); N = 16 · 2 heads = 32;
B = 32 · 4 slots = 128; L = 128 · 3 blocks = 384. The others follow the same walk.

SGLang's buffers (MHATokenToKVPool._kv_buffer_shapes): default [slots, heads,
head_dim] with slots = blocks · page size; heads first [pages, heads, page size,
head_dim]. Both are plain contiguous tensors.

The engines' own way of reading K for (block b, slot n, head h): vLLM splits the
layer's view after moving slots before heads (transpose(1, 2), then the first
head_dim values of the content); SGLang default reads k_buffer[b · page size + n],
heads first reads k_buffer[b, :, n].
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from kvkernels.layouts import (
    LAYOUTS,
    SGLANG_COMMIT,
    VLLM_COMMIT,
    CacheSpec,
    kv_views,
    make_cache,
    make_index,
    planes,
)

SPEC = CacheSpec(
    num_layers=2, num_blocks=3, block_size=4, num_kv_heads=2, head_dim=8, dtype=torch.float16
)

EXPECTED_VLLM_STRIDES = {
    "vllm_LBNHC": ((128, 16, 32, 1), 384),
    "vllm_LBHNC": ((128, 64, 16, 1), 384),
    "vllm_BLNHC": ((256, 16, 32, 1), 128),
    "vllm_BLHNC": ((256, 64, 16, 1), 128),
}


def code(b: int, n: int, h: int, d: int, kv: int) -> float:
    """A distinct small integer per element, exact in fp16."""
    return float((((b * 4 + n) * 2 + h) * 8 + d) * 2 + kv)


def fill_through_views(name: str) -> list[tuple[torch.Tensor, ...]]:
    layout = LAYOUTS[name]
    cache = make_cache(layout, SPEC)
    for layer in cache:
        k, v = kv_views(layout, SPEC, layer)
        for b in range(SPEC.num_blocks):
            for n in range(SPEC.block_size):
                for h in range(SPEC.num_kv_heads):
                    for d in range(SPEC.head_dim):
                        k[b, n, h, d] = code(b, n, h, d, 0)
                        v[b, n, h, d] = code(b, n, h, d, 1)
    return cache


def engine_read(
    name: str, layer: tuple[torch.Tensor, ...], b: int, n: int, h: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V of (block, slot, head), read the way the engine's attention reads them."""
    d = SPEC.head_dim
    if name.startswith("vllm"):
        (view,) = layer
        # vLLM's own expression; PyTorch's stubs leave split untyped.
        key_cache, value_cache = view.transpose(1, 2).split(d, dim=-1)  # type: ignore[no-untyped-call]
        return key_cache[b, n, h], value_cache[b, n, h]
    k_buffer, v_buffer = layer
    if name == "sglang_nhd":
        slot = b * SPEC.block_size + n
        return k_buffer[slot, h], v_buffer[slot, h]
    return k_buffer[b, h, n], v_buffer[b, h, n]


def test_every_layout_cites_the_pinned_engine_code() -> None:
    for layout in LAYOUTS.values():
        assert layout.sources
        commit = VLLM_COMMIT if layout.engine == "vllm" else SGLANG_COMMIT
        assert all(s.commit == commit and s.path and s.symbol for s in layout.sources)


@pytest.mark.parametrize("name", sorted(EXPECTED_VLLM_STRIDES))
def test_vllm_strides_match_the_hand_worked_values(name: str) -> None:
    cache = make_cache(LAYOUTS[name], SPEC)
    strides, layer_step = EXPECTED_VLLM_STRIDES[name]
    first, second = cache[0][0], cache[1][0]
    assert tuple(first.shape) == (3, 2, 4, 16)
    assert first.stride() == strides
    assert second.storage_offset() - first.storage_offset() == layer_step
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def test_sglang_buffer_shapes() -> None:
    nhd = make_cache(LAYOUTS["sglang_nhd"], SPEC)
    hnd = make_cache(LAYOUTS["sglang_hnd"], SPEC)
    assert all(tuple(t.shape) == (12, 2, 8) and t.is_contiguous() for t in nhd[0])
    assert all(tuple(t.shape) == (3, 2, 4, 8) and t.is_contiguous() for t in hnd[0])
    assert len(nhd) == len(hnd) == SPEC.num_layers


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_views_name_the_elements_the_engine_reads(name: str) -> None:
    cache = fill_through_views(name)
    for layer in cache:
        for b in range(SPEC.num_blocks):
            for n in range(SPEC.block_size):
                for h in range(SPEC.num_kv_heads):
                    key, value = engine_read(name, layer, b, n, h)
                    expected_k = torch.tensor([code(b, n, h, d, 0) for d in range(8)])
                    expected_v = torch.tensor([code(b, n, h, d, 1) for d in range(8)])
                    assert torch.equal(key.float(), expected_k)
                    assert torch.equal(value.float(), expected_v)


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_views_do_not_copy(name: str) -> None:
    layout = LAYOUTS[name]
    layer = make_cache(layout, SPEC)[1]
    storage = layer[0].untyped_storage().data_ptr()
    k, _ = kv_views(layout, SPEC, layer)
    assert k.untyped_storage().data_ptr() == storage
    assert all(
        p.untyped_storage().data_ptr() in {t.untyped_storage().data_ptr() for t in layer}
        for p in planes(layout, SPEC, layer)
    )


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_each_plane_block_is_one_run_holding_exactly_that_block(name: str) -> None:
    layout = LAYOUTS[name]
    cache = fill_through_views(name)
    for layer in cache:
        layer_planes = planes(layout, SPEC, layer)
        assert len(layer_planes) == (1 if layout.kv_split == "joint" else 2)
        for plane in layer_planes:
            assert plane.shape[0] == SPEC.num_blocks
            for b in range(SPEC.num_blocks):
                assert plane[b].is_contiguous()
                values = plane[b].float().flatten()
                # Every value in block b's run belongs to block b.
                blocks = (values // 2).long() // (4 * 2 * 8)
                assert torch.equal(blocks, torch.full_like(blocks, b))


def test_planes_of_the_all_layer_orders_step_over_every_layer() -> None:
    layout = LAYOUTS["vllm_BLNHC"]
    plane = planes(layout, SPEC, make_cache(layout, SPEC)[1])[0]
    block_values = 4 * 2 * 16
    assert plane.stride(0) == SPEC.num_layers * block_values


def test_fp8_caches_give_planes_and_kv_views_but_not_other_8_bit_types() -> None:
    spec = CacheSpec(2, 3, 4, 2, 8, torch.float8_e4m3fn)
    layout = LAYOUTS["vllm_LBNHC"]
    layer = make_cache(layout, spec)[0]
    assert planes(layout, spec, layer)[0].dtype == torch.float8_e4m3fn
    k, v = kv_views(layout, spec, layer)
    assert k.dtype == v.dtype == torch.float8_e4m3fn and tuple(k.shape) == (3, 4, 2, 8)
    e5m2 = CacheSpec(2, 3, 4, 2, 8, torch.float8_e5m2)
    with pytest.raises(ValueError, match="views"):
        kv_views(layout, e5m2, make_cache(layout, e5m2)[0])


# --- the index ---------------------------------------------------------------------

BLOCKS = [[7, 2, 9], [7, 2, 5, 11]]
LENGTHS = [40, 60]
INDEX_SPEC = CacheSpec(1, 16, 16, 1, 8, torch.float16)


def test_vllm_block_table() -> None:
    index = make_index(LAYOUTS["vllm_LBNHC"], INDEX_SPEC, BLOCKS, LENGTHS)
    assert index["block_table"].dtype == torch.int32
    assert index["block_table"].tolist() == [[7, 2, 9, 0], [7, 2, 5, 11]]  # 0: the null block
    assert index["seq_lens"].dtype == torch.int32
    assert index["seq_lens"].tolist() == [40, 60]


def test_sglang_slot_list_with_pages_of_16() -> None:
    index = make_index(LAYOUTS["sglang_nhd"], INDEX_SPEC, BLOCKS, LENGTHS)
    assert index["kv_indptr"].dtype == torch.int32
    assert index["kv_indptr"].tolist() == [0, 40, 100]
    slots = index["kv_indices"]
    assert slots.dtype == torch.int64
    expected_r0 = (
        [7 * 16 + i for i in range(16)]
        + [2 * 16 + i for i in range(16)]
        + [9 * 16 + i for i in range(8)]
    )
    assert slots[:40].tolist() == expected_r0
    # r1: 60 tokens fill blocks 7, 2 and 5, and 12 slots of block 11.
    assert slots[40:].tolist()[-12:] == [11 * 16 + i for i in range(12)]


def test_sglang_slot_list_with_pages_of_1() -> None:
    spec = CacheSpec(1, 64, 1, 1, 8, torch.float16)
    index = make_index(LAYOUTS["sglang_nhd"], spec, [[5, 9, 3], [8]], [3, 1])
    assert index["kv_indptr"].tolist() == [0, 3, 4]
    assert index["kv_indices"].tolist() == [5, 9, 3, 8]


def test_index_rejects_bad_requests() -> None:
    layout = LAYOUTS["vllm_LBNHC"]
    with pytest.raises(ValueError, match="need 3 blocks"):
        make_index(layout, INDEX_SPEC, [[7, 2]], [40])
    with pytest.raises(ValueError, match="outside"):
        make_index(layout, INDEX_SPEC, [[0, 2, 3]], [40])  # block 0 is reserved
    with pytest.raises(ValueError, match="outside"):
        make_index(layout, INDEX_SPEC, [[1, 2, 16]], [40])
    with pytest.raises(ValueError, match="lengths"):
        make_index(layout, INDEX_SPEC, [[1, 2, 3]], [40, 1])
