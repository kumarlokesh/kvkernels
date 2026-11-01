"""The kv_gather benchmark suite: its parameter grid, byte model and implementations."""

from __future__ import annotations

import pytest

from bench.schema import Params
from bench.suites import kv_gather as suite


def test_every_case_fits_the_cache_budget() -> None:
    params = list(suite.grid())
    assert params
    for p in params:
        shape = suite.block_shape(str(p["layout"]), int(p["block"]), int(p["kv_heads"]), 128)
        assert shape[0] == p["block"]
        n = int(p["num_ids"])
        block_bytes = (suite.bytes_moved(p) - 4 * n) // (2 * n)
        assert suite.cache_bytes(str(p["layout"]), block_bytes, int(p["num_ids"])) <= (
            suite.MAX_CACHE_BYTES
        )


def test_large_all_layer_caches_are_left_out() -> None:
    # 32 layers · 16 slots · 8 heads · 2·128 values · 2 bytes = 2 MiB per block;
    # 32,768 ids would need 64 GiB.
    big = [
        p
        for p in suite.grid()
        if p["layout"] == "vllm_BLNHC"
        and p["block"] == 16
        and p["kv_heads"] == 8
        and p["head_dim"] == 128
        and p["num_ids"] == 32768
    ]
    assert big == []


def test_bytes_moved_counts_one_read_one_write_and_the_ids() -> None:
    p: Params = {
        "op": "gather",
        "layout": "vllm_LBNHC",
        "block": 16,
        "kv_heads": 8,
        "head_dim": 128,
        "num_ids": 512,
        "order": "random",
        "dtype": "float16",
    }
    block = 16 * 8 * 256 * 2
    assert block == 65536
    assert suite.bytes_moved(p) == 2 * 512 * block + 4 * 512


def test_every_op_and_layout_appears() -> None:
    seen = {(p["op"], p["layout"]) for p in suite.grid()}
    assert seen == {(op, layout) for op in ("gather", "scatter") for layout in suite.BLOCK_SIZES}


def test_only_the_plain_copy_gets_the_contiguous_source() -> None:
    params: Params = {
        "op": "gather",
        "layout": "vllm_LBNHC",
        "block": 16,
        "kv_heads": 8,
        "head_dim": 128,
        "num_ids": 64,
        "order": "random",
        "dtype": "float16",
    }
    gather = suite.make_case(params, with_dma=False)
    assert gather.setups is not None and set(gather.setups) == {"copy_ceiling"}
    assert suite.make_case({**params, "op": "scatter"}, with_dma=False).setups is None


@pytest.mark.gpu
@pytest.mark.parametrize("op", ["gather", "scatter"])
@pytest.mark.parametrize(
    ("layout", "block", "heads"),
    [("vllm_LBNHC", 16, 8), ("vllm_BLNHC", 16, 8), ("sglang_nhd", 1, 8)],
)
def test_every_implementation_matches_the_reference(
    op: str, layout: str, block: int, heads: int
) -> None:
    from bench.harness import check_case

    params: Params = {
        "op": op,
        "layout": layout,
        "block": block,
        "kv_heads": heads,
        "head_dim": 128,
        "num_ids": 512,
        "order": "random",
        "dtype": "float16",
    }
    case = suite.make_case(params, with_dma=suite.batched_dma_available())
    check_case(case)
