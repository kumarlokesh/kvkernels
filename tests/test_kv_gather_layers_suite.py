"""The kv_gather_layers suite: its grid, byte model and, on small inputs, its implementations."""

from __future__ import annotations

import dataclasses
import os

import pytest

from bench.schema import Params
from bench.suites import kv_gather_layers as suite

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


def test_every_layout_and_op_appears_and_caches_fit_in_8_gib() -> None:
    params = list(suite.grid())
    assert {(p["op"], p["layout"]) for p in params} == {
        (op, layout) for op in ("gather", "scatter") for layout, *_ in suite.SHAPES
    }
    for p in params:
        planes = suite.LAYERS * suite.planes_per_layer(str(p["layout"]))
        cache = planes * int(p["blocks"]) * suite.block_bytes(p)
        rows = planes * int(p["num_ids"]) * suite.block_bytes(p)
        assert cache <= 8 << 30 and rows <= 4 << 30, p


def test_bytes_moved_counts_every_plane() -> None:
    p: Params = {"op": "gather", "layout": "sglang_nhd", "block": 1, "num_ids": 1024}
    block = 1 * 8 * 128 * 2
    assert suite.bytes_moved(p) == 2 * 1024 * block * 64 + 4 * 1024


@pytest.mark.skipif(not INTERPRETING, reason="runs the kernels under the Triton interpreter")
@pytest.mark.parametrize("op", ["gather", "scatter"])
@pytest.mark.parametrize("layout", ["vllm_LBNHC", "vllm_BLNHC", "sglang_nhd"])
def test_every_implementation_agrees_on_tiny_blocks(layout: str, op: str) -> None:
    pytest.importorskip("torch")
    from bench import harness

    p = next(q for q in suite.grid() if q["layout"] == layout and q["op"] == op)
    p = {**p, "num_ids": 6}
    case = dataclasses.replace(
        suite.make_case(p),
        setup=lambda: suite.make_inputs(p, tiny_blocks=True, device="cpu"),
        host_setup=None,
    )
    harness.check_case(case)
