"""The kv_offload suite: its grid and byte count. The implementations need a GPU (pinned memory)."""

from __future__ import annotations

from bench.suites import kv_offload as suite


def test_grid_covers_both_directions_both_layouts_and_three_request_lengths() -> None:
    params = list(suite.grid())
    assert len(params) == 2 * len(suite.LAYOUTS) * len(suite.TOKENS)
    assert {(p["op"], p["layout"], p["tokens"]) for p in params} == {
        (op, layout, tokens)
        for op in ("offload", "reload")
        for layout in suite.LAYOUTS
        for tokens in suite.TOKENS
    }


def test_reload_cases_check_the_loaded_blocks_and_offload_cases_the_host_buffer() -> None:
    for p in suite.grid():
        case = suite.make_case(p, with_dma=True)
        assert case.outputs == (("loaded",) if p["op"] == "reload" else ("host",))
        assert ("vllm_batched_dma" in case.impls) == str(p["layout"]).startswith("vllm")
        first = next(iter(case.impls))
        assert first == (
            "torch_copy_then_index_copy" if p["op"] == "reload" else "torch_index_select_then_copy"
        )


def test_bytes_moved_is_the_request_kv_of_every_layer() -> None:
    # 1,024 tokens of Llama 3.1 8B: 32 layers · 8 KV heads · K and V · 128 · 2 bytes per token.
    for p in suite.grid():
        if p["tokens"] == 1024:
            assert suite.bytes_moved(p) == 1024 * 32 * 8 * 2 * 128 * 2 == 128 << 20
