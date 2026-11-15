"""The tree-shaped cascade suite: its grid, its trees, and the cascade through it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from bench.suites import cascade_tree as suite  # noqa: E402


def test_grid_covers_the_planned_cases() -> None:
    params = list(suite.grid())
    length_sets = sum(len(sets) for sets in suite.SHAPES.values())
    assert len(params) == length_sets * 2 * len(suite.BATCHES) * 2 * 2  # models, suffixes, T
    assert len({tuple(sorted(p.items())) for p in params}) == len(params)
    for p in params:
        assert int(p["batch"]) % max(suite.fanouts(str(p["shape"]), int(p["batch"]))) == 0


def test_a_three_level_tree_shares_each_node_with_the_requests_below_it() -> None:
    p = suite._params(
        shape="three_levels", lens=(1024, 2048, 4096), batch=64, suffix=64, query_tokens=1
    )
    tree = suite.tree_of(p)
    assert [len(nodes) for nodes in tree.levels] == [1, 4, 16]
    assert [n for _, n, _ in tree.levels[2]] == [4] * 16
    rows = tree.request_blocks()
    assert rows[0][:64] == rows[63][:64]  # the system prompt: 1,024 tokens, 64 KV blocks
    assert rows[0][:192] == rows[15][:192] and rows[0][64:192] != rows[16][64:192]
    every = [b for nodes in tree.levels for _, _, blocks in nodes for b in blocks]
    every += [b for blocks in tree.suffix_blocks for b in blocks]
    assert len(every) == len(set(every)) and 0 not in every
    # The RFC's example: 78,848 tokens, 4 KiB of K and V each per layer.
    assert suite.bytes_moved(p) == 78_848 * 4096


@pytest.mark.parametrize("query_tokens", [1, 2])
def test_the_cascade_matches_per_request_attention(
    query_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kvkernels.ops.cascade
    from tests.ops.cascade_cases import DEVICE, INTERPRETING

    if not INTERPRETING and not torch.cuda.is_available():
        pytest.skip("needs the Triton interpreter or a CUDA GPU")
    monkeypatch.setattr(kvkernels.ops.cascade, "MIN_SHARED_TOKENS", 16)
    p = suite._params(
        shape="three_levels", lens=(16, 16, 32), batch=16, suffix=16, query_tokens=query_tokens
    )
    p.update(q_heads=8, kv_heads=2, head_dim=64)
    case = suite.make_case(p, with_flashinfer=False, device=DEVICE)
    expected = case.setup()
    case.impls["torch_per_request"](expected)
    # Batch 16: one request per history, so the nodes are the system prompt and
    # the 4 tool lists; two levels find the system prompt only. With 2 query
    # tokens the threshold is 32 tokens: the 16-token system prompt is folded
    # into the tool lists' items.
    tree_items, two_level_items = (5, 1) if query_tokens == 1 else (4, 0)
    for name, node_items in (
        ("kvkernels_tree", tree_items),
        ("kvkernels_two_level", two_level_items),
    ):
        t = case.prepared(name, case.setup())
        case.impls[name](t)
        assert torch.allclose(t["out"].float(), expected["out"].float(), atol=2e-3, rtol=0), name
        cascade = case.impls[name].__self__.cascade  # type: ignore[attr-defined]
        assert int(cascade.counts[3]) == node_items, name
