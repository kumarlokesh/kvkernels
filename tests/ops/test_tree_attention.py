"""The cascade over a prefix tree, and several query tokens per request.

Batches here share KV blocks the way a radix-tree prefix cache makes them
share: requests walk down a random tree of KV-block spans, then add KV blocks
of their own. The plan must find the same tree nodes as the Python reference,
and attention through it must match the fp32 reference, with each query token
seeing only the keys up to its own position.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache, make_index
from kvkernels.ops.cascade import CascadePlan, PrefixGroup, cascade_decode_attention
from kvkernels.reference.attention import RequestKV, attention_ref32, attention_ref_low
from kvkernels.reference.cascade_plan import find_prefix_tree_ref
from kvkernels.tolerance import assert_close
from tests.ops.cascade_cases import DEVICE, INTERPRETING
from tests.ops.test_cascade_plan import Batch, Setup, explicit_batch, plan_device

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)

Item = tuple[int, int, int, int]  # first sorted row, rows, first KV block, end KV block


def tree_batch(
    rng: random.Random,
    *,
    requests: int,
    block_size: int,
    depth: int = 4,
    fanout: int = 2,
    min_tail: int = 1,
    padded: Sequence[int] = (),
) -> Batch:
    """Requests that walk 0 … depth levels down a random tree of KV-block spans, then
    add 0-2 full KV blocks and a tail of min_tail … block_size - 1 tokens of their own."""
    next_block = 1
    spans: dict[tuple[int, ...], list[int]] = {}

    def take(n: int) -> list[int]:
        nonlocal next_block
        out = list(range(next_block, next_block + n))
        next_block += n
        return out

    rows: list[list[int]] = []
    lens: list[int] = []
    for r in range(requests):
        if r in padded:
            rows.append([])
            lens.append(0)
            continue
        path: tuple[int, ...] = ()
        blocks: list[int] = []
        for _ in range(rng.randint(0, depth)):
            path = (*path, rng.randrange(fanout))
            if path not in spans:
                spans[path] = take(rng.randint(1, 3))
            blocks += spans[path]
        own_full, tail = rng.randint(0, 2), rng.randint(min_tail, max(min_tail, block_size - 1))
        rows.append(blocks + take(own_full + 1))
        lens.append((len(blocks) + own_full) * block_size + tail)
    return Batch(rows, lens, block_size, next_block + 1)


def build(
    batch: Batch,
    *,
    query_tokens: int = 1,
    layout: str = "vllm_LBNHC",
    q_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 64,
    seed: int = 0,
) -> Setup:
    gen = torch.Generator().manual_seed(seed)
    lay = LAYOUTS[layout]
    spec = CacheSpec(1, batch.num_blocks, batch.block_size, kv_heads, head_dim, torch.float16)
    k, v = kv_views(lay, spec, make_cache(lay, spec, DEVICE)[0])
    k.copy_(torch.randn(tuple(k.shape), generator=gen).half())
    v.copy_(torch.randn(tuple(v.shape), generator=gen).half())
    real = [r for r, n in enumerate(batch.seq_lens) if n > 0]
    index = make_index(
        lay,
        spec,
        [batch.request_blocks[r] for r in real],
        [batch.seq_lens[r] for r in real],
        DEVICE,
    )
    seq_lens = torch.tensor(batch.seq_lens, dtype=torch.int32, device=DEVICE)
    if lay.index_kind == "block_table":
        table = torch.zeros(
            (len(batch.seq_lens), index["block_table"].shape[1]), dtype=torch.int32, device=DEVICE
        )
        table[real] = index["block_table"]
        index = {"block_table": table, "seq_lens": seq_lens}
    else:
        assert len(real) == len(batch.seq_lens), "slot-list batches have no padded rows here"
        index = {**index, "seq_lens": seq_lens}
    rows = len(batch.seq_lens) * query_tokens
    q = torch.randn((rows, q_heads, head_dim), generator=gen).half().to(DEVICE)
    requests = [
        RequestKV([], 0, blocks, n)
        for blocks, n in zip(batch.request_blocks, batch.seq_lens, strict=True)
    ]
    return Setup(q, k, v, index, requests)


def new_plan(
    setup: Setup, batch: Batch, *, query_tokens: int = 1, **kwargs: int | bool
) -> CascadePlan:
    args: dict[str, int | bool] = {
        "max_requests": max(len(batch.seq_lens), 1),
        "max_groups": max(len(batch.seq_lens), 1),
        "max_blocks": sum(len(b) for b in batch.request_blocks) + 1,
        "num_q_heads": setup.q.shape[1],
        "num_kv_heads": setup.k.shape[2],
        "head_dim": setup.q.shape[2],
        "block_size": batch.block_size,
        "num_blocks": batch.num_blocks,
        "min_shared_tokens": batch.block_size,
        "query_tokens": query_tokens,
    }
    args.update(kwargs)
    return CascadePlan(device=torch.device(DEVICE), **args)  # type: ignore[arg-type]


def attend(setup: Setup, plan: CascadePlan) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(setup.q)
    lse = torch.empty(setup.q.shape[:2], dtype=torch.float32, device=DEVICE)
    cascade_decode_attention(setup.q, setup.k, setup.v, plan, out, lse)
    return out, lse


def check(setup: Setup, out: torch.Tensor, lse: torch.Tensor, query_tokens: int = 1) -> None:
    out_low, lse_low = attention_ref_low(
        setup.q, setup.k, setup.v, setup.requests, query_tokens=query_tokens
    )
    out32, lse32 = attention_ref32(
        setup.q, setup.k, setup.v, setup.requests, query_tokens=query_tokens
    )
    assert not torch.isnan(out).any(), "NaN in out"
    assert_close(out, out_low, out32)
    assert_close(lse, lse_low, lse32)


def device_items(plan: CascadePlan, setup: Setup) -> tuple[list[int], list[Item]]:
    perm = plan.perm[: plan.num_requests].tolist()
    items = []
    for g in range(int(plan.counts[3])):
        first, rows = int(plan.item_req_start[g]), int(plan.item_num_reqs[g])
        if "block_table" in setup.index:
            base, unit = perm[first] * setup.index["block_table"].stride(0), 1
        else:
            base, unit = int(setup.index["kv_indptr"][perm[first]]), plan.block_size
        start = (int(plan.item_block_start[g]) - base) // unit
        items.append((first, rows, start, start + int(plan.item_info[0, g]) // plan.block_size))
    return perm, items


def expected_items(batch: Batch, plan: CascadePlan) -> tuple[list[int], list[Item]]:
    perm, items = find_prefix_tree_ref(
        batch.request_blocks,
        batch.seq_lens,
        block_size=batch.block_size,
        min_shared_tokens=plan.min_shared_tokens,
        max_groups=plan.max_groups,
        max_levels=plan.max_levels,
        query_tokens=plan.query_tokens,
    )
    return perm, [(i.first, i.rows, i.start, i.end) for i in items]


def host_update(plan: CascadePlan, batch: Batch, items: Sequence[Item]) -> None:
    """The host plan for a batch already in tree order, with the given node items."""
    bs = batch.block_size
    covered = [0] * len(batch.seq_lens)
    groups = []
    for first, rows, start, end in items:
        groups.append(
            PrefixGroup(
                range(first, first + rows),
                batch.request_blocks[first][start:end],
                (end - start) * bs,
                start=start * bs,
            )
        )
        for r in range(first, first + rows):
            covered[r] = end
    plan.update(
        groups=groups,
        suffix_blocks=[b[c:] for b, c in zip(batch.request_blocks, covered, strict=True)],
        suffix_lens=[n - c * bs for n, c in zip(batch.seq_lens, covered, strict=True)],
    )


def in_tree_order(batch: Batch, plan: CascadePlan) -> Batch:
    perm, _ = expected_items(batch, plan)
    return Batch(
        [batch.request_blocks[r] for r in perm],
        [batch.seq_lens[r] for r in perm],
        batch.block_size,
        batch.num_blocks,
    )


# The batch drawn in the RFC: block size 16.
EXAMPLE_ROWS = [[7, 2, 9, 4, 1], [7, 2, 9, 4, 8], [7, 2, 9, 6, 3], [7, 2, 5, 10], [3, 8, 1]]
EXAMPLE_LENS = [75, 70, 72, 60, 40]


# --- the tree ---------------------------------------------------------------------


def test_the_worked_example() -> None:
    batch = explicit_batch(EXAMPLE_ROWS, EXAMPLE_LENS, 16)
    setup = build(batch)
    plan = new_plan(setup, batch)
    plan_device(plan, setup)
    # Sorted: r4 | r3 | r0 r1 | r2. A: rows 1-4, blocks 0-1; B: rows 2-4, block 2;
    # C: rows 2-3, block 3.
    assert device_items(plan, setup) == (
        [4, 3, 0, 1, 2],
        [(1, 4, 0, 2), (2, 3, 2, 3), (2, 2, 3, 4)],
    )
    assert device_items(plan, setup) == expected_items(batch, plan)
    check(setup, *attend(setup, plan))


@pytest.mark.parametrize(("max_levels", "items"), [(1, 1), (2, 2), (3, 3), (8, 3)])
def test_rounds_stop_at_max_levels(max_levels: int, items: int) -> None:
    batch = explicit_batch(EXAMPLE_ROWS, EXAMPLE_LENS, 16)
    setup = build(batch)
    plan = new_plan(setup, batch, max_levels=max_levels)
    plan_device(plan, setup)
    assert len(device_items(plan, setup)[1]) == items
    assert device_items(plan, setup) == expected_items(batch, plan)
    check(setup, *attend(setup, plan))


def test_a_short_node_is_folded_into_its_children() -> None:
    # The root shares one KV block, each child two more: with a threshold of two
    # KV blocks the root is no item, and each child's item starts at block 0.
    rows = [[1, 2, 3, 10], [1, 2, 3, 11], [1, 4, 5, 12], [1, 4, 5, 13], [1, 6]]
    batch = explicit_batch(rows, [14, 15, 13, 14, 6], 4)
    setup = build(batch)
    plan = new_plan(setup, batch, min_shared_tokens=8)
    plan_device(plan, setup)
    assert device_items(plan, setup)[1] == [(0, 2, 0, 3), (2, 2, 0, 3)]
    assert device_items(plan, setup) == expected_items(batch, plan)
    check(setup, *attend(setup, plan))


def test_nodes_past_max_groups_are_folded() -> None:
    batch = explicit_batch(EXAMPLE_ROWS, EXAMPLE_LENS, 16)
    setup = build(batch)
    plan = new_plan(setup, batch, max_groups=2)
    plan_device(plan, setup)
    assert device_items(plan, setup)[1] == [(1, 4, 0, 2), (2, 3, 2, 3)]
    check(setup, *attend(setup, plan))


@pytest.mark.parametrize("seed", range(8))
def test_trees_match_the_reference_on_random_batches(seed: int) -> None:
    rng = random.Random(seed)
    batch = tree_batch(rng, requests=rng.randint(2, 24), block_size=4, padded=(1,))
    setup = build(batch, seed=seed)
    plan = new_plan(setup, batch, max_levels=rng.choice([2, 4, 8]))
    plan_device(plan, setup)
    assert device_items(plan, setup) == expected_items(batch, plan)
    check(setup, *attend(setup, plan))


@pytest.mark.parametrize(("layout", "block_size"), [("sglang_nhd", 1), ("sglang_hnd", 4)])
def test_trees_from_slot_lists(layout: str, block_size: int) -> None:
    batch = tree_batch(random.Random(3), requests=12, block_size=block_size)
    setup = build(batch, layout=layout)
    plan = new_plan(setup, batch, min_shared_tokens=max(block_size, 2))
    plan_device(plan, setup)
    assert device_items(plan, setup) == expected_items(batch, plan)
    check(setup, *attend(setup, plan))


@pytest.mark.parametrize("num_sms", [1, 64])  # 64: small batches split their tokens
def test_bit_identical_to_the_host_plan_with_nested_groups(num_sms: int) -> None:
    probe = tree_batch(random.Random(12), requests=14, block_size=4)
    setup = build(probe)
    batch = in_tree_order(probe, new_plan(setup, probe))
    setup = build(batch)
    plan = new_plan(setup, batch, num_sms=num_sms, min_split_tokens=8)
    plan_device(plan, setup)
    perm, items = device_items(plan, setup)
    assert perm == list(range(14))
    assert any(start > 0 for _, _, start, _ in items), "the batch should have nested nodes"
    device_out, device_lse = attend(setup, plan)
    host_update(plan, batch, items)
    host_out, host_lse = attend(setup, plan)
    assert torch.equal(device_out, host_out)
    assert torch.equal(device_lse, host_lse)
    check(setup, device_out, device_lse)


# --- several query tokens per request ---------------------------------------------


@pytest.mark.parametrize("query_tokens", [1, 2, 4, 8])
def test_query_tokens_see_keys_up_to_their_own_position(query_tokens: int) -> None:
    rng = random.Random(20 + query_tokens)
    batch = tree_batch(rng, requests=10, block_size=4, min_tail=1)
    setup = build(batch, query_tokens=query_tokens)
    # A threshold of one token: every node of a KV block or more is an item.
    plan = new_plan(setup, batch, query_tokens=query_tokens, min_shared_tokens=1)
    plan_device(plan, setup)
    perm, items = device_items(plan, setup)
    assert items, "the batch should have node items"
    assert (perm, items) == expected_items(batch, plan)
    check(setup, *attend(setup, plan), query_tokens=query_tokens)


def test_a_range_with_no_visible_keys_gives_an_empty_partial() -> None:
    # 64 SMs and 4-token ranges split each suffix finely: the last range holds
    # only the newest keys, which the request's first query tokens do not see.
    rows = [[1, 2, 3, 4, 5, 6, 7, 8]] * 1 + [[10, 11, 12, 13, 14, 15, 16, 17]]
    batch = explicit_batch(rows, [32, 30], 4)
    setup = build(batch, query_tokens=8)
    plan = new_plan(setup, batch, query_tokens=8, num_sms=64, min_split_tokens=4, block_n=16)
    plan.min_split_floor, plan.split_tokens_per_row = 4, 0  # ranges as short as 4 tokens
    plan_device(plan, setup)
    assert int(plan.item_info[1, : int(plan.counts[4])].max()) > 1, "suffixes should split"
    check(setup, *attend(setup, plan), query_tokens=8)


def test_query_tokens_in_the_worked_example() -> None:
    batch = explicit_batch(EXAMPLE_ROWS, EXAMPLE_LENS, 16)
    setup = build(batch, query_tokens=4)
    plan = new_plan(setup, batch, query_tokens=4, min_shared_tokens=4)
    plan_device(plan, setup)
    device_out, device_lse = attend(setup, plan)
    check(setup, device_out, device_lse, query_tokens=4)


def test_batch_invariance_with_query_tokens() -> None:
    batch = tree_batch(random.Random(30), requests=6, block_size=4, min_tail=3)
    setup = build(batch, query_tokens=3)
    plan = new_plan(
        setup, batch, query_tokens=3, batch_invariant=True, num_sms=64, min_split_tokens=4
    )
    plan_device(plan, setup)
    out, lse = attend(setup, plan)
    check(setup, out, lse, query_tokens=3)
    for r in range(6):
        rows = slice(3 * r, 3 * r + 3)
        alone = Setup(
            setup.q[rows].clone(),
            setup.k,
            setup.v,
            {
                "block_table": setup.index["block_table"][r : r + 1].clone(),
                "seq_lens": setup.index["seq_lens"][r : r + 1].clone(),
            },
            [setup.requests[r]],
        )
        plan_device(plan, alone)
        out_r, lse_r = attend(alone, plan)
        assert torch.equal(out_r, out[rows]), f"request {r}"
        assert torch.equal(lse_r, lse[rows]), f"request {r}"


# --- input checks -----------------------------------------------------------------


def test_update_rejects_groups_that_are_not_a_tree() -> None:
    batch = explicit_batch([[1, 2, 3], [1, 2, 4], [1, 5]], [12, 12, 8], 4)
    setup = build(batch, query_tokens=2)
    plan = new_plan(setup, batch, query_tokens=2)
    outer = PrefixGroup(range(0, 3), [1], 4)
    with pytest.raises(ValueError, match="overlap"):
        plan.update(
            groups=[outer, PrefixGroup(range(0, 2), [1, 2], 8)],
            suffix_blocks=[[3], [4], [5]],
            suffix_lens=[4, 4, 4],
        )
    with pytest.raises(ValueError, match="gap"):
        plan.update(
            groups=[outer, PrefixGroup(range(0, 2), [3], 4, start=8)],
            suffix_blocks=[[3], [4], [5]],
            suffix_lens=[4, 4, 4],
        )
    with pytest.raises(ValueError, match="not inside"):
        plan.update(
            groups=[PrefixGroup(range(0, 2), [1], 4), PrefixGroup(range(1, 3), [2], 4, start=4)],
            suffix_blocks=[[2, 3], [3], [5]],
            suffix_lens=[8, 4, 4],
        )
    with pytest.raises(ValueError, match="first new query token"):
        plan.update(
            groups=[outer, PrefixGroup(range(0, 2), [2, 3], 8, start=4)],
            suffix_blocks=[[], [4], [5]],
            suffix_lens=[0, 4, 4],
        )
    with pytest.raises(ValueError, match="multiple of the block size"):
        plan.update(
            groups=[PrefixGroup(range(0, 2), [2], 4, start=2)],
            suffix_blocks=[[3], [4], [5]],
            suffix_lens=[4, 4, 4],
        )


def test_q_must_hold_every_query_token() -> None:
    batch = explicit_batch([[1, 2], [3, 4]], [8, 8], 4)
    setup = build(batch, query_tokens=2)
    plan = new_plan(setup, batch, query_tokens=2)
    plan_device(plan, setup)
    with pytest.raises(ValueError, match="query_tokens"):
        cascade_decode_attention(setup.q[:2], setup.k, setup.v, plan, setup.q[:2].clone())
    with pytest.raises(ValueError, match="query_tokens"):
        new_plan(setup, batch, query_tokens=17)


# --- GPU tier ---------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("layout", "query_tokens"), [("vllm_LBNHC", 1), ("vllm_LBNHC", 4), ("sglang_nhd", 1)]
)
def test_full_size(layout: str, query_tokens: int) -> None:
    block_size = 16 if layout.startswith("vllm") else 1
    batch = tree_batch(random.Random(1), requests=96, block_size=block_size, depth=5, fanout=3)
    setup = build(
        batch, layout=layout, q_heads=32, kv_heads=8, head_dim=128, query_tokens=query_tokens
    )
    plan = new_plan(setup, batch, query_tokens=query_tokens, min_shared_tokens=16)
    plan_device(plan, setup)
    assert device_items(plan, setup) == expected_items(batch, plan)
    check(setup, *attend(setup, plan), query_tokens=query_tokens)


@pytest.mark.gpu
def test_planning_and_attention_capture_in_one_cuda_graph() -> None:
    first = tree_batch(random.Random(4), requests=24, block_size=16)
    second = tree_batch(random.Random(5), requests=24, block_size=16)
    num_blocks = max(first.num_blocks, second.num_blocks)
    first.num_blocks = second.num_blocks = num_blocks
    a = build(first, q_heads=32, kv_heads=8, head_dim=128, query_tokens=2)
    b = build(second, q_heads=32, kv_heads=8, head_dim=128, query_tokens=2, seed=1)
    width = max(a.index["block_table"].shape[1], b.index["block_table"].shape[1])
    table = torch.zeros((24, width), dtype=torch.int32, device=DEVICE)
    lens = torch.zeros(24, dtype=torch.int32, device=DEVICE)
    q = torch.empty_like(a.q)

    def load(s: Setup) -> None:
        table.zero_()
        table[:, : s.index["block_table"].shape[1]] = s.index["block_table"]
        lens.copy_(s.index["seq_lens"])
        q.copy_(s.q)

    plan = new_plan(a, first, query_tokens=2, min_shared_tokens=16)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=DEVICE)

    def step() -> None:
        plan.plan_from_block_table(table, lens)
        cascade_decode_attention(q, a.k, a.v, plan, out, lse)

    load(a)
    step()  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for s in (a, b):
        load(s)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check(Setup(q, a.k, a.v, s.index, s.requests), out, lse, query_tokens=2)
