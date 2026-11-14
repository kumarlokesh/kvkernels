"""Planning the cascade on the GPU from block tables and slot lists.

Each batch is a set of requests that share leading KV blocks the way prefix
caching makes them share: some requests take the first few KV blocks of a
common chain, then KV blocks of their own. The plan must find the same groups
as the Python reference, and attention through it must match the fp32
reference over each request's whole row.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache, make_index
from kvkernels.ops.cascade import CascadePlan, PrefixGroup, cascade_decode_attention
from kvkernels.reference.attention import RequestKV, attention_ref32, attention_ref_low
from kvkernels.reference.cascade_plan import FoundGroup, find_prefix_groups_ref
from kvkernels.tolerance import assert_close
from tests.ops.cascade_cases import DEVICE, INTERPRETING

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)


@dataclass
class Batch:
    request_blocks: list[list[int]]  # empty for a padded row
    seq_lens: list[int]  # 0 for a padded row
    block_size: int
    num_blocks: int


def chains_batch(
    rng: random.Random,
    *,
    requests: int,
    block_size: int,
    chains: int = 3,
    share: float = 0.7,
    padded: Sequence[int] = (),
) -> Batch:
    """Requests that take 1 … all KV blocks of a random chain, then KV blocks of their own."""
    next_block = 1

    def take(n: int) -> list[int]:
        nonlocal next_block
        out = list(range(next_block, next_block + n))
        next_block += n
        return out

    roots = [take(rng.randint(1, 6)) for _ in range(chains)]
    rows: list[list[int]] = []
    lens: list[int] = []
    for r in range(requests):
        if r in padded:
            rows.append([])
            lens.append(0)
            continue
        prefix = rng.choice(roots)[: rng.randint(1, 6)] if rng.random() < share else []
        own_full, tail = rng.randint(0, 3), rng.randint(0, block_size - 1)
        if not prefix and own_full == 0 and tail == 0:
            tail = 1
        rows.append(prefix + take(own_full + (tail > 0)))
        lens.append((len(prefix) + own_full) * block_size + tail)
    return Batch(rows, lens, block_size, next_block + 1)


def explicit_batch(rows: list[list[int]], lens: list[int], block_size: int) -> Batch:
    return Batch(rows, lens, block_size, max((max(r) for r in rows if r), default=1) + 2)


@dataclass
class Setup:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    index: dict[str, torch.Tensor]
    requests: list[RequestKV]


def build(
    batch: Batch,
    *,
    layout: str = "vllm_LBNHC",
    q_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
) -> Setup:
    gen = torch.Generator().manual_seed(seed)
    lay = LAYOUTS[layout]
    spec = CacheSpec(1, batch.num_blocks, batch.block_size, kv_heads, head_dim, dtype)
    k, v = kv_views(lay, spec, make_cache(lay, spec, DEVICE)[0])
    k.copy_(torch.randn(tuple(k.shape), generator=gen).to(dtype))
    v.copy_(torch.randn(tuple(v.shape), generator=gen).to(dtype))
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
        if len(real) != len(batch.seq_lens):
            raise ValueError("slot-list batches have no padded rows here")
        index = {**index, "seq_lens": seq_lens}
    q = torch.randn((len(batch.seq_lens), q_heads, head_dim), generator=gen).to(
        dtype=dtype, device=DEVICE
    )
    requests = [
        RequestKV([], 0, blocks, n)
        for blocks, n in zip(batch.request_blocks, batch.seq_lens, strict=True)
    ]
    return Setup(q, k, v, index, requests)


def new_plan(setup: Setup, batch: Batch, **kwargs: int | bool) -> CascadePlan:
    args: dict[str, int | bool] = {
        "max_requests": max(len(batch.seq_lens), 1),
        "max_groups": max(len(batch.seq_lens) // 2, 1),
        "max_blocks": sum(len(b) for b in batch.request_blocks) + 1,
        "num_q_heads": setup.q.shape[1],
        "num_kv_heads": setup.k.shape[2],
        "head_dim": setup.q.shape[2],
        "block_size": batch.block_size,
        "num_blocks": batch.num_blocks,
        "min_shared_tokens": batch.block_size,
    }
    args.update(kwargs)
    return CascadePlan(device=torch.device(DEVICE), **args)  # type: ignore[arg-type]


def plan_device(plan: CascadePlan, setup: Setup) -> None:
    if "block_table" in setup.index:
        plan.plan_from_block_table(setup.index["block_table"], setup.index["seq_lens"])
    else:
        plan.plan_from_slots(
            setup.index["kv_indptr"], setup.index["kv_indices"], setup.index["seq_lens"]
        )


def attend(setup: Setup, plan: CascadePlan) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(setup.q)
    lse = torch.empty(setup.q.shape[:2], dtype=torch.float32, device=DEVICE)
    cascade_decode_attention(setup.q, setup.k, setup.v, plan, out, lse)
    return out, lse


def check(setup: Setup, out: torch.Tensor, lse: torch.Tensor) -> None:
    out_low, lse_low = attention_ref_low(setup.q, setup.k, setup.v, setup.requests)
    out32, lse32 = attention_ref32(setup.q, setup.k, setup.v, setup.requests)
    assert not torch.isnan(out).any(), "NaN in out"
    assert_close(out, out_low, out32)
    assert_close(lse, lse_low, lse32)


def found_groups(plan: CascadePlan) -> tuple[list[int], list[FoundGroup]]:
    perm = plan.perm[: plan.num_requests].tolist()
    groups = [
        FoundGroup(
            int(plan.item_req_start[g]),
            int(plan.item_num_reqs[g]),
            int(plan.item_info[0, g]) // plan.block_size,
        )
        for g in range(int(plan.counts[3]))
    ]
    return perm, groups


def expected_groups(batch: Batch, plan: CascadePlan) -> tuple[list[int], list[FoundGroup]]:
    return find_prefix_groups_ref(
        batch.request_blocks,
        batch.seq_lens,
        block_size=batch.block_size,
        min_shared_tokens=plan.min_shared_tokens,
        max_groups=plan.max_groups,
    )


# --- groups -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_groups_match_the_reference_on_random_batches(seed: int) -> None:
    rng = random.Random(seed)
    batch = chains_batch(rng, requests=rng.randint(1, 24), block_size=4, padded=(2,))
    setup = build(batch, seed=seed)
    plan = new_plan(setup, batch)
    plan_device(plan, setup)
    assert found_groups(plan) == expected_groups(batch, plan)
    check(setup, *attend(setup, plan))


def test_the_worked_example() -> None:
    batch = explicit_batch(
        [[7, 2, 9, 4], [3, 8, 1], [7, 2, 9, 6], [7, 2, 5], [3, 8, 5]], [60, 40, 62, 40, 35], 16
    )
    setup = build(batch)
    plan = new_plan(setup, batch, min_shared_tokens=32)
    plan_device(plan, setup)
    assert found_groups(plan) == ([1, 4, 0, 2, 3], [FoundGroup(0, 2, 2), FoundGroup(2, 3, 2)])
    check(setup, *attend(setup, plan))


@pytest.mark.parametrize(
    ("name", "rows", "lens", "min_tokens", "groups"),
    [
        ("no sharing", [[1, 2], [3, 4], [5]], [8, 7, 3], 4, []),
        ("every request shares", [[1, 2, 3], [1, 2, 4], [1, 2, 5]], [12, 11, 9], 4, [(0, 3, 2)]),
        ("below the threshold", [[1, 2], [1, 3], [5, 6], [5, 6]], [8, 8, 8, 8], 8, [(2, 2, 2)]),
        ("a request inside its first block", [[1], [1, 2]], [3, 6], 4, []),
        (
            "a shared block that one request has not filled",
            [[1, 2], [1, 2]],
            [8, 6],
            4,
            [(0, 2, 1)],
        ),
        ("a whole request is the prefix", [[1, 2], [1, 2, 3]], [8, 11], 4, [(0, 2, 2)]),
        ("padded rows", [[], [1, 2], [], [1, 3]], [0, 8, 0, 7], 4, [(0, 2, 1)]),
    ],
)
def test_named_batches(
    name: str,
    rows: list[list[int]],
    lens: list[int],
    min_tokens: int,
    groups: list[tuple[int, ...]],
) -> None:
    batch = explicit_batch(rows, lens, 4)
    setup = build(batch)
    plan = new_plan(setup, batch, min_shared_tokens=min_tokens)
    plan_device(plan, setup)
    perm, found = found_groups(plan)
    assert found == [FoundGroup(*g) for g in groups], name
    assert (perm, found) == expected_groups(batch, plan), name
    check(setup, *attend(setup, plan))


def test_groups_past_max_groups_stay_ungrouped() -> None:
    rows = [[1, 2], [1, 3], [4, 5], [4, 6], [7, 8], [7, 9]]
    batch = explicit_batch(rows, [8] * 6, 4)
    setup = build(batch)
    plan = new_plan(setup, batch, max_groups=2, min_shared_tokens=4)
    plan_device(plan, setup)
    assert found_groups(plan)[1] == [FoundGroup(0, 2, 1), FoundGroup(2, 2, 1)]
    check(setup, *attend(setup, plan))


# --- index forms and layouts ------------------------------------------------------


@pytest.mark.parametrize(
    ("layout", "block_size"),
    [("vllm_LBHNC", 4), ("vllm_BLNHC", 4), ("sglang_nhd", 1), ("sglang_hnd", 4)],
)
def test_index_forms_and_layouts(layout: str, block_size: int) -> None:
    rng = random.Random(7)
    batch = chains_batch(rng, requests=10, block_size=block_size)
    setup = build(batch, layout=layout)
    plan = new_plan(setup, batch)
    plan_device(plan, setup)
    assert found_groups(plan) == expected_groups(batch, plan)
    check(setup, *attend(setup, plan))


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(4, 4), (16, 1)])
def test_gqa_group_sizes(q_heads: int, kv_heads: int) -> None:
    batch = chains_batch(random.Random(3), requests=9, block_size=4)
    setup = build(batch, q_heads=q_heads, kv_heads=kv_heads)
    plan = new_plan(setup, batch)
    plan_device(plan, setup)
    check(setup, *attend(setup, plan))


# --- against the host plan --------------------------------------------------------


def host_update(plan: CascadePlan, batch: Batch, groups: Sequence[FoundGroup]) -> None:
    """The host plan for an already sorted batch, with the given groups."""
    shared = [0] * len(batch.seq_lens)
    host_groups = []
    for g in groups:
        host_groups.append(
            PrefixGroup(
                range(g.first, g.first + g.rows),
                batch.request_blocks[g.first][: g.blocks],
                g.blocks * batch.block_size,
            )
        )
        for r in range(g.first, g.first + g.rows):
            shared[r] = g.blocks
    plan.update(
        groups=host_groups,
        suffix_blocks=[b[s:] for b, s in zip(batch.request_blocks, shared, strict=True)],
        suffix_lens=[n - s * batch.block_size for n, s in zip(batch.seq_lens, shared, strict=True)],
    )


@pytest.mark.parametrize("num_sms", [1, 64])  # 64: small batches split their tokens
def test_bit_identical_to_the_host_plan_on_a_sorted_batch(num_sms: int) -> None:
    rng = random.Random(11)
    batch = chains_batch(rng, requests=12, block_size=4)
    order = sorted(
        range(12),
        key=lambda r: (batch.request_blocks[r][0] if batch.seq_lens[r] >= 4 else 2**31, r),
    )
    batch = Batch(
        [batch.request_blocks[r] for r in order],
        [batch.seq_lens[r] for r in order],
        batch.block_size,
        batch.num_blocks,
    )
    setup = build(batch)
    plan = new_plan(setup, batch, num_sms=num_sms, min_split_tokens=8)
    plan_device(plan, setup)
    perm, groups = found_groups(plan)
    assert perm == list(range(12))
    assert groups, "the batch should have groups"
    device_out, device_lse = attend(setup, plan)
    host_update(plan, batch, groups)
    host_out, host_lse = attend(setup, plan)
    assert torch.equal(device_out, host_out)
    assert torch.equal(device_lse, host_lse)
    check(setup, device_out, device_lse)


def test_one_plan_alternates_between_host_and_device_planning() -> None:
    batch = chains_batch(random.Random(5), requests=8, block_size=4)
    setup = build(batch)
    plan = new_plan(setup, batch)
    plan.update(groups=[], suffix_blocks=batch.request_blocks, suffix_lens=batch.seq_lens)
    check(setup, *attend(setup, plan))
    plan_device(plan, setup)
    check(setup, *attend(setup, plan))
    plan.update(groups=[], suffix_blocks=batch.request_blocks, suffix_lens=batch.seq_lens)
    check(setup, *attend(setup, plan))


# --- capacity fallbacks -----------------------------------------------------------


def test_split_k_falls_back_to_one_range_when_parts_do_not_fit() -> None:
    shared = list(range(1, 13))
    batch = explicit_batch([[*shared, 13], [*shared, 14], list(range(15, 27))], [52, 50, 48], 4)
    setup = build(batch)
    rows = 3 * setup.q.shape[1]
    roomy = new_plan(setup, batch, num_sms=64, min_split_tokens=4)
    plan_device(roomy, setup)
    tight = new_plan(setup, batch, num_sms=64, min_split_tokens=4, max_parts=2 * rows)
    plan_device(tight, setup)
    assert int(roomy.item_info[1, : int(roomy.counts[4])].max()) > 1, "roomy plan should split"
    assert int(tight.item_info[1, : int(tight.counts[4])].max()) == 1
    check(setup, *attend(setup, roomy))
    check(setup, *attend(setup, tight))


# --- batch-invariant mode ---------------------------------------------------------


def test_batch_invariant_output_does_not_depend_on_the_batch() -> None:
    rows = [[1, 2, 3, 4, 5], [1, 2, 3, 6], [1, 2, 7], [8, 9]]
    lens = [19, 16, 10, 6]
    batch = explicit_batch(rows, lens, 4)
    setup = build(batch)
    plan = new_plan(setup, batch, batch_invariant=True, num_sms=64, min_split_tokens=4)
    plan_device(plan, setup)
    assert int(plan.counts[3]) == 0, "a batch-invariant plan makes no groups"
    out, lse = attend(setup, plan)
    check(setup, out, lse)
    for r in range(len(lens)):
        alone = Setup(
            setup.q[r : r + 1].clone(),
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
        assert torch.equal(out_r[0], out[r]), f"request {r}"
        assert torch.equal(lse_r[0], lse[r]), f"request {r}"


def test_batch_invariant_plan_rejects_named_groups() -> None:
    batch = explicit_batch([[1, 2], [1, 3]], [8, 8], 4)
    setup = build(batch)
    plan = new_plan(setup, batch, batch_invariant=True)
    with pytest.raises(ValueError, match="batch-invariant"):
        plan.update(
            groups=[PrefixGroup(range(0, 2), [1], 4)],
            suffix_blocks=[[2], [3]],
            suffix_lens=[4, 4],
        )


# --- input checks -----------------------------------------------------------------


def test_device_planning_rejects_bad_inputs() -> None:
    batch = explicit_batch([[1, 2], [1, 3]], [8, 8], 4)
    setup = build(batch)
    plan = new_plan(setup, batch)
    table, lens = setup.index["block_table"], setup.index["seq_lens"]
    with pytest.raises(ValueError, match="int32"):
        plan.plan_from_block_table(table.long(), lens)
    with pytest.raises(ValueError, match="int32"):
        plan.plan_from_block_table(table, lens.long())
    with pytest.raises(ValueError, match="contiguous"):
        plan.plan_from_block_table(table.t().contiguous().t(), lens)
    with pytest.raises(ValueError, match="rows"):
        plan.plan_from_block_table(table[:1], lens)
    with pytest.raises(ValueError, match="max_requests"):
        plan.plan_from_block_table(table.repeat(2, 1), lens.repeat(2))
    with pytest.raises(ValueError, match="slot_indptr"):
        plan.plan_from_slots(lens.repeat(3), table.flatten(), lens)
    with pytest.raises(ValueError, match="max_parts"):
        new_plan(setup, batch, max_parts=3)


def test_a_cuda_device_without_an_index_is_the_current_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kvkernels.ops.cascade import same_device

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cuda, cuda0, cuda1 = torch.device("cuda"), torch.device("cuda:0"), torch.device("cuda:1")
    assert same_device(cuda, cuda0) and same_device(cuda0, cuda)
    assert not same_device(cuda, cuda1) and not same_device(cuda0, cuda1)
    assert same_device(torch.device("cpu"), torch.device("cpu"))
    assert not same_device(torch.device("cpu"), cuda0)


# --- GPU tier ---------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["vllm_LBNHC", "sglang_nhd"])
def test_full_size(layout: str) -> None:
    block_size = 16 if layout.startswith("vllm") else 1
    rng = random.Random(1)
    batch = chains_batch(rng, requests=96, block_size=block_size, chains=6)
    setup = build(batch, layout=layout, q_heads=32, kv_heads=8, head_dim=128)
    plan = new_plan(setup, batch, min_shared_tokens=16)
    plan_device(plan, setup)
    assert found_groups(plan) == expected_groups(batch, plan)
    check(setup, *attend(setup, plan))


@pytest.mark.gpu
def test_batch_invariant_on_the_gpu() -> None:
    batch = chains_batch(random.Random(2), requests=32, block_size=16)
    setup = build(batch, q_heads=32, kv_heads=8, head_dim=128)
    plan = new_plan(setup, batch, batch_invariant=True)
    plan_device(plan, setup)
    out, lse = attend(setup, plan)
    for r in (0, 7, 31):
        alone = Setup(
            setup.q[r : r + 1].clone(),
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
        assert torch.equal(out_r[0], out[r]) and torch.equal(lse_r[0], lse[r]), f"request {r}"


@pytest.mark.gpu
def test_planning_replays_one_graph_while_the_batch_size_changes() -> None:
    """An engine passes the first rows of the same block table and length buffers each step."""
    batch = chains_batch(random.Random(6), requests=24, block_size=16)
    s = build(batch, q_heads=32, kv_heads=8, head_dim=128)
    table, lens = s.index["block_table"], s.index["seq_lens"]
    plan = new_plan(s, batch, max_groups=12, min_shared_tokens=16)
    for n in (24, 24, 17, 9, 24, 1):
        step = Setup(
            s.q[:n], s.k, s.v, {"block_table": table[:n], "seq_lens": lens[:n]}, s.requests[:n]
        )
        plan_device(plan, step)
        check(step, *attend(step, plan))
        fresh = new_plan(s, batch, max_groups=12, min_shared_tokens=16)
        plan_device(fresh, step)  # a first call launches the kernels directly
        assert found_groups(plan) == found_groups(fresh)
        assert torch.equal(plan.counts, fresh.counts)
        rows = n * plan.num_q_heads
        assert torch.equal(plan.merge_start[:rows], fresh.merge_start[:rows])
    assert len(plan.graphs.graphs) == 1


@pytest.mark.gpu
def test_planning_and_attention_capture_in_one_cuda_graph() -> None:
    first = chains_batch(random.Random(4), requests=24, block_size=16)
    second = chains_batch(random.Random(5), requests=24, block_size=16)
    num_blocks = max(first.num_blocks, second.num_blocks)
    first.num_blocks = second.num_blocks = num_blocks
    a = build(first, q_heads=32, kv_heads=8, head_dim=128)
    b = build(second, q_heads=32, kv_heads=8, head_dim=128, seed=1)
    width = max(a.index["block_table"].shape[1], b.index["block_table"].shape[1])
    table = torch.zeros((24, width), dtype=torch.int32, device=DEVICE)
    lens = torch.zeros(24, dtype=torch.int32, device=DEVICE)
    q = torch.empty_like(a.q)
    k, v = a.k, a.v

    def load(s: Setup) -> None:
        table.zero_()
        table[:, : s.index["block_table"].shape[1]] = s.index["block_table"]
        lens.copy_(s.index["seq_lens"])
        q.copy_(s.q)

    plan = new_plan(a, first, max_groups=12, min_shared_tokens=16)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=DEVICE)

    def step() -> None:
        plan.plan_from_block_table(table, lens)
        cascade_decode_attention(q, k, v, plan, out, lse)

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
        check(Setup(q, k, v, s.index, s.requests), out, lse)
