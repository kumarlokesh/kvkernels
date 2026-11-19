"""Cascade decode attention against an fp32 reference over each request's prefix and suffix.

Each request's keys are its group's shared prefix followed by its own suffix.
The cascade computes attention over the prefix once per group and over each
suffix once per request, then merges; the result must match attention over the
joined keys within the tolerance rule err ≤ 2 · err(ref_low) + floor, for both
the output and the LSE.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

import kvkernels.ops.cascade as cascade_module
from kvkernels.ops.cascade import (
    CascadePlan,
    PrefixGroup,
    cascade_decode_attention,
    default_prefix_block_m,
)
from kvkernels.tolerance import assert_close
from tests.ops.cascade_cases import DEVICE, INTERPRETING, Case, make_case, make_plan, references

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)


def run(
    case: Case, plan: CascadePlan | None = None, **plan_args: int
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = plan or make_plan(case, **plan_args)
    out = torch.empty_like(case.q)
    lse = torch.empty(case.q.shape[:2], dtype=torch.float32, device=DEVICE)
    cascade_decode_attention(case.q, case.k, case.v, plan, out, lse)
    return out, lse


def check(case: Case, out: torch.Tensor, lse: torch.Tensor) -> None:
    (out_low, lse_low), (out32, lse32) = references(case)
    assert not torch.isnan(out).any(), "NaN in out"
    assert not torch.isnan(lse).any(), "NaN in lse"
    assert_close(out, out_low, out32)
    assert_close(lse, lse_low, lse32)


def test_one_group_with_ragged_suffixes() -> None:
    case = make_case()
    check(case, *run(case))


def test_batch_of_one() -> None:
    case = make_case(group_lens=[(1, 30)], suffix_lens=[7])
    check(case, *run(case))


def test_two_groups_and_a_request_outside_every_group() -> None:
    case = make_case(group_lens=[(2, 16), (3, 33)], suffix_lens=[4, 0, 6, 1, 12, 8])
    check(case, *run(case))


def test_a_request_with_no_keys_gets_zero_and_minus_infinity() -> None:
    case = make_case(group_lens=[(2, 12)], suffix_lens=[3, 0, 0])  # request 2: no prefix, no suffix
    out, lse = run(case)
    check(case, out, lse)
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    assert torch.isneginf(lse[2]).all()


def test_prefix_ending_inside_a_block() -> None:
    case = make_case(group_lens=[(3, 23)], suffix_lens=[2, 5, 1], block_size=4)
    check(case, *run(case))


@pytest.mark.parametrize(
    ("q_heads", "kv_heads"), [(2, 2), (8, 2), (16, 2)], ids=["gqa1", "gqa4", "gqa8"]
)
def test_gqa_group_sizes(q_heads: int, kv_heads: int) -> None:
    case = make_case(q_heads=q_heads, kv_heads=kv_heads)
    check(case, *run(case))


@pytest.mark.parametrize("layout", ["vllm_LBHNC", "vllm_BLNHC", "sglang_hnd"])
def test_other_layouts(layout: str) -> None:
    case = make_case(layout=layout)
    check(case, *run(case))


def test_sglang_with_one_token_pages() -> None:
    case = make_case(layout="sglang_nhd", block_size=1, group_lens=[(3, 17)], suffix_lens=[2, 6, 3])
    check(case, *run(case))


def test_split_k_gives_the_same_answer() -> None:
    case = make_case(group_lens=[(2, 60)], suffix_lens=[30, 45])
    whole_out, whole_lse = run(case, num_sms=1, min_split_tokens=1024)
    split_out, split_lse = run(case, num_sms=64, min_split_tokens=8)
    check(case, split_out, split_lse)
    torch.testing.assert_close(split_out.float(), whole_out.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(split_lse, whole_lse, rtol=1e-4, atol=1e-4)


def test_one_plan_serves_successive_batches() -> None:
    first = make_case(group_lens=[(3, 20)], suffix_lens=[5, 2, 7], seed=1)
    plan = CascadePlan(
        max_requests=6,
        max_groups=2,
        max_blocks=64,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        block_size=4,
        num_blocks=64,
        device=torch.device(DEVICE),
    )
    plan.update(
        groups=first.groups, suffix_blocks=first.suffix_blocks, suffix_lens=first.suffix_lens
    )
    check(first, *run(first, plan))
    second = make_case(group_lens=[(2, 9), (2, 14)], suffix_lens=[1, 3, 0, 4, 6], seed=2)
    plan.update(
        groups=second.groups, suffix_blocks=second.suffix_blocks, suffix_lens=second.suffix_lens
    )
    check(second, *run(second, plan))


def test_plan_rejects_bad_batches() -> None:
    case = make_case()
    plan = make_plan(case)
    with pytest.raises(ValueError, match="overlap"):
        plan.update(
            groups=[PrefixGroup(range(0, 2), [1, 2], 8), PrefixGroup(range(1, 3), [3], 4)],
            suffix_blocks=case.suffix_blocks,
            suffix_lens=case.suffix_lens,
        )
    with pytest.raises(ValueError, match="blocks"):
        plan.update(
            groups=[PrefixGroup(range(0, 2), [1], 9)],
            suffix_blocks=case.suffix_blocks,
            suffix_lens=case.suffix_lens,
        )
    with pytest.raises(ValueError, match="outside"):
        plan.update(
            groups=[PrefixGroup(range(0, 2), [case.num_blocks], 2)],
            suffix_blocks=case.suffix_blocks,
            suffix_lens=case.suffix_lens,
        )
    with pytest.raises(ValueError, match="max_requests"):
        plan.update(groups=[], suffix_blocks=[[1]] * 9, suffix_lens=[1] * 9)


def test_run_rejects_inputs_that_do_not_match_the_plan() -> None:
    case = make_case()
    plan = make_plan(case)
    out = torch.empty_like(case.q)
    with pytest.raises(ValueError, match="requests"):
        cascade_decode_attention(case.q[:2], case.k, case.v, plan, out[:2])
    with pytest.raises(ValueError, match="dtype"):
        cascade_decode_attention(case.q, case.k.float(), case.v, plan, out)
    with pytest.raises(ValueError, match="block_size"):
        cascade_decode_attention(case.q, case.k[:, :2], case.v[:, :2], plan, out)


def test_op_is_registered_and_declares_what_it_writes() -> None:
    schema = torch.ops.kvkernels.cascade_decode_attention.default._schema
    written = [a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write]
    assert written == ["workspace", "out", "lse_out"]


@pytest.mark.parametrize(
    ("capability", "head_dim", "rows"),
    [((7, 5), 128, 64), ((7, 5), 64, 128), ((8, 0), 128, 128), ((8, 9), 128, 128)],
)
def test_prefix_tile_is_64_rows_where_128_rows_do_not_fit_in_shared_memory(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], head_dim: int, rows: int
) -> None:
    monkeypatch.setattr(cascade_module, "_capability", lambda index: capability)
    assert default_prefix_block_m(torch.device("cuda", 0), head_dim) == rows
    assert default_prefix_block_m(torch.device("cpu"), head_dim) == 128  # the interpreter


# --- GPU tier ---------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_full_size(head_dim: int, dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16 on this GPU")
    case = make_case(
        group_lens=[(16, 4096), (8, 1000)],
        suffix_lens=[(37 * r) % 700 for r in range(32)],
        q_heads=32,
        kv_heads=8,
        head_dim=head_dim,
        block_size=16,
        dtype=dtype,
    )
    check(case, *run(case))


@pytest.mark.gpu
def test_capture_and_replay_in_a_cuda_graph_across_plan_updates() -> None:
    first = make_case(
        group_lens=[(8, 512)],
        suffix_lens=[64] * 12,
        q_heads=32,
        kv_heads=8,
        head_dim=128,
        block_size=16,
        seed=3,
    )
    plan = make_plan(first)
    out = torch.empty_like(first.q)
    lse = torch.empty(first.q.shape[:2], dtype=torch.float32, device=DEVICE)
    cascade_decode_attention(first.q, first.k, first.v, plan, out, lse)  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cascade_decode_attention(first.q, first.k, first.v, plan, out, lse)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    check(first, out, lse)
