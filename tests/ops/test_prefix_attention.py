"""The attention kernel over work items, exercised through each kind of work item alone.

A batch with no suffixes runs only prefix work items (many requests' rows over a
shared block list); a batch with no prefix groups runs only suffix work items
(one request's rows each). Both must match the reference by themselves.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.ops.cascade import cascade_decode_attention
from tests.ops.cascade_cases import INTERPRETING, make_case, make_plan
from tests.ops.test_cascade import check, run

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)


def test_prefix_items_alone() -> None:
    case = make_case(group_lens=[(4, 37)], suffix_lens=[0, 0, 0, 0])
    check(case, *run(case))


def test_suffix_items_alone() -> None:
    case = make_case(group_lens=[], suffix_lens=[13, 1, 40, 7])
    check(case, *run(case))


def test_a_group_taller_than_one_row_tile() -> None:
    # 20 requests · 4 query heads per KV head = 80 rows: tiles of 64 rows with
    # 64-row prefix tiles, one 128-row tile by default.
    case = make_case(group_lens=[(20, 25)], suffix_lens=[2] * 20)
    check(case, *run(case))
    check(case, *run(case, prefix_block_m=64))


# Groups of 6, 12 and 40 requests with a GQA group of 4: 24, 48 and 160 rows per
# KV head. With 128-row tiles and a smallest tile of 32, the 24 rows run in a
# 32-row tile, the 48 in a 64-row tile, and the 160 in one 128-row tile and one
# 32-row tile for the last 32 rows.
SHORT_TILE_GROUPS = [(6, 21), (12, 30), (40, 18)]


@pytest.mark.parametrize("min_block_m", [128, 64, 32])
def test_tasks_with_few_rows_run_in_shorter_tiles(min_block_m: int) -> None:
    case = make_case(group_lens=SHORT_TILE_GROUPS, suffix_lens=[3, 0] * 29)
    check(case, *run(case, prefix_min_block_m=min_block_m))


@pytest.mark.parametrize("min_block_m", [64, 32])
def test_shorter_tiles_with_split_ranges(min_block_m: int) -> None:
    case = make_case(group_lens=[(6, 150), (12, 90)], suffix_lens=[0] * 18, block_size=4)
    check(case, *run(case, num_sms=32, min_split_tokens=16, prefix_min_block_m=min_block_m))


@pytest.mark.parametrize("min_block_m", [96, 16, 8])
def test_a_smallest_tile_that_is_not_a_half_or_quarter_is_rejected(min_block_m: int) -> None:
    case = make_case()
    with pytest.raises(ValueError, match="prefix_min_block_m"):
        run(case, prefix_min_block_m=min_block_m)


def test_a_prefix_spanning_many_key_tiles_with_split_ranges() -> None:
    case = make_case(group_lens=[(3, 150)], suffix_lens=[0, 0, 0], block_size=4)
    check(case, *run(case, num_sms=32, min_split_tokens=16))


def test_a_head_dimension_that_is_not_contiguous_is_rejected() -> None:
    # The kernel reads head_dim as contiguous, so that K and V load 16 bytes at a
    # time; a view with another stride there must be refused, not misread.
    case = make_case()
    plan = make_plan(case)
    strided = torch.empty(
        (*case.k.shape[:3], 2 * case.k.shape[3]), dtype=case.k.dtype, device=case.k.device
    )[..., ::2]
    strided.copy_(case.k)
    out = torch.empty_like(case.q)
    with pytest.raises(ValueError, match="contiguous head dimension"):
        cascade_decode_attention(case.q, strided, case.v, plan, out)
