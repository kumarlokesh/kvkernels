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

from tests.ops.cascade_cases import INTERPRETING, make_case
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
    # 20 requests · 4 query heads per KV head = 80 rows: two tiles of 64 rows, one prefix.
    case = make_case(group_lens=[(20, 25)], suffix_lens=[2] * 20)
    check(case, *run(case))


def test_a_prefix_spanning_many_key_tiles_with_split_ranges() -> None:
    case = make_case(group_lens=[(3, 150)], suffix_lens=[0, 0, 0], block_size=4)
    check(case, *run(case, num_sms=32, min_split_tokens=16))
