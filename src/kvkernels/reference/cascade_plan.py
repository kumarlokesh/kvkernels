"""Reference for finding prefix groups from the engine's index.

Requests are sorted by (first KV block, request). Each sorted row is compared
with the row before it over the KV blocks both have filled completely; a run is
a stretch of rows with the same first KV block, and its shared count is the
smallest comparison inside it:

    block_size 16          full blocks   sorted   shared with previous row
    r0  [7 2 9 4]  60 tok      3          r1 [3 8 1]     -    ┐ run: shared 2
    r1  [3 8 1]    40 tok      2          r4 [3 8 5]     2    ┘
    r2  [7 2 9 6]  62 tok      3          r0 [7 2 9 4]   -    ┐
    r3  [7 2 5]    40 tok      2          r2 [7 2 9 6]   3    │ run: shared min(3, 2) = 2
    r4  [3 8 5]    35 tok      2          r3 [7 2 5]     2    ┘

A run of two or more rows whose shared count reaches the threshold is a group,
up to max_groups groups in sorted order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FoundGroup:
    """A group: sorted rows first … first + rows - 1, sharing their first `blocks` KV blocks."""

    first: int
    rows: int
    blocks: int


def find_prefix_groups_ref(
    request_blocks: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    *,
    block_size: int,
    min_shared_tokens: int,
    max_groups: int,
) -> tuple[list[int], list[FoundGroup]]:
    """(perm, groups): sorted row i is request perm[i]; groups in sorted-row order."""
    full = [n // block_size for n in seq_lens]
    no_key = 2**31 - 1

    def key(r: int) -> tuple[int, int]:
        return (request_blocks[r][0] if full[r] > 0 else no_key, r)

    perm = sorted(range(len(seq_lens)), key=key)

    def common(a: int, b: int) -> int:
        n = min(full[a], full[b])
        for j in range(n):
            if request_blocks[a][j] != request_blocks[b][j]:
                return j
        return n

    shared = [0] + [common(perm[i - 1], perm[i]) for i in range(1, len(perm))]
    min_blocks = max(1, math.ceil(min_shared_tokens / block_size))
    groups: list[FoundGroup] = []
    i = 0
    while i < len(perm):
        end = i + 1
        while end < len(perm) and shared[end] > 0:
            end += 1
        if end - i >= 2:
            run_shared = min(shared[i + 1 : end])
            if run_shared >= min_blocks and len(groups) < max_groups:
                groups.append(FoundGroup(i, end - i, run_shared))
        i = end
    return perm, groups
