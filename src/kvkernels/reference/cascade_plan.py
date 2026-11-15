"""References for finding prefix groups, and the prefix tree, from the engine's index.

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


@dataclass(frozen=True)
class TreeItem:
    """A node that became a work item: sorted rows first … first + rows - 1 share
    KV blocks start … end - 1, found in round `level` (1 is the first)."""

    level: int
    first: int
    rows: int
    start: int
    end: int


def find_prefix_tree_ref(
    request_blocks: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    *,
    block_size: int,
    min_shared_tokens: int,
    max_groups: int,
    max_levels: int,
    query_tokens: int = 1,
) -> tuple[list[int], list[TreeItem]]:
    """(perm, items): sorted row i is request perm[i]; node items in (round, row) order.

    Round 1 sorts requests by (first KV block, request). Round k + 1 sorts the
    rows of each node from round k by (KV block at the node's depth, request),
    in place. A run of two or more rows with the same key is a node; its depth is
    the smallest common prefix of neighbouring rows, compared over the KV blocks
    that end at or before each request's first new query token. A node becomes
    an item when it holds at least min_shared_tokens · query_tokens tokens
    beyond the last item above it, up to max_groups items; otherwise its rows'
    next item starts where the last one ended.
    """
    n = len(seq_lens)
    usable = [max(length - query_tokens + 1, 0) // block_size for length in seq_lens]
    no_key = 2**31 - 1

    def common(a: int, b: int) -> int:
        m = min(usable[a], usable[b])
        for j in range(m):
            if request_blocks[a][j] != request_blocks[b][j]:
                return j
        return m

    perm = list(range(n))
    # Segments: (first row, rows, depth) of the last round's nodes; round 1 has one.
    segments = [(0, n, 0)] if n else []
    nodes: list[
        list[tuple[int, int, int, int]]
    ] = []  # per round: (first, rows, depth, parent depth)
    for _ in range(max_levels):
        found: list[tuple[int, int, int, int]] = []
        next_segments = []
        for first, rows, depth in segments:

            def key(r: int, depth: int = depth) -> tuple[int, int]:
                return (request_blocks[r][depth] if depth < usable[r] else no_key, r)

            part = sorted(perm[first : first + rows], key=key)
            perm[first : first + rows] = part
            i = 0
            while i < rows:
                k = key(part[i])[0]
                end = i + 1
                while end < rows and k != no_key and key(part[end])[0] == k:
                    end += 1
                if end - i >= 2:
                    node_depth = min(common(part[j - 1], part[j]) for j in range(i + 1, end))
                    found.append((first + i, end - i, node_depth, depth))
                    next_segments.append((first + i, end - i, node_depth))
                i = end
        if not found:
            break
        nodes.append(sorted(found))
        segments = next_segments

    min_blocks = max(1, math.ceil(min_shared_tokens * query_tokens / block_size))
    covered = [0] * n  # by sorted row: KV blocks held by the row's items so far
    items: list[TreeItem] = []
    for level, found in enumerate(nodes, start=1):
        for first, rows, node_depth, _ in found:
            start = covered[first]
            if node_depth - start >= min_blocks and len(items) < min(max_groups, n):
                items.append(TreeItem(level, first, rows, start, node_depth))
                for row in range(first, first + rows):
                    covered[row] = node_depth
    return perm, items
