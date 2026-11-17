"""Planning the cascade on the GPU from the engine's own index.

Requests that share a prefix share the leading entries of their index rows:
prefix caching hands two requests the same physical KV block exactly where
their tokens, and every token before them, agree. Two rows are therefore equal
on a run of leading KV blocks and differ from the first unequal one on, which
lets planning find the whole prefix tree with sorts and binary searches, with
no tokens and no tree from the engine. Three launches turn the index into the
plan's buffers, with no value copied to the host:

    index rows, lengths (device)
         │
         ▼ tree    rounds of: sort rows by (node of the last round, KV block   ──▶ perm, node and
         │         at that node's depth); runs of equal keys are the next         depth per row
         │         level's nodes; each node's depth is the smallest common        and round
         │         prefix of neighbouring rows, by binary search; one program
         ▼ items   node items (a node with too few tokens of its own is        ──▶ items, counts,
         │         folded into its children); one suffix item per request;        merge rows
         │         split-K ranges; offsets; one program
         ▼ fill    each item's tasks and merge entries; one program per item  ──▶ tasks, merge list

Rounds for an example batch (block size 16):

    r0 [7 2 9 4 1]  r1 [7 2 9 4 8]  r2 [7 2 9 6 3]  r3 [7 2 5 10]  r4 [3 8 1]

    round  key                order               nodes      common prefix with the previous row
    1      first block        r4|r0 r1 r2 r3      A: r0-r3   r1 4, r2 3, r3 2  ─▶ A ends at block 2
    2      (A, block 2)       r4|r3|r0 r1 r2      B: r0-r2   r1 4, r2 3        ─▶ B ends at block 3
    3      (B, block 3)       r4|r3|r0 r1|r2      C: r0 r1   r1 4              ─▶ C ends at block 4

Each round's sort keeps every earlier node's rows in place, so every node is a
range of sorted rows. Only KV blocks that end at or before each request's first
new query token are compared, so a node never holds a key that one of its
query tokens must not see.

Two index forms are read in place, never copied:

    block table   one row of KV block ids per request (vLLM); entry j of request r
                  holds tokens j · block_size … j · block_size + block_size - 1
    slot list     one slot per token, rows back to back (SGLang); slot s is
                  token s % page_size of page s // page_size

For a slot list, entries are compared once per page: the engine allocates
pages whole and places token t at offset t % page_size of its page, so two
requests with the same first slot in a page share the whole page.

Sizes are fixed by the plan's maxima and the batch size is read from the
device, so the launches capture in a CUDA graph, and the plan records its own
graph to replay on later steps (PlanGraphs). Nothing raises on the device:
when the batch's split-K partial results or tasks would exceed the plan's
capacity, every item falls back to one range, and nodes past max_groups are
folded into their children.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from kvkernels.ops.cascade import CascadePlan

MAX_PLAN_REQUESTS = 4096  # every request's sort key is held by one program
_NO_KEY = tl.constexpr(0x7FFFFFFF)  # key block of a row that cannot share further
_BIG = tl.constexpr(0x7FFFFFFF)
# Key limit of a node item: every query token sees all its keys. Small enough that
# adding a query token's index cannot overflow int32.
_NO_LIMIT = tl.constexpr(1 << 30)


@triton.jit
def _maximum(a, b):  # type: ignore[no-untyped-def]
    return tl.maximum(a, b)


@triton.jit
def _minimum(a, b):  # type: ignore[no-untyped-def]
    return tl.minimum(a, b)


@triton.jit
def _segment_min(flag_a, a, flag_b, b):  # type: ignore[no-untyped-def]
    """Running minimum that restarts at every element whose flag is set."""
    return flag_a | flag_b, tl.where(flag_b != 0, b, tl.minimum(a, b))


@triton.jit
def _row_base(index_ptr, starts_ptr, request, stride_r, PER_TOKEN: tl.constexpr):  # type: ignore[no-untyped-def]
    """Offset of a request's first index entry."""
    if PER_TOKEN:
        return tl.load(starts_ptr + request).to(tl.int64)
    return request.to(tl.int64) * stride_r


@triton.jit
def _usable_blocks(seq_lens_ptr, request, ok, block_size, query_tokens):  # type: ignore[no-untyped-def]
    """KV blocks that end at or before the request's first new query token."""
    length = tl.load(seq_lens_ptr + request, mask=ok, other=0)
    return tl.maximum(length - query_tokens + 1, 0) // block_size


@triton.jit
def _tree_kernel(  # type: ignore[no-untyped-def]
    index_ptr,
    starts_ptr,
    seq_lens_ptr,
    perm_ptr,
    tree_ptr,  # [2, MAX_LEVELS, max_requests]: node's first row (-1: none), node depth
    counts_ptr,
    num_requests_ptr,
    stride_r,
    block_size,
    query_tokens,
    max_requests,
    MAX_LEVELS: tl.constexpr,
    PER_TOKEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    num_requests = tl.load(num_requests_ptr)
    i = tl.arange(0, BLOCK_R)
    ok = i < num_requests
    prev = tl.maximum(i - 1, 0)
    nxt = tl.minimum(i + 1, BLOCK_R - 1)
    step = block_size if PER_TOKEN else 1  # a slot list is compared at each page's first slot
    request = i
    # A row's segment: the first sorted row of its node from the last round, or
    # itself once it shares nothing further. Round 1 puts every real row in one.
    seg = tl.where(ok, 0, i)
    depth = tl.zeros_like(i)  # KV blocks of the row's node from the last round
    active = ok
    level = tl.sum(tl.zeros_like(i), 0)  # an int32 scalar, so the loop can carry it
    found = level + 1
    while (level < MAX_LEVELS) & (found > 0):
        usable = _usable_blocks(seq_lens_ptr, request, ok, block_size, query_tokens)
        base = _row_base(index_ptr, starts_ptr, tl.where(ok, request, 0), stride_r, PER_TOKEN)
        has = active & (depth < usable)
        block = tl.load(index_ptr + base + depth.to(tl.int64) * step, mask=has, other=0)
        block = tl.where(has, block.to(tl.int64), _NO_KEY)
        # Segment, key block and request in one integer: the sort refines the
        # last round's order, and ties keep request order.
        key = tl.sort((seg.to(tl.int64) << 43) | (block << 12) | request.to(tl.int64))
        request = (key & 0xFFF).to(tl.int32)
        block = (key >> 12) & 0x7FFFFFFF
        group_key = key >> 12  # segment and key block
        same_prev = ok & (i > 0) & (block != _NO_KEY) & (tl.gather(group_key, prev, 0) == group_key)
        same_next = tl.gather(same_prev.to(tl.int32), nxt, 0) != 0
        same_next = same_next & (i + 1 < BLOCK_R)
        is_start = ok & ~same_prev
        is_end = ok & ~same_next
        start_pos = tl.associative_scan(tl.where(is_start, i, 0), 0, _maximum)
        end_pos = tl.associative_scan(tl.where(is_end, i, BLOCK_R - 1), 0, _minimum, reverse=True)
        in_node = ok & (end_pos > start_pos)

        # Common prefix with the previous row, in KV blocks, by binary search:
        # both rows hold the key block at `depth`, and equality holds on a
        # leading run of blocks only.
        usable = _usable_blocks(seq_lens_ptr, request, ok, block_size, query_tokens)
        base = _row_base(index_ptr, starts_ptr, tl.where(ok, request, 0), stride_r, PER_TOKEN)
        base_prev = tl.gather(base, prev, 0)
        lo = tl.where(same_prev, depth + 1, 0)
        hi = tl.where(same_prev, tl.minimum(usable, tl.gather(usable, prev, 0)), 0)
        while tl.max(hi - lo, 0) > 0:
            mid = (lo + hi + 1) // 2
            go = hi > lo
            at = (mid - 1).to(tl.int64) * step
            a = tl.load(index_ptr + base_prev + at, mask=go, other=0)
            b = tl.load(index_ptr + base + at, mask=go, other=0)
            lo = tl.where(go & (a == b), mid, lo)
            hi = tl.where(go & (a != b), mid - 1, hi)
        _, running = tl.associative_scan(
            (is_start.to(tl.int32), tl.where(is_start, _BIG, lo)), 0, _segment_min
        )
        node_depth = tl.gather(running, end_pos, 0)

        out = tree_ptr + level * max_requests + i
        tl.store(out, tl.where(in_node, start_pos, -1), mask=ok)
        tl.store(out + MAX_LEVELS * max_requests, tl.where(in_node, node_depth, 0), mask=ok)
        active = in_node
        seg = tl.where(in_node, start_pos, i)
        depth = tl.where(in_node, node_depth, depth)
        found = tl.sum(in_node.to(tl.int32), 0)
        level += 1
    tl.store(perm_ptr + i, request, mask=ok)
    tl.store(counts_ptr + 5, level)


@triton.jit
def _ranges(  # type: ignore[no-untyped-def]
    length,
    rows,
    item_rows,
    budget_rows,
    base_programs,
    allow_split,
    target,
    min_split,
    split_floor,
    split_per_row,
    block_n,
    MAX_SPLITS: tl.constexpr,
    INVARIANT: tl.constexpr,
):
    """Split-K token span and number of ranges for items of the given lengths.

    rows: query rows of the item's tile; target: programs of its launch that run at once.
    """
    if INVARIANT:
        splits = tl.cdiv(length, min_split)
    else:
        min_tokens = tl.minimum(min_split, tl.maximum(split_floor, split_per_row * rows))
        cap = tl.minimum(tl.maximum(tl.cdiv(length, min_tokens), 1), MAX_SPLITS)
        # The item's partial results (8 bytes per row and head dimension per range)
        # are no larger than its K and V, or fit in a share of L2: ranges times rows
        # at most budget_rows.
        item_rows = tl.maximum(item_rows, 1)
        by_kv = length // tl.maximum(split_per_row * item_rows, 1)
        by_bytes = tl.maximum(by_kv, budget_rows // item_rows)
        cap = tl.minimum(cap, tl.maximum(by_bytes, 1))
        # The s whose programs fill the launch's rounds best: ceil(base · s / R) / s
        # smallest, compared as ceil(base · s / R) · best_s < best_rounds · s.
        base = tl.maximum(base_programs, 1)
        splits = tl.full(length.shape, 1, tl.int32)
        best_rounds = tl.cdiv(base, target) + splits * 0
        for s in tl.static_range(2, MAX_SPLITS + 1):
            rounds = tl.cdiv(base * s, target)
            better = (s <= cap) & (rounds * splits < best_rounds * s)
            splits = tl.where(better, s, splits)
            best_rounds = tl.where(better, rounds, best_rounds)
        # A launch that fills a round already gets no split: more ranges add merge work.
        splits = tl.where(base_programs >= target, 1, splits)
    splits = tl.minimum(tl.maximum(splits, 1), MAX_SPLITS)
    splits = tl.where(allow_split, splits, 1)
    span = tl.cdiv(tl.cdiv(tl.maximum(length, 1), splits), block_n) * block_n
    return span, tl.where(length > 0, tl.cdiv(length, span), 0)


@triton.jit
def _store_items(  # type: ignore[no-untyped-def]
    item_req_start_ptr,
    item_num_reqs_ptr,
    item_index_start_ptr,
    item_info_ptr,
    max_items,
    at,
    mask,
    req_start,
    num_reqs,
    index_start,
    length,
    ranges,
    span,
    task_start,
    part_start,
    merge_offset,
):
    """Write the items held by the masked lanes to item slots `at`."""
    tl.store(item_req_start_ptr + at, req_start, mask=mask)
    tl.store(item_num_reqs_ptr + at, num_reqs, mask=mask)
    tl.store(item_index_start_ptr + at, index_start.to(tl.int32), mask=mask)
    tl.store(item_info_ptr + at, length, mask=mask)
    tl.store(item_info_ptr + max_items + at, ranges, mask=mask)
    tl.store(item_info_ptr + 2 * max_items + at, span, mask=mask)
    tl.store(item_info_ptr + 3 * max_items + at, task_start, mask=mask)
    tl.store(item_info_ptr + 4 * max_items + at, part_start, mask=mask)
    tl.store(item_info_ptr + 5 * max_items + at, merge_offset, mask=mask)


@triton.jit
def _items_kernel(  # type: ignore[no-untyped-def]
    index_ptr,
    starts_ptr,
    seq_lens_ptr,
    perm_ptr,
    tree_ptr,
    item_req_start_ptr,
    item_num_reqs_ptr,
    item_index_start_ptr,
    item_key_limit_ptr,
    item_info_ptr,  # [6, max_items]: length, ranges, span, task start, part start, merge offset
    counts_ptr,  # prefix tasks, suffix tasks, merge rows, node items, items, rounds
    merge_start_ptr,
    merge_end_ptr,
    num_requests_ptr,
    stride_r,
    block_size,
    query_tokens,
    min_shared_blocks,
    max_groups,
    max_requests,
    max_items,
    max_tasks,
    max_parts,
    prefix_target,
    suffix_target,
    min_split,
    split_floor,
    split_per_row,
    block_n,
    kv_heads,
    q_heads,
    small_rows,
    part_budget_rows,
    GROUP: tl.constexpr,
    MAX_LEVELS: tl.constexpr,
    PREFIX_BM: tl.constexpr,
    SUFFIX_BM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PER_TOKEN: tl.constexpr,
    INVARIANT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    num_requests = tl.load(num_requests_ptr)
    rounds = tl.load(counts_ptr + 5)
    i = tl.arange(0, BLOCK_R)
    ok = i < num_requests
    nxt = tl.minimum(i + 1, BLOCK_R - 1)
    request = tl.load(perm_ptr + i, mask=ok, other=0)
    length = tl.load(seq_lens_ptr + request, mask=ok, other=0)
    base = _row_base(index_ptr, starts_ptr, request, stride_r, PER_TOKEN)
    unit = block_size if PER_TOKEN else 1  # index entries per KV block
    rows_per_request = GROUP * query_tokens

    # Node items, round by round. A node becomes an item when it has at least
    # min_shared_blocks KV blocks beyond the last item above it (covered); a
    # node below that is folded: its children start where covered ends. The
    # item that holds each row in each round replaces the node's first row in
    # the tree buffer, for the merge offsets below.
    covered = tl.zeros_like(i)
    num_groups = tl.sum(tl.zeros_like(i), 0)
    for level in range(rounds):
        at_level = tree_ptr + level * max_requests + i
        first = tl.load(at_level, mask=ok, other=-1)
        node_depth = tl.load(at_level + MAX_LEVELS * max_requests, mask=ok, other=0)
        in_node = ok & (first >= 0)
        safe_first = tl.where(in_node, first, 0)
        is_first = in_node & (first == i)
        last = in_node & ((tl.gather(first, nxt, 0) != first) | (i + 1 >= num_requests))
        end_pos = tl.associative_scan(tl.where(last, i, BLOCK_R - 1), 0, _minimum, reverse=True)
        own = node_depth - covered
        qualifies = (i < 0) if INVARIANT else is_first & (own >= min_shared_blocks)
        index = num_groups + tl.cumsum(qualifies.to(tl.int32), 0) - 1
        is_item = qualifies & (index < tl.minimum(max_groups, max_requests))
        num_groups += tl.sum(is_item.to(tl.int32), 0)
        member = in_node & (tl.gather(is_item.to(tl.int32), safe_first, 0) != 0)
        at = tl.where(is_item, index, 0)
        tl.store(item_req_start_ptr + at, i, mask=is_item)
        tl.store(item_num_reqs_ptr + at, end_pos - i + 1, mask=is_item)
        tl.store(item_index_start_ptr + at, (base + covered * unit).to(tl.int32), mask=is_item)
        tl.store(item_key_limit_ptr + at, _NO_LIMIT, mask=is_item)
        tl.store(item_info_ptr + at, own * block_size, mask=is_item)
        tl.store(at_level, tl.where(member, tl.gather(index, safe_first, 0), -1), mask=ok)
        covered = tl.where(member, node_depth, covered)

    # Lanes below read what other lanes stored above.
    tl.debug_barrier()

    # Suffix items: each request's tokens after its last node item.
    suffix_len = tl.where(ok, length - covered * block_size, 0)
    has_suffix = suffix_len > 0
    suffix_index = num_groups + tl.cumsum(has_suffix.to(tl.int32), 0) - 1
    num_items = num_groups + tl.sum(has_suffix.to(tl.int32), 0)

    # Node items as lanes: lane j holds item j.
    is_group = i < num_groups
    group_len = tl.load(item_info_ptr + i, mask=is_group, other=0)
    group_rows = tl.load(item_num_reqs_ptr + i, mask=is_group, other=0) * rows_per_request
    # A node item of at most small_rows rows runs in the suffix launch's tiles.
    small = is_group & (group_rows <= small_rows)
    large = is_group & (group_rows > small_rows)
    group_tiles = tl.where(
        large, tl.cdiv(group_rows, PREFIX_BM), tl.where(small, tl.cdiv(group_rows, SUFFIX_BM), 0)
    )
    suffix_tiles = tl.where(has_suffix, tl.cdiv(rows_per_request, SUFFIX_BM), 0)

    # Split-K, with a fallback to one range per item when the batch would not fit.
    # Each launch's programs are counted alone: the prefix and suffix launches
    # run one after the other.
    p_base = kv_heads * tl.sum(tl.where(large, group_tiles, 0), 0)
    s_base = kv_heads * (tl.sum(tl.where(small, group_tiles, 0), 0) + tl.sum(suffix_tiles, 0))
    g_base = tl.where(small, s_base, p_base)
    g_target = tl.where(small, suffix_target, prefix_target)
    g_tile_rows = tl.minimum(group_rows, tl.where(small, SUFFIX_BM, PREFIX_BM))
    s_tile_rows = tl.minimum(rows_per_request, SUFFIX_BM)
    _, g_ranges = _ranges(
        group_len,
        g_tile_rows,
        group_rows,
        part_budget_rows,
        g_base,
        True,
        g_target,
        min_split,
        split_floor,
        split_per_row,
        block_n,
        MAX_SPLITS,
        INVARIANT,
    )
    _, s_ranges = _ranges(
        suffix_len,
        s_tile_rows,
        rows_per_request,
        part_budget_rows,
        s_base,
        True,
        suffix_target,
        min_split,
        split_floor,
        split_per_row,
        block_n,
        MAX_SPLITS,
        INVARIANT,
    )
    g_parts = tl.where(is_group, group_rows * kv_heads * g_ranges, 0)
    s_parts = rows_per_request * kv_heads * s_ranges
    g_tasks = group_tiles * g_ranges
    fits = (
        (tl.sum(g_parts, 0) + tl.sum(s_parts, 0) <= max_parts)
        & (tl.sum(tl.where(large, g_tasks, 0), 0) <= max_tasks)
        & (tl.sum(tl.where(small, g_tasks, 0), 0) + tl.sum(suffix_tiles * s_ranges, 0) <= max_tasks)
    )
    g_span1, g_ranges1 = _ranges(
        group_len,
        g_tile_rows,
        group_rows,
        part_budget_rows,
        g_base,
        fits,
        g_target,
        min_split,
        split_floor,
        split_per_row,
        block_n,
        MAX_SPLITS,
        INVARIANT,
    )
    s_span1, s_ranges1 = _ranges(
        suffix_len,
        s_tile_rows,
        rows_per_request,
        part_budget_rows,
        s_base,
        fits,
        suffix_target,
        min_split,
        split_floor,
        split_per_row,
        block_n,
        MAX_SPLITS,
        INVARIANT,
    )
    g_ranges1 = tl.where(is_group, g_ranges1, 0)
    g_parts = tl.where(is_group, group_rows * kv_heads * g_ranges1, 0)
    s_parts = rows_per_request * kv_heads * s_ranges1
    g_tasks = group_tiles * g_ranges1
    s_tasks = suffix_tiles * s_ranges1
    large_tasks = tl.where(large, g_tasks, 0)
    small_tasks = tl.where(small, g_tasks, 0)

    # Offsets: node items first, then suffix items in sorted-row order. The
    # prefix task table holds the large node items; the suffix task table holds
    # the small node items, then the suffix items.
    g_task_start = tl.where(
        large, tl.cumsum(large_tasks, 0) - large_tasks, tl.cumsum(small_tasks, 0) - small_tasks
    )
    s_task_start = tl.sum(small_tasks, 0) + tl.cumsum(s_tasks, 0) - s_tasks
    g_part_start = tl.cumsum(g_parts, 0) - g_parts
    s_part_start = tl.sum(g_parts, 0) + tl.cumsum(s_parts, 0) - s_parts
    item_at = tl.where(is_group, i, 0)
    tl.store(item_info_ptr + max_items + item_at, g_ranges1, mask=is_group)
    tl.store(item_info_ptr + 2 * max_items + item_at, g_span1, mask=is_group)
    tl.store(item_info_ptr + 3 * max_items + item_at, g_task_start, mask=is_group)
    tl.store(item_info_ptr + 4 * max_items + item_at, g_part_start, mask=is_group)

    # Merge lists: a row holds its node items' ranges, ancestors first, then
    # its suffix's. An item's merge offset is the ranges of the items above it.
    path_ranges = tl.zeros_like(i)
    for level in range(rounds):
        at_level = tree_ptr + level * max_requests + i
        item = tl.load(at_level, mask=ok, other=-1)
        member = item >= 0
        item_ranges = tl.where(member, tl.gather(g_ranges1, tl.where(member, item, 0), 0), 0)
        is_first = member & (tl.load(item_req_start_ptr + tl.where(member, item, 0)) == i)
        tl.store(item_info_ptr + 5 * max_items + item, path_ranges, mask=is_first)
        path_ranges += item_ranges

    s_at = tl.where(has_suffix, suffix_index, 0)
    # A suffix item's token t sees its tokens up to suffix_len - query_tokens + t.
    _store_items(
        item_req_start_ptr,
        item_num_reqs_ptr,
        item_index_start_ptr,
        item_info_ptr,
        max_items,
        s_at,
        has_suffix,
        i,
        tl.full(i.shape, 1, tl.int32),
        base + covered * unit,
        suffix_len,
        s_ranges1,
        s_span1,
        s_task_start,
        s_part_start,
        path_ranges,
    )
    tl.store(item_key_limit_ptr + s_at, suffix_len - query_tokens, mask=has_suffix)

    per_row = path_ranges + tl.where(has_suffix, s_ranges1, 0)
    rows_of_request = query_tokens * q_heads  # merge rows: (request · T + token) · q_heads + head
    row_start = tl.cumsum(per_row * rows_of_request, 0) - per_row * rows_of_request
    for r in range(rows_of_request):
        row = request * rows_of_request + r
        first_entry = row_start + r * per_row
        tl.store(merge_start_ptr + row, first_entry, mask=ok)
        tl.store(merge_end_ptr + row, first_entry + per_row, mask=ok)

    tl.store(counts_ptr + 0, tl.sum(large_tasks, 0))
    tl.store(counts_ptr + 1, tl.sum(small_tasks, 0) + tl.sum(s_tasks, 0))
    tl.store(counts_ptr + 2, num_requests * rows_of_request)
    tl.store(counts_ptr + 3, num_groups)
    tl.store(counts_ptr + 4, num_items)


@triton.jit
def _fill_kernel(  # type: ignore[no-untyped-def]
    perm_ptr,
    item_req_start_ptr,
    item_num_reqs_ptr,
    item_info_ptr,
    counts_ptr,
    # [2, 5, max_tasks]: prefix, then suffix tasks; columns item, row start,
    # token start, token end, part base.
    tasks_ptr,
    merge_start_ptr,
    merge_indices_ptr,
    max_items,
    max_tasks,
    kv_heads,
    q_heads,
    query_tokens,
    small_rows,
    GROUP: tl.constexpr,
    PREFIX_BM: tl.constexpr,
    SUFFIX_BM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item = tl.program_id(0)
    if item >= tl.load(counts_ptr + 4):
        return
    # Programs (item, c) for c < num_programs(1) share the item's tasks and
    # merge entries, BLOCK at a time: an item of many rows and ranges has tens
    # of thousands of entries.
    part = tl.program_id(1)
    stride = tl.num_programs(1) * BLOCK
    req_start = tl.load(item_req_start_ptr + item)
    rows_per_request = GROUP * query_tokens
    rows = tl.load(item_num_reqs_ptr + item) * rows_per_request
    # Large node items use the prefix launch; small ones and suffix items the suffix launch.
    is_large = (item < tl.load(counts_ptr + 3)) & (rows > small_rows)
    bm = tl.where(is_large, PREFIX_BM, SUFFIX_BM)
    table = tasks_ptr + tl.where(is_large, 0, 5 * max_tasks)
    length = tl.load(item_info_ptr + item)
    ranges = tl.load(item_info_ptr + max_items + item)
    span = tl.load(item_info_ptr + 2 * max_items + item)
    task_start = tl.load(item_info_ptr + 3 * max_items + item)
    part_start = tl.load(item_info_ptr + 4 * max_items + item)
    merge_offset = tl.load(item_info_ptr + 5 * max_items + item)
    lanes = tl.arange(0, BLOCK)

    # Tasks: row tiles outer, token ranges inner; a task's partial results are
    # kv_heads · tile_rows consecutive slots.
    num_tasks = tl.cdiv(rows, bm) * ranges
    for first in range(part * BLOCK, num_tasks, stride):
        k = first + lanes
        live = k < num_tasks
        tile = k // ranges
        rng = k % ranges
        tile_rows = tl.minimum(bm, rows - tile * bm)
        tok_start = rng * span
        at = task_start + k
        tl.store(table + at, item + k * 0, mask=live)
        tl.store(table + max_tasks + at, tile * bm, mask=live)
        tl.store(table + 2 * max_tasks + at, tok_start, mask=live)
        tl.store(table + 3 * max_tasks + at, tl.minimum(tok_start + span, length), mask=live)
        part_base = part_start + tile * ranges * kv_heads * bm + rng * kv_heads * tile_rows
        tl.store(table + 4 * max_tasks + at, part_base, mask=live)

    # Merge entries: (item row, KV head, range) ──▶ the row's list, after merge_offset entries.
    num_entries = rows * kv_heads * ranges
    for first in range(part * BLOCK, num_entries, stride):
        e = first + lanes
        live = e < num_entries
        rng = e % ranges
        rest = e // ranges
        kv = rest % kv_heads
        item_row = rest // kv_heads
        tile = item_row // bm
        tile_rows = tl.minimum(bm, rows - tile * bm)
        slot = (
            part_start
            + tile * ranges * kv_heads * bm
            + rng * kv_heads * tile_rows
            + kv * tile_rows
            + item_row % bm
        )
        request = tl.load(perm_ptr + req_start + item_row // rows_per_request, mask=live, other=0)
        token = (item_row // GROUP) % query_tokens
        row = (request * query_tokens + token) * q_heads + kv * GROUP + item_row % GROUP
        dest = tl.load(merge_start_ptr + row, mask=live, other=0) + merge_offset + rng
        tl.store(merge_indices_ptr + dest, slot, mask=live)


FILL_PROGRAMS = 8  # programs per item in the fill launch


def _warps(block: int) -> int:
    return 4 if block <= 512 else 8 if block <= 2048 else 16


Launch = Callable[[], None]


def planning_launches(
    plan: CascadePlan,
    *,
    index: torch.Tensor,
    starts: torch.Tensor | None,
    seq_lens: torch.Tensor,
    stride_r: int,
) -> list[tuple[str, Launch]]:
    """The three planning launches, by name, in order; see the module docstring.

    They read the batch size from plan.planned_requests, which the caller sets
    first. starts is None for a block table (row r begins at r · stride_r) and
    holds each request's first slot for a slot list.
    """
    per_token = starts is not None
    starts_arg = starts if starts is not None else seq_lens  # unused for a block table
    block_r = max(16, triton.next_power_of_2(plan.max_requests))
    interpreting = plan.device.type != "cuda"
    max_items = plan.max_groups + plan.max_requests
    rows_args = (index, starts_arg, seq_lens, plan.perm)

    def tree() -> None:
        _tree_kernel[(1,)](
            *rows_args,
            plan.tree,
            plan.counts,
            plan.planned_requests,
            stride_r,
            plan.block_size,
            plan.query_tokens,
            plan.max_requests,
            MAX_LEVELS=plan.max_levels,
            PER_TOKEN=per_token,
            BLOCK_R=block_r,
            num_warps=_warps(block_r),
        )

    def items() -> None:
        _items_kernel[(1,)](
            *rows_args,
            plan.tree,
            plan.item_req_start,
            plan.item_num_reqs,
            plan.item_block_start,
            plan.item_key_limit,
            plan.item_info,
            plan.counts,
            plan.merge_start,
            plan.merge_end,
            plan.planned_requests,
            stride_r,
            plan.block_size,
            plan.query_tokens,
            plan.min_shared_blocks,
            plan.max_groups,
            plan.max_requests,
            max_items,
            plan.max_tasks,
            plan.max_parts,
            plan.target_programs(plan.prefix_block_m),
            plan.target_programs(plan.suffix_block_m),
            plan.min_split_tokens,
            plan.min_split_floor,
            plan.split_tokens_per_row,
            plan.block_n,
            plan.num_kv_heads,
            plan.num_q_heads,
            plan.small_item_rows,
            plan.part_budget_rows,
            GROUP=plan.group,
            MAX_LEVELS=plan.max_levels,
            PREFIX_BM=plan.prefix_block_m,
            SUFFIX_BM=plan.suffix_block_m,
            MAX_SPLITS=plan.max_splits,
            PER_TOKEN=per_token,
            INVARIANT=plan.batch_invariant,
            BLOCK_R=block_r,
            num_warps=_warps(block_r),
        )

    def fill() -> None:
        _fill_kernel[(max_items, FILL_PROGRAMS)](
            plan.perm,
            plan.item_req_start,
            plan.item_num_reqs,
            plan.item_info,
            plan.counts,
            plan.tasks,
            plan.merge_start,
            plan.merge_indices,
            max_items,
            plan.max_tasks,
            plan.num_kv_heads,
            plan.num_q_heads,
            plan.query_tokens,
            plan.small_item_rows,
            GROUP=plan.group,
            PREFIX_BM=plan.prefix_block_m,
            SUFFIX_BM=plan.suffix_block_m,
            BLOCK=64 if interpreting else 256,
        )

    return [("tree", tree), ("items", items), ("fill", fill)]


MAX_PLAN_GRAPHS = 8  # input sets one plan keeps a graph for
_MAX_SEEN = 64  # input sets remembered while waiting for a second call


class PlanGraphs:
    """The planning launches recorded in CUDA graphs, one graph per set of inputs.

    Launching three Triton kernels from Python costs about 30 µs of CPU each; a
    graph replay costs one launch. A graph holds the addresses of the index and
    length tensors it was recorded with, so it serves only calls that pass the
    same tensors. An engine does: it keeps its block table and lengths in
    persistent buffers and passes a slice of the first rows each step. The batch
    size, which changes every step, is read from the device.

        call 1 with inputs X   launch the three kernels (and compile them)
        call 2 with inputs X   record the three launches in a graph, replay it
        call 3+ with inputs X  replay

    Waiting for the second call keeps a one-off call from paying for a
    recording, which synchronizes the device. Past MAX_PLAN_GRAPHS input sets,
    calls launch the kernels directly. Inside another stream's capture the
    kernels are launched directly too, and the outer graph records them.
    """

    def __init__(self) -> None:
        self.graphs: dict[tuple[object, ...], torch.cuda.CUDAGraph] = {}
        self.seen: set[tuple[object, ...]] = set()

    def run(self, key: tuple[object, ...], launches: list[Launch]) -> None:
        graph = self.graphs.get(key)
        if graph is None and key in self.seen and len(self.graphs) < MAX_PLAN_GRAPHS:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                for launch in launches:
                    launch()
            self.graphs[key] = graph
        if graph is not None:
            graph.replay()
            return
        if len(self.seen) < _MAX_SEEN:
            self.seen.add(key)
        for launch in launches:
            launch()


def plan_on_device(
    plan: CascadePlan,
    *,
    index: torch.Tensor,
    starts: torch.Tensor | None,
    seq_lens: torch.Tensor,
    stride_r: int,
) -> None:
    """Fill the plan's buffers from an index on the device; see the module docstring.

    starts is None for a block table (row r begins at r · stride_r) and holds
    each request's first slot for a slot list. On a GPU the launches replay
    from a graph when the same tensors were passed before (PlanGraphs).
    """
    plan.planned_requests.fill_(seq_lens.shape[0])
    launches = [
        launch
        for _, launch in planning_launches(
            plan, index=index, starts=starts, seq_lens=seq_lens, stride_r=stride_r
        )
    ]
    if plan.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        for launch in launches:
            launch()
        return
    key = (
        index.data_ptr(),
        index.dtype,
        starts.data_ptr() if starts is not None else 0,
        seq_lens.data_ptr(),
        stride_r,
    )
    plan.graphs.run(key, launches)
