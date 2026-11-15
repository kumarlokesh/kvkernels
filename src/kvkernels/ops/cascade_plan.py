"""Planning the cascade on the GPU from the engine's own index.

Requests that share a prefix share the leading entries of their index rows:
prefix caching hands two requests the same physical KV block exactly where
their tokens, and every token before them, agree. Planning therefore needs no
tokens and no prefix tree. Four launches turn the index into the plan's
buffers, with no value copied to the host:

    index rows, lengths (device)
         │
         ▼ order   sort requests by (first KV block, request); one program    ──▶ perm
         ▼ common  KV blocks each sorted row shares with the previous row,    ──▶ common
         │         counting blocks both have filled; one program per row
         ▼ items   runs of rows with the same first KV block; groups; one     ──▶ items, counts,
         │         suffix item per request; split-K ranges; offsets;              merge rows
         │         one program
         ▼ fill    each item's tasks and merge entries; one program per item  ──▶ tasks, merge list

Two index forms are read in place, never copied:

    block table   one row of KV block ids per request (vLLM); entry j of request r
                  holds tokens j · block_size … j · block_size + block_size - 1
    slot list     one slot per token, rows back to back (SGLang); slot s is
                  token s % page_size of page s // page_size

For a slot list, entries are compared once per page: the engine allocates
pages whole and places token t at offset t % page_size of its page, so two
requests with the same first slot in a page share the whole page.

A group shares a whole number of KV blocks, so every member's suffix starts at
the beginning of a KV block. Sizes are fixed by the plan's maxima and the batch
size is read from the device, so the four launches capture in a CUDA graph, and
the plan records its own graph to replay on later steps (PlanGraphs). Nothing
raises on the device: when the batch's split-K partial results or tasks would
exceed the plan's capacity, every item falls back to one range, and runs past
max_groups stay ungrouped.
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
_NO_KEY = tl.constexpr(0x7FFFFFFF)  # sort key of a request with no full KV block: never grouped


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
def _order_kernel(  # type: ignore[no-untyped-def]
    index_ptr,
    starts_ptr,
    seq_lens_ptr,
    perm_ptr,
    num_requests_ptr,
    stride_r,
    block_size,
    PER_TOKEN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    num_requests = tl.load(num_requests_ptr)
    i = tl.arange(0, BLOCK_R)
    ok = i < num_requests
    length = tl.load(seq_lens_ptr + i, mask=ok, other=0)
    has_full = ok & (length >= block_size)
    base = _row_base(index_ptr, starts_ptr, tl.where(ok, i, 0), stride_r, PER_TOKEN)
    first = tl.load(index_ptr + base, mask=has_full, other=0).to(tl.int64)
    key = tl.where(has_full, first, _NO_KEY)
    # Key and request in one integer: the sort is exact and keeps request order on ties.
    packed = tl.sort((key << 32) | i.to(tl.int64))
    tl.store(perm_ptr + i, (packed & 0xFFFFFFFF).to(tl.int32), mask=ok)


@triton.jit
def _common_kernel(  # type: ignore[no-untyped-def]
    index_ptr,
    starts_ptr,
    seq_lens_ptr,
    perm_ptr,
    common_ptr,
    num_requests_ptr,
    stride_r,
    block_size,
    PER_TOKEN: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Program i: KV blocks that sorted row i shares with sorted row i - 1 from the start."""
    i = tl.program_id(0)
    if i >= tl.load(num_requests_ptr):
        return
    if i == 0:
        tl.store(common_ptr, 0)
        return
    a = tl.load(perm_ptr + i - 1)
    b = tl.load(perm_ptr + i)
    full = tl.minimum(tl.load(seq_lens_ptr + a), tl.load(seq_lens_ptr + b)) // block_size
    base_a = _row_base(index_ptr, starts_ptr, a, stride_r, PER_TOKEN)
    base_b = _row_base(index_ptr, starts_ptr, b, stride_r, PER_TOKEN)
    step = block_size if PER_TOKEN else 1  # a slot list is compared at each page's first slot
    shared = full
    cols = tl.arange(0, BLOCK_W)
    start = full * 0  # a tensor, so the while loop can update it
    while start < shared:
        j = start + cols
        in_range = j < full
        x = tl.load(index_ptr + base_a + j.to(tl.int64) * step, mask=in_range, other=0)
        y = tl.load(index_ptr + base_b + j.to(tl.int64) * step, mask=in_range, other=0)
        shared = tl.minimum(shared, tl.min(tl.where(in_range & (x != y), j, full), axis=0))
        start += BLOCK_W
    tl.store(common_ptr + i, shared)


@triton.jit
def _ranges(  # type: ignore[no-untyped-def]
    length,
    base_programs,
    allow_split,
    target,
    min_split,
    block_n,
    MAX_SPLITS: tl.constexpr,
    INVARIANT: tl.constexpr,
):
    """Split-K token span and number of ranges for items of the given lengths."""
    by_length = tl.cdiv(length, min_split)
    if INVARIANT:
        splits = by_length
    else:
        by_fill = tl.cdiv(target, tl.maximum(base_programs, 1))
        splits = tl.where(base_programs >= target, 1, tl.minimum(by_length, by_fill))
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
    common_ptr,
    item_req_start_ptr,
    item_num_reqs_ptr,
    item_index_start_ptr,
    item_info_ptr,  # [6, max_items]: length, ranges, span, task start, part start, merge offset
    counts_ptr,  # prefix tasks, suffix tasks, merge rows, groups, items
    merge_start_ptr,
    merge_end_ptr,
    num_requests_ptr,
    stride_r,
    block_size,
    min_shared_blocks,
    max_groups,
    max_items,
    max_tasks,
    max_parts,
    target_programs,
    min_split,
    block_n,
    kv_heads,
    q_heads,
    GROUP: tl.constexpr,
    PREFIX_BM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    PER_TOKEN: tl.constexpr,
    INVARIANT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    num_requests = tl.load(num_requests_ptr)
    i = tl.arange(0, BLOCK_R)
    ok = i < num_requests
    request = tl.load(perm_ptr + i, mask=ok, other=0)
    length = tl.load(seq_lens_ptr + request, mask=ok, other=0)
    common = tl.load(common_ptr + i, mask=ok, other=0)
    common_next = tl.load(common_ptr + i + 1, mask=i + 1 < num_requests, other=0)

    # Runs: maximal stretches of sorted rows with the same first KV block.
    is_start = ok & (common == 0)
    is_end = ok & (common_next == 0)
    start_pos = tl.associative_scan(tl.where(is_start, i, 0), 0, _maximum)
    end_pos = tl.associative_scan(tl.where(is_end, i, BLOCK_R - 1), 0, _minimum, reverse=True)
    big = 0x7FFFFFFF
    _, running = tl.associative_scan(
        (is_start.to(tl.int32), tl.where(is_start, big, common)), 0, _segment_min
    )
    run_shared = tl.gather(running, end_pos, 0)  # smallest common count in the run
    run_size = end_pos - start_pos + 1

    # Groups: runs of two or more requests sharing at least the threshold.
    if INVARIANT:
        qualifies = i < 0
    else:
        qualifies = is_start & (run_size >= 2) & (run_shared >= min_shared_blocks)
    group_index = tl.cumsum(qualifies.to(tl.int32), 0) - 1
    is_group = qualifies & (group_index < max_groups)
    num_groups = tl.sum(is_group.to(tl.int32), 0)
    in_group = ok & (tl.gather(is_group.to(tl.int32), start_pos, 0) != 0)
    shared_blocks = tl.where(in_group, run_shared, 0)
    group_len = tl.where(is_group, run_shared * block_size, 0)  # set at a group's first row
    group_rows = run_size * GROUP
    group_tiles = tl.where(is_group, tl.cdiv(group_rows, PREFIX_BM), 0)
    suffix_len = tl.where(ok, length - shared_blocks * block_size, 0)
    has_suffix = suffix_len > 0
    suffix_index = num_groups + tl.cumsum(has_suffix.to(tl.int32), 0) - 1
    num_items = num_groups + tl.sum(has_suffix.to(tl.int32), 0)

    # Split-K, with a fallback to one range per item when the batch would not fit.
    # Each launch's programs are counted alone: the prefix and suffix launches
    # run one after the other.
    g_base = kv_heads * tl.sum(group_tiles, 0)
    s_base = kv_heads * tl.sum(has_suffix.to(tl.int32), 0)
    _, g_ranges = _ranges(
        group_len, g_base, True, target_programs, min_split, block_n, MAX_SPLITS, INVARIANT
    )
    _, s_ranges = _ranges(
        suffix_len, s_base, True, target_programs, min_split, block_n, MAX_SPLITS, INVARIANT
    )
    g_parts = group_rows * kv_heads * g_ranges
    s_parts = GROUP * kv_heads * s_ranges
    fits = (
        (tl.sum(tl.where(is_group, g_parts, 0), 0) + tl.sum(s_parts, 0) <= max_parts)
        & (tl.sum(group_tiles * g_ranges, 0) <= max_tasks)
        & (tl.sum(s_ranges, 0) <= max_tasks)
    )
    g_span1, g_ranges1 = _ranges(
        group_len, g_base, fits, target_programs, min_split, block_n, MAX_SPLITS, INVARIANT
    )
    s_span1, s_ranges1 = _ranges(
        suffix_len, s_base, fits, target_programs, min_split, block_n, MAX_SPLITS, INVARIANT
    )
    g_parts = tl.where(is_group, group_rows * kv_heads * g_ranges1, 0)
    s_parts = GROUP * kv_heads * s_ranges1
    g_tasks = group_tiles * g_ranges1

    # Offsets: groups first, then suffix items, each in sorted-row order.
    g_task_start = tl.cumsum(g_tasks, 0) - g_tasks
    s_task_start = tl.cumsum(s_ranges1, 0) - s_ranges1
    g_part_start = tl.cumsum(g_parts, 0) - g_parts
    s_part_start = tl.sum(g_parts, 0) + tl.cumsum(s_parts, 0) - s_parts

    base = _row_base(index_ptr, starts_ptr, request, stride_r, PER_TOKEN)
    skip = shared_blocks * block_size if PER_TOKEN else shared_blocks
    g_at = tl.where(is_group, group_index, 0)
    s_at = tl.where(has_suffix, suffix_index, 0)
    member_ranges = tl.where(in_group, tl.gather(g_ranges1, start_pos, 0), 0)
    _store_items(
        item_req_start_ptr,
        item_num_reqs_ptr,
        item_index_start_ptr,
        item_info_ptr,
        max_items,
        g_at,
        is_group,
        i,
        run_size,
        base,
        group_len,
        g_ranges1,
        g_span1,
        g_task_start,
        g_part_start,
        tl.zeros_like(i),
    )
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
        base + skip,
        suffix_len,
        s_ranges1,
        s_span1,
        s_task_start,
        s_part_start,
        member_ranges,
    )

    # Merge lists: row (request, head) holds its group's ranges, then its suffix's.
    per_row = member_ranges + tl.where(has_suffix, s_ranges1, 0)
    row_start = tl.cumsum(per_row * q_heads, 0) - per_row * q_heads
    for h in range(q_heads):
        row = request * q_heads + h
        first = row_start + h * per_row
        tl.store(merge_start_ptr + row, first, mask=ok)
        tl.store(merge_end_ptr + row, first + per_row, mask=ok)

    tl.store(counts_ptr + 0, tl.sum(g_tasks, 0))
    tl.store(counts_ptr + 1, tl.sum(s_ranges1, 0))
    tl.store(counts_ptr + 2, num_requests * q_heads)
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
    GROUP: tl.constexpr,
    PREFIX_BM: tl.constexpr,
    SUFFIX_BM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item = tl.program_id(0)
    if item >= tl.load(counts_ptr + 4):
        return
    is_prefix = item < tl.load(counts_ptr + 3)
    bm = tl.where(is_prefix, PREFIX_BM, SUFFIX_BM)
    table = tasks_ptr + tl.where(is_prefix, 0, 5 * max_tasks)
    req_start = tl.load(item_req_start_ptr + item)
    rows = tl.load(item_num_reqs_ptr + item) * GROUP
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
    for first in range(0, num_tasks, BLOCK):
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
    for first in range(0, num_entries, BLOCK):
        e = first + lanes
        live = e < num_entries
        rng = e % ranges
        rest = e // ranges
        kv = rest % kv_heads
        item_row = rest // kv_heads
        tile = item_row // bm
        tile_rows = tl.minimum(bm, rows - tile * bm)
        part = (
            part_start
            + tile * ranges * kv_heads * bm
            + rng * kv_heads * tile_rows
            + kv * tile_rows
            + item_row % bm
        )
        request = tl.load(perm_ptr + req_start + item_row // GROUP, mask=live, other=0)
        row = request * q_heads + kv * GROUP + item_row % GROUP
        dest = tl.load(merge_start_ptr + row, mask=live, other=0) + merge_offset + rng
        tl.store(merge_indices_ptr + dest, part, mask=live)


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
    """The four planning launches, by name, in order; see the module docstring.

    They read the batch size from plan.planned_requests, which the caller sets
    first. starts is None for a block table (row r begins at r · stride_r) and
    holds each request's first slot for a slot list.
    """
    per_token = starts is not None
    starts_arg = starts if starts is not None else seq_lens  # unused for a block table
    block_r = max(16, triton.next_power_of_2(plan.max_requests))
    interpreting = plan.device.type != "cuda"
    max_items = plan.max_groups + plan.max_requests
    order_args = (index, starts_arg, seq_lens, plan.perm)

    def order() -> None:
        _order_kernel[(1,)](
            *order_args,
            plan.planned_requests,
            stride_r,
            plan.block_size,
            PER_TOKEN=per_token,
            BLOCK_R=block_r,
            num_warps=_warps(block_r),
        )

    def common() -> None:
        _common_kernel[(plan.max_requests,)](
            *order_args,
            plan.common,
            plan.planned_requests,
            stride_r,
            plan.block_size,
            PER_TOKEN=per_token,
            BLOCK_W=16 if interpreting else 64,
        )

    def items() -> None:
        _items_kernel[(1,)](
            *order_args,
            plan.common,
            plan.item_req_start,
            plan.item_num_reqs,
            plan.item_block_start,
            plan.item_info,
            plan.counts,
            plan.merge_start,
            plan.merge_end,
            plan.planned_requests,
            stride_r,
            plan.block_size,
            plan.min_shared_blocks,
            plan.max_groups,
            max_items,
            plan.max_tasks,
            plan.max_parts,
            plan.programs_per_sm * plan.num_sms,
            plan.min_split_tokens,
            plan.block_n,
            plan.num_kv_heads,
            plan.num_q_heads,
            GROUP=plan.group,
            PREFIX_BM=plan.prefix_block_m,
            MAX_SPLITS=plan.max_splits,
            PER_TOKEN=per_token,
            INVARIANT=plan.batch_invariant,
            BLOCK_R=block_r,
            num_warps=_warps(block_r),
        )

    def fill() -> None:
        _fill_kernel[(max_items,)](
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
            GROUP=plan.group,
            PREFIX_BM=plan.prefix_block_m,
            SUFFIX_BM=plan.suffix_block_m,
            BLOCK=64 if interpreting else 256,
        )

    return [("order", order), ("common", common), ("items", items), ("fill", fill)]


MAX_PLAN_GRAPHS = 8  # input sets one plan keeps a graph for
_MAX_SEEN = 64  # input sets remembered while waiting for a second call


class PlanGraphs:
    """The planning launches recorded in CUDA graphs, one graph per set of inputs.

    Launching four Triton kernels from Python costs about 30 µs of CPU each; a
    graph replay costs one launch. A graph holds the addresses of the index and
    length tensors it was recorded with, so it serves only calls that pass the
    same tensors. An engine does: it keeps its block table and lengths in
    persistent buffers and passes a slice of the first rows each step. The batch
    size, which changes every step, is read from the device.

        call 1 with inputs X   launch the four kernels (and compile them)
        call 2 with inputs X   record the four launches in a graph, replay it
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
