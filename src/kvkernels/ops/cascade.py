"""Cascade decode attention: shared KV read once per tree node, suffixes once per request.

Requests that share leading KV blocks form a prefix tree. Each node of the tree
(requests below it, the KV blocks they share beyond its parent) is one work
item, and each request's own remaining tokens are another. The cascade computes
attention over every work item, then merges each request's partial results:

    work items                                 partial results   merge
    rows of r0-r3 · node A blocks, range 1  ──▶ p0
    rows of r0-r3 · node A blocks, range 2  ──▶ p1                r0: p0 p1 p2 p4 ──▶ out[r0]
    rows of r0-r1 · node B blocks (in A)    ──▶ p2
    rows of r0    · r0's suffix blocks      ──▶ p4
    …

A request may have several new query tokens (query_tokens, the same for every
request of a plan): token t sees the request's keys up to length - query_tokens
+ t, so the causal rule applies in its suffix item only. The cache may be fp8
(e4m3) with one K and one V scale per layer.

A plan, computed once per decode step and used by every layer, lists the tasks
(one tile of an item's rows over one token range), assigns each partial result
its slot in the workspace, and lists each (request, query head) row's partials
for the merge. Its buffers are allocated once for a maximum
batch and rewritten in place each step, so the launches can be captured in a
CUDA graph and replayed after later steps. Two ways to fill them:

    plan_from_block_table, plan_from_slots   on the GPU, from the index the engine
                                             already passes to attention; groups are
                                             found from shared leading KV blocks
    update                                   on the host, from groups the caller names

Requests are read through a permutation, so a group's requests need not be
consecutive in the batch; out[r] is always request r.

Three launches: prefix tasks, suffix tasks, merge. The two attention launches
run the same kernel with different tiles:

    launch    rows per tile    warps   stages   programs per SM
    prefix    128, 64 or 32    4       2        2                many requests' rows per item
    suffix     16              4       2        2                one request's GQA group per item

Both launches read keys in tiles of 32 tokens (block_n). A prefix task of at
most 64 or 32 rows runs in a 64- or 32-row tile (prefix_min_block_m, default a
quarter of the prefix tile; see prefix_attention). A group of at most
small_item_rows rows (four requests with a GQA group of 4) runs in the suffix
launch's tiles instead.

With 4 warps each warp holds 32 query rows of a 128-row tile. Compiled for
sm_89 the prefix launch uses 255 registers per thread, 48 bytes of stack and
40 KB of shared memory, so two programs fit on an SM; with 64-token key tiles
it spilled 1 KB per thread. On an L4, against 8 warps, 3 stages and 64-token
key tiles, it took 0.83-0.88x the time at batch 64-128 with prefixes of 8,192
and more, and 0.97x over the probe grid.

Split-K cuts an item's tokens into s ranges, each its own task, so a launch
with few tasks still fills the GPU. Each launch is counted alone, since the
two run one after the other. With base programs before splitting and R
programs that run at once (prefix_programs_per_sm · SMs for the prefix launch,
programs_per_sm · SMs for the suffix launch), a launch takes about
ceil(base · s / R) rounds of ranges 1/s as long, so s is the value from 1 to
its cap that makes ceil(base · s / R) / s smallest, ties to the smaller:

    base 32, R 116     s = 1: 1 round      1/1 = 1.00
                       s = 3: 1 round      1/3 = 0.33
                       s = 7: 2 rounds     2/7 = 0.29   ◀ chosen

A launch with base >= R fills a round already and is not split: more ranges
would add partial results for the merge to read. The cap is MAX_SPLITS and
ceil(length / shortest range), with the shortest range
clamp(SPLIT_TOKENS_PER_ROW · tile rows, MIN_SPLIT_FLOOR, min_split_tokens).
On an L4 over 144 cases (batch 1-128, prefix 512-32,768, both models) the
round rule took 0.935x the time of the earlier one (ceil(2 SMs / base) ranges
of at least 512 tokens); the guard took a further 0.99x over 72 cases with
suffixes of 128-2,048 tokens.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from kvkernels.ops._registry import registered_op
from kvkernels.ops.cascade_plan import MAX_PLAN_REQUESTS, PlanGraphs, plan_on_device
from kvkernels.ops.merge_states import merge_ragged
from kvkernels.ops.prefix_attention import attend

MAX_SPLITS = 16
MAX_QUERY_TOKENS = 16  # more new tokens per request is prefill
MAX_LEVELS = 8  # rounds of tree finding a plan may ask for
DEFAULT_LEVELS = 4
# Split-K ranges are at least SPLIT_TOKENS_PER_ROW tokens per query row of the
# item's tile, and at least MIN_SPLIT_FLOOR tokens, up to min_split_tokens.
SPLIT_TOKENS_PER_ROW = 2
MIN_SPLIT_FLOOR = 128
# Shortest shared prefix that forms a group. On an L4 (48 MB of L2), grouping
# requests that share 16-256 tokens took 0.98-1.07x the time of reading the
# shared tokens with each request; from 512 tokens it took 0.61-1.03x.
MIN_SHARED_TOKENS = 512
TASK_WAVES = 8  # attention programs per KV head: TASK_WAVES · programs_per_sm · SMs / KV heads
MERGE_PROGRAMS_PER_SM = 32  # merge programs are small; many fit on one SM
# Warps and pipeline stages of each attention launch; see the module docstring.
PREFIX_WARPS, PREFIX_STAGES = 4, 2
SUFFIX_WARPS, SUFFIX_STAGES = 4, 2
HEAD_DIMS = (64, 128)
_IN_DTYPES = (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class PrefixGroup:
    """Requests that share tokens start … start + length - 1, held in `blocks`.

    requests are consecutive request indices. Groups may nest: a group inside
    another covers a subset of its requests and starts where it ends. start is
    a multiple of the block size; length may end inside the last block.
    """

    requests: range
    blocks: Sequence[int]
    length: int
    start: int = 0


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def same_device(a: torch.device, b: torch.device) -> bool:
    """Whether two devices are the same; a CUDA device without an index is the current one."""
    if a.type != b.type:
        return False
    if a.type != "cuda" or a.index == b.index:
        return True
    current = torch.cuda.current_device()
    return (a.index if a.index is not None else current) == (
        b.index if b.index is not None else current
    )


class CascadePlan:
    """Work items, split-K ranges and merge lists for one decode step, in fixed device buffers."""

    def __init__(
        self,
        *,
        max_requests: int,
        max_groups: int,
        max_blocks: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        device: torch.device,
        max_tasks: int | None = None,
        max_parts: int | None = None,
        num_sms: int | None = None,
        programs_per_sm: int = 2,
        prefix_programs_per_sm: int = 2,
        min_split_tokens: int = 512,
        prefix_block_m: int = 128,
        prefix_min_block_m: int | None = None,
        suffix_block_m: int = 16,
        block_n: int | None = None,
        min_shared_tokens: int | None = None,
        small_item_rows: int = 16,
        batch_invariant: bool = False,
        query_tokens: int = 1,
        max_levels: int = DEFAULT_LEVELS,
    ) -> None:
        if num_q_heads % num_kv_heads:
            raise ValueError(f"num_q_heads {num_q_heads} is not a multiple of {num_kv_heads}")
        self.group = num_q_heads // num_kv_heads
        if not 1 <= self.group <= suffix_block_m:
            raise ValueError(f"GQA group {self.group} must be 1 to {suffix_block_m}")
        if not 1 <= query_tokens <= MAX_QUERY_TOKENS:
            raise ValueError(f"query_tokens {query_tokens} must be 1 to {MAX_QUERY_TOKENS}")
        if not 1 <= max_levels <= MAX_LEVELS:
            raise ValueError(f"max_levels {max_levels} must be 1 to {MAX_LEVELS}")
        if head_dim not in HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
        if prefix_block_m == suffix_block_m:
            raise ValueError("prefix and suffix row tiles must differ in height")
        prefix_min_block_m = prefix_min_block_m or max(16, prefix_block_m // 4)
        if prefix_min_block_m not in (prefix_block_m, prefix_block_m // 2, prefix_block_m // 4):
            raise ValueError(
                f"prefix_min_block_m {prefix_min_block_m}: must be prefix_block_m, /2 or /4"
            )
        if prefix_min_block_m < 16:
            raise ValueError(f"prefix_min_block_m {prefix_min_block_m} is below 16")
        self.max_requests = max_requests
        self.max_groups = max_groups
        self.max_blocks = max_blocks
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device
        self.max_splits = MAX_SPLITS
        self.min_shared_tokens = (
            MIN_SHARED_TOKENS if min_shared_tokens is None else min_shared_tokens
        )
        self.small_item_rows = small_item_rows
        self.query_tokens = query_tokens
        self.max_levels = max_levels
        # A node's own tokens must pay for the partial results of its rows, which
        # grow with the query tokens per request.
        self.min_shared_blocks = max(
            1, math.ceil(self.min_shared_tokens * query_tokens / block_size)
        )
        self.batch_invariant = batch_invariant
        rows_per_request = self.group * query_tokens
        suffix_tiles = math.ceil(rows_per_request / suffix_block_m)
        self.max_tasks = max_tasks or (64 * max_groups + MAX_SPLITS * max_requests * suffix_tiles)
        self.max_parts = max_parts or (2 * max_requests * query_tokens * num_q_heads * MAX_SPLITS)
        # Without split-K every plan fits: one part per row for each node item on
        # its path and one for its suffix. A batch-invariant plan must fit with
        # split-K, which it never turns off.
        splits = MAX_SPLITS if batch_invariant else 1
        levels = 0 if batch_invariant else max_levels
        need_parts = (levels + 1) * max_requests * query_tokens * num_q_heads * splits
        small_groups = 0 if batch_invariant else min(max_groups, max_requests // 2)
        need_tasks = max(
            max_requests * suffix_tiles * splits
            + small_groups * math.ceil(small_item_rows / suffix_block_m),
            0
            if batch_invariant
            else max_groups + math.ceil(levels * max_requests * rows_per_request / prefix_block_m),
        )
        if self.max_parts < need_parts:
            raise ValueError(f"max_parts {self.max_parts} is below {need_parts}")
        if self.max_tasks < need_tasks:
            raise ValueError(f"max_tasks {self.max_tasks} is below {need_tasks}")
        if num_sms is None:
            num_sms = (
                torch.cuda.get_device_properties(device).multi_processor_count
                if device.type == "cuda"
                else 8
            )
        self.num_sms = num_sms
        self.programs_per_sm = programs_per_sm
        self.prefix_programs_per_sm = prefix_programs_per_sm
        self.min_split_tokens = min_split_tokens
        self.split_tokens_per_row = SPLIT_TOKENS_PER_ROW
        self.min_split_floor = MIN_SPLIT_FLOOR
        self.prefix_block_m = prefix_block_m
        self.prefix_min_block_m = prefix_min_block_m
        self.suffix_block_m = suffix_block_m
        self.block_n = block_n or (16 if _interpreting() else 32)
        self.max_rows = max_requests * query_tokens * num_q_heads
        # Fixed grids that loop over the batch's work, a few times larger than
        # what the GPU runs at once so that uneven tasks still spread out.
        resident = programs_per_sm * num_sms
        self.task_programs = min(self.max_tasks, math.ceil(TASK_WAVES * resident / num_kv_heads))
        self.merge_programs = min(self.max_rows, MERGE_PROGRAMS_PER_SM * num_sms)

        def ints(n: int) -> torch.Tensor:
            return torch.zeros(max(n, 1), dtype=torch.int32, device=device)

        items = max_groups + max_requests
        self.blocks = ints(max_blocks)
        self.perm = ints(max_requests)  # row i of the sorted batch is request perm[i]
        # Device planning, per round and sorted row: the first row of the row's
        # node (-1: none) and the node's depth in KV blocks.
        self.tree = torch.zeros(
            (2, max_levels, max(max_requests, 1)), dtype=torch.int32, device=device
        )
        self.item_req_start, self.item_num_reqs, self.item_block_start = (
            ints(items),
            ints(items),
            ints(items),
        )
        # A suffix item's token t sees its tokens up to key_limit + t; node items see all.
        self.item_key_limit = ints(items)
        # Device planning: length, ranges, span, task start, part start, merge offset per item.
        self.item_info = torch.zeros((6, items), dtype=torch.int32, device=device)
        # Item, row start, token start, token end, part base; prefix tasks, then suffix tasks.
        self.tasks = torch.zeros((2, 5, self.max_tasks), dtype=torch.int32, device=device)
        self.prefix_tasks = tuple(self.tasks[0])
        self.suffix_tasks = tuple(self.tasks[1])
        self.counts = ints(6)  # prefix tasks, suffix tasks, merge rows, node items, items, rounds
        self.planned_requests = ints(1)  # device planning: the batch size, read by the kernels
        self.graphs = PlanGraphs()  # device planning: recorded launches, replayed each step
        self.merge_start = ints(self.max_rows)
        self.merge_end = ints(self.max_rows)
        self.merge_indices = ints(self.max_parts)
        self.part_o = torch.zeros((self.max_parts, head_dim), dtype=torch.float32, device=device)
        self.part_lse = torch.zeros(self.max_parts, dtype=torch.float32, device=device)
        self.num_requests = 0
        # Whether the prefix launch can have tasks. An empty launch still costs
        # about 2 us on an L4 (batch 1, Llama 3.1 8B: 19.7 us with it, per call
        # replayed from a CUDA graph), so it is left out when no node can exist.
        self.prefix_launch = True
        self.index = self.blocks  # what item_block_start points into
        self.per_token = False  # index holds one slot per token (a slot list)

    # ----------------------------------------------------------------------------------

    def _check(
        self,
        groups: Sequence[PrefixGroup],
        suffix_blocks: Sequence[Sequence[int]],
        suffix_lens: Sequence[int],
    ) -> list[int]:
        """Check the groups form a tree; return their order, parents before children."""
        num_req = len(suffix_lens)
        if num_req > self.max_requests:
            raise ValueError(f"{num_req} requests exceed max_requests {self.max_requests}")
        if len(suffix_blocks) != num_req:
            raise ValueError(f"{len(suffix_blocks)} suffix block lists for {num_req} requests")
        if len(groups) > self.max_groups:
            raise ValueError(f"{len(groups)} groups exceed max_groups {self.max_groups}")
        for gi, g in enumerate(groups):
            if g.requests.step != 1 or g.requests.start < 0 or g.requests.stop > num_req:
                raise ValueError(f"group {gi}: requests {g.requests} outside 0 … {num_req - 1}")
            if len(g.requests) == 0:
                raise ValueError(f"group {gi} has no requests")
            if g.start < 0 or g.start % self.block_size:
                raise ValueError(f"group {gi}: start {g.start} is not a multiple of the block size")
        # Each request's groups, outermost first, must hold tokens 0 … end with no gap.
        path: list[list[int]] = [[] for _ in range(num_req)]
        for gi in sorted(range(len(groups)), key=lambda gi: groups[gi].start):
            for r in groups[gi].requests:
                end = groups[path[r][-1]].start + groups[path[r][-1]].length if path[r] else 0
                if groups[gi].start < end:
                    raise ValueError(
                        f"groups {path[r][-1]} and {gi} overlap at request {r}, token "
                        f"{groups[gi].start}"
                    )
                if groups[gi].start > end:
                    raise ValueError(
                        f"group {gi} starts at token {groups[gi].start}, but request {r}'s "
                        f"groups end at {end}: a gap"
                    )
                if path[r]:
                    outer = groups[path[r][-1]].requests
                    if not (
                        outer.start <= groups[gi].requests.start
                        and groups[gi].requests.stop <= outer.stop
                    ):
                        raise ValueError(f"group {gi} is not inside group {path[r][-1]}")
                path[r].append(gi)
        depth = [0] * len(groups)
        for p in path:
            for d, gi in enumerate(p):
                depth[gi] = d
        for r, p in enumerate(path):
            if p and self.query_tokens > 1 and suffix_lens[r] < self.query_tokens - 1:
                raise ValueError(
                    f"request {r}: groups hold keys after its first new query token "
                    f"(suffix {suffix_lens[r]} tokens, query_tokens {self.query_tokens})"
                )
        lists = [(f"group {gi}", g.blocks, g.length) for gi, g in enumerate(groups)]
        lists += [
            (f"request {r}", b, n)
            for r, (b, n) in enumerate(zip(suffix_blocks, suffix_lens, strict=True))
        ]
        for name, blocks, length in lists:
            if length < 0 or len(blocks) * self.block_size < length:
                raise ValueError(f"{name}: {length} tokens do not fit in {len(blocks)} blocks")
            for b in blocks:
                if not 0 <= b < self.num_blocks:
                    raise ValueError(f"{name}: block {b} is outside 0 … {self.num_blocks - 1}")
        return sorted(range(len(groups)), key=lambda gi: (depth[gi], groups[gi].requests.start))

    def min_range_tokens(self, rows: int) -> int:
        """Shortest split-K range for an item whose tile holds rows query rows."""
        by_rows = max(self.min_split_floor, self.split_tokens_per_row * rows)
        return min(self.min_split_tokens, by_rows)

    def target_programs(self, block_m: int) -> int:
        """Programs of one launch that run at once: what split-K aims to fill."""
        per_sm = (
            self.prefix_programs_per_sm if block_m == self.prefix_block_m else self.programs_per_sm
        )
        return per_sm * self.num_sms

    def _splits(self, length: int, base_programs: int, rows: int, block_m: int) -> int:
        if self.batch_invariant:
            return max(1, min(MAX_SPLITS, math.ceil(length / self.min_split_tokens)))
        resident = self.target_programs(block_m)
        if base_programs >= resident:
            return 1  # the launch fills a round already; more ranges add merge work
        # The ranges whose programs fill the launch's rounds best: time is about
        # (rounds of programs) x (tokens per range) = ceil(base · s / R) / s.
        cap = max(1, min(MAX_SPLITS, math.ceil(length / self.min_range_tokens(min(rows, block_m)))))
        base = max(base_programs, 1)
        best_s, best_rounds = 1, math.ceil(base / resident)
        for s in range(2, cap + 1):
            rounds = math.ceil(base * s / resident)
            if rounds * best_s < best_rounds * s:
                best_s, best_rounds = s, rounds
        return best_s

    def update(
        self,
        *,
        groups: Sequence[PrefixGroup],
        suffix_blocks: Sequence[Sequence[int]],
        suffix_lens: Sequence[int],
    ) -> None:
        """Plan one decode step on the host. Requests are numbered 0 … len(suffix_lens) - 1."""
        if self.batch_invariant and groups:
            raise ValueError("a batch-invariant plan makes no groups")
        order = self._check(groups, suffix_blocks, suffix_lens)
        num_req = len(suffix_lens)
        g, hkv, hq, nt = self.group, self.num_kv_heads, self.num_q_heads, self.query_tokens
        per_req = g * nt  # item rows per request
        no_limit = 1 << 30  # node items: every query token sees all keys

        # Work items, parents before children, then suffixes: (first request,
        # requests, block list, length, row-tile height, key limit).
        items: list[tuple[int, int, Sequence[int], int, int, int]] = []
        for grp in (groups[gi] for gi in order):
            if grp.length > 0:
                items.append(
                    (
                        grp.requests.start,
                        len(grp.requests),
                        grp.blocks,
                        grp.length,
                        self.suffix_block_m
                        if len(grp.requests) * per_req <= self.small_item_rows
                        else self.prefix_block_m,
                        no_limit,
                    )
                )
        for r, (blocks, length) in enumerate(zip(suffix_blocks, suffix_lens, strict=True)):
            if length > 0:
                items.append((r, 1, blocks, length, self.suffix_block_m, length - nt))
        # Programs of each launch before splitting; the launches run one after the other.
        base_programs = {self.prefix_block_m: 0, self.suffix_block_m: 0}
        for _, n, _, _, bm, _ in items:
            base_programs[bm] += math.ceil(n * per_req / bm) * hkv

        flat: list[int] = []
        req_start, num_reqs, block_start = [], [], []
        tasks: dict[int, list[list[int]]] = {
            self.prefix_block_m: [[], [], [], [], []],
            self.suffix_block_m: [[], [], [], [], []],
        }
        key_limit: list[int] = []
        parts_of_row: list[list[int]] = [[] for _ in range(num_req * nt * hq)]
        next_part = 0
        for index, (first_req, n_req, blocks, length, bm, limit) in enumerate(items):
            req_start.append(first_req)
            num_reqs.append(n_req)
            block_start.append(len(flat))
            key_limit.append(limit)
            flat.extend(blocks[: math.ceil(length / self.block_size)])
            splits = self._splits(length, base_programs[bm], n_req * per_req, bm)
            span = math.ceil(math.ceil(length / splits) / self.block_n) * self.block_n
            ranges = [(s, min(s + span, length)) for s in range(0, length, span)]
            rows = n_req * per_req
            for row_start in range(0, rows, bm):
                tile_rows = min(bm, rows - row_start)
                for tok_start, tok_end in ranges:
                    table = tasks[bm]
                    for column, value in zip(
                        table, (index, row_start, tok_start, tok_end, next_part), strict=True
                    ):
                        column.append(value)
                    for kv in range(hkv):
                        for i in range(tile_rows):
                            item_row = row_start + i
                            request = first_req + item_row // per_req
                            token = (item_row // g) % nt
                            head = kv * g + item_row % g
                            parts_of_row[(request * nt + token) * hq + head].append(
                                next_part + kv * tile_rows + i
                            )
                    next_part += hkv * tile_rows

        prefix_tasks, suffix_tasks = tasks[self.prefix_block_m], tasks[self.suffix_block_m]
        if len(flat) > self.max_blocks:
            raise ValueError(f"{len(flat)} blocks in the batch exceed max_blocks {self.max_blocks}")
        if max(len(prefix_tasks[0]), len(suffix_tasks[0])) > self.max_tasks:
            raise ValueError(f"the batch needs more than max_tasks {self.max_tasks} tasks")
        if next_part > self.max_parts:
            raise ValueError(
                f"the batch needs {next_part} partial results, above max_parts {self.max_parts}"
            )

        starts: list[int] = []
        indices: list[int] = []
        for parts in parts_of_row:
            starts.append(len(indices))
            indices.extend(parts)

        def put(buffer: torch.Tensor, values: Sequence[int]) -> None:
            if values:
                buffer[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))

        put(self.blocks, flat)
        put(self.perm, range(num_req))
        put(self.item_req_start, req_start)
        put(self.item_num_reqs, num_reqs)
        put(self.item_block_start, block_start)
        put(self.item_key_limit, key_limit)
        for buffers, columns in (
            (self.prefix_tasks, prefix_tasks),
            (self.suffix_tasks, suffix_tasks),
        ):
            for buffer, column in zip(buffers, columns, strict=True):
                put(buffer, column)
        put(self.merge_start, starts)
        put(self.merge_end, [*starts[1:], len(indices)])
        put(self.merge_indices, indices)
        put(self.counts, [len(prefix_tasks[0]), len(suffix_tasks[0]), num_req * nt * hq])
        self.num_requests = num_req
        # As for device planning: by the batch size alone, so a CUDA graph captured
        # before a later update still holds the prefix launch the update needs.
        self.prefix_launch = num_req >= 2 and not self.batch_invariant
        self.index, self.per_token = self.blocks, False

    def plan_from_block_table(self, block_table: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """Plan one decode step on the GPU from a vLLM-style index, without a host sync.

        block_table: [requests, width] int32, row r holding request r's KV block ids
        (entries past its length are never read). seq_lens: [requests] int32; a
        request of length 0 is a padded row and gets out = 0. Requests whose rows
        start with the same KV blocks form a group when they share at least
        min_shared_tokens; the group covers the whole KV blocks every member
        shares. The attention reads block_table in place until the next plan, so
        keep it unchanged until then.
        """
        if block_table.dim() != 2 or block_table.dtype != torch.int32:
            raise ValueError(f"block_table must be 2-D int32, got {block_table.dtype}")
        if block_table.shape[1] > 0 and block_table.stride(1) != 1:
            raise ValueError("block_table rows must be contiguous")
        self._plan_on_device(block_table, None, seq_lens, block_table.stride(0))

    def plan_from_slots(
        self, slot_indptr: torch.Tensor, slots: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        """Plan one decode step on the GPU from an SGLang-style index, without a host sync.

        slots: one slot per token, int32 or int64, request r's tokens at
        slots[slot_indptr[r] : slot_indptr[r] + seq_lens[r]]; slot s is token
        s % block_size of page s // block_size. slot_indptr: [requests + 1] or
        [requests] int32. Groups are found as in plan_from_block_table, comparing
        each page's first slot; the attention reads slots in place.
        """
        if slots.dim() != 1 or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"slots must be 1-D int32 or int64, got {slots.dtype}")
        if slot_indptr.dim() != 1 or slot_indptr.dtype != torch.int32:
            raise ValueError(f"slot_indptr must be 1-D int32, got {slot_indptr.dtype}")
        if slot_indptr.shape[0] not in (seq_lens.shape[0], seq_lens.shape[0] + 1):
            raise ValueError(
                f"slot_indptr has {slot_indptr.shape[0]} entries for {seq_lens.shape[0]} requests"
            )
        self._plan_on_device(slots, slot_indptr, seq_lens, 0)

    def _plan_on_device(
        self,
        index: torch.Tensor,
        starts: torch.Tensor | None,
        seq_lens: torch.Tensor,
        stride_r: int,
    ) -> None:
        if seq_lens.dim() != 1 or seq_lens.dtype != torch.int32:
            raise ValueError(f"seq_lens must be 1-D int32, got {seq_lens.dtype}")
        num_req = seq_lens.shape[0]
        if num_req > self.max_requests:
            raise ValueError(f"{num_req} requests exceed max_requests {self.max_requests}")
        if starts is None and index.shape[0] != num_req:
            raise ValueError(f"block_table has {index.shape[0]} rows for {num_req} requests")
        if self.max_requests > MAX_PLAN_REQUESTS:
            raise ValueError(f"planning on the GPU supports up to {MAX_PLAN_REQUESTS} requests")
        for t in (index, seq_lens) + ((starts,) if starts is not None else ()):
            if not same_device(t.device, self.device):
                raise ValueError(f"index tensors must be on {self.device}, got {t.device}")
        plan_on_device(self, index=index, starts=starts, seq_lens=seq_lens, stride_r=stride_r)
        self.num_requests = num_req
        # A node needs two requests. The decision rests on the batch size alone,
        # which a captured CUDA graph fixes.
        self.prefix_launch = num_req >= 2 and not self.batch_invariant
        self.index, self.per_token = index, starts is not None

    # ----------------------------------------------------------------------------------

    def tensors(self) -> list[torch.Tensor]:
        """Read-only plan tensors, in the order the registered op expects."""
        return [
            self.index,
            self.perm,
            self.item_req_start,
            self.item_num_reqs,
            self.item_block_start,
            self.item_key_limit,
            *self.prefix_tasks,
            *self.suffix_tasks,
            self.counts,
            self.merge_start,
            self.merge_end,
            self.merge_indices,
        ]

    def settings(self) -> list[int]:
        return [
            self.block_size,
            self.group,
            self.max_tasks,
            self.max_rows,
            self.prefix_block_m,
            self.suffix_block_m,
            self.block_n,
            int(self.per_token),
            self.task_programs,
            self.merge_programs,
            self.prefix_min_block_m,
            self.query_tokens,
            int(self.prefix_launch),
        ]


@registered_op("kvkernels::cascade_decode_attention", mutates_args=("workspace", "out", "lse_out"))
def _cascade_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_tensors: list[torch.Tensor],
    workspace: list[torch.Tensor],
    out: torch.Tensor,
    lse_out: torch.Tensor | None,
    sm_scale: float,
    settings: list[int],
    kv_scales: list[torch.Tensor],
) -> None:
    index, perm, req_start, num_reqs, block_start, key_limit = plan_tensors[:6]
    prefix_tasks = tuple(plan_tensors[6:11])
    suffix_tasks = tuple(plan_tensors[11:16])
    counts, merge_start, merge_end, indices = plan_tensors[16:20]
    part_o, part_lse = workspace
    block_size, group, max_tasks, max_rows, prefix_bm, suffix_bm, block_n, per_token = settings[:8]
    task_programs, merge_programs, prefix_min_bm, query_tokens, prefix_launch = settings[8:13]
    common = {
        "index": index,
        "perm": perm,
        "per_token": bool(per_token),
        "item_req_start": req_start,
        "item_num_reqs": num_reqs,
        "item_block_start": block_start,
        "item_key_limit": key_limit,
        "query_tokens": query_tokens,
        "part_o": part_o,
        "part_lse": part_lse,
        "sm_scale": sm_scale,
        "block_size": block_size,
        "group": group,
        "max_tasks": max_tasks,
        "block_n": block_n,
        "num_programs": task_programs,
        "k_scale": kv_scales[0] if kv_scales else None,
        "v_scale": kv_scales[1] if kv_scales else None,
    }
    if prefix_launch:
        attend(
            q,
            k,
            v,
            tasks=prefix_tasks,  # type: ignore[arg-type]
            num_tasks=counts[0:1],
            block_m=prefix_bm,
            min_block_m=prefix_min_bm,
            num_warps=PREFIX_WARPS,
            num_stages=PREFIX_STAGES,
            **common,  # type: ignore[arg-type]
        )
    attend(
        q,
        k,
        v,
        tasks=suffix_tasks,  # type: ignore[arg-type]
        num_tasks=counts[1:2],
        block_m=suffix_bm,
        num_warps=SUFFIX_WARPS,
        num_stages=SUFFIX_STAGES,
        **common,  # type: ignore[arg-type]
    )
    merge_ragged(
        part_o,
        part_lse,
        merge_start,
        merge_end,
        indices,
        counts[2:3],
        out,
        lse_out,
        max_rows=max_rows,
        num_programs=merge_programs,
    )


@_cascade_op.register_fake
def _cascade_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_tensors: list[torch.Tensor],
    workspace: list[torch.Tensor],
    out: torch.Tensor,
    lse_out: torch.Tensor | None,
    sm_scale: float,
    settings: list[int],
    kv_scales: list[torch.Tensor],
) -> None:
    return None


def cascade_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: CascadePlan,
    out: torch.Tensor,
    lse_out: torch.Tensor | None = None,
    *,
    sm_scale: float | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Decode attention for the batch the plan was last filled for.

    q: [requests · query_tokens, q_heads, head_dim], fp16 or bf16; request r's
    new tokens are rows r · query_tokens … r · query_tokens + query_tokens - 1,
    oldest first, and token t sees the request's keys up to position
    length - query_tokens + t. k, v: one layer's K and V views from
    kvkernels.layouts, [blocks, block_size, kv_heads, head_dim], same dtype; the
    new tokens' K and V are already in them. out: like q. lse_out, if given:
    [requests · query_tokens, q_heads], fp32, natural log. A request with no keys
    gets out = 0 and lse = -inf. sm_scale defaults to 1 / sqrt(head_dim). Runs
    as the custom op kvkernels::cascade_decode_attention.

    An fp8 cache: k and v are float8_e4m3fn views holding K / k_scale and
    V / v_scale, with k_scale and v_scale one-element fp32 tensors on q's device
    (an engine's per-layer scales).
    """
    rows = plan.num_requests * plan.query_tokens
    if q.dim() != 3 or q.shape[0] != rows:
        raise ValueError(
            f"q must be [{rows} = requests · query_tokens, q_heads, head_dim], got {tuple(q.shape)}"
        )
    if q.shape[1:] != (plan.num_q_heads, plan.head_dim):
        raise ValueError(f"q heads and head_dim {tuple(q.shape[1:])} do not match the plan")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    kv_ok = k.dtype == v.dtype and k.dtype in (q.dtype, torch.float8_e4m3fn)
    if q.dtype not in _IN_DTYPES or not kv_ok:
        raise ValueError(
            "q must be fp16 or bf16, and k and v q's dtype or both float8_e4m3fn; "
            f"got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    scales: list[torch.Tensor] = []
    if kv_fp8:
        for name, given in (("k_scale", k_scale), ("v_scale", v_scale)):
            if given is None or given.numel() != 1 or given.dtype != torch.float32:
                raise ValueError(f"an fp8 cache needs {name}: a one-element fp32 tensor")
        assert k_scale is not None and v_scale is not None
        scales = [k_scale.reshape(1), v_scale.reshape(1)]
    for name, t in (("k", k), ("v", v)):
        if t.dim() != 4 or t.shape[2:] != (plan.num_kv_heads, plan.head_dim):
            raise ValueError(
                f"{name} must be [blocks, block_size, {plan.num_kv_heads}, {plan.head_dim}]"
            )
        if t.shape[1] != plan.block_size:
            raise ValueError(f"{name} has block_size {t.shape[1]}, the plan {plan.block_size}")
    if tuple(out.shape) != tuple(q.shape) or out.dtype != q.dtype:
        raise ValueError("out must have q's shape and dtype")
    if lse_out is not None and (
        tuple(lse_out.shape) != tuple(q.shape[:2]) or lse_out.dtype != torch.float32
    ):
        raise ValueError(f"lse_out must be fp32 {tuple(q.shape[:2])}")
    if not _interpreting() and not q.is_cuda:
        raise ValueError("tensors must be on a CUDA device")
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(plan.head_dim)
    _cascade_op(
        q,
        k,
        v,
        plan.tensors(),
        [plan.part_o, plan.part_lse],
        out,
        lse_out,
        scale,
        plan.settings(),
        scales,
    )
