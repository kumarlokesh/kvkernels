"""Cascade decode attention: shared prefixes read once per group, suffixes once per request.

Each request's keys are its group's shared prefix followed by its own suffix.
The cascade computes attention over every work item, then merges each request's
partial results:

    work items                                   partial results     merge
    rows of r0-r3 · prefix A blocks, range 1  ──▶ p0
    rows of r0-r3 · prefix A blocks, range 2  ──▶ p1                 r0: p0, p1, p3 ──▶ out[r0]
    rows of r0    · r0's suffix blocks        ──▶ p3
    …

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
run the same kernel with different row-tile heights: a prefix item has many
requests' rows, a suffix item only one request's GQA group.

Split-K: when the tasks without splitting would give fewer than
programs_per_sm · SMs programs, each item's tokens are cut into ranges of at
least min_split_tokens, up to MAX_SPLITS, so small batches still fill the GPU.
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
HEAD_DIMS = (64, 128)
_IN_DTYPES = (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class PrefixGroup:
    """Requests that share a prefix: consecutive request indices, the prefix's blocks and length."""

    requests: range
    blocks: Sequence[int]
    length: int  # tokens; may end inside the last block


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
        min_split_tokens: int = 512,
        prefix_block_m: int = 64,
        suffix_block_m: int = 16,
        block_n: int | None = None,
        min_shared_tokens: int = 64,
        batch_invariant: bool = False,
    ) -> None:
        if num_q_heads % num_kv_heads:
            raise ValueError(f"num_q_heads {num_q_heads} is not a multiple of {num_kv_heads}")
        self.group = num_q_heads // num_kv_heads
        if not 1 <= self.group <= suffix_block_m:
            raise ValueError(f"GQA group {self.group} must be 1 to {suffix_block_m}")
        if head_dim not in HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
        if prefix_block_m == suffix_block_m:
            raise ValueError("prefix and suffix row tiles must differ in height")
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
        self.min_shared_tokens = min_shared_tokens
        self.min_shared_blocks = max(1, math.ceil(min_shared_tokens / block_size))
        self.batch_invariant = batch_invariant
        self.max_tasks = max_tasks or (64 * max_groups + MAX_SPLITS * max_requests)
        self.max_parts = max_parts or (2 * max_requests * num_q_heads * MAX_SPLITS)
        # Without split-K every plan fits: one prefix and one suffix part per row.
        # A batch-invariant plan must fit with split-K, which it never turns off.
        splits = MAX_SPLITS if batch_invariant else 1
        need_parts = (1 if batch_invariant else 2) * max_requests * num_q_heads * splits
        need_tasks = max(
            max_requests * splits,
            0
            if batch_invariant
            else max_groups + math.ceil(max_requests * self.group / prefix_block_m),
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
        self.min_split_tokens = min_split_tokens
        self.prefix_block_m = prefix_block_m
        self.suffix_block_m = suffix_block_m
        self.block_n = block_n or (16 if _interpreting() else 64)
        self.max_rows = max_requests * num_q_heads

        def ints(n: int) -> torch.Tensor:
            return torch.zeros(max(n, 1), dtype=torch.int32, device=device)

        items = max_groups + max_requests
        self.blocks = ints(max_blocks)
        self.perm = ints(max_requests)  # row i of the sorted batch is request perm[i]
        self.common = ints(max_requests)  # device planning: KV blocks shared with the previous row
        self.item_req_start, self.item_num_reqs, self.item_block_start = (
            ints(items),
            ints(items),
            ints(items),
        )
        # Device planning: length, ranges, span, task start, part start, merge offset per item.
        self.item_info = torch.zeros((6, items), dtype=torch.int32, device=device)
        # Item, row start, token start, token end, part base; prefix tasks, then suffix tasks.
        self.tasks = torch.zeros((2, 5, self.max_tasks), dtype=torch.int32, device=device)
        self.prefix_tasks = tuple(self.tasks[0])
        self.suffix_tasks = tuple(self.tasks[1])
        self.counts = ints(5)  # prefix tasks, suffix tasks, merge rows, groups, items
        self.planned_requests = ints(1)  # device planning: the batch size, read by the kernels
        self.graphs = PlanGraphs()  # device planning: recorded launches, replayed each step
        self.merge_start = ints(self.max_rows)
        self.merge_end = ints(self.max_rows)
        self.merge_indices = ints(self.max_parts)
        self.part_o = torch.zeros((self.max_parts, head_dim), dtype=torch.float32, device=device)
        self.part_lse = torch.zeros(self.max_parts, dtype=torch.float32, device=device)
        self.num_requests = 0
        self.index = self.blocks  # what item_block_start points into
        self.per_token = False  # index holds one slot per token (a slot list)

    # ----------------------------------------------------------------------------------

    def _check(
        self,
        groups: Sequence[PrefixGroup],
        suffix_blocks: Sequence[Sequence[int]],
        suffix_lens: Sequence[int],
    ) -> None:
        num_req = len(suffix_lens)
        if num_req > self.max_requests:
            raise ValueError(f"{num_req} requests exceed max_requests {self.max_requests}")
        if len(suffix_blocks) != num_req:
            raise ValueError(f"{len(suffix_blocks)} suffix block lists for {num_req} requests")
        if len(groups) > self.max_groups:
            raise ValueError(f"{len(groups)} groups exceed max_groups {self.max_groups}")
        owner: dict[int, int] = {}
        for gi, g in enumerate(groups):
            if g.requests.step != 1 or g.requests.start < 0 or g.requests.stop > num_req:
                raise ValueError(f"group {gi}: requests {g.requests} outside 0 … {num_req - 1}")
            for r in g.requests:
                if r in owner:
                    raise ValueError(f"groups {owner[r]} and {gi} overlap at request {r}")
                owner[r] = gi
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

    def _splits(self, length: int, base_programs: int) -> int:
        by_length = math.ceil(length / self.min_split_tokens)
        if self.batch_invariant:
            return max(1, min(MAX_SPLITS, by_length))
        target = self.programs_per_sm * self.num_sms
        if base_programs >= target:
            return 1
        return max(1, min(MAX_SPLITS, by_length, math.ceil(target / max(base_programs, 1))))

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
        self._check(groups, suffix_blocks, suffix_lens)
        num_req = len(suffix_lens)
        g, hkv, hq = self.group, self.num_kv_heads, self.num_q_heads

        # Work items: (first request, requests, block list, length, row-tile height).
        items: list[tuple[int, int, Sequence[int], int, int]] = []
        for grp in groups:
            if grp.length > 0:
                items.append(
                    (
                        grp.requests.start,
                        len(grp.requests),
                        grp.blocks,
                        grp.length,
                        self.prefix_block_m,
                    )
                )
        for r, (blocks, length) in enumerate(zip(suffix_blocks, suffix_lens, strict=True)):
            if length > 0:
                items.append((r, 1, blocks, length, self.suffix_block_m))
        base_programs = sum(math.ceil(n * g / bm) * hkv for _, n, _, _, bm in items)

        flat: list[int] = []
        req_start, num_reqs, block_start = [], [], []
        tasks: dict[int, list[list[int]]] = {
            self.prefix_block_m: [[], [], [], [], []],
            self.suffix_block_m: [[], [], [], [], []],
        }
        parts_of_row: list[list[int]] = [[] for _ in range(num_req * hq)]
        next_part = 0
        for index, (first_req, n_req, blocks, length, bm) in enumerate(items):
            req_start.append(first_req)
            num_reqs.append(n_req)
            block_start.append(len(flat))
            flat.extend(blocks[: math.ceil(length / self.block_size)])
            splits = self._splits(length, base_programs)
            span = math.ceil(math.ceil(length / splits) / self.block_n) * self.block_n
            ranges = [(s, min(s + span, length)) for s in range(0, length, span)]
            rows = n_req * g
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
                            request = first_req + item_row // g
                            head = kv * g + item_row % g
                            parts_of_row[request * hq + head].append(next_part + kv * tile_rows + i)
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
        for buffers, columns in (
            (self.prefix_tasks, prefix_tasks),
            (self.suffix_tasks, suffix_tasks),
        ):
            for buffer, column in zip(buffers, columns, strict=True):
                put(buffer, column)
        put(self.merge_start, starts)
        put(self.merge_end, [*starts[1:], len(indices)])
        put(self.merge_indices, indices)
        put(self.counts, [len(prefix_tasks[0]), len(suffix_tasks[0]), num_req * hq])
        self.num_requests = num_req
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
) -> None:
    index, perm, req_start, num_reqs, block_start = plan_tensors[:5]
    prefix_tasks = tuple(plan_tensors[5:10])
    suffix_tasks = tuple(plan_tensors[10:15])
    counts, merge_start, merge_end, indices = plan_tensors[15:19]
    part_o, part_lse = workspace
    block_size, group, max_tasks, max_rows, prefix_bm, suffix_bm, block_n, per_token = settings
    common = {
        "index": index,
        "perm": perm,
        "per_token": bool(per_token),
        "item_req_start": req_start,
        "item_num_reqs": num_reqs,
        "item_block_start": block_start,
        "part_o": part_o,
        "part_lse": part_lse,
        "sm_scale": sm_scale,
        "block_size": block_size,
        "group": group,
        "max_tasks": max_tasks,
        "block_n": block_n,
    }
    attend(q, k, v, tasks=prefix_tasks, num_tasks=counts[0:1], block_m=prefix_bm, **common)  # type: ignore[arg-type]
    attend(q, k, v, tasks=suffix_tasks, num_tasks=counts[1:2], block_m=suffix_bm, **common)  # type: ignore[arg-type]
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
) -> None:
    """Decode attention for the batch the plan was last filled for.

    q: [requests, q_heads, head_dim], fp16 or bf16. k, v: one layer's K and V
    views from kvkernels.layouts, [blocks, block_size, kv_heads, head_dim], same
    dtype. out: like q. lse_out, if given: [requests, q_heads], fp32, natural log.
    A request with no keys gets out = 0 and lse = -inf. sm_scale defaults to
    1 / sqrt(head_dim). Runs as the custom op kvkernels::cascade_decode_attention.
    """
    if q.dim() != 3 or q.shape[0] != plan.num_requests:
        raise ValueError(
            f"q must be [{plan.num_requests} requests, q_heads, head_dim], got {tuple(q.shape)}"
        )
    if q.shape[1:] != (plan.num_q_heads, plan.head_dim):
        raise ValueError(f"q heads and head_dim {tuple(q.shape[1:])} do not match the plan")
    if q.dtype not in _IN_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q, k and v need one dtype, fp16 or bf16; got {q.dtype}, {k.dtype}, {v.dtype}"
        )
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
    )
