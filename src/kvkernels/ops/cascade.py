"""Cascade decode attention: shared prefixes read once per group, suffixes once per request.

Each request's keys are its group's shared prefix followed by its own suffix.
The cascade computes attention over every work item, then merges each request's
partial results:

    work items                                   partial results     merge
    rows of r0-r3 · prefix A blocks, range 1  ──▶ p0
    rows of r0-r3 · prefix A blocks, range 2  ──▶ p1                 r0: p0, p1, p3 ──▶ out[r0]
    rows of r0    · r0's suffix blocks        ──▶ p3
    …

A plan, computed on the host once per decode step and used by every layer,
lists the tasks (one tile of an item's rows over one token range), assigns each
partial result its slot in the workspace, and lists each (request, query head)
row's partials for the merge. Its buffers are allocated once for a maximum
batch and rewritten in place by each update, so the launches can be captured in
a CUDA graph and replayed after later updates.

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
    ) -> None:
        if num_q_heads % num_kv_heads:
            raise ValueError(f"num_q_heads {num_q_heads} is not a multiple of {num_kv_heads}")
        self.group = num_q_heads // num_kv_heads
        if not 1 <= self.group <= suffix_block_m:
            raise ValueError(f"GQA group {self.group} must be 1 to {suffix_block_m}")
        if head_dim not in HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
        self.max_requests = max_requests
        self.max_groups = max_groups
        self.max_blocks = max_blocks
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device
        self.max_tasks = max_tasks or (64 * max_groups + MAX_SPLITS * max_requests)
        self.max_parts = max_parts or (2 * max_requests * num_q_heads * MAX_SPLITS)
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
        self.item_req_start, self.item_num_reqs, self.item_block_start = (
            ints(items),
            ints(items),
            ints(items),
        )
        self.prefix_tasks = tuple(ints(self.max_tasks) for _ in range(5))
        self.suffix_tasks = tuple(ints(self.max_tasks) for _ in range(5))
        self.counts = ints(3)  # prefix tasks, suffix tasks, merge rows
        self.merge_indptr = ints(self.max_rows + 1)
        self.merge_indices = ints(self.max_parts)
        self.part_o = torch.zeros((self.max_parts, head_dim), dtype=torch.float32, device=device)
        self.part_lse = torch.zeros(self.max_parts, dtype=torch.float32, device=device)
        self.num_requests = 0

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
        target = self.programs_per_sm * self.num_sms
        if base_programs >= target:
            return 1
        by_length = math.ceil(length / self.min_split_tokens)
        return max(1, min(MAX_SPLITS, by_length, math.ceil(target / max(base_programs, 1))))

    def update(
        self,
        *,
        groups: Sequence[PrefixGroup],
        suffix_blocks: Sequence[Sequence[int]],
        suffix_lens: Sequence[int],
    ) -> None:
        """Plan one decode step. Requests are numbered 0 … len(suffix_lens) - 1."""
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
        if self.prefix_block_m == self.suffix_block_m:
            raise ValueError("prefix and suffix row tiles must differ in height")
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

        indptr = [0]
        indices: list[int] = []
        for parts in parts_of_row:
            indices.extend(parts)
            indptr.append(len(indices))

        def put(buffer: torch.Tensor, values: Sequence[int]) -> None:
            if values:
                buffer[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))

        put(self.blocks, flat)
        put(self.item_req_start, req_start)
        put(self.item_num_reqs, num_reqs)
        put(self.item_block_start, block_start)
        for buffers, columns in (
            (self.prefix_tasks, prefix_tasks),
            (self.suffix_tasks, suffix_tasks),
        ):
            for buffer, column in zip(buffers, columns, strict=True):
                put(buffer, column)
        put(self.merge_indptr, indptr)
        put(self.merge_indices, indices)
        put(self.counts, [len(prefix_tasks[0]), len(suffix_tasks[0]), num_req * hq])
        self.num_requests = num_req

    # ----------------------------------------------------------------------------------

    def tensors(self) -> list[torch.Tensor]:
        """Read-only plan tensors, in the order the registered op expects."""
        return [
            self.blocks,
            self.item_req_start,
            self.item_num_reqs,
            self.item_block_start,
            *self.prefix_tasks,
            *self.suffix_tasks,
            self.counts,
            self.merge_indptr,
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
        ]


@torch.library.custom_op(
    "kvkernels::cascade_decode_attention", mutates_args=("workspace", "out", "lse_out")
)
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
    blocks, req_start, num_reqs, block_start = plan_tensors[:4]
    prefix_tasks = tuple(plan_tensors[4:9])
    suffix_tasks = tuple(plan_tensors[9:14])
    counts, indptr, indices = plan_tensors[14:17]
    part_o, part_lse = workspace
    block_size, group, max_tasks, max_rows, prefix_bm, suffix_bm, block_n = settings
    common = {
        "blocks": blocks,
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
    merge_ragged(part_o, part_lse, indptr, indices, counts[2:3], out, lse_out, max_rows=max_rows)


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
    """Decode attention for the batch the plan was last updated with.

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
