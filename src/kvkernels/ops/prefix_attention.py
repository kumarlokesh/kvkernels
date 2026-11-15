"""Decode attention over work items: query rows that read the same list of KV blocks.

A work item is a set of query rows and one block list: a prefix group (every
request of the group, each with its GQA group of query heads, over the group's
shared blocks) or one request's suffix (its own rows over its own blocks). A
task is one tile of an item's rows over one token range of its block list. An
item's requests are consecutive rows of the sorted batch; request = perm[row].

Grid: (task programs, KV heads); each program loops over tasks. One task:

    resident                                     per tile of BLOCK_N tokens
    q    BLOCK_M rows · head_dim, gathered:      token t ──▶ block = list[t // block_size],
         row i = (request, query head) of               slot = t % block_size
                                                 (slot list: s = list[t], block =
                                                 s // block_size, slot = s % block_size)
         the item, for this KV head              K, V of this KV head ◀── device memory
    m, l running max and sum per row             s = q · kᵀ · scale (base 2); online softmax
    acc  BLOCK_M · head_dim, fp32                acc = acc · 2^(m - m_new) + p · v

    end: each real row writes acc / l and lse = m · ln 2 + ln l to its part slot

A task with few rows runs in a shorter tile. The kernel holds the task body
compiled for BLOCK_M rows and, down to MIN_BLOCK_M, for half and a quarter of
that, and each task takes the shortest that holds its rows:

    BLOCK_M 128, MIN_BLOCK_M 32     task rows   1-32 ─▶ 32-row tile
                                                33-64 ─▶ 64-row tile
                                               65-128 ─▶ 128-row tile

Warps and pipeline stages stay those of the launch. On an L4 this made groups
of 17 to 32 rows (Llama 3.1 8B, batch 8) up to 17% faster; 64-row tiles for
groups of 64 rows gained nothing over 128-row tiles.

K and V are the per-layer views of kvkernels.layouts ([blocks, slots, kv_heads,
head_dim], any strides), so one kernel reads every supported layout. The number
of tasks is read from the device, and the grid is a fixed number of programs
that loop over tasks, so a captured CUDA graph serves any batch up to the
plan's capacity. Sizing the grid to that capacity instead launched up to
100,000 programs per layer that exited at once, which cost 0.27 ms per layer
on an L4 for a plan sized for 256 requests.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LN2 = 0.6931471805599453
LOG2E = 1.4426950408889634


@triton.jit
def _task(  # type: ignore[no-untyped-def]
    q_ptr,
    k_ptr,
    v_ptr,
    index_ptr,
    perm_ptr,
    part_o_ptr,
    part_lse_ptr,
    kv_head,
    req_start,
    block_start,
    row_start,
    tile_rows,
    tok_start,
    tok_end,
    part_base,
    stride_qr,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    scale_log2e,
    block_size,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PER_TOKEN: tl.constexpr,
):
    """One task: tile_rows (at most BLOCK_M) rows of an item over tokens [tok_start, tok_end)."""
    # Row i of the tile is (request, query head) of this item, for this KV head.
    rows = tl.arange(0, BLOCK_M)
    row_ok = rows < tile_rows
    item_row = row_start + rows
    request = tl.load(perm_ptr + req_start + item_row // GROUP, mask=row_ok, other=0)
    q_head = kv_head * GROUP + item_row % GROUP
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + request[:, None].to(tl.int64) * stride_qr
        + q_head[:, None] * stride_qh
        + dims[None, :] * stride_qd,
        mask=row_ok[:, None],
        other=0.0,
    )

    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)  # noqa: E741
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for start in range(tok_start, tok_end, BLOCK_N):
        tokens = start + cols
        tok_ok = tokens < tok_end
        if PER_TOKEN:
            entry = tl.load(index_ptr + block_start + tokens, mask=tok_ok, other=0).to(tl.int64)
            block = entry // block_size
            slot = entry % block_size
        else:
            block = tl.load(index_ptr + block_start + tokens // block_size, mask=tok_ok, other=0)
            block = block.to(tl.int64)
            slot = tokens % block_size
        k = tl.load(
            k_ptr
            + block[None, :] * stride_kb
            + slot[None, :] * stride_ks
            + kv_head * stride_kh
            + dims[:, None] * stride_kd,
            mask=tok_ok[None, :],
            other=0.0,
        )
        s = tl.dot(q, k) * scale_log2e
        s = tl.where(tok_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)  # noqa: E741
        v = tl.load(
            v_ptr
            + block[:, None] * stride_vb
            + slot[:, None] * stride_vs
            + kv_head * stride_vh
            + dims[None, :] * stride_vd,
            mask=tok_ok[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    # Every task covers at least one token, so each real row has l > 0.
    out = acc / l[:, None]
    lse = m * 0.6931471805599453 + tl.log(l)  # m is in base-2 units
    part = part_base + kv_head * tile_rows + rows
    tl.store(part_o_ptr + part[:, None] * HEAD_DIM + dims[None, :], out, mask=row_ok[:, None])
    tl.store(part_lse_ptr + part, lse, mask=row_ok)


@triton.jit
def _work_item_kernel(  # type: ignore[no-untyped-def]
    q_ptr,
    k_ptr,
    v_ptr,
    index_ptr,
    perm_ptr,
    item_req_start_ptr,
    item_num_reqs_ptr,
    item_block_start_ptr,
    task_item_ptr,
    task_row_start_ptr,
    task_tok_start_ptr,
    task_tok_end_ptr,
    task_part_base_ptr,
    num_tasks_ptr,
    part_o_ptr,
    part_lse_ptr,
    stride_qr,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    scale_log2e,
    block_size,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MIN_BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PER_TOKEN: tl.constexpr,
):
    kv_head = tl.program_id(1)
    num_tasks = tl.load(num_tasks_ptr)
    # Program p runs tasks p, p + programs, … below num_tasks, so the grid does
    # not grow with the plan's capacity (empty programs cost time in a graph).
    task = tl.program_id(0)
    while task < num_tasks:
        item = tl.load(task_item_ptr + task)
        row_start = tl.load(task_row_start_ptr + task)
        tok_start = tl.load(task_tok_start_ptr + task)
        tok_end = tl.load(task_tok_end_ptr + task)
        part_base = tl.load(task_part_base_ptr + task).to(tl.int64)
        req_start = tl.load(item_req_start_ptr + item)
        block_start = tl.load(item_block_start_ptr + item).to(tl.int64)
        item_rows = tl.load(item_num_reqs_ptr + item) * GROUP
        tile_rows = tl.minimum(item_rows - row_start, BLOCK_M)
        # A task with few rows runs a copy of the task compiled for a shorter
        # tile: half or a quarter of BLOCK_M, down to MIN_BLOCK_M.
        if MIN_BLOCK_M == BLOCK_M:
            _task(
                q_ptr,
                k_ptr,
                v_ptr,
                index_ptr,
                perm_ptr,
                part_o_ptr,
                part_lse_ptr,
                kv_head,
                req_start,
                block_start,
                row_start,
                tile_rows,
                tok_start,
                tok_end,
                part_base,
                stride_qr,
                stride_qh,
                stride_qd,
                stride_kb,
                stride_ks,
                stride_kh,
                stride_kd,
                stride_vb,
                stride_vs,
                stride_vh,
                stride_vd,
                scale_log2e,
                block_size,
                GROUP,
                HEAD_DIM,
                BLOCK_M,
                BLOCK_N,
                PER_TOKEN,
            )
        elif MIN_BLOCK_M * 2 == BLOCK_M:
            if tile_rows <= MIN_BLOCK_M:
                _task(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    index_ptr,
                    perm_ptr,
                    part_o_ptr,
                    part_lse_ptr,
                    kv_head,
                    req_start,
                    block_start,
                    row_start,
                    tile_rows,
                    tok_start,
                    tok_end,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_qd,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_kd,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    stride_vd,
                    scale_log2e,
                    block_size,
                    GROUP,
                    HEAD_DIM,
                    MIN_BLOCK_M,
                    BLOCK_N,
                    PER_TOKEN,
                )
            else:
                _task(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    index_ptr,
                    perm_ptr,
                    part_o_ptr,
                    part_lse_ptr,
                    kv_head,
                    req_start,
                    block_start,
                    row_start,
                    tile_rows,
                    tok_start,
                    tok_end,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_qd,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_kd,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    stride_vd,
                    scale_log2e,
                    block_size,
                    GROUP,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    PER_TOKEN,
                )
        else:
            if tile_rows <= MIN_BLOCK_M:
                _task(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    index_ptr,
                    perm_ptr,
                    part_o_ptr,
                    part_lse_ptr,
                    kv_head,
                    req_start,
                    block_start,
                    row_start,
                    tile_rows,
                    tok_start,
                    tok_end,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_qd,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_kd,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    stride_vd,
                    scale_log2e,
                    block_size,
                    GROUP,
                    HEAD_DIM,
                    MIN_BLOCK_M,
                    BLOCK_N,
                    PER_TOKEN,
                )
            elif tile_rows <= BLOCK_M // 2:
                _task(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    index_ptr,
                    perm_ptr,
                    part_o_ptr,
                    part_lse_ptr,
                    kv_head,
                    req_start,
                    block_start,
                    row_start,
                    tile_rows,
                    tok_start,
                    tok_end,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_qd,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_kd,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    stride_vd,
                    scale_log2e,
                    block_size,
                    GROUP,
                    HEAD_DIM,
                    BLOCK_M // 2,
                    BLOCK_N,
                    PER_TOKEN,
                )
            else:
                _task(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    index_ptr,
                    perm_ptr,
                    part_o_ptr,
                    part_lse_ptr,
                    kv_head,
                    req_start,
                    block_start,
                    row_start,
                    tile_rows,
                    tok_start,
                    tok_end,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_qd,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_kd,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    stride_vd,
                    scale_log2e,
                    block_size,
                    GROUP,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    PER_TOKEN,
                )
        task += tl.num_programs(0)


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    index: torch.Tensor,
    perm: torch.Tensor,
    per_token: bool,
    item_req_start: torch.Tensor,
    item_num_reqs: torch.Tensor,
    item_block_start: torch.Tensor,
    tasks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    num_tasks: torch.Tensor,
    part_o: torch.Tensor,
    part_lse: torch.Tensor,
    sm_scale: float,
    block_size: int,
    group: int,
    max_tasks: int,
    block_m: int,
    block_n: int,
    min_block_m: int | None = None,
    num_programs: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """Launch min(max_tasks, num_programs) programs per KV head; see the module docstring.

    min_block_m, block_m // 2 or block_m // 4 (default block_m): a task of at most
    that many rows runs in a tile of that height instead of block_m.
    """
    min_block_m = min_block_m or block_m
    if min_block_m not in (block_m, block_m // 2, block_m // 4) or min_block_m < 16:
        raise ValueError(f"min_block_m {min_block_m}: must be block_m, /2 or /4, and at least 16")
    task_item, task_row_start, task_tok_start, task_tok_end, task_part_base = tasks
    grid = max(1, min(max_tasks, num_programs or max_tasks))
    _work_item_kernel[(grid, k.shape[2])](
        q,
        k,
        v,
        index,
        perm,
        item_req_start,
        item_num_reqs,
        item_block_start,
        task_item,
        task_row_start,
        task_tok_start,
        task_tok_end,
        task_part_base,
        num_tasks,
        part_o,
        part_lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        sm_scale * LOG2E,
        block_size,
        GROUP=group,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        MIN_BLOCK_M=min_block_m,
        BLOCK_N=block_n,
        PER_TOKEN=per_token,
        num_warps=num_warps,
        num_stages=num_stages,
    )
