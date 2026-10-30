"""Decode attention over work items: query rows that read the same list of KV blocks.

A work item is a set of query rows and one block list: a prefix group (every
request of the group, each with its GQA group of query heads, over the group's
shared blocks) or one request's suffix (its own rows over its own blocks). A
task is one tile of an item's rows over one token range of its block list.

Grid: (tasks, KV heads). One program:

    resident                                     per tile of BLOCK_N tokens
    q    BLOCK_M rows · head_dim, gathered:      token t ──▶ block = list[t // block_size],
         row i = (request, query head) of               slot = t % block_size
         the item, for this KV head              K, V of this KV head ◀── device memory
    m, l running max and sum per row             s = q · kᵀ · scale (base 2); online softmax
    acc  BLOCK_M · head_dim, fp32                acc = acc · 2^(m - m_new) + p · v

    end: each real row writes acc / l and lse = m · ln 2 + ln l to its part slot

K and V are the per-layer views of kvkernels.layouts ([blocks, slots, kv_heads,
head_dim], any strides), so one kernel reads every supported layout. The grid is
sized to the plan's capacity: programs past the number of tasks exit at once,
which keeps a captured CUDA graph valid for any batch up to that capacity.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LN2 = 0.6931471805599453
LOG2E = 1.4426950408889634


@triton.jit
def _work_item_kernel(  # type: ignore[no-untyped-def]
    q_ptr,
    k_ptr,
    v_ptr,
    blocks_ptr,
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
    BLOCK_N: tl.constexpr,
):
    task = tl.program_id(0)
    kv_head = tl.program_id(1)
    if task >= tl.load(num_tasks_ptr):
        return
    item = tl.load(task_item_ptr + task)
    row_start = tl.load(task_row_start_ptr + task)
    tok_start = tl.load(task_tok_start_ptr + task)
    tok_end = tl.load(task_tok_end_ptr + task)
    part_base = tl.load(task_part_base_ptr + task).to(tl.int64)
    req_start = tl.load(item_req_start_ptr + item)
    block_start = tl.load(item_block_start_ptr + item)
    item_rows = tl.load(item_num_reqs_ptr + item) * GROUP
    tile_rows = tl.minimum(item_rows - row_start, BLOCK_M)

    # Row i of the tile is (request, query head) of this item, for this KV head.
    rows = tl.arange(0, BLOCK_M)
    row_ok = rows < tile_rows
    item_row = row_start + rows
    request = req_start + item_row // GROUP
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
        block = tl.load(blocks_ptr + block_start + tokens // block_size, mask=tok_ok, other=0)
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


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    blocks: torch.Tensor,
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
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """Launch one program per (task slot up to max_tasks, KV head); see the module docstring."""
    task_item, task_row_start, task_tok_start, task_tok_end, task_part_base = tasks
    _work_item_kernel[(max_tasks, k.shape[2])](
        q,
        k,
        v,
        blocks,
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
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
