"""Decode attention over work items: query rows that read the same list of KV blocks.

A work item is a set of query rows and one block list: a tree node (every
request below it, each with its query tokens times its GQA group of query
heads, over the node's blocks) or one request's suffix (its own rows over its
own blocks, with the causal rule over its newest keys when it has several query
tokens). A task is one tile of an item's rows over one token range of its block
list. An item's requests are consecutive rows of the sorted batch; request =
perm[row]. An fp8 cache's tiles convert to q's dtype unscaled: K's scale
multiplies the scores and V's the output.

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
    key_limit,
    part_base,
    stride_qr,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    scale_log2e,
    out_scale,
    block_size,
    GROUP: tl.constexpr,
    QUERY_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PER_TOKEN: tl.constexpr,
):
    """One task: tile_rows (at most BLOCK_M) rows of an item over tokens [tok_start, tok_end).

    With several query tokens per request, token t sees the item's tokens up to
    key_limit + t (the causal rule of a suffix item; node items pass a limit past
    their end).
    """
    # Row i of the tile is (request, query token, query head) of this item, for
    # this KV head: item row = request · QUERY_TOKENS · GROUP + token · GROUP + head.
    rows = tl.arange(0, BLOCK_M)
    row_ok = rows < tile_rows
    item_row = row_start + rows
    request = tl.load(
        perm_ptr + req_start + item_row // (GROUP * QUERY_TOKENS), mask=row_ok, other=0
    )
    token = (item_row // GROUP) % QUERY_TOKENS
    q_head = kv_head * GROUP + item_row % GROUP
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + (request * QUERY_TOKENS + token)[:, None].to(tl.int64) * stride_qr
        + q_head[:, None] * stride_qh
        + dims[None, :],
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
            + dims[:, None],
            mask=tok_ok[None, :],
            other=0.0,
        ).to(q.dtype)  # an fp8 tile converts exactly; its scale is in scale_log2e
        s = tl.dot(q, k) * scale_log2e
        if QUERY_TOKENS > 1:
            visible = tok_ok[None, :] & (tokens[None, :] <= key_limit + token[:, None])
            s = tl.where(visible, s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            # A row that has seen no key yet keeps m = -inf; subtract 0 instead.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m - m_safe)
            p = tl.exp2(s - m_safe[:, None])
        else:
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
            + dims[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        ).to(q.dtype)  # an fp8 tile's scale is applied to the output, in out_scale
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m = m_new

    if QUERY_TOKENS > 1:
        # A row whose range ends before its first visible key saw nothing: an
        # empty partial result (out 0, lse -inf), which the merge skips.
        seen = l > 0
        out = tl.where(seen[:, None], acc * out_scale / tl.where(seen, l, 1.0)[:, None], 0.0)
        lse = tl.where(seen, m * 0.6931471805599453 + tl.log(tl.where(seen, l, 1.0)), m)
    else:
        # Every task covers at least one token, so each real row has l > 0.
        out = acc * out_scale / l[:, None]
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
    item_key_limit_ptr,
    task_item_ptr,
    task_row_start_ptr,
    task_tok_start_ptr,
    task_tok_end_ptr,
    task_part_base_ptr,
    num_tasks_ptr,
    part_o_ptr,
    part_lse_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_qr,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    scale_log2e,
    block_size,
    GROUP: tl.constexpr,
    QUERY_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MIN_BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PER_TOKEN: tl.constexpr,
    KV_FP8: tl.constexpr,
):
    kv_head = tl.program_id(1)
    # An fp8 cache holds K / k_scale and V / v_scale: K's scale multiplies the
    # scores and V's the output, so each tile converts to q's dtype unscaled.
    if KV_FP8:
        score_scale = scale_log2e * tl.load(k_scale_ptr)
        out_scale = tl.load(v_scale_ptr)
    else:
        score_scale = scale_log2e
        out_scale = 1.0
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
        item_rows = tl.load(item_num_reqs_ptr + item) * (GROUP * QUERY_TOKENS)
        key_limit = tl.load(item_key_limit_ptr + item) if QUERY_TOKENS > 1 else 0
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
                key_limit,
                part_base,
                stride_qr,
                stride_qh,
                stride_kb,
                stride_ks,
                stride_kh,
                stride_vb,
                stride_vs,
                stride_vh,
                score_scale,
                out_scale,
                block_size,
                GROUP,
                QUERY_TOKENS,
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
                    key_limit,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    score_scale,
                    out_scale,
                    block_size,
                    GROUP,
                    QUERY_TOKENS,
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
                    key_limit,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    score_scale,
                    out_scale,
                    block_size,
                    GROUP,
                    QUERY_TOKENS,
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
                    key_limit,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    score_scale,
                    out_scale,
                    block_size,
                    GROUP,
                    QUERY_TOKENS,
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
                    key_limit,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    score_scale,
                    out_scale,
                    block_size,
                    GROUP,
                    QUERY_TOKENS,
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
                    key_limit,
                    part_base,
                    stride_qr,
                    stride_qh,
                    stride_kb,
                    stride_ks,
                    stride_kh,
                    stride_vb,
                    stride_vs,
                    stride_vh,
                    score_scale,
                    out_scale,
                    block_size,
                    GROUP,
                    QUERY_TOKENS,
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
    query_tokens: int = 1,
    item_key_limit: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Launch min(max_tasks, num_programs) programs per KV head; see the module docstring.

    min_block_m, block_m // 2 or block_m // 4 (default block_m): a task of at most
    that many rows runs in a tile of that height instead of block_m.
    """
    # The kernel reads head_dim as contiguous: with its stride a runtime argument,
    # Triton cannot prove it is 1 and loads K and V 2 bytes at a time.
    if q.stride(2) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        raise ValueError("q, k and v need a contiguous head dimension (stride 1)")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if kv_fp8 and (k_scale is None or v_scale is None):
        raise ValueError("an fp8 cache needs k_scale and v_scale")
    if query_tokens > 1 and item_key_limit is None:
        raise ValueError("several query tokens per request need item_key_limit")
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
        item_key_limit if item_key_limit is not None else item_block_start,  # unread at T = 1
        task_item,
        task_row_start,
        task_tok_start,
        task_tok_end,
        task_part_base,
        num_tasks,
        part_o,
        part_lse,
        k_scale if kv_fp8 else part_lse,  # unread without fp8
        v_scale if kv_fp8 else part_lse,
        *q.stride()[:2],
        *k.stride()[:3],
        *v.stride()[:3],
        sm_scale * LOG2E,
        block_size,
        GROUP=group,
        QUERY_TOKENS=query_tokens,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        MIN_BLOCK_M=min_block_m,
        BLOCK_N=block_n,
        PER_TOKEN=per_token,
        KV_FP8=kv_fp8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
