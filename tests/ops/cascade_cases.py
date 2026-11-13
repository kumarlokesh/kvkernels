"""Building cascade test cases: a cache, its K and V views, block lists and the reference.

A case is a batch of requests on one layer's cache built by kvkernels.layouts.
Prefix groups get their own blocks, shared by every request in the group; each
request's suffix gets its own blocks. Block 0 is never used (both engines
reserve it).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache
from kvkernels.ops.cascade import CascadePlan, PrefixGroup
from kvkernels.reference.attention import RequestKV, attention_ref32, attention_ref_low

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING else "cuda"


@dataclass
class Case:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    groups: list[PrefixGroup]
    suffix_blocks: list[list[int]]
    suffix_lens: list[int]
    requests: list[RequestKV]
    block_size: int
    num_blocks: int


def make_case(
    *,
    layout: str = "vllm_LBNHC",
    group_lens: Sequence[tuple[int, int]] = ((3, 20),),
    suffix_lens: Sequence[int] = (5, 0, 9, 3),
    q_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 4,
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
) -> Case:
    """group_lens: (number of requests, prefix length) per group, assigned to requests in order."""
    gen = torch.Generator().manual_seed(seed)
    next_block = 1

    def take(tokens: int) -> list[int]:
        nonlocal next_block
        count = math.ceil(tokens / block_size)
        blocks = list(range(next_block, next_block + count))
        next_block += count
        return blocks

    groups: list[PrefixGroup] = []
    prefix_of: dict[int, tuple[list[int], int]] = {}
    first = 0
    for size, length in group_lens:
        blocks = take(length)
        groups.append(
            PrefixGroup(requests=range(first, first + size), blocks=blocks, length=length)
        )
        for r in range(first, first + size):
            prefix_of[r] = (blocks, length)
        first += size
    suffix_blocks = [take(n) for n in suffix_lens]
    num_blocks = next_block + 1

    spec = CacheSpec(1, num_blocks, block_size, kv_heads, head_dim, dtype)
    lay = LAYOUTS[layout]
    layer = make_cache(lay, spec, DEVICE)[0]
    k, v = kv_views(lay, spec, layer)
    k.copy_(torch.randn(tuple(k.shape), generator=gen).to(dtype))
    v.copy_(torch.randn(tuple(v.shape), generator=gen).to(dtype))
    q = torch.randn((len(suffix_lens), q_heads, head_dim), generator=gen).to(
        dtype=dtype, device=DEVICE
    )
    requests = [
        RequestKV(
            prefix_blocks=prefix_of.get(r, ([], 0))[0],
            prefix_len=prefix_of.get(r, ([], 0))[1],
            suffix_blocks=suffix_blocks[r],
            suffix_len=suffix_lens[r],
        )
        for r in range(len(suffix_lens))
    ]
    return Case(q, k, v, groups, suffix_blocks, list(suffix_lens), requests, block_size, num_blocks)


def make_plan(case: Case, **kwargs: Any) -> CascadePlan:
    plan = CascadePlan(
        max_requests=max(len(case.suffix_lens), 1),
        max_groups=len(case.groups) + 2,  # room for tests that submit other groups
        max_blocks=case.num_blocks * 2,
        num_q_heads=case.q.shape[1],
        num_kv_heads=case.k.shape[2],
        head_dim=case.q.shape[2],
        block_size=case.block_size,
        num_blocks=case.num_blocks,
        device=torch.device(DEVICE),
        **kwargs,
    )
    plan.update(groups=case.groups, suffix_blocks=case.suffix_blocks, suffix_lens=case.suffix_lens)
    return plan


def references(
    case: Case,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    return (
        attention_ref_low(case.q, case.k, case.v, case.requests),
        attention_ref32(case.q, case.k, case.v, case.requests),
    )
