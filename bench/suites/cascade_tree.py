"""Tree-shaped batches: the cascade with nodes at several levels against two levels and FlashInfer.

One decode step of one layer. Requests share KV blocks along a tree: every
request walks down the same number of levels, and each level's node is shared
by the requests below it. Then each request has a suffix of its own:

    three_levels   1 system prompt ─▶ 4 tool lists ─▶ 16 histories ─▶ batch / 16 requests each
    sampling       1 system prompt ─▶ batch / 8 prompts ─▶ 8 samples each
    flat           1 system prompt ─▶ every request

Node and suffix KV blocks are distinct physical blocks drawn in a random order.
The tree is described to every implementation from the same block lists.
Each request has query_tokens new tokens (1, or 4 for a speculative-decoding
verification step), and token t sees the request's keys up to position
length - query_tokens + t.

Implementations, each checked against the first within the fp32-reference
tolerance rule:

    flashinfer_per_request   FlashInfer batch decode (one token) or batch prefill
                             with a causal mask (several), each request over all its
                             keys; first, and so the reference, when FlashInfer runs
    torch_per_request        PyTorch SDPA per request; first only without FlashInfer
    kvkernels_tree           the cascade planned on the GPU, up to 4 levels of nodes
    kvkernels_two_level      the same with one level: the requests every node shares
    flashinfer_multilevel    FlashInfer's multi-level cascade, one level per tree
                             depth and one for the suffixes

Planning on the GPU is inside the timed call, as for kvkernels_gpu_plan in the
cascade suite; FlashInfer's plan runs on the host and is recorded as plan_us.
bytes_moved counts each node's KV once and each suffix once.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bench.harness import Case, Tensors
from bench.schema import Params
from bench.suites.cascade import FLASHINFER_WORKSPACE_BYTES, MODELS, _paged

if TYPE_CHECKING:
    import torch

    from kvkernels.ops.cascade import CascadePlan

BLOCK_SIZE = 16
LAYOUT = "vllm_LBNHC"
BATCHES = (16, 32, 64, 128)
QUERY_TOKENS = (1, 4)
SUFFIXES = (64, 512)
# Shape name to node lengths per level, outermost first.
SHAPES: dict[str, tuple[tuple[int, ...], ...]] = {
    "three_levels": ((1024, 2048, 4096), (4096, 4096, 16384)),
    "sampling": ((2048, 4096),),
    "flat": ((4096,),),
}


def fanouts(shape: str, batch: int) -> tuple[int, ...]:
    """Nodes at each level."""
    return {"three_levels": (1, 4, 16), "sampling": (1, batch // 8), "flat": (1,)}[shape]


def _params(
    *,
    shape: str,
    lens: Sequence[int],
    batch: int,
    suffix: int,
    query_tokens: int,
    model: str = "llama-3.1-8b",
) -> Params:
    q_heads, kv_heads, head_dim = MODELS[model]
    return {
        "model": model,
        "shape": shape,
        "lens": "-".join(str(n) for n in lens),
        "batch": batch,
        "suffix_len": suffix,
        "query_tokens": query_tokens,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "layout": LAYOUT,
        "block": BLOCK_SIZE,
        "dtype": "float16",
    }


def grid() -> Iterator[Params]:
    for shape, length_sets in SHAPES.items():
        for lens in length_sets:
            for model in MODELS:
                for batch in BATCHES:
                    for suffix in SUFFIXES:
                        for tokens in QUERY_TOKENS:
                            yield _params(
                                shape=shape,
                                lens=lens,
                                batch=batch,
                                suffix=suffix,
                                query_tokens=tokens,
                                model=model,
                            )


@dataclass(frozen=True)
class Tree:
    """The batch's nodes per level, as (first request, requests, KV blocks), and suffixes."""

    levels: list[list[tuple[int, int, list[int]]]]
    level_lens: list[int]
    suffix_blocks: list[list[int]]
    suffix_len: int
    block_size: int

    @property
    def size(self) -> int:
        return len(self.suffix_blocks)

    def request_blocks(self) -> list[list[int]]:
        out = []
        for r in range(self.size):
            blocks: list[int] = []
            for nodes in self.levels:
                blocks += next(b for first, n, b in nodes if first <= r < first + n)
            out.append(blocks + self.suffix_blocks[r])
        return out

    def request_lens(self) -> list[int]:
        return [sum(self.level_lens) + self.suffix_len] * self.size


def tree_of(p: Params) -> Tree:
    """The same tree for every call with these params: blocks in a fixed random order."""
    import torch

    shape, batch, block = str(p["shape"]), int(p["batch"]), int(p["block"])
    lens = [int(n) for n in str(p["lens"]).split("-")]
    suffix = int(p["suffix_len"])
    counts = fanouts(shape, batch)
    needed = sum(c * math.ceil(n / block) for c, n in zip(counts, lens, strict=True))
    needed += batch * math.ceil(suffix / block)
    order = (torch.randperm(needed, generator=torch.Generator().manual_seed(1)) + 1).tolist()
    taken = 0

    def take(n: int) -> list[int]:
        nonlocal taken
        taken += n
        return order[taken - n : taken]

    levels = []
    for count, length in zip(counts, lens, strict=True):
        per = batch // count
        levels.append([(j * per, per, take(math.ceil(length / block))) for j in range(count)])
    suffix_blocks = [take(math.ceil(suffix / block)) for _ in range(batch)]
    return Tree(levels, lens, suffix_blocks, suffix, block)


def bytes_moved(p: Params) -> int:
    lens = [int(n) for n in str(p["lens"]).split("-")]
    counts = fanouts(str(p["shape"]), int(p["batch"]))
    tokens = sum(c * n for c, n in zip(counts, lens, strict=True))
    tokens += int(p["batch"]) * int(p["suffix_len"])
    return tokens * 2 * int(p["kv_heads"]) * int(p["head_dim"]) * 2


def flops(p: Params) -> int:
    keys = sum(int(n) for n in str(p["lens"]).split("-")) + int(p["suffix_len"])
    rows = int(p["batch"]) * int(p["query_tokens"]) * int(p["q_heads"])
    return 4 * rows * keys * int(p["head_dim"])


def make_inputs(p: Params, *, nhd_copies: bool = False, device: str = "cuda") -> Tensors:
    import torch

    from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache

    tree = tree_of(p)
    q_heads, kv_heads, head_dim = (int(p[k]) for k in ("q_heads", "kv_heads", "head_dim"))
    num_blocks = max(b for blocks in tree.request_blocks() for b in blocks) + 1
    layout = LAYOUTS[LAYOUT]
    spec = CacheSpec(1, num_blocks, tree.block_size, kv_heads, head_dim, torch.float16)
    k, v = kv_views(layout, spec, make_cache(layout, spec, device)[0])
    gen = torch.Generator(device=device).manual_seed(0)
    for view in (k, v):
        view.copy_(torch.randn(tuple(view.shape), generator=gen, device=device, dtype=view.dtype))
    rows = tree.size * int(p["query_tokens"])
    q = torch.randn((rows, q_heads, head_dim), generator=gen, device=device, dtype=torch.float16)
    inputs: Tensors = {"q": q, "k": k, "v": v, "out": torch.zeros_like(q)}
    if nhd_copies:
        inputs["k_nhd"] = k.contiguous()
        inputs["v_nhd"] = v.contiguous()
    return inputs


def _sm_scale(t: Tensors) -> float:
    return 1.0 / math.sqrt(t["q"].shape[-1])


# --- implementations ----------------------------------------------------------------


class KvkernelsTree:
    def __init__(self, p: Params, *, max_levels: int) -> None:
        self.p = p
        self.max_levels = max_levels
        self.cascade: CascadePlan | None = None
        self.table: torch.Tensor | None = None
        self.lens: torch.Tensor | None = None

    def plan(self, t: Tensors) -> None:
        import torch

        from kvkernels.ops.cascade import CascadePlan

        tree = tree_of(self.p)
        q, k = t["q"], t["k"]
        rows = tree.request_blocks()
        width = max(map(len, rows))
        self.table = torch.tensor(
            [r + [0] * (width - len(r)) for r in rows], dtype=torch.int32, device=q.device
        )
        self.lens = torch.tensor(tree.request_lens(), dtype=torch.int32, device=q.device)
        self.cascade = CascadePlan(
            max_requests=tree.size,
            max_groups=tree.size,
            max_blocks=1,
            num_q_heads=q.shape[1],
            num_kv_heads=k.shape[2],
            head_dim=q.shape[2],
            block_size=k.shape[1],
            num_blocks=k.shape[0],
            device=torch.device(q.device),
            query_tokens=int(self.p["query_tokens"]),
            max_levels=self.max_levels,
        )

    def run(self, t: Tensors) -> None:
        from kvkernels.ops.cascade import cascade_decode_attention

        assert self.cascade is not None and self.table is not None and self.lens is not None
        self.cascade.plan_from_block_table(self.table, self.lens)
        cascade_decode_attention(t["q"], t["k"], t["v"], self.cascade, t["out"])


class FlashInferPerRequest:
    def __init__(self, p: Params) -> None:
        self.p = p
        self.wrapper: Any = None

    def plan(self, t: Tensors) -> None:
        import torch
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        tree = tree_of(self.p)
        q, k = t["q"], t["k"]
        tokens = int(self.p["query_tokens"])
        workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
        indptr, indices, last = _paged(
            tree.request_blocks(), tree.request_lens(), tree.block_size, q.device
        )
        common = {"q_data_type": q.dtype, "kv_data_type": k.dtype, "sm_scale": _sm_scale(t)}
        if tokens == 1:
            self.wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
            self.wrapper.plan(
                indptr, indices, last, q.shape[1], k.shape[2], q.shape[2], tree.block_size, **common
            )
        else:
            self.wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
            qo = torch.arange(0, tree.size * tokens + 1, tokens, dtype=torch.int32, device=q.device)
            self.wrapper.plan(
                qo,
                indptr,
                indices,
                last,
                q.shape[1],
                k.shape[2],
                q.shape[2],
                tree.block_size,
                causal=True,
                **common,
            )

    def run(self, t: Tensors) -> None:
        self.wrapper.run(t["q"], (t["k_nhd"], t["v_nhd"]), out=t["out"])


class FlashInferMultiLevel:
    def __init__(self, p: Params) -> None:
        self.p = p
        self.wrapper: Any = None

    def plan(self, t: Tensors) -> None:
        import torch
        from flashinfer import MultiLevelCascadeAttentionWrapper

        tree = tree_of(self.p)
        q, k = t["q"], t["k"]
        tokens = int(self.p["query_tokens"])
        dev = q.device
        workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        self.wrapper = MultiLevelCascadeAttentionWrapper(len(tree.levels) + 1, workspace, "NHD")
        qo: list[torch.Tensor] = []
        indptr: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        last: list[torch.Tensor] = []
        for nodes, length in zip(tree.levels, tree.level_lens, strict=True):
            starts = [first * tokens for first, _, _ in nodes] + [tree.size * tokens]
            paged = _paged([b for _, _, b in nodes], [length] * len(nodes), tree.block_size, dev)
            qo.append(torch.tensor(starts, dtype=torch.int32, device=dev))
            for out, value in zip((indptr, indices, last), paged, strict=True):
                out.append(value)
        paged = _paged(tree.suffix_blocks, [tree.suffix_len] * tree.size, tree.block_size, dev)
        qo.append(torch.arange(0, tree.size * tokens + 1, tokens, dtype=torch.int32, device=dev))
        for out, value in zip((indptr, indices, last), paged, strict=True):
            out.append(value)
        self.wrapper.plan(
            qo,
            indptr,
            indices,
            last,
            q.shape[1],
            k.shape[2],
            q.shape[2],
            tree.block_size,
            causal=tokens > 1,
            sm_scale=_sm_scale(t),
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )

    def run(self, t: Tensors) -> None:
        # The wrapper allocates its output on every call.
        t["out"] = self.wrapper.run(t["q"], (t["k_nhd"], t["v_nhd"]))


def _request_kv(p: Params) -> list[Any]:
    from kvkernels.reference.attention import RequestKV

    tree = tree_of(p)
    return [
        RequestKV([], 0, blocks, n)
        for blocks, n in zip(tree.request_blocks(), tree.request_lens(), strict=True)
    ]


def torch_per_request(p: Params) -> Any:
    def run(t: Tensors) -> None:
        from kvkernels.reference.attention import attention_ref_low

        out, _ = attention_ref_low(
            t["q"], t["k"], t["v"], _request_kv(p), query_tokens=int(p["query_tokens"])
        )
        t["out"].copy_(out)

    return run


def reference_fp32(p: Params) -> Any:
    def run(t: Tensors) -> Tensors:
        from kvkernels.reference.attention import attention_ref32

        out, _ = attention_ref32(
            t["q"], t["k"], t["v"], _request_kv(p), query_tokens=int(p["query_tokens"])
        )
        return {"out": out}

    return run


# --- cases --------------------------------------------------------------------------


def make_case(p: Params, *, with_flashinfer: bool, device: str = "cuda") -> Case:
    objects: dict[str, Any] = {}
    if with_flashinfer:
        objects["flashinfer_per_request"] = FlashInferPerRequest(p)
    else:
        objects["torch_per_request"] = torch_per_request(p)
    objects["kvkernels_tree"] = KvkernelsTree(p, max_levels=4)
    objects["kvkernels_two_level"] = KvkernelsTree(p, max_levels=1)
    if with_flashinfer:
        objects["flashinfer_multilevel"] = FlashInferMultiLevel(p)
    impls, plans = {}, {}
    for name, obj in objects.items():
        if hasattr(obj, "run"):
            impls[name] = obj.run
            plans[name] = obj.plan
        else:
            impls[name] = obj
    return Case(
        params=p,
        setup=lambda: make_inputs(p, nhd_copies=with_flashinfer, device=device),
        impls=impls,
        outputs=("out",),
        check="tolerance",
        bytes_moved=bytes_moved,
        flops=flops,
        reference_fp32=reference_fp32(p),
        plans=plans,
    )


def cases() -> Iterator[Case]:
    from bench.suites.cascade import flashinfer_runs

    with_flashinfer = flashinfer_runs()
    if not with_flashinfer:
        print("cascade_tree: FlashInfer is not available; torch_per_request is the reference")
    for p in grid():
        yield make_case(p, with_flashinfer=with_flashinfer)
