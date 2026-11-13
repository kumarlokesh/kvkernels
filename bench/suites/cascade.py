"""Shared-prefix decode attention: the cascade against per-request decode and FlashInfer.

One decode step of one layer. Requests come in groups; each group shares a
prefix, and each request has its own suffix after it:

    group 0: requests 0 … B/G-1     prefix (P tokens, its own KV blocks), then a suffix each
    group 1: …

Every request's prefix blocks and suffix blocks are distinct physical KV blocks,
drawn in a random order from one layer's cache, so neighbouring tokens across a
block boundary are not neighbours in memory.

Cases (RFC grid):

    main      batch 1 … 128, prefix 512 … 32,768, suffix 0 … 2,048, one group,
              Llama 3.1 8B and Llama 3.2 1B attention shapes, vLLM's default layout
              with 16-token KV blocks, fp16
    groups    4 groups of 16 at batch 64
    GQA       group sizes 1 and 8 at batch 32
    layout    SGLang's default (NHD, one-token pages) at batch 32
    bf16      one shape
    threshold batch 2 and 8, prefix 16 … 256, suffix 512: the shared length at
              which a group starts to beat reading the prefix once per request

Implementations, each checked against the first within the fp32-reference
tolerance rule (output only; the LSE is checked in the op's tests):

    flashinfer_decode       FlashInfer batch decode, each request over its prefix
                            and suffix; first, and so the reference, when FlashInfer
                            runs on this machine
    torch_per_request       PyTorch SDPA per request; first only when FlashInfer does not run
    kvkernels               the cascade, groups named on the host
    kvkernels_gpu_plan      the cascade planned on the GPU from the block table (or
                            slot list) each step; planning is inside the timed call
    kvkernels_per_request   the same op with every request its own group: per-request
                            decode with kvkernels' kernel, separating the cascade's
                            effect from the kernel's quality
    flashinfer_decode_tc    FlashInfer batch decode with its tensor-core kernel
    flashinfer_cascade      FlashInfer's two-level cascade; only when every suffix is
                            non-empty, since each request needs a KV page at the second level
    vllm_triton_decode      vLLM's Triton attention kernel, copied, on the engine layout
    kvkernels_split…,       batch 1 and 4 only: the cascade with other split-K
    kvkernels_pps…          settings (minimum range length; programs per SM)
    kvkernels_gpu_grouped,  threshold cases only: planned on the GPU with every
    kvkernels_gpu_ungrouped run of shared KV blocks a group, and with none

FlashInfer runs on a copy of K and V in its own layout (NHD, contiguous), with
the same values. Planning (kvkernels' plan update, FlashInfer's plan, vLLM's
block table and scratch) is excluded from timing and recorded as plan_us; for
kvkernels_gpu_plan, plan_us is building the index an engine already has.

bytes_moved and flops follow the roofline model of a cascade: each group's prefix
read once, each suffix read once, every query head's two matrix products over
its prefix and suffix. The same numbers apply to every implementation, so a
per-request baseline's GB/s is the traffic the cascade needs divided by the
baseline's time.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bench.harness import Case, Tensors
from bench.schema import Params
from kvkernels.roofline import decode_attention_cost, shared_prefix_attention_cost

if TYPE_CHECKING:
    import torch

    from kvkernels.ops.cascade import CascadePlan, PrefixGroup

MODELS = {"llama-3.1-8b": (32, 8, 128), "llama-3.2-1b": (32, 8, 64)}
BATCHES = (1, 4, 16, 32, 64, 128)
PREFIXES = (512, 2048, 8192, 32768)
SUFFIXES = (0, 128, 512, 2048)
BLOCK_SIZE = 16
DEFAULT_LAYOUT = "vllm_LBNHC"
THRESHOLD_PREFIXES = (16, 32, 64, 128, 256)
NEVER_GROUP = 1 << 30
# The split-K settings timed at batch 1 and 4: (minimum range length, programs per SM).
SPLIT_SETTINGS = {
    "kvkernels_split256": (256, 2),
    "kvkernels_split1024": (1024, 2),
    "kvkernels_pps1": (512, 1),
    "kvkernels_pps4": (512, 4),
}
FLASHINFER_WORKSPACE_BYTES = 128 << 20  # FlashInfer's documented recommendation


def _params(
    *,
    batch: int,
    prefix: int,
    suffix: int,
    model: str = "llama-3.1-8b",
    groups: int = 1,
    kv_heads: int | None = None,
    layout: str = DEFAULT_LAYOUT,
    block: int = BLOCK_SIZE,
    dtype: str = "float16",
) -> Params:
    q_heads, model_kv_heads, head_dim = MODELS[model]
    return {
        "model": model,
        "batch": batch,
        "groups": groups,
        "prefix_len": prefix,
        "suffix_len": suffix,
        "q_heads": q_heads,
        "kv_heads": kv_heads or model_kv_heads,
        "head_dim": head_dim,
        "layout": layout,
        "block": block,
        "dtype": dtype,
    }


def grid() -> Iterator[Params]:
    for model in MODELS:
        for batch in BATCHES:
            for prefix in PREFIXES:
                for suffix in SUFFIXES:
                    yield _params(batch=batch, prefix=prefix, suffix=suffix, model=model)
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            yield _params(batch=64, prefix=prefix, suffix=suffix, groups=4)
            for kv_heads in (32, 4):  # GQA groups of 1 and 8
                yield _params(batch=32, prefix=prefix, suffix=suffix, kv_heads=kv_heads)
            yield _params(batch=32, prefix=prefix, suffix=suffix, layout="sglang_nhd", block=1)
    yield _params(batch=32, prefix=8192, suffix=512, dtype="bfloat16")
    for batch in (2, 8):
        for prefix in THRESHOLD_PREFIXES:
            yield _params(batch=batch, prefix=prefix, suffix=512)


def is_threshold_sweep(p: Params) -> bool:
    return int(p["prefix_len"]) in THRESHOLD_PREFIXES and int(p["batch"]) in (2, 8)


def is_split_sweep(p: Params) -> bool:
    return (
        int(p["batch"]) <= 4
        and p["groups"] == 1
        and p["layout"] == DEFAULT_LAYOUT
        and p["dtype"] == "float16"
    )


# --- the cost model -------------------------------------------------------------


def _element_size(dtype: str) -> int:
    return {"float16": 2, "bfloat16": 2}[dtype]


def _cost(p: Params) -> tuple[int, int]:
    batch, groups = int(p["batch"]), int(p["groups"])
    shape = (int(p["q_heads"]), int(p["kv_heads"]))
    head_dim, b = int(p["head_dim"]), _element_size(str(p["dtype"]))
    pf, pb = shared_prefix_attention_cost(
        batch // groups, *shape, int(p["prefix_len"]), head_dim, b
    )
    sf, sb = decode_attention_cost(batch, *shape, int(p["suffix_len"]), head_dim, b)
    return groups * pf + sf, groups * pb + sb


def bytes_moved(p: Params) -> int:
    return _cost(p)[1]


def flops(p: Params) -> int:
    return _cost(p)[0]


# --- inputs ---------------------------------------------------------------------


def make_inputs(
    p: Params, *, small: bool = False, nhd_copies: bool = False, device: str = "cuda"
) -> Tensors:
    """Deterministic inputs for one case.

    The batch's structure is kept on the host, as an engine keeps it, in int64
    CPU tensors: group_starts (first request of each group, then the batch size),
    prefix_lens and prefix_table per group, suffix_lens and suffix_table per
    request, tables padded with 0. small keeps the batch and the shape of the
    model but shortens every prefix to two KV blocks and every suffix to at most
    one, for measuring the CPU cost of a call. nhd_copies adds FlashInfer's
    copies of K and V, k_nhd and v_nhd.
    """
    import torch

    from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache

    batch, groups, block = int(p["batch"]), int(p["groups"]), int(p["block"])
    prefix, suffix = int(p["prefix_len"]), int(p["suffix_len"])
    if small:
        prefix, suffix = 2 * block, min(suffix, block)
    q_heads, kv_heads, head_dim = (int(p[k]) for k in ("q_heads", "kv_heads", "head_dim"))
    dtype = getattr(torch, str(p["dtype"]))

    prefix_blocks, suffix_blocks = math.ceil(prefix / block), math.ceil(suffix / block)
    needed = groups * prefix_blocks + batch * suffix_blocks
    order = (torch.randperm(needed, generator=torch.Generator().manual_seed(1)) + 1).tolist()
    prefix_table = torch.tensor(order[: groups * prefix_blocks]).view(groups, prefix_blocks)
    suffix_table = torch.tensor(order[groups * prefix_blocks :], dtype=torch.int64).view(
        batch, suffix_blocks
    )

    layout = LAYOUTS[str(p["layout"])]
    spec = CacheSpec(1, needed + 1, block, kv_heads, head_dim, dtype)  # block 0 reserved
    k, v = kv_views(layout, spec, make_cache(layout, spec, device)[0])
    gen = torch.Generator(device=device).manual_seed(0)
    for view in (k, v):
        view.copy_(torch.randn(tuple(view.shape), generator=gen, device=device, dtype=dtype))
    q = torch.randn((batch, q_heads, head_dim), generator=gen, device=device, dtype=dtype)

    inputs: Tensors = {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.zeros_like(q),
        "group_starts": torch.arange(0, batch + 1, batch // groups),
        "prefix_lens": torch.full((groups,), prefix),
        "prefix_table": prefix_table,
        "suffix_lens": torch.full((batch,), suffix),
        "suffix_table": suffix_table,
    }
    if nhd_copies:
        inputs["k_nhd"] = k.contiguous()
        inputs["v_nhd"] = v.contiguous()
    return inputs


@dataclass(frozen=True)
class Batch:
    """The host-side description of the batch in a case's inputs."""

    group_starts: list[int]
    prefix_lens: list[int]
    prefix_blocks: list[list[int]]
    suffix_lens: list[int]
    suffix_blocks: list[list[int]]
    block_size: int

    @property
    def size(self) -> int:
        return len(self.suffix_lens)

    def group_of(self, request: int) -> int:
        return next(g for g in range(len(self.prefix_lens)) if request < self.group_starts[g + 1])

    def prefix_groups(self) -> list[PrefixGroup]:
        from kvkernels.ops.cascade import PrefixGroup

        return [
            PrefixGroup(range(start, stop), blocks, length)
            for start, stop, blocks, length in zip(
                self.group_starts[:-1],
                self.group_starts[1:],
                self.prefix_blocks,
                self.prefix_lens,
                strict=True,
            )
        ]

    def per_request_groups(self) -> list[PrefixGroup]:
        """Every request its own group, over its group's prefix."""
        from kvkernels.ops.cascade import PrefixGroup

        groups = []
        for r in range(self.size):
            g = self.group_of(r)
            groups.append(PrefixGroup(range(r, r + 1), self.prefix_blocks[g], self.prefix_lens[g]))
        return groups

    def request_blocks(self) -> list[list[int]]:
        """Each request's prefix blocks followed by its suffix blocks.

        A request's tokens are contiguous in this list only when its prefix fills
        whole KV blocks, which every case in the grid does.
        """
        out = []
        for r in range(self.size):
            g = self.group_of(r)
            if self.prefix_lens[g] % self.block_size:
                raise ValueError("per-request block lists need block-aligned prefixes")
            out.append(self.prefix_blocks[g] + self.suffix_blocks[r])
        return out

    def request_lens(self) -> list[int]:
        return [self.prefix_lens[self.group_of(r)] + self.suffix_lens[r] for r in range(self.size)]


def batch_of(t: Tensors) -> Batch:
    return Batch(
        group_starts=t["group_starts"].tolist(),
        prefix_lens=t["prefix_lens"].tolist(),
        prefix_blocks=t["prefix_table"].tolist(),
        suffix_lens=t["suffix_lens"].tolist(),
        suffix_blocks=t["suffix_table"].tolist(),
        block_size=int(t["k"].shape[1]),
    )


def request_kv(t: Tensors) -> list[Any]:
    from kvkernels.reference.attention import RequestKV

    b = batch_of(t)
    out = []
    for r in range(b.size):
        g = b.group_of(r)
        out.append(
            RequestKV(b.prefix_blocks[g], b.prefix_lens[g], b.suffix_blocks[r], b.suffix_lens[r])
        )
    return out


def _sm_scale(t: Tensors) -> float:
    return 1.0 / math.sqrt(t["q"].shape[-1])


# --- implementations: each a plan (host, once per step) and a run ---------------


class Kvkernels:
    def __init__(
        self, *, per_request: bool = False, min_split_tokens: int = 512, programs_per_sm: int = 2
    ) -> None:
        self.per_request = per_request
        self.settings = {"min_split_tokens": min_split_tokens, "programs_per_sm": programs_per_sm}
        self.cascade: CascadePlan | None = None
        self.key: tuple[Any, ...] = ()

    def plan(self, t: Tensors) -> None:
        import torch

        from kvkernels.ops.cascade import MAX_SPLITS, CascadePlan

        b = batch_of(t)
        groups = b.per_request_groups() if self.per_request else b.prefix_groups()
        q, k = t["q"], t["k"]
        key = (tuple(q.shape), tuple(k.shape), len(groups), q.device)
        if self.cascade is None or key != self.key:
            rows = sum(len(g.requests) for g in groups) * (q.shape[1] // k.shape[2])
            self.cascade = CascadePlan(
                max_requests=b.size,
                max_groups=len(groups),
                max_blocks=sum(len(g.blocks) for g in groups) + sum(map(len, b.suffix_blocks)),
                num_q_heads=q.shape[1],
                num_kv_heads=k.shape[2],
                head_dim=q.shape[2],
                block_size=k.shape[1],
                num_blocks=k.shape[0],
                device=torch.device(q.device),
                max_tasks=MAX_SPLITS * (len(groups) + math.ceil(rows / 64) + b.size),
                min_split_tokens=self.settings["min_split_tokens"],
                programs_per_sm=self.settings["programs_per_sm"],
            )
            self.key = key
        self.cascade.update(groups=groups, suffix_blocks=b.suffix_blocks, suffix_lens=b.suffix_lens)

    def run(self, t: Tensors) -> None:
        from kvkernels.ops.cascade import cascade_decode_attention

        assert self.cascade is not None
        cascade_decode_attention(t["q"], t["k"], t["v"], self.cascade, t["out"])


class KvkernelsGpuPlan:
    """Plans every step on the GPU from the index, inside the timed call."""

    def __init__(self, *, layout: str, min_shared_tokens: int = 64) -> None:
        self.layout = layout
        self.min_shared_tokens = min_shared_tokens
        self.cascade: CascadePlan | None = None
        self.index: dict[str, torch.Tensor] = {}

    def plan(self, t: Tensors) -> None:
        import torch

        from kvkernels.layouts import LAYOUTS, CacheSpec, make_index
        from kvkernels.ops.cascade import CascadePlan

        b = batch_of(t)
        q, k = t["q"], t["k"]
        lens = b.request_lens()
        spec = CacheSpec(1, k.shape[0], k.shape[1], k.shape[2], k.shape[3], k.dtype)
        self.index = make_index(LAYOUTS[self.layout], spec, b.request_blocks(), lens, q.device)
        self.index["seq_lens"] = torch.tensor(lens, dtype=torch.int32, device=q.device)
        self.cascade = CascadePlan(
            max_requests=b.size,
            max_groups=max(b.size // 2, 1),
            max_blocks=1,
            num_q_heads=q.shape[1],
            num_kv_heads=k.shape[2],
            head_dim=q.shape[2],
            block_size=k.shape[1],
            num_blocks=k.shape[0],
            device=torch.device(q.device),
            min_shared_tokens=self.min_shared_tokens,
        )

    def run(self, t: Tensors) -> None:
        from kvkernels.ops.cascade import cascade_decode_attention

        assert self.cascade is not None
        if "block_table" in self.index:
            self.cascade.plan_from_block_table(self.index["block_table"], self.index["seq_lens"])
        else:
            self.cascade.plan_from_slots(
                self.index["kv_indptr"], self.index["kv_indices"], self.index["seq_lens"]
            )
        cascade_decode_attention(t["q"], t["k"], t["v"], self.cascade, t["out"])


def torch_per_request(t: Tensors) -> None:
    from kvkernels.reference.attention import attention_ref_low

    out, _ = attention_ref_low(t["q"], t["k"], t["v"], request_kv(t))
    t["out"].copy_(out)


def _paged(
    block_lists: Sequence[Sequence[int]], lens: Sequence[int], page: int, device: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's paged-KV index: page indptr, page indices, tokens in each last page."""
    import torch

    indptr = [0]
    for blocks in block_lists:
        indptr.append(indptr[-1] + len(blocks))
    flat = [b for blocks in block_lists for b in blocks]
    last = [(n - 1) % page + 1 if n else 0 for n in lens]
    return (
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(flat, dtype=torch.int32, device=device),
        torch.tensor(last, dtype=torch.int32, device=device),
    )


class FlashInferDecode:
    def __init__(self, *, tensor_cores: bool) -> None:
        self.tensor_cores = tensor_cores
        self.wrapper: Any = None

    def plan(self, t: Tensors) -> None:
        import torch
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        q, k = t["q"], t["k"]
        if self.wrapper is None:
            workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
            self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=self.tensor_cores
            )
        b = batch_of(t)
        indptr, indices, last = _paged(b.request_blocks(), b.request_lens(), b.block_size, q.device)
        self.wrapper.plan(
            indptr,
            indices,
            last,
            q.shape[1],
            k.shape[2],
            q.shape[2],
            b.block_size,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            sm_scale=_sm_scale(t),
        )

    def run(self, t: Tensors) -> None:
        self.wrapper.run(t["q"], (t["k_nhd"], t["v_nhd"]), out=t["out"])


class FlashInferCascade:
    def __init__(self) -> None:
        self.wrapper: Any = None

    def plan(self, t: Tensors) -> None:
        import torch
        from flashinfer import MultiLevelCascadeAttentionWrapper

        q, k = t["q"], t["k"]
        if self.wrapper is None:
            workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
            self.wrapper = MultiLevelCascadeAttentionWrapper(2, workspace, "NHD")
        b = batch_of(t)
        top = _paged(b.prefix_blocks, b.prefix_lens, b.block_size, q.device)
        bottom = _paged(b.suffix_blocks, b.suffix_lens, b.block_size, q.device)
        qo = [
            torch.tensor(b.group_starts, dtype=torch.int32, device=q.device),
            torch.arange(b.size + 1, dtype=torch.int32, device=q.device),
        ]
        self.wrapper.plan(
            qo,
            [top[0], bottom[0]],
            [top[1], bottom[1]],
            [top[2], bottom[2]],
            q.shape[1],
            k.shape[2],
            q.shape[2],
            b.block_size,
            sm_scale=_sm_scale(t),
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )

    def run(self, t: Tensors) -> None:
        # The wrapper allocates its output on every call.
        t["out"] = self.wrapper.run(t["q"], (t["k_nhd"], t["v_nhd"]))


class VllmTritonDecode:
    def __init__(self) -> None:
        self.step: Any = None

    def plan(self, t: Tensors) -> None:
        import torch

        from bench.baselines.vllm.unified_attention_adapter import plan_decode

        b = batch_of(t)
        q, k = t["q"], t["k"]
        self.step = plan_decode(
            b.request_blocks(),
            b.request_lens(),
            q.shape[1],
            k.shape[2],
            q.shape[2],
            torch.device(q.device),
        )

    def run(self, t: Tensors) -> None:
        from bench.baselines.vllm.unified_attention_adapter import decode

        decode(t["q"], t["k"], t["v"], t["out"], self.step, _sm_scale(t))


@functools.cache
def flashinfer_runs() -> bool:
    """True when FlashInfer imports and compiles its kernels on this machine.

    FlashInfer compiles on first use and needs the CUDA compiler for that; a
    machine without it, or without a GPU, leaves FlashInfer out.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    p = _params(batch=2, prefix=32, suffix=16, model="llama-3.2-1b", block=16)
    try:
        inputs = make_inputs(p, nhd_copies=True)
        for impl in (FlashInferDecode(tensor_cores=False), FlashInferCascade()):
            impl.plan(inputs)
            impl.run(inputs)
        torch.cuda.synchronize()
    except Exception as exc:  # compiler missing, unsupported GPU, …
        print(f"cascade: FlashInfer does not run here ({type(exc).__name__}: {exc}); left out")
        return False
    return True


# --- cases -----------------------------------------------------------------------


def reference_fp32(t: Tensors) -> Tensors:
    from kvkernels.reference.attention import attention_ref32

    out, _ = attention_ref32(t["q"], t["k"], t["v"], request_kv(t))
    return {"out": out}


def implementations(p: Params, *, with_flashinfer: bool) -> dict[str, Any]:
    """Name to implementation object (or plain function), in check order."""
    impls: dict[str, Any] = {}
    if with_flashinfer:
        impls["flashinfer_decode"] = FlashInferDecode(tensor_cores=False)
    else:
        impls["torch_per_request"] = torch_per_request
    impls["kvkernels"] = Kvkernels()
    layout = str(p["layout"])
    impls["kvkernels_gpu_plan"] = KvkernelsGpuPlan(layout=layout)
    impls["kvkernels_per_request"] = Kvkernels(per_request=True)
    if with_flashinfer:
        impls["flashinfer_decode_tc"] = FlashInferDecode(tensor_cores=True)
        if int(p["suffix_len"]) > 0:
            impls["flashinfer_cascade"] = FlashInferCascade()
    impls["vllm_triton_decode"] = VllmTritonDecode()
    if is_split_sweep(p):
        for name, (min_split, per_sm) in SPLIT_SETTINGS.items():
            impls[name] = Kvkernels(min_split_tokens=min_split, programs_per_sm=per_sm)
    if is_threshold_sweep(p):
        impls["kvkernels_gpu_grouped"] = KvkernelsGpuPlan(layout=layout, min_shared_tokens=1)
        impls["kvkernels_gpu_ungrouped"] = KvkernelsGpuPlan(
            layout=layout, min_shared_tokens=NEVER_GROUP
        )
    return impls


def make_case(p: Params, *, with_flashinfer: bool, device: str = "cuda") -> Case:
    objects = implementations(p, with_flashinfer=with_flashinfer)
    impls: dict[str, Callable[[Tensors], None]] = {}
    plans: dict[str, Callable[[Tensors], None]] = {}
    for name, obj in objects.items():
        if callable(obj) and not hasattr(obj, "run"):
            impls[name] = obj
        else:
            impls[name] = obj.run
            plans[name] = obj.plan
    return Case(
        params=p,
        setup=lambda: make_inputs(p, nhd_copies=with_flashinfer, device=device),
        impls=impls,
        outputs=("out",),
        check="tolerance",
        bytes_moved=bytes_moved,
        flops=flops,
        reference_fp32=reference_fp32,
        host_setup=lambda: make_inputs(p, small=True, nhd_copies=with_flashinfer, device=device),
        plans=plans,
    )


def cases() -> Iterator[Case]:
    with_flashinfer = flashinfer_runs()
    if not with_flashinfer:
        print("cascade: FlashInfer is not available; torch_per_request is the reference")
    for p in grid():
        yield make_case(p, with_flashinfer=with_flashinfer)
