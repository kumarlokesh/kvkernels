"""Offloading one request's KV, every layer, from the GPU cache to pinned host memory.

Each case builds a 32-layer Llama 3.1 8B cache (8 KV heads, head_dim 128, fp16)
with kvkernels.layouts and copies one request's KV blocks of every plane into
one pinned host buffer, rows of plane p at host[p]:

    torch_index_select_then_copy   per plane, index_select into a GPU staging
                                   buffer; then one copy of the buffer to the host
                                   (reference)
    kvkernels_staging              one plane-set gather into the staging buffer;
                                   then one copy to the host
    kvkernels_staging_chunked      four plane sets of 8 layers; chunk i is copied to
                                   the host on a second stream while chunk i+1 is
                                   gathered
    kvkernels_to_host              one plane-set gather whose row buffers are the
                                   host buffer: the kernel's stores cross PCIe
    vllm_batched_dma               vLLM v0.33.0's offload: one cuMemcpyBatchAsync
                                   call, one job per KV block per plane, from the
                                   cache straight to the host buffer, on a side
                                   stream (vLLM layout only; needs a driver for
                                   CUDA 12.8 or later)
    copy_ceiling                   one copy to the host of a GPU buffer that already
                                   holds the same bytes: the PCIe bandwidth the
                                   others are compared with

Plane sets and job lists are built in the plan step, outside the timed call.
bytes_moved counts the bytes that cross PCIe once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from bench.harness import Case, Tensors
from bench.schema import Params

LAYERS = 32
HEADS, HEAD_DIM = 8, 128
CHUNKS = 4
TOKENS = (1024, 4096, 16384)
LAYOUTS = {"vllm_LBNHC": 16, "sglang_nhd": 1}  # layout: tokens per KV block


def planes_per_layer(layout: str) -> int:
    return 2 if layout.startswith("sglang") else 1


def block_bytes(p: Params) -> int:
    content = HEAD_DIM if str(p["layout"]).startswith("sglang") else 2 * HEAD_DIM
    return int(p["block"]) * HEADS * content * 2  # fp16


def num_ids(p: Params) -> int:
    return int(p["tokens"]) // int(p["block"])


def bytes_moved(p: Params) -> int:
    return num_ids(p) * block_bytes(p) * LAYERS * planes_per_layer(str(p["layout"]))


def make_inputs(
    params: Params, *, tiny_blocks: bool = False, filled_staging: bool = False
) -> Tensors:
    """The cache's planes, the request's KV block ids, a GPU staging buffer, a pinned host buffer.

    tiny_blocks keeps the planes and ids but shrinks every block to 16 bytes, for
    measuring the CPU cost of a call. filled_staging fills the staging buffer with
    the request's blocks, the source copy_ceiling copies.
    """
    import torch

    from kvkernels.layouts import LAYOUTS as ALL_LAYOUTS
    from kvkernels.layouts import CacheSpec, make_cache, planes

    name = str(params["layout"])
    n = num_ids(params)
    blocks = 2 * n
    if tiny_blocks:
        block, heads, head_dim = 1, 1, 4
    else:
        block, heads, head_dim = int(params["block"]), HEADS, HEAD_DIM
    layout = ALL_LAYOUTS[name]
    spec = CacheSpec(LAYERS, blocks, block, heads, head_dim, torch.float16)
    cache = make_cache(layout, spec, "cuda")
    all_planes = [plane for layer in cache for plane in planes(layout, spec, layer)]
    gen = torch.Generator(device="cuda").manual_seed(0)
    for tensor in {id(t): t for layer in cache for t in layer}.values():
        tensor.view(torch.uint8).random_(0, 256, generator=gen)

    ids = torch.randperm(blocks, generator=torch.Generator().manual_seed(1))[:n].to("cuda")
    shape = (len(all_planes), n, *all_planes[0].shape[1:])
    staging = torch.empty(shape, dtype=torch.float16, device="cuda")
    if filled_staging:
        for plane, rows in zip(all_planes, staging, strict=True):
            rows.copy_(plane[ids])
    host = torch.zeros(shape, dtype=torch.float16).pin_memory()
    inputs: dict[str, Any] = {
        "planes": all_planes,
        "ids": ids.to(torch.int32),
        "ids_long": ids,
        "staging": staging,
        "host": host,
    }
    return inputs


# --- implementations -------------------------------------------------------


def torch_index_select_then_copy(t: Tensors) -> None:
    import torch

    planes: Any = t["planes"]
    for plane, rows in zip(planes, t["staging"], strict=True):
        torch.index_select(
            plane.reshape(plane.shape[0], -1), 0, t["ids_long"], out=rows.view(rows.shape[0], -1)
        )
    t["host"].copy_(t["staging"], non_blocking=True)


def copy_ceiling(t: Tensors) -> None:
    t["host"].copy_(t["staging"], non_blocking=True)


class KvkernelsStaging:
    def plan(self, t: Tensors) -> None:
        from kvkernels.ops.plane_set import PlaneSet

        planes: Any = t["planes"]
        t["plane_set"] = PlaneSet(planes, list(t["staging"]))  # type: ignore[assignment]

    def run(self, t: Tensors) -> None:
        plane_set: Any = t["plane_set"]
        plane_set.gather(t["ids"])
        t["host"].copy_(t["staging"], non_blocking=True)


class KvkernelsStagingChunked:
    """CHUNKS plane sets; each chunk's copy to the host overlaps the next chunk's gather."""

    def __init__(self) -> None:
        self.side: Any = None

    def plan(self, t: Tensors) -> None:
        import torch

        from kvkernels.ops.plane_set import PlaneSet

        planes: Any = t["planes"]
        per_chunk = len(planes) // CHUNKS
        bounds = [(c * per_chunk, (c + 1) * per_chunk) for c in range(CHUNKS)]
        t["chunks"] = [  # type: ignore[assignment]
            (PlaneSet(planes[a:b], list(t["staging"][a:b])), a, b) for a, b in bounds
        ]
        if self.side is None:
            self.side = torch.cuda.Stream()  # type: ignore[no-untyped-call]

    def run(self, t: Tensors) -> None:
        import torch

        current = torch.cuda.current_stream()
        chunks: Any = t["chunks"]
        for plane_set, a, b in chunks:
            plane_set.gather(t["ids"])
            self.side.wait_stream(current)
            with torch.cuda.stream(self.side):
                t["host"][a:b].copy_(t["staging"][a:b], non_blocking=True)
        current.wait_stream(self.side)


class KvkernelsToHost:
    def plan(self, t: Tensors) -> None:
        from kvkernels.ops.plane_set import PlaneSet

        planes: Any = t["planes"]
        t["plane_set"] = PlaneSet(planes, list(t["host"]))  # type: ignore[assignment]

    def run(self, t: Tensors) -> None:
        plane_set: Any = t["plane_set"]
        plane_set.gather(t["ids"])


class VllmBatchedDma:
    """vLLM v0.33.0's GPU-to-CPU offload: one batched copy, one job per KV block per plane."""

    def __init__(self) -> None:
        self.side: Any = None

    def plan(self, t: Tensors) -> None:
        import torch

        planes: Any = t["planes"]
        ids = t["ids_long"].cpu()
        srcs, dsts = [], []
        for plane, rows in zip(planes, t["host"], strict=True):
            size = rows[0].numel() * rows.element_size()
            stride = plane.stride(0) * plane.element_size()
            srcs.append(plane.data_ptr() + ids * stride)
            dsts.append(rows.data_ptr() + torch.arange(ids.numel()) * size)
        size = t["host"][0][0].numel() * t["host"].element_size()
        src = torch.cat(srcs)
        t["dma_jobs"] = (torch.cat(dsts), src, torch.full_like(src, size))  # type: ignore[assignment]
        if self.side is None:
            self.side = torch.cuda.Stream()  # type: ignore[no-untyped-call]

    def run(self, t: Tensors) -> None:
        import torch

        from bench.baselines.vllm.batched_dma import copy_batch

        dst, src, sizes = t["dma_jobs"]
        current = torch.cuda.current_stream()
        self.side.wait_stream(current)
        copy_batch(dst, src, sizes, self.side.cuda_stream, src_access_any=False)
        current.wait_stream(self.side)


# --- cases ------------------------------------------------------------------


def make_case(params: Params, *, with_dma: bool) -> Case:
    staging, chunked, to_host = KvkernelsStaging(), KvkernelsStagingChunked(), KvkernelsToHost()
    impls: dict[str, Callable[[Tensors], None]] = {
        "torch_index_select_then_copy": torch_index_select_then_copy,
        "kvkernels_staging": staging.run,
        "kvkernels_staging_chunked": chunked.run,
        "kvkernels_to_host": to_host.run,
        "copy_ceiling": copy_ceiling,
    }
    plans: dict[str, Callable[[Tensors], None]] = {
        "kvkernels_staging": staging.plan,
        "kvkernels_staging_chunked": chunked.plan,
        "kvkernels_to_host": to_host.plan,
    }
    if with_dma and str(params["layout"]).startswith("vllm"):
        dma = VllmBatchedDma()
        impls["vllm_batched_dma"] = dma.run
        plans["vllm_batched_dma"] = dma.plan
    return Case(
        params=params,
        setup=lambda: make_inputs(params),
        impls=impls,
        outputs=("host",),
        check="exact",
        bytes_moved=bytes_moved,
        host_setup=lambda: make_inputs(params, tiny_blocks=True, filled_staging=True),
        plans=plans,
        setups={"copy_ceiling": lambda: make_inputs(params, filled_staging=True)},
    )


def grid() -> Iterator[Params]:
    for layout, block in LAYOUTS.items():
        for tokens in TOKENS:
            yield {
                "op": "offload",
                "layout": layout,
                "layers": LAYERS,
                "block": block,
                "kv_heads": HEADS,
                "head_dim": HEAD_DIM,
                "tokens": tokens,
                "dtype": "float16",
            }


def cases() -> Iterator[Case]:
    from bench.baselines.vllm.batched_dma import available

    with_dma = available()
    if not with_dma:
        print("kv_offload: the driver has no cuMemcpyBatchAsync (CUDA 12.8); no vLLM baseline")
    for params in grid():
        yield make_case(params, with_dma=with_dma)
