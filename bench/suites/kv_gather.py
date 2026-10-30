"""Gather and scatter of KV blocks: kvkernels against PyTorch, vLLM's copy kernel and DMA.

Each case moves one plane, the unit the op launches on, from a cache built by
kvkernels.layouts the way the engine allocates it:

    vllm_LBNHC   vLLM's default: one layer's KV, K and V together, each block one run
                 [blocks, block_size, heads, 2·head_dim]
    vllm_BLNHC   layer 1 of a cache that stores the 32 layers of a block together;
                 consecutive blocks of one layer are 32 layers' worth of bytes apart
    sglang_nhd   SGLang's default: the K buffer of one layer viewed as pages
                 [pages, page_size, heads, head_dim]; V is a second, identical plane

Implementations, all checked bit for bit against the first:

    gather:  torch_index_select (reference), kvkernels, torch_advanced_indexing,
             copy_ceiling (a plain copy of the same bytes, no indexing),
             vllm_swap_blocks, cuda_batched_dma, cuda_batched_dma_build_jobs
    scatter: torch_index_copy (reference), kvkernels, vllm_swap_blocks,
             cuda_batched_dma, cuda_batched_dma_build_jobs

vllm_swap_blocks runs only for blocks whose size and stride are multiples of 8
bytes, the condition vLLM itself uses it under. The two batched-DMA entries need
NVIDIA's cuda-bindings and a driver for CUDA 12.8 or later. Both copy the same
job list in one cuMemcpyBatchAsync call; cuda_batched_dma builds the list once per
set of inputs, cuda_batched_dma_build_jobs rebuilds it on every call, so the
difference in host_us is the cost of building it on the host.

Every plane spans at least 640 MiB, eight times the largest L2 among the target
GPUs, so most blocks are read from device memory. Cases whose cache would exceed
8 GiB (all layers for vllm_BLNHC, K and V for sglang_nhd) are left out.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

from bench.harness import Case, Tensors
from bench.schema import Params

MiB = 1 << 20
GiB = 1 << 30
MIN_CACHE_BYTES = 640 * MiB
MAX_CACHE_BYTES = 8 * GiB
ALL_LAYERS = 32

NUM_IDS = (64, 512, 4096, 32768)
ORDERS = ("random", "sorted")
HEADS_AND_DIMS = ((8, 128), (8, 64), (1, 128))
BLOCK_SIZES = {"vllm_LBNHC": (16, 32), "vllm_BLNHC": (16, 32), "sglang_nhd": (1, 16)}


def block_shape(layout: str, block: int, heads: int, head_dim: int) -> tuple[int, ...]:
    if layout.startswith("sglang"):
        return (block, heads, head_dim)
    return (block, heads, 2 * head_dim)


def layers_per_block(layout: str) -> int:
    return ALL_LAYERS if layout == "vllm_BLNHC" else 1


def num_blocks(layout: str, block_bytes: int, num_ids: int) -> int:
    per_block = block_bytes * layers_per_block(layout)
    return max(num_ids, math.ceil(MIN_CACHE_BYTES / per_block))


def cache_bytes(layout: str, block_bytes: int, num_ids: int) -> int:
    """Bytes the cache allocates: every layer of it, and both K and V for SGLang."""
    buffers = 2 if layout.startswith("sglang") else 1
    blocks = num_blocks(layout, block_bytes, num_ids)
    return blocks * block_bytes * layers_per_block(layout) * buffers


def _dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


def _element_size(dtype: str) -> int:
    return {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]


def make_inputs(params: Params, *, tiny_blocks: bool = False, device: str = "cuda") -> Tensors:
    """Deterministic inputs: the same params always give the same bytes and ids.

    tiny_blocks keeps the number of ids but shrinks every block to 32 bytes, for
    measuring the host cost of a call without waiting on the GPU.
    """
    import torch

    from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache, planes

    name = str(params["layout"])
    dtype = _dtype(str(params["dtype"]))
    n = int(params["num_ids"])
    layers = layers_per_block(name)
    if tiny_blocks:
        block, heads, head_dim = 1, 1, 8
        blocks = max(n, 2)
    else:
        block, heads, head_dim = (int(params[k]) for k in ("block", "kv_heads", "head_dim"))
        shape = block_shape(name, block, heads, head_dim)
        blocks = num_blocks(name, math.prod(shape) * _element_size(str(params["dtype"])), n)

    layout = LAYOUTS[name]
    spec = CacheSpec(layers, blocks, block, heads, head_dim, dtype)
    layer = make_cache(layout, spec, device)[1 if layers > 1 else 0]
    plane = planes(layout, spec, layer)[0]
    gen = torch.Generator(device=device).manual_seed(0)
    plane.view(torch.uint8).random_(0, 256, generator=gen)
    shape = tuple(plane.shape[1:])

    order = torch.randperm(blocks, generator=torch.Generator().manual_seed(1))[:n]
    if params["order"] == "sorted":
        order = order.sort().values
    ids = order.to(device=device, dtype=torch.int32)
    rows = torch.empty((n, *shape), dtype=dtype, device=device)
    inputs: Tensors = {"plane": plane, "ids": ids, "ids_long": ids.long()}
    if params["op"] == "gather":
        inputs["out"] = rows.zero_()
        inputs["contiguous"] = plane[ids.long()].contiguous()
    else:
        rows.view(torch.uint8).random_(0, 256, generator=gen)
        inputs["src"] = rows
    return inputs


def _rows_2d(t: Any) -> Any:
    return t.view(t.shape[0], -1)


# --- implementations -------------------------------------------------------


def torch_index_select(t: Tensors) -> None:
    import torch

    torch.index_select(_rows_2d(t["plane"]), 0, t["ids_long"], out=_rows_2d(t["out"]))


def torch_advanced_indexing(t: Tensors) -> None:
    _rows_2d(t["out"]).copy_(_rows_2d(t["plane"])[t["ids_long"]])


def copy_ceiling(t: Tensors) -> None:
    t["out"].copy_(t["contiguous"])


def torch_index_copy(t: Tensors) -> None:
    _rows_2d(t["plane"]).index_copy_(0, t["ids_long"], _rows_2d(t["src"]))


def kvkernels_gather(t: Tensors) -> None:
    from kvkernels.ops.kv_gather import gather_kv_blocks

    gather_kv_blocks([t["plane"]], t["ids"], [t["out"]])


def kvkernels_scatter(t: Tensors) -> None:
    from kvkernels.ops.kv_gather import scatter_kv_blocks

    scatter_kv_blocks([t["src"]], t["ids"], [t["plane"]])


def _block_and_stride_bytes(t: Tensors) -> tuple[int, int]:
    plane = t["plane"]
    block = math.prod(plane.shape[1:]) * plane.element_size()
    return block, plane.stride(0) * plane.element_size()


def job_addresses(t: Tensors, op: str) -> tuple[Any, Any, Any]:
    """(source addresses, destination addresses, sizes) as int64 device tensors, one per id."""
    import torch

    block, stride = _block_and_stride_bytes(t)
    ids = t["ids_long"]
    rows = torch.arange(ids.numel(), device=ids.device, dtype=torch.int64)
    in_plane = t["plane"].data_ptr() + ids * stride
    if op == "gather":
        src, dst = in_plane, t["out"].data_ptr() + rows * block
    else:
        src, dst = t["src"].data_ptr() + rows * block, in_plane
    return src, dst, torch.full_like(ids, block)


def _cache_key(t: Tensors, op: str) -> tuple[int, ...]:
    other = t["out"] if op == "gather" else t["src"]
    return (t["plane"].data_ptr(), other.data_ptr(), t["ids"].data_ptr(), t["ids"].numel())


def vllm_swap_blocks(op: str) -> Callable[[Tensors], None]:
    from bench.baselines.vllm.swap_blocks_adapter import chunk_bytes, copy_jobs

    jobs: dict[tuple[int, ...], tuple[Any, Any, Any]] = {}

    def run(t: Tensors) -> None:
        key = _cache_key(t, op)
        if key not in jobs:
            jobs.clear()
            jobs[key] = job_addresses(t, op)
        src, dst, sizes = jobs[key]
        copy_jobs(src, dst, sizes, chunk_bytes(_block_and_stride_bytes(t)[0]))

    return run


def batched_dma_available() -> bool:
    try:
        from cuda.bindings import driver

        # Raises, rather than returning an error code, when no NVIDIA driver is installed.
        err, version = driver.cuDriverGetVersion()
    except Exception:
        return False
    return bool(err == driver.CUresult.CUDA_SUCCESS and version >= 12080)


def cuda_batched_dma(op: str, *, rebuild_jobs: bool) -> Callable[[Tensors], None]:
    """One cuMemcpyBatchAsync call for every block.

    The call refuses the legacy default stream, which is the stream PyTorch uses
    by default, so it runs on a side stream ordered after and before the current
    one. The two stream waits are part of the measured time.
    """
    import torch
    from cuda.bindings import driver

    attrs = driver.CUmemcpyAttributes()
    attrs.srcAccessOrder = driver.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
    side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    jobs: dict[tuple[int, ...], tuple[list[Any], list[Any], list[int]]] = {}

    def build(t: Tensors) -> tuple[list[Any], list[Any], list[int]]:
        src, dst, sizes = job_addresses(t, op)
        return (
            [driver.CUdeviceptr(a) for a in src.tolist()],
            [driver.CUdeviceptr(a) for a in dst.tolist()],
            sizes.tolist(),
        )

    def run(t: Tensors) -> None:
        key = _cache_key(t, op)
        if rebuild_jobs:
            srcs, dsts, sizes = build(t)
        else:
            if key not in jobs:
                jobs.clear()
                jobs[key] = build(t)
            srcs, dsts, sizes = jobs[key]
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        result = driver.cuMemcpyBatchAsync(
            dsts, srcs, sizes, len(sizes), [attrs], [0], 1, driver.CUstream(side.cuda_stream)
        )
        err = result[0] if isinstance(result, tuple) else result
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyBatchAsync failed: {result}")
        current.wait_stream(side)

    return run


# --- cases ------------------------------------------------------------------


def bytes_moved(p: Params) -> int:
    """Each block read once and written once, plus 4 bytes per id."""
    shape = block_shape(str(p["layout"]), int(p["block"]), int(p["kv_heads"]), int(p["head_dim"]))
    block = math.prod(shape) * _element_size(str(p["dtype"]))
    n = int(p["num_ids"])
    return 2 * n * block + 4 * n


def make_case(params: Params, *, with_dma: bool) -> Case:
    op = str(params["op"])
    shape = block_shape(
        str(params["layout"]),
        int(params["block"]),
        int(params["kv_heads"]),
        int(params["head_dim"]),
    )
    block = math.prod(shape) * _element_size(str(params["dtype"]))
    stride = block * layers_per_block(str(params["layout"]))

    impls: dict[str, Callable[[Tensors], None]]
    if op == "gather":
        impls = {
            "torch_index_select": torch_index_select,
            "kvkernels": kvkernels_gather,
            "torch_advanced_indexing": torch_advanced_indexing,
            "copy_ceiling": copy_ceiling,
        }
        outputs: tuple[str, ...] = ("out",)
    else:
        impls = {"torch_index_copy": torch_index_copy, "kvkernels": kvkernels_scatter}
        outputs = ("plane",)
    if block % 8 == 0 and stride % 8 == 0:
        impls["vllm_swap_blocks"] = vllm_swap_blocks(op)
    if with_dma:
        impls["cuda_batched_dma"] = cuda_batched_dma(op, rebuild_jobs=False)
        impls["cuda_batched_dma_build_jobs"] = cuda_batched_dma(op, rebuild_jobs=True)

    return Case(
        params=params,
        setup=lambda: make_inputs(params),
        impls=impls,
        outputs=outputs,
        check="exact",
        bytes_moved=bytes_moved,
        host_setup=lambda: make_inputs(params, tiny_blocks=True),
    )


def grid() -> Iterator[Params]:
    """Every parameter combination whose cache fits in MAX_CACHE_BYTES."""
    for op in ("gather", "scatter"):
        for layout, block_sizes in BLOCK_SIZES.items():
            for block in block_sizes:
                for heads, head_dim in HEADS_AND_DIMS:
                    shape = block_shape(layout, block, heads, head_dim)
                    block_bytes = math.prod(shape) * _element_size("float16")
                    for n in NUM_IDS:
                        if cache_bytes(layout, block_bytes, n) > MAX_CACHE_BYTES:
                            continue
                        for order in ORDERS:
                            yield {
                                "op": op,
                                "layout": layout,
                                "block": block,
                                "kv_heads": heads,
                                "head_dim": head_dim,
                                "num_ids": n,
                                "order": order,
                                "dtype": "float16",
                            }
        # bf16 once per op: the bytes moved are the same, so the time should be too.
        yield {
            "op": op,
            "layout": "vllm_LBNHC",
            "block": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "num_ids": 512,
            "order": "random",
            "dtype": "bfloat16",
        }


def cases() -> Iterator[Case]:
    with_dma = batched_dma_available()
    if not with_dma:
        print("kv_gather: cuda-bindings or a CUDA 12.8 driver is missing; no batched-DMA baseline")
    for params in grid():
        yield make_case(params, with_dma=with_dma)
