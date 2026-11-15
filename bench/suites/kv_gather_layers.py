"""Gather and scatter of every layer's KV blocks at once: one launch against one per plane.

An offload copies the same KV blocks out of every layer. Each case builds a
32-layer cache with kvkernels.layouts and copies the named blocks of every
plane:

    vllm_LBNHC   32 planes, one per layer, each its own tensor
    vllm_BLNHC   32 planes interleaved in one buffer (all layers of a block together)
    sglang_nhd   64 planes, K and V of each layer, one token per page

Implementations, all checked bit for bit against the first:

    gather:  torch_index_select (reference, one call per plane), kvkernels_plane_set
             (one launch for every plane), kvkernels_per_plane (one gather_kv_blocks
             call per plane)
    scatter: torch_index_copy (reference), kvkernels_plane_set, kvkernels_per_plane

The plane set is built in the plan step, which the harness times separately
(plan_us) from the call (host_us and GPU time). Every plane's rows are one slice
of a single rows tensor, so a gather has one output to compare. A scatter
writes 32 to 64 tensors; its output, "scattered", is read after the call: the
scattered blocks of every plane, gathered back by PyTorch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from bench.harness import Case, Tensors
from bench.schema import Params

LAYERS = 32
HEADS, HEAD_DIM = 8, 128
# (layout, page or block size, KV blocks per plane, ids): the vLLM cases move
# 4 to 64 MiB per layer; the SGLang cases name tokens, as SGLang's offload does.
SHAPES = (
    ("vllm_LBNHC", 16, 2048, (64, 512, 1024)),
    ("vllm_BLNHC", 16, 2048, (64, 512, 1024)),
    ("sglang_nhd", 1, 32768, (1024, 8192, 16384)),
)


def planes_per_layer(layout: str) -> int:
    return 2 if layout.startswith("sglang") else 1


def block_bytes(p: Params) -> int:
    content = HEAD_DIM if str(p["layout"]).startswith("sglang") else 2 * HEAD_DIM
    return int(p["block"]) * HEADS * content * 2  # fp16


def bytes_moved(p: Params) -> int:
    """Each block of each plane read once and written once, plus 4 bytes per id."""
    planes = LAYERS * planes_per_layer(str(p["layout"]))
    n = int(p["num_ids"])
    return 2 * n * block_bytes(p) * planes + 4 * n


class _Inputs(dict[str, Any]):
    """The case's tensors; "scattered" is computed when the harness reads it, after the call."""

    def __missing__(self, key: str) -> Any:
        if key != "scattered":
            raise KeyError(key)
        import torch

        return torch.stack([p.reshape(p.shape[0], -1)[self["ids_long"]] for p in self["planes"]])


def make_inputs(params: Params, *, tiny_blocks: bool = False, device: str = "cuda") -> Tensors:
    """Every plane of a 32-layer cache, the ids, and one rows tensor holding every plane's rows.

    tiny_blocks keeps the planes and the number of ids but shrinks every block
    to 16 bytes, for measuring the CPU cost of a call.
    """
    import torch

    from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache, planes

    name = str(params["layout"])
    n = int(params["num_ids"])
    if tiny_blocks:
        block, heads, head_dim, blocks = 1, 1, 4, max(2 * n, 2)
    else:
        block, heads, head_dim, blocks = (
            int(params["block"]),
            HEADS,
            HEAD_DIM,
            int(params["blocks"]),
        )
    layout = LAYOUTS[name]
    spec = CacheSpec(LAYERS, blocks, block, heads, head_dim, torch.float16)
    cache = make_cache(layout, spec, device)
    all_planes = [plane for layer in cache for plane in planes(layout, spec, layer)]
    gen = torch.Generator(device=device).manual_seed(0)
    for tensor in {id(t): t for layer in cache for t in layer}.values():
        tensor.view(torch.uint8).random_(0, 256, generator=gen)

    ids = torch.randperm(blocks, generator=torch.Generator().manual_seed(1))[:n]
    shape = tuple(all_planes[0].shape[1:])
    rows = torch.empty((len(all_planes), n, *shape), dtype=torch.float16, device=device)
    if params["op"] == "gather":
        rows.zero_()
    else:
        rows.view(torch.uint8).random_(0, 256, generator=gen)
    return _Inputs(
        ids=ids.to(device=device, dtype=torch.int32),
        ids_long=ids.to(device=device),
        rows=rows,
        planes=all_planes,
    )


# --- implementations -------------------------------------------------------


def torch_index_select(t: Tensors) -> None:
    import torch

    for plane, rows in zip(t["planes"], t["rows"], strict=True):
        torch.index_select(
            plane.reshape(plane.shape[0], -1), 0, t["ids_long"], out=rows.view(rows.shape[0], -1)
        )


def torch_index_copy(t: Tensors) -> None:
    for plane, rows in zip(t["planes"], t["rows"], strict=True):
        plane.view(plane.shape[0], -1).index_copy_(0, t["ids_long"], rows.view(rows.shape[0], -1))


def kvkernels_per_plane(op: str) -> Callable[[Tensors], None]:
    def gather(t: Tensors) -> None:
        from kvkernels.ops.kv_gather import gather_kv_blocks

        for plane, rows in zip(t["planes"], t["rows"], strict=True):
            gather_kv_blocks([plane], t["ids"], [rows])

    def scatter(t: Tensors) -> None:
        from kvkernels.ops.kv_gather import scatter_kv_blocks

        for plane, rows in zip(t["planes"], t["rows"], strict=True):
            scatter_kv_blocks([rows], t["ids"], [plane])

    return gather if op == "gather" else scatter


class KvkernelsPlaneSet:
    """One plane set per set of inputs, built in the plan step."""

    def __init__(self, op: str) -> None:
        self.op = op

    def plan(self, t: Tensors) -> None:
        from kvkernels.ops.plane_set import PlaneSet

        planes: Any = t["planes"]  # a list of tensors in the inputs dict
        t["plane_set"] = PlaneSet(planes, list(t["rows"]))  # type: ignore[assignment]

    def run(self, t: Tensors) -> None:
        plane_set: Any = t["plane_set"]
        if self.op == "gather":
            plane_set.gather(t["ids"])
        else:
            plane_set.scatter(t["ids"])


# --- cases ------------------------------------------------------------------


def make_case(params: Params) -> Case:
    op = str(params["op"])
    plane_set = KvkernelsPlaneSet(op)
    reference = torch_index_select if op == "gather" else torch_index_copy
    return Case(
        params=params,
        setup=lambda: make_inputs(params),
        impls={
            f"torch_{'index_select' if op == 'gather' else 'index_copy'}": reference,
            "kvkernels_plane_set": plane_set.run,
            "kvkernels_per_plane": kvkernels_per_plane(op),
        },
        outputs=("rows",) if op == "gather" else ("scattered",),
        check="exact",
        bytes_moved=bytes_moved,
        host_setup=lambda: make_inputs(params, tiny_blocks=True),
        plans={"kvkernels_plane_set": plane_set.plan},
    )


def grid() -> Iterator[Params]:
    for op in ("gather", "scatter"):
        for layout, block, blocks, id_counts in SHAPES:
            for n in id_counts:
                yield {
                    "op": op,
                    "layout": layout,
                    "layers": LAYERS,
                    "block": block,
                    "blocks": blocks,
                    "kv_heads": HEADS,
                    "head_dim": HEAD_DIM,
                    "num_ids": n,
                    "dtype": "float16",
                }


def cases() -> Iterator[Case]:
    for params in grid():
        yield make_case(params)
