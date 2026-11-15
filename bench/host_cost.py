"""Where the CPU time of one gather_kv_blocks call goes.

Usage on a GPU machine: ``python -m bench.host_cost``.

The benchmark harness records host_us, the CPU time per call, for the whole
public op. This tool times the same call with one layer removed at a time, so
the difference between two rows is the cost of that layer:

    registered custom op              the op through PyTorch's custom-op dispatch
    gather_kv_blocks                  the public op, as the harness times it
      └─ checks + per-plane copy      the op's body, called directly
         └─ per-plane copy            one plane's launch, without the checks
            └─ direct launch          the kernel launched with a fixed tile
    autotuned launch                  the same launch through Triton's autotuner

Two more rows give the scale: an empty custom op (dispatch alone) and
torch.index_select on the same inputs. Inputs are small (512 ids, 16-byte KV
blocks), as the harness uses for host_us, so the GPU work stays short.

Two rows copy 32 layers, one plane each, as an offload does: a plane set (one
launch for all of them) and 32 gather_kv_blocks calls. The plane set's target
is under 2x one single-plane gather_kv_blocks call; the last line prints the
ratio.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from bench.harness import Tensors, host_us_per_call
from kvkernels.ops import kv_gather
from kvkernels.ops.plane_set import PlaneSet

NUM_IDS = 512
NUM_BLOCKS = 4096
CALLS = 2000
LAYERS = 32


@torch.library.custom_op("kvkernels_probe::empty", mutates_args=("outs",))
def _empty_op(
    planes: list[torch.Tensor], block_ids: torch.Tensor, outs: list[torch.Tensor], validate: bool
) -> None:
    return None


def parts(device: str = "cuda") -> dict[str, Callable[[], None]]:
    """Each row's call, on fresh inputs. The autotuner has already picked its tile."""
    plane = torch.zeros(NUM_BLOCKS, 1, 8, dtype=torch.float16, device=device)
    ids = torch.randperm(NUM_BLOCKS, device=device)[:NUM_IDS].to(torch.int32)
    out = torch.empty(NUM_IDS, 1, 8, dtype=torch.float16, device=device)
    planes, outs = [plane], [out]
    kv_gather.gather_kv_blocks(planes, ids, outs)  # autotune before timing

    tile = kv_gather._SMALLEST_TILE
    meta = {"IDS_INDEX_SRC": True, "ALIGN": 2, "WORD": kv_gather._WORD_TYPES[8]}
    args = (plane, out, ids, 2, 2, 2)  # 16-byte blocks: 2 words each, 2 words apart

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (NUM_IDS, 1)

    def registered() -> None:
        kv_gather._gather_op.custom_op(planes, ids, outs, False)

    def public() -> None:
        kv_gather.gather_kv_blocks(planes, ids, outs)

    def body() -> None:
        kv_gather._gather_op.body(planes, ids, outs, False)

    def copy_plane() -> None:
        kv_gather._copy_plane(plane, out, ids, gather=True)

    def direct() -> None:
        kv_gather._copy_blocks_kernel[(NUM_IDS, 1)](*args, **meta, TILE=tile)

    def autotuned() -> None:
        kv_gather._copy_blocks_autotuned[grid](*args, **meta)

    def empty_op() -> None:
        _empty_op(planes, ids, outs, False)

    flat_plane, flat_out, ids_long = plane.view(NUM_BLOCKS, -1), out.view(NUM_IDS, -1), ids.long()

    layer_planes = [torch.zeros_like(plane) for _ in range(LAYERS)]
    layer_outs = [torch.empty_like(out) for _ in range(LAYERS)]
    plane_set = PlaneSet(layer_planes, layer_outs)
    plane_set.gather(ids)  # autotune before timing

    def plane_set_all_layers() -> None:
        plane_set.gather(ids)

    def per_layer_calls() -> None:
        for layer_plane, layer_out in zip(layer_planes, layer_outs, strict=True):
            kv_gather.gather_kv_blocks([layer_plane], ids, [layer_out])

    def index_select() -> None:
        torch.index_select(flat_plane, 0, ids_long, out=flat_out)

    return {
        "registered custom op": registered,
        "gather_kv_blocks": public,
        "checks + per-plane copy": body,
        "per-plane copy": copy_plane,
        "direct launch": direct,
        "autotuned launch": autotuned,
        "empty custom op": empty_op,
        "torch.index_select": index_select,
        f"plane set, {LAYERS} layers": plane_set_all_layers,
        f"gather_kv_blocks x {LAYERS} layers": per_layer_calls,
    }


def _as_impl(fn: Callable[[], None]) -> Callable[[Tensors], None]:
    return lambda _inputs: fn()


def measure(calls: int = CALLS, device: str = "cuda") -> dict[str, float]:
    return {name: host_us_per_call(_as_impl(fn), {}, calls) for name, fn in parts(device).items()}


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    result = measure()
    for name, us in result.items():
        print(f"{name:32s} {us:8.1f} us per call")
    ratio = result[f"plane set, {LAYERS} layers"] / result["gather_kv_blocks"]
    print(f"plane set over {LAYERS} layers / one single-plane call: {ratio:.2f} (target below 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
