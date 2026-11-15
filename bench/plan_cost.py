"""The cost of planning one cascade step, on the host and on the GPU.

Usage on a GPU machine: ``python -m bench.plan_cost``.

Each batch is requests in groups of 16. A group shares a prefix of 256 KV
blocks of 16 tokens, and each request has 256 KV blocks of its own, so every
block table row has 512 entries. One row per batch size:

    host plan          CPU time of the host plan: groups named by the caller,
                       buffers built in Python and copied to the GPU
    launches, CPU      CPU time of the four planning kernels launched directly
    GPU plan, CPU      CPU time of planning from the block table, called eagerly;
                       after the second call the plan replays its own graph
    GPU plan, GPU      GPU time of planning from the block table
    step in a graph    GPU time of one replay of planning plus one layer's
                       attention, captured together in one CUDA graph

A second table gives each planning kernel's GPU time, from a cold L2.

The batch sizes run to 256, the largest batch the planning targets name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from bench.harness import Tensors, host_us_per_call, read_flush_samples, summarize
from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache
from kvkernels.ops.cascade import CascadePlan, PrefixGroup, cascade_decode_attention
from kvkernels.ops.cascade_plan import planning_launches

BATCHES = (16, 64, 128, 256)
GROUP_SIZE = 16
BLOCK_SIZE = 16
PREFIX_BLOCKS = 256
OWN_BLOCKS = 256
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128  # Llama 3.1 8B
CALLS = 200


@dataclass
class Row:
    batch: int
    host_plan_us: float
    launches_cpu_us: float
    gpu_plan_cpu_us: float
    gpu_plan_us: float
    graph_step_us: float
    kernel_us: dict[str, float]


def _as_impl(fn: Callable[[], None]) -> Callable[[Tensors], None]:
    return lambda _inputs: fn()


def measure_batch(batch: int, *, calls: int = CALLS, device: str = "cuda") -> Row:
    groups = max(batch // GROUP_SIZE, 1)
    size = batch // groups
    num_blocks = 1 + groups * PREFIX_BLOCKS + batch * OWN_BLOCKS
    ids = torch.randperm(num_blocks - 1, generator=torch.Generator().manual_seed(0)) + 1
    prefixes = ids[: groups * PREFIX_BLOCKS].view(groups, PREFIX_BLOCKS)
    own = ids[groups * PREFIX_BLOCKS :].view(batch, OWN_BLOCKS)
    rows = torch.cat([prefixes.repeat_interleave(size, dim=0), own], dim=1)
    table = rows.to(torch.int32).to(device)
    lens = torch.full((batch,), (PREFIX_BLOCKS + OWN_BLOCKS) * BLOCK_SIZE, dtype=torch.int32)
    lens = lens.to(device)

    layout = LAYOUTS["vllm_LBNHC"]
    spec = CacheSpec(1, num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, torch.float16)
    k, v = kv_views(layout, spec, make_cache(layout, spec, device)[0])
    q = torch.randn((batch, Q_HEADS, HEAD_DIM), dtype=torch.float16, device=device)
    out = torch.empty_like(q)
    plan = CascadePlan(
        max_requests=batch,
        max_groups=groups,
        max_blocks=groups * PREFIX_BLOCKS + batch * OWN_BLOCKS,
        num_q_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        num_blocks=num_blocks,
        device=torch.device(device),
    )
    host_groups = [
        PrefixGroup(
            range(g * size, (g + 1) * size), prefixes[g].tolist(), PREFIX_BLOCKS * BLOCK_SIZE
        )
        for g in range(groups)
    ]
    suffix_blocks = own.tolist()
    suffix_lens = [OWN_BLOCKS * BLOCK_SIZE] * batch

    def host_plan() -> None:
        plan.update(groups=host_groups, suffix_blocks=suffix_blocks, suffix_lens=suffix_lens)

    def gpu_plan() -> None:
        plan.plan_from_block_table(table, lens)

    def step() -> None:
        gpu_plan()
        cascade_decode_attention(q, k, v, plan, out)

    launches = planning_launches(
        plan, index=table, starts=None, seq_lens=lens, stride_r=table.stride(0)
    )

    def launch_all() -> None:
        for _, launch in launches:
            launch()

    host_us = host_us_per_call(_as_impl(host_plan), {}, calls)
    gpu_plan()  # sets the batch size the launches read
    launches_us = host_us_per_call(_as_impl(launch_all), {}, calls)
    gpu_cpu_us = host_us_per_call(_as_impl(gpu_plan), {}, calls)
    gpu_us = summarize(read_flush_samples(gpu_plan)).median_ms * 1000
    kernel_us = {
        name: summarize(read_flush_samples(launch)).median_ms * 1000 for name, launch in launches
    }
    step()  # compile every kernel before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph_us = summarize(read_flush_samples(graph.replay)).median_ms * 1000
    return Row(batch, host_us, launches_us, gpu_cpu_us, gpu_us, graph_us, kernel_us)


def measure(batches: tuple[int, ...] = BATCHES, calls: int = CALLS) -> list[Row]:
    return [measure_batch(b, calls=calls) for b in batches]


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    heads = ("batch", "host plan", "launches, CPU", "GPU plan, CPU", "GPU plan, GPU", "graph step")
    print(" ".join(f"{h:>{w}}" for h, w in zip(heads, (6, 12, 14, 14, 14, 12), strict=True)))
    rows = measure()
    for r in rows:
        print(
            f"{r.batch:>6} {r.host_plan_us:>10.1f}us {r.launches_cpu_us:>12.1f}us"
            f" {r.gpu_plan_cpu_us:>12.1f}us {r.gpu_plan_us:>12.1f}us {r.graph_step_us:>10.1f}us"
        )
    names = list(rows[0].kernel_us)
    print("\nGPU time per planning kernel")
    print(f"{'batch':>6} " + " ".join(f"{n:>10}" for n in names))
    for r in rows:
        print(f"{r.batch:>6} " + " ".join(f"{r.kernel_us[n]:>8.1f}us" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
