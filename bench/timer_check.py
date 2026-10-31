"""Check whether the do_bench L2 flush hides kernel launch overhead.

Usage on a GPU machine: ``python -m bench.timer_check`` (or ``make check-timer``).

For each probe kernel, compares three medians:

- do_bench: the harness timer (flush, start event, kernel, end event).
- graph: a CUDA graph of [flush, kernel] x N minus a graph of [flush] x N, divided
  by N. No launch overhead; the fallback timer for small kernels.
- ncu: Nsight Compute's gpu__time_duration for the kernel alone, with caches
  flushed (--cache-control all) and clocks left alone (--clock-control none);
  by default ncu locks clocks to base, which would make it disagree with
  do_bench for a reason unrelated to launch overhead.

If do_bench is within 10% of ncu for every small probe, the flush hides launch
overhead and the harness timer stays. Otherwise kernels under 50 us move to the
graph timer.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

TOLERANCE = 0.10  # largest accepted gap between do_bench and ncu
SMALL_KERNEL_US = 50.0
NVTX_RANGE = "probe"
NCU_LAUNCHES = 50
GRAPH_CALLS = 100
GRAPH_REPLAYS = 20

_UNIT_TO_NS = {
    "ns": 1.0,
    "nsecond": 1.0,
    "us": 1e3,
    "usecond": 1e3,
    "ms": 1e6,
    "msecond": 1e6,
    "s": 1e9,
    "second": 1e9,
}


@dataclass(frozen=True)
class ProbeResult:
    name: str
    do_bench_us: float
    graph_us: float
    ncu_us: float

    @property
    def overhead_fraction(self) -> float:
        """How much longer do_bench measures than ncu, as a fraction of ncu."""
        return self.do_bench_us / self.ncu_us - 1.0

    @property
    def hidden(self) -> bool:
        return abs(self.overhead_fraction) <= TOLERANCE


def parse_ncu_durations_ns(csv_text: str) -> list[float]:
    """gpu__time_duration values, in ns, from ``ncu --csv`` output.

    ncu mixes "==PROF==" log lines into its output; only quoted CSV lines are read.
    """
    rows = [line for line in csv_text.splitlines() if line.startswith('"')]
    durations: list[float] = []
    for row in csv.DictReader(io.StringIO("\n".join(rows))):
        if not row.get("Metric Name", "").startswith("gpu__time_duration"):
            continue
        unit = row.get("Metric Unit", "").strip()
        if unit not in _UNIT_TO_NS:
            raise ValueError(f"unknown ncu unit {unit!r}")
        value = float(row["Metric Value"].replace(",", ""))
        durations.append(value * _UNIT_TO_NS[unit])
    return durations


def decision(results: Sequence[ProbeResult]) -> str:
    small = [r for r in results if r.ncu_us < SMALL_KERNEL_US]
    if all(r.hidden for r in small):
        return "keep do_bench: the flush hides launch overhead for every probe"
    return f"switch kernels under {SMALL_KERNEL_US:.0f} us to the graph timer"


# --- GPU side -----------------------------------------------------------------


def _probes() -> dict[str, Callable[[], Callable[[], None]]]:
    """Each entry allocates its inputs and returns the function to time."""

    def one_program() -> Callable[[], None]:
        import torch
        import triton
        import triton.language as tl

        @triton.jit  # type: ignore[untyped-decorator]
        def _one_program_kernel(out_ptr, N: tl.constexpr):  # type: ignore[no-untyped-def]
            offs = tl.arange(0, N)
            tl.store(out_ptr + offs, offs.to(tl.float32))

        out = torch.empty(128, device="cuda")

        def fn() -> None:
            _one_program_kernel[(1,)](out, N=128)

        return fn

    def gather_64() -> Callable[[], None]:
        # torch.index_select stands in for kvkernels' own gather kernel: a
        # 64-block gather of 64 KiB KV blocks from a 1 GiB plane.
        import torch

        plane = torch.empty(16384, 32768, dtype=torch.float16, device="cuda")
        ids = torch.randperm(16384, device="cuda")[:64]
        out = torch.empty(64, 32768, dtype=torch.float16, device="cuda")

        def fn() -> None:
            torch.index_select(plane, 0, ids, out=out)

        return fn

    return {"one_program": one_program, "gather_64": gather_64}


def _flush_fn() -> Callable[[], None]:
    from triton import runtime

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    def flush() -> None:
        runtime.driver.active.clear_cache(cache)

    return flush


def do_bench_us(fn: Callable[[], None]) -> float:
    from bench.harness import do_bench_samples, summarize, timed_samples

    return summarize(timed_samples(fn, do_bench_samples)).median_ms * 1e3


def graph_us(fn: Callable[[], None]) -> float:
    import torch

    flush = _flush_fn()

    def capture(body: Callable[[], None]) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(GRAPH_CALLS):
                body()
        return graph

    def both() -> None:
        flush()
        fn()

    fn()
    flush()
    torch.cuda.synchronize()
    g_both = capture(both)
    g_flush = capture(flush)

    def replay_ms(graph: torch.cuda.CUDAGraph) -> float:
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    samples = [(replay_ms(g_both) - replay_ms(g_flush)) / GRAPH_CALLS for _ in range(GRAPH_REPLAYS)]
    return statistics.median(samples) * 1e3


def ncu_us(probe: str) -> float:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu not found; install Nsight Compute or add it to PATH")
    cmd = [
        ncu,
        "--metrics",
        "gpu__time_duration.sum",
        "--csv",
        "--cache-control",
        "all",
        "--clock-control",
        "none",
        "--nvtx",
        "--nvtx-include",
        f"{NVTX_RANGE}/",
        sys.executable,
        "-m",
        "bench.timer_check",
        "--ncu-target",
        probe,
    ]
    out = run_ncu(cmd)
    durations = parse_ncu_durations_ns(out)
    if len(durations) < NCU_LAUNCHES // 2:
        raise RuntimeError(f"ncu reported {len(durations)} launches for {probe}; output:\n{out}")
    return statistics.median(durations) / 1e3


def run_ncu(cmd: list[str]) -> str:
    """ncu's stdout; if ncu fails, an error that carries the end of its output."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
        raise RuntimeError(f"ncu exited with {proc.returncode}:\n{tail}")
    return proc.stdout


def ncu_target(probe: str) -> None:
    """Run inside ncu: launch the probe NCU_LAUNCHES times inside an NVTX range."""
    import torch

    fn = _probes()[probe]()
    fn()
    torch.cuda.synchronize()
    for _ in range(NCU_LAUNCHES):
        with torch.cuda.nvtx.range(NVTX_RANGE):
            fn()
    torch.cuda.synchronize()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that do_bench hides launch overhead")
    parser.add_argument("--ncu-target", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.ncu_target:
        ncu_target(args.ncu_target)
        return 0

    from bench import env

    try:
        info = env.collect()
    except RuntimeError as exc:
        print(f"timer_check: {exc}", file=sys.stderr)
        return 1
    if env.counters_restricted():
        print("timer_check: performance counters are restricted; ncu cannot run", file=sys.stderr)
        return 1

    results = []
    for name, make in _probes().items():
        fn = make()
        results.append(ProbeResult(name, do_bench_us(fn), graph_us(fn), ncu_us(name)))

    print(
        f"GPU {info.gpu_name}, driver {info.driver_version}, torch {info.torch_version},"
        f" triton {info.triton_version}"
    )
    print(
        f"{'probe':12s} {'do_bench us':>12s} {'graph us':>10s}"
        f" {'ncu us':>10s} {'do_bench/ncu':>13s}"
    )
    for r in results:
        print(
            f"{r.name:12s} {r.do_bench_us:12.2f} {r.graph_us:10.2f} {r.ncu_us:10.2f}"
            f" {r.overhead_fraction:+12.1%}"
        )
    print(decision(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
