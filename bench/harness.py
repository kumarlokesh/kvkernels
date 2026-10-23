"""Benchmark harness.

Usage: ``python -m bench.harness --suite <name> [--filter key=value ...]``

A suite is a module ``bench/suites/<name>.py`` with ``cases() -> Iterable[Case]``.
For each case, the harness checks every implementation against the first one
listed (the reference), then times each one with a cold L2 and records median,
p10 and p90 with derived metrics.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from bench.schema import SCHEMA_VERSION, BenchResult, EnvInfo, Measurement, Params, utc_now
from kvkernels.specs import GPUSpec, peak_tensor_tflops

if TYPE_CHECKING:
    import torch

Tensors = dict[str, "torch.Tensor"]
Impl = Callable[[Tensors], None]
Timer = Callable[[Callable[[], None]], list[float]]

TIMER_NAME = "do_bench-cold-l2"
WARMUP_MS = 25  # covers compilation and autotuning
REP_MS = 200  # gives at least MIN_SAMPLES for kernels under about 1 ms
MIN_SAMPLES = 100
MAX_SPREAD = 1.10  # p90 / median above this marks a case as noisy
EXTRA_RUNS = 2  # reruns allowed for a noisy case
HOST_CALLS = 1000  # calls averaged for the CPU cost per call


@dataclass(frozen=True)
class Case:
    params: Params
    setup: Callable[[], Tensors]
    impls: Mapping[str, Impl]
    outputs: tuple[str, ...]
    check: Literal["exact", "tolerance"]
    bytes_moved: Callable[[Params], int] | None = None
    flops: Callable[[Params], int] | None = None
    # Required when check == "tolerance": the reference computed in fp32.
    # The first impl serves as the low-precision reference.
    reference_fp32: Callable[[Tensors], Tensors] | None = None
    # Small inputs for measuring the wrapper's CPU cost per call.
    host_setup: Callable[[], Tensors] | None = None


class CorrectnessError(AssertionError):
    """An implementation disagrees with the reference; the suite stops."""


@dataclass(frozen=True)
class Timing:
    median_ms: float
    p10_ms: float
    p90_ms: float
    samples: int

    @property
    def spread(self) -> float:
        return self.p90_ms / self.median_ms if self.median_ms > 0 else float("inf")


def summarize(times_ms: Sequence[float]) -> Timing:
    if not times_ms:
        raise ValueError("no timing samples")
    p10, p50, p90 = np.percentile(np.asarray(times_ms, dtype=np.float64), [10, 50, 90])
    return Timing(median_ms=float(p50), p10_ms=float(p10), p90_ms=float(p90), samples=len(times_ms))


def do_bench_samples(fn: Callable[[], None], rep_ms: int = REP_MS) -> list[float]:
    """Raw per-sample times in ms from triton.testing.do_bench.

    do_bench clears L2 before every sample by zeroing a large buffer, so each
    sample reads its inputs from HBM, as a kernel does in serving.
    """
    from triton.testing import do_bench

    times: list[float] = list(do_bench(fn, warmup=WARMUP_MS, rep=rep_ms, return_mode="all"))
    return times


def timed_samples(fn: Callable[[], None], timer: Timer) -> list[float]:
    """Run the timer; if it returns fewer than MIN_SAMPLES, retry with a longer rep."""
    times = timer(fn)
    if len(times) >= MIN_SAMPLES or timer is not do_bench_samples:
        return times
    scale = MIN_SAMPLES / max(len(times), 1) * 1.2
    return do_bench_samples(fn, rep_ms=int(REP_MS * scale) + 1)


def time_with_reruns(fn: Callable[[], None], timer: Timer) -> tuple[Timing, bool]:
    """Time fn; rerun a noisy case and keep the run with the lowest spread."""
    best = summarize(timed_samples(fn, timer))
    for _ in range(EXTRA_RUNS):
        if best.spread <= MAX_SPREAD:
            break
        again = summarize(timed_samples(fn, timer))
        if again.spread < best.spread:
            best = again
    return best, best.spread > MAX_SPREAD


def host_us_per_call(impl: Impl, inputs: Tensors, calls: int = HOST_CALLS) -> float:
    """CPU time per call of an implementation, in microseconds.

    Reported separately from GPU time: engines run decode under CUDA graphs,
    where this cost is paid once at capture.
    """
    import torch

    impl(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        impl(inputs)
    elapsed = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return elapsed / calls * 1e6


def check_case(case: Case) -> None:
    """Compare every implementation's outputs with the first one.

    Exact cases compare bits; tolerance cases use kvkernels.tolerance.check_close.
    """
    from kvkernels.tolerance import assert_exact, check_close

    names = list(case.impls)
    if not names:
        raise ValueError(f"case {case.params} has no implementations")
    if case.check == "tolerance" and case.reference_fp32 is None:
        raise ValueError(f"case {case.params}: tolerance check needs reference_fp32")

    ref_inputs = case.setup()
    case.impls[names[0]](ref_inputs)
    ref32 = case.reference_fp32(case.setup()) if case.reference_fp32 is not None else None

    for name in names[1:]:
        inputs = case.setup()
        case.impls[name](inputs)
        for out in case.outputs:
            where = f"{name} vs {names[0]}, output {out!r}, params {case.params}"
            if case.check == "exact":
                try:
                    assert_exact(inputs[out], ref_inputs[out])
                except AssertionError as exc:
                    raise CorrectnessError(f"{where}: {exc}") from exc
            else:
                assert ref32 is not None
                result = check_close(inputs[out], ref_inputs[out], ref32[out])
                if not result.ok:
                    raise CorrectnessError(
                        f"{where}: error {result.err:.3e} > allowed {result.allowed:.3e}"
                    )


def _ratio(value: float | None, peak: float | None) -> float | None:
    if value is None or peak is None or peak <= 0:
        return None
    return value / peak


def measure(
    case: Case,
    name: str,
    timing: Timing,
    noisy: bool,
    spec: GPUSpec | None,
    measured_peak_gbps: float | None,
    host_us: float | None,
) -> Measurement:
    """Build one Measurement with derived metrics from the median time.

    GB/s and TFLOP/s use decimal units (1e9, 1e12), as vendor datasheets do.
    """
    seconds = timing.median_ms * 1e-3
    nbytes = case.bytes_moved(case.params) if case.bytes_moved else None
    nflops = case.flops(case.params) if case.flops else None
    gbps = nbytes / seconds / 1e9 if nbytes is not None and seconds > 0 else None
    tflops = nflops / seconds / 1e12 if nflops is not None and seconds > 0 else None
    dtype = case.params.get("dtype")
    peak_compute = (
        peak_tensor_tflops(spec, dtype) if spec is not None and isinstance(dtype, str) else None
    )
    return Measurement(
        params=dict(case.params),
        impl=name,
        median_ms=timing.median_ms,
        p10_ms=timing.p10_ms,
        p90_ms=timing.p90_ms,
        samples=timing.samples,
        bytes_moved=nbytes,
        flops=nflops,
        gbps=gbps,
        tflops=tflops,
        pct_datasheet_bw=_ratio(gbps, spec.peak_hbm_gbps if spec else None),
        pct_measured_bw=_ratio(gbps, measured_peak_gbps),
        pct_peak_compute=_ratio(tflops, peak_compute),
        host_us=host_us,
        noisy=noisy,
    )


def run_suite(
    name: str,
    cases: Iterable[Case],
    spec: GPUSpec | None,
    *,
    env: EnvInfo,
    git_sha: str,
    flush_bytes: int,
    measured_peak_gbps: float | None,
    timer: Timer = do_bench_samples,
) -> BenchResult:
    if spec is not None and flush_bytes <= 2 * spec.l2_bytes:
        raise RuntimeError(
            f"L2 flush buffer ({flush_bytes} B) must exceed twice the L2 ({spec.l2_bytes} B)"
        )
    measurements: list[Measurement] = []
    for case in cases:
        check_case(case)
        for impl_name, impl in case.impls.items():
            inputs = case.setup()
            timing, noisy = time_with_reruns(functools.partial(impl, inputs), timer)
            host_us = (
                host_us_per_call(impl, case.host_setup()) if case.host_setup is not None else None
            )
            measurements.append(
                measure(case, impl_name, timing, noisy, spec, measured_peak_gbps, host_us)
            )
            print(
                f"{name} {case.params} {impl_name}: median {timing.median_ms * 1e3:.2f} us"
                f" (p10 {timing.p10_ms * 1e3:.2f}, p90 {timing.p90_ms * 1e3:.2f},"
                f" n={timing.samples}){' NOISY' if noisy else ''}",
                flush=True,
            )
    return BenchResult(
        schema_version=SCHEMA_VERSION,
        suite=name,
        git_sha=git_sha,
        created_utc=utc_now(),
        timer=TIMER_NAME,
        flush_bytes=flush_bytes,
        env=env,
        measured_peak_gbps=measured_peak_gbps,
        measurements=measurements,
    )


def git_sha(repo: Path = Path(".")) -> str:
    """HEAD commit, prefixed with "dirty-" when the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"dirty-{sha}" if status.strip() else sha


def flush_buffer_bytes() -> int:
    """Size of the buffer do_bench zeroes before each sample (256 MiB in Triton 3.8.0)."""
    from triton import runtime

    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    return int(cache.numel() * cache.element_size())


def measure_copy_peak(timer: Timer = do_bench_samples) -> float:
    """Measured peak bandwidth in GB/s: a 1 GiB device copy, one read and one write."""
    from bench.suites.smoke import PROBE_BYTES, copy_case

    case = copy_case(PROBE_BYTES)
    inputs = case.setup()
    impl = case.impls["torch_copy"]
    timing = summarize(timed_samples(functools.partial(impl, inputs), timer))
    return 2 * PROBE_BYTES / (timing.median_ms * 1e-3) / 1e9


def load_cases(suite: str) -> list[Case]:
    module = importlib.import_module(f"bench.suites.{suite}")
    return list(module.cases())


def parse_filters(items: Sequence[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"filter must look like key=value, got {item!r}")
        filters[key] = value
    return filters


def matches(params: Params, filters: Mapping[str, str]) -> bool:
    return all(str(params.get(k)) == v for k, v in filters.items())


def main(argv: Sequence[str] | None = None) -> int:
    from bench import env as envmod
    from bench.schema import write_result
    from kvkernels.specs import lookup

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="run even if nvidia-smi lists other compute processes (e.g. inside a container)",
    )
    args = parser.parse_args(argv)

    info = envmod.collect()
    others = envmod.other_compute_pids()
    if others and not args.allow_busy_gpu:
        print(f"harness: other processes are using the GPU: {others}", file=sys.stderr)
        return 2

    filters = parse_filters(args.filter)
    cases = [c for c in load_cases(args.suite) if matches(c.params, filters)]
    if not cases:
        print("harness: no cases match", file=sys.stderr)
        return 2

    peak = measure_copy_peak()
    print(f"measured copy peak: {peak:.1f} GB/s", flush=True)
    result = run_suite(
        args.suite,
        cases,
        lookup(info.gpu_name),
        env=info,
        git_sha=git_sha(),
        flush_bytes=flush_buffer_bytes(),
        measured_peak_gbps=peak,
    )
    path = write_result(result, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
