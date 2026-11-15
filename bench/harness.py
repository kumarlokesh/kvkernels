"""Benchmark harness.

Usage: ``python -m bench.harness --suite <name> [--filter key=value ...]``

A suite is a module ``bench/suites/<name>.py`` with ``cases() -> Iterable[Case]``;
a suite named ``experiments-<name>`` is ``experiments/<name>/bench_suite.py``.
For each case, the harness checks every implementation against the first one
listed (the reference), then times each one with a cold L2 and records median,
p10 and p90 with derived metrics. The reference's outputs wait in host memory
while the others run, so the GPU holds one implementation's inputs at a time. A
case that still runs out of GPU memory is recorded under left_out with the
reason, and the suite goes on.

An implementation can have a plan: host-side preparation that an engine runs
once per batch and reuses across layers. The plan runs on fresh inputs before the
implementation, outside the timed region; its CPU time is recorded as plan_us.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from bench.schema import (
    SCHEMA_VERSION,
    BenchResult,
    EnvInfo,
    LeftOut,
    Measurement,
    Params,
    utc_now,
)
from kvkernels.specs import GPUSpec, peak_tensor_tflops

if TYPE_CHECKING:
    import torch

Tensors = dict[str, "torch.Tensor"]
Impl = Callable[[Tensors], None]
Timer = Callable[[Callable[[], None]], list[float]]

TIMER_NAME = "events-read-flush-cold-l2"
WARMUP_MS = 25  # covers compilation and autotuning
REP_MS = 200  # gives at least MIN_SAMPLES for kernels under about 1 ms
MIN_SAMPLES = 100
MAX_SPREAD = 1.10  # p90 / median above this marks a case as noisy
# ... unless p90 is within this of the median: for kernels of a few microseconds,
# that gap is the resolution of the CUDA event timer (about 0.5 us per event).
MIN_NOISE_MS = 0.002
EXTRA_RUNS = 2  # reruns allowed for a noisy case
HOST_CALLS = 1000  # calls averaged for the CPU cost per call
HOST_SECONDS = 1.0  # or fewer calls, once this much time has passed
PLAN_CALLS = 20  # plan calls averaged for plan_us; a plan can cost milliseconds


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
    # Per implementation: host preparation run on the inputs before the
    # implementation and never timed with it (see the module docstring).
    plans: Mapping[str, Impl] | None = None
    # Per implementation: its own inputs, for one that needs tensors the others do
    # not; the outputs must match setup's.
    setups: Mapping[str, Callable[[], Tensors]] | None = None

    def setup_for(self, impl: str) -> Tensors:
        if self.setups and impl in self.setups:
            return self.setups[impl]()
        return self.setup()

    def prepared(self, impl: str, inputs: Tensors) -> Tensors:
        """Run the implementation's plan on inputs, if it has one, and return them."""
        if self.plans and impl in self.plans:
            self.plans[impl](inputs)
        return inputs


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

    @property
    def noisy(self) -> bool:
        return self.spread > MAX_SPREAD and self.p90_ms - self.median_ms > MIN_NOISE_MS


def summarize(times_ms: Sequence[float]) -> Timing:
    if not times_ms:
        raise ValueError("no timing samples")
    p10, p50, p90 = np.percentile(np.asarray(times_ms, dtype=np.float64), [10, 50, 90])
    return Timing(median_ms=float(p50), p10_ms=float(p10), p90_ms=float(p90), samples=len(times_ms))


def do_bench_samples(fn: Callable[[], None], rep_ms: int = REP_MS) -> list[float]:
    """Raw per-sample times in ms from triton.testing.do_bench (zero flush).

    No longer the harness timer: do_bench clears L2 by writing zeros, which
    leaves L2 full of modified lines that the timed call then writes back, so
    it reads a memory-bound kernel high (a 442 us gather as 614 us on an L4).
    Kept for bench.timer_check, which compares the two.
    """
    from triton.testing import do_bench

    times: list[float] = list(do_bench(fn, warmup=WARMUP_MS, rep=rep_ms, return_mode="all"))
    return times


_READ_FLUSH: list[Callable[[], None]] = []  # made once per process


def read_flush() -> Callable[[], None]:
    """Clear L2 by reading (summing) a buffer larger than it.

    L2 is left holding unmodified lines of the buffer, which cost the next call
    nothing to evict, so the timed call reads its inputs from device memory and
    pays for no one else's writes.
    """
    if not _READ_FLUSH:
        import torch
        from triton import runtime

        cache = runtime.driver.active.get_empty_cache_for_benchmark()
        total = torch.empty((), dtype=torch.int64, device=cache.device)

        def flush() -> None:
            torch.sum(cache, dim=0, dtype=torch.int64, out=total)

        _READ_FLUSH.append(flush)
    return _READ_FLUSH[0]


def read_flush_samples(fn: Callable[[], None], rep_ms: int = REP_MS) -> list[float]:
    """The harness timer: raw per-sample times in ms with a cold, clean L2.

    The same loop as triton.testing.do_bench with the read flush in place of
    its zero flush:

        GPU stream  … │ read 256 MiB │ start │ fn │ end │ read 256 MiB │ start │ fn │ end │ …
                                     └──────────────────┘ one sample

    The flush keeps the GPU busy while the CPU queues the start event and the
    call, so a sample holds no launch overhead.
    """
    import torch

    flush = read_flush()
    fn()
    torch.cuda.synchronize()
    event = torch.cuda.Event
    begin, finish = event(enable_timing=True), event(enable_timing=True)  # type: ignore[no-untyped-call]
    begin.record()
    for _ in range(5):
        flush()
        fn()
    finish.record()
    torch.cuda.synchronize()
    estimate_ms = max(float(begin.elapsed_time(finish)) / 5, 1e-3)
    for _ in range(max(1, int(WARMUP_MS / estimate_ms))):
        fn()
    n = max(1, int(rep_ms / estimate_ms))
    starts = [event(enable_timing=True) for _ in range(n)]  # type: ignore[no-untyped-call]
    ends = [event(enable_timing=True) for _ in range(n)]  # type: ignore[no-untyped-call]
    for start, end in zip(starts, ends, strict=True):
        flush()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [float(a.elapsed_time(b)) for a, b in zip(starts, ends, strict=True)]


def timed_samples(fn: Callable[[], None], timer: Timer) -> list[float]:
    """Run the timer; if it returns fewer than MIN_SAMPLES, retry with a longer rep."""
    times = timer(fn)
    if len(times) >= MIN_SAMPLES or timer not in (read_flush_samples, do_bench_samples):
        return times
    scale = MIN_SAMPLES / max(len(times), 1) * 1.2
    return timer(fn, rep_ms=int(REP_MS * scale) + 1)


def time_with_reruns(fn: Callable[[], None], timer: Timer) -> tuple[Timing, bool]:
    """Time fn; rerun a noisy case and keep the run with the lowest spread."""
    best = summarize(timed_samples(fn, timer))
    for _ in range(EXTRA_RUNS):
        if not best.noisy:
            break
        again = summarize(timed_samples(fn, timer))
        if again.spread < best.spread:
            best = again
    return best, best.noisy


def host_us_per_call(
    impl: Impl, inputs: Tensors, calls: int = HOST_CALLS, seconds: float = HOST_SECONDS
) -> float:
    """CPU time per call of an implementation, in microseconds.

    Averages over `calls` calls, or over the calls made before `seconds` have
    passed if that comes first. Without the time limit, an implementation whose
    calls each queue milliseconds of GPU work (one copy job per block, tens of
    thousands of blocks) would spend minutes here, and once the GPU's launch
    queue filled, each call could wait for the GPU and the average would include
    that wait.

    Reported separately from GPU time: engines run decode under CUDA graphs,
    where this cost is paid once at capture.
    """
    import torch

    impl(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    made = 0
    start = time.perf_counter()
    while made < calls:
        impl(inputs)
        made += 1
        if time.perf_counter() - start >= seconds:
            break
    elapsed = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return elapsed / made * 1e6


def plan_us_per_call(plan: Impl, inputs: Tensors, calls: int = PLAN_CALLS) -> float:
    """CPU time per call of a plan on the case's own inputs, in microseconds.

    Unlike host_us, a plan's cost grows with the batch, so it runs on the full inputs.
    """
    import torch

    plan(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        plan(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / calls * 1e6


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

    def outputs_on_host(inputs: Tensors) -> Tensors:
        import torch

        # .cpu() of a tensor already in host memory returns it without waiting for
        # the GPU, so wait here for copies still in flight to a pinned host output.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {out: inputs[out].cpu() for out in case.outputs}

    ref_inputs = case.prepared(names[0], case.setup_for(names[0]))
    case.impls[names[0]](ref_inputs)
    ref = outputs_on_host(ref_inputs)
    del ref_inputs
    ref32 = (
        outputs_on_host(case.reference_fp32(case.setup()))
        if case.reference_fp32 is not None
        else None
    )

    for name in names[1:]:
        inputs = case.prepared(name, case.setup_for(name))
        case.impls[name](inputs)
        got = outputs_on_host(inputs)
        del inputs
        for out in case.outputs:
            where = f"{name} vs {names[0]}, output {out!r}, params {case.params}"
            if case.check == "exact":
                try:
                    assert_exact(got[out], ref[out])
                except AssertionError as exc:
                    raise CorrectnessError(f"{where}: {exc}") from exc
            else:
                assert ref32 is not None
                result = check_close(got[out], ref[out], ref32[out])
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
    plan_us: float | None = None,
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
        plan_us=plan_us,
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
    timer: Timer = read_flush_samples,
) -> BenchResult:
    if spec is not None and flush_bytes <= 2 * spec.l2_bytes:
        raise RuntimeError(
            f"L2 flush buffer ({flush_bytes} B) must exceed twice the L2 ({spec.l2_bytes} B)"
        )
    import torch

    measurements: list[Measurement] = []
    left_out: list[LeftOut] = []
    for case in cases:
        try:
            measurements.extend(_run_case(name, case, spec, measured_peak_gbps, timer))
        except torch.OutOfMemoryError as exc:
            reason = f"out of GPU memory: {str(exc).splitlines()[0]}"
            left_out.append(LeftOut(params=case.params, reason=reason))
            print(f"{name} {case.params}: left out, {reason}", flush=True)
            torch.cuda.empty_cache()
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
        left_out=left_out,
    )


def _run_case(
    name: str,
    case: Case,
    spec: GPUSpec | None,
    measured_peak_gbps: float | None,
    timer: Timer,
) -> list[Measurement]:
    """Check one case, then time each implementation; all or nothing."""
    check_case(case)
    measurements = []
    for impl_name, impl in case.impls.items():
        inputs = case.prepared(impl_name, case.setup_for(impl_name))
        timing, noisy = time_with_reruns(functools.partial(impl, inputs), timer)
        host_us = (
            host_us_per_call(impl, case.prepared(impl_name, case.host_setup()))
            if case.host_setup is not None
            else None
        )
        plan = case.plans.get(impl_name) if case.plans else None
        plan_us = plan_us_per_call(plan, inputs) if plan is not None else None
        del inputs
        measurements.append(
            measure(case, impl_name, timing, noisy, spec, measured_peak_gbps, host_us, plan_us)
        )
        print(
            f"{name} {case.params} {impl_name}: median {timing.median_ms * 1e3:.2f} us"
            f" (p10 {timing.p10_ms * 1e3:.2f}, p90 {timing.p90_ms * 1e3:.2f},"
            f" n={timing.samples}){' NOISY' if noisy else ''}",
            flush=True,
        )
    return measurements


def git_sha(repo: Path = Path(".")) -> str:
    """HEAD commit, prefixed with "dirty-" when the working tree has uncommitted changes.

    KVKERNELS_GIT_SHA, when set, is used instead: a copy of the source without
    .git (a Kaggle session) carries the value computed where the copy was made.
    """
    if sha := os.environ.get("KVKERNELS_GIT_SHA"):
        return sha
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
    """Size of the buffer the timer reads before each sample (Triton's 256 MiB benchmark buffer)."""
    from triton import runtime

    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    return int(cache.numel() * cache.element_size())


def measure_copy_peak(timer: Timer = read_flush_samples) -> float:
    """Measured peak bandwidth in GB/s: a 1 GiB device copy, one read and one write."""
    from bench.suites.smoke import PROBE_BYTES, copy_case

    case = copy_case(PROBE_BYTES)
    inputs = case.setup()
    impl = case.impls["torch_copy"]
    timing = summarize(timed_samples(functools.partial(impl, inputs), timer))
    return 2 * PROBE_BYTES / (timing.median_ms * 1e-3) / 1e9


EXPERIMENTS_PREFIX = "experiments-"


def suite_module(suite: str) -> str:
    """Module that defines a suite.

    A suite named experiments-<name> lives next to its kernel, in
    experiments/<name>/bench_suite.py; every other suite in bench/suites/<name>.py.
    """
    if suite.startswith(EXPERIMENTS_PREFIX):
        return f"experiments.{suite.removeprefix(EXPERIMENTS_PREFIX)}.bench_suite"
    return f"bench.suites.{suite}"


def load_cases(suite: str) -> Iterator[Case]:
    """The suite's cases, made one at a time.

    Never collect them in a list: an implementation keeps the GPU memory its
    plan allocated (a FlashInfer workspace is 128 MiB), so a list of every case
    holds every case's memory until the suite ends. On a 24 GB GPU the cascade
    suite ran out of memory a third of the way through that way.
    """
    module = importlib.import_module(suite_module(suite))
    yield from module.cases()


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
    peak = measure_copy_peak()
    print(f"measured copy peak: {peak:.1f} GB/s", flush=True)
    result = run_suite(
        args.suite,
        (c for c in load_cases(args.suite) if matches(c.params, filters)),
        lookup(info.gpu_name),
        env=info,
        git_sha=git_sha(),
        flush_bytes=flush_buffer_bytes(),
        measured_peak_gbps=peak,
    )
    if not result.measurements and not result.left_out:
        print("harness: no cases match", file=sys.stderr)
        return 2
    path = write_result(result, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
