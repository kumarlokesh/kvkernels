from __future__ import annotations

import dataclasses
import gc
import time
import weakref
from collections.abc import Callable, Iterator

import pytest

torch = pytest.importorskip("torch")

from bench import harness  # noqa: E402
from bench.harness import Case, CorrectnessError, Tensors  # noqa: E402
from bench.schema import EnvInfo  # noqa: E402
from kvkernels.specs import GPUSpec  # noqa: E402

ENV = EnvInfo("cpu", "cpu", "-", "-", "-", "-", None, None, "0" * 64)

SPEC = GPUSpec(
    slug="fake",
    name="Fake GPU",
    peak_hbm_gbps=1000.0,
    peak_fp16_tensor_tflops_dense_fp16acc=200.0,
    peak_fp16_tensor_tflops_dense_fp32acc=100.0,
    peak_bf16_tensor_tflops_dense_fp32acc=100.0,
    l2_bytes=50 << 20,
    smem_per_sm_bytes=228 << 10,
    num_sms=100,
    sources=("https://example.invalid/fake",),
)


def fixed_timer(*runs: list[float]) -> Callable[[Callable[[], None]], list[float]]:
    """A timer that returns the given sample lists in order, without calling fn."""
    it: Iterator[list[float]] = iter(runs)
    return lambda fn: next(it)


def copy_case(impls: dict[str, Callable[[Tensors], None]] | None = None) -> Case:
    def setup() -> Tensors:
        src = torch.arange(16, dtype=torch.float32)
        return {"src": src, "dst": torch.zeros_like(src)}

    def good(t: Tensors) -> None:
        t["dst"].copy_(t["src"])

    return Case(
        params={"bytes": 1000, "dtype": "fp16"},
        setup=setup,
        impls=impls or {"ref": good, "other": good},
        outputs=("dst",),
        check="exact",
        bytes_moved=lambda p: int(p["bytes"]),
        flops=lambda p: 2_000_000,
    )


def test_summarize_reports_percentiles() -> None:
    t = harness.summarize([float(i) for i in range(1, 102)])
    assert (t.p10_ms, t.median_ms, t.p90_ms, t.samples) == (11.0, 51.0, 91.0, 101)


def test_derived_metrics_use_median_and_decimal_units() -> None:
    timing = harness.Timing(median_ms=1e-3, p10_ms=1e-3, p90_ms=1e-3, samples=100)
    m = harness.measure(copy_case(), "ref", timing, False, SPEC, 500.0, None)
    assert m.gbps == pytest.approx(1000 / 1e-6 / 1e9)  # 1 GB/s
    assert m.pct_datasheet_bw == pytest.approx(1.0 / 1000)
    assert m.pct_measured_bw == pytest.approx(1.0 / 500)
    assert m.tflops == pytest.approx(2e6 / 1e-6 / 1e12)  # 2 TFLOP/s
    assert m.pct_peak_compute == pytest.approx(2.0 / 100)  # fp16 with fp32 accumulation


def test_percentages_are_empty_without_spec_or_peak() -> None:
    timing = harness.Timing(1.0, 1.0, 1.0, 100)
    m = harness.measure(copy_case(), "ref", timing, False, None, None, None)
    assert m.gbps is not None
    assert (m.pct_datasheet_bw, m.pct_measured_bw, m.pct_peak_compute) == (None, None, None)


def test_quiet_case_runs_once() -> None:
    timer = fixed_timer([1.0] * 100)
    timing, noisy = harness.time_with_reruns(lambda: None, timer)
    assert not noisy and timing.samples == 100


def test_noisy_case_reruns_and_keeps_lowest_spread() -> None:
    noisy_run = [1.0] * 80 + [2.0] * 20  # p90 / median = 2.0
    less_noisy = [1.0] * 85 + [1.5] * 15  # p90 / median = 1.5
    timer = fixed_timer(noisy_run, less_noisy, noisy_run)
    timing, noisy = harness.time_with_reruns(lambda: None, timer)
    assert noisy
    assert timing.spread == pytest.approx(1.5)


def test_noisy_case_that_settles_is_not_flagged() -> None:
    timer = fixed_timer([1.0] * 80 + [2.0] * 20, [1.0] * 100)
    _, noisy = harness.time_with_reruns(lambda: None, timer)
    assert not noisy


def test_a_high_spread_within_the_timer_resolution_is_not_noisy() -> None:
    # A 5 us kernel whose p90 is 1 us above its median: spread 1.2, gap 0.001 ms.
    timer = fixed_timer([0.005] * 80 + [0.006] * 20)
    timing, noisy = harness.time_with_reruns(lambda: None, timer)
    assert timing.spread == pytest.approx(1.2)
    assert not noisy


def test_wrong_implementation_stops_the_suite() -> None:
    def good(t: Tensors) -> None:
        t["dst"].copy_(t["src"])

    def off_by_one(t: Tensors) -> None:
        t["dst"].copy_(t["src"] + 1)

    case = copy_case({"ref": good, "bad": off_by_one})
    with pytest.raises(CorrectnessError, match="bad vs ref"):
        harness.run_suite(
            "t",
            [case],
            None,
            env=ENV,
            git_sha="x",
            flush_bytes=1,
            measured_peak_gbps=None,
            timer=fixed_timer(),
        )


def test_a_case_out_of_gpu_memory_is_left_out_and_the_suite_goes_on() -> None:
    def good(t: Tensors) -> None:
        t["dst"].copy_(t["src"])

    def too_big(t: Tensors) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 8.00 GiB.\nmore")

    big = dataclasses.replace(copy_case({"ref": good, "big": too_big}), params={"bytes": 8})
    result = harness.run_suite(
        "t",
        [big, copy_case()],
        None,
        env=ENV,
        git_sha="x",
        flush_bytes=1,
        measured_peak_gbps=None,
        timer=fixed_timer([1.0] * 100, [2.0] * 100),
    )
    assert [m.impl for m in result.measurements] == ["ref", "other"]
    assert [lo.params for lo in result.left_out] == [{"bytes": 8}]
    assert result.left_out[0].reason == (
        "out of GPU memory: CUDA out of memory. Tried to allocate 8.00 GiB."
    )


def test_a_finished_case_is_released_before_the_next_one_runs() -> None:
    # A case keeps whatever its implementations' plans allocated; holding finished
    # cases would hold that GPU memory until the suite ends.
    refs: list[weakref.ref[Case]] = []
    released: list[bool] = []

    def good(t: Tensors) -> None:
        t["dst"].copy_(t["src"])

    def check_first_is_gone(t: Tensors) -> None:
        gc.collect()
        released.append(refs[0]() is None)
        good(t)

    def remember(case: Case) -> Case:
        refs.append(weakref.ref(case))
        return case

    def cases() -> Iterator[Case]:
        yield remember(copy_case())
        yield copy_case({"ref": check_first_is_gone, "other": good})

    harness.run_suite(
        "t",
        cases(),
        None,
        env=ENV,
        git_sha="x",
        flush_bytes=1,
        measured_peak_gbps=None,
        timer=fixed_timer(*([[1.0] * 100] * 4)),
    )
    assert released and all(released)


def test_git_sha_comes_from_the_environment_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KVKERNELS_GIT_SHA", "dirty-0123abc")
    assert harness.git_sha() == "dirty-0123abc"


def test_host_cost_stops_at_the_call_count_or_the_time_limit() -> None:
    calls = 0

    def impl(t: Tensors) -> None:
        nonlocal calls
        calls += 1

    harness.host_us_per_call(impl, {}, calls=50)
    assert calls == 1 + 50  # one call before timing starts

    def slow(t: Tensors) -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.01)

    calls = 0
    host_us = harness.host_us_per_call(slow, {}, calls=1000, seconds=0.05)
    assert 1 + 1 <= calls < 1 + 1000
    assert host_us >= 10_000


def test_an_implementation_with_its_own_setup_gets_its_own_inputs() -> None:
    seen: dict[str, set[str]] = {}

    def record(name: str) -> Callable[[Tensors], None]:
        def impl(t: Tensors) -> None:
            seen.setdefault(name, set()).update(t)
            t["dst"].copy_(t["src"])

        return impl

    def extra() -> Tensors:
        return {**copy_case().setup(), "extra": torch.zeros(1)}

    case = dataclasses.replace(copy_case({"a": record("a"), "b": record("b")}), setups={"b": extra})
    harness.check_case(case)
    assert seen == {"a": {"src", "dst"}, "b": {"src", "dst", "extra"}}


def test_tolerance_check_needs_fp32_reference() -> None:
    case = dataclasses.replace(copy_case(), check="tolerance")
    with pytest.raises(ValueError, match="reference_fp32"):
        harness.check_case(case)


def test_run_suite_builds_one_measurement_per_impl() -> None:
    result = harness.run_suite(
        "t",
        [copy_case()],
        SPEC,
        env=ENV,
        git_sha="abc",
        flush_bytes=256 << 20,
        measured_peak_gbps=None,
        timer=fixed_timer([1.0] * 100, [2.0] * 100),
    )
    assert [m.impl for m in result.measurements] == ["ref", "other"]
    assert [m.median_ms for m in result.measurements] == [1.0, 2.0]
    assert result.timer == "events-read-flush-cold-l2"


def test_a_plan_runs_before_its_implementation_and_is_timed_on_its_own() -> None:
    def plan(t: Tensors) -> None:
        t["planned"] = t["src"].clone()

    def ref(t: Tensors) -> None:
        t["dst"].copy_(t["src"])

    def needs_plan(t: Tensors) -> None:
        t["dst"].copy_(t["planned"])  # KeyError if the plan has not run

    case = dataclasses.replace(
        copy_case({"ref": ref, "planned": needs_plan}),
        plans={"planned": plan},
    )
    result = harness.run_suite(
        "t",
        [case],
        None,
        env=ENV,
        git_sha="x",
        flush_bytes=1,
        measured_peak_gbps=None,
        timer=fixed_timer([1.0] * 100, [1.0] * 100),
    )
    unplanned, planned = result.measurements
    assert unplanned.plan_us is None
    assert planned.plan_us is not None and planned.plan_us > 0


def test_flush_buffer_must_exceed_twice_l2() -> None:
    with pytest.raises(RuntimeError, match="twice the L2"):
        harness.run_suite(
            "t",
            [],
            SPEC,
            env=ENV,
            git_sha="x",
            flush_bytes=SPEC.l2_bytes * 2,
            measured_peak_gbps=None,
        )


def test_filters() -> None:
    assert harness.parse_filters(["a=1", "b=x"]) == {"a": "1", "b": "x"}
    assert harness.matches({"a": 1, "b": "x"}, {"a": "1"})
    assert not harness.matches({"a": 2}, {"a": "1"})
    with pytest.raises(ValueError):
        harness.parse_filters(["nonsense"])


def test_smoke_suite_is_discoverable() -> None:
    loaded = harness.load_cases("smoke")
    assert isinstance(loaded, Iterator) and not isinstance(loaded, list)  # one case at a time
    cases = list(loaded)
    assert [c.params["bytes"] for c in cases][-1] == 1 << 30


@pytest.mark.gpu
def test_the_timer_returns_enough_samples_on_gpu() -> None:
    x = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)

    def copy() -> None:
        y.copy_(x)

    times = harness.timed_samples(copy, harness.read_flush_samples)
    assert len(times) >= harness.MIN_SAMPLES


def test_suite_module_maps_experiment_suites_next_to_their_kernel() -> None:
    from bench.harness import suite_module

    assert suite_module("smoke") == "bench.suites.smoke"
    assert suite_module("experiments-matmul") == "experiments.matmul.bench_suite"
