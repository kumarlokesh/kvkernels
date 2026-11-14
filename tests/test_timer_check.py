from __future__ import annotations

import pytest

from bench.timer_check import TIMERS, ProbeResult, decision, parse_ncu_durations_ns

NCU_OUTPUT = """==PROF== Connected to process 4242
==PROF== Profiling "_one_program_kernel": 0%....50%....100% - 1 pass
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","4242","python","127.0.0.1","_one_program_kernel","1","7","Command line profiler metrics","gpu__time_duration.sum","nsecond","2,048"
"1","4242","python","127.0.0.1","_one_program_kernel","1","7","Command line profiler metrics","gpu__time_duration.sum","usecond","3.5"
==PROF== Disconnected from process 4242
"""


def test_parse_ncu_durations_reads_values_and_units() -> None:
    assert parse_ncu_durations_ns(NCU_OUTPUT) == [2048.0, 3500.0]


def test_parse_ncu_rejects_unknown_unit() -> None:
    bad = NCU_OUTPUT.replace('"usecond"', '"fortnight"')
    with pytest.raises(ValueError, match="fortnight"):
        parse_ncu_durations_ns(bad)


def probe(name: str, ncu: float, **timers: float) -> ProbeResult:
    return ProbeResult(name, ncu, {t: timers.get(t, 1e9) for t in TIMERS})


def test_within_ten_percent_of_ncu() -> None:
    assert probe("p", 10.0, do_bench=10.9).within("do_bench")
    assert not probe("p", 10.0, do_bench=11.5).within("do_bench")
    assert not probe("p", 10.0, do_bench=8.5).within("do_bench")


def test_decision_considers_only_small_kernels() -> None:
    small_ok = probe("a", 5.0, do_bench=5.2)
    large_off = probe("b", 100.0, do_bench=130.0)
    assert decision([small_ok, large_off]).startswith("do_bench")


def test_decision_takes_the_first_timer_in_order_that_agrees() -> None:
    a = probe("a", 5.0, do_bench=8.0, events_read=5.3, graph_read=5.0)
    b = probe("b", 20.0, do_bench=40.0, events_read=21.0, graph_read=20.0)
    assert decision([a, b]).startswith("events_read")
    c = probe("c", 5.0, do_bench=8.0)
    assert decision([a, c]).startswith("no timer")
