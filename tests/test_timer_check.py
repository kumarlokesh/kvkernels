from __future__ import annotations

import pytest

from bench.timer_check import ProbeResult, decision, parse_ncu_durations_ns

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


def test_within_ten_percent_is_hidden() -> None:
    assert ProbeResult("p", do_bench_us=10.9, graph_us=10.0, ncu_us=10.0).hidden
    assert not ProbeResult("p", do_bench_us=11.5, graph_us=10.0, ncu_us=10.0).hidden


def test_decision_considers_only_small_kernels() -> None:
    small_ok = ProbeResult("a", 5.2, 5.0, 5.0)
    large_off = ProbeResult("b", 130.0, 100.0, 100.0)
    assert decision([small_ok, large_off]).startswith("keep do_bench")
    small_off = ProbeResult("c", 8.0, 5.0, 5.0)
    assert decision([small_ok, small_off]).startswith("switch")
