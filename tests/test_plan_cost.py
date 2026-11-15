"""The planning-cost probe runs and reports every measurement."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bench import plan_cost  # noqa: E402


@pytest.mark.gpu
def test_every_measurement_is_positive() -> None:
    (row,) = plan_cost.measure(batches=(16,), calls=5)
    assert row.batch == 16
    times = (row.host_plan_us, row.launches_cpu_us, row.gpu_plan_cpu_us, row.gpu_plan_us)
    assert min(*times, row.graph_step_us, *row.kernel_us.values()) > 0
    assert list(row.kernel_us) == ["tree", "items", "fill"]


@pytest.mark.gpu
def test_planning_a_tree_is_timed_for_each_max_levels() -> None:
    times = plan_cost.measure_levels(batch=32, levels=(1, 4))
    assert set(times) == {1, 4} and min(times.values()) > 0
