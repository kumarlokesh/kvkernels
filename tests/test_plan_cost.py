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
    assert list(row.kernel_us) == ["order", "common", "items", "fill"]
