"""The host-cost probe runs and reports every layer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bench import host_cost  # noqa: E402


@pytest.mark.gpu
def test_every_layer_is_timed() -> None:
    result = host_cost.measure(calls=20)
    assert list(result) == list(host_cost.parts())
    assert all(us > 0 for us in result.values())
