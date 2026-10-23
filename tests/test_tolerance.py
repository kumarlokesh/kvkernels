from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kvkernels.tolerance import (  # noqa: E402
    FLOORS,
    assert_close,
    assert_exact,
    check_close,
    max_abs_error,
)


def test_allowed_error_is_twice_the_low_precision_error_plus_floor() -> None:
    ref32 = torch.tensor([1.0, 2.0, 3.0])
    ref_low = torch.tensor([1.0, 2.0, 3.01], dtype=torch.float16)
    result = check_close(ref_low.clone(), ref_low, ref32)
    low_err = max_abs_error(ref_low, ref32)
    assert result.allowed == pytest.approx(2 * low_err + FLOORS[torch.float16])
    assert result.ok


def test_error_above_allowed_fails() -> None:
    ref32 = torch.zeros(4)
    ref_low = torch.zeros(4, dtype=torch.float16)
    out = torch.tensor([0.0, 0.0, 0.0, 0.01], dtype=torch.float16)
    assert not check_close(out, ref_low, ref32).ok
    with pytest.raises(AssertionError):
        assert_close(out, ref_low, ref32)


def test_nan_in_output_fails_but_matching_nan_passes() -> None:
    ref32 = torch.tensor([float("nan"), 1.0])
    ref_low = ref32.half()
    assert check_close(torch.tensor([float("nan"), 1.0]).half(), ref_low, ref32).ok
    assert not check_close(torch.tensor([1.0, float("nan")]).half(), ref_low, ref32).ok


def test_matching_infinities_pass() -> None:
    ref32 = torch.tensor([float("-inf"), 0.0])
    assert check_close(ref32.half(), ref32.half(), ref32).ok


def test_empty_tensors_pass() -> None:
    empty = torch.empty(0)
    assert check_close(empty, empty, empty).ok


def test_exact_compares_bits() -> None:
    a = torch.tensor([0.0, 1.0])
    assert_exact(a, a.clone())
    with pytest.raises(AssertionError):
        assert_exact(torch.tensor([0.0]), torch.tensor([-0.0]))
    with pytest.raises(AssertionError):
        assert_exact(a, a.half())
