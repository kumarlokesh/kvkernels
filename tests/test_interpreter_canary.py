"""Canary: bf16 tl.dot is wrong under the Triton 3.8.0 interpreter.

The interpreter stores bf16 values as raw 16-bit integers and its tl.dot multiplies
them without converting, so bf16 tests run on GPUs only. If this test fails after
a Triton upgrade, the interpreter has been fixed: allow bf16 in CPU tests and
delete this test.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="checks the Triton interpreter only"
)


@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):  # type: ignore[name-defined]
    r = tl.arange(0, N)
    a = tl.load(a_ptr + r[:, None] * N + r[None, :])
    b = tl.load(b_ptr + r[:, None] * N + r[None, :])
    tl.store(c_ptr + r[:, None] * N + r[None, :], tl.dot(a, b))


def _dot_error(dtype: Any, n: int = 16) -> float:
    torch.manual_seed(0)
    a = torch.randn(n, n).to(dtype)
    b = torch.randn(n, n).to(dtype)
    c = torch.empty(n, n, dtype=torch.float32)
    _dot_kernel[(1,)](a, b, c, N=n)
    return float((c - a.float() @ b.float()).abs().max())


def test_fp16_dot_is_exact_under_interpreter() -> None:
    assert _dot_error(torch.float16) == 0.0


def test_bf16_dot_is_still_wrong_under_interpreter() -> None:
    assert _dot_error(torch.bfloat16) > 1.0
