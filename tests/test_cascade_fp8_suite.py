"""The fp8 cascade suite: every implementation sees the same dequantized values."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from bench.suites import cascade as cascade_suite  # noqa: E402
from bench.suites import cascade_fp8 as suite  # noqa: E402


def test_grid() -> None:
    params = list(suite.grid())
    assert len(params) == len(suite.BATCHES) * len(suite.PREFIXES) * len(suite.SUFFIXES)


def test_fp8_and_fp16_runs_agree_on_a_small_case() -> None:
    from tests.ops.cascade_cases import DEVICE, INTERPRETING

    if not INTERPRETING and not torch.cuda.is_available():
        pytest.skip("needs the Triton interpreter or a CUDA GPU")
    p = cascade_suite._params(batch=4, prefix=64, suffix=16, model="llama-3.2-1b")
    case = suite.make_case(p, with_flashinfer=False, device=DEVICE)
    t = case.setup()
    assert t["k8"].dtype == torch.float8_e4m3fn
    assert torch.equal((t["k8"].float() * t["k_scale"]).half(), t["k"])
    outs = {}
    for name in ("kvkernels", "kvkernels_fp8"):
        t = case.prepared(name, case.setup())
        case.impls[name](t)
        outs[name] = t["out"]
    assert torch.equal(outs["kvkernels"], outs["kvkernels_fp8"])
