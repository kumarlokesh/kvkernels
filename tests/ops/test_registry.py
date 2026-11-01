"""When an op call goes through PyTorch's custom-op dispatch and when it skips it."""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from kvkernels.ops._registry import registered_op

calls: list[str] = []


@registered_op("kvkernels_test::record", mutates_args=("out",))
def _record(x: torch.Tensor, out: torch.Tensor) -> None:
    calls.append("body")
    out.copy_(x)


@_record.register_fake
def _record_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    calls.append("fake")


def test_an_eager_call_runs_the_body_directly() -> None:
    calls.clear()
    x, out = torch.ones(4), torch.zeros(4)
    _record(x, out)
    assert calls == ["body"]
    assert torch.equal(out, x)


def test_a_call_on_fake_tensors_goes_through_the_registered_op() -> None:
    calls.clear()
    with FakeTensorMode() as mode:
        x, out = mode.from_tensor(torch.ones(4)), mode.from_tensor(torch.zeros(4))
        _record(x, out)
    assert calls == ["fake"]


def test_the_registered_op_is_reachable_by_name() -> None:
    calls.clear()
    x, out = torch.ones(4), torch.zeros(4)
    torch.ops.kvkernels_test.record(x, out)
    assert calls == ["body"]
