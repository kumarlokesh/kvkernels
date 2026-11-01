"""Register an op with PyTorch, and call it without PyTorch's dispatch when nothing needs it.

Each op is registered as a PyTorch custom op, so torch.compile and engines treat a
call as one opaque operation that writes into the tensors named in mutates_args.
Going through the registration costs CPU time on every call: on the host of a
Kaggle T4, a gather call spent 67 µs of its 164 µs there (bench/host_cost.py).
A plain eager call needs none of it, so the op runs its body directly:

    torch.compile tracing ───────────┐
    a dispatch mode (fake tensors) ──┴──▶ the registered custom op
    otherwise ───────────────────────────▶ the body, called directly

The direct call skips the version-counter bump PyTorch's autograd uses to detect
in-place changes to saved tensors; the ops are for inference and define no
gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, ParamSpec

import torch

P = ParamSpec("P")


class RegisteredOp(Generic[P]):
    def __init__(self, name: str, body: Callable[P, None], mutates_args: Sequence[str]) -> None:
        self.body = body
        self.custom_op = torch.library.custom_op(name, body, mutates_args=tuple(mutates_args))

    def register_fake(self, fake: Callable[P, None]) -> Callable[P, None]:
        self.custom_op.register_fake(fake)
        return fake

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        if torch.compiler.is_compiling() or torch._C._len_torch_dispatch_stack() > 0:
            self.custom_op(*args, **kwargs)
        else:
            self.body(*args, **kwargs)


def registered_op(
    name: str, *, mutates_args: Sequence[str]
) -> Callable[[Callable[P, None]], RegisteredOp[P]]:
    """Decorator: register the function as custom op `name` and return a RegisteredOp."""

    def wrap(body: Callable[P, None]) -> RegisteredOp[P]:
        return RegisteredOp(name, body, mutates_args)

    return wrap
