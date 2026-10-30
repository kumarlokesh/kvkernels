from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip gpu-marked tests when no CUDA device is visible."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def _language_objects() -> list[Any]:
    """The objects whose attributes Triton's interpreter replaces while it runs a kernel."""
    try:
        import triton.language as tl
    except ImportError:
        return []
    return [tl, tl.core, tl.math, tl.tensor, tl.dtype, tl.core.tensor_descriptor_base]


def _snapshot() -> list[tuple[Any, dict[str, Any]]]:
    return [(obj, dict(vars(obj))) for obj in _language_objects()]


# Taken when pytest imports this file, before any kernel runs.
_CLEAN_LANGUAGE = _snapshot()


@pytest.fixture(autouse=True)
def clean_triton_language() -> Iterator[None]:
    """Undo the interpreter's leftover replacements of triton.language before each test.

    Under TRITON_INTERPRET=1, Triton 3.8.0 swaps triton.language functions for
    interpreter versions while a kernel runs. A call from a kernel into another
    @triton.jit function (tl.max, tl.sum and tl.cdiv are such functions) swaps
    them again and never swaps them back. Real compilation in the same process,
    such as the IR dump tool's tests, then fails.
    """
    for obj, clean in _CLEAN_LANGUAGE:
        current = vars(obj)
        for name in [n for n in current if n not in clean]:
            delattr(obj, name)
        for name, value in clean.items():
            if current.get(name) is not value:
                setattr(obj, name, value)
    yield
