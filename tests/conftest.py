from __future__ import annotations

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
