"""Skip gpu-marked tests when no CUDA device is visible, as under tests/."""

from tests.conftest import pytest_collection_modifyitems

__all__ = ["pytest_collection_modifyitems"]
