"""The v0.1.0 public API: every listed name imports from its module."""

from __future__ import annotations

import importlib
from importlib.metadata import version

import pytest

import kvkernels
from scripts.wheel_smoke import PUBLIC


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_every_public_name_imports(module: str) -> None:
    mod = importlib.import_module(module)
    for name in PUBLIC[module]:
        assert hasattr(mod, name), f"{module}.{name}"
    assert sorted(mod.__all__) == sorted(PUBLIC[module])


def test_version_comes_from_the_installed_metadata() -> None:
    assert kvkernels.__version__ == version("kvkernels")
