"""The scripts in examples/ run as written, on the CPU under the Triton interpreter."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("torch")

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def test_there_are_examples() -> None:
    assert [p.name for p in EXAMPLES] == ["cascade.py", "gather.py"]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(path: Path) -> None:
    runpy.run_path(str(path), run_name="__main__")
