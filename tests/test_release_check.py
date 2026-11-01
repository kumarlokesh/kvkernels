"""The release workflow's tag and changelog checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import release_check


def project(tmp_path: Path, version: str, changelog: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    return tmp_path


CHANGELOG = """# Changelog

## [Unreleased]

## [0.1.0] - 2026-11-20

### Added

- the ops

Tested on: L4 (torch 2.14.1, Triton 3.8.0)

## [0.0.1] - 2026-11-01

Tested on: nothing
"""


def test_a_matching_tag_returns_the_section(tmp_path: Path) -> None:
    section = release_check.check("v0.1.0", project(tmp_path, "0.1.0", CHANGELOG))
    assert section.startswith("### Added")
    assert section.endswith("Tested on: L4 (torch 2.14.1, Triton 3.8.0)")
    assert "0.0.1" not in section


def test_a_tag_that_differs_from_the_version_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        release_check.check("v0.1.1", project(tmp_path, "0.1.0", CHANGELOG))


def test_a_section_without_a_tested_on_line_fails(tmp_path: Path) -> None:
    changelog = CHANGELOG.replace("Tested on: L4 (torch 2.14.1, Triton 3.8.0)\n", "")
    with pytest.raises(ValueError, match="Tested on"):
        release_check.check("v0.1.0", project(tmp_path, "0.1.0", changelog))


def test_a_version_without_a_section_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no section"):
        release_check.check("v0.2.0", project(tmp_path, "0.2.0", CHANGELOG))


def test_the_repository_changelog_has_an_unreleased_section() -> None:
    text = (release_check.ROOT / "CHANGELOG.md").read_text()
    assert "## [Unreleased]" in text
