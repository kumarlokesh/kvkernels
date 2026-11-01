"""Checks the release workflow runs on a tag before it builds anything.

    python scripts/release_check.py --tag v0.1.0 [--notes notes.md]

Fails unless the tag is "v" followed by the project version, and CHANGELOG.md has
a section for that version that ends with a "Tested on:" line naming where the
GPU tests passed. With --notes, writes that section's text, for the GitHub
release.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def project_version(root: Path = ROOT) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def changelog_section(changelog: str, version: str) -> str:
    """The text under "## [version]", up to the next "## " heading."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)",
        changelog,
        re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    return match.group(1).strip()


def check(tag: str, root: Path = ROOT) -> str:
    """The changelog section for the tag; raises ValueError if a check fails."""
    version = project_version(root)
    if tag != f"v{version}":
        raise ValueError(f"tag {tag} does not match the project version {version}")
    section = changelog_section((root / "CHANGELOG.md").read_text(), version)
    lines = [line for line in section.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith("Tested on:"):
        raise ValueError(f"the {version} section of CHANGELOG.md must end with a 'Tested on:' line")
    return section


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--notes", type=Path, help="write the changelog section here")
    args = parser.parse_args(argv)
    try:
        section = check(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.notes:
        args.notes.write_text(section + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
