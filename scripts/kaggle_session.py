"""Run scripts/gpu_session.py on a Kaggle GPU, from a development machine.

Needs the Kaggle CLI (``uv tool install kaggle``) and an API token
(``KAGGLE_API_TOKEN`` or ``~/.kaggle/access_token``).

    python scripts/kaggle_session.py push --user <name> [--steps ...]
    python scripts/kaggle_session.py push --user <name> --command "<shell command>"
    python scripts/kaggle_session.py status --user <name>
    python scripts/kaggle_session.py fetch --user <name>
    python scripts/gpu_session.py collect gpu-sessions/kaggle/<time>/session

push writes a private script that carries the source bundle inside it (a
gzipped tar, base64-encoded, about 0.4 MB), and pushes it as a new version of
one private notebook, which Kaggle then runs:

    this machine                          Kaggle (private)
    run script + bundle ──▶ notebook <user>/kvkernels-gpu-session, GPU, internet on
                               │  unpack the bundle,
                               │  uv sync --extra cu126 --extra bench --extra baselines,
                               │  gpu_session.py run --out /kaggle/working/session
                               ▼
    fetch ◀─────────────── notebook output: manifest, logs, result files

The bundle travels inside the script rather than as a Kaggle dataset: Kaggle
processes a new dataset version asynchronously, and a notebook started before
that finishes sees an empty input directory.

With --command, the notebook runs that command in the installed environment
instead of the session, and its output is in the notebook log
(kvkernels-gpu-session.log in what fetch downloads). Use it for a one-off
measurement; it writes nothing that collect accepts.

The bundle holds only what the session runs: the package, tests, benchmarks,
experiments, scripts and the build files.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "gpu-sessions" / "kaggle"
KERNEL_SLUG = "kvkernels-gpu-session"
BUNDLE_FILES = ("pyproject.toml", "uv.lock", "Makefile", "README.md", "LICENSE")
BUNDLE_DIRS = ("src", "tests", "bench", "experiments", "scripts")
SKIPPED_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
# Kaggle's only GPU (two Tesla T4s); the CLI accepts other names and runs them on a CPU.
ACCELERATORS = ("NvidiaTeslaT4",)
MAX_SECONDS = 12 * 3600  # Kaggle's longest session


def bundle_paths(root: Path = ROOT) -> Iterator[Path]:
    """The files the session needs, relative to root, in a fixed order."""
    for name in BUNDLE_FILES:
        yield Path(name)
    for top in BUNDLE_DIRS:
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root)
            if path.is_file() and not SKIPPED_PARTS & set(rel.parts) and path.suffix != ".pyc":
                yield rel


def bundle_bytes(root: Path = ROOT) -> bytes:
    """A gzipped tar of bundle_paths under kvkernels/, with fixed owners and times."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for rel in bundle_paths(root):
            data = (root / rel).read_bytes()
            info = tarfile.TarInfo(f"kvkernels/{rel.as_posix()}")
            info.size = len(data)
            info.mode = 0o755 if rel.suffix == ".sh" else 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def kernel_metadata(user: str, accelerator: str) -> dict[str, object]:
    return {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": accelerator,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


RUN_SCRIPT = '''"""On Kaggle: unpack the source, install the GPU environment, run the session."""

import base64
import io
import os
import subprocess
import sys
import tarfile

STEPS = {steps!r}
COMMAND = {command!r}
GIT_SHA = {git_sha!r}
BUNDLE = {bundle!r}

with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as tar:
    tar.extractall("/tmp", filter="data")
repo = "/tmp/kvkernels"


def sh(command):
    print("$", command, flush=True)
    return subprocess.run(command, shell=True, cwd=repo).returncode


# The source has no .git; result files record the commit the bundle was made from.
os.environ["KVKERNELS_GIT_SHA"] = GIT_SHA

INSTALL = "uv sync --locked --extra cu126 --extra bench --extra baselines"
if sh("nvidia-smi"):
    sys.exit("no GPU attached")
if sh("pip install -q uv==0.13.5") or sh(INSTALL):
    sys.exit("environment install failed")
# uv run keeps the CUDA torch just installed instead of syncing to PyPI's.
os.environ["UV_NO_SYNC"] = "1"

# FlashInfer links the kernels it compiles with -lcuda, and the image has no
# unversioned libcuda.so, so give the linker one: a link to the driver library
# that a CUDA process loads, read from that process's memory map.
FIND_LIBCUDA = (
    "import torch; torch.cuda.init(); "
    "print(next(line.split()[-1] for line in open('/proc/self/maps') if 'libcuda.so' in line))"
)
found = subprocess.run(
    ["uv", "run", "python", "-c", FIND_LIBCUDA], cwd=repo, capture_output=True, text=True
)
lines = found.stdout.split()
driver = lines[-1] if found.returncode == 0 and lines else None
if driver:
    os.makedirs("/tmp/cuda-link", exist_ok=True)
    if not os.path.exists("/tmp/cuda-link/libcuda.so"):
        os.symlink(driver, "/tmp/cuda-link/libcuda.so")
    os.environ["LIBRARY_PATH"] = "/tmp/cuda-link:" + os.environ.get("LIBRARY_PATH", "")
print("libcuda for linking:", driver or "not found", found.stderr[-500:], flush=True)
if COMMAND:
    sys.exit(sh(f"PYTHONUNBUFFERED=1 uv run {{COMMAND}}"))
steps = f" --steps {{STEPS}}" if STEPS else ""
# Unbuffered, so the notebook log shows when each line was printed.
session = "PYTHONUNBUFFERED=1 uv run python scripts/gpu_session.py run"
sys.exit(sh(f"{{session}} --out /kaggle/working/session{{steps}}"))
'''


def run_script(
    steps: str | None, bundle: bytes, command: str | None = None, sha: str = "unknown"
) -> str:
    return RUN_SCRIPT.format(
        steps=steps or "",
        command=command or "",
        git_sha=sha,
        bundle=base64.b64encode(bundle).decode(),
    )


def git_sha(root: Path = ROOT) -> str:
    """HEAD, prefixed with "dirty-" when the bundled files differ from it (as bench.harness)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = git("status", "--porcelain", "--", *BUNDLE_FILES, *BUNDLE_DIRS).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"dirty-{sha}" if dirty else sha


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found; install it with `uv tool install kaggle`")
    print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run(["kaggle", *args], check=check, capture_output=True, text=True)


def push(
    user: str, accelerator: str, steps: str | None, timeout: int, command: str | None = None
) -> None:
    folder = WORK / "kernel"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "run.py").write_text(run_script(steps, bundle_bytes(), command, git_sha()))
    metadata = kernel_metadata(user, accelerator)
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    result = kaggle(
        "kernels", "push", "-p", str(folder), "--accelerator", accelerator, "-t", str(timeout)
    )
    print(result.stdout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("push", "status", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--user", required=True, help="Kaggle username")
        if name == "push":
            p.add_argument("--accelerator", choices=ACCELERATORS, default="NvidiaTeslaT4")
            p.add_argument("--steps", help="passed to gpu_session.py run --steps")
            p.add_argument("--command", help="run this with `uv run` instead of the session")
            p.add_argument("--timeout", type=int, default=MAX_SECONDS, help="seconds")
    args = parser.parse_args(argv)

    kernel = f"{args.user}/{KERNEL_SLUG}"
    if args.action == "push":
        if args.command and args.steps:
            parser.error("--command and --steps cannot be used together")
        push(args.user, args.accelerator, args.steps, args.timeout, args.command)
    elif args.action == "status":
        print(kaggle("kernels", "status", kernel).stdout)
    else:
        target = WORK / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        kaggle("kernels", "output", kernel, "-p", str(target), "-o", "-q")
        print(
            f"downloaded to {target}; then: python scripts/gpu_session.py collect {target}/session"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
