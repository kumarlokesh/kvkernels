"""One GPU session: the GPU-bound steps of the roadmap, in order, each with a log.

On the GPU machine, from the repository root, after ``make install-cu126`` (or cu130):

    python scripts/gpu_session.py run --out <dir> [--steps env,test-gpu,...]

Back on a development machine, after copying <dir> over:

    python scripts/gpu_session.py collect <dir>

Steps, in the order they run:

    env            scripts/env_report.py; always runs, and runs first
    test-gpu       the package's GPU tests (tests/)
    test-gpu-experiments
                   the experiments' GPU tests (experiments/); about an hour on a T4,
                   so it runs only when chosen or when matmul or attention runs
    smoke          the smoke suite (make bench-smoke)
    timer          the timer check (make check-timer); needs Nsight Compute counters
    kv_gather      the kv_gather suite
    matmul         the K2 matmul suite; needs test-gpu-experiments
    attention      the K2 fused attention suite; needs test-gpu-experiments
    cascade        the cascade suite
    dram           the device-memory bytes run (make dram-cascade); needs the counters
    old-torch      the package's GPU tests (tests/ only) again with torch 2.13.0 and
                   Triton 3.7.1, in a separate environment (the release's lowest
                   supported versions); experiments/ follows the lock file only

Each step runs the command its Makefile target runs, without make, which some
GPU images lack. If env fails (no usable GPU), nothing else runs. A step that writes results
runs only after test-gpu (and, for the experiment suites, test-gpu-experiments)
has passed in the same session; choosing such a step adds the test steps it
needs. collect copies results only from a session where test-gpu passed, so no
result reaches results/ from a machine where the GPU tests failed. Every test
step sets KVKERNELS_REQUIRE_GPU=1, so a missing GPU fails it instead of skipping
every test. Any other failing step
is recorded and the session goes on with the next one.

The output directory holds manifest.json (each step's command, exit code,
duration and log file), one log per step, and a copy of every result file the
session wrote, under results/. The manifest and the copy are brought up to date
after every step, so a session stopped partway (a provider's time limit) keeps
what its finished steps wrote.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_TORCH_ENV = Path("/tmp/kvkernels-torch-2.13")
OLD_TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


@dataclass(frozen=True)
class Step:
    name: str
    command: str  # run with the shell, from the repository root
    writes_results: bool = False
    needs: tuple[str, ...] = ()  # test steps that must pass first


def _tests(path: str) -> str:
    return f"KVKERNELS_REQUIRE_GPU=1 uv run pytest {path} -m gpu -v --durations=20"


RESULT_TESTS = ("test-gpu",)
EXPERIMENT_TESTS = ("test-gpu", "test-gpu-experiments")


def _suite(name: str) -> str:
    return f"uv run python -m bench.harness --suite {name}"


STEPS = (
    Step("env", "uv run python scripts/env_report.py"),
    Step("test-gpu", _tests("tests")),
    Step("test-gpu-experiments", _tests("experiments")),
    Step("smoke", _suite("smoke"), writes_results=True, needs=RESULT_TESTS),
    Step("timer", "uv run python -m bench.timer_check"),
    Step("kv_gather", _suite("kv_gather"), writes_results=True, needs=RESULT_TESTS),
    Step("matmul", _suite("experiments-matmul"), writes_results=True, needs=EXPERIMENT_TESTS),
    Step(
        "attention",
        _suite("experiments-fused_attention_fwd"),
        writes_results=True,
        needs=EXPERIMENT_TESTS,
    ),
    Step("cascade", _suite("cascade"), writes_results=True, needs=RESULT_TESTS),
    Step("dram", "uv run python -m bench.dram_bytes", writes_results=True, needs=RESULT_TESTS),
    Step(
        "old-torch",
        f"uv venv -q --allow-existing {OLD_TORCH_ENV}"
        f" && uv pip install -q -p {OLD_TORCH_ENV} --index-url {OLD_TORCH_INDEX} torch==2.13.0"
        f" && uv pip install -q -p {OLD_TORCH_ENV} triton==3.7.1 numpy pytest"
        f" && KVKERNELS_REQUIRE_GPU=1 PYTHONPATH=src:."
        f" {OLD_TORCH_ENV}/bin/python -m pytest tests -m gpu -v -p no:cacheprovider",
    ),
)
STEP_NAMES = tuple(s.name for s in STEPS)


@dataclass
class StepRecord:
    name: str
    command: str
    returncode: int | None  # None when the step did not run
    seconds: float
    log: str | None
    skipped: str | None = None  # why the step did not run


@dataclass
class Manifest:
    started_utc: str
    steps: list[StepRecord] = field(default_factory=list)

    def passed(self, name: str) -> bool:
        return any(s.name == name and s.returncode == 0 for s in self.steps)

    def write(self, out: Path) -> None:
        (out / "manifest.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @staticmethod
    def read(out: Path) -> Manifest:
        data = json.loads((out / "manifest.json").read_text())
        return Manifest(data["started_utc"], [StepRecord(**s) for s in data["steps"]])


def select(names: Sequence[str] | None) -> list[Step]:
    """The steps to run, in session order; env and the tests they need are included."""
    if names is None:
        return list(STEPS)
    unknown = sorted(set(names) - set(STEP_NAMES))
    if unknown:
        raise ValueError(f"unknown steps {unknown}; known: {', '.join(STEP_NAMES)}")
    wanted = {"env", *names}
    wanted |= {need for s in STEPS if s.name in wanted for need in s.needs}
    return [s for s in STEPS if s.name in wanted]


Runner = Callable[[str, Path], int]


def shell_runner(command: str, log: Path) -> int:
    """Run command from the repository root; print its output and save it to log."""
    with log.open("w") as f:
        f.write(f"$ {command}\n")
        f.flush()
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=ROOT,
            # A pipe makes Python buffer its output; unbuffered, the log shows progress live.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def run(
    steps: Sequence[Step],
    out: Path,
    runner: Runner = shell_runner,
    results: Path = ROOT / "results",
) -> Manifest:
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    for step in steps:
        if step.name != "env" and not manifest.passed("env"):
            manifest.steps.append(
                StepRecord(step.name, step.command, None, 0.0, None, "env did not pass: no GPU")
            )
            manifest.write(out)
            continue
        failed = [need for need in step.needs if not manifest.passed(need)]
        if failed:
            reason = f"{', '.join(failed)} did not pass"
            manifest.steps.append(StepRecord(step.name, step.command, None, 0.0, None, reason))
            manifest.write(out)
            continue
        print(f"\n=== {step.name}: {step.command}", flush=True)
        log = out / f"{step.name}.log"
        start = time.monotonic()
        code = runner(step.command, log)
        manifest.steps.append(
            StepRecord(step.name, step.command, code, time.monotonic() - start, log.name)
        )
        if results.is_dir():
            shutil.copytree(results, out / "results", dirs_exist_ok=True)
        manifest.write(out)
        print(f"=== {step.name}: exit {code}", flush=True)
    return manifest


def collect(session: Path, results: Path = ROOT / "results") -> list[Path]:
    """Copy a session's result files into results/; refuse a session whose GPU tests failed.

    Each GPU directory also gets the session's environment report, as
    env-report-<session start>.txt, next to its result files.
    """
    manifest = Manifest.read(session)
    for name in ("env", "test-gpu"):
        if not manifest.passed(name):
            raise RuntimeError(f"{session}: step {name!r} did not pass; no results are copied")
    copied: list[Path] = []
    source = session / "results"
    files = sorted(source.rglob("*.json")) if source.is_dir() else []
    for path in files:
        target = results / path.relative_to(source)
        if target.exists():
            continue  # already collected, possibly from this session; never overwrite
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    stamp = manifest.started_utc.replace("-", "").replace(":", "")
    for gpu in sorted({p.relative_to(results).parts[0] for p in copied}):
        report = results / gpu / f"env-report-{stamp}.txt"
        if not report.exists():
            shutil.copy2(session / "env.log", report)
    return copied


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the session's steps on this GPU machine")
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--steps", help=f"comma-separated, from: {', '.join(STEP_NAMES)}")
    p_collect = sub.add_parser("collect", help="copy a passed session's results into results/")
    p_collect.add_argument("session", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        names = args.steps.split(",") if args.steps else None
        manifest = run(select(names), args.out)
        failed = [s.name for s in manifest.steps if s.returncode not in (0, None)]
        skipped = [s.name for s in manifest.steps if s.skipped]
        print(f"\nsession done; failed: {failed or 'none'}; skipped: {skipped or 'none'}")
        return 0 if manifest.passed("test-gpu") else 1
    copied = collect(args.session)
    for path in copied:
        print(f"copied {path}")
    print(f"{len(copied)} result files copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
