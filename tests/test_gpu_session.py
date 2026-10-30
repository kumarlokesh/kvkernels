"""The GPU session runner and its Kaggle driver, without a GPU or the network."""

from __future__ import annotations

import ast
import base64
import io
import tarfile
from pathlib import Path

import pytest

from scripts import gpu_session, kaggle_session


def test_env_always_runs_first() -> None:
    names = [s.name for s in gpu_session.select(["cascade", "test-gpu"])]
    assert names == ["env", "test-gpu", "cascade"]
    assert [s.name for s in gpu_session.select(None)] == list(gpu_session.STEP_NAMES)
    with pytest.raises(ValueError, match="nonsense"):
        gpu_session.select(["nonsense"])


def fake_runner(exit_codes: dict[str, int]) -> gpu_session.Runner:
    def run(command: str, log: Path) -> int:
        step = next(s for s in gpu_session.STEPS if s.command == command)
        log.write_text(f"output of {step.name}\n")
        return exit_codes.get(step.name, 0)

    return run


def test_result_steps_are_skipped_when_the_gpu_tests_fail(tmp_path: Path) -> None:
    steps = gpu_session.select(["test-gpu", "kv_gather", "old-torch"])
    manifest = gpu_session.run(steps, tmp_path, fake_runner({"test-gpu": 1}))
    by_name = {s.name: s for s in manifest.steps}
    assert by_name["test-gpu"].returncode == 1
    assert by_name["kv_gather"].returncode is None and by_name["kv_gather"].skipped
    assert by_name["old-torch"].returncode == 0  # writes no results, so it still runs
    saved = gpu_session.Manifest.read(tmp_path)
    assert [s.name for s in saved.steps] == ["env", "test-gpu", "kv_gather", "old-torch"]


def write_session(path: Path, test_gpu_code: int) -> None:
    manifest = gpu_session.Manifest(
        "2026-11-01T10:00:00Z",
        [
            gpu_session.StepRecord("env", "env", 0, 1.0, "env.log"),
            gpu_session.StepRecord("test-gpu", "tests", test_gpu_code, 1.0, "test-gpu.log"),
        ],
    )
    path.mkdir(parents=True)
    manifest.write(path)
    (path / "env.log").write_text("gpu_name Tesla T4\n")
    result = path / "results" / "t4" / "smoke" / "20261101T100500Z.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n")


def test_nothing_runs_without_a_gpu(tmp_path: Path) -> None:
    manifest = gpu_session.run(gpu_session.select(None), tmp_path, fake_runner({"env": 1}))
    assert [s.name for s in manifest.steps if s.returncode is not None] == ["env"]
    assert all(s.skipped for s in manifest.steps[1:])


def test_both_test_steps_require_a_gpu() -> None:
    commands = {s.name: s.command for s in gpu_session.STEPS}
    assert "KVKERNELS_REQUIRE_GPU=1" in commands["test-gpu"]
    assert "KVKERNELS_REQUIRE_GPU=1" in commands["old-torch"]


def test_collect_copies_results_and_the_env_report(tmp_path: Path) -> None:
    session, results = tmp_path / "session", tmp_path / "results"
    write_session(session, test_gpu_code=0)
    copied = gpu_session.collect(session, results)
    assert copied == [results / "t4" / "smoke" / "20261101T100500Z.json"]
    assert (results / "t4" / "env-report-20261101T100000Z.txt").read_text().startswith("gpu_name")
    assert gpu_session.collect(session, results) == []  # never overwrites


def test_collect_refuses_a_session_whose_gpu_tests_failed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    write_session(session, test_gpu_code=1)
    with pytest.raises(RuntimeError, match="test-gpu"):
        gpu_session.collect(session, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_bundle_holds_the_code_and_no_documents() -> None:
    with tarfile.open(fileobj=io.BytesIO(kaggle_session.bundle_bytes())) as tar:
        names = tar.getnames()
    assert "kvkernels/pyproject.toml" in names and "kvkernels/uv.lock" in names
    assert "kvkernels/src/kvkernels/ops/cascade.py" in names
    assert "kvkernels/scripts/gpu_session.py" in names
    for banned in ("rfcs/", "deep-dives/", "blog/", "notes/", "AGENTS.md", "roadmap", "results/"):
        assert not any(banned in n for n in names), banned
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_kernel_is_private_with_gpu_and_internet() -> None:
    meta = kaggle_session.kernel_metadata("someone", "NvidiaL4")
    assert meta["id"] == "someone/kvkernels-gpu-session"
    assert (meta["is_private"], meta["enable_gpu"], meta["enable_internet"]) == (
        "true",
        "true",
        "true",
    )
    assert meta["machine_shape"] == "NvidiaL4"
    assert meta["dataset_sources"] == []  # the source travels inside the script


def test_run_script_is_valid_python_and_carries_the_bundle() -> None:
    bundle = kaggle_session.bundle_bytes()
    script = kaggle_session.run_script("env,test-gpu", bundle)
    tree = ast.parse(script)
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert values["STEPS"] == "env,test-gpu"
    assert base64.b64decode(values["BUNDLE"]) == bundle
    assert "STEPS = ''" in kaggle_session.run_script(None, b"")
