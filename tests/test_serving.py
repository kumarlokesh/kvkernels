"""The serving driver's commands, parsing and result files, without vLLM or a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bench import serving
from bench.schema import EnvInfo

ENV = EnvInfo("NVIDIA L4", "l4", "580", "12.9", "2.14.0", "3.8.0", None, None, "0" * 64)


def test_workloads_are_the_planned_four() -> None:
    by_name = {w.name: w for w in serving.WORKLOADS}
    assert by_name["shared_system_prompt"] == serving.Workload(
        "shared_system_prompt", 1, 2048, 128, 256
    )
    assert by_name["few_shot_4"].prefixes == 4 and by_name["few_shot_4"].prefix_len == 1024
    assert by_name["long_document"].prefix_len == 8192
    assert by_name["control"].prefixes == 0


def test_bench_command_fixes_lengths_and_greedy_output() -> None:
    w = serving.WORKLOADS[0]
    cmd = serving.bench_command(
        "m", w, concurrency=32, num_prompts=256, result_dir=Path("/tmp/r"), result_file="x.json"
    )
    joined = " ".join(cmd)
    assert "--dataset-name prefix_repetition" in joined
    assert "--prefix-repetition-prefix-len 2048" in joined
    assert "--max-concurrency 32" in joined and "--num-prompts 256" in joined
    assert "--ignore-eos" in cmd and "--temperature 0" in joined
    control = " ".join(
        serving.bench_command(
            "m", serving.WORKLOADS[3], concurrency=16, num_prompts=8, result_dir=Path("/r"),
            result_file="y.json",
        )
    )  # fmt: skip
    assert "--dataset-name random" in control and "--random-input-len 128" in control


def test_serve_command_selects_the_backend_and_caps_the_context() -> None:
    cmd = serving.serve_command("m", "CUSTOM")
    assert Path(cmd[0]).name == "vllm"
    assert cmd[1:3] == ["serve", "m"]
    assert cmd[cmd.index("--attention-backend") + 1] == "CUSTOM"
    assert cmd[cmd.index("--max-model-len") + 1] == str(serving.MAX_MODEL_LEN)


def test_a_record_comes_from_vllm_bench_output() -> None:
    data = {
        "completed": 128,
        "failed": 0,
        "duration": 41.5,
        "output_throughput": 1520.25,
        "median_tpot_ms": 30.1,
        "p99_tpot_ms": 44.0,
        "median_ttft_ms": 210.0,
        "generated_texts": ["a"] * 128,
    }
    r = serving.record_from_bench(
        data, backend="CUSTOM", workload="w", concurrency=16, num_prompts=128
    )
    assert r.output_throughput == 1520.25 and r.median_tpot_ms == 30.1 and r.completed == 128


def test_agreement_counts_equal_texts_against_the_reference_backend() -> None:
    texts = {
        ("FLASH_ATTN", "w", 16): ["a", "b", "c", "d"],
        ("TRITON_ATTN", "w", 16): ["a", "b", "x", "d"],
        ("CUSTOM", "w", 16): ["a", "y", "z", "d"],
        ("CUSTOM", "w", 32): ["a"],  # no reference run at 32: left out
    }
    got = {(a.backend, a.concurrency): (a.equal, a.total) for a in serving.agreement(texts)}
    assert got == {("TRITON_ATTN", 16): (3, 4), ("CUSTOM", 16): (2, 4)}


def test_gsm8k_accuracy_comes_from_lm_eval_results() -> None:
    data = {
        "results": {
            "gsm8k": {"exact_match,strict-match": 0.78, "exact_match_stderr,strict-match": 0.011}
        }
    }
    a = serving.accuracy_from_lm_eval(data, "CUSTOM")
    assert (a.accuracy, a.stderr, a.task) == (0.78, 0.011, "gsm8k_5shot_strict")


def test_result_file_round_trips(tmp_path: Path) -> None:
    record = serving.ServingRecord("CUSTOM", "w", 16, 128, 128, 0, 40.0, 1500.0, 30.0, 45.0, 200.0)
    result = serving.ServingResult(
        schema_version=serving.SERVING_SCHEMA_VERSION,
        suite="serving",
        git_sha="abc",
        created_utc="2026-11-13T22:15:00Z",
        env=ENV,
        engine="vllm",
        engine_version=serving.ENGINE_VERSION,
        model="m",
        model_revision="r",
        max_model_len=serving.MAX_MODEL_LEN,
        records=[record],
        agreement=[serving.AgreementRecord("CUSTOM", "w", 16, "FLASH_ATTN", 3, 4)],
        accuracy=[serving.AccuracyRecord("CUSTOM", "gsm8k_5shot_strict", 0.78, 0.011)],
    )
    path = serving.write_result(result, tmp_path)
    assert path == tmp_path / "l4" / "serving" / "20261113T221500Z.json"
    assert serving.read_result(path) == result
    with pytest.raises(FileExistsError):
        serving.write_result(result, tmp_path)


def test_tools_come_from_the_venv_of_the_running_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "vllm").touch()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    assert serving.tool("vllm") == str(tmp_path / "bin" / "vllm")
    assert serving.tool("lm_eval") == "lm_eval"  # not installed there: found on PATH


def test_the_bench_saves_generated_texts_for_the_agreement_count(tmp_path: Path) -> None:
    w = serving.WORKLOADS[0]
    cmd = serving.bench_command(
        "m", w, concurrency=16, num_prompts=128, result_dir=tmp_path, result_file="r.json"
    )
    assert "--save-detailed" in cmd
