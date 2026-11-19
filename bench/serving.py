"""Serving benchmark: kvkernels' vLLM backend against vLLM's own, on shared-prefix workloads.

Usage on a GPU machine, in an environment with vllm==0.33.0, kvkernels and
kvkernels-vllm installed (see integrations/vllm), after scripts/env_report.py:

    python -m bench.serving --model meta-llama/Llama-3.1-8B-Instruct

For each backend one vLLM server runs; for each workload the prefix cache is
filled with one request per prefix, then vLLM's serving benchmark runs at each
concurrency:

    for backend in FLASH_ATTN, TRITON_ATTN, CUSTOM (kvkernels):
        vllm serve <model> --attention-backend <backend>
        for workload in WORKLOADS:
            warm-up: one request per prefix
            for concurrency in CONCURRENCIES:
                vllm bench serve … ──▶ one ServingRecord

Outputs are greedy and every request generates exactly its output length, so
the same prompt gives comparable text across backends: agreement is the share
of requests whose text equals the first backend's. With --gsm8k, GSM8K 5-shot
accuracy is measured through each server with lm-eval.

The result goes to results/<gpu>/serving/<time>.json. Server logs and vLLM's
raw benchmark files stay in a directory printed at the start.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.schema import EnvInfo, SchemaError, _build, utc_now

SERVING_SCHEMA_VERSION = 2  # 2 adds kv_cache_dtype; a version 1 file reads as "auto"
ENGINE = "vllm"
ENGINE_VERSION = "0.33.0"
BACKENDS = ("FLASH_ATTN", "TRITON_ATTN", "CUSTOM")
CONCURRENCIES = (16, 32, 64)
PROMPTS_PER_CONCURRENCY = 8
# vLLM refuses to start unless one request of the model's full context fits in the
# KV cache; Llama 3.1's 131,072 tokens need about 16 GB, more than an L4 has left.
MAX_MODEL_LEN = 16384
PORT = 8123
SEED = 0


@dataclass(frozen=True)
class Workload:
    name: str
    prefixes: int  # 0: no shared prefix
    prefix_len: int
    own_len: int  # each request's own prompt tokens
    output_len: int

    def dataset_args(self) -> list[str]:
        if self.prefixes == 0:
            return [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(self.own_len),
                "--random-output-len",
                str(self.output_len),
                "--random-range-ratio",
                "0.0",
            ]
        return [
            "--dataset-name",
            "prefix_repetition",
            "--prefix-repetition-prefix-len",
            str(self.prefix_len),
            "--prefix-repetition-suffix-len",
            str(self.own_len),
            "--prefix-repetition-num-prefixes",
            str(self.prefixes),
            "--prefix-repetition-output-len",
            str(self.output_len),
        ]


WORKLOADS = (
    Workload("shared_system_prompt", 1, 2048, 128, 256),
    Workload("few_shot_4", 4, 1024, 128, 256),
    Workload("long_document", 1, 8192, 64, 128),
    Workload("control", 0, 0, 128, 256),
)


@dataclass(frozen=True)
class ServingRecord:
    backend: str
    workload: str
    concurrency: int
    num_prompts: int
    completed: int
    failed: int
    duration_s: float
    output_throughput: float  # output tokens per second
    median_tpot_ms: float
    p99_tpot_ms: float
    median_ttft_ms: float


@dataclass(frozen=True)
class AgreementRecord:
    backend: str
    workload: str
    concurrency: int
    reference: str  # the backend whose text counts as the reference
    equal: int
    total: int


@dataclass(frozen=True)
class AccuracyRecord:
    backend: str
    task: str
    accuracy: float
    stderr: float


@dataclass(frozen=True)
class ServingResult:
    schema_version: int
    suite: str
    git_sha: str
    created_utc: str
    env: EnvInfo
    engine: str
    engine_version: str
    model: str
    model_revision: str
    max_model_len: int
    records: list[ServingRecord]
    agreement: list[AgreementRecord]
    accuracy: list[AccuracyRecord]
    kv_cache_dtype: str  # vLLM's --kv-cache-dtype for every server of the run


# --- commands ---------------------------------------------------------------------


def tool(name: str) -> str:
    """A command installed next to this Python, so it runs without activating its venv."""
    path = Path(sys.executable).parent / name  # not resolved: the venv's bin, not the base's
    return str(path) if path.exists() else name


def serve_command(
    model: str, backend: str, port: int = PORT, kv_cache_dtype: str = "auto"
) -> list[str]:
    cache = [] if kv_cache_dtype == "auto" else ["--kv-cache-dtype", kv_cache_dtype]
    return [
        tool("vllm"),
        "serve",
        model,
        "--attention-backend",
        backend,
        "--port",
        str(port),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--seed",
        str(SEED),
        *cache,
    ]


def bench_command(
    model: str,
    workload: Workload,
    *,
    concurrency: int,
    num_prompts: int,
    result_dir: Path,
    result_file: str,
    port: int = PORT,
) -> list[str]:
    return [
        tool("vllm"),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        model,
        "--port",
        str(port),
        *workload.dataset_args(),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--temperature",
        "0",
        "--seed",
        str(SEED),
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "50,99",
        "--save-result",
        "--save-detailed",  # keeps generated_texts, which the agreement count compares
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_file,
    ]


def gsm8k_command(model: str, out_dir: Path, port: int = PORT) -> list[str]:
    return [
        tool("lm_eval"),
        "--model",
        "local-completions",
        "--model_args",
        f"model={model},base_url=http://localhost:{port}/v1/completions,"
        "num_concurrent=32,tokenized_requests=False",
        "--tasks",
        "gsm8k",
        "--num_fewshot",
        "5",
        "--output_path",
        str(out_dir),
    ]


# --- parsing ----------------------------------------------------------------------


def record_from_bench(
    data: dict[str, Any], *, backend: str, workload: str, concurrency: int, num_prompts: int
) -> ServingRecord:
    """One record from the JSON that `vllm bench serve --save-result` writes."""
    return ServingRecord(
        backend=backend,
        workload=workload,
        concurrency=concurrency,
        num_prompts=num_prompts,
        completed=int(data["completed"]),
        failed=int(data.get("failed", 0)),
        duration_s=float(data["duration"]),
        output_throughput=float(data["output_throughput"]),
        median_tpot_ms=float(data["median_tpot_ms"]),
        p99_tpot_ms=float(data["p99_tpot_ms"]),
        median_ttft_ms=float(data["median_ttft_ms"]),
    )


def agreement(
    texts: dict[tuple[str, str, int], list[str]], reference: str = BACKENDS[0]
) -> list[AgreementRecord]:
    """Per (backend, workload, concurrency): how many texts equal the reference backend's."""
    out = []
    for (backend, workload, concurrency), mine in sorted(texts.items()):
        theirs = texts.get((reference, workload, concurrency))
        if backend == reference or theirs is None:
            continue
        equal = sum(a == b for a, b in zip(mine, theirs, strict=False))
        out.append(
            AgreementRecord(
                backend, workload, concurrency, reference, equal, min(len(mine), len(theirs))
            )
        )
    return out


def accuracy_from_lm_eval(data: dict[str, Any], backend: str) -> AccuracyRecord:
    """GSM8K accuracy with strict answer matching, from lm-eval's results JSON."""
    task = data["results"]["gsm8k"]
    return AccuracyRecord(
        backend=backend,
        task="gsm8k_5shot_strict",
        accuracy=float(task["exact_match,strict-match"]),
        stderr=float(task["exact_match_stderr,strict-match"]),
    )


# --- result file ------------------------------------------------------------------


def result_path(result: ServingResult, root: Path = Path("results")) -> Path:
    stamp = result.created_utc.replace("-", "").replace(":", "")
    return root / result.env.gpu_slug / result.suite / f"{stamp}.json"


def write_result(result: ServingResult, root: Path = Path("results")) -> Path:
    path = result_path(result, root)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True) + "\n")
    return path


def read_result(path: Path) -> ServingResult:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("schema_version") == 1:
        data = {**data, "schema_version": SERVING_SCHEMA_VERSION, "kv_cache_dtype": "auto"}
    if not isinstance(data, dict) or data.get("schema_version") != SERVING_SCHEMA_VERSION:
        raise SchemaError(f"{path}: not a serving result of schema {SERVING_SCHEMA_VERSION}")
    result: ServingResult = _build(ServingResult, data, "result")
    return result


# --- running ----------------------------------------------------------------------


def wait_ready(port: int, server: subprocess.Popen[bytes], timeout_s: float = 1800.0) -> None:
    """Wait for the server's health endpoint; the first start downloads the model."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {server.returncode}")
        with (
            contextlib.suppress(OSError),
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r,
        ):
            if r.status == 200:
                return
        time.sleep(5)
    raise TimeoutError(f"vllm serve was not ready after {timeout_s:.0f} s")


@contextlib.contextmanager
def server(model: str, backend: str, log: Path, kv_cache_dtype: str = "auto") -> Iterator[None]:
    with log.open("wb") as out:
        command = serve_command(model, backend, kv_cache_dtype=kv_cache_dtype)
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        try:
            wait_ready(PORT, proc)
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def model_revision(model: str) -> str:
    from huggingface_hub import HfApi

    sha = HfApi().model_info(model).sha
    return sha or "unknown"


def run(
    model: str,
    *,
    gsm8k: bool,
    workdir: Path,
    backends: Sequence[str] = BACKENDS,
    kv_cache_dtype: str = "auto",
) -> ServingResult:
    from bench.env import collect
    from bench.harness import git_sha

    records: list[ServingRecord] = []
    texts: dict[tuple[str, str, int], list[str]] = {}
    accuracy: list[AccuracyRecord] = []
    for backend in backends:
        with server(model, backend, workdir / f"serve-{backend}.log", kv_cache_dtype):
            for workload in WORKLOADS:
                warm = bench_command(
                    model,
                    workload,
                    concurrency=1,
                    num_prompts=max(workload.prefixes, 1),
                    result_dir=workdir,
                    result_file=f"warm-{backend}-{workload.name}.json",
                )
                subprocess.run(warm, check=True)
                for c in CONCURRENCIES:
                    n = PROMPTS_PER_CONCURRENCY * c
                    name = f"{backend}-{workload.name}-{c}.json"
                    cmd = bench_command(
                        model,
                        workload,
                        concurrency=c,
                        num_prompts=n,
                        result_dir=workdir,
                        result_file=name,
                    )
                    subprocess.run(cmd, check=True)
                    data = json.loads((workdir / name).read_text())
                    records.append(
                        record_from_bench(
                            data,
                            backend=backend,
                            workload=workload.name,
                            concurrency=c,
                            num_prompts=n,
                        )
                    )
                    texts[(backend, workload.name, c)] = list(data.get("generated_texts", []))
            if gsm8k:
                out_dir = workdir / f"gsm8k-{backend}"
                subprocess.run(gsm8k_command(model, out_dir), check=True)
                found = sorted(out_dir.rglob("results_*.json"))
                if not found:
                    raise FileNotFoundError(f"lm-eval wrote no results under {out_dir}")
                accuracy.append(accuracy_from_lm_eval(json.loads(found[-1].read_text()), backend))
    return ServingResult(
        schema_version=SERVING_SCHEMA_VERSION,
        suite="serving",
        git_sha=git_sha(),
        created_utc=utc_now(),
        env=collect(),
        engine=ENGINE,
        engine_version=ENGINE_VERSION,
        model=model,
        model_revision=model_revision(model),
        max_model_len=MAX_MODEL_LEN,
        records=records,
        agreement=agreement(texts, reference=backends[0]),
        accuracy=accuracy,
        kv_cache_dtype=kv_cache_dtype,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--gsm8k", action="store_true", help="also measure GSM8K accuracy")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--kv-cache-dtype", default="auto", help="vLLM's --kv-cache-dtype, for every server"
    )
    parser.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help="comma-separated; the first is the reference for the agreement count",
    )
    args = parser.parse_args(argv)
    backends = args.backends.split(",")
    if unknown := set(backends) - set(BACKENDS):
        parser.error(f"unknown backends {sorted(unknown)}; choose from {BACKENDS}")
    import vllm

    if vllm.__version__ != ENGINE_VERSION:
        raise SystemExit(f"needs vllm {ENGINE_VERSION}, found {vllm.__version__}")
    workdir = Path(tempfile.mkdtemp(prefix="kvkernels-serving-"))
    print(f"server logs and raw benchmark files: {workdir}")
    result = run(
        args.model,
        gsm8k=args.gsm8k,
        workdir=workdir,
        backends=backends,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    path = write_result(result, args.results)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
