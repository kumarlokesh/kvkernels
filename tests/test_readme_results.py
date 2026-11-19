"""The README script, on result files written into a temporary directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import readme_results

README = """# x

<!-- results:kv_gather:begin -->
old text
<!-- results:kv_gather:end -->

kept

<!-- results:cascade:begin -->
<!-- results:cascade:end -->

<!-- results:serving:begin -->
<!-- results:serving:end -->
"""


def env(slug: str) -> dict[str, Any]:
    return {
        "gpu_name": slug.upper(),
        "gpu_slug": slug,
        "torch_version": "2.14.1",
        "triton_version": "3.8.0",
    }


def gather(layout: str, impl: str, gbps: float, num_ids: int = 512) -> dict[str, Any]:
    params = {**readme_results.KV_GATHER_ROW, "layout": layout, "num_ids": num_ids}
    return {"params": params, "impl": impl, "gbps": gbps, "pct_measured_bw": gbps / 200}


def cascade(batch: int, impl: str, ms: float) -> dict[str, Any]:
    params = {**readme_results.CASCADE_ROW, "batch": batch}
    return {"params": params, "impl": impl, "median_ms": ms, "pct_datasheet_bw": 0.5}


def write(root: Path, gpu: str, suite: str, stamp: str, measurements: list[Any]) -> None:
    path = root / "results" / gpu / suite / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "env": env(gpu),
        "git_sha": "a" * 40,
        "measured_peak_gbps": 200.0,
        "measurements": measurements,
    }
    path.write_text(json.dumps(result))


def serving_record(workload: str, concurrency: int, backend: str, tok_s: float) -> dict[str, Any]:
    return {
        "workload": workload,
        "concurrency": concurrency,
        "backend": backend,
        "output_throughput": tok_s,
    }


def write_serving(root: Path, stamp: str, dtype: str | None, records: list[Any]) -> None:
    path = root / "results" / "l4" / "serving" / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "env": env("l4"),
        "git_sha": "b" * 40,
        "engine_version": "0.33.0",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "records": records,
        "accuracy": [
            {"backend": "CUSTOM", "accuracy": 0.704, "stderr": 0.013},
            {"backend": "TRITON_ATTN", "accuracy": 0.701, "stderr": 0.013},
        ],
    }
    if dtype is not None:  # schema 1 files have no kv_cache_dtype
        result["kv_cache_dtype"] = dtype
    path.write_text(json.dumps(result))


def render(root: Path) -> str:
    return readme_results.render(README, root / "results", root)[0]


def test_each_gpu_gets_a_table_from_its_newest_file(tmp_path: Path) -> None:
    write(tmp_path, "l4", "kv_gather", "20261101T000000Z", [gather("vllm_LBNHC", "kvkernels", 100)])
    write(tmp_path, "l4", "kv_gather", "20261102T000000Z", [gather("vllm_LBNHC", "kvkernels", 180)])
    write(
        tmp_path, "h100", "kv_gather", "20261101T000000Z", [gather("sglang_nhd", "kvkernels", 50)]
    )
    text, files = readme_results.render(README, tmp_path / "results", tmp_path)
    assert "old text" not in text and "kept" in text
    tables = files["results/README.md"].decode()
    assert "| vLLM default | 180 (90%) |" in tables
    assert "| 100 " not in tables  # the older L4 file is not used
    assert tables.index("H100,") < tables.index("L4,")
    assert "results/l4/kv_gather/20261102T000000Z.json" in tables
    assert "No results yet." in text.split("results:cascade:begin")[1]


def test_rendering_twice_changes_nothing(tmp_path: Path) -> None:
    write(tmp_path, "l4", "kv_gather", "20261101T000000Z", [gather("vllm_LBNHC", "kvkernels", 100)])
    once, figures = readme_results.render(README, tmp_path / "results", tmp_path)
    again, figures_again = readme_results.render(once, tmp_path / "results", tmp_path)
    assert again == once
    assert figures_again == figures  # the same SVG bytes, so the check mode can compare


def test_the_gather_section_has_a_summary_a_chart_per_theme_and_the_numbers(
    tmp_path: Path,
) -> None:
    measurements = [
        gather("vllm_LBNHC", "kvkernels", 180),
        gather("vllm_LBNHC", "vllm_swap_blocks", 170),
        gather("vllm_LBNHC", "copy_ceiling", 190),
    ]
    write(tmp_path, "l4", "kv_gather", "20261101T000000Z", measurements)
    text, figures = readme_results.render(README, tmp_path / "results", tmp_path)
    assert "kvkernels gathers 512 KV blocks from vLLM's default layout at 180 GB/s" in text
    assert "vLLM copy kernel 170 GB/s" in text
    charts = ["figures/l4/kv_gather-dark.svg", "figures/l4/kv_gather-light.svg"]
    assert sorted(figures) == [*charts, "results/README.md"]
    assert all(figures[chart].startswith(b"<?xml") for chart in charts)
    assert 'srcset="figures/l4/kv_gather-dark.svg"' in text
    assert "| vLLM default |" not in text  # the full table lives in results/README.md
    tables = figures["results/README.md"].decode()
    assert "| vLLM default | 180 (90%) |" in tables
    assert "results/l4/kv_gather/20261101T000000Z.json" in tables


def test_the_gather_shown_is_the_smallest_whose_output_is_four_times_l2(tmp_path: Path) -> None:
    # L4: 48 MiB of L2. 512 blocks write 32 MiB, 2,048 write 128 MiB, 4,096 write 256 MiB.
    measurements = [
        gather("vllm_LBNHC", "kvkernels", gbps, num_ids=n)
        for n, gbps in ((512, 431), (2048, 299), (4096, 240), (32768, 234))
    ]
    write(tmp_path, "l4", "kv_gather", "20261101T000000Z", measurements)
    text, files = readme_results.render(README, tmp_path / "results", tmp_path)
    assert "kvkernels gathers 4,096 KV blocks from vLLM's default layout at 240 GB/s" in text
    tables = files["results/README.md"].decode()
    assert "Gather of 4,096 KV blocks" in tables and "| vLLM default | 240 (120%) |" in tables


def test_cascade_rows_and_speedup(tmp_path: Path) -> None:
    measurements = [
        cascade(1, "kvkernels", 0.5),
        cascade(1, "flashinfer_decode", 1.0),
        cascade(32, "kvkernels", 2.0),
    ]
    write(tmp_path, "l4", "cascade", "20261101T000000Z", measurements)
    text = render(tmp_path)
    assert "| 1 | 500.0 | 1000.0 | - | 2.00x | 50% |" in text
    assert "| 32 | 2000.0 | - | - | - | 50% |" in text


def test_the_cascade_section_has_a_summary_and_a_chart_at_batch_64(tmp_path: Path) -> None:
    measurements = [
        cascade(64, "kvkernels", 0.769),
        cascade(64, "flashinfer_decode", 2.373),
        cascade(64, "flashinfer_cascade", 0.716),
        cascade(128, "kvkernels", 1.5),
    ]
    write(tmp_path, "l4", "cascade", "20261101T000000Z", measurements)
    text, files = readme_results.render(README, tmp_path / "results", tmp_path)
    assert (
        "decodes 64 requests that share an 8,192-token prefix in 769 µs with kvkernels"
        " (FlashInfer's decode 2373 µs, FlashInfer's cascade 716 µs)." in text
    )
    assert "figures/l4/cascade-light.svg" in files and "figures/l4/cascade-dark.svg" in files
    assert "| 64 | 769.0 |" not in text  # the table lives in results/README.md
    assert "| 64 | 769.0 | 2373.0 | 716.0 | 3.09x | 50% |" in files["results/README.md"].decode()


def test_serving_takes_the_newest_file_of_each_kv_cache_dtype(tmp_path: Path) -> None:
    def records(ours: float, triton: float, flash: float | None) -> list[Any]:
        rows = []
        for workload, scale in (("long_document", 1.0), ("control", 0.0)):
            for c in (16, 64):
                rows.append(serving_record(workload, c, "CUSTOM", ours if scale else 100.0))
                rows.append(serving_record(workload, c, "TRITON_ATTN", triton if scale else 99.0))
                if flash is not None:
                    rows.append(
                        serving_record(workload, c, "FLASH_ATTN", flash if scale else 100.5)
                    )
        return rows

    write_serving(tmp_path, "20261101T000000Z", None, records(500, 400, 300))
    write_serving(tmp_path, "20261102T000000Z", "auto", records(666, 447, 358))
    write_serving(tmp_path, "20261103T000000Z", "fp8", records(680, 494, None))
    text, files = readme_results.render(README, tmp_path / "results", tmp_path)
    serving = text.split("results:serving:begin")[1]
    assert "produces 666 output tokens/s when 64 requests share" in serving
    assert "(+49% over vLLM Triton, +86% over FlashAttention)" in serving
    assert "With nothing shared it is within 1.0% of vLLM's backends" in serving
    assert "GSM8K kvkernels 70.4%, vLLM Triton 70.1%" in serving
    assert "with an fp8 KV cache" in serving and "(+38% over vLLM Triton)" in serving
    assert "produces 500" not in serving  # the older default-dtype file is not used
    assert serving.index("20261102T000000Z") < serving.index("20261103T000000Z")
    assert "figures/l4/serving-light.svg" in files  # one chart, for the default dtype
    tables = files["results/README.md"].decode()
    assert (
        "| long document: 8,192 shared tokens | 64 | 666 | 447 | 358 | +49.0% | +86.0% |" in tables
    )
    assert "GSM8K 5-shot, strict match: kvkernels 70.4% ± 1.3%, vLLM Triton 70.1% ± 1.3%." in tables


def test_no_attention_table_for_a_gpu_without_tensor_cores_in_triton(tmp_path: Path) -> None:
    write(tmp_path, "t4", "cascade", "20261101T000000Z", [cascade(1, "kvkernels", 0.5)])
    assert "No results yet." in render(tmp_path).split("results:cascade:begin")[1]


def test_missing_markers_are_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results:cascade"):
        readme_results.render(
            "<!-- results:kv_gather:begin -->\n<!-- results:kv_gather:end -->", tmp_path, tmp_path
        )
