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
"""


def env(slug: str) -> dict[str, Any]:
    return {
        "gpu_name": slug.upper(),
        "gpu_slug": slug,
        "torch_version": "2.14.1",
        "triton_version": "3.8.0",
    }


def gather(layout: str, impl: str, gbps: float) -> dict[str, Any]:
    params = {**readme_results.KV_GATHER_ROW, "layout": layout}
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
    assert "No results yet." in text  # cascade


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


def test_no_attention_table_for_a_gpu_without_tensor_cores_in_triton(tmp_path: Path) -> None:
    write(tmp_path, "t4", "cascade", "20261101T000000Z", [cascade(1, "kvkernels", 0.5)])
    assert "No results yet." in render(tmp_path).split("results:cascade:begin")[1]


def test_missing_markers_are_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results:cascade"):
        readme_results.render(
            "<!-- results:kv_gather:begin -->\n<!-- results:kv_gather:end -->", tmp_path, tmp_path
        )
