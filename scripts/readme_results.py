"""Write the README's results tables from the committed result files.

    python scripts/readme_results.py           rewrite README.md
    python scripts/readme_results.py --check   exit 1 if README.md differs from what
                                               the result files give

Each suite's tables sit between two markers in README.md:

    <!-- results:<suite>:begin -->
    one table per GPU, each from the newest file in results/<gpu>/<suite>/,
    with a line naming the file, GPU, torch, Triton and commit
    <!-- results:<suite>:end -->

Each table comes from one result file, so numbers from different runs never share
a table. Text outside the markers is left alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

Result = dict[str, Any]
Measurement = dict[str, Any]

# On compute capability 7.5, Triton compiles the attention kernel without tensor
# cores, so the README shows no attention results for these GPUs.
NO_ATTENTION_RESULTS = {"t4"}
MISSING = "-"  # a cell with no measurement

KV_GATHER_LAYOUTS = {
    "vllm_LBNHC": "vLLM default",
    "vllm_BLNHC": "vLLM, layers per block",
    "sglang_nhd": "SGLang default",
}
KV_GATHER_IMPLS = {
    "kvkernels": "kvkernels",
    "torch_index_select": "index_select",
    "torch_advanced_indexing": "advanced indexing",
    "vllm_swap_blocks": "vLLM copy kernel",
    "cuda_batched_dma": "cuMemcpyBatchAsync",
    "copy_ceiling": "plain copy (no indexing)",
}
KV_GATHER_ROW = {
    "op": "gather",
    "block": 16,
    "kv_heads": 8,
    "head_dim": 128,
    "num_ids": 512,
    "order": "random",
    "dtype": "float16",
}
CASCADE_ROW = {
    "model": "llama-3.1-8b",
    "groups": 1,
    "prefix_len": 8192,
    "suffix_len": 512,
    "kv_heads": 8,
    "layout": "vllm_LBNHC",
    "dtype": "float16",
}
CASCADE_IMPLS = {
    "kvkernels": "kvkernels",
    "flashinfer_decode": "FlashInfer decode",
    "flashinfer_cascade": "FlashInfer cascade",
}


def newest_results(results: Path, suite: str) -> list[Path]:
    """The newest result file of the suite for each GPU, in GPU order."""
    newest = []
    for gpu in sorted(p for p in results.iterdir() if p.is_dir()) if results.is_dir() else []:
        files = sorted((gpu / suite).glob("*.json"))
        if files:
            newest.append(files[-1])
    return newest


def matches(m: Measurement, fixed: dict[str, Any]) -> bool:
    return all(m["params"].get(k) == v for k, v in fixed.items())


def find(result: Result, impl: str, fixed: dict[str, Any]) -> Measurement | None:
    found = [m for m in result["measurements"] if m["impl"] == impl and matches(m, fixed)]
    return found[0] if found else None


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def source_line(path: Path, result: Result, root: Path) -> str:
    env = result["env"]
    return (
        f"{env['gpu_name']}, torch {env['torch_version']}, Triton {env['triton_version']},"
        f" commit `{result['git_sha'][:14]}`; from `{path.relative_to(root).as_posix()}`"
    )


def kv_gather_table(result: Result) -> list[str] | None:
    impls = [i for i in KV_GATHER_IMPLS if any(m["impl"] == i for m in result["measurements"])]
    rows = []
    for layout, name in KV_GATHER_LAYOUTS.items():
        cells = [name]
        for impl in impls:
            m = find(result, impl, {**KV_GATHER_ROW, "layout": layout})
            if m is None or m["gbps"] is None:
                cells.append(MISSING)
            elif m["pct_measured_bw"] is None:
                cells.append(f"{m['gbps']:.0f}")
            else:
                cells.append(f"{m['gbps']:.0f} ({m['pct_measured_bw']:.0%})")
        if any(c != MISSING for c in cells[1:]):
            rows.append(cells)
    if not rows:
        return None
    return [
        "Gather of 512 KV blocks in random order, 16-token KV blocks, 8 KV heads,"
        " head_dim 128, fp16. GB/s, and in brackets the share of the measured copy"
        f" bandwidth ({result['measured_peak_gbps']:.0f} GB/s).",
        "",
        *table(["Layout", *(KV_GATHER_IMPLS[i] for i in impls)], rows),
    ]


def cascade_table(result: Result) -> list[str] | None:
    if result["env"]["gpu_slug"] in NO_ATTENTION_RESULTS:
        return None
    batches = sorted(
        {m["params"]["batch"] for m in result["measurements"] if matches(m, CASCADE_ROW)}
    )
    rows = []
    for batch in batches:
        fixed = {**CASCADE_ROW, "batch": batch}
        found = {impl: find(result, impl, fixed) for impl in CASCADE_IMPLS}
        cells = [str(batch)]
        for m in found.values():
            cells.append(MISSING if m is None else f"{m['median_ms'] * 1e3:.1f}")
        ours, decode = found["kvkernels"], found["flashinfer_decode"]
        speedup = decode["median_ms"] / ours["median_ms"] if ours and decode else None
        cells.append(MISSING if speedup is None else f"{speedup:.2f}x")
        pct = ours["pct_datasheet_bw"] if ours else None
        cells.append(MISSING if pct is None else f"{pct:.0%}")
        rows.append(cells)
    if not rows:
        return None
    header = [
        "Batch",
        *(f"{name} (µs)" for name in CASCADE_IMPLS.values()),
        "speedup over FlashInfer decode",
        "kvkernels, share of datasheet bandwidth",
    ]
    return [
        "One decode step of one layer: Llama 3.1 8B attention (32 query heads,"
        " 8 KV heads, head_dim 128), one shared prefix of 8,192 tokens, a 512-token"
        " suffix per request, vLLM's layout with 16-token KV blocks, fp16. Median"
        " time; bandwidth counts the bytes a cascade needs (the prefix read once).",
        "",
        *table(header, rows),
    ]


TABLES: dict[str, Callable[[Result], list[str] | None]] = {
    "kv_gather": kv_gather_table,
    "cascade": cascade_table,
}


def section(suite: str, results: Path, root: Path) -> str:
    parts = []
    for path in newest_results(results, suite):
        result = json.loads(path.read_text())
        lines = TABLES[suite](result)
        if lines:
            parts.append("\n".join([source_line(path, result, root), "", *lines]))
    return "\n\n".join(parts) if parts else "No results yet."


def render(readme: str, results: Path, root: Path) -> str:
    for suite in TABLES:
        begin, end = f"<!-- results:{suite}:begin -->", f"<!-- results:{suite}:end -->"
        found = re.search(re.escape(begin) + r".*?" + re.escape(end), readme, re.DOTALL)
        if found is None:
            raise ValueError(f"README has no {begin} … {end} markers")
        body = f"{begin}\n{section(suite, results, root)}\n{end}"
        readme = readme[: found.start()] + body + readme[found.end() :]
    return readme


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    args = parser.parse_args(argv)
    path = ROOT / "README.md"
    current = path.read_text()
    wanted = render(current, ROOT / "results", ROOT)
    if args.check:
        if wanted != current:
            print("README.md results tables are out of date; run `make readme`", file=sys.stderr)
            return 1
        return 0
    path.write_text(wanted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
