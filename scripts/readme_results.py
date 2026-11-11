"""Write the README's results, their charts, and the full tables, from the result files.

    python scripts/readme_results.py           rewrite the files below
    python scripts/readme_results.py --check   exit 1 if any differs from what the
                                               result files give

For each suite and GPU, from the newest file in results/<gpu>/<suite>/:

    README.md, between <!-- results:<suite>:begin/end -->   a one-line summary and a chart
    figures/<gpu>/<suite>-{light,dark}.svg                  the chart, for each GitHub theme
    results/README.md                                       the full table, and the file,
                                                            GPU, torch, Triton and commit

Each chart and table comes from one result file, so numbers from different runs
never share one. Text outside the markers is left alone.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
        f"<sub>{env['gpu_name']}, torch {env['torch_version']}, Triton"
        f" {env['triton_version']}, commit <code>{result['git_sha'][:14]}</code>; from"
        f" <code>{path.relative_to(root).as_posix()}</code></sub>"
    )


def picture(light: str, dark: str, alt: str) -> str:
    """A chart that follows GitHub's light or dark theme."""
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="{dark}">'
        f'<img alt="{alt}" src="{light}" width="720"></picture>'
    )


# Chart colors: one accent for kvkernels, gray for the baselines, on GitHub's page
# colors. The accent passes the contrast check against both surfaces.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "accent": "#2a78d6",
        "other": "#d8d6ce",
        "on_accent": "#ffffff",
        "on_other": "#0b0b0b",
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "accent": "#3987e5",
        "other": "#3d3d3a",
        "on_accent": "#ffffff",
        "on_other": "#f0efec",
    },
}


def bar_chart(
    bars: Sequence[tuple[str, float]], ceiling: tuple[str, float] | None, title: str, unit: str
) -> dict[str, bytes]:
    """Horizontal bars, kvkernels first in the accent color; one SVG per theme.

    The SVG is the same bytes for the same input: fixed ids, no date, text drawn as
    paths, so the check mode can compare files.
    """
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    out = {}
    for theme, c in THEMES.items():
        with plt.rc_context({"svg.hashsalt": "kvkernels", "svg.fonttype": "path", "font.size": 10}):
            fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(bars) + 1.1))
            fig.patch.set_facecolor(c["surface"])
            ax.set_facecolor(c["surface"])
            names = [name for name, _ in bars][::-1]
            values = [value for _, value in bars][::-1]
            colors = [c["accent"] if n == "kvkernels" else c["other"] for n in names]
            ax.barh(names, values, height=0.6, color=colors, zorder=2)
            top = max([*values, ceiling[1] if ceiling else 0.0]) * 1.04
            # Values sit inside the bar ends, clear of the ceiling line.
            for y, (name, value) in enumerate(zip(names, values, strict=True)):
                ours = name == "kvkernels"
                ax.text(
                    value - top * 0.012,
                    y,
                    f"{value:.0f} {unit}",
                    ha="right",
                    va="center",
                    color=c["on_accent"] if ours else c["on_other"],
                    fontweight="bold" if ours else "normal",
                    zorder=4,
                )
            if ceiling:
                label, value = ceiling
                ax.axvline(value, color=c["muted"], linestyle=(0, (4, 3)), linewidth=1, zorder=3)
                ax.text(
                    value,
                    len(bars) - 0.4,
                    f"{label}: {value:.0f} {unit}",
                    ha="right",
                    va="bottom",
                    color=c["secondary"],
                )
            ax.set_xlim(0, top)
            ax.set_ylim(-0.6, len(bars) + 0.2)
            ax.set_xlabel(unit, color=c["muted"], loc="right")
            ax.set_title(title, loc="left", color=c["ink"], fontsize=11, pad=14)
            ax.tick_params(colors=c["muted"], length=0)
            for tick in ax.get_yticklabels():
                tick.set_color(c["ink"])
            ax.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": None})
            plt.close(fig)
        out[theme] = buffer.getvalue()
    return out


Files = dict[str, bytes]  # path relative to the repository root -> file contents


@dataclass(frozen=True)
class Section:
    readme: list[str]  # for README.md; empty when the suite has nothing to show there
    table: list[str]  # for results/README.md
    figures: Files


def kv_gather_section(result: Result) -> Section | None:
    table_lines = kv_gather_table(result)
    if table_lines is None:
        return None
    fixed = {**KV_GATHER_ROW, "layout": "vllm_LBNHC"}
    found = {impl: find(result, impl, fixed) for impl in KV_GATHER_IMPLS}
    ours = found["kvkernels"]
    if ours is None:
        return Section([], table_lines, {})
    gpu, gpu_name = result["env"]["gpu_slug"], result["env"]["gpu_name"]
    bars = sorted(
        (
            (KV_GATHER_IMPLS[impl], m["gbps"])
            for impl, m in found.items()
            if m is not None and impl != "copy_ceiling"
        ),
        key=lambda bar: bar[0] != "kvkernels",
    )
    bars = [bars[0], *sorted(bars[1:], key=lambda bar: -bar[1])]
    copy = found["copy_ceiling"]
    ceiling = ("plain copy, no indexing", copy["gbps"]) if copy else None
    title = f"Gather of 512 KV blocks (64 KiB each, random order) on a {gpu_name}"
    charts = bar_chart(bars, ceiling, title, "GB/s")
    paths = {theme: f"figures/{gpu}/kv_gather-{theme}.svg" for theme in charts}

    summary = (
        f"On a {gpu_name}, kvkernels gathers 512 KV blocks from vLLM's default layout at"
        f" {ours['gbps']:.0f} GB/s"
    )
    if ours["pct_measured_bw"] is not None:
        summary += f", {ours['pct_measured_bw']:.0%} of the GPU's measured copy bandwidth"
    baselines = [f"{name} {value:.0f} GB/s" for name, value in bars[1:] if name in BASELINE_SUMMARY]
    if baselines:
        summary += " (" + ", ".join(baselines) + ")"
    lines = [summary + ".", "", picture(paths["light"], paths["dark"], title)]
    return Section(lines, table_lines, {paths[theme]: svg for theme, svg in charts.items()})


# Baselines named in the one-line summary; the chart shows them all.
BASELINE_SUMMARY = {"index_select", "vLLM copy kernel", "cuMemcpyBatchAsync"}


def cascade_section(result: Result) -> Section | None:
    table_lines = cascade_table(result)
    return None if table_lines is None else Section(table_lines, table_lines, {})


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


SECTIONS: dict[str, tuple[str, Callable[[Result], Section | None]]] = {
    "kv_gather": ("KV block gather", kv_gather_section),
    "cascade": ("Cascade decode attention", cascade_section),
}
TABLES_PATH = "results/README.md"
TABLES_HEADER = """# Results

Generated by `make readme` from the newest result file of each suite and GPU in
this directory. Each table comes from one file, named under it.
"""


def render(readme: str, results: Path, root: Path) -> tuple[str, Files]:
    """The README with every results section rewritten, and the other files it needs."""
    files: Files = {}
    tables = [TABLES_HEADER]
    for suite, (heading, make) in SECTIONS.items():
        begin, end = f"<!-- results:{suite}:begin -->", f"<!-- results:{suite}:end -->"
        found = re.search(re.escape(begin) + r".*?" + re.escape(end), readme, re.DOTALL)
        if found is None:
            raise ValueError(f"README has no {begin} … {end} markers")
        parts = []
        tables.append(f"## {heading}\n")
        for path in newest_results(results, suite):
            result = json.loads(path.read_text())
            made = make(result)
            if made is None:
                continue
            source = source_line(path, result, root)
            if made.readme:
                parts.append("\n".join([*made.readme, "", source]))
            tables.append("\n".join([*made.table, "", source, ""]))
            files.update(made.figures)
        if len(tables[-1]) and tables[-1].startswith("## "):
            tables.append("No results yet.\n")
        text = "\n\n".join(parts) if parts else "No results yet."
        readme = readme[: found.start()] + f"{begin}\n{text}\n{end}" + readme[found.end() :]
    files[TABLES_PATH] = "\n".join(tables).encode()
    return readme, files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    args = parser.parse_args(argv)
    path = ROOT / "README.md"
    current = path.read_text()
    wanted, files = render(current, ROOT / "results", ROOT)
    if args.check:
        stale = [
            name
            for name, data in files.items()
            if not (ROOT / name).exists() or (ROOT / name).read_bytes() != data
        ]
        if wanted != current or stale:
            print("README.md results are out of date; run `make readme`", file=sys.stderr)
            return 1
        return 0
    path.write_text(wanted)
    for name, data in files.items():
        (ROOT / name).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / name).write_bytes(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
