"""Charts from one result file.

Usage: ``python -m bench.plot results/<gpu>/<suite>/<stamp>.json --x <param> [--filter k=v]``

One result file per invocation: numbers from different suite runs, or from
different torch or Triton versions, are never compared in one chart.

Writes ``figures/<gpu>/<suite>/<stamp>-latency.{png,svg}`` and, when the suite
reports bandwidth, ``<stamp>-bandwidth.{png,svg}``. Noisy measurements (p90 /
median above the harness threshold) are drawn as hollow markers.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.harness import matches, parse_filters
from bench.schema import BenchResult, Measurement, read_result


@dataclass(frozen=True)
class Point:
    x: float
    latency_us: float
    pct_bw: float | None
    noisy: bool


def series(
    result: BenchResult, x: str, filters: Mapping[str, str] | None = None
) -> dict[str, list[Point]]:
    """Group measurements into one line per implementation and fixed parameters.

    The line label is the implementation name followed by every parameter except
    x, so two points share a line only when they differ in x alone.
    """
    lines: dict[str, list[Point]] = {}
    for m in result.measurements:
        if filters and not matches(m.params, filters):
            continue
        if x not in m.params:
            raise KeyError(f"parameter {x!r} not in measurement params {sorted(m.params)}")
        raw = m.params[x]
        if not isinstance(raw, int):
            raise TypeError(f"x parameter {x!r} must be an integer, got {raw!r}")
        lines.setdefault(_label(m, x), []).append(
            Point(float(raw), m.median_ms * 1e3, _pct_bw(m), m.noisy)
        )
    for points in lines.values():
        points.sort(key=lambda p: p.x)
    return lines


def _label(m: Measurement, x: str) -> str:
    rest = ", ".join(f"{k}={v}" for k, v in sorted(m.params.items()) if k != x)
    return f"{m.impl} ({rest})" if rest else m.impl


def _pct_bw(m: Measurement) -> float | None:
    pct = m.pct_measured_bw if m.pct_measured_bw is not None else m.pct_datasheet_bw
    return None if pct is None else pct * 100


def _draw(ax: Any, lines: dict[str, list[Point]], value: str) -> None:
    for label, points in lines.items():
        xs = [p.x for p in points]
        ys = [getattr(p, value) for p in points]
        (line,) = ax.plot(xs, ys, label=label)
        for p, y in zip(points, ys, strict=True):
            ax.plot(
                p.x,
                y,
                marker="o",
                color=line.get_color(),
                markerfacecolor="none" if p.noisy else line.get_color(),
            )


def plot(result: BenchResult, x: str, out_root: Path, filters: Mapping[str, str]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("plotting needs the [bench] extra: uv sync --extra bench") from exc

    lines = series(result, x, filters)
    if not lines:
        raise SystemExit("no measurements match")
    stamp = result.created_utc.replace("-", "").replace(":", "")
    out_dir = out_root / result.env.gpu_slug / result.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    env = result.env
    title = (
        f"{result.suite} on {env.gpu_name} (torch {env.torch_version}, triton {env.triton_version})"
    )
    written: list[Path] = []

    charts = [("latency", "latency_us", "median latency (µs)")]
    if any(p.pct_bw is not None for pts in lines.values() for p in pts):
        uses_measured = any(m.pct_measured_bw is not None for m in result.measurements)
        peak = "measured copy peak" if uses_measured else "datasheet peak"
        charts.append(("bandwidth", "pct_bw", f"achieved bandwidth (% of {peak})"))

    for kind, value, ylabel in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        _draw(ax, lines, value)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        for ext in ("png", "svg"):
            path = out_dir / f"{stamp}-{kind}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot one benchmark result file.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--x", required=True, help="integer parameter for the x axis")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    for path in plot(read_result(args.result), args.x, args.out, parse_filters(args.filter)):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
