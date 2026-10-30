"""Device-memory traffic of shared-prefix decode attention, measured with Nsight Compute.

Usage: ``python -m bench.dram_bytes [--out results]`` on a GPU whose performance
counters are open (see bench/env.py).

The cascade suite's cost model assumes per-request decode reads the shared
prefix once per request. When the prefix fits in L2, later requests can find it
there, so the bytes read from device memory are fewer. This run measures them.
For each shape and implementation it runs one call inside an NVTX range under
ncu and adds up, over every kernel the call launches:

    dram__bytes_read.sum     bytes read from device memory
    dram__bytes_write.sum    bytes written to device memory

ncu flushes the caches before each kernel (--cache-control all), so a launch
never finds data a previous launch left in L2; reuse within one launch, the
case in question, is measured. Clocks are left alone (--clock-control none).

Shapes: Llama 3.1 8B attention, batch 32 and 128, prefix 4,096 and 32,768,
suffix 512. Implementations: flashinfer_decode (when FlashInfer runs),
kvkernels_per_request and kvkernels, as in the cascade suite. The result goes to
results/<gpu>/cascade-dram/<timestamp>.json with the suite's model bytes next to
each measurement.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bench.schema import EnvInfo, Params, utc_now

NVTX_RANGE = "dram"
SHAPES = [(batch, prefix) for batch in (32, 128) for prefix in (4096, 32768)]
SUFFIX = 512
IMPLS = ("flashinfer_decode", "kvkernels_per_request", "kvkernels")
METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")
_UNIT_TO_BYTES = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}


@dataclass(frozen=True)
class Traffic:
    params: Params
    impl: str
    kernels: int
    read_bytes: int
    write_bytes: int
    model_bytes: int  # the cascade suite's bytes_moved for the same params


@dataclass(frozen=True)
class DramResult:
    suite: str
    git_sha: str
    created_utc: str
    env: EnvInfo
    measurements: list[Traffic]


def parse_ncu_bytes(csv_text: str) -> dict[str, list[float]]:
    """Per metric in METRICS, its value in bytes for each kernel, from ``ncu --csv`` output.

    ncu mixes "==PROF==" log lines into its output; only quoted CSV lines are read.
    """
    rows = [line for line in csv_text.splitlines() if line.startswith('"')]
    values: dict[str, list[float]] = {m: [] for m in METRICS}
    for row in csv.DictReader(io.StringIO("\n".join(rows))):
        name = row.get("Metric Name", "")
        if name not in values:
            continue
        unit = row.get("Metric Unit", "").strip()
        if unit not in _UNIT_TO_BYTES:
            raise ValueError(f"unknown ncu unit {unit!r}")
        values[name].append(float(row["Metric Value"].replace(",", "")) * _UNIT_TO_BYTES[unit])
    return values


def shape_params(batch: int, prefix: int) -> Params:
    from bench.suites.cascade import _params

    return _params(batch=batch, prefix=prefix, suffix=SUFFIX)


def ncu_target(batch: int, prefix: int, impl_name: str) -> None:
    """Run inside ncu: plan, one call outside the range to compile, one call inside it."""
    import torch

    from bench.suites import cascade

    p = shape_params(batch, prefix)
    with_flashinfer = impl_name.startswith("flashinfer")
    impl = cascade.implementations(p, with_flashinfer=with_flashinfer)[impl_name]
    inputs = cascade.make_inputs(p, nhd_copies=with_flashinfer)
    impl.plan(inputs)
    impl.run(inputs)
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range(NVTX_RANGE):
        impl.run(inputs)
    torch.cuda.synchronize()


def measure(batch: int, prefix: int, impl_name: str) -> Traffic:
    from bench.suites.cascade import bytes_moved

    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu not found; install Nsight Compute or add it to PATH")
    cmd = [
        ncu,
        "--metrics",
        ",".join(METRICS),
        "--csv",
        "--cache-control",
        "all",
        "--clock-control",
        "none",
        "--nvtx",
        "--nvtx-include",
        f"{NVTX_RANGE}/",
        sys.executable,
        "-m",
        "bench.dram_bytes",
        "--ncu-target",
        str(batch),
        str(prefix),
        impl_name,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    values = parse_ncu_bytes(out)
    reads, writes = values[METRICS[0]], values[METRICS[1]]
    if not reads or len(reads) != len(writes):
        raise RuntimeError(f"ncu reported no matching kernels for {impl_name}; output:\n{out}")
    p = shape_params(batch, prefix)
    return Traffic(p, impl_name, len(reads), round(sum(reads)), round(sum(writes)), bytes_moved(p))


def write(result: DramResult, root: Path) -> Path:
    stamp = datetime.strptime(result.created_utc, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%dT%H%M%SZ")
    path = root / result.env.gpu_slug / result.suite / f"{stamp}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--ncu-target", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.ncu_target:
        batch, prefix, impl_name = args.ncu_target
        ncu_target(int(batch), int(prefix), impl_name)
        return 0

    from bench import env
    from bench.harness import git_sha
    from bench.suites.cascade import flashinfer_runs

    info = env.collect()
    if env.counters_restricted():
        print("dram_bytes: performance counters are restricted; ncu cannot run", file=sys.stderr)
        return 1
    impls = [i for i in IMPLS if flashinfer_runs() or not i.startswith("flashinfer")]
    measurements = []
    for batch, prefix in SHAPES:
        for impl_name in impls:
            t = measure(batch, prefix, impl_name)
            print(
                f"batch {batch} prefix {prefix} {impl_name}: read {t.read_bytes / 1e6:.1f} MB,"
                f" model {t.model_bytes / 1e6:.1f} MB, {t.kernels} kernels",
                flush=True,
            )
            measurements.append(t)
    result = DramResult("cascade-dram", git_sha(), utc_now(), info, measurements)
    print(f"wrote {write(result, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
