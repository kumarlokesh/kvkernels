"""Print the environment of this GPU machine.

Run first on every rented GPU: ``uv run python scripts/env_report.py``.
Exits non-zero when there is no usable GPU.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import env
from kvkernels.specs import lookup


def main() -> int:
    try:
        info = env.collect()
    except RuntimeError as exc:
        print(f"env_report: {exc}", file=sys.stderr)
        return 1

    for key, value in dataclasses.asdict(info).items():
        print(f"{key:16s} {value}")

    spec = lookup(info.gpu_name)
    print(
        f"{'spec':16s} {spec.slug if spec else 'not in SPECS (datasheet percentages stay empty)'}"
    )

    others = env.other_compute_pids()
    print(f"{'other_processes':16s} {others or 'none'}")

    restricted = env.counters_restricted()
    if restricted is None:
        print(f"{'perf_counters':16s} unknown (ncu gave no answer and no {env.NVIDIA_PARAMS})")
    elif restricted:
        print(f"{'perf_counters':16s} RESTRICTED: Nsight Compute will fail with ERR_NVGPUCTRPERM")
        print("                 kernel profiling needs a provider that allows counter access")
    else:
        print(f"{'perf_counters':16s} available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
