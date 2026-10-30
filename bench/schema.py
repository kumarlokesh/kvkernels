"""Versioned result schema.

Results are JSON with sorted keys, one file per suite run, at
``results/<gpu_slug>/<suite>/<YYYYMMDDTHHMMSSZ>.json``.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

Params = dict[str, int | str]


@dataclass(frozen=True)
class EnvInfo:
    gpu_name: str
    gpu_slug: str
    driver_version: str
    cuda_version: str
    torch_version: str
    triton_version: str
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None
    hostname_hash: str


@dataclass(frozen=True)
class Measurement:
    params: Params
    impl: str
    median_ms: float
    p10_ms: float
    p90_ms: float
    samples: int
    bytes_moved: int | None
    flops: int | None
    gbps: float | None
    tflops: float | None
    pct_datasheet_bw: float | None
    pct_measured_bw: float | None
    pct_peak_compute: float | None
    host_us: float | None
    plan_us: float | None  # CPU time of the implementation's per-batch plan, if it has one
    noisy: bool


@dataclass(frozen=True)
class BenchResult:
    schema_version: int
    suite: str
    git_sha: str
    created_utc: str
    timer: str
    flush_bytes: int
    env: EnvInfo
    measured_peak_gbps: float | None
    measurements: list[Measurement]


class SchemaError(ValueError):
    """A result file does not match the schema."""


def utc_now() -> str:
    """Current UTC time in the format stored in created_utc, e.g. 2026-10-23T22:15:00Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def result_path(result: BenchResult, root: Path = Path("results")) -> Path:
    created = datetime.strptime(result.created_utc, "%Y-%m-%dT%H:%M:%SZ")
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    return root / result.env.gpu_slug / result.suite / f"{stamp}.json"


def to_json(result: BenchResult) -> str:
    return json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True) + "\n"


def write_result(result: BenchResult, root: Path = Path("results")) -> Path:
    path = result_path(result, root)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(result))
    return path


def read_result(path: Path) -> BenchResult:
    data = json.loads(path.read_text())
    return from_dict(data)


def from_dict(data: Any) -> BenchResult:
    if not isinstance(data, dict):
        raise SchemaError("result must be a JSON object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version {version!r}; this code reads version {SCHEMA_VERSION}"
        )
    result: BenchResult = _build(BenchResult, data, "result")
    return result


def _build(cls: type[Any], data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise SchemaError(f"{where}: expected an object")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    missing = sorted(set(names) - set(data))
    extra = sorted(set(data) - set(names))
    if missing or extra:
        raise SchemaError(f"{where}: missing fields {missing}, unknown fields {extra}")
    return cls(**{n: _check(hints[n], data[n], f"{where}.{n}") for n in names})


def _check(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        for option in args:
            try:
                return _check(option, value, where)
            except SchemaError:
                continue
        raise SchemaError(f"{where}: {value!r} does not match {tp}")
    if tp is type(None):
        if value is not None:
            raise SchemaError(f"{where}: expected null")
        return None
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return _build(tp, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise SchemaError(f"{where}: expected a list")
        return [_check(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise SchemaError(f"{where}: expected an object")
        return {
            _check(args[0], k, where): _check(args[1], v, f"{where}.{k}") for k, v in value.items()
        }
    if tp is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"{where}: expected a boolean")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{where}: expected an integer")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(f"{where}: expected a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise SchemaError(f"{where}: expected a string")
        return value
    raise SchemaError(f"{where}: unsupported type {tp}")
