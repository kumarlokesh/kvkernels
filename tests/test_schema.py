from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from bench.schema import (
    SCHEMA_VERSION,
    BenchResult,
    EnvInfo,
    Measurement,
    SchemaError,
    from_dict,
    read_result,
    result_path,
    to_json,
    write_result,
)


def make_result(**overrides: object) -> BenchResult:
    env = EnvInfo(
        gpu_name="NVIDIA L4",
        gpu_slug="l4",
        driver_version="580.95",
        cuda_version="13.0",
        torch_version="2.14.1",
        triton_version="3.8.0",
        sm_clock_mhz=2040,
        mem_clock_mhz=None,
        hostname_hash="0" * 64,
    )
    m = Measurement(
        params={"bytes": 1024, "dtype": "fp16"},
        impl="torch_copy",
        median_ms=0.5,
        p10_ms=0.45,
        p90_ms=0.6,
        samples=120,
        bytes_moved=2048,
        flops=None,
        gbps=4.096e-3,
        tflops=None,
        pct_datasheet_bw=None,
        pct_measured_bw=0.9,
        pct_peak_compute=None,
        host_us=None,
        noisy=False,
    )
    fields = dict(
        schema_version=SCHEMA_VERSION,
        suite="smoke",
        git_sha="dirty-abc123",
        created_utc="2026-10-23T22:15:00Z",
        timer="do_bench-cold-l2",
        flush_bytes=256 << 20,
        env=env,
        measured_peak_gbps=250.0,
        measurements=[m],
    )
    fields.update(overrides)
    return BenchResult(**fields)  # type: ignore[arg-type]


def test_round_trip(tmp_path: Path) -> None:
    result = make_result()
    path = write_result(result, tmp_path)
    assert read_result(path) == result


def test_path_is_gpu_suite_timestamp(tmp_path: Path) -> None:
    path = result_path(make_result(), tmp_path)
    assert path == tmp_path / "l4" / "smoke" / "20261023T221500Z.json"


def test_json_is_stable_and_sorted() -> None:
    text = to_json(make_result())
    assert text == to_json(make_result())
    keys = list(json.loads(text))
    assert keys == sorted(keys)


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    write_result(make_result(), tmp_path)
    with pytest.raises(FileExistsError):
        write_result(make_result(), tmp_path)


def test_rejects_unknown_schema_version() -> None:
    data = dataclasses.asdict(make_result())
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(SchemaError, match="schema_version"):
        from_dict(data)


def test_rejects_missing_and_unknown_fields() -> None:
    data = dataclasses.asdict(make_result())
    del data["env"]["gpu_slug"]
    data["env"]["extra"] = 1
    with pytest.raises(SchemaError, match="gpu_slug"):
        from_dict(data)


def test_rejects_wrong_types() -> None:
    data = dataclasses.asdict(make_result())
    data["measurements"][0]["samples"] = "120"
    with pytest.raises(SchemaError, match="samples"):
        from_dict(data)
    data = dataclasses.asdict(make_result())
    data["measurements"][0]["noisy"] = 0
    with pytest.raises(SchemaError, match="noisy"):
        from_dict(data)


def test_accepts_integer_for_float_field() -> None:
    data = dataclasses.asdict(make_result())
    data["measurements"][0]["median_ms"] = 1
    assert from_dict(data).measurements[0].median_ms == 1.0
