from __future__ import annotations

import hashlib

import pytest

from bench.env import (
    hostname_hash,
    parse_compute_pids,
    parse_smi_gpu,
    profiling_restricted,
)
from kvkernels.specs import gpu_slug


def test_parse_smi_gpu() -> None:
    gpu = parse_smi_gpu("NVIDIA L4, 580.95.05, 2040, 6251")
    assert (gpu.name, gpu.driver_version, gpu.sm_clock_mhz, gpu.mem_clock_mhz) == (
        "NVIDIA L4",
        "580.95.05",
        2040,
        6251,
    )


def test_parse_smi_gpu_unknown_clock() -> None:
    assert parse_smi_gpu("NVIDIA L4, 580.95.05, [N/A], 6251").sm_clock_mhz is None


def test_parse_smi_gpu_rejects_bad_line() -> None:
    with pytest.raises(ValueError):
        parse_smi_gpu("NVIDIA L4, 580")


def test_parse_compute_pids() -> None:
    assert parse_compute_pids("1234\n5678\n") == [1234, 5678]
    assert parse_compute_pids("") == []


def test_hostname_is_hashed() -> None:
    assert hostname_hash("box") == hashlib.sha256(b"box").hexdigest()


@pytest.mark.parametrize(
    ("params", "euid", "restricted"),
    [
        ("RmProfilingAdminOnly: 1\n", 1000, True),
        ("RmProfilingAdminOnly: 1\n", 0, False),
        ("RmProfilingAdminOnly: 0\n", 1000, False),
        ("SomethingElse: 1\n", 1000, False),
    ],
)
def test_profiling_restricted(params: str, euid: int, restricted: bool) -> None:
    assert profiling_restricted(params, euid) is restricted


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("NVIDIA H100 80GB HBM3", "h100-sxm"),
        ("NVIDIA H100 PCIe", "h100-pcie"),
        ("NVIDIA A100-SXM4-80GB", "a100-sxm-80gb"),
        ("NVIDIA GeForce RTX 4090", "rtx4090"),
        ("NVIDIA L4", "l4"),
        ("NVIDIA L40S", "l40s"),
        ("NVIDIA RTX A6000", "rtx-a6000"),
    ],
)
def test_gpu_slug(name: str, slug: str) -> None:
    assert gpu_slug(name) == slug
