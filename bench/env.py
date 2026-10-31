"""Environment metadata and GPU checks for benchmark runs."""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from bench.schema import EnvInfo
from kvkernels.specs import gpu_slug

NVIDIA_PARAMS = Path("/proc/driver/nvidia/params")


@dataclass(frozen=True)
class SmiGpu:
    name: str
    driver_version: str
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None


def hostname_hash(hostname: str | None = None) -> str:
    """sha256 of the hostname; the raw hostname is never stored."""
    host = socket.gethostname() if hostname is None else hostname
    return hashlib.sha256(host.encode()).hexdigest()


def _int_or_none(text: str) -> int | None:
    text = text.strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_smi_gpu(line: str) -> SmiGpu:
    """Parse one line of ``nvidia-smi --format=csv,noheader,nounits``
    with ``--query-gpu=name,driver_version,clocks.sm,clocks.mem``.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        raise ValueError(f"unexpected nvidia-smi line: {line!r}")
    name, driver, sm, mem = parts
    return SmiGpu(name, driver, _int_or_none(sm), _int_or_none(mem))


def parse_compute_pids(output: str) -> list[int]:
    """Parse ``nvidia-smi --query-compute-apps=pid --format=csv,noheader``."""
    return [int(tok) for tok in output.split() if tok.strip().isdigit()]


def profiling_restricted(params_text: str, euid: int) -> bool:
    """True when performance counters need admin rights and this user is not root.

    The NVIDIA driver exposes the setting as ``RmProfilingAdminOnly`` in
    /proc/driver/nvidia/params. When it is 1, Nsight Compute fails for non-root
    users with ERR_NVGPUCTRPERM.
    """
    for line in params_text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "RmProfilingAdminOnly":
            return value.strip() == "1" and euid != 0
    return False


def _smi(*args: str) -> str:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found; this needs a machine with an NVIDIA GPU")
    return subprocess.run(["nvidia-smi", *args], check=True, capture_output=True, text=True).stdout


def query_gpu(index: int = 0) -> SmiGpu:
    out = _smi(
        f"--id={index}",
        "--query-gpu=name,driver_version,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    )
    return parse_smi_gpu(out.strip().splitlines()[0])


def other_compute_pids(index: int = 0) -> list[int]:
    """Compute processes on the GPU other than this one; benchmarks need an idle GPU."""
    out = _smi(f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader")
    return [pid for pid in parse_compute_pids(out) if pid != os.getpid()]


# One small kernel for Nsight Compute to profile.
COUNTER_PROBE = "import torch; torch.ones(8, device='cuda').add_(1); torch.cuda.synchronize()"


def counter_probe_result(returncode: int, output: str) -> bool | None:
    """Whether ncu was refused counter access, from its exit code and output; None if unclear."""
    if "ERR_NVGPUCTRPERM" in output:
        return True
    return False if returncode == 0 else None


def counters_restricted() -> bool | None:
    """Whether Nsight Compute will be refused counter access; None if unknown.

    With ncu on PATH, this profiles one small kernel and looks for
    ERR_NVGPUCTRPERM. The driver setting alone is not enough: inside a container
    (a Kaggle notebook) root is refused when the container lacks the host's
    permission. Without ncu, it falls back to the driver setting.
    """
    ncu = shutil.which("ncu")
    if ncu is not None:
        cmd = [ncu, "--metrics", "gpu__time_duration.sum", sys.executable, "-c", COUNTER_PROBE]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None
        return counter_probe_result(proc.returncode, proc.stdout + proc.stderr)
    if not NVIDIA_PARAMS.exists():
        return None
    return profiling_restricted(NVIDIA_PARAMS.read_text(), os.geteuid())


def collect(index: int = 0) -> EnvInfo:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible to torch")
    gpu = query_gpu(index)
    name = torch.cuda.get_device_name(index)
    return EnvInfo(
        gpu_name=name,
        gpu_slug=gpu_slug(name),
        driver_version=gpu.driver_version,
        cuda_version=str(torch.version.cuda),
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        sm_clock_mhz=gpu.sm_clock_mhz,
        mem_clock_mhz=gpu.mem_clock_mhz,
        hostname_hash=hostname_hash(),
    )
