"""GPU specifications used for roofline and percent-of-peak figures.

Every entry in ``SPECS`` comes from a vendor datasheet or architecture whitepaper
cited in ``source_url``. A GPU without an entry still benchmarks; its results
leave the datasheet-based percentages empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    slug: str
    name: str
    peak_hbm_gbps: float
    peak_fp16_tensor_tflops_dense_fp16acc: float
    peak_fp16_tensor_tflops_dense_fp32acc: float
    peak_bf16_tensor_tflops_dense_fp32acc: float
    l2_bytes: int
    smem_per_sm_bytes: int
    num_sms: int
    source_url: str


SPECS: dict[str, GPUSpec] = {}


def gpu_slug(device_name: str) -> str:
    """Map a device name, as torch.cuda.get_device_name() reports it, to a short slug.

    Known GPUs get the slugs used in specs and result paths ("h100-sxm", "rtx4090").
    Any other name becomes its lowercase words joined by hyphens.
    """
    name = device_name.lower()
    if "h100" in name:
        if "pcie" in name:
            return "h100-pcie"
        if "nvl" in name:
            return "h100-nvl"
        return "h100-sxm"
    if "a100" in name and "sxm" in name and "80gb" in name:
        return "a100-sxm-80gb"
    if "rtx 4090" in name:
        return "rtx4090"
    if re.search(r"\bl4\b", name):
        return "l4"
    words = re.findall(r"[a-z0-9]+", name.replace("nvidia", ""))
    return "-".join(words) or "unknown"


def lookup(device_name: str) -> GPUSpec | None:
    """Return the spec for a device name, or None when the GPU is not in SPECS."""
    return SPECS.get(gpu_slug(device_name))


def detect() -> GPUSpec:
    """Return the spec of the current CUDA device; raise if it is not in SPECS."""
    import torch

    name = torch.cuda.get_device_name()
    spec = lookup(name)
    if spec is None:
        raise LookupError(f"no GPUSpec for {name!r} (slug {gpu_slug(name)!r}); add it to SPECS")
    return spec


def peak_tensor_tflops(spec: GPUSpec, dtype: str) -> float | None:
    """Dense tensor-core peak for a dtype with fp32 accumulation, or None if not tabulated."""
    if dtype in ("fp16", "float16"):
        return spec.peak_fp16_tensor_tflops_dense_fp32acc
    if dtype in ("bf16", "bfloat16"):
        return spec.peak_bf16_tensor_tflops_dense_fp32acc
    return None
