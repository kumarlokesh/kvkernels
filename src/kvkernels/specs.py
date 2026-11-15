"""GPU specifications used for roofline and percent-of-peak figures.

Every number comes from an NVIDIA datasheet, architecture whitepaper or CUDA
documentation page listed in the entry's ``sources``. Tensor-core throughput is
the dense rate (datasheets often headline the 2:4 sparsity rate, which is twice
as high). Bandwidth is in decimal GB/s, as datasheets state it. Cache and shared
memory sizes are in bytes; the whitepapers' "KB" and "MB" for these mean 1024
and 1024² bytes.

A GPU without an entry still benchmarks; its results leave the datasheet-based
percentages empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KiB = 1024
MiB = 1024 * KiB

_TURING_WP = (
    "https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/"
    "turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf"
)
_AMPERE_WP = (
    "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/"
    "nvidia-ampere-architecture-whitepaper.pdf"
)
_A100_80GB_DS = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/"
    "a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf"
)
_ADA_WP = "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
_HOPPER_WP = "https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper"
_H100_PAGE = "https://www.nvidia.com/en-us/data-center/h100/"
_CC_TABLE = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html"
)


@dataclass(frozen=True)
class GPUSpec:
    slug: str
    name: str
    peak_hbm_gbps: float
    peak_fp16_tensor_tflops_dense_fp16acc: float
    peak_fp16_tensor_tflops_dense_fp32acc: float
    peak_bf16_tensor_tflops_dense_fp32acc: float | None  # None: no bf16 tensor cores
    l2_bytes: int
    smem_per_sm_bytes: int
    num_sms: int
    sources: tuple[str, ...]
    notes: str = ""


SPECS: dict[str, GPUSpec] = {
    spec.slug: spec
    for spec in (
        GPUSpec(
            slug="t4",
            name="NVIDIA T4",
            peak_hbm_gbps=320.0,
            peak_fp16_tensor_tflops_dense_fp16acc=65.0,
            peak_fp16_tensor_tflops_dense_fp32acc=65.0,
            peak_bf16_tensor_tflops_dense_fp32acc=None,
            l2_bytes=4096 * KiB,
            smem_per_sm_bytes=64 * KiB,
            num_sms=40,
            sources=(_TURING_WP, _ADA_WP, _CC_TABLE),
            notes=(
                "Both whitepapers give 320 GB/s; the T4 datasheet gives 300 GB/s, which is "
                "320e9 B/s expressed in GiB/s (inference). L2 from the Ada whitepaper, Table 5."
            ),
        ),
        GPUSpec(
            slug="l4",
            name="NVIDIA L4",
            peak_hbm_gbps=300.0,
            peak_fp16_tensor_tflops_dense_fp16acc=121.0,
            peak_fp16_tensor_tflops_dense_fp32acc=121.0,
            peak_bf16_tensor_tflops_dense_fp32acc=121.0,
            l2_bytes=49152 * KiB,
            smem_per_sm_bytes=100 * KiB,
            num_sms=58,
            sources=(_ADA_WP, _CC_TABLE),
            notes=(
                "The Ada whitepaper (Appendix D) gives one fp16/bf16 rate without naming the "
                "accumulation precision. GeForce Ada parts run fp32 accumulation at half rate; "
                "the L4 does not: a 4,096-square fp16 matmul reached 76.3 TFLOPS with fp32 "
                "accumulation and 78.8 with fp16 (Triton, 2026-11-15)."
            ),
        ),
        GPUSpec(
            slug="rtx4090",
            name="NVIDIA GeForce RTX 4090",
            peak_hbm_gbps=1008.0,
            peak_fp16_tensor_tflops_dense_fp16acc=330.3,
            peak_fp16_tensor_tflops_dense_fp32acc=165.2,
            peak_bf16_tensor_tflops_dense_fp32acc=165.2,
            l2_bytes=73728 * KiB,
            smem_per_sm_bytes=100 * KiB,
            num_sms=128,
            sources=(_ADA_WP, _CC_TABLE),
            notes="Ada whitepaper, Appendix A. fp32 accumulation runs at half the fp16 rate.",
        ),
        GPUSpec(
            slug="a100-sxm-80gb",
            name="NVIDIA A100-SXM4-80GB",
            peak_hbm_gbps=2039.0,
            peak_fp16_tensor_tflops_dense_fp16acc=312.0,
            peak_fp16_tensor_tflops_dense_fp32acc=312.0,
            peak_bf16_tensor_tflops_dense_fp32acc=312.0,
            l2_bytes=40960 * KiB,
            smem_per_sm_bytes=164 * KiB,
            num_sms=108,
            sources=(_A100_80GB_DS, _AMPERE_WP, _CC_TABLE),
            notes=(
                "Bandwidth from the 80GB datasheet; SMs, L2 and tensor rates from the whitepaper."
            ),
        ),
        GPUSpec(
            slug="h100-sxm",
            name="NVIDIA H100 80GB HBM3",
            peak_hbm_gbps=3350.0,
            peak_fp16_tensor_tflops_dense_fp16acc=989.5,
            peak_fp16_tensor_tflops_dense_fp32acc=989.5,
            peak_bf16_tensor_tflops_dense_fp32acc=989.5,
            l2_bytes=50 * MiB,
            smem_per_sm_bytes=228 * KiB,
            num_sms=132,
            sources=(_H100_PAGE, _HOPPER_WP, _CC_TABLE),
            notes=(
                "Bandwidth and tensor rates from the product page (1,979 TFLOPS with sparsity, "
                "halved). The whitepaper's own figures (3000 GB/s, 1000 TFLOPS) predate launch "
                "and are marked not finalized; SMs and L2 come from it."
            ),
        ),
    )
}


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
    if re.search(r"\bt4\b", name):
        return "t4"
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


def peak_tensor_tflops(spec: GPUSpec, dtype: str, acc_dtype: str = "fp32") -> float | None:
    """Dense tensor-core peak for an input dtype and accumulation dtype.

    None when the GPU has no tensor-core path for that combination, or the dtype is
    not one the table covers.
    """
    if dtype in ("fp16", "float16"):
        if acc_dtype in ("fp16", "float16"):
            return spec.peak_fp16_tensor_tflops_dense_fp16acc
        if acc_dtype in ("fp32", "float32"):
            return spec.peak_fp16_tensor_tflops_dense_fp32acc
    if dtype in ("bf16", "bfloat16") and acc_dtype in ("fp32", "float32"):
        return spec.peak_bf16_tensor_tflops_dense_fp32acc
    return None
