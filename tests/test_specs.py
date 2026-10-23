from __future__ import annotations

import pytest

from kvkernels.specs import SPECS, GPUSpec, gpu_slug, lookup, peak_tensor_tflops


@pytest.mark.parametrize("spec", SPECS.values(), ids=list(SPECS))
def test_every_entry_cites_sources(spec: GPUSpec) -> None:
    assert spec.sources
    assert all(url.startswith("https://") for url in spec.sources)


@pytest.mark.parametrize("spec", SPECS.values(), ids=list(SPECS))
def test_numbers_are_positive_and_consistent(spec: GPUSpec) -> None:
    assert spec.peak_hbm_gbps > 0
    assert spec.l2_bytes > 0 and spec.smem_per_sm_bytes > 0 and spec.num_sms > 0
    # fp32 accumulation is never faster than fp16 accumulation.
    assert (
        0 < spec.peak_fp16_tensor_tflops_dense_fp32acc <= spec.peak_fp16_tensor_tflops_dense_fp16acc
    )


@pytest.mark.parametrize("spec", SPECS.values(), ids=list(SPECS))
def test_slug_round_trips_through_the_device_name(spec: GPUSpec) -> None:
    assert gpu_slug(spec.name) == spec.slug
    assert lookup(spec.name) is spec


def test_geforce_halves_fp32_accumulation() -> None:
    rtx = SPECS["rtx4090"]
    assert peak_tensor_tflops(rtx, "fp16", "fp16") == 330.3
    assert peak_tensor_tflops(rtx, "fp16", "fp32") == 165.2


def test_t4_has_no_bf16_tensor_cores() -> None:
    assert peak_tensor_tflops(SPECS["t4"], "bf16") is None


def test_unknown_dtype_has_no_peak() -> None:
    assert peak_tensor_tflops(SPECS["h100-sxm"], "fp32") is None
    assert peak_tensor_tflops(SPECS["h100-sxm"], "bf16", "fp16") is None


def test_tesla_t4_name() -> None:
    assert gpu_slug("Tesla T4") == "t4"


@pytest.mark.gpu
def test_spec_matches_the_device() -> None:
    """SM count and L2 size in the table agree with what the driver reports."""
    import torch

    spec = lookup(torch.cuda.get_device_name())
    if spec is None:
        pytest.skip("GPU not in SPECS")
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == spec.num_sms
    assert props.L2_cache_size == spec.l2_bytes
