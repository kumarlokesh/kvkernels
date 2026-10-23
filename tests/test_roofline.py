from __future__ import annotations

import pytest

from kvkernels.roofline import (
    arithmetic_intensity,
    attainable_tflops,
    bound,
    decode_attention_cost,
    peak_tflops,
    ridge_point,
    shared_prefix_attention_cost,
)
from kvkernels.specs import SPECS

H100 = SPECS["h100-sxm"]
RTX4090 = SPECS["rtx4090"]


def test_decode_cost_by_hand() -> None:
    # One request, 4096 tokens, 32 query heads, 8 KV heads, head_dim 128, fp16.
    flops, nbytes = decode_attention_cost(1, 32, 8, 4096, 128, 2)
    assert nbytes == 2 * 4096 * 8 * 128 * 2 == 16_777_216
    assert flops == 4 * 4096 * 32 * 128 == 67_108_864
    assert arithmetic_intensity(flops, nbytes) == 4.0  # the GQA group size


def test_decode_intensity_does_not_depend_on_batch_or_context() -> None:
    small = arithmetic_intensity(*decode_attention_cost(1, 32, 8, 512, 128, 2))
    large = arithmetic_intensity(*decode_attention_cost(64, 32, 8, 32768, 128, 2))
    assert small == large == 4.0


def test_shared_prefix_intensity_grows_with_batch() -> None:
    one = arithmetic_intensity(*shared_prefix_attention_cost(1, 32, 8, 4096, 128, 2))
    many = arithmetic_intensity(*shared_prefix_attention_cost(32, 32, 8, 4096, 128, 2))
    assert one == 4.0
    assert many == 128.0


def test_shared_prefix_reads_the_prefix_once() -> None:
    _, shared_bytes = shared_prefix_attention_cost(32, 32, 8, 4096, 128, 2)
    _, naive_bytes = decode_attention_cost(32, 32, 8, 4096, 128, 2)
    assert naive_bytes == 32 * shared_bytes


@pytest.mark.parametrize(
    ("slug", "dtype", "acc", "expected"),
    [
        ("h100-sxm", "bf16", "fp32", 989.5e12 / 3350e9),
        ("a100-sxm-80gb", "bf16", "fp32", 312e12 / 2039e9),
        ("rtx4090", "fp16", "fp16", 330.3e12 / 1008e9),
        ("rtx4090", "fp16", "fp32", 165.2e12 / 1008e9),
        ("l4", "fp16", "fp32", 121e12 / 300e9),
        ("t4", "fp16", "fp32", 65e12 / 320e9),
    ],
)
def test_ridge_points(slug: str, dtype: str, acc: str, expected: float) -> None:
    assert ridge_point(SPECS[slug], dtype, acc) == pytest.approx(expected)


def test_attainable_follows_bandwidth_below_the_ridge() -> None:
    # AI 4 on an H100: 4 FLOP/B x 3350 GB/s = 13.4 TFLOP/s.
    assert attainable_tflops(4.0, H100, "bf16") == pytest.approx(13.4)


def test_attainable_is_capped_at_peak_above_the_ridge() -> None:
    assert attainable_tflops(10_000.0, H100, "bf16") == peak_tflops(H100, "bf16")


def test_bound_flips_at_the_ridge() -> None:
    ridge = ridge_point(H100, "bf16")
    assert bound(ridge * 0.99, H100, "bf16") == "memory"
    assert bound(ridge * 1.01, H100, "bf16") == "compute"


def test_fp32_accumulation_moves_the_geforce_ridge() -> None:
    assert ridge_point(RTX4090, "fp16", "fp32") == pytest.approx(
        ridge_point(RTX4090, "fp16", "fp16") / 2, rel=1e-3
    )


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        decode_attention_cost(1, 30, 8, 4096, 128, 2)  # 30 is not a multiple of 8
    with pytest.raises(ValueError):
        arithmetic_intensity(1, 0)
    with pytest.raises(ValueError):
        peak_tflops(SPECS["t4"], "bf16")
