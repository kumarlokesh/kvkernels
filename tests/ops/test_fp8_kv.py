"""The cascade over an fp8 (e4m3) KV cache with one K and one V scale per layer.

The cache holds K / k_scale and V / v_scale in float8_e4m3fn, as vLLM's and
SGLang's fp8 caches do. With power-of-two scales, the dequantized values are
exact in fp16 and bf16, so attention over the fp8 cache must equal attention
over a 16-bit cache holding the dequantized values, bit for bit, and both must
match the fp32 reference within the tolerance for the query's dtype.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.ops.cascade import cascade_decode_attention
from tests.ops.cascade_cases import DEVICE, INTERPRETING
from tests.ops.test_cascade_plan import Setup, explicit_batch, plan_device
from tests.ops.test_tree_attention import attend, build, check, new_plan, tree_batch

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)

E4M3_MAX = 448.0


def quantize(view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(fp8 values, scale, dequantized values in view's dtype); the scale a power of two."""
    amax = float(view.float().abs().max())
    scale = 2.0 ** round(torch.log2(torch.tensor(amax / E4M3_MAX)).item())
    fp8 = (view.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    scale_t = torch.tensor([scale], dtype=torch.float32, device=view.device)
    return fp8, scale_t, (fp8.float() * scale).to(view.dtype)


def fp8_setup(setup: Setup) -> tuple[Setup, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The setup with K and V replaced by their dequantized values, and the fp8 copies."""
    k8, k_scale, k_deq = quantize(setup.k)
    v8, v_scale, v_deq = quantize(setup.v)
    setup.k.copy_(k_deq)
    setup.v.copy_(v_deq)
    return setup, k8, v8, k_scale, v_scale


def attend_fp8(
    setup: Setup,
    plan: object,
    k8: torch.Tensor,
    v8: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(setup.q)
    lse = torch.empty(setup.q.shape[:2], dtype=torch.float32, device=DEVICE)
    cascade_decode_attention(setup.q, k8, v8, plan, out, lse, k_scale=ks, v_scale=vs)  # type: ignore[arg-type]
    return out, lse


@pytest.mark.parametrize(
    ("layout", "query_tokens", "dtype"),
    [
        ("vllm_LBNHC", 1, torch.float16),
        ("vllm_LBNHC", 4, torch.float16),
        ("vllm_LBNHC", 1, torch.bfloat16),
        ("sglang_nhd", 1, torch.float16),
    ],
)
def test_fp8_equals_the_dequantized_cache_bit_for_bit(
    layout: str, query_tokens: int, dtype: torch.dtype
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 needs a GPU that supports it (the interpreter does not)")
    block_size = 4 if layout.startswith("vllm") else 1
    batch = tree_batch(random.Random(40), requests=10, block_size=block_size, min_tail=query_tokens)
    setup = build(batch, layout=layout, query_tokens=query_tokens)
    if dtype != torch.float16:
        setup = Setup(
            setup.q.to(dtype), setup.k.to(dtype), setup.v.to(dtype), setup.index, setup.requests
        )
    setup, k8, v8, ks, vs = fp8_setup(setup)
    plan = new_plan(setup, batch, query_tokens=query_tokens, min_shared_tokens=1)
    plan_device(plan, setup)
    out16, lse16 = attend(setup, plan)
    out8, lse8 = attend_fp8(setup, plan, k8, v8, ks, vs)
    assert torch.equal(out8, out16)
    assert torch.equal(lse8, lse16)
    check(setup, out8, lse8, query_tokens=query_tokens)


def test_scales_that_are_not_powers_of_two() -> None:
    # The dequantized values are not exact in fp16 here; the fp8 result must
    # still match the fp32 reference over K · k_scale and V · v_scale.
    batch = tree_batch(random.Random(41), requests=6, block_size=4)
    setup = build(batch)
    k8 = (setup.k.float() / 0.013).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    v8 = (setup.v.float() / 0.011).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    ks = torch.tensor([0.013], dtype=torch.float32, device=DEVICE)
    vs = torch.tensor([0.011], dtype=torch.float32, device=DEVICE)
    plan = new_plan(setup, batch, min_shared_tokens=1)
    plan_device(plan, setup)
    out8, _ = attend_fp8(setup, plan, k8, v8, ks, vs)
    from kvkernels.reference.attention import attention_ref32

    deq = Setup(setup.q, k8.float() * 0.013, v8.float() * 0.011, setup.index, setup.requests)
    out32, _ = attention_ref32(deq.q, deq.k, deq.v, deq.requests)
    # Scores rounded to fp16 inputs: the fp16 tolerance, at twice the usual size
    # because the reference's K and V are not rounded to fp16.
    assert float((out8.float() - out32).abs().max()) < 4e-3


def test_fp8_needs_both_scales_and_an_fp8_view() -> None:
    batch = explicit_batch([[1, 2], [1, 3]], [8, 8], 4)
    setup = build(batch)
    setup, k8, v8, ks, _ = fp8_setup(setup)
    plan = new_plan(setup, batch)
    plan_device(plan, setup)
    out = torch.empty_like(setup.q)
    with pytest.raises(ValueError, match="v_scale"):
        cascade_decode_attention(setup.q, k8, v8, plan, out, k_scale=ks)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        cascade_decode_attention(setup.q, k8.view(torch.uint8), v8.view(torch.uint8), plan, out)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        cascade_decode_attention(setup.q, k8, setup.v, plan, out, k_scale=ks, v_scale=ks)
