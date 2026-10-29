"""Fused attention forward against its references.

The kernel module (kernel.py) provides:

    CONFIGS: list[dict[str, int]]
        the autotuning list; keys BLOCK_M, BLOCK_N, num_stages, num_warps
    attention_fwd(q, k, v, out, lse, *, causal, sm_scale=None, config=None) -> None
        q: [batch, q_heads, seq_q, head_dim]; k, v: [batch, kv_heads, seq_k, head_dim];
        q_heads a multiple of kv_heads; head_dim 64 or 128; fp16 or bf16.
        out: like q. lse: [batch, q_heads, seq_q], fp32.
        causal needs seq_q == seq_k. sm_scale defaults to 1 / sqrt(head_dim).
        config, one entry of CONFIGS, replaces autotuning.

Tolerance, for out and for lse: err(kernel) ≤ 2 · err(ref_low) + floor.
Tests without the gpu mark keep every tensor at or below 65,536 elements, with
head_dim 64 and sequences of at most 256.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from bench.schema import Params
from experiments.fused_attention_fwd.reference import (
    attention_ref32,
    attention_ref_low,
    repeat_kv,
)
from kvkernels.tolerance import assert_close

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING or not torch.cuda.is_available() else "cuda"

# (batch, q_heads, kv_heads, seq_q, seq_k, head_dim)
SMALL_SHAPES = [
    (1, 2, 1, 128, 128, 64),
    (2, 4, 2, 100, 100, 64),
    (4, 2, 2, 128, 128, 64),
    (1, 1, 1, 1, 1, 64),
]
# Fewer queries than keys, as in the shared-prefix pass; non-causal only.
SHORT_QUERIES = (2, 4, 2, 16, 200, 64)


def kernel_module() -> ModuleType:
    """kernel.py, or a skip while it has not been written; other import errors fail."""
    try:
        return importlib.import_module("experiments.fused_attention_fwd.kernel")
    except ModuleNotFoundError as exc:
        if exc.name == "experiments.fused_attention_fwd.kernel":
            pytest.skip("experiments/fused_attention_fwd/kernel.py is not written yet")
        raise


needs_kernel_runtime = pytest.mark.skipif(
    not (INTERPRETING or torch.cuda.is_available()),
    reason="needs the Triton interpreter or a CUDA GPU",
)


def inputs(
    shape: tuple[int, int, int, int, int, int], dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    batch, q_heads, kv_heads, seq_q, seq_k, head_dim = shape
    gen = torch.Generator().manual_seed(seed)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(size, generator=gen).to(dtype=dtype, device=DEVICE)

    q = randn(batch, q_heads, seq_q, head_dim)
    k = randn(batch, kv_heads, seq_k, head_dim)
    v = randn(batch, kv_heads, seq_k, head_dim)
    out = torch.empty_like(q)
    lse = torch.empty((batch, q_heads, seq_q), dtype=torch.float32, device=DEVICE)
    return q, k, v, out, lse


def run_and_check(
    shape: tuple[int, int, int, int, int, int], dtype: torch.dtype, causal: bool
) -> None:
    kernel = kernel_module()
    q, k, v, out, lse = inputs(shape, dtype)
    kernel.attention_fwd(q, k, v, out, lse, causal=causal)
    assert not torch.isnan(out).any(), "NaN in out"
    assert not torch.isnan(lse).any(), "NaN in lse"
    out_low, lse_low = attention_ref_low(q, k, v, causal=causal)
    out32, lse32 = attention_ref32(q, k, v, causal=causal)
    assert_close(out, out_low, out32)
    assert_close(lse, lse_low, lse32)


# --- the references themselves (no kernel needed) ---------------------------


def test_ref32_matches_sdpa_in_fp64() -> None:
    q, k, v, _, _ = inputs((2, 4, 2, 50, 50, 64), torch.float16)
    for causal in (False, True):
        out, _ = attention_ref32(q, k, v, causal=causal)
        expected = F.scaled_dot_product_attention(
            q.double(), repeat_kv(k, 2).double(), repeat_kv(v, 2).double(), is_causal=causal
        )
        torch.testing.assert_close(out, expected.float(), rtol=1e-6, atol=1e-6)


def test_ref32_lse_is_logsumexp_of_scaled_scores() -> None:
    q, k, v, _, _ = inputs((1, 2, 2, 30, 40, 64), torch.float16)
    _, lse = attention_ref32(q, k, v, causal=False)
    scores = (q.double() @ k.double().transpose(-1, -2)) / 8.0  # sqrt(64)
    torch.testing.assert_close(lse, torch.logsumexp(scores, -1).float(), rtol=1e-6, atol=1e-6)


def test_gqa_equals_repeating_kv_heads() -> None:
    q, k, v, _, _ = inputs((1, 4, 2, 20, 20, 64), torch.float16)
    grouped = attention_ref32(q, k, v, causal=True)
    repeated = attention_ref32(q, repeat_kv(k, 2), repeat_kv(v, 2), causal=True)
    for a, b in zip(grouped, repeated, strict=True):
        assert torch.equal(a, b)


def test_first_causal_row_sees_only_the_first_key() -> None:
    q, k, v, _, _ = inputs((1, 2, 2, 16, 16, 64), torch.float16)
    out, lse = attention_ref32(q, k, v, causal=True)
    torch.testing.assert_close(out[:, :, 0], v[:, :, 0].float(), rtol=0, atol=1e-6)
    score = (q[:, :, 0].double() * k[:, :, 0].double()).sum(-1) / 8.0
    torch.testing.assert_close(lse[:, :, 0], score.float(), rtol=1e-6, atol=1e-6)


def test_chunked_references_match_one_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.fused_attention_fwd import reference

    q, k, v, _, _ = inputs((1, 2, 1, 64, 64, 64), torch.float16)
    whole = attention_ref32(q, k, v, causal=True)
    monkeypatch.setattr(reference, "CHUNK_ELEMENTS", 2 * 64 * 5)  # 5 rows per chunk
    chunked = attention_ref32(q, k, v, causal=True)
    for a, b in zip(whole, chunked, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# --- the kernel ----------------------------------------------------------------


@needs_kernel_runtime
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SMALL_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_reference(shape: tuple[int, int, int, int, int, int], causal: bool) -> None:
    run_and_check(shape, torch.float16, causal)


@needs_kernel_runtime
def test_fewer_queries_than_keys() -> None:
    run_and_check(SHORT_QUERIES, torch.float16, causal=False)


@needs_kernel_runtime
def test_first_causal_row_equals_the_first_value() -> None:
    kernel = kernel_module()
    q, k, v, out, lse = inputs((1, 4, 2, 64, 64, 64), torch.float16)
    kernel.attention_fwd(q, k, v, out, lse, causal=True)
    torch.testing.assert_close(out[:, :, 0], repeat_kv(v, 2)[:, :, 0], rtol=0, atol=1e-3)


@needs_kernel_runtime
def test_every_configuration_is_correct() -> None:
    kernel = kernel_module()
    q, k, v, out, lse = inputs((1, 2, 1, 100, 100, 64), torch.float16)
    out_low, lse_low = attention_ref_low(q, k, v, causal=True)
    out32, lse32 = attention_ref32(q, k, v, causal=True)
    for config in kernel.CONFIGS:
        out.zero_()
        kernel.attention_fwd(q, k, v, out, lse, causal=True, config=config)
        assert_close(out, out_low, out32)
        assert_close(lse, lse_low, lse32)


@needs_kernel_runtime
def test_rejects_bad_arguments() -> None:
    kernel = kernel_module()
    q, k, v, out, lse = inputs((1, 4, 2, 16, 16, 64), torch.float16)
    with pytest.raises(ValueError, match="multiple"):
        kernel.attention_fwd(q[:, :3], k, v, out[:, :3], lse[:, :3], causal=False)
    with pytest.raises(ValueError, match="causal"):
        kernel.attention_fwd(q[:, :, :8], k, v, out[:, :, :8], lse[:, :, :8], causal=True)
    with pytest.raises(ValueError, match="head_dim"):
        kernel.attention_fwd(
            q[..., :48], k[..., :48], v[..., :48], out[..., :48], lse, causal=False
        )
    with pytest.raises(ValueError, match="lse"):
        kernel.attention_fwd(q, k, v, out, lse.half(), causal=False)
    with pytest.raises(ValueError, match="dtype"):
        kernel.attention_fwd(q, k.float(), v, out, lse, causal=False)


# --- GPU tier ---------------------------------------------------------------------

GPU_SEQS = [512, 1000, 2048, 4096, 8192]


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_heads", [8, 32])
@pytest.mark.parametrize("seq", GPU_SEQS)
def test_full_shapes(seq: int, kv_heads: int, head_dim: int, causal: bool) -> None:
    run_and_check((4, 32, kv_heads, seq, seq, head_dim), torch.float16, causal)


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_bf16(causal: bool) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU has no bf16 support")
    run_and_check((4, 32, 8, 2048, 2048, 128), torch.bfloat16, causal)


# --- the benchmark suite (no kernel needed) --------------------------------------


def test_suite_grid() -> None:
    from experiments.fused_attention_fwd import bench_suite

    params = list(bench_suite.grid())
    assert len(params) == 5 * 2 * 2 * 2 + 1
    assert sum(bench_suite.is_sweep_shape(p) for p in params) == 2  # causal and not
    full: Params = {"seq": 1024, "head_dim": 64, "causal": "no"}
    assert bench_suite.flops(full) == 4 * 4 * 32 * 1024 * 1024 * 64
    assert bench_suite.flops({**full, "causal": "yes"}) == bench_suite.flops(full) // 2
