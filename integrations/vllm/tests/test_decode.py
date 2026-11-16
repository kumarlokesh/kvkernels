"""Decode through kvkernels on vLLM's per-layer cache tensor, without vLLM.

The cache comes from kvkernels.layouts, which builds vLLM's default layout the
way vLLM allocates it: one tensor per layer with axes (block, head, slot,
content), content being K then V.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache, make_index
from kvkernels.layouts import kv_views as layout_views
from kvkernels.reference.attention import RequestKV, attention_ref32, attention_ref_low
from kvkernels.tolerance import assert_close
from kvkernels_vllm.decode import check_decode, decode, kv_views, new_plan, unsupported

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING else "cuda"
pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)

BLOCK = 16
# Requests 0-2 share KV blocks 1-4 (64 tokens); request 3 shares nothing.
ROWS = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6, 7], [1, 2, 3, 4], [8, 9]]
LENS = [70, 90, 64, 20]


def setup(
    q_heads: int = 8, kv_heads: int = 2, head_dim: int = 64, dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], list[RequestKV]]:
    gen = torch.Generator().manual_seed(0)
    layout = LAYOUTS["vllm_LBNHC"]
    spec = CacheSpec(1, 12, BLOCK, kv_heads, head_dim, dtype)
    (kv_cache,) = make_cache(layout, spec, DEVICE)[0]
    kv_cache.copy_(torch.randn(tuple(kv_cache.shape), generator=gen).to(dtype))
    index = make_index(layout, spec, ROWS, LENS, DEVICE)
    q = torch.randn((len(LENS), q_heads, head_dim), generator=gen).to(dtype=dtype, device=DEVICE)
    requests = [RequestKV([], 0, row, n) for row, n in zip(ROWS, LENS, strict=True)]
    return kv_cache, q, index, requests


def run(
    kv_cache: torch.Tensor, q: torch.Tensor, index: dict[str, torch.Tensor], **plan_args: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[2]
    plan = new_plan(
        max_requests=8,
        num_q_heads=q.shape[1],
        num_kv_heads=kv_cache.shape[1],
        head_dim=head_dim,
        block_size=BLOCK,
        device=torch.device(DEVICE),
        **plan_args,
    )
    plan.plan_from_block_table(index["block_table"], index["seq_lens"])
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=DEVICE)
    decode(q, kv_cache, plan, out, head_size=head_dim, scale=head_dim**-0.5, lse_out=lse)
    return out, lse


def test_views_are_the_layout_module_views() -> None:
    kv_cache, _, _, _ = setup()
    spec = CacheSpec(1, 12, BLOCK, 2, 64, torch.float16)
    for ours, theirs in zip(
        kv_views(kv_cache, 64), layout_views(LAYOUTS["vllm_LBNHC"], spec, [kv_cache]), strict=True
    ):
        assert ours.shape == theirs.shape and ours.stride() == theirs.stride()
        assert ours.data_ptr() == theirs.data_ptr()


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_decode_matches_the_reference(batch_invariant: bool) -> None:
    kv_cache, q, index, requests = setup()
    out, lse = run(kv_cache, q, index, batch_invariant=batch_invariant)
    k, v = kv_views(kv_cache, 64)
    out_low, lse_low = attention_ref_low(q, k, v, requests)
    out32, lse32 = attention_ref32(q, k, v, requests)
    assert_close(out, out_low, out32)
    assert_close(lse, lse_low, lse32)


def test_the_check_passes_a_correct_result_and_fails_a_wrong_one() -> None:
    kv_cache, q, index, requests = setup()
    out, lse = run(kv_cache, q, index)
    k, v = kv_views(kv_cache, 64)
    engine_out, _ = attention_ref_low(q, k, v, requests)  # stands in for vLLM's kernel
    args = (q, kv_cache, index["block_table"], index["seq_lens"])
    check_decode(*args, out, lse, engine_out, head_size=64, scale=64**-0.5)
    wrong = out.clone()
    wrong[1] += 0.05
    with pytest.raises(AssertionError, match="exceeds"):
        check_decode(*args, wrong, lse, engine_out, head_size=64, scale=64**-0.5)


def test_unsupported_lists_every_reason() -> None:
    ok = {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "kv_cache_dtype": "auto",
        "sliding_window": False,
        "alibi": False,
        "soft_cap": False,
        "sinks": False,
        "decoder": True,
    }
    assert unsupported(**ok) == []  # type: ignore[arg-type]
    bad = {
        **ok,
        "head_size": 96,
        "num_kv_heads": 1,
        "kv_cache_dtype": "fp8_e5m2",
        "sliding_window": True,
    }
    reasons = unsupported(**bad)  # type: ignore[arg-type]
    assert len(reasons) == 4
    assert any("96" in r for r in reasons) and any("fp8_e5m2" in r for r in reasons)


def test_a_plan_larger_than_one_sorting_program_is_refused() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        new_plan(
            max_requests=8192,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            block_size=BLOCK,
            device=torch.device(DEVICE),
        )
