"""Merging partial attention results.

Each partial result is an output o and an LSE for one key set. Merging the
partials of disjoint key sets gives attention over their union:

    lse = log Σ_j exp(lse_j)          out = Σ_j exp(lse_j - lse) · o_j

An empty key set has lse = -inf and contributes nothing; if every set is empty
the merged output is 0 and the merged lse is -inf, with no NaN.
"""

from __future__ import annotations

import math
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from kvkernels.ops.merge_states import (
    merge_attention_states,
    merge_attention_states_n,
)
from kvkernels.reference.attention import (
    RequestKV,
    attention_ref32,
    merge_states_ref,
)

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING else "cuda"

pytestmark = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)

NEG_INF = float("-inf")


def states(
    splits: int, tokens: int = 5, heads: int = 3, head_dim: int = 64, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    o = torch.randn((splits, tokens, heads, head_dim), generator=gen).to(DEVICE)
    lse = (torch.randn((splits, tokens, heads), generator=gen) * 3).to(DEVICE)
    return o, lse


def outputs(
    o: torch.Tensor, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(o.shape[1:], dtype=dtype, device=DEVICE)
    lse_out = torch.empty(o.shape[1:-1], dtype=torch.float32, device=DEVICE)
    return out, lse_out


def check(out: torch.Tensor, lse_out: torch.Tensor, o: torch.Tensor, lse: torch.Tensor) -> None:
    ref, ref_lse = merge_states_ref(o, lse)
    assert not torch.isnan(out).any() and not torch.isnan(lse_out).any()
    tol = 1e-5 if out.dtype == torch.float32 else 2e-3
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(lse_out, ref_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("splits", [1, 2, 7, 16])
def test_n_way_matches_reference(splits: int) -> None:
    o, lse = states(splits)
    out, lse_out = outputs(o)
    merge_attention_states_n(o, lse, out, lse_out)
    check(out, lse_out, o, lse)


def test_two_way_matches_n_way() -> None:
    o, lse = states(2)
    out2, lse2 = outputs(o)
    outn, lsen = outputs(o)
    merge_attention_states(o[0], lse[0], o[1], lse[1], out2, lse2)
    merge_attention_states_n(o, lse, outn, lsen)
    torch.testing.assert_close(out2, outn, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse2, lsen, rtol=1e-6, atol=1e-6)


def test_identical_states_keep_the_output_and_add_log_two() -> None:
    o, lse = states(1)
    o2, lse2 = o.expand(2, *o.shape[1:]).contiguous(), lse.expand(2, *lse.shape[1:]).contiguous()
    out, lse_out = outputs(o2)
    merge_attention_states_n(o2, lse2, out, lse_out)
    torch.testing.assert_close(out, o[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse_out, lse[0] + math.log(2), rtol=1e-6, atol=1e-6)


def test_an_empty_side_returns_the_other_side() -> None:
    o, lse = states(2)
    lse[1] = NEG_INF
    o[1] = 1e30  # an empty side's output must be ignored, whatever it holds
    out, lse_out = outputs(o)
    merge_attention_states(o[0], lse[0], o[1], lse[1], out, lse_out)
    torch.testing.assert_close(out, o[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse_out, lse[0], rtol=0, atol=0)


def test_all_sides_empty_give_zero_and_minus_infinity() -> None:
    o, lse = states(3)
    lse.fill_(NEG_INF)
    out, lse_out = outputs(o)
    merge_attention_states_n(o, lse, out, lse_out)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse_out).all()


def test_rows_empty_in_some_splits_only() -> None:
    o, lse = states(4)
    lse[0, 1] = NEG_INF
    lse[:, 2, 0] = NEG_INF  # one row empty everywhere
    out, lse_out = outputs(o)
    merge_attention_states_n(o, lse, out, lse_out)
    check(out, lse_out, o, lse)
    assert torch.equal(out[2, 0], torch.zeros_like(out[2, 0]))


def test_lse_far_apart_do_not_overflow() -> None:
    o, lse = states(2)
    lse[0] = 1000.0
    lse[1] = -1000.0
    out, lse_out = outputs(o)
    merge_attention_states_n(o, lse, out, lse_out)
    torch.testing.assert_close(out, o[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse_out, lse[0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("head_dim", [64, 80, 128])
def test_head_dims_including_one_not_a_power_of_two(head_dim: int) -> None:
    o, lse = states(3, head_dim=head_dim)
    out, lse_out = outputs(o)
    merge_attention_states_n(o, lse, out, lse_out)
    check(out, lse_out, o, lse)


def test_half_precision_output_and_no_lse_output() -> None:
    o, lse = states(3)
    out = torch.empty(o.shape[1:], dtype=torch.float16, device=DEVICE)
    merge_attention_states_n(o, lse, out)
    ref, _ = merge_states_ref(o, lse)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)


def test_merging_two_halves_of_the_keys_is_attention_over_all_of_them() -> None:
    gen = torch.Generator().manual_seed(4)
    blocks, slots, kv_heads, head_dim, q_heads = 6, 4, 2, 64, 4
    k = torch.randn((blocks, slots, kv_heads, head_dim), generator=gen).to(DEVICE)
    v = torch.randn((blocks, slots, kv_heads, head_dim), generator=gen).to(DEVICE)
    q = torch.randn((1, q_heads, head_dim), generator=gen).to(DEVICE)
    whole = RequestKV([1, 2, 3], 12, [4, 5], 7)
    first = RequestKV([1, 2, 3], 12, [], 0)
    second = RequestKV([], 0, [4, 5], 7)
    out_all, lse_all = attention_ref32(q, k, v, [whole])
    o1, l1 = attention_ref32(q, k, v, [first])
    o2, l2 = attention_ref32(q, k, v, [second])
    out, lse_out = outputs(o1[None])
    merge_attention_states(o1, l1, o2, l2, out, lse_out)
    torch.testing.assert_close(out, out_all, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_out, lse_all, rtol=1e-5, atol=1e-5)


def test_rejects_bad_arguments() -> None:
    o, lse = states(2)
    out, lse_out = outputs(o)
    with pytest.raises(ValueError, match="lse"):
        merge_attention_states_n(o, lse.half(), out, lse_out)
    with pytest.raises(ValueError, match="shape"):
        merge_attention_states_n(o, lse[:, :3], out, lse_out)
    with pytest.raises(ValueError, match="out"):
        merge_attention_states_n(o, lse, out[:2], lse_out)
    with pytest.raises(ValueError, match="splits"):
        merge_attention_states_n(o[:0], lse[:0], out, lse_out)


def test_ops_are_registered_and_declare_what_they_write() -> None:
    for name in ("merge_attention_states", "merge_attention_states_n"):
        schema = getattr(torch.ops.kvkernels, name).default._schema
        written = [a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write]
        assert written == ["out", "lse_out"], name


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_full_size(dtype: torch.dtype) -> None:
    o, lse = states(16, tokens=128, heads=32, head_dim=128)
    out, lse_out = outputs(o, dtype)
    merge_attention_states_n(o, lse, out, lse_out)
    check(out, lse_out, o, lse)
