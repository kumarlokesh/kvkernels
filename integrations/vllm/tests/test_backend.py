"""The vLLM backend: registration, what it refuses, its metadata split, and a GPU run.

Skipped where vLLM 0.33.0 is not installed. The GPU test serves a small model
with every decode call checked against vLLM's Triton kernel and fp32.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch

from kvkernels_vllm import BACKEND, register
from kvkernels_vllm.decode import kv_views


def test_register_points_the_custom_backend_at_kvkernels() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from kvkernels_vllm.backend import KvkernelsAttentionBackend

    register()
    register()  # vLLM may load plugins more than once per process
    assert AttentionBackendEnum.CUSTOM.get_path() == BACKEND
    assert AttentionBackendEnum.CUSTOM.get_class() is KvkernelsAttentionBackend
    assert AttentionBackendEnum[KvkernelsAttentionBackend.get_name()] is AttentionBackendEnum.CUSTOM


def test_the_backend_declares_only_what_the_cascade_supports() -> None:
    from vllm.v1.attention.backend import AttentionCGSupport, AttentionType

    from kvkernels_vllm.backend import KvkernelsAttentionBackend as B

    assert B.supports_head_size(128) and B.supports_head_size(64)
    assert not B.supports_head_size(96)
    assert B.supports_attn_type(AttentionType.DECODER)
    assert not B.supports_attn_type(AttentionType.ENCODER_ONLY)
    assert not B.supports_sliding_window() and not B.supports_sink()
    assert not B.supports_dtype(torch.float32)
    assert not B.supports_kv_cache_dtype("fp8")
    assert B.get_builder_cls()._cudagraph_support == (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_the_impl_refuses_a_sliding_window() -> None:
    from kvkernels_vllm.backend import KvkernelsAttentionImpl

    args = (32, 128, 128**-0.5, 8, None)
    KvkernelsAttentionImpl(*args, None, "auto")
    with pytest.raises(ValueError, match="sliding window"):
        KvkernelsAttentionImpl(*args, 4096, "auto")


def test_rows_slices_the_prefill_requests_out_of_a_mixed_batch() -> None:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    from kvkernels_vllm.backend import _rows

    # Requests 0-1 decode one token each; request 2 prefills 3 tokens, request 3 prefills 2.
    meta = TritonAttentionMetadata(
        num_actual_tokens=7,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 1, 2, 5, 7], dtype=torch.int32),
        max_seq_len=40,
        seq_lens=torch.tensor([30, 40, 3, 12], dtype=torch.int32),
        block_table=torch.arange(16, dtype=torch.int32).view(4, 4),
        slot_mapping=torch.arange(7),
        seq_threshold_3D=8,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(1),
        softmax_segm_max=torch.empty(1),
        softmax_segm_expsum=torch.empty(1),
        causal=True,
    )
    prefill = _rows(meta, 2, 4, 2, 7)
    assert prefill.num_actual_tokens == 5
    assert prefill.query_start_loc.tolist() == [0, 3, 5]
    assert prefill.seq_lens.tolist() == [3, 12]
    assert prefill.block_table.tolist() == [[8, 9, 10, 11], [12, 13, 14, 15]]
    assert prefill.slot_mapping.tolist() == [2, 3, 4, 5, 6]
    decodes = _rows(meta, 0, 2, 0, 2)
    assert decodes.query_start_loc.tolist() == [0, 1, 2]
    assert decodes.seq_lens.tolist() == [30, 40]


def _builder_config(max_num_seqs: int) -> Any:
    """The parts of vLLM's config the Triton builder and ours read, with CUDA graphs off."""
    from vllm.config import CUDAGraphMode

    model = SimpleNamespace(
        get_num_attention_heads=lambda parallel: 8,
        get_num_kv_heads=lambda parallel: 2,
        get_head_size=lambda: 64,
        rswa_window=None,
    )
    return SimpleNamespace(
        model_config=model,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            static_forward_context={},
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=None,
    )


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, blocks: list[int], visible: int
) -> torch.Tensor:
    """fp64 attention of one query token over the first `visible` tokens of a block list."""
    keys = k[blocks].flatten(0, 1)[:visible].double()  # [tokens, kv_heads, d]
    values = v[blocks].flatten(0, 1)[:visible].double()
    group = q.shape[0] // keys.shape[1]
    keys, values = keys.repeat_interleave(group, 1), values.repeat_interleave(group, 1)
    scores = torch.einsum("hd,thd->ht", q.double(), keys) * q.shape[-1] ** -0.5
    return torch.einsum("ht,thd->hd", torch.softmax(scores, -1), values)


def _metadata_attention(
    self: Any,
    layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    meta: Any,
    output: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    """Causal attention for each request of vLLM metadata: its query tokens are the last
    of its seq_lens tokens, its keys in its block-table row."""
    k, v = kv_views(kv_cache, 64)
    starts = meta.query_start_loc.tolist()
    for r, n in enumerate(meta.seq_lens.tolist()):
        blocks = meta.block_table[r].tolist()
        ql = starts[r + 1] - starts[r]
        for i in range(ql):
            t = starts[r] + i
            output[t] = _reference(query[t], k, v, blocks, n - ql + i + 1).to(output.dtype)
    return output


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)
def test_a_mixed_batch_through_the_builder_and_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.kv_cache_interface import KVQuantMode

    from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache
    from kvkernels_vllm.backend import KvkernelsAttentionImpl, KvkernelsMetadataBuilder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    block = 16
    spec = CacheSpec(1, 16, block, 2, 64, torch.float16)
    (kv_cache,) = make_cache(LAYOUTS["vllm_LBNHC"], spec, device)[0]
    kv_cache.copy_(torch.randn(tuple(kv_cache.shape), generator=gen).to(torch.float16))

    # Requests 0-2 decode one token; 0 and 1 share KV blocks 1-4. Request 3 prefills 5
    # tokens at positions 35-39, after 35 cached ones.
    rows = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [7, 8], [9, 10, 11]]
    lens = [70, 66, 20, 40]
    query_lens = [1, 1, 1, 5]
    table = torch.zeros((4, 5), dtype=torch.int32)
    for r, row in enumerate(rows):
        table[r, : len(row)] = torch.tensor(row)
    starts = torch.tensor([0, 1, 2, 3, 8], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        num_reqs=4,
        num_actual_tokens=8,
        max_query_len=5,
        max_seq_len=70,
        block_table_tensor=table.to(device),
        slot_mapping=torch.zeros(8, dtype=torch.int64, device=device),
    )
    kv_spec = SimpleNamespace(block_size=block, kv_quant_mode=KVQuantMode.NONE)
    builder = KvkernelsMetadataBuilder(
        kv_spec, ["layer.0"], _builder_config(8), torch.device(device)
    )
    meta = builder.build(0, common)
    assert (meta.num_decodes, meta.num_decode_tokens) == (3, 3)
    assert meta.prefill is not None and meta.prefill.num_actual_tokens == 5
    assert int(builder.plan.counts[3]) == 1, "requests 0 and 1 share 64 tokens: one group"

    if device == "cpu":
        # Without a GPU, vLLM replaces triton with a placeholder and its prefill kernel
        # cannot run. Stand in for it with attention computed from the metadata the
        # backend hands it, read the way vLLM's kernel reads it.
        monkeypatch.setattr(TritonAttentionImpl, "forward", _metadata_attention)
    impl = KvkernelsAttentionImpl(8, 64, 64**-0.5, 2, None, None, "auto")
    one = torch.ones((), dtype=torch.float32, device=device)
    layer = SimpleNamespace(_k_scale=one, _v_scale=one, _q_scale=one)
    query = torch.randn((8, 8, 64), generator=gen).to(dtype=torch.float16, device=device)
    output = torch.zeros_like(query)
    impl.forward(layer, query, query, query, kv_cache, meta, output)  # type: ignore[arg-type]

    k, v = kv_views(kv_cache, 64)
    token = 0
    for r, (row, n, ql) in enumerate(zip(rows, lens, query_lens, strict=True)):
        for i in range(ql):
            want = _reference(query[token], k, v, row, n - ql + i + 1)
            torch.testing.assert_close(
                output[token].double(), want, atol=2e-2, rtol=2e-2, msg=f"request {r} token {i}"
            )
            token += 1


SMALL_MODEL = "Qwen/Qwen3-0.6B"  # 16 query heads, 8 KV heads, head_dim 128; not gated
SYSTEM = "You are a careful assistant. Answer in one short sentence. " * 12


@pytest.mark.gpu
def test_every_decode_call_matches_vllm_and_fp32_in_a_server_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("KVKERNELS_VLLM_CHECK", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    register()
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    llm = LLM(
        SMALL_MODEL,
        attention_backend="CUSTOM",
        enforce_eager=True,
        dtype=dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    prompts = [f"{SYSTEM}\nQuestion {i}: name a prime number above {10 * i}." for i in range(12)]
    prompts += ["Count from one to five.", "Name a colour."]  # no shared prefix
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
    assert all(o.outputs[0].token_ids for o in outputs)


@pytest.mark.gpu
def test_decode_batches_run_in_full_cuda_graphs() -> None:
    from vllm import LLM, SamplingParams

    if os.environ.get("KVKERNELS_VLLM_CHECK") == "1":
        pytest.skip("the check copies to the host, which a CUDA graph cannot capture")
    register()
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    llm = LLM(
        SMALL_MODEL,
        attention_backend="CUSTOM",
        dtype=dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    prompts = [f"{SYSTEM}\nQuestion {i}: name a city." for i in range(24)]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
    assert all(len(o.outputs[0].token_ids) > 0 for o in outputs)
