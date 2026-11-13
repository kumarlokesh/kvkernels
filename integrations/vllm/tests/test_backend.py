"""The vLLM backend: registration, what it refuses, its metadata split, and a GPU run.

Skipped where vLLM 0.33.0 is not installed. The GPU test serves a small model
with every decode call checked against vLLM's Triton kernel and fp32.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch

from kvkernels_vllm import BACKEND, register


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
