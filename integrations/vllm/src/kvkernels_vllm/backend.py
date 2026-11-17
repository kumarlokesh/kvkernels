"""vLLM attention backend: single-token decodes through kvkernels, the rest through vLLM's Triton.

One step, with decodes ordered first (the builder asks vLLM for that):

    query tokens   │ r0 │ r1 │ r2 │ r3 │ r4 r4 r4 r4 r4 │ r5 r5 r5 │
                   └─ decodes, 1 token ┘ └── prefill and extend ──┘
                             │                       │
                             ▼                       ▼
                    kvkernels cascade,       vLLM's Triton kernel on the
                    planned in build()       prefill rows' slice of the metadata

KV writes stay with vLLM's Triton backend (it writes before forward is called).
The builder plans on the GPU from the decode rows of vLLM's block table, with no
host sync, so full CUDA graphs of decode batches stay valid: vLLM builds
metadata eagerly before each replay, and the replay reads the plan's buffers at
the addresses it recorded.

Set KVKERNELS_VLLM_CHECK=1, with --enforce-eager, to also run vLLM's Triton
kernel and an fp32 reference on every decode call and fail on a mismatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import torch
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from kvkernels.ops.cascade import CascadePlan
from kvkernels_vllm.decode import HEAD_SIZES, check_decode, decode, new_plan, unsupported

CHECK_ENV = "KVKERNELS_VLLM_CHECK"


@dataclass
class Calls:
    """Python-side calls in this process: decode batches run, and those checked.

    Under a full CUDA graph Python runs forward only while capturing, so a
    replay adds nothing here.
    """

    decode: int = 0
    checked: int = 0


calls = Calls()


def _checking() -> bool:
    return os.environ.get(CHECK_ENV) == "1"


@dataclass
class KvkernelsMetadata(TritonAttentionMetadata):
    num_decodes: int = 0
    num_decode_tokens: int = 0
    plan: CascadePlan | None = None
    prefill: TritonAttentionMetadata | None = None  # the prefill rows, for vLLM's kernel
    decode_check: TritonAttentionMetadata | None = None  # the decode rows, in check mode


def _rows(
    meta: TritonAttentionMetadata, first_req: int, last_req: int, first_tok: int, last_tok: int
) -> TritonAttentionMetadata:
    """vLLM's metadata for requests first_req … last_req - 1 alone."""
    causal = meta.causal
    if isinstance(causal, torch.Tensor):
        causal = causal[first_req:last_req]
    return replace(
        meta,
        num_actual_tokens=last_tok - first_tok,
        query_start_loc=meta.query_start_loc[first_req : last_req + 1] - first_tok,
        seq_lens=meta.seq_lens[first_req:last_req],
        block_table=meta.block_table[first_req:last_req],
        slot_mapping=meta.slot_mapping[first_tok:last_tok],
        causal=causal,
    )


class KvkernelsMetadataBuilder(TritonAttentionMetadataBuilder):
    # Full graphs for batches of single-token decodes; mixed batches run piecewise.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_draft_decode_metadata_update = False

    def __init__(
        self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: torch.device
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.speculative_config is not None:
            raise ValueError("the kvkernels backend does not support speculative decoding")
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.plan = new_plan(
            max_requests=vllm_config.scheduler_config.max_num_seqs,
            num_q_heads=self.num_heads_q,
            num_kv_heads=self.num_heads_kv,
            head_dim=self.headdim,
            block_size=self.block_size,
            device=device,
            batch_invariant=bool(envs.VLLM_BATCH_INVARIANT),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KvkernelsMetadata:
        meta = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=1
        )
        num_reqs = num_decodes + num_prefills
        if num_decodes:
            self.plan.plan_from_block_table(
                meta.block_table[:num_decodes], meta.seq_lens[:num_decodes]
            )
        prefill = (
            _rows(meta, num_decodes, num_reqs, num_decode_tokens, meta.num_actual_tokens)
            if num_prefills
            else None
        )
        decode_check = (
            _rows(meta, 0, num_decodes, 0, num_decode_tokens)
            if num_decodes and _checking()
            else None
        )
        return KvkernelsMetadata(
            **{f.name: getattr(meta, f.name) for f in fields(meta)},
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            plan=self.plan if num_decodes else None,
            prefill=prefill,
            decode_check=decode_check,
        )


def _kv_scales(layer: torch.nn.Module) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The layer's K and V scales for an fp8 cache: one-element fp32 tensors."""
    scales = (getattr(layer, "_k_scale", None), getattr(layer, "_v_scale", None))
    k_scale, v_scale = (t if isinstance(t, torch.Tensor) else None for t in scales)
    return k_scale, v_scale


class KvkernelsAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        reasons = unsupported(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            kv_cache_dtype=self.kv_cache_dtype,
            sliding_window=self.sliding_window != (-1, -1),
            alibi=self.alibi_slopes is not None,
            soft_cap=bool(self.logits_soft_cap),
            sinks=self.sinks is not None,
            decoder=self.attn_type == AttentionType.DECODER,
        )
        if reasons:
            raise ValueError("the kvkernels backend does not support: " + "; ".join(reasons))
        # With an fp8 cache vLLM would quantize the query to fp8 for a backend that
        # accepts it; the cascade takes fp16 or bf16 queries and scales the fp8 KV.
        self.supports_quant_query_input = False

    def fused_output_quant_supported(self, quant_key: Any) -> bool:
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KvkernelsMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("the kvkernels backend does not quantize its output")
        n = attn_metadata.num_decode_tokens
        if n:
            calls.decode += 1
            assert attn_metadata.plan is not None
            k_scale, v_scale = _kv_scales(layer)
            q = query[:n].view(n, self.num_heads, self.head_size)
            out = output[:n].view(n, self.num_heads, self.head_size)
            check = attn_metadata.decode_check
            lse = (
                torch.empty((n, self.num_heads), dtype=torch.float32, device=q.device)
                if check
                else None
            )
            decode(
                q,
                kv_cache,
                attn_metadata.plan,
                out,
                head_size=self.head_size,
                scale=self.scale,
                lse_out=lse,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            if check is not None:
                assert lse is not None
                engine_out = torch.empty_like(out)
                super().forward(layer, q, key[:n], value[:n], kv_cache, check, engine_out)
                check_decode(
                    q,
                    kv_cache,
                    check.block_table,
                    check.seq_lens,
                    out,
                    lse,
                    engine_out,
                    head_size=self.head_size,
                    scale=self.scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                calls.checked += 1
        if attn_metadata.prefill is not None:
            super().forward(
                layer, query[n:], key[n:], value[n:], kv_cache, attn_metadata.prefill, output[n:]
            )
        return output


class KvkernelsAttentionBackend(TritonAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"  # vLLM looks the backend up by this name in its enum

    @staticmethod
    def get_impl_cls() -> type[KvkernelsAttentionImpl]:
        return KvkernelsAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[KvkernelsMetadataBuilder]:
        return KvkernelsMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in HEAD_SIZES

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return bool(attn_type == AttentionType.DECODER)

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True
