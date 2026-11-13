"""kvkernels' cascade decode attention as a vLLM attention backend.

vLLM calls register() in every process it starts, through the
vllm.general_plugins entry point. Select the backend with
`vllm serve <model> --attention-backend CUSTOM`.
"""

from __future__ import annotations

BACKEND = "kvkernels_vllm.backend.KvkernelsAttentionBackend"


def register() -> None:
    """Point vLLM's CUSTOM attention backend at kvkernels. Safe to call more than once."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND)
