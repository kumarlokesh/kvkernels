"""Stand-ins for the vLLM modules that the copied vLLM files import.

vLLM is not installed (it needs a different torch). Before importing a copied
file, install() registers small modules under vLLM's names in sys.modules, each
supplying only what the copied files use, with the behaviour vLLM 0.32.0 gives
them on a CUDA GPU:

    vllm.triton_utils                    triton and triton.language
    vllm._custom_ops                     swap_blocks_batch, which raises: vLLM's C++ copy
                                         is not available, and the adapters never call it
    vllm.envs                            VLLM_BATCH_INVARIANT = False, vLLM's default
    vllm.logger                          init_logger, a standard-library logger
    vllm.platforms                       current_platform: fp8_dtype() is float8_e4m3fn;
                                         is_device_capability_family(c) compares the GPU's
                                         major version with c // 10
    vllm.v1.kv_cache_interface           KVQuantMode, the same names and values
    vllm.v1.attention.ops.triton_attention_helpers
                                         the copied helpers file
"""

from __future__ import annotations

import enum
import logging
import sys
import types
from typing import Any

import torch
import triton
import triton.language as tl


class KVQuantMode(enum.IntEnum):
    """vllm/v1/kv_cache_interface.py at the pinned commit: names and values only."""

    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4
    NVFP4 = 5
    TURBOQUANT_K8V4 = 6
    TURBOQUANT_4BIT_NC = 7
    TURBOQUANT_K3V4_NC = 8
    TURBOQUANT_3BIT_NC = 9
    NVFP4_DS_MLA = 10
    ULTRAQUANT_4BIT = 11


class _CudaPlatform:
    @staticmethod
    def fp8_dtype() -> torch.dtype:
        return torch.float8_e4m3fn

    @staticmethod
    def is_device_capability_family(capability: int, device_id: int = 0) -> bool:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(device_id)
        return (major * 10 + minor) // 10 == capability // 10


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _unavailable(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("vLLM's C++ copy is not available outside vLLM")


def install() -> None:
    """Register the stand-ins once per process; later calls do nothing."""
    if "vllm" in sys.modules:
        return
    modules = {
        "vllm": _module("vllm"),
        "vllm._custom_ops": _module("vllm._custom_ops", swap_blocks_batch=_unavailable),
        "vllm.triton_utils": _module("vllm.triton_utils", triton=triton, tl=tl),
        "vllm.envs": _module("vllm.envs", VLLM_BATCH_INVARIANT=False),
        "vllm.logger": _module("vllm.logger", init_logger=logging.getLogger),
        "vllm.platforms": _module("vllm.platforms", current_platform=_CudaPlatform()),
        "vllm.v1": _module("vllm.v1"),
        "vllm.v1.kv_cache_interface": _module(
            "vllm.v1.kv_cache_interface", KVQuantMode=KVQuantMode
        ),
        "vllm.v1.attention": _module("vllm.v1.attention"),
        "vllm.v1.attention.ops": _module("vllm.v1.attention.ops"),
    }
    sys.modules.update(modules)
    for name, module in modules.items():  # attribute access: `import vllm.envs as envs`
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    # Imported after vllm.triton_utils exists, which it needs.
    from bench.baselines.vllm import triton_attention_helpers

    sys.modules["vllm.v1.attention.ops.triton_attention_helpers"] = triton_attention_helpers
    modules["vllm.v1.attention.ops"].triton_attention_helpers = triton_attention_helpers  # type: ignore[attr-defined]
