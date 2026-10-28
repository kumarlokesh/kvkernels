"""Run vLLM's Triton address-list copy kernel outside vLLM.

The copied file imports two vLLM modules. ``vllm.triton_utils`` only re-exports
``triton`` and ``triton.language``. ``vllm._custom_ops`` holds vLLM's C++ copy,
which the file's own launcher calls for batches under 16 jobs; this adapter
never calls that launcher. Stand-ins for both are registered before the import.

The kernel copies a list of jobs, each a (source address, destination address,
size in bytes) triple. A gather becomes one job per id:

    job i:  plane + block_ids[i] · block stride  ──▶  out + i · block bytes,  block bytes

vLLM launches at most 12 programs, a number tuned for copies over PCIe. For a
copy inside device memory that cap would measure the cap, so this adapter
launches one program per job. The chunk size follows vLLM's caller: the block
size rounded up to a power of two, at most 8 KiB.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import torch
import triton
import triton.language as tl

# vLLM's caller uses the kernel only when every size is a multiple of 8 bytes.
WORD_BYTES = 8
MAX_CHUNK_BYTES = 8192


def _install_stand_ins() -> None:
    if "vllm" in sys.modules:
        return

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("vLLM's C++ copy is not available outside vLLM")

    vllm = types.ModuleType("vllm")
    custom_ops = types.ModuleType("vllm._custom_ops")
    triton_utils = types.ModuleType("vllm.triton_utils")
    custom_ops.swap_blocks_batch = unavailable  # type: ignore[attr-defined]
    triton_utils.triton = triton  # type: ignore[attr-defined]
    triton_utils.tl = tl  # type: ignore[attr-defined]
    vllm._custom_ops = custom_ops  # type: ignore[attr-defined]
    vllm.triton_utils = triton_utils  # type: ignore[attr-defined]
    sys.modules.update(
        {"vllm": vllm, "vllm._custom_ops": custom_ops, "vllm.triton_utils": triton_utils}
    )


def supports(block_bytes: int, stride_bytes: int) -> bool:
    """True when vLLM's caller would use this kernel for blocks of this size."""
    return block_bytes % WORD_BYTES == 0 and stride_bytes % WORD_BYTES == 0


def gather_jobs(
    plane: torch.Tensor, block_ids: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source addresses, destination addresses and sizes for a gather, on the device."""
    block_bytes = out[0].numel() * out.element_size() if out.shape[0] else 0
    stride_bytes = plane.stride(0) * plane.element_size()
    ids = block_ids.long()
    rows = torch.arange(ids.numel(), device=ids.device, dtype=torch.int64)
    src = plane.data_ptr() + ids * stride_bytes
    dst = out.data_ptr() + rows * block_bytes
    sizes = torch.full_like(ids, block_bytes)
    return src, dst, sizes


def chunk_bytes(block_bytes: int) -> int:
    return int(min(triton.next_power_of_2(block_bytes), MAX_CHUNK_BYTES))


def copy_jobs(src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor, chunk: int) -> None:
    """Launch the copied kernel with one program per job."""
    _install_stand_ins()
    from bench.baselines.vllm import swap_blocks_triton

    n = src.numel()
    swap_blocks_triton._swap_blocks_kernel[(n,)](src, dst, sizes, n, BYTES_PER_CHUNK=chunk)
