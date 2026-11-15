"""The driver call vLLM v0.33.0 makes to offload KV blocks: one cuMemcpyBatchAsync for all jobs.

vLLM's C++ op swap_blocks_batch (csrc/cache_kernels.cu at v0.33.0, commit
4ba787ec) takes three int64 CPU tensors (source addresses, destination
addresses, sizes), looks up cuMemcpyBatchAsync with cuGetProcAddress at the
CUDA 12.8 ABI, and passes the tensors' memory to it unconverted, with one
attribute set (source access order of the stream for GPU to CPU) for every job,
on the caller's stream. This module makes the same call through ctypes, so the
per-call CPU cost stays that of vLLM's op; NVIDIA's cuda-bindings package
converts every job list to C on each call, which for tens of thousands of jobs
would put milliseconds of host work inside the measured time.

    jobs (built once)          int64 arrays: dst[i], src[i], size[i]
         │
         ▼ cuMemcpyBatchAsync(dst, src, size, n, &attrs, &index 0, 1, &fail, stream)
    copy engine                each job a separate copy, in any order
"""

from __future__ import annotations

import ctypes
import functools
from typing import Any

CUDA_12_8 = 12080
SRC_ACCESS_ORDER_STREAM = 1  # CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
SRC_ACCESS_ORDER_ANY = 3  # CU_MEMCPY_SRC_ACCESS_ORDER_ANY


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemcpyAttributes(ctypes.Structure):
    """CUmemcpyAttributes of CUDA 12.8: access order, source and destination hints, flags."""

    _fields_ = [
        ("src_access_order", ctypes.c_int),
        ("src_loc_hint", _MemLocation),
        ("dst_loc_hint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,  # CUresult
    ctypes.c_void_p,  # CUdeviceptr* dsts
    ctypes.c_void_p,  # CUdeviceptr* srcs
    ctypes.c_void_p,  # size_t* sizes
    ctypes.c_size_t,  # count
    ctypes.POINTER(_MemcpyAttributes),
    ctypes.POINTER(ctypes.c_size_t),  # attrsIdxs
    ctypes.c_size_t,  # numAttrs
    ctypes.POINTER(ctypes.c_size_t),  # failIdx
    ctypes.c_void_p,  # CUstream
)


@functools.cache
def _batch_fn() -> Any:
    """cuMemcpyBatchAsync at the CUDA 12.8 ABI, or None when the driver lacks it."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    get_proc = libcuda.cuGetProcAddress_v2
    get_proc.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
    ]
    get_proc.restype = ctypes.c_int
    fn = ctypes.c_void_p()
    status = ctypes.c_int()
    if get_proc(b"cuMemcpyBatchAsync", ctypes.byref(fn), CUDA_12_8, 0, ctypes.byref(status)):
        return None
    if not fn.value:
        return None
    return _BATCH_FN(fn.value)


def available() -> bool:
    return _batch_fn() is not None


def copy_batch(dst: Any, src: Any, sizes: Any, stream: int, *, src_access_any: bool) -> None:
    """Submit len(sizes) copies in one call. dst, src, sizes: contiguous int64 CPU tensors."""
    n = int(sizes.numel())
    if n == 0:
        return
    attrs = _MemcpyAttributes()
    attrs.src_access_order = SRC_ACCESS_ORDER_ANY if src_access_any else SRC_ACCESS_ORDER_STREAM
    index = ctypes.c_size_t(0)
    fail = ctypes.c_size_t(0)
    result = _batch_fn()(
        dst.data_ptr(),
        src.data_ptr(),
        sizes.data_ptr(),
        n,
        ctypes.byref(attrs),
        ctypes.byref(index),
        1,
        ctypes.byref(fail),
        stream,
    )
    if result != 0:
        raise RuntimeError(f"cuMemcpyBatchAsync failed at job {fail.value} with error {result}")
