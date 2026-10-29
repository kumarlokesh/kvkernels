"""Fused attention forward against PyTorch's SDPA backends.

Run as ``python -m bench.harness --suite experiments-fused_attention_fwd``.

Cases: batch 4, 32 query heads, 8 or 32 KV heads, sequence ∈ {512, 1,000, 2,048,
4,096, 8,192}, head_dim 64 or 128, causal and not, fp16; one bf16 case.

PyTorch does not report which backend scaled_dot_product_attention picked, so
each backend is forced in turn with torch.nn.attention.sdpa_kernel and timed as
its own implementation: sdpa_flash, sdpa_efficient, sdpa_cudnn. A backend that
rejects a case (a trial call on small inputs of the same dtype, head_dim, GQA and
causal setting raises) is left out of that case, and the suite prints it. The
first backend that accepts the case is the reference the others and the kernel
are checked against. Only the output is checked here; the LSE, which SDPA does
not return, is checked in test_kernel.py.

On two shapes (sequence 4,096, head_dim 128, 8 KV heads, causal and not) every
autotuning configuration is also timed on its own, as triton_cfg<i>.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Any

from bench.harness import Case, Tensors
from bench.schema import Params

BATCH = 4
Q_HEADS = 32
KV_HEADS = (8, 32)
SEQS = (512, 1000, 2048, 4096, 8192)
HEAD_DIMS = (64, 128)
BACKENDS = ("flash", "efficient", "cudnn")


def is_sweep_shape(p: Params) -> bool:
    return (
        p["seq"] == 4096 and p["head_dim"] == 128 and p["kv_heads"] == 8 and p["dtype"] == "float16"
    )


def kernel() -> Any:
    """experiments/fused_attention_fwd/kernel.py, loaded when a case runs."""
    return importlib.import_module("experiments.fused_attention_fwd.kernel")


def flops(p: Params) -> int:
    """QKᵀ and PV, 2 flops per multiply-add; halved when causal."""
    full = 4 * BATCH * Q_HEADS * int(p["seq"]) * int(p["seq"]) * int(p["head_dim"])
    return full // 2 if p["causal"] == "yes" else full


def grid() -> Iterator[Params]:
    for seq in SEQS:
        for kv_heads in KV_HEADS:
            for head_dim in HEAD_DIMS:
                for causal in ("no", "yes"):
                    yield {
                        "seq": seq,
                        "kv_heads": kv_heads,
                        "head_dim": head_dim,
                        "causal": causal,
                        "dtype": "float16",
                    }
    yield {"seq": 2048, "kv_heads": 8, "head_dim": 128, "causal": "yes", "dtype": "bfloat16"}


def make_inputs(p: Params, *, seq: int | None = None, device: str = "cuda") -> Tensors:
    import torch

    dtype = getattr(torch, str(p["dtype"]))
    s = int(p["seq"]) if seq is None else seq
    d = int(p["head_dim"])
    kv = int(p["kv_heads"])
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(size, generator=gen, device=device).to(dtype)

    q = randn(BATCH, Q_HEADS, s, d)
    return {
        "q": q,
        "k": randn(BATCH, kv, s, d),
        "v": randn(BATCH, kv, s, d),
        "out": torch.empty_like(q),
        "lse": torch.empty((BATCH, Q_HEADS, s), dtype=torch.float32, device=device),
    }


def _sdpa_backend(name: str) -> Any:
    from torch.nn.attention import SDPBackend

    return {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[name]


def sdpa(name: str, causal: bool) -> Callable[[Tensors], None]:
    def run(t: Tensors) -> None:
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel

        gqa = t["q"].shape[1] != t["k"].shape[1]
        with sdpa_kernel(_sdpa_backend(name)):
            t["out"] = F.scaled_dot_product_attention(
                t["q"], t["k"], t["v"], is_causal=causal, enable_gqa=gqa
            )

    return run


def accepts(name: str, p: Params) -> bool:
    """True when the backend runs this case's dtype, head_dim, GQA and causal setting."""
    run = sdpa(name, p["causal"] == "yes")
    try:
        run(make_inputs(p, seq=64))
    except RuntimeError:
        return False
    return True


def triton(causal: bool, config: dict[str, int] | None = None) -> Callable[[Tensors], None]:
    def run(t: Tensors) -> None:
        kernel().attention_fwd(
            t["q"], t["k"], t["v"], t["out"], t["lse"], causal=causal, config=config
        )

    return run


def reference_fp32(causal: bool) -> Callable[[Tensors], Tensors]:
    def run(t: Tensors) -> Tensors:
        from experiments.fused_attention_fwd.reference import attention_ref32

        out, _ = attention_ref32(t["q"], t["k"], t["v"], causal=causal)
        return {"out": out}

    return run


def make_case(p: Params, backends: list[str], configs: list[dict[str, Any]]) -> Case:
    causal = p["causal"] == "yes"
    impls: dict[str, Callable[[Tensors], None]] = {f"sdpa_{backends[0]}": sdpa(backends[0], causal)}
    impls["triton"] = triton(causal)
    for name in backends[1:]:
        impls[f"sdpa_{name}"] = sdpa(name, causal)
    if is_sweep_shape(p):
        for i, config in enumerate(configs):
            impls[f"triton_cfg{i}"] = triton(causal, config)
    return Case(
        params=p,
        setup=lambda: make_inputs(p),
        impls=impls,
        outputs=("out",),
        check="tolerance",
        flops=flops,
        reference_fp32=reference_fp32(causal),
        host_setup=lambda: make_inputs(p, seq=64),
    )


def cases() -> Iterator[Case]:
    configs: list[dict[str, Any]] = list(kernel().CONFIGS)
    for p in grid():
        backends = [name for name in BACKENDS if accepts(name, p)]
        left_out = sorted(set(BACKENDS) - set(backends))
        if left_out:
            print(f"experiments-fused_attention_fwd {p}: backends left out: {', '.join(left_out)}")
        if not backends:
            print(f"experiments-fused_attention_fwd {p}: no SDPA backend runs this case; skipped")
            continue
        yield make_case(p, backends, configs)
