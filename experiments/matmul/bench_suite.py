"""Matmul against cuBLAS (torch.matmul).

Run as ``python -m bench.harness --suite experiments-matmul``.

Cases, all fp16 unless noted:

    square       M = N = K ∈ {512, …, 8,192}; 4,096 also in bf16
    projection   the Llama 3.1 8B projections (q/k/v, o, gate/up, down) with 16, 128 and
                 2,048 tokens
    fp16acc      square 4,096 and 8,192 with the kernel accumulating in fp16, the
                 measurement that shows whether a GPU halves fp32-accumulate throughput

Implementations in each square and projection case, checked against torch_matmul:
torch_matmul, triton (autotuned), and on two shapes (square 4,096 and the down
projection with 16 tokens) triton_cfg<i> for every autotuning configuration on its
own. An fp16acc case holds only triton_fp16acc: it is meant to be less accurate
than cuBLAS, so the harness cannot check it against torch_matmul; its correctness
is tested against an fp16-accumulating reference in test_kernel.py.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Any

from bench.harness import Case, Tensors
from bench.schema import Params

SQUARE = (512, 1024, 2048, 4096, 8192)
PROJECTIONS = {
    "qkv": (6144, 4096),
    "o": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
}
TOKENS = (16, 128, 2048)
FP16ACC_SIZES = (4096, 8192)
# The two shapes on which every configuration is timed on its own.
SWEEP_SHAPES = {(4096, 4096, 4096), (16, 4096, 14336)}


def kernel() -> Any:
    """experiments/matmul/kernel.py, loaded when a case runs."""
    return importlib.import_module("experiments.matmul.kernel")


def flops(p: Params) -> int:
    return 2 * int(p["M"]) * int(p["N"]) * int(p["K"])


def make_inputs(p: Params, device: str = "cuda") -> Tensors:
    import torch

    dtype = getattr(torch, str(p["dtype"]))
    gen = torch.Generator(device=device).manual_seed(0)
    m, n, k = int(p["M"]), int(p["N"]), int(p["K"])
    a = torch.randn((m, k), generator=gen, device=device).to(dtype)
    b = torch.randn((k, n), generator=gen, device=device).to(dtype)
    return {"a": a, "b": b, "out": torch.empty((m, n), dtype=dtype, device=device)}


def small_inputs(p: Params) -> Tensors:
    return make_inputs({**p, "M": 64, "N": 64, "K": 64})


def torch_matmul(t: Tensors) -> None:
    import torch

    torch.matmul(t["a"], t["b"], out=t["out"])


def triton(t: Tensors) -> None:
    kernel().matmul(t["a"], t["b"], t["out"])


def triton_fp16acc(t: Tensors) -> None:
    kernel().matmul(t["a"], t["b"], t["out"], fp16_accumulate=True)


def triton_with(config: dict[str, int]) -> Callable[[Tensors], None]:
    def run(t: Tensors) -> None:
        kernel().matmul(t["a"], t["b"], t["out"], config=config)

    return run


def reference_fp32(t: Tensors) -> Tensors:
    from experiments.matmul.reference import matmul_ref32

    return {"out": matmul_ref32(t["a"], t["b"])}


def grid() -> Iterator[Params]:
    for size in SQUARE:
        yield {"case": "square", "M": size, "N": size, "K": size, "dtype": "float16", "acc": "fp32"}
    yield {"case": "square", "M": 4096, "N": 4096, "K": 4096, "dtype": "bfloat16", "acc": "fp32"}
    for name, (n, k) in PROJECTIONS.items():
        for tokens in TOKENS:
            yield {
                "case": f"projection_{name}",
                "M": tokens,
                "N": n,
                "K": k,
                "dtype": "float16",
                "acc": "fp32",
            }
    for size in FP16ACC_SIZES:
        yield {
            "case": "fp16acc",
            "M": size,
            "N": size,
            "K": size,
            "dtype": "float16",
            "acc": "fp16",
        }


def make_case(p: Params, configs: list[dict[str, int]] | None = None) -> Case:
    impls: dict[str, Callable[[Tensors], None]]
    if p["acc"] == "fp16":
        impls = {"triton_fp16acc": triton_fp16acc}
    else:
        impls = {"torch_matmul": torch_matmul, "triton": triton}
        shape = (int(p["M"]), int(p["N"]), int(p["K"]))
        if configs is not None and shape in SWEEP_SHAPES and p["dtype"] == "float16":
            for i, config in enumerate(configs):
                impls[f"triton_cfg{i}"] = triton_with(config)
    return Case(
        params=p,
        setup=lambda: make_inputs(p),
        impls=impls,
        outputs=("out",),
        check="tolerance",
        flops=flops,
        reference_fp32=reference_fp32,
        host_setup=lambda: small_inputs(p),
    )


def cases() -> Iterator[Case]:
    configs: list[dict[str, Any]] = list(kernel().CONFIGS)
    for p in grid():
        yield make_case(p, configs)
