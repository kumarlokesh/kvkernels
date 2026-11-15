"""The cascade over an fp8 (e4m3) KV cache against the same values in fp16.

Cases are the cascade suite's (one group, Llama 3.1 8B, vLLM's default layout)
at batch 1, 16, 64 and 128, prefixes of 2,048 and 32,768 tokens and suffixes of
128 and 512. K and V are drawn as in the cascade suite, then quantized to e4m3
with one power-of-two scale each, and the fp16 cache holds the dequantized
values, which are exact in fp16: every implementation attends over the same
numbers, so the outputs agree within the usual tolerance and the times compare
the bytes read.

    flashinfer_decode    FlashInfer batch decode over the fp16 cache (the reference)
    kvkernels            the cascade over the fp16 cache, groups named on the host
    kvkernels_fp8        the same plan over the fp8 cache with its two scales

bytes_moved is the fp16 cascade's traffic for every implementation, so the fp8
cascade's GB/s reads as the fp16 traffic it replaces per second.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bench.harness import Case, Tensors
from bench.schema import Params
from bench.suites import cascade as c

BATCHES = (1, 16, 64, 128)
PREFIXES = (2048, 32768)
SUFFIXES = (128, 512)
E4M3_MAX = 448.0


def grid() -> Iterator[Params]:
    for batch in BATCHES:
        for prefix in PREFIXES:
            for suffix in SUFFIXES:
                yield c._params(batch=batch, prefix=prefix, suffix=suffix)


def make_inputs(p: Params, *, with_flashinfer: bool, device: str = "cuda") -> Tensors:
    import torch

    t = c.make_inputs(p, device=device)
    for name in ("k", "v"):
        view = t[name]
        amax = float(view.float().abs().max())
        scale = 2.0 ** round(torch.log2(torch.tensor(amax / E4M3_MAX)).item())
        fp8 = (view.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        view.copy_((fp8.float() * scale).to(view.dtype))
        t[f"{name}8"] = fp8
        t[f"{name}_scale"] = torch.tensor([scale], dtype=torch.float32, device=device)
    if with_flashinfer:
        t["k_nhd"] = t["k"].contiguous()
        t["v_nhd"] = t["v"].contiguous()
    return t


class KvkernelsFp8(c.Kvkernels):
    def run(self, t: Tensors) -> None:
        from kvkernels.ops.cascade import cascade_decode_attention

        assert self.cascade is not None
        cascade_decode_attention(
            t["q"],
            t["k8"],
            t["v8"],
            self.cascade,
            t["out"],
            k_scale=t["k_scale"],
            v_scale=t["v_scale"],
        )


def make_case(p: Params, *, with_flashinfer: bool, device: str = "cuda") -> Case:
    objects: dict[str, Any] = {}
    if with_flashinfer:
        objects["flashinfer_decode"] = c.FlashInferDecode(tensor_cores=False)
    else:
        objects["torch_per_request"] = c.torch_per_request
    objects["kvkernels"] = c.Kvkernels()
    objects["kvkernels_fp8"] = KvkernelsFp8()
    impls, plans = {}, {}
    for name, obj in objects.items():
        if hasattr(obj, "run"):
            impls[name] = obj.run
            plans[name] = obj.plan
        else:
            impls[name] = obj
    return Case(
        params=p,
        setup=lambda: make_inputs(p, with_flashinfer=with_flashinfer, device=device),
        impls=impls,
        outputs=("out",),
        check="tolerance",
        bytes_moved=c.bytes_moved,
        flops=c.flops,
        reference_fp32=c.reference_fp32,
        plans=plans,
    )


def cases() -> Iterator[Case]:
    with_flashinfer = c.flashinfer_runs()
    for p in grid():
        yield make_case(p, with_flashinfer=with_flashinfer)
