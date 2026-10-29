"""The compiler-stage dump tool: argument types, the alignment rule, and the files it writes.

Compilation needs no GPU, so these run in the interpreter tier too (the tool turns
the interpreter's kernel object back into a compilable one).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

import torch  # noqa: E402

from scripts.dump_kernel_ir import (  # noqa: E402
    Example,
    divisible_by_16,
    dump,
    jit_function,
    signature_and_attrs,
    triton_type,
)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):  # type: ignore[name-defined]
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add_example(n: int, num_warps: int = 8) -> Example:
    x = torch.empty(4096, dtype=torch.float16)
    return Example(
        name=f"add_n{n}",
        kernel=add_kernel,
        args={"x_ptr": x, "y_ptr": x.clone(), "out_ptr": x.clone(), "n": n},
        constexprs={"BLOCK": 1024},
        num_warps=num_warps,
        num_stages=2,
    )


def test_triton_types() -> None:
    assert triton_type(torch.empty(2, dtype=torch.float16)) == "*fp16"
    assert triton_type(torch.empty(2, dtype=torch.bfloat16)) == "*bf16"
    assert triton_type(torch.empty(2, dtype=torch.int32)) == "*i32"
    assert triton_type(7) == "i32"
    assert triton_type(2**31) == "i64"
    assert triton_type(True) == "i1"
    assert triton_type(0.5) == "fp32"


def test_alignment_rule() -> None:
    t = torch.empty(64, dtype=torch.float16)  # PyTorch aligns allocations to at least 16 bytes
    assert divisible_by_16(t)
    assert not divisible_by_16(t[1:])  # 2 bytes past an aligned start
    assert divisible_by_16(4096)
    assert not divisible_by_16(1000)
    assert not divisible_by_16(True)


def test_signature_marks_the_aligned_arguments() -> None:
    example = add_example(1000)
    signature, attrs = signature_and_attrs(
        jit_function(add_kernel), example.args, example.constexprs
    )
    assert signature == {
        "x_ptr": "*fp16",
        "y_ptr": "*fp16",
        "out_ptr": "*fp16",
        "n": "i32",
        "BLOCK": "constexpr",
    }
    assert set(attrs) == {(0,), (1,), (2,)}  # the three pointers; n = 1000 is not a multiple of 16


def test_rejects_missing_and_unknown_arguments() -> None:
    fn = jit_function(add_kernel)
    with pytest.raises(ValueError, match="no example value"):
        signature_and_attrs(fn, {"x_ptr": torch.empty(1)}, {"BLOCK": 8})
    args = dict(add_example(16).args, extra=1)
    with pytest.raises(ValueError, match="not parameters"):
        signature_and_attrs(fn, args, {"BLOCK": 8})


def test_dump_writes_every_stage(tmp_path: Path) -> None:
    out = dump(add_example(4096), "sm_80", tmp_path)
    for name in ("ttir", "ttgir", "llir", "ptx", "cubin", "sass"):
        assert (out / f"kernel.{name}").stat().st_size > 0, name
    assert ".target sm_80" in (out / "kernel.ptx").read_text()
    assert "LDG" in (out / "kernel.sass").read_text()
    assert "REG:" in (out / "resources.txt").read_text()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["num_warps"] == 8 and meta["num_stages"] == 2
    assert meta["divisible_by_16"] == ["x_ptr", "y_ptr", "out_ptr", "n"]


def test_alignment_decides_the_load_width(tmp_path: Path) -> None:
    # 1,024 fp16 values over 8 warps: 4 per thread, so aligned loads are 8 bytes wide.
    aligned = (dump(add_example(4096), "sm_90", tmp_path) / "kernel.sass").read_text()
    unaligned = (dump(add_example(1000), "sm_90", tmp_path) / "kernel.sass").read_text()
    assert "LDG.E.64" in aligned
    assert "LDG.E.U16" not in aligned
    assert "LDG.E.U16" in unaligned


@pytest.mark.gpu
def test_matches_the_launched_kernel() -> None:
    n = 4096
    x = torch.randn(n, dtype=torch.float16, device="cuda")
    y = torch.randn(n, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)
    launched = add_kernel[(n // 1024,)](x, y, out, n, BLOCK=1024, num_warps=8, num_stages=2)
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if arch not in ("sm_75", "sm_80", "sm_89", "sm_90"):
        pytest.skip(f"{arch} is not a target architecture")
    from scripts.dump_kernel_ir import ARCHS, compile_example

    example = Example(
        name="add",
        kernel=add_kernel,
        args={"x_ptr": x, "y_ptr": y, "out_ptr": out, "n": n},
        constexprs={"BLOCK": 1024},
        num_warps=8,
        num_stages=2,
    )
    assert compile_example(example, ARCHS[arch]).asm["ptx"] == launched.asm["ptx"]
