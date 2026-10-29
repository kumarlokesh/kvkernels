"""Compile a Triton kernel without a GPU and write every compiler stage.

Usage:
    python scripts/dump_kernel_ir.py MODULE:FUNCTION [--arch sm_80 ...] [--out dumps]

FUNCTION() returns a list of Example: a kernel, example argument values, its
constexprs, num_warps and num_stages. For each example and architecture this writes

    dumps/<example>/<arch>/
        kernel.ttir  kernel.ttgir  kernel.llir  kernel.ptx  kernel.cubin
        kernel.sass       SASS from the cubin (nvdisasm, shipped in Triton's wheel)
        resources.txt     registers, shared memory, spills (cuobjdump)
        metadata.json     arch, warps, stages, shared bytes, signature, attributes

The compiled code has to match what a launch produces, so the argument types and
alignment facts are derived from the example values the way the launcher derives
them: a tensor becomes a pointer to its dtype, an integer becomes i32 or i64, and
a tensor whose address, or an integer whose value, is a multiple of 16 is marked
divisible by 16 (Triton 3.8.0, BaseBackend.get_tensor_specialization and
get_int_specialization). Without that mark, loads that could be 16 bytes wide
compile to one element at a time.

No GPU is needed: triton.compile targets any architecture from the CPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARCHS = {"sm_75": 75, "sm_80": 80, "sm_89": 89, "sm_90": 90}

_POINTER_TYPES = {
    "float16": "*fp16",
    "bfloat16": "*bf16",
    "float32": "*fp32",
    "float64": "*fp64",
    "float8_e4m3fn": "*fp8e4nv",
    "float8_e5m2": "*fp8e5",
    "int8": "*i8",
    "int16": "*i16",
    "int32": "*i32",
    "int64": "*i64",
    "uint8": "*u8",
    "bool": "*i1",
}

DIVISIBILITY = 16


@dataclass(frozen=True)
class Example:
    name: str
    kernel: Any  # a @triton.jit function
    args: Mapping[str, Any]  # every non-constexpr parameter, by name
    constexprs: Mapping[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 3


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "data_ptr") and hasattr(value, "dtype")


def triton_type(value: Any) -> str:
    """The signature type the launcher gives an argument value."""
    if _is_tensor(value):
        name = str(value.dtype).removeprefix("torch.")
        if name not in _POINTER_TYPES:
            raise TypeError(f"no Triton pointer type for {value.dtype}")
        return _POINTER_TYPES[name]
    if isinstance(value, bool):
        return "i1"
    if isinstance(value, int):
        if -(2**31) <= value < 2**31:
            return "i32"
        if -(2**63) <= value < 2**63:
            return "i64"
        return "u64"
    if isinstance(value, float):
        return "fp32"
    raise TypeError(f"no Triton type for {type(value).__name__}")


def divisible_by_16(value: Any) -> bool:
    """The launcher's alignment rule for one argument."""
    if _is_tensor(value):
        return bool(value.data_ptr() % DIVISIBILITY == 0)
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value % DIVISIBILITY == 0
    return False


def jit_function(kernel: Any) -> Any:
    """A compilable JITFunction, also under TRITON_INTERPRET=1.

    The interpreter replaces @triton.jit functions with objects triton.compile
    cannot use; both keep the original Python function as `.fn`.
    """
    from triton.runtime.jit import JITFunction

    if isinstance(kernel, JITFunction):
        return kernel
    return JITFunction(kernel.fn)


def signature_and_attrs(
    fn: Any, args: Mapping[str, Any], constexprs: Mapping[str, Any]
) -> tuple[dict[str, str], dict[tuple[int, ...], list[list[Any]]]]:
    """Triton signature and attributes for a kernel's parameters, in order."""
    signature: dict[str, str] = {}
    attrs: dict[tuple[int, ...], list[list[Any]]] = {}
    for index, name in enumerate(fn.arg_names):
        if name in constexprs:
            signature[name] = "constexpr"
            continue
        if name not in args:
            raise ValueError(f"no example value for parameter {name!r}")
        value = args[name]
        signature[name] = triton_type(value)
        if divisible_by_16(value):
            attrs[(index,)] = [["tt.divisibility", DIVISIBILITY]]
    unknown = set(args) | set(constexprs)
    unknown -= set(fn.arg_names)
    if unknown:
        raise ValueError(f"not parameters of {fn.__name__}: {sorted(unknown)}")
    return signature, attrs


def compile_example(example: Example, arch: int) -> Any:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = jit_function(example.kernel)
    signature, attrs = signature_and_attrs(fn, example.args, example.constexprs)
    source = ASTSource(fn=fn, signature=signature, constexprs=dict(example.constexprs), attrs=attrs)
    return triton.compile(
        source,
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": example.num_warps, "num_stages": example.num_stages},
    )


def _nvidia_tool(name: str) -> Path | None:
    import triton

    path = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / name
    return path if path.exists() else (Path(found) if (found := shutil.which(name)) else None)


def _run_tool(name: str, *args: str) -> str:
    tool = _nvidia_tool(name)
    if tool is None:
        return f"{name} not found\n"
    done = subprocess.run([str(tool), *args], capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else f"{name} failed:\n{done.stderr}"


def dump(example: Example, arch_name: str, out_root: Path) -> Path:
    """Compile one example for one architecture and write its stages; return the directory."""
    if arch_name not in ARCHS:
        raise ValueError(f"unknown arch {arch_name}; choose from {sorted(ARCHS)}")
    compiled = compile_example(example, ARCHS[arch_name])
    out = out_root / example.name / arch_name
    out.mkdir(parents=True, exist_ok=True)
    asm = compiled.asm
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        (out / f"kernel.{stage}").write_text(asm[stage])
    cubin = out / "kernel.cubin"
    cubin.write_bytes(asm["cubin"])
    (out / "kernel.sass").write_text(_run_tool("nvdisasm", "-c", str(cubin)))
    (out / "resources.txt").write_text(_run_tool("cuobjdump", "--dump-resource-usage", str(cubin)))
    fn = jit_function(example.kernel)
    signature, attrs = signature_and_attrs(fn, example.args, example.constexprs)
    meta = compiled.metadata
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "kernel": fn.__name__,
                "arch": arch_name,
                "num_warps": meta.num_warps,
                "num_stages": meta.num_stages,
                "shared_bytes": meta.shared,
                "signature": signature,
                "constexprs": {k: str(v) for k, v in example.constexprs.items()},
                "divisible_by_16": [fn.arg_names[i[0]] for i in attrs],
            },
            indent=2,
        )
        + "\n"
    )
    return out


def load_examples(target: str) -> list[Example]:
    module_name, sep, function_name = target.partition(":")
    if not sep:
        raise ValueError(f"expected MODULE:FUNCTION, got {target!r}")
    root = str(Path.cwd())
    if root not in sys.path:
        sys.path.insert(0, root)
    examples = getattr(importlib.import_module(module_name), function_name)()
    return list(examples)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("target", help="MODULE:FUNCTION returning a list of Example")
    parser.add_argument("--arch", action="append", choices=sorted(ARCHS), help="default: all")
    parser.add_argument("--out", type=Path, default=Path("dumps"))
    args = parser.parse_args(argv)
    for example in load_examples(args.target):
        for arch in args.arch or sorted(ARCHS):
            print(dump(example, arch, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
