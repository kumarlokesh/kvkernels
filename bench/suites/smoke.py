"""Smoke suite: device-to-device copies. The 1 GiB case is the measured-peak probe."""

from __future__ import annotations

from collections.abc import Iterator

from bench.harness import Case, Tensors

MiB = 1 << 20
PROBE_BYTES = 1024 * MiB
SIZES = (64 * MiB, 256 * MiB, PROBE_BYTES)


def copy_case(nbytes: int, device: str = "cuda") -> Case:
    def setup() -> Tensors:
        import torch

        src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device)
        return {"src": src, "dst": torch.empty_like(src)}

    def torch_copy(t: Tensors) -> None:
        t["dst"].copy_(t["src"])

    return Case(
        params={"bytes": nbytes},
        setup=setup,
        impls={"torch_copy": torch_copy},
        outputs=("dst",),
        check="exact",
        bytes_moved=lambda p: 2 * int(p["bytes"]),
    )


def cases() -> Iterator[Case]:
    for nbytes in SIZES:
        yield copy_case(nbytes)
