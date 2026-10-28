"""Reference gather and scatter of whole KV blocks.

A plane is a tensor whose first dimension indexes KV blocks. Gather copies the
blocks named by ``block_ids`` into consecutive rows of ``out``; scatter copies
consecutive rows of ``src`` into the blocks named by ``block_ids``. Both index
the first dimension of a byte view of each tensor, which copies whole blocks bit
for bit and works for every dtype, fp8 included.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """The same memory as uint8; the last dimension grows by the element size."""
    return t.view(torch.uint8)


def gather_kv_blocks_ref(
    planes: Sequence[torch.Tensor], block_ids: torch.Tensor, outs: Sequence[torch.Tensor]
) -> None:
    """outs[p][i] = planes[p][block_ids[i]] for every plane p and row i."""
    ids = block_ids.long()
    for plane, out in zip(planes, outs, strict=True):
        _bytes(out).copy_(_bytes(plane)[ids])


def scatter_kv_blocks_ref(
    srcs: Sequence[torch.Tensor], block_ids: torch.Tensor, planes: Sequence[torch.Tensor]
) -> None:
    """planes[p][block_ids[i]] = srcs[p][i] for every plane p and row i.

    With a repeated id, the block's final contents are undefined, as in the op.
    """
    ids = block_ids.long()
    for src, plane in zip(srcs, planes, strict=True):
        _bytes(plane)[ids] = _bytes(src)
