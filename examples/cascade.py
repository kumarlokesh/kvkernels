"""Decode attention for four requests that share a 64-token prefix.

Run on a GPU: python examples/cascade.py
Without a GPU, under the Triton interpreter: TRITON_INTERPRET=1 python examples/cascade.py
"""

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache
from kvkernels.ops import CascadePlan, cascade_decode_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

layout = LAYOUTS["vllm_LBNHC"]
spec = CacheSpec(
    num_layers=32, num_blocks=64, block_size=16, num_kv_heads=8, head_dim=128, dtype=torch.float16
)
cache = make_cache(layout, spec, device=DEVICE)

# Once, for the largest batch the engine runs.
plan = CascadePlan(
    max_requests=4,
    max_groups=2,
    max_blocks=16,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=16,
    num_blocks=64,
    device=torch.device(DEVICE),
)

# Every decode step, from the engine's block table: requests 0-3 start with the
# same four KV blocks (64 tokens of a shared prompt), then KV blocks of their own.
block_table = torch.tensor(
    [[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 6, 0], [1, 2, 3, 4, 7, 8], [1, 2, 3, 4, 0, 0]],
    dtype=torch.int32,
    device=DEVICE,
)
seq_lens = torch.tensor([67, 73, 84, 64], dtype=torch.int32, device=DEVICE)
plan.plan_from_block_table(block_table, seq_lens)  # on the GPU, no host sync

# Every layer: one query token per request.
q = torch.randn(4, 32, 128, dtype=torch.float16, device=DEVICE)
k, v = kv_views(layout, spec, cache[0])  # [64 blocks, 16 slots, 8 heads, 128]
out = torch.empty_like(q)
cascade_decode_attention(q, k, v, plan, out)

print("attention output", tuple(out.shape))
