"""Decode attention for four requests that share a 40-token prefix.

Run on a GPU: python examples/cascade.py
Without a GPU, under the Triton interpreter: TRITON_INTERPRET=1 python examples/cascade.py
"""

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, kv_views, make_cache
from kvkernels.ops import CascadePlan, PrefixGroup, cascade_decode_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

layout = LAYOUTS["vllm_LBNHC"]
spec = CacheSpec(
    num_layers=32, num_blocks=64, block_size=16, num_kv_heads=8, head_dim=128, dtype=torch.float16
)
cache = make_cache(layout, spec, device=DEVICE)

# Once, for the largest batch the engine runs.
plan = CascadePlan(
    max_requests=4,
    max_groups=1,
    max_blocks=16,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=16,
    num_blocks=64,
    device=torch.device(DEVICE),
)

# Every decode step, on the host: requests 0-3 share a 40-token prefix in blocks
# 1-3, and each has its own suffix.
plan.update(
    groups=[PrefixGroup(requests=range(0, 4), blocks=[1, 2, 3], length=40)],
    suffix_blocks=[[4], [5], [6, 7], [8]],
    suffix_lens=[3, 9, 20, 0],
)

# Every layer: one query token per request.
q = torch.randn(4, 32, 128, dtype=torch.float16, device=DEVICE)
k, v = kv_views(layout, spec, cache[0])  # [64 blocks, 16 slots, 8 heads, 128]
out = torch.empty_like(q)
cascade_decode_attention(q, k, v, plan, out)

print("attention output", tuple(out.shape))
