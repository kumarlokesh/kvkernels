"""Copy one request's KV blocks out of a vLLM-layout cache into a contiguous buffer.

Run on a GPU: python examples/gather.py
Without a GPU, under the Triton interpreter: TRITON_INTERPRET=1 python examples/gather.py
"""

import torch

from kvkernels.layouts import LAYOUTS, CacheSpec, make_cache, planes
from kvkernels.ops import gather_kv_blocks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A Llama 3.1 8B cache as vLLM allocates it: 16-token KV blocks, 8 KV heads of 128.
layout = LAYOUTS["vllm_LBNHC"]
spec = CacheSpec(
    num_layers=32, num_blocks=64, block_size=16, num_kv_heads=8, head_dim=128, dtype=torch.float16
)
cache = make_cache(layout, spec, device=DEVICE)

# Copy one request's KV blocks of layer 0 into a contiguous buffer, ready to offload.
layer = planes(layout, spec, cache[0])  # [64 blocks, 16, 8, 256]: K and V together
ids = torch.tensor([5, 1, 6], dtype=torch.int32, device=DEVICE)
staging = [torch.empty(len(ids), *p.shape[1:], dtype=p.dtype, device=DEVICE) for p in layer]
gather_kv_blocks(layer, ids, staging)  # staging[0][i] = layer[0][ids[i]]

assert torch.equal(staging[0].view(torch.int16), layer[0][ids.long()].view(torch.int16))
print("gathered blocks", ids.tolist(), "into", tuple(staging[0].shape))
