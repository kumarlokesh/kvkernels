"""Check an installed kvkernels: every public name imports, and two ops run once.

Run in an environment that has only the built wheel and CPU torch installed:

    python scripts/wheel_smoke.py

The ops run on the CPU under the Triton interpreter, so no GPU is needed. Run
from the repository root, the script still imports the installed package, since
Python puts the script's own directory on the path, not src/.
"""

from __future__ import annotations

import importlib
import os
import sys

# The v0.1.0 public API, by module.
PUBLIC = {
    "kvkernels.ops": [
        "CascadePlan",
        "PrefixGroup",
        "cascade_decode_attention",
        "gather_kv_blocks",
        "merge_attention_states",
        "merge_attention_states_n",
        "scatter_kv_blocks",
    ],
    "kvkernels.layouts": [
        "LAYOUTS",
        "CacheSpec",
        "PagedKVLayout",
        "kv_views",
        "make_cache",
        "make_index",
        "planes",
    ],
    "kvkernels.reference": [
        "FoundGroup",
        "RequestKV",
        "attention_ref32",
        "attention_ref_low",
        "find_prefix_groups_ref",
        "gather_kv_blocks_ref",
        "gather_tokens",
        "merge_states_ref",
        "request_kv",
        "scatter_kv_blocks_ref",
    ],
}
REGISTERED_OPS = [
    "gather_kv_blocks",
    "scatter_kv_blocks",
    "merge_attention_states",
    "merge_attention_states_n",
    "cascade_decode_attention",
]


def main() -> int:
    os.environ["TRITON_INTERPRET"] = "1"  # before Triton is imported
    import torch

    import kvkernels

    for module, names in PUBLIC.items():
        mod = importlib.import_module(module)
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            print(f"{module} is missing {missing}", file=sys.stderr)
            return 1
    for name in REGISTERED_OPS:
        getattr(torch.ops.kvkernels, name)  # raises if the op is not registered

    from kvkernels.ops import gather_kv_blocks, merge_attention_states
    from kvkernels.reference import merge_states_ref

    plane = torch.arange(8 * 4 * 16, dtype=torch.float16).view(8, 4, 16)
    ids = torch.tensor([5, 1, 6], dtype=torch.int32)
    out = torch.empty(3, 4, 16, dtype=torch.float16)
    gather_kv_blocks([plane], ids, [out])
    if not torch.equal(out, plane[ids.long()]):
        print("gather_kv_blocks gave a wrong result", file=sys.stderr)
        return 1

    gen = torch.Generator().manual_seed(0)
    o = torch.randn(2, 3, 2, 64, generator=gen)
    lse = torch.randn(2, 3, 2, generator=gen)
    merged = torch.empty(3, 2, 64)
    merge_attention_states(o[0], lse[0], o[1], lse[1], merged)
    ref, _ = merge_states_ref(o, lse)
    torch.testing.assert_close(merged, ref, rtol=1e-5, atol=1e-5)

    print(f"kvkernels {kvkernels.__version__} from {kvkernels.__file__}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
