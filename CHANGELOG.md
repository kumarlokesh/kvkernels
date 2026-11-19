# Changelog

All notable changes to kvkernels are listed here, in the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versions follow
semantic versioning for 0.x releases: a patch release never changes the public
API; a minor release may, and lists every such change under Changed or Removed.
Each version ends with a "Tested on:" line naming the GPUs where the GPU tests
passed on the released source, with the torch and Triton versions of that run.

## [Unreleased]

The first release.

### Added

- `gather_kv_blocks` and `scatter_kv_blocks`: copy KV blocks between a paged
  cache and contiguous rows, as raw bytes, for every vLLM and SGLang layout.
- `PlaneSet`: gather or scatter the same KV blocks in every plane of a cache
  (all layers, K and V) in one launch. The planes and row buffers are checked
  once, so each call checks only the ids, and calls can be captured in a CUDA
  graph. Row buffers may be pinned host memory: a gather then writes straight
  to the host over PCIe, and a scatter reads from it.
- `merge_attention_states` and `merge_attention_states_n`: combine partial
  attention results over disjoint key sets using their LSE.
- `cascade_decode_attention`, `CascadePlan` and `PrefixGroup`: decode attention
  that reads each shared prefix once for the requests that share it, reads each
  request's own tokens once, and merges the results. Split-K fills the GPU at
  small batches, and calls can be captured in a CUDA graph across plan updates.
- `CascadePlan.plan_from_block_table` and `CascadePlan.plan_from_slots`: plan
  each decode step on the GPU from a vLLM block table or an SGLang slot list,
  with no host sync. Requests that share at least `min_shared_tokens` leading
  tokens (default 512) are grouped without naming the groups. The planning
  launches are recorded in a CUDA graph on the second call with the same
  tensors and replayed after that.
- Several levels of shared prefix: planning on the GPU finds a prefix tree
  (`CascadePlan(max_levels=...)`, default 4; 1 gives one level of groups), and
  each node is read once for the requests below it. `PrefixGroup.start` places
  nested groups in a host plan.
- Several new query tokens per request (`CascadePlan(query_tokens=...)`, 1 to
  16, for speculative decoding's verification step): q, out and lse_out hold
  `requests · query_tokens` rows, and token t sees the request's keys up to
  position length - query_tokens + t.
- fp8 (e4m3) K and V, with one K and one V scale per layer
  (`cascade_decode_attention(..., k_scale=, v_scale=)`), as vLLM's and SGLang's
  default fp8 caches store them. Needs compute capability 8.9 or later, where
  Triton has an e4m3 type; below it the op raises a ValueError.
- `CascadePlan(batch_invariant=True)`: no groups, and split-K by each request's
  length alone, so a request's output does not depend on its batch.
- `kvkernels.layouts`: the KV cache layouts of vLLM 0.32.0 and SGLang 0.5.22,
  and the views the ops take of them, fp8 included.
- `kvkernels.reference`: PyTorch references for every op.
- Support for torch 2.13 and later and Triton 3.7 and later, on Python 3.11 and
  later, on NVIDIA GPUs of compute capability 7.5 to 9.0. The cascade needs q,
  K and V with a contiguous head dimension, which every supported layout has.
