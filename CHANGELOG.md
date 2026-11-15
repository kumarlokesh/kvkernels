# Changelog

All notable changes to kvkernels are listed here, in the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versions follow
semantic versioning for 0.x releases: a patch release never changes the public
API; a minor release may, and lists every such change under Changed or Removed.
Each version ends with a "Tested on:" line naming the GPUs where the GPU tests
passed on the released source, with the torch and Triton versions of that run.

## [Unreleased]

### Added

- `gather_kv_blocks` and `scatter_kv_blocks`: copy KV blocks between a paged
  cache and contiguous rows, as raw bytes, for every vLLM and SGLang layout.
- `merge_attention_states` and `merge_attention_states_n`: combine partial
  attention results over disjoint key sets using their LSE.
- `cascade_decode_attention`, `CascadePlan` and `PrefixGroup`: decode attention
  that reads each shared prefix once per group of requests, with split-K for
  small batches and CUDA graph capture across plan updates.
- `CascadePlan.plan_from_block_table` and `CascadePlan.plan_from_slots`: plan
  the cascade on the GPU from a vLLM block table or an SGLang slot list. Requests
  that share leading KV blocks are grouped without naming groups and without a
  host sync, so planning and attention capture in one CUDA graph.
- `CascadePlan(batch_invariant=True)`: no groups and split-K by each request's
  length alone, so a request's output does not depend on its batch.
- Planning from a block table or slot list replays its four launches from a CUDA
  graph the plan records on the second call with the same tensors, so a step's
  planning costs one graph launch of CPU time.
- `PlaneSet`: gather or scatter the same KV blocks in every plane of a cache
  (all layers, K and V) in one launch. The planes and row buffers are checked
  once and their addresses written to a device table, so each call checks only
  the ids; calls can be captured in a CUDA graph.
- `kvkernels.layouts`: the KV cache layouts of vLLM 0.32.0 and SGLang 0.5.22,
  and the views the ops take of them.
- `kvkernels.reference`: PyTorch references for every op.
- Support for torch 2.13 and later and Triton 3.7 and later, on Python 3.11 and
  later.

### Changed

- The cascade's attention and merge kernels launch a fixed number of programs
  that loop over the batch's work, instead of one program per slot of the
  plan's capacity. Under a CUDA graph sized for 256 requests this removes about
  0.27 ms per layer at batch 16 on an L4.
- The cascade's prefix tiles are 128 query rows with 8 warps and 3 pipeline
  stages (64 rows and 4 warps before). On an L4 one layer at batch 64 and 128
  with a 32,768-token prefix takes 1.5-1.6x less time; batch 16 takes up to 3%
  more and batch 1 4 us more. `CascadePlan(prefix_block_m=64)` gives 64-row
  prefix tiles, still with 8 warps.
- Split-K counts the prefix launch's programs and the suffix launch's programs
  separately. With both counted together, a batch with many suffixes left its
  prefix unsplit: batch 128, 32,768-token prefix, 128-token suffixes took
  3.3 ms per layer on an L4 and now takes 1.8 ms.
