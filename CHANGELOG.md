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
  the ids; calls can be captured in a CUDA graph. Row buffers may be pinned host
  memory: a gather then writes straight to the host over PCIe, and a scatter
  reads from it. That keeps the kernel on the SMs for the whole transfer; on
  an L4 it slowed a matmul on another stream 1.7x, against 1.16x for gathers of
  8 layers at a time into a GPU buffer, each copied to the host on a second
  stream.
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
  more. `CascadePlan(prefix_block_m=64)` gives 64-row prefix tiles, still with
  8 warps. A group of at most `small_item_rows` query rows (default 16: four
  requests with a GQA group of 4) runs in the suffix launch's 16-row tiles.
- Split-K counts the prefix launch's programs and the suffix launch's programs
  separately. With both counted together, a batch with many suffixes left its
  prefix unsplit: batch 128, 32,768-token prefix, 128-token suffixes took
  3.3 ms per layer on an L4 and now takes 1.8 ms.
- A prefix task of at most 64 or 32 query rows runs in a 64- or 32-row tile
  inside the same launch (`CascadePlan(prefix_min_block_m=...)`, default a
  quarter of the prefix tile). On an L4, groups of 17 to 32 rows (Llama 3.1 8B
  at batch 8) take up to 17% less time.
- Split-K chooses the number of token ranges per item that best fills whole
  rounds of the programs that run at once (one prefix program per SM, two
  suffix programs per SM by default: `prefix_programs_per_sm`,
  `programs_per_sm`), and the shortest range is 4 tokens per query row of the
  item's tile, between 128 and `min_split_tokens` (512). On an L4 over batch
  1-128 and prefixes of 512-32,768 tokens the cascade takes 0.96x the time of
  the earlier rule (geometric mean of 267 cases), 0.74x at best and 1.06x at
  worst; batch 1 with a 512-token prefix takes 24.6 us per layer instead of
  30.7.
