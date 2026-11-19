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
- The cascade reads every level of a prefix tree once: planning on the GPU
  finds the tree from the block table or slot list (`CascadePlan(max_levels=4)`;
  1 gives the earlier one level of groups), and each node is a work item over
  the requests below it. `PrefixGroup.start` names nested groups for the host
  plan.
- Several new query tokens per request (`CascadePlan(query_tokens=...)`, 1 to
  16, for speculative decoding's verification step): q, out and lse_out hold
  `requests · query_tokens` rows, and token t sees the request's keys up to
  position length - query_tokens + t.
- fp8 (e4m3) K and V in the cascade, with one K and one V scale per layer
  (`cascade_decode_attention(..., k_scale=, v_scale=)`), as vLLM's and SGLang's
  default fp8 caches store them; K and V views accept float8_e4m3fn.
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
  30.7. A launch whose programs already fill the GPU once is not split.
- The cascade's prefix launch runs 4 warps and 2 pipeline stages (8 and 3
  before), and both attention launches read keys in tiles of 32 tokens (64
  before), so two prefix programs fit on an SM; split-K ranges are at least 2
  tokens per query row. On an L4 the cascade takes 0.97x the time over the
  suite (geometric mean of 267 cases), 0.83x at best and 1.14x at worst
  (batch 128, 2,048-token prefix, 128-token suffixes); batch 1 with a
  512-token prefix takes 21.5 us per layer.
- Requests are grouped only when they share at least 512 tokens
  (`min_shared_tokens`, 64 before). On an L4, groups sharing 16-256 tokens
  took 0.98-1.07x the time of reading those tokens with each request; from 512
  tokens they took 0.61-1.03x.
- A plan for one request, or a batch-invariant plan, leaves out the prefix
  launch, which has no work then: on an L4 it took 2.0 us of the 19.7 us of a
  batch-1 call (Llama 3.1 8B, 512 tokens) replayed from a CUDA graph.
- Planning on the GPU runs three launches instead of four: one launch finds the
  prefix tree by sorting and binary search, and the fill launch splits each
  item's tasks and merge entries over 8 programs.
- Split-K bounds each item's partial results: s ranges of an item are allowed
  when its partial results are no larger than its K and V, or fit in a third of
  L2. A 4,096-token prefix shared by 64 requests with 4 query tokens each was
  split into 9 ranges whose partial results (75 MB) left an L4's 48 MB L2.
- The cascade's attention kernel requires q, K and V with a contiguous head
  dimension (every supported layout has one) and loads K and V 16 bytes at a
  time; with the stride an argument it loaded 2 bytes at a time. On an L4 the
  prefix pass issues 43% fewer instructions and long prefixes take 2-3% less
  time.
- GPU planning runs its tree and items launches with 8 warps for plans of 129
  to 2,048 requests (4 warps up to 512 before). On an L4, planning a batch of
  256 takes 48 us instead of 51, a three-level tree at batch 256 takes 70 us
  instead of 77, and a batch of 512 takes 80 us instead of 96.

