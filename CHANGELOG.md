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
- `kvkernels.layouts`: the KV cache layouts of vLLM 0.32.0 and SGLang 0.5.22,
  and the views the ops take of them.
- `kvkernels.reference`: PyTorch references for every op.
- Support for torch 2.13 and later and Triton 3.7 and later, on Python 3.11 and
  later.
