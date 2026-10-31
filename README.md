# kvkernels

Triton kernels for the KV-cache layer of LLM inference: paged KV movement and
shared-prefix (cascade) decode attention. Every kernel is checked against a
PyTorch reference and measured against the roofline.

## The problem

Requests in a decode batch often share a prefix: a system prompt, tool
definitions, few-shot examples. A standard paged decode kernel reads that
prefix's K and V from GPU memory once per request. A cascade kernel reads it
once per batch:

1. Attend all queries in the batch to the shared prefix in one pass. The batch
   turns many matrix-vector products into one matrix multiply that tensor cores
   can run.
2. Attend each query to its own suffix.
3. Merge the two partial results exactly, using their log-sum-exp.

One layer, 32 requests, 4,096-token prefix, 8 KV heads, head_dim 128, fp16:

```
prefix K and V     2 · 4096 · 8 · 128 · 2 bytes   ≈  16.8 MB
standard decode    read 32 times                  ≈ 537   MB
cascade            read once                      ≈  16.8 MB
```

These figures are a model. Part of the repeated reads in standard decode can
hit L2, so the measured gap will be smaller; measuring it is part of this
project. Measured results will replace this section.

## Kernels

| Kernel | What it does | Status |
|---|---|---|
| Paged KV gather and scatter | Copies scattered KV blocks into a contiguous buffer, and back | planned |
| Attention-state merge | Combines partial attention outputs using their log-sum-exp | planned |
| Shared-prefix decode attention | Prefix pass, suffix pass and merge, on a real paged layout | planned |

## How results are produced

- Each kernel has a pure-PyTorch reference. Data-movement kernels must match it
  bit for bit; attention kernels must match an fp32 reference within a fixed
  tolerance.
- Each published number comes from a committed result JSON that records the GPU,
  driver, CUDA, PyTorch and Triton versions.
- Each benchmark reports percent of the roofline as well as speedup over a
  baseline.

## Development

Triton runs on Linux only. On macOS, open the repository in the devcontainer
(`.devcontainer/`).

```bash
make install     # CPU torch, for the devcontainer and CI
make test        # CPU tests under the Triton interpreter
make lint        # ruff and mypy
```

On a GPU machine:

```bash
make install-cu130                      # or install-cu126 for older drivers
uv run python scripts/env_report.py     # GPU, versions, Nsight counter access
make test-gpu
make bench-smoke                        # writes results/<gpu>/smoke/<timestamp>.json
```

Every GPU step, in order, with a log each (results are written only after the
GPU tests pass):

```bash
uv run python scripts/gpu_session.py run --out session/
```

On a free Kaggle GPU (Tesla T4), from a development machine, with the Kaggle CLI
and an API token:

```bash
python scripts/kaggle_session.py push --user <kaggle-name>
python scripts/kaggle_session.py status --user <kaggle-name>
python scripts/kaggle_session.py fetch --user <kaggle-name>
python scripts/gpu_session.py collect gpu-sessions/kaggle/<time>/session
```

## Prior work

Cascade attention is available in [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
and the decomposition is described in the Hydragen paper
([arXiv:2402.05099](https://arxiv.org/abs/2402.05099)). vLLM had an opt-in cascade
path and removed it in October 2026
([#61126](https://github.com/vllm-project/vllm/pull/61126)). kvkernels implements it in
Triton on the KV layouts SGLang and vLLM use, and measures it against the roofline.

## License

[Apache License 2.0](LICENSE)
