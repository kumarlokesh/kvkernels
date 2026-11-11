# kvkernels

Triton kernels for the KV cache of LLM inference: paged KV block gather and
scatter, and shared-prefix (cascade) decode attention, on the KV cache layouts
vLLM and SGLang use.

## Install

```bash
pip install kvkernels
```

Needs torch 2.13+ and, on Linux, Triton 3.7+, so it installs next to vLLM 0.32.0
or SGLang 0.5.22. Supports Python 3.11+ and NVIDIA GPUs of compute capability
7.5 to 9.0.

## What it does

**Gather and scatter KV blocks.** Copy a request's scattered KV blocks into one
contiguous buffer, before offloading them or sending them to another GPU, and
back. Raw bytes, so every dtype works, on every vLLM and SGLang layout.

```
paged KV cache   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ …
                       ▲               ▲   ▲
staging buffer   │ 5 │ 1 │ 6 │                       gather_kv_blocks(planes, [5, 1, 6], staging)
```

**Decode with a shared prefix.** When requests share a prefix (a system prompt),
the cascade reads its K and V once for the whole batch, reads each request's own
suffix once, and merges the two results exactly.

```
             shared prefix              own suffix
request 0    ████████████████████████   ░░          ─┐
request 1    ████████████████████████   ░░░░░        ├─▶ merge ─▶ out
request 2    ████████████████████████   ░░░░░░░░░   ─┘
             └── read once for all ─┘   └ per request ┘

plan.update(groups=[PrefixGroup(requests=range(0, 3), blocks=[1, 2, 3], length=40)], ...)
cascade_decode_attention(q, k, v, plan, out)        # each layer
```

Runnable versions: [examples/gather.py](examples/gather.py),
[examples/cascade.py](examples/cascade.py). Every op writes into tensors the
caller provides, can be captured in a CUDA graph, and has a PyTorch reference in
`kvkernels.reference`.

## Results

Every chart comes from one benchmark run, which records the GPU and software
versions. All numbers: [results/README.md](results/README.md).

### KV block gather

<!-- results:kv_gather:begin -->
On a Tesla T4, kvkernels gathers 512 KV blocks from vLLM's default layout at 223 GB/s, 94% of the GPU's measured copy bandwidth (index_select 232 GB/s, vLLM copy kernel 213 GB/s, cuMemcpyBatchAsync 58 GB/s).

<picture><source media="(prefers-color-scheme: dark)" srcset="figures/t4/kv_gather-dark.svg"><img alt="Gather of 512 KV blocks (64 KiB each, random order) on a Tesla T4" src="figures/t4/kv_gather-light.svg" width="720"></picture>

<sub>Tesla T4, torch 2.14.1+cu126, Triton 3.8.0, commit <code>dirty-615f2a83</code>; from <code>results/t4/kv_gather/20261031T051548Z.json</code></sub>
<!-- results:kv_gather:end -->

### Cascade decode attention

<!-- results:cascade:begin -->
No results yet.
<!-- results:cascade:end -->

## Limitations

- On the T4 (compute capability 7.5), Triton uses no tensor cores, so the
  attention ops are correct but slower than FlashInfer there.
- Decode only; no prefill kernels.
- Outside a CUDA graph, each call costs tens of microseconds of CPU time.

## Development

On macOS, use the devcontainer; Triton runs on Linux only.

```bash
make install && make test && make lint     # CPU, under the Triton interpreter
make install-cu130 && make test-gpu        # on a GPU machine
uv run python scripts/gpu_session.py run --out session/   # every GPU step, with logs
```

`scripts/kaggle_session.py` runs the GPU session on a free Kaggle T4.

## License

[Apache License 2.0](LICENSE)
