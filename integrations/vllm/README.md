# kvkernels-vllm

kvkernels' cascade decode attention inside an unmodified vLLM 0.33.0 server.
Single-token decode rows go to the cascade, planned on the GPU from vLLM's
block table; prefill and extend rows and KV writes stay with vLLM's Triton
backend.

```bash
uv pip install -e . -e ../..        # this package and kvkernels, next to vllm==0.33.0
vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend CUSTOM
```

The backend refuses, at startup, models and settings the cascade does not
support: head sizes other than 64 and 128, more than 16 query heads per KV head,
sliding windows, ALiBi, logit soft caps, attention sinks, speculative decoding.
An fp8 KV cache (`--kv-cache-dtype fp8` or `fp8_e4m3`) is supported on GPUs of
compute capability 8.9 or later; the query stays in the model's dtype.

`KVKERNELS_VLLM_CHECK=1` with `--enforce-eager` also runs vLLM's Triton kernel
and an fp32 reference on every decode call, and fails on a mismatch.
