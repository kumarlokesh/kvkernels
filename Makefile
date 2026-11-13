.PHONY: install install-cu126 install-cu130 install-oldest install-vllm-cpu test test-gpu \
	test-oldest test-vllm-cpu lint fmt \
	readme bench-smoke check-timer dram-cascade plot

# torch is a package dependency, so a plain `uv run` would replace the torch build
# that `make install*` chose with PyPI's. uv run uses the environment as installed.
export UV_NO_SYNC := 1

# CPU torch: the devcontainer, CI and macOS.
install:
	uv sync --extra cpu --extra bench

# CUDA torch: rented GPU machines. Pick the build the machine's driver supports.
# The baselines extra (FlashInfer) compiles kernels on first use and needs nvcc.
install-cu126:
	uv sync --extra cu126 --extra bench --extra baselines

install-cu130:
	uv sync --extra cu130 --extra bench --extra baselines

# The lowest versions the package supports (torch 2.13.0, Triton 3.7.1, Python 3.11),
# CPU torch, in a separate environment outside the lock file.
OLDEST_ENV ?= .venv-oldest
install-oldest:
	uv venv -q --allow-existing -p 3.11 $(OLDEST_ENV)
	uv pip install -q -p $(OLDEST_ENV) --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
	uv pip install -q -p $(OLDEST_ENV) triton==3.7.1 numpy matplotlib pandas pytest
	uv pip install -q -p $(OLDEST_ENV) --no-deps -e .

# vLLM 0.33.0 with CPU torch, in a separate environment, to run the vLLM backend's
# tests against vLLM's own classes without a GPU. vLLM's dependencies pull CUDA builds
# of torchvision and torchaudio, which are swapped for the CPU builds afterwards.
VLLM_ENV ?= .venv-vllm
CPU_TORCH := --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision torchaudio==2.11.0
install-vllm-cpu:
	uv venv -q --allow-existing -p 3.12 $(VLLM_ENV)
	uv pip install -q -p $(VLLM_ENV) $(CPU_TORCH)
	uv pip install -q -p $(VLLM_ENV) vllm==0.33.0 pytest
	uv pip install -q -p $(VLLM_ENV) $(CPU_TORCH) --reinstall-package torchvision --reinstall-package torchaudio
	uv pip install -q -p $(VLLM_ENV) --no-deps -e . -e integrations/vllm

test:
	TRITON_INTERPRET=1 uv run pytest -m "not gpu"

test-vllm-cpu:
	TRITON_INTERPRET=1 $(VLLM_ENV)/bin/python -m pytest -p no:cacheprovider integrations/vllm/tests -m "not gpu"

test-oldest:
	TRITON_INTERPRET=1 $(OLDEST_ENV)/bin/python -m pytest -m "not gpu"

test-gpu:
	uv run pytest -m gpu

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff format .
	uv run ruff check --fix .

# Rewrites the README's results tables from the newest committed result files.
readme:
	uv run python scripts/readme_results.py

bench-smoke:
	uv run python -m bench.harness --suite smoke

# Checks that the benchmark timer leaves out kernel launch overhead; needs Nsight Compute (ncu).
check-timer:
	uv run python -m bench.timer_check

# Device-memory bytes of the cascade and per-request decode; needs Nsight Compute (ncu).
dram-cascade:
	uv run python -m bench.dram_bytes

# Usage: make plot RESULT=results/<gpu>/<suite>/<timestamp>.json X=<param>
plot:
	uv run python -m bench.plot $(RESULT) --x $(X)
