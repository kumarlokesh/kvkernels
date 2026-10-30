.PHONY: install install-cu126 install-cu130 test test-gpu lint fmt bench-smoke check-timer dram-cascade plot

# CPU torch: the devcontainer, CI and macOS.
install:
	uv sync --extra cpu --extra bench

# CUDA torch: rented GPU machines. Pick the build the machine's driver supports.
# The baselines extra (FlashInfer) compiles kernels on first use and needs nvcc.
install-cu126:
	uv sync --extra cu126 --extra bench --extra baselines

install-cu130:
	uv sync --extra cu130 --extra bench --extra baselines

test:
	TRITON_INTERPRET=1 uv run pytest -m "not gpu"

test-gpu:
	uv run pytest -m gpu

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff format .
	uv run ruff check --fix .

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
