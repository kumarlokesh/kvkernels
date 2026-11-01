"""Triton kernels for paged KV-cache movement and shared-prefix decode attention.

The public API lives in three modules: kvkernels.ops (the Triton ops),
kvkernels.layouts (engine KV cache layouts and their views) and
kvkernels.reference (PyTorch references for the ops). Importing this package
alone needs neither Triton nor a GPU.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kvkernels")
except PackageNotFoundError:  # run from a source tree that was never installed
    __version__ = "0.0.0+unknown"
