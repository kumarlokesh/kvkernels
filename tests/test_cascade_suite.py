"""The cascade benchmark suite: its grid, cost model, inputs and implementations.

Under the Triton interpreter the implementations that need no FlashInfer
(kvkernels in each setting, vLLM's copied kernel and the PyTorch reference) run
on small batches and are checked the way the harness checks them on a GPU.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from bench import harness
from bench.schema import EnvInfo, Params
from bench.suites import cascade as suite

INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRETING else "cuda"
needs_kernels = pytest.mark.skipif(
    not INTERPRETING and not torch.cuda.is_available(),
    reason="needs the Triton interpreter or a CUDA GPU",
)


def tiny(**changes: int | str) -> Params:
    p: Params = {
        "model": "tiny",
        "batch": 3,
        "groups": 1,
        "prefix_len": 32,
        "suffix_len": 8,
        "q_heads": 8,
        "kv_heads": 2,
        "head_dim": 64,
        "layout": "vllm_LBNHC",
        "block": 16,
        "dtype": "float16",
    }
    p.update(changes)
    return p


def test_grid_covers_the_planned_cases() -> None:
    params = list(suite.grid())
    main = 2 * 6 * 4 * 4  # models, batches, prefixes, suffixes
    extra = 4 * 4 * 4  # per (prefix, suffix): groups, GQA 1, GQA 8, SGLang layout
    assert len(params) == main + extra + 1  # and one bf16 shape
    assert len({tuple(sorted(p.items())) for p in params}) == len(params)
    for p in params:
        assert int(p["batch"]) % int(p["groups"]) == 0
        assert int(p["prefix_len"]) % int(p["block"]) == 0  # per-request block lists need it
    assert {(p["q_heads"], p["kv_heads"]) for p in params} == {(32, 8), (32, 32), (32, 4)}
    assert {p["layout"] for p in params} == {"vllm_LBNHC", "sglang_nhd"}


def test_cost_model_reads_the_prefix_once_per_group() -> None:
    p = suite._params(batch=32, prefix=8192, suffix=512)
    prefix_bytes = 2 * 8192 * 8 * 128 * 2  # K and V once, 32 MiB
    suffix_bytes = 32 * 2 * 512 * 8 * 128 * 2  # each request's own, 64 MiB
    assert suite.bytes_moved(p) == prefix_bytes + suffix_bytes
    assert suite.flops(p) == 32 * 4 * (8192 + 512) * 32 * 128
    four = suite._params(batch=64, prefix=8192, suffix=512, groups=4)
    assert suite.bytes_moved(four) == 4 * prefix_bytes + 2 * suffix_bytes


def test_inputs_give_every_prefix_and_suffix_its_own_blocks() -> None:
    t = suite.make_inputs(tiny(batch=4, groups=2), device="cpu")
    b = suite.batch_of(t)
    assert b.group_starts == [0, 2, 4]
    assert [len(blocks) for blocks in b.prefix_blocks] == [2, 2]
    assert [len(blocks) for blocks in b.suffix_blocks] == [1, 1, 1, 1]
    every = [x for blocks in b.prefix_blocks + b.suffix_blocks for x in blocks]
    assert sorted(every) == list(range(1, 9))  # distinct, block 0 never used
    assert b.request_lens() == [40, 40, 40, 40]
    assert b.request_blocks()[3] == b.prefix_blocks[1] + b.suffix_blocks[3]


def test_small_inputs_keep_the_batch_and_shorten_the_sequences() -> None:
    t = suite.make_inputs(tiny(prefix_len=320, suffix_len=40), small=True, device="cpu")
    b = suite.batch_of(t)
    assert b.size == 3
    assert b.prefix_lens == [32] and b.suffix_lens == [16, 16, 16]


def test_flashinfer_entries_follow_availability_and_empty_suffixes() -> None:
    names = list(suite.implementations(tiny(), with_flashinfer=True))
    assert names[0] == "flashinfer_decode"
    assert "flashinfer_cascade" in names
    assert "flashinfer_cascade" not in suite.implementations(
        tiny(suffix_len=0), with_flashinfer=True
    )
    without = list(suite.implementations(tiny(), with_flashinfer=False))
    assert without[0] == "torch_per_request"
    assert not any(name.startswith("flashinfer") for name in without)


def test_split_settings_run_only_at_small_batch() -> None:
    small = suite.implementations(
        suite._params(batch=4, prefix=512, suffix=0), with_flashinfer=False
    )
    large = suite.implementations(
        suite._params(batch=16, prefix=512, suffix=0), with_flashinfer=False
    )
    assert set(suite.SPLIT_SETTINGS) <= set(small)
    assert not set(suite.SPLIT_SETTINGS) & set(large)


@needs_kernels
@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"batch": 4, "groups": 2, "suffix_len": 0},
        {"layout": "sglang_nhd", "block": 1, "prefix_len": 12, "suffix_len": 5},
    ],
    ids=["one_group", "two_groups_no_suffix", "sglang_one_token_pages"],
)
def test_every_implementation_matches_the_reference(changes: dict[str, int | str]) -> None:
    case = suite.make_case(tiny(**changes), with_flashinfer=False, device=DEVICE)
    harness.check_case(case)


@needs_kernels
def test_planned_implementations_report_plan_time() -> None:
    # Without host_setup: measuring host_us runs each implementation 1,000 times.
    case = dataclasses.replace(
        suite.make_case(tiny(batch=16), with_flashinfer=False, device=DEVICE), host_setup=None
    )
    result = harness.run_suite(
        "cascade",
        [case],
        None,
        env=EnvInfo("cpu", "cpu", "-", "-", "-", "-", None, None, "0" * 64),
        git_sha="x",
        flush_bytes=1,
        measured_peak_gbps=None,
        timer=lambda fn: [1.0] * 100,
    )
    plan_us = {m.impl: m.plan_us for m in result.measurements}
    assert plan_us.pop("torch_per_request") is None
    assert plan_us and all(us is not None and us > 0 for us in plan_us.values())


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [4, 32])
def test_every_implementation_matches_on_a_full_shape(batch: int) -> None:
    p = suite._params(batch=batch, prefix=2048, suffix=128)
    harness.check_case(suite.make_case(p, with_flashinfer=suite.flashinfer_runs()))
