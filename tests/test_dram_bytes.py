"""Reading device-memory bytes from Nsight Compute's CSV output."""

from __future__ import annotations

import pytest

from bench.dram_bytes import parse_ncu_bytes

# Two kernels, as ncu --csv prints them, with a log line mixed in.
NCU_CSV = """==PROF== Connected to process 4242
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","4242","python","127.0.0.1","_work_item_kernel","1","7","Command line profiler metrics","dram__bytes_read.sum","Mbyte","33.55"
"0","4242","python","127.0.0.1","_work_item_kernel","1","7","Command line profiler metrics","dram__bytes_write.sum","Kbyte","524.29"
"1","4242","python","127.0.0.1","_merge_ragged_kernel","1","7","Command line profiler metrics","dram__bytes_read.sum","byte","1,048,576"
"1","4242","python","127.0.0.1","_merge_ragged_kernel","1","7","Command line profiler metrics","dram__bytes_write.sum","byte","262,144"
==PROF== Disconnected from process 4242
"""


def test_reads_each_kernels_bytes_in_bytes() -> None:
    values = parse_ncu_bytes(NCU_CSV)
    assert values["dram__bytes_read.sum"] == pytest.approx([33.55e6, 1048576.0])
    assert values["dram__bytes_write.sum"] == pytest.approx([524.29e3, 262144.0])


def test_rejects_an_unknown_unit() -> None:
    bad = NCU_CSV.replace('"Mbyte"', '"Mibyte"')
    with pytest.raises(ValueError, match="Mibyte"):
        parse_ncu_bytes(bad)
