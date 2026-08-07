# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from functools import cache
from pathlib import Path

import pytest
import torch

import tvm
from tirx_kernels.megakernel import moe
from tirx_kernels.megakernel.moe import IKET_EVENT_NAMES
from tvm.support.nvcc import compile_cuda
from tvm.tirx.cuda.iket import IketProfiler, IketProfileResult

COMMON_EVENTS = frozenset(
    {
        "count-and-sort",
        "group-gemm-down",
        "group-gemm-gate-up-silu",
        "init-etensor",
        "moe-align",
        "moe-gating",
        "topk-softmax",
    }
)
EXPECTED_EVENTS = {
    "static": COMMON_EVENTS | {"wait-etensor-init"},
    "dynamic": COMMON_EVENTS | {"fetch", "push"},
    "unfused": COMMON_EVENTS | {"wait-etensor-init"},
}
EXPECTED_EVENT_SETS = set(map(frozenset, EXPECTED_EVENTS.values()))


@cache
def _module(scheduler: str, enable_iket: bool) -> tvm.IRModule:
    mk = moe.MegaKernelMOE(moe.MEGAKERNEL_MOE_BENCH_CONFIG, world_size=1, enable_iket=enable_iket)
    mk._compile_batch_size = 1
    return mk.get_module(scheduler)


def _sources(root) -> str:
    pending = [root]
    sources = []
    while pending:
        module = pending.pop()
        pending.extend(module.imports)
        try:
            sources.append(module.inspect_source())
        except RuntimeError:
            pass
    return "\n".join(sources)


def _cuda_source(root) -> str:
    modules = root._collect_from_import_tree(  # pylint: disable=protected-access
        lambda module: module.kind == "cuda"
    )
    assert len(modules) == 1
    return modules[0].inspect_source("cuda")


def _require_sm100() -> None:
    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        pytest.skip("MegaKernelMOE IKET verification requires CUDA in PyTorch and TVM")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MegaKernelMOE IKET verification requires an SM100 GPU")


@pytest.mark.parametrize(
    ("scheduler", "num_params"), (("static", 20), ("dynamic", 22), ("unfused", 20))
)
def test_megakernel_moe_iket_annotations_and_plain_abi(scheduler, num_params) -> None:
    plain_func = _module(scheduler, False)["main"]
    iket_func = _module(scheduler, True)["main"]

    assert len(plain_func.params) == num_params
    assert len(iket_func.params) == num_params
    plain_script = plain_func.script(show_meta=False)
    iket_script = iket_func.script(show_meta=False)
    declarations = set(re.findall(r'T\.cuda\.iket\.range_push\("([^"]+)"', iket_script))
    assert declarations == EXPECTED_EVENTS[scheduler]
    assert declarations <= set(IKET_EVENT_NAMES)
    assert "iket" not in plain_script.lower()
    assert "profiler_buffer" not in plain_script
    assert "profiler_buffer" not in iket_script
    assert "clock64" not in iket_script


def test_megakernel_moe_public_runner_has_no_profiler_switch() -> None:
    assert "profiler_on" not in inspect.signature(moe.run_test).parameters
    assert "profiler_on" not in inspect.signature(moe.run_bench).parameters


def test_megakernel_moe_module_entry_uses_orchestrator_defaults(
    tmp_path, monkeypatch, capsys
) -> None:
    captured = {}

    def fake_run(main, **kwargs):
        captured.update(main=main, kwargs=kwargs)
        return IketProfileResult(
            output_dir=Path(kwargs["output_dir"]),
            postprocess=kwargs["postprocess"],
            json_traces=(tmp_path / "trace.json",),
            perfetto_traces=(tmp_path / "trace.pftrace",),
            html_reports=(tmp_path / "trace.html",),
        )

    monkeypatch.setattr(moe.iket, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["megakernel_moe"])
    moe.main()

    assert captured["main"].func is moe._profile_iket_workload
    assert captured["main"].args[0].scheduler == "dynamic"
    assert captured["main"].args[0].batch_size == 1
    assert captured["main"].args[0].repeat == 1
    assert captured["kwargs"] == {
        "output_dir": "/tmp/megakernel-moe-iket",
        "postprocess": "all",
        "clobber": True,
        "timeout": 600.0,
        "keep": False,
        "max_ts_cnt_per_warp": None,
    }
    output = capsys.readouterr().out
    assert "IKET output directory: /tmp/megakernel-moe-iket" in output
    assert f"IKET artifact: {tmp_path / 'trace.json'}" in output


@pytest.mark.parametrize("scheduler", ("static", "dynamic", "unfused"))
def test_megakernel_moe_official_iket_metadata(scheduler) -> None:
    _require_sm100()
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    executable = IketProfiler().compile(_module(scheduler, True), target=target)
    source = _sources(executable.mod)
    assert "__iket_meta_info" in source
    declaration_names = set(re.findall(r"__iket_evt_decl_([a-z0-9_]+)_\d+_attrs", source))
    assert declaration_names == {name.replace("-", "_") for name in EXPECTED_EVENTS[scheduler]}
    assert len(re.findall(r"__iket_evt_decl_[a-z0-9_]+_\d+_attrs", source)) == len(
        EXPECTED_EVENTS[scheduler]
    )

    helper_start = source.index("template <unsigned int EventId>")
    helper_end = source.index('extern "C" __global__', helper_start)
    helper = source[helper_start:helper_end]
    assert "activemask" not in helper
    assert "elect.sync" not in helper
    assert "__shfl" not in helper
    assert "st.weak.shared.u32 [%%r], %%t" in helper


def test_megakernel_moe_plain_dynamic_cubin_is_annotation_invariant() -> None:
    _require_sm100()
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    sources = []
    for enable_iket in (False, True):
        executable = tvm.compile(
            _module("dynamic", enable_iket), target=target, tir_pipeline="tirx"
        )
        source = _cuda_source(executable.mod)
        assert "iket" not in source.lower()
        sources.append(source)

    cubins = [
        compile_cuda(source, target_format="cubin", arch="sm_100a", compiler="nvrtc")
        for source in sources
    ]
    assert cubins[0] == cubins[1]


def test_megakernel_moe_external_trace_contract() -> None:
    trace_path = os.environ.get("TIRX_MEGAKERNEL_MOE_IKET_OFFICIAL_TRACE_JSON")
    if not trace_path:
        pytest.skip("set TIRX_MEGAKERNEL_MOE_IKET_OFFICIAL_TRACE_JSON to a run-iket trace")

    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    strings = trace["stringTable"]
    observed = []
    for launch in trace["launches"]:
        names = frozenset(strings[item["rangeNameIdx"]] for item in launch["ranges"])
        if names not in EXPECTED_EVENT_SETS:
            continue
        observed.append(names)
        assert all(item["startTs"] <= item["endTs"] for item in launch["ranges"])
        assert all(
            len(item["internalEvents"]) == 2
            and item["rangeType"] == 2
            and item["internalEvents"][0]["eventId"] != 31
            and item["internalEvents"][1]["eventId"] == 31
            and item["internalEvents"][0]["timestamp"] == item["startTs"]
            and item["internalEvents"][1]["timestamp"] == item["endTs"]
            for item in launch["ranges"]
        )

    assert observed
