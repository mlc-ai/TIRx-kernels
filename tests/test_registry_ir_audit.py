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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest import SkipTest

from tirx_kernels.low_level_ir import LowLevelIRFinding, LowLevelIRReport
from tirx_kernels.registry_audit import ConfigAuditResult, audit_registered_kernels, main
from tvm.script import tirx as T


@T.prim_func
def _legal_kernel():
    values = T.alloc_local((1,), "float32")
    values[0] = T.float32(1)


@T.prim_func
def _forbidden_load(source: T.Buffer((2,), "float32")):
    values = T.alloc_local((1,), "float32")
    T.evaluate(T.address_of(source[0]))
    values[0] = source[1]


_DEFAULT_GET_KERNEL = object()


def _get_legal_kernel(**_parameters):
    return _legal_kernel


def _module(name, configs, get_kernel=_DEFAULT_GET_KERNEL, **overrides):
    if get_kernel is _DEFAULT_GET_KERNEL:
        get_kernel = _get_legal_kernel
    values = {
        "__name__": f"test_kernels.{name}",
        "KERNEL_META": {"name": name, "category": "test", "compute_capability": 10},
        "CONFIGS": configs,
        "get_kernel": get_kernel,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_audit_calls_every_declared_config_without_the_label() -> None:
    calls = []

    def get_kernel(**parameters):
        calls.append(parameters)
        return _legal_kernel

    configs = [{"label": "small", "size": 1}, {"label": "large", "size": 4, "axes": (2, 2)}]
    report = audit_registered_kernels(
        {"example": _module("example", configs, get_kernel=get_kernel)}
    )

    assert report.ok
    assert report.complete_registry
    assert calls == [{"size": 1}, {"size": 4, "axes": (2, 2)}]
    assert report.counts()["audited_configs"] == len(configs)
    assert report.kernels[0].decision == "skip"
    assert report.kernels[0].complete
    assert [result.label for result in report.kernels[0].configs] == ["small", "large"]
    assert report.to_dict()["kernels"][0]["configs"][1]["parameters"]["axes"] == [2, 2]


def test_one_bad_function_rewrites_the_whole_multifunction_config() -> None:
    def get_kernel(**_parameters):
        return {"entry": _legal_kernel, "launches": (_legal_kernel, [_forbidden_load])}

    report = audit_registered_kernels(
        {"nested": _module("nested", [{"label": "case"}], get_kernel=get_kernel)}
    )

    assert not report.ok
    kernel = report.kernels[0]
    config = kernel.configs[0]
    assert kernel.complete
    assert kernel.decision == "rewrite"
    assert config.status == "violation"
    assert config.ir is not None
    assert len(config.ir.checked_functions) == 3
    finding = config.ir.violations[0]
    assert finding.kind == "buffer_load"
    assert finding.scope == "global"
    assert "launches" in finding.function

    machine = report.to_dict()
    assert machine["summary"]["forbidden_loads"] == 1
    assert machine["summary"]["address_only_loads"] == 1
    assert machine["kernels"][0]["configs"][0]["ir"]["counts"] == {
        "checked_functions": 3,
        "violations": 1,
        "tile_primitives": 0,
        "forbidden_loads": 1,
        "forbidden_stores": 0,
        "address_only_loads": 1,
    }


def test_get_kernel_errors_are_recorded_and_do_not_stop_later_configs() -> None:
    calls = []

    def get_kernel(mode):
        calls.append(mode)
        if mode == "skip":
            raise SkipTest("GPU unavailable")
        if mode == "error":
            raise RuntimeError("construction failed")
        return _legal_kernel

    configs = [
        {"label": "skipped", "mode": "skip"},
        {"label": "broken", "mode": "error"},
        {"label": "valid", "mode": "pass"},
    ]
    report = audit_registered_kernels({"errors": _module("errors", configs, get_kernel=get_kernel)})

    assert not report.ok
    assert calls == ["skip", "error", "pass"]
    assert [result.status for result in report.kernels[0].configs] == ["error", "error", "pass"]
    assert [result.errors[0].error_type for result in report.kernels[0].configs[:2]] == [
        "SkipTest",
        "RuntimeError",
    ]
    assert report.kernels[0].decision == "incomplete"
    assert report.counts()["error_configs"] == 2


def test_empty_kernel_return_is_an_error() -> None:
    report = audit_registered_kernels(
        {"empty": _module("empty", [{"label": "empty"}], get_kernel=lambda **_parameters: [])}
    )

    config = report.kernels[0].configs[0]
    assert not report.ok
    assert config.status == "error"
    assert config.errors[0].stage == "inspection"
    assert "TIRx PrimFunc" in config.errors[0].message
    assert report.kernels[0].decision == "incomplete"


def test_registry_module_and_config_integrity_errors_cannot_pass() -> None:
    registry = {
        "duplicate": _module("duplicate", [{"label": "same"}, {"label": "same"}]),
        "empty_configs": _module("empty_configs", []),
        "missing_get_kernel": _module("missing_get_kernel", [{"label": "case"}], get_kernel=None),
        "wrong_name": _module(
            "different_name",
            [{"label": "case"}],
            KERNEL_META={"name": "different_name", "category": "test", "compute_capability": 10},
        ),
    }

    report = audit_registered_kernels(registry)

    assert not report.ok
    assert {kernel.decision for kernel in report.kernels} == {"incomplete"}
    by_name = {kernel.name: kernel for kernel in report.kernels}
    assert all(result.status == "error" for result in by_name["duplicate"].configs)
    assert by_name["empty_configs"].declared_config_count == 0
    assert by_name["empty_configs"].audited_config_count == 0
    assert any(error.stage == "get_kernel" for error in by_name["missing_get_kernel"].errors)
    assert any("does not match" in error.message for error in by_name["wrong_name"].errors)


def test_filters_are_explicitly_marked_as_incomplete_registry_scope() -> None:
    registry = {
        "alpha": _module("alpha", [{"label": "one"}]),
        "beta": _module("beta", [{"label": "one"}, {"label": "two"}]),
    }

    report = audit_registered_kernels(registry, kernel="beta", config="two")

    assert report.ok
    assert not report.complete_registry
    assert report.discovered_kernel_count == len(registry)
    assert [kernel.name for kernel in report.kernels] == ["beta"]
    assert report.kernels[0].declared_config_count == 2
    assert report.kernels[0].audited_config_count == 1
    assert report.kernels[0].decision == "incomplete"
    assert report.to_dict()["scope"] == {
        "complete_registry": False,
        "kernel": "beta",
        "config": "two",
    }


def test_json_cli_is_deterministic_and_uses_contract_exit_codes() -> None:
    passing_registry = {
        "zeta": _module("zeta", [{"label": "second"}]),
        "alpha": _module("alpha", [{"label": "first"}]),
    }
    first_stdout = io.StringIO()
    second_stdout = io.StringIO()

    first_code = main(["--format", "json"], registry=passing_registry, stdout=first_stdout)
    second_code = main(["--format", "json"], registry=passing_registry, stdout=second_stdout)

    assert first_code == 0
    assert second_code == 0
    assert first_stdout.getvalue() == second_stdout.getvalue()
    payload = json.loads(first_stdout.getvalue())
    assert payload["ok"] is True
    assert [kernel["name"] for kernel in payload["kernels"]] == ["alpha", "zeta"]

    failing_stdout = io.StringIO()
    assert (
        main(
            ["--format", "text"],
            registry={
                "bad": _module(
                    "bad", [{"label": "case"}], get_kernel=lambda **_parameters: _forbidden_load
                )
            },
            stdout=failing_stdout,
        )
        == 1
    )
    assert "REWRITE" in failing_stdout.getvalue()

    stderr = io.StringIO()
    assert (
        main(
            ["--kernel", "missing"], registry=passing_registry, stdout=io.StringIO(), stderr=stderr
        )
        == 2
    )
    assert "was not found" in stderr.getvalue()


def test_machine_report_removes_process_local_source_name_addresses() -> None:
    finding = LowLevelIRFinding(
        function="root",
        kind="buffer_load",
        node_type="BufferLoad",
        scope="global",
        span="Span(SourceName(kernel.py, 0x1a2B), 4, 4, 1, 2)",
    )
    result = ConfigAuditResult(
        index=0,
        label="case",
        parameters={},
        status="violation",
        errors=(),
        ir=LowLevelIRReport(
            checked_functions=("root",), violations=(finding,), address_only_loads=()
        ),
    )

    span = result.to_dict()["ir"]["violations"][0]["span"]
    assert span == "Span(SourceName(kernel.py), 4, 4, 1, 2)"


def test_json_cli_can_write_a_machine_readable_report(tmp_path) -> None:
    output = tmp_path / "audit.json"
    registry = {"one": _module("one", [{"label": "case"}])}

    code = main(
        ["--format", "json", "--output", str(output)], registry=registry, stdout=io.StringIO()
    )

    assert code == 0
    assert json.loads(output.read_text())["summary"]["audited_configs"] == len(
        registry["one"].CONFIGS
    )
