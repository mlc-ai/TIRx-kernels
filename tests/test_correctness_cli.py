# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
from types import ModuleType
from unittest import SkipTest

import pytest

from tirx_kernels.test.__main__ import (
    CorrectnessSelectionError,
    main,
    run_correctness_suite,
    write_correctness_report,
)


def _module(name: str, outcomes: list[tuple[str, str]]) -> ModuleType:
    module = ModuleType(name)
    module.CONFIGS = [{"label": label, "outcome": outcome} for label, outcome in outcomes]

    def run_test(outcome: str) -> None:
        if outcome == "skip":
            raise SkipTest("reference unavailable")
        if outcome == "fail":
            raise RuntimeError("reference mismatch")

    module.run_test = run_test
    return module


def test_suite_reports_each_public_outcome_and_treats_skips_as_incomplete() -> None:
    registry = {
        "fixture": _module("fixture", [("pass", "pass"), ("skip", "skip"), ("fail", "fail")])
    }

    report = run_correctness_suite(registry)

    assert report["ok"] is False
    assert (report["passed"], report["skipped"], report["failed"]) == (1, 1, 1)
    assert report["selected_configs"] == report["declared_configs"] == 3
    assert [result["status"] for result in report["results"]] == ["PASS", "SKIP", "FAIL"]
    assert report["results"][1]["reason"] == "reference unavailable"
    assert report["results"][2]["error_type"] == "RuntimeError"


def test_filtered_all_pass_scope_is_successful_but_not_a_complete_registry_run() -> None:
    registry = {
        "first": _module("first", [("target", "pass"), ("other", "fail")]),
        "second": _module("second", [("target", "pass")]),
    }

    report = run_correctness_suite(registry, kernel_filter="first", config_filter="target")

    assert report["ok"] is True
    assert report["scope"]["complete_registry"] is False
    assert report["selected_kernels"] == 1
    assert report["selected_configs"] == report["passed"] == 1


@pytest.mark.parametrize(
    ("kernel_filter", "config_filter", "message"),
    (("missing", None, "was not found"), (None, "missing", "selected no declared")),
)
def test_absent_public_selection_is_rejected(kernel_filter, config_filter, message) -> None:
    registry = {"fixture": _module("fixture", [("present", "pass")])}

    with pytest.raises(CorrectnessSelectionError, match=message):
        run_correctness_suite(registry, kernel_filter=kernel_filter, config_filter=config_filter)


@pytest.mark.parametrize(
    "configs",
    (
        [],
        [{"outcome": "pass"}],
        [{"label": "same", "outcome": "pass"}, {"label": "same", "outcome": "pass"}],
    ),
)
def test_malformed_or_ambiguous_public_config_matrix_is_rejected(configs) -> None:
    module = _module("fixture", [("placeholder", "pass")])
    module.CONFIGS = configs

    with pytest.raises(CorrectnessSelectionError):
        run_correctness_suite({"fixture": module})


def test_json_evidence_round_trips_without_losing_result_details(tmp_path) -> None:
    report = run_correctness_suite({"fixture": _module("fixture", [("pass", "pass")])})

    output = write_correctness_report(report, tmp_path / "nested" / "correctness.json")

    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_cli_returns_failure_for_skip_and_writes_the_same_machine_readable_evidence(
    tmp_path,
) -> None:
    output = tmp_path / "correctness.json"
    stdout = io.StringIO()

    code = main(
        ["--kernel", "fixture", "--json", "--output", str(output)],
        registry={"fixture": _module("fixture", [("skipped", "skip")])},
        stdout=stdout,
    )

    assert code == 1
    emitted = json.loads(stdout.getvalue())
    assert emitted == json.loads(output.read_text(encoding="utf-8"))
    assert emitted["ok"] is False
    assert emitted["skipped"] == 1
