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

import copy
import io
import json
from types import SimpleNamespace

import pytest

import tvm
from tirx_kernels.registry_manifest import (
    RegistryManifestCaptureError,
    RegistryManifestValidationError,
    capture_registry_manifest,
    compare_registry_manifests,
    main,
    render_registry_manifest,
    validate_registry_manifest,
)
from tvm.script import tirx as T


@T.prim_func
def _first_kernel():
    value = T.alloc_local((1,), "float32")
    value[0] = T.float32(1)


@T.prim_func
def _second_kernel():
    value = T.alloc_local((1,), "float32")
    value[0] = T.float32(2)


def _module(
    name: str, configs: list[dict], *, get_kernel=_first_kernel, metadata: dict | None = None
):
    if get_kernel is _first_kernel or get_kernel is _second_kernel:

        def public_get_kernel(**_parameters):
            return get_kernel

    else:
        public_get_kernel = get_kernel
    return SimpleNamespace(
        KERNEL_META=metadata or {"name": name, "category": "test", "compute_capability": 10},
        CONFIGS=configs,
        get_kernel=public_get_kernel,
    )


def _single_kernel_manifest(
    *, metadata: dict | None = None, configs: list[dict] | None = None, get_kernel=_first_kernel
):
    return capture_registry_manifest(
        {
            "example": _module(
                "example",
                configs or [{"label": "case", "size": 1}],
                metadata=metadata,
                get_kernel=get_kernel,
            )
        }
    )


def test_capture_records_every_public_config_and_pre_lowering_return_shape() -> None:
    calls = []

    def nested_return(**parameters):
        calls.append(parameters)
        return {
            "entry": _first_kernel,
            "launches": (
                tvm.IRModule({"zeta": _second_kernel, "alpha": _first_kernel}),
                [_second_kernel],
            ),
        }

    registry = {
        "zeta": _module(
            "zeta",
            [{"label": "nested", "axes": (2, 3), "enabled": True}],
            get_kernel=nested_return,
            metadata={
                "name": "zeta",
                "category": "test",
                "compute_capability": 10,
                "description": "public description",
                "_build_cache": object(),
            },
        ),
        "alpha": _module("alpha", [{"label": "large", "size": 4}, {"label": "small", "size": 1}]),
    }

    manifest = capture_registry_manifest(registry)

    assert [kernel["name"] for kernel in manifest["kernels"]] == sorted(registry)
    assert manifest["summary"]["kernels"] == len(registry)
    assert manifest["summary"]["configs"] == sum(
        len(module.CONFIGS) for module in registry.values()
    )
    zeta = next(kernel for kernel in manifest["kernels"] if kernel["name"] == "zeta")
    assert zeta["metadata"] == {
        "name": "zeta",
        "category": "test",
        "compute_capability": 10,
        "description": "public description",
    }
    assert calls == [{"axes": (2, 3), "enabled": True}]
    assert zeta["configs"][0]["parameters"] == {"axes": [2, 3], "enabled": True}
    assert zeta["configs"][0]["result"] == {
        "function_count": 4,
        "container": {
            "kind": "mapping",
            "entries": [
                {"key": "entry", "container": {"kind": "prim_func"}},
                {
                    "key": "launches",
                    "container": {
                        "kind": "tuple",
                        "items": [
                            {"kind": "ir_module", "globals": ["alpha", "zeta"]},
                            {"kind": "list", "items": [{"kind": "prim_func"}]},
                        ],
                    },
                },
            ],
        },
    }


def test_capture_is_deterministic_without_treating_declaration_order_as_identity() -> None:
    first_registry = {
        "zeta": _module("zeta", [{"label": "two", "size": 2}]),
        "alpha": _module("alpha", [{"label": "second", "size": 2}, {"label": "first", "size": 1}]),
    }
    second_registry = {
        "alpha": _module("alpha", [{"label": "first", "size": 1}, {"label": "second", "size": 2}]),
        "zeta": _module("zeta", [{"label": "two", "size": 2}]),
    }

    first = capture_registry_manifest(first_registry)
    second = capture_registry_manifest(second_registry)

    assert render_registry_manifest(first) == render_registry_manifest(second)
    assert compare_registry_manifests(first, second).compatible


@pytest.mark.parametrize(
    ("module", "error_type", "message"),
    [
        (
            _module("example", [{"label": "same"}, {"label": "same"}]),
            RegistryManifestValidationError,
            "labels must be unique",
        ),
        (
            _module("example", [{"label": "bad", "value": object()}]),
            RegistryManifestValidationError,
            "not JSON serializable",
        ),
        (
            _module(
                "different",
                [{"label": "case"}],
                metadata={"name": "different", "category": "test", "compute_capability": 10},
            ),
            RegistryManifestValidationError,
            "must match registry name",
        ),
        (
            _module("example", [{"label": "case"}], get_kernel=lambda **_parameters: []),
            RegistryManifestValidationError,
            "at least one TIRx PrimFunc",
        ),
        (
            _module("example", [{"label": "case"}], get_kernel=lambda **_parameters: 1),
            RegistryManifestValidationError,
            "expected a TIRx PrimFunc",
        ),
    ],
)
def test_capture_rejects_invalid_public_registry_contracts(module, error_type, message) -> None:
    with pytest.raises(error_type, match=message):
        capture_registry_manifest({"example": module})


def test_capture_reports_public_get_kernel_failure() -> None:
    def cannot_construct(**_parameters):
        raise RuntimeError("construction failed")

    with pytest.raises(RegistryManifestCaptureError, match="construction failed"):
        capture_registry_manifest(
            {"example": _module("example", [{"label": "case"}], get_kernel=cannot_construct)}
        )


def test_compare_reports_missing_and_added_public_identities() -> None:
    before = capture_registry_manifest(
        {
            "alpha": _module("alpha", [{"label": "one"}, {"label": "two"}]),
            "beta": _module("beta", [{"label": "case"}]),
        }
    )
    after = capture_registry_manifest(
        {
            "alpha": _module("alpha", [{"label": "two"}, {"label": "three"}]),
            "gamma": _module("gamma", [{"label": "case"}]),
        }
    )

    report = compare_registry_manifests(before, after)

    assert not report.compatible
    assert {difference.code for difference in report.differences} == {
        "kernel_missing",
        "kernel_added",
        "config_missing",
        "config_added",
    }


@pytest.mark.parametrize(
    ("after", "expected_codes"),
    [
        (
            _single_kernel_manifest(
                metadata={
                    "name": "example",
                    "category": "test",
                    "compute_capability": 10,
                    "description": "changed",
                }
            ),
            {"metadata_changed"},
        ),
        (_single_kernel_manifest(configs=[{"label": "case", "size": 2}]), {"parameters_changed"}),
        (
            _single_kernel_manifest(
                get_kernel=lambda **_parameters: [_first_kernel, _second_kernel]
            ),
            {"function_count_changed", "container_shape_changed"},
        ),
        (
            _single_kernel_manifest(get_kernel=lambda **_parameters: [_first_kernel]),
            {"container_shape_changed"},
        ),
    ],
)
def test_compare_rejects_public_contract_changes(after, expected_codes) -> None:
    before = _single_kernel_manifest()

    report = compare_registry_manifests(before, after)

    assert not report.compatible
    assert {difference.code for difference in report.differences} == expected_codes


@pytest.mark.parametrize("duplicate", ["kernel", "config"])
def test_serialized_manifest_rejects_duplicate_public_identities(duplicate) -> None:
    manifest = _single_kernel_manifest()
    if duplicate == "kernel":
        manifest["kernels"].append(copy.deepcopy(manifest["kernels"][0]))
    else:
        manifest["kernels"][0]["configs"].append(
            copy.deepcopy(manifest["kernels"][0]["configs"][0])
        )

    with pytest.raises(RegistryManifestValidationError, match="must be unique"):
        validate_registry_manifest(manifest)


def test_serialized_manifest_rejects_function_count_inconsistent_with_container() -> None:
    manifest = _single_kernel_manifest()
    manifest["kernels"][0]["configs"][0]["result"]["function_count"] += 1

    with pytest.raises(RegistryManifestValidationError, match="container describes"):
        validate_registry_manifest(manifest)


def test_export_cli_emits_deterministic_machine_readable_json() -> None:
    registry = {
        "example": _module("example", [{"label": "two", "size": 2}, {"label": "one", "size": 1}])
    }
    first_stdout = io.StringIO()
    second_stdout = io.StringIO()

    assert main(["export"], registry=registry, stdout=first_stdout) == 0
    assert main(["export"], registry=registry, stdout=second_stdout) == 0

    assert first_stdout.getvalue() == second_stdout.getvalue()
    payload = json.loads(first_stdout.getvalue())
    assert payload == capture_registry_manifest(registry)


def test_compare_cli_uses_compatibility_and_invalid_artifact_exit_codes(tmp_path) -> None:
    before = _single_kernel_manifest()
    incompatible = _single_kernel_manifest(configs=[{"label": "case", "size": 2}])
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    invalid_path = tmp_path / "invalid.json"
    before_path.write_text(render_registry_manifest(before))
    after_path.write_text(render_registry_manifest(incompatible))
    invalid_path.write_text("{}")

    output = io.StringIO()
    code = main(["compare", str(before_path), str(after_path), "--format", "json"], stdout=output)

    assert code == 1
    report = json.loads(output.getvalue())
    assert report["compatible"] is False
    assert [difference["code"] for difference in report["differences"]] == ["parameters_changed"]

    stderr = io.StringIO()
    assert (
        main(["compare", str(before_path), str(invalid_path)], stdout=io.StringIO(), stderr=stderr)
        == 2
    )
    assert "ERROR:" in stderr.getvalue()
