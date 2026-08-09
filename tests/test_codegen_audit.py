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

import hashlib
import io
import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from tirx_kernels.codegen_audit import (
    analyze_instruction_features,
    audit_registered_codegen,
    compare_codegen_manifests,
    main,
    parse_ptxas_verbose,
)
from tvm.script import tirx as T


def _observable_kernel():
    @T.prim_func
    def kernel():
        T.device_entry()
        _bx = T.cta_id([3])
        _tx = T.thread_id([32])
        pool = T.SMEMPool()
        _scratch = pool.alloc([32], "uint32")
        pool.commit()
        T.ptx.bar.sync(0, T.uint32(32))

    return kernel


def _registry(get_kernel=None):
    if get_kernel is None:

        def get_kernel(**_kwargs):
            return _observable_kernel()

    module = SimpleNamespace(
        __name__="example.audit_kernel",
        KERNEL_META={"name": "audit_kernel", "category": "example", "compute_capability": 10},
        CONFIGS=[{"label": "declared", "shape": 3}],
        get_kernel=get_kernel,
    )
    return {"audit_kernel": module}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_manifest() -> dict:
    ptx_counts = {
        "atomic": 1,
        "barrier": 2,
        "descriptor": 3,
        "direct_memory": 4,
        "fence": 5,
        "tcgen05": 6,
        "tma": 7,
    }
    sass_counts = {
        "atomic": 1,
        "barrier": 2,
        "descriptor_or_tma": 3,
        "direct_memory": 4,
        "tcgen05": 5,
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "ok": True,
        "errors": [],
        "kernel": {
            "name": "audit_kernel",
            "module": "example.audit_kernel",
            "metadata": {"name": "audit_kernel", "category": "example", "compute_capability": 10},
            "source_functions": ["root"],
        },
        "config": {"label": "declared", "shape": 3},
        "target": {
            "kind": "cuda",
            "arch": "sm_100a",
            "compute_capability": 10,
            "pipeline": "tirx",
            "tvm_version": "test",
            "tvm_python": "/opt/test/tvm/__init__.py",
        },
        "launches": [
            {
                "function": "kernel_kernel",
                "launch_index": 0,
                "grid": [3, 1, 1],
                "block": [128, 1, 1],
                "cluster": [2, 1, 1],
                "preferred_cluster": [2, 1, 1],
                "dynamic_shared_memory_bytes": 4096,
                "flags": {"cooperative_launch": False, "programmatic_dependent_launch": True},
                "parameter_tags": [
                    "blockIdx.x",
                    "threadIdx.x",
                    "clusterCtaIdx.x",
                    "preferredClusterCtaIdx.x",
                    "tirx.use_dyn_shared_memory",
                    "tirx.use_programtic_dependent_launch",
                ],
            }
        ],
        "resources": {
            "kernel_kernel": {
                "registers": 128,
                "barriers": 8,
                "stack_frame_bytes": 64,
                "cumulative_stack_size_bytes": 64,
                "spill_store_bytes": 32,
                "spill_load_bytes": 16,
                "memory_bytes": {"cmem[0]": 128},
            }
        },
        "instructions": {
            "cuda": {"tensor_map_type_occurrences": 2},
            "ptx": {
                "instruction_count": 100,
                "families": {
                    name: {"count": count, "mnemonics": {f"{name}.typed": count}}
                    for name, count in ptx_counts.items()
                },
                "cache_modifiers": {"count": 2, "values": {"L1::evict_last": 2}},
                "memory_order_qualifiers": {"count": 3, "values": {"relaxed": 2, "release": 1}},
            },
            "sass": {
                "instruction_count": 80,
                "families": {
                    name: {"count": count, "mnemonics": {f"{name}.TYPED": count}}
                    for name, count in sass_counts.items()
                },
            },
        },
        "tools": {},
        "artifacts": {},
        "run_artifacts": {},
    }


def test_instruction_report_classifies_generated_semantics() -> None:
    cuda = "CUtensorMap input_map; CUtensorMap output_map;"
    ptx = """
    ld.global.L1::no_allocate.acquire.u32 %r1, [%rd1];
    st.global.release.gpu.u32 [%rd2], %r1;
    mbarrier.init.shared.b64 [%r2], 1;
    fence.proxy.async.shared::cta;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r2], [%rd1];
    tcgen05.mma.cta_group::1.kind::f16 [%r3], %rd4, %rd5, %r6, 0;
    atom.release.gpu.global.add.u32 %r7, [%rd2], 1;
    """
    sass = """
    /*0000*/                   LDG.E R2, desc[UR4][R2.64] ;
    /*0010*/                   UMMA.16816 R4, R6 ;
    /*0020*/                   ATOMG.E.ADD R8, [R10], R12 ;
    /*0030*/                   BAR.SYNC 0x0 ;
    """

    report = analyze_instruction_features(cuda, ptx, sass)

    assert report["cuda"] == {"tensor_map_type_occurrences": 2}
    assert report["ptx"]["families"]["barrier"]["count"] == 1
    assert report["ptx"]["families"]["fence"]["count"] == 1
    assert report["ptx"]["families"]["tma"]["count"] == 1
    assert report["ptx"]["families"]["tcgen05"]["count"] == 1
    assert report["ptx"]["families"]["atomic"]["count"] == 1
    assert report["ptx"]["families"]["direct_memory"]["count"] == 2
    assert report["ptx"]["cache_modifiers"] == {"count": 1, "values": {"L1::no_allocate": 1}}
    assert report["ptx"]["memory_order_qualifiers"]["values"] == {"acquire": 1, "release": 2}
    assert report["sass"]["families"]["barrier"]["count"] == 1
    assert report["sass"]["families"]["tcgen05"]["count"] == 1
    assert report["sass"]["families"]["atomic"]["count"] == 1


def test_ptxas_report_preserves_per_entry_resources_and_spills() -> None:
    log = """
ptxas info    : Compiling entry function 'first_kernel' for 'sm_100a'
ptxas info    : Function properties for first_kernel
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 72 registers, used 3 barriers, 16 bytes cumulative stack size, 40 bytes cmem[0]
ptxas info    : Compiling entry function 'second_kernel' for 'sm_100a'
ptxas info    : Function properties for second_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, 8 bytes smem, 32 bytes cmem[0]
ptxas info    : Function properties for _Z24out_of_line_device_helperv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""

    resources = parse_ptxas_verbose(log)

    assert resources == {
        "first_kernel": {
            "memory_bytes": {"cmem[0]": 40},
            "stack_frame_bytes": 16,
            "spill_store_bytes": 8,
            "spill_load_bytes": 4,
            "registers": 72,
            "barriers": 3,
            "cumulative_stack_size_bytes": 16,
        },
        "second_kernel": {
            "memory_bytes": {"cmem[0]": 32, "smem": 8},
            "stack_frame_bytes": 0,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "registers": 24,
            "barriers": 0,
        },
    }


def test_registry_config_produces_complete_reproducible_artifact_set(tmp_path: Path) -> None:
    assert all(shutil.which(tool) for tool in ("nvcc", "ptxas", "nvdisasm"))
    output = tmp_path / "evidence"
    repeated_output = tmp_path / "repeated-evidence"

    report = audit_registered_codegen(
        "audit_kernel", "declared", output, registry=_registry(), arch="sm_100a"
    )
    repeated = audit_registered_codegen(
        "audit_kernel", "declared", repeated_output, registry=_registry(), arch="sm_100a"
    )

    assert report.ok
    assert repeated.ok
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == report.to_dict()
    assert repeated.to_dict() == manifest
    assert manifest["kernel"]["name"] == "audit_kernel"
    assert manifest["config"] == {"label": "declared", "shape": 3}
    assert manifest["target"]["arch"] == "sm_100a"
    assert manifest["launches"] == [
        {
            "function": "kernel_kernel",
            "launch_index": 0,
            "grid": [3, 1, 1],
            "block": [32, 1, 1],
            "cluster": [1, 1, 1],
            "preferred_cluster": [1, 1, 1],
            "dynamic_shared_memory_bytes": 128,
            "flags": {"cooperative_launch": False, "programmatic_dependent_launch": False},
            "parameter_tags": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"],
        }
    ]
    assert manifest["resources"]["kernel_kernel"]["registers"] > 0
    assert manifest["resources"]["kernel_kernel"]["spill_store_bytes"] == 0
    assert manifest["resources"]["kernel_kernel"]["spill_load_bytes"] == 0
    assert manifest["instructions"]["ptx"]["families"]["barrier"]["count"] >= 1
    assert manifest["instructions"]["sass"]["families"]["barrier"]["count"] >= 1

    expected = {"kernel.cu", "kernel.ptx", "kernel.cubin", "kernel.sass", "nvcc.txt", "ptxas.txt"}
    assert set(manifest["artifacts"]) == expected
    assert set(manifest["run_artifacts"]) == {"kernel.nvcc.ptx", "ptxas.raw.txt", "timing.json"}
    for name in expected:
        artifact = manifest["artifacts"][name]
        path = output / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _digest(path) == artifact["sha256"]


def test_unavailable_codegen_capability_leaves_incomplete_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    output = tmp_path / "evidence"

    report = audit_registered_codegen(
        "audit_kernel", "declared", output, registry=_registry(), arch="sm_100a"
    )

    assert not report.ok
    assert report.status == "incomplete"
    assert report.errors
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert manifest["ok"] is False
    assert not {"kernel.ptx", "kernel.cubin", "kernel.sass"} & set(manifest["artifacts"])


def test_public_kernel_construction_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    def fail(**_kwargs):
        raise RuntimeError("declared configuration cannot be constructed")

    stdout = io.StringIO()
    stderr = io.StringIO()
    output = tmp_path / "evidence"
    exit_code = main(
        [
            "--kernel",
            "audit_kernel",
            "--config",
            "declared",
            "--output",
            str(output),
            "--format",
            "json",
        ],
        registry=_registry(fail),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stderr.getvalue() == ""
    rendered = json.loads(stdout.getvalue())
    assert rendered["status"] == "incomplete"
    assert rendered["errors"][0]["stage"] == "get_kernel"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == rendered


def test_unknown_declared_label_is_a_selection_error(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    output = tmp_path / "evidence"

    exit_code = main(
        ["--kernel", "audit_kernel", "--config", "not-declared", "--output", str(output)],
        registry=_registry(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert "not-declared" in stderr.getvalue()
    assert not output.exists()


def test_manifest_comparison_accepts_resource_improvements_and_typed_mnemonic_changes() -> None:
    baseline = _comparison_manifest()
    candidate = deepcopy(baseline)
    resources = candidate["resources"]["kernel_kernel"]
    resources.update(
        {
            "registers": 120,
            "stack_frame_bytes": 48,
            "cumulative_stack_size_bytes": 48,
            "spill_store_bytes": 24,
            "spill_load_bytes": 8,
            "memory_bytes": {"cmem[0]": 96},
        }
    )
    candidate["instructions"]["ptx"]["instruction_count"] += 17
    candidate["instructions"]["ptx"]["families"]["direct_memory"].update(
        {"count": 6, "mnemonics": {"ld.shared.v2.f32": 6}}
    )
    candidate["instructions"]["sass"]["instruction_count"] += 9
    candidate["instructions"]["sass"]["families"]["direct_memory"].update(
        {"count": 6, "mnemonics": {"LDS.U.128": 6}}
    )
    candidate["instructions"]["cuda"]["tensor_map_type_occurrences"] = 1

    report = compare_codegen_manifests(baseline, candidate)
    rendered = report.to_dict()

    assert report.ok
    assert rendered["status"] == "pass"
    assert rendered["failures"] == []
    assert {
        change["path"]
        for change in rendered["changes"]
        if change["rule"] in {"not_increased", "not_new_or_increased"}
    } == {
        "/resources/kernel_kernel/allocated_registers_per_warp",
        "/resources/kernel_kernel/stack_frame_bytes",
        "/resources/kernel_kernel/cumulative_stack_size_bytes",
        "/resources/kernel_kernel/spill_store_bytes",
        "/resources/kernel_kernel/spill_load_bytes",
        "/resources/kernel_kernel/memory_bytes/cmem[0]",
    }
    assert {
        change["path"]
        for change in rendered["changes"]
        if change["rule"] == "observed_nonnegative_integer"
    } == {
        "/instructions/cuda/tensor_map_type_occurrences",
        "/instructions/ptx/families/direct_memory/count",
        "/instructions/sass/families/direct_memory/count",
        "/resources/kernel_kernel/registers",
    }


def test_manifest_comparison_gates_sm100_physical_register_allocation() -> None:
    cases = (
        (167, 168, True, 5376, 5376),
        (128, 129, False, 4096, 4352),
        (168, 169, False, 5376, 5632),
    )
    for (
        baseline_registers,
        candidate_registers,
        expected_ok,
        baseline_alloc,
        candidate_alloc,
    ) in cases:
        baseline = _comparison_manifest()
        candidate = deepcopy(baseline)
        baseline["resources"]["kernel_kernel"]["registers"] = baseline_registers
        candidate["resources"]["kernel_kernel"]["registers"] = candidate_registers

        report = compare_codegen_manifests(baseline, candidate).to_dict()
        allocation = next(
            check
            for check in report["checks"]
            if check["path"] == "/resources/kernel_kernel/allocated_registers_per_warp"
        )
        raw = next(
            check
            for check in report["checks"]
            if check["path"] == "/resources/kernel_kernel/registers"
        )

        assert report["ok"] is expected_ok
        assert allocation["baseline"] == baseline_alloc
        assert allocation["candidate"] == candidate_alloc
        assert allocation["status"] == ("pass" if expected_ok else "fail")
        assert raw["rule"] == "observed_nonnegative_integer"
        assert raw["status"] == "pass"


def test_manifest_comparison_hard_fails_observable_codegen_regressions() -> None:
    def set_launch(field, value):
        return lambda manifest: manifest["launches"][0].__setitem__(field, value)

    def increase_resource(field):
        return lambda manifest: manifest["resources"]["kernel_kernel"].__setitem__(
            field, manifest["resources"]["kernel_kernel"][field] + 1
        )

    regressions = [
        ("/target/arch", lambda value: value["target"].__setitem__("arch", "sm_90a")),
        ("/launches/kernel_kernel/0/grid", set_launch("grid", [4, 1, 1])),
        ("/launches/kernel_kernel/0/block", set_launch("block", [256, 1, 1])),
        ("/launches/kernel_kernel/0/cluster", set_launch("cluster", [1, 1, 1])),
        ("/launches/kernel_kernel/0/preferred_cluster", set_launch("preferred_cluster", [1, 1, 1])),
        (
            "/launches/kernel_kernel/0/dynamic_shared_memory_bytes",
            set_launch("dynamic_shared_memory_bytes", 8192),
        ),
        (
            "/launches/kernel_kernel/0/flags",
            set_launch(
                "flags", {"cooperative_launch": True, "programmatic_dependent_launch": True}
            ),
        ),
        ("/resources/kernel_kernel/barriers", increase_resource("barriers")),
        ("/resources/kernel_kernel/allocated_registers_per_warp", increase_resource("registers")),
        ("/resources/kernel_kernel/stack_frame_bytes", increase_resource("stack_frame_bytes")),
        (
            "/resources/kernel_kernel/cumulative_stack_size_bytes",
            increase_resource("cumulative_stack_size_bytes"),
        ),
        ("/resources/kernel_kernel/spill_store_bytes", increase_resource("spill_store_bytes")),
        ("/resources/kernel_kernel/spill_load_bytes", increase_resource("spill_load_bytes")),
        (
            "/resources/kernel_kernel/memory_bytes/smem",
            lambda value: value["resources"]["kernel_kernel"]["memory_bytes"].__setitem__(
                "smem", 1
            ),
        ),
        *[
            (
                f"/instructions/ptx/families/{family}/count",
                lambda value, family=family: value["instructions"]["ptx"]["families"][
                    family
                ].__setitem__(
                    "count", value["instructions"]["ptx"]["families"][family]["count"] + 1
                ),
            )
            for family in ("atomic", "barrier", "descriptor", "fence", "tcgen05", "tma")
        ],
        (
            "/instructions/ptx/cache_modifiers/values",
            lambda value: value["instructions"]["ptx"]["cache_modifiers"].__setitem__(
                "values", {"cg": 2}
            ),
        ),
        (
            "/instructions/ptx/memory_order_qualifiers/values",
            lambda value: value["instructions"]["ptx"]["memory_order_qualifiers"].__setitem__(
                "values", {"acquire": 2, "release": 1}
            ),
        ),
        (
            "/instructions/sass/families/tcgen05/count",
            lambda value: value["instructions"]["sass"]["families"]["tcgen05"].__setitem__(
                "count", 6
            ),
        ),
    ]

    for expected_failure, mutate in regressions:
        baseline = _comparison_manifest()
        candidate = deepcopy(baseline)
        mutate(candidate)

        report = compare_codegen_manifests(baseline, candidate).to_dict()

        assert report["status"] == "fail"
        assert expected_failure in {failure["path"] for failure in report["failures"]}


def test_manifest_comparison_rejects_invalid_or_mismatched_evidence() -> None:
    invalid_cases = [
        ("/schema_version", lambda value: value.__setitem__("schema_version", 2)),
        (
            "/status",
            lambda value: (
                value.__setitem__("status", "incomplete"),
                value.__setitem__("ok", False),
            ),
        ),
        ("/kernel", lambda value: value["kernel"].__setitem__("name", "other_kernel")),
        ("/config", lambda value: value["config"].__setitem__("shape", 4)),
        (
            "/instructions/ptx/families/keys",
            lambda value: value["instructions"]["ptx"]["families"].pop("tma"),
        ),
        (
            "/resources/kernel_kernel/spill_load_bytes",
            lambda value: value["resources"]["kernel_kernel"].pop("spill_load_bytes"),
        ),
        (
            "/instructions/sass/families/direct_memory/count",
            lambda value: value["instructions"]["sass"]["families"]["direct_memory"].__setitem__(
                "count", -1
            ),
        ),
    ]

    for expected_failure, mutate in invalid_cases:
        baseline = _comparison_manifest()
        candidate = deepcopy(baseline)
        mutate(candidate)

        report = compare_codegen_manifests(baseline, candidate).to_dict()

        assert not report["ok"]
        assert expected_failure in {failure["path"] for failure in report["failures"]}


def test_comparison_cli_writes_deterministic_json_and_returns_gate_status(tmp_path: Path) -> None:
    baseline = _comparison_manifest()
    candidate = deepcopy(baseline)
    candidate["resources"]["kernel_kernel"]["spill_load_bytes"] = 8
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    outputs = [tmp_path / "comparison-1.json", tmp_path / "comparison-2.json"]
    rendered_outputs = []
    for output in outputs:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output),
                "--format",
                "json",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        assert exit_code == 0
        assert stderr.getvalue() == ""
        rendered_outputs.append(json.loads(stdout.getvalue()))

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert rendered_outputs[0] == rendered_outputs[1]
    assert rendered_outputs[0]["status"] == "pass"

    regressed = deepcopy(candidate)
    regressed["resources"]["kernel_kernel"]["registers"] = 129
    regressed_path = tmp_path / "regressed.json"
    regressed_path.write_text(json.dumps(regressed), encoding="utf-8")
    failed_output = tmp_path / "failed-comparison.json"
    exit_code = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(regressed_path),
            "--output",
            str(failed_output),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 1
    assert json.loads(failed_output.read_text(encoding="utf-8"))["status"] == "fail"
