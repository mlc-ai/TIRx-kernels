# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CPU checks for explicit, frozen Thor source-reference selection."""

import importlib
import importlib.metadata
import json
import subprocess
import sysconfig
from types import SimpleNamespace
from unittest import SkipTest

import pytest

from tirx_kernels import reference_requirements as variants


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test initialized CUDA"))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def frozen_variant(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "csrc/jit/device_runtime.hpp": "int get_arch_major() { return get_arch_pair().first; }\n",
        "deep_gemm/include/deep_gemm/kernel.cuh": "// frozen device program\n",
        "deep_gemm/__init__.py": "# frozen Python API\n",
        "setup.py": "# frozen build\n",
        "third-party/tilelang_ops/utils.py": "# frozen optional helper\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "frozen",
    )
    policy = {**variants.VARIANTS["deep-gemm"], "revision": git(root, "rev-parse", "HEAD")}
    monkeypatch.setitem(variants.VARIANTS, "deep-gemm", policy)
    git(root, "remote", "add", "origin", policy["url"])
    host = root / "csrc/jit/device_runtime.hpp"
    host.write_text(
        variants.adapted_text("deep-gemm", "csrc/jit/device_runtime.hpp", host.read_text())
    )
    for name in ("cute", "cutlass"):
        target = root / "third-party/cutlass/include" / name
        target.mkdir(parents=True)
        (root / "deep_gemm/include" / name).symlink_to(target, target_is_directory=True)
    extension = root / "deep_gemm/_C.test.so"
    extension.write_bytes(b"fixture extension bytes")
    log = root / "build.log"
    log.write_text(
        "c++ -c csrc/python_api.cpp -std=c++17 -O3\n"
        "c++ -shared python_api.o -o deep_gemm/_C.fixture.so\n"
    )
    ninja = root / "build.ninja"
    ninja.write_text("cflags = -O3")
    manifest = {
        "schema": 1,
        "name": "deep-gemm",
        "cuda_arch": "sm_110a",
        "root": str(root),
        "source_revision": policy["revision"],
        "source_url": policy["url"],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": importlib.metadata.version("torch"),
        "source_inventory": variants.source_inventory("deep-gemm", root),
        "extension": {
            "relative_path": "deep_gemm/_C.test.so",
            "sha256": variants.sha256(extension),
        },
        "build": {
            "log": str(log),
            "log_sha256": variants.sha256(log),
            "ninja_files": [{"path": str(ninja), "sha256": variants.sha256(ninja)}],
            "jit_target": "actual_device_arch",
            "compiler_commands": variants.deepgemm_build_commands(log.read_text()),
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(variants, "_loaded_manifests", {})
    return root, path, manifest


def test_manifest_accepts_only_frozen_host_adaptation(frozen_variant):
    root, path, manifest = frozen_variant
    assert variants.validate_variant("deep-gemm", path) == manifest
    assert manifest["source_inventory"]["host_patch_files"] == ["csrc/jit/device_runtime.hpp"]
    assert len(manifest["source_inventory"]["device_file_sha256"]) == 1
    assert manifest["source_inventory"]["omitted_files"] == []


@pytest.mark.parametrize(
    "relative",
    [
        "deep_gemm/include/deep_gemm/kernel.cuh",
        "deep_gemm/__init__.py",
        "setup.py",
        "csrc/jit/device_runtime.hpp",
    ],
)
def test_source_mutations_rejected(frozen_variant, relative):
    root, path, _ = frozen_variant
    with (root / relative).open("a") as stream:
        stream.write("\n// mutation\n")
    with pytest.raises(RuntimeError, match="unexpected source change"):
        variants.validate_variant("deep-gemm", path)


@pytest.mark.parametrize("relative", ["deep_gemm/_C.test.so", "build.ninja", "build.log"])
def test_build_mutations_rejected(frozen_variant, relative):
    root, path, _ = frozen_variant
    with (root / relative).open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(RuntimeError, match=r"hash|changed"):
        variants.validate_variant("deep-gemm", path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_revision", "0" * 40),
        ("source_url", "wrong"),
        ("cuda_arch", "sm_100a"),
        ("torch_version", "0.0"),
        ("python_soabi", "wrong"),
    ],
)
def test_identity_mutations_rejected(frozen_variant, key, value):
    _, path, manifest = frozen_variant
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        variants.validate_variant("deep-gemm", path)


def test_legacy_omissions_require_explicit_record(frozen_variant):
    root, path, manifest = frozen_variant
    (root / "third-party/tilelang_ops/utils.py").unlink()
    with pytest.raises(RuntimeError, match="missing frozen source file"):
        variants.validate_variant("deep-gemm", path)
    inventory = variants.source_inventory("deep-gemm", root, allow_legacy_omissions=True)
    assert inventory["omitted_files"] == ["third-party/tilelang_ops/utils.py"]
    manifest["source_inventory"] = inventory
    path.write_text(json.dumps(manifest))
    assert variants.validate_variant("deep-gemm", path)["source_inventory"] == inventory


def test_thor_requires_explicit_manifest(monkeypatch):
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.delenv("TIRX_DEEP_GEMM_VARIANT_MANIFEST", raising=False)
    with pytest.raises(RuntimeError, match="requires TIRX_DEEP_GEMM_VARIANT_MANIFEST"):
        variants.load_reference("deep-gemm")


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("name", ["deep-gemm", "flash-mla"])
def test_native_import_stays_native(monkeypatch, arch, name):
    policy = variants.VARIANTS[name]
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", arch)
    monkeypatch.delenv(policy["environment"], raising=False)
    module = object()
    imports = []

    def import_module(import_name):
        imports.append(import_name)
        return module

    monkeypatch.setattr(variants.importlib, "import_module", import_module)
    monkeypatch.setattr(
        variants, "validate_variant", lambda *_: pytest.fail("native import checked a Thor build")
    )
    assert variants.load_reference(name) is module
    assert imports == [policy["import"]]
    monkeypatch.setenv(policy["environment"], "not-valid-on-native")
    with pytest.raises(RuntimeError, match="only valid for sm_110a"):
        variants.load_reference(name)
    assert imports == [policy["import"]]


def test_import_path_and_loaded_extension_must_match(frozen_variant, monkeypatch):
    root, path, _ = frozen_variant
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setenv("TIRX_DEEP_GEMM_VARIANT_MANIFEST", str(path))
    monkeypatch.setattr(
        variants.importlib.util, "find_spec", lambda _: SimpleNamespace(origin="/wrong/__init__.py")
    )
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        variants.load_reference("deep-gemm")
    monkeypatch.setattr(
        variants.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(root / "deep_gemm/__init__.py")),
    )
    extension = SimpleNamespace(__file__="/wrong/_C.so")
    module = SimpleNamespace()
    monkeypatch.setattr(
        variants.importlib,
        "import_module",
        lambda name: extension if name.endswith("._C") else module,
    )
    with pytest.raises(RuntimeError, match="different extension"):
        variants.load_reference("deep-gemm")
    extension.__file__ = str(root / "deep_gemm/_C.test.so")
    assert variants.load_reference("deep-gemm") is module
    assert variants.reference_provenance("deep-gemm")["manifest_sha256"] == variants.sha256(path)
    monkeypatch.setenv("TIRX_DEEP_GEMM_VARIANT_MANIFEST", str(root / "other.json"))
    with pytest.raises(RuntimeError, match="cannot change"):
        variants.load_reference("deep-gemm")


def test_runtime_git_requirement_uses_selected_flashmla_source(tmp_path, monkeypatch):
    from tirx_kernels.reference_requirements import unmet_reference_requirements

    root = tmp_path / "flash-mla"
    root.mkdir()
    (root / "flash_mla").mkdir()
    (root / "flash_mla/__init__.py").write_text("# fixture\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "frozen",
    )
    policy = variants.VARIANTS["flash-mla"]
    git(root, "remote", "add", "origin", policy["url"])
    monkeypatch.syspath_prepend(str(root))
    requirement = (
        {
            "package": "flash-mla",
            "import": "flash_mla",
            "git": {"url": policy["url"], "commit": git(root, "rev-parse", "HEAD")},
        },
    )
    assert unmet_reference_requirements(requirement) == ()
    requirement[0]["git"]["commit"] = "0" * 40
    assert unmet_reference_requirements(requirement)


def test_megamoe_thor_rejects_multi_gpu_before_allocation(monkeypatch):
    import torch

    from tirx_kernels.deepgemm import sm100_fp8_fp4_mega_moe as module
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe import data
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe.spec import MegaMoeConfig

    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setattr(
        torch.cuda, "device_count", lambda: pytest.fail("GPU probe before TP1 guard")
    )
    assert module._make_config().num_processes == 1
    with pytest.raises(SkipTest, match="only num_processes=1"):
        module._make_config(num_processes=2)
    with pytest.raises(SkipTest, match="only num_processes=1"):
        data._run_distributed(MegaMoeConfig(num_processes=2), "test")
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_100a")
    assert module._make_config(num_processes=2).num_processes == 2


def test_thor_gemm_keeps_math_and_strict_source_checks(monkeypatch):
    import torch

    from tirx_kernels.deepgemm import fp8_gemm_1d1d as module
    from tirx_kernels.deepgemm._sm100_fp8_fp4_gemm_1d1d import data as helpers

    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    data = dict(
        d=torch.zeros(1),
        c=None,
        ref=torch.ones(1),
        a_dtype="fp8",
        b_dtype="fp8",
        M=1,
        N=1,
        K=1,
        major_a="K",
        major_b="K",
        cd_dtype="bf16",
        accumulate=False,
    )
    monkeypatch.setattr(module, "prepare_data", lambda **kwargs: data)
    monkeypatch.setattr(module, "_tirx_launch", lambda *_: lambda: data["d"].fill_(1))
    thresholds = []
    original = helpers.assert_within_threshold

    def check(diff, *args, **kwargs):
        thresholds.append(kwargs.get("threshold"))
        return original(diff, *args, **kwargs)

    monkeypatch.setattr(helpers, "assert_within_threshold", check)
    monkeypatch.setattr(helpers, "deepgemm_launch_normal", lambda _: (None, torch.ones(1)))
    module.run_test()
    assert thresholds == [helpers.max_diff_threshold("fp8", "fp8"), None]
    monkeypatch.setattr(helpers, "deepgemm_launch_normal", lambda _: (None, torch.full((1,), 0.9)))
    with pytest.raises(AssertionError):
        module.run_test()


@pytest.mark.parametrize("output_index", [0, 1, 2])
def test_flashmla_prefill_source_gate_checks_every_output(monkeypatch, output_index):
    import torch

    from tirx_kernels.flashmla.utils import _flashmla_bench as helper

    tensors = tuple(torch.ones(2) for _ in range(3))
    case = dict(zip(("out", "max_logits", "lse"), tensors, strict=True))
    source = [value.clone() for value in tensors]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(helper, "run_flashmla_sparse_prefill_outputs", lambda _: source)
    helper.validate_flashmla_sparse_prefill(case, tensors, output_rtol=3.01 / 128)
    source[output_index][0] = 0
    with pytest.raises(AssertionError):
        helper.validate_flashmla_sparse_prefill(case, tensors, output_rtol=3.01 / 128)


def test_flashmla_host_patch_preserves_translation_unit_list():
    source = """subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])
flags = ["-gencode", "arch=compute_100f,code=sm_100f"]
sources = ["csrc/sm100/fwd.cu", "csrc/smxx/combine.cu"]
"""
    adapted = variants.adapted_text("flash-mla", "setup.py", source)
    assert 'sources = ["csrc/sm100/fwd.cu", "csrc/smxx/combine.cu"]' in adapted
    assert "arch=compute_110a,code=sm_110a" in adapted
    assert "subprocess.run" not in adapted
    with pytest.raises(RuntimeError, match="no longer matches"):
        variants.adapted_text("flash-mla", "setup.py", adapted)


def test_deepgemm_rejects_failed_or_missing_build_commands():
    with pytest.raises(RuntimeError, match="actual host compile and link commands"):
        variants.deepgemm_build_commands("RuntimeError: Error compiling objects for extension")
