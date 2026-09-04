# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MSA must select Thor's compiler spelling before CuTe captures its target."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from tirx_kernels.msa.utils import _msa_bench as msa


@pytest.fixture
def pinned_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(msa, "_MSA_THOR_TARGET", None)
    monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", "sm_110a")
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    for name in tuple(sys.modules):
        if name == "cutlass" or name.startswith("cutlass."):
            monkeypatch.delitem(sys.modules, name)
    for package, version in (("nvidia-cutlass-dsl", "4.5.3"), ("quack-kernels", "0.5.0")):
        directory = tmp_path / f"{package.replace('-', '_')}-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(f"Name: {package}\nVersion: {version}\n")
    files = {
        "cuda/__init__.py": "",
        "cuda/bindings/__init__.py": "",
        "cuda/bindings/driver.py": (
            "def cuInit(*args):\n"
            "    raise AssertionError('CPU metadata tests must not initialize CUDA')\n"
        ),
        # Like real CuTe 4.5, importing the package captures the environment.
        "cutlass/__init__.py": "import os\ncaptured_target = os.environ.get('CUTE_DSL_ARCH')\n",
        "cutlass/base_dsl/__init__.py": "",
        "cutlass/base_dsl/arch.py": (
            "import os\n"
            "class Arch:\n"
            "    @staticmethod\n"
            "    def from_string(value):\n"
            "        result = Arch()\n"
            "        result.value = (os.environ.get('TEST_MSA_CANONICAL', 'sm_101a')\n"
            "                        if value in ('sm_110a', 'sm_101a') else value)\n"
            "        return result\n"
            "    def to_string(self):\n"
            "        return self.value\n"
        ),
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return [str(tmp_path)]


@pytest.mark.parametrize("requested", [None, "sm_110a", "sm_101a"])
@pytest.mark.parametrize("canonical", ["sm_101a", "sm_110a"])
def test_target_is_resolved_before_parent_import(pinned_prefix, monkeypatch, requested, canonical):
    if requested is not None:
        monkeypatch.setenv("CUTE_DSL_ARCH", requested)
    monkeypatch.setenv("TEST_MSA_CANONICAL", canonical)
    msa._configure_msa_thor_target(pinned_prefix)
    assert "cutlass" not in sys.modules
    assert os.environ["CUTE_DSL_ARCH"] == canonical

    # A fresh process observes the canonical value when the package imports.
    child = msa.subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import cutlass; print(cutlass.captured_target)",
            pinned_prefix[0],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.stdout.strip() == canonical
    monkeypatch.setattr(msa.subprocess, "run", lambda *a, **kw: pytest.fail("repeated query"))
    msa._configure_msa_thor_target(pinned_prefix)


@pytest.mark.parametrize("arch", [None, "sm_100a", "sm_103a", "sm_107a", "sm_120a"])
def test_other_architectures_preserve_environment(pinned_prefix, monkeypatch, arch):
    if arch is None:
        monkeypatch.delenv("TIRX_PREPARE_CUDA_ARCH")
    else:
        monkeypatch.setenv("TIRX_PREPARE_CUDA_ARCH", arch)
    monkeypatch.setenv("CUTE_DSL_ARCH", "unchanged")
    monkeypatch.setattr(msa.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected query"))
    msa._configure_msa_thor_target([])
    assert os.environ["CUTE_DSL_ARCH"] == "unchanged"
    assert "cutlass" not in sys.modules


def test_explicit_conflicting_target_fails_before_parent_import(pinned_prefix, monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    with pytest.raises(RuntimeError, match="conflicts with prepared Thor"):
        msa._configure_msa_thor_target(pinned_prefix)
    assert os.environ["CUTE_DSL_ARCH"] == "sm_100a"
    assert "cutlass" not in sys.modules


@pytest.mark.parametrize("change", ["prefix", "environment"])
def test_configuration_cannot_drift_mid_process(pinned_prefix, monkeypatch, change):
    msa._configure_msa_thor_target(pinned_prefix)
    if change == "prefix":
        pinned_prefix = [pinned_prefix[0] + "-other"]
    else:
        monkeypatch.setenv("CUTE_DSL_ARCH", "sm_110a")
    with pytest.raises(RuntimeError, match="changed after target configuration"):
        msa._configure_msa_thor_target(pinned_prefix)


def test_missing_frozen_dependency_fails(pinned_prefix, tmp_path):
    with pytest.raises(RuntimeError, match="requires its frozen"):
        msa._configure_msa_thor_target([])
    metadata = tmp_path / "quack_kernels-0.5.0.dist-info" / "METADATA"
    metadata.write_text("Name: quack-kernels\nVersion: 0.5.1\n")
    with pytest.raises(RuntimeError, match=r"quack-kernels==0\.5\.0"):
        msa._configure_msa_thor_target(pinned_prefix)
    assert "CUTE_DSL_ARCH" not in os.environ


@pytest.mark.parametrize("captured", ["sm_110a", "sm_101a"])
def test_existing_singleton_is_never_reconfigured(pinned_prefix, monkeypatch, captured):
    monkeypatch.setenv("CUTE_DSL_ARCH", captured)
    instance = SimpleNamespace(envar=SimpleNamespace(arch=captured))
    dsl = type("CuTeDSL", (), {})
    dsl._instances = {dsl: instance}
    arch = SimpleNamespace(from_string=lambda value: SimpleNamespace(to_string=lambda: "sm_101a"))
    monkeypatch.setitem(
        sys.modules, "cutlass", SimpleNamespace(__file__=pinned_prefix[0] + "/cutlass/__init__.py")
    )
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.arch", SimpleNamespace(Arch=arch))
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl", SimpleNamespace(CuTeDSL=dsl))
    monkeypatch.setattr(msa.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected query"))
    if captured == "sm_110a":
        with pytest.raises(RuntimeError, match="fresh worker with CUTE_DSL_ARCH=sm_101a"):
            msa._configure_msa_thor_target(pinned_prefix)
    else:
        msa._configure_msa_thor_target(pinned_prefix)
    assert dsl._instances[dsl] is instance
    assert instance.envar.arch == captured
    assert os.environ["CUTE_DSL_ARCH"] == captured


def test_wrong_compiler_prefix_is_rejected(pinned_prefix, monkeypatch):
    monkeypatch.setattr(
        msa.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps(
                {"canonical": "sm_101a", "requested": None, "module": "/other/cutlass/__init__.py"}
            )
        ),
    )
    with pytest.raises(RuntimeError, match="outside the pinned prefix"):
        msa._configure_msa_thor_target(pinned_prefix)
    assert "CUTE_DSL_ARCH" not in os.environ


def test_source_paths_are_selected_before_target_query(pinned_prefix, monkeypatch, tmp_path):
    monkeypatch.setattr(msa, "cutedsl_paths", lambda: pinned_prefix)
    monkeypatch.setattr(msa, "msa_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())

    def query(paths):
        assert paths == pinned_prefix
        assert paths[0] in sys.path
        assert str(tmp_path / "python" / "fmha_sm100" / "cute") in sys.path
        assert "cutlass" not in sys.modules

    monkeypatch.setattr(msa, "_configure_msa_thor_target", query)
    msa.ensure_msa_importable()


@pytest.mark.parametrize("already_configured", [False, True])
def test_initialized_wrong_device_is_rejected_before_mutation(
    pinned_prefix, monkeypatch, already_configured
):
    if already_configured:
        msa._configure_msa_thor_target(pinned_prefix)
    before = os.environ.get("CUTE_DSL_ARCH")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: True, get_device_capability=lambda: (10, 0))
        ),
    )
    monkeypatch.setattr(msa.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected query"))
    with pytest.raises(RuntimeError, match=r"requires CUDA capability \(11, 0\), got \(10, 0\)"):
        msa._configure_msa_thor_target(pinned_prefix)
    assert os.environ.get("CUTE_DSL_ARCH") == before


def test_uninitialized_torch_is_not_queried(pinned_prefix, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_initialized=lambda: False,
                get_device_capability=lambda: pytest.fail("unexpected CUDA query"),
            )
        ),
    )
    msa._configure_msa_thor_target(pinned_prefix)
    assert os.environ["CUTE_DSL_ARCH"] == "sm_101a"
