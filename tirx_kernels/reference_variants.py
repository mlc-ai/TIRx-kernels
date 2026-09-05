# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Frozen DeepGEMM and FlashMLA host adapters for separately built Thor references.

The original source packages remain the default on native architectures. Thor
workers select an explicit build manifest and import path before interpreter
startup; this module never changes a device identity or replaces a loaded module.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sysconfig
from pathlib import Path

VARIANTS = {
    "deep-gemm": {
        "import": "deep_gemm",
        "extension": "deep_gemm._C",
        "revision": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
        "url": "https://github.com/deepseek-ai/DeepGEMM.git",
        "environment": "TIRX_DEEP_GEMM_VARIANT_MANIFEST",
        "device_root": "deep_gemm/include/deep_gemm",
        # The original private diagnostic copy omitted these unused Python files.
        # New clones retain them; registration records omissions explicitly.
        "legacy_omissions": (
            "third-party/tilelang_ops/__init__.py",
            "third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py",
            "third-party/tilelang_ops/utils.py",
        ),
    },
    "flash-mla": {
        "import": "flash_mla",
        "extension": "flash_mla.cuda",
        "revision": "9241ae3ef9bac614dd25e45e507e089f888280e0",
        "url": "https://github.com/deepseek-ai/FlashMLA.git",
        "environment": "TIRX_FLASH_MLA_VARIANT_MANIFEST",
        "device_root": "csrc/sm100",
        "legacy_omissions": (),
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def adapted_text(name: str, relative: str, original: str) -> str:
    """Apply only the host changes exercised by the pinned source probes."""
    replacements = {
        "deep-gemm": {
            "csrc/jit/device_runtime.hpp": (
                (
                    "return get_arch_pair().first;",
                    "const auto [major, minor] = get_arch_pair();\n"
                    "        // Audit adapter: preserve real JIT target, select the original SM100 host schedule on Thor.\n"
                    "        return (major == 11 and minor == 0) ? 10 : major;",
                ),
            )
        },
        "flash-mla": {
            "csrc/api/common.h": (
                ("return major == 10;", "return major == 10 || (major == 11 && minor == 0);"),
            ),
            "setup.py": (
                (
                    '["-gencode", "arch=compute_100f,code=sm_100f"]',
                    '["-gencode", "arch=compute_110a,code=sm_110a"]',
                ),
                (
                    'subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])',
                    'assert Path("csrc/cutlass/include").is_dir()',
                ),
            ),
        },
    }
    for old, new in replacements[name].get(relative, ()):
        if original.count(old) != 1:
            raise RuntimeError(f"frozen host patch no longer matches {name}/{relative}")
        original = original.replace(old, new)
    return original


def source_inventory(name: str, root: Path, *, allow_legacy_omissions: bool = False) -> dict:
    """Compare every tracked parent file to its pin, including non-device files."""
    policy = VARIANTS[name]
    if git_output(root, "rev-parse", "HEAD") != policy["revision"]:
        raise RuntimeError(f"{name} source revision does not match the frozen pin")
    if git_output(root, "remote", "get-url", "origin") != policy["url"]:
        raise RuntimeError(f"{name} source origin does not match the frozen pin")
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", "HEAD"])
    hashes, device_hashes, patches, missing, submodules, symlinks = {}, {}, [], [], {}, {}
    for entry in entries.rstrip(b"\0").split(b"\0"):
        info, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = info.decode().split()
        relative = raw_path.decode()
        path = root / relative
        if kind == "commit":
            actual = git_output(path, "rev-parse", "HEAD")
            if actual != oid:
                raise RuntimeError(f"submodule revision mismatch: {relative}")
            if subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"]).returncode:
                raise RuntimeError(f"submodule tracked files changed: {relative}")
            submodules[relative] = oid
            if path.is_symlink():
                symlinks[relative] = str(path.resolve())
            continue
        original = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid])
        if not path.exists() and not path.is_symlink():
            if allow_legacy_omissions and relative in policy["legacy_omissions"]:
                missing.append(relative)
                continue
            raise RuntimeError(f"missing frozen source file: {relative}")
        expected = original
        if relative in {"csrc/jit/device_runtime.hpp", "csrc/api/common.h", "setup.py"}:
            expected = adapted_text(name, relative, original.decode()).encode()
        actual = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
        if actual != expected:
            raise RuntimeError(f"unexpected source change: {name}/{relative}")
        digest = hashlib.sha256(actual).hexdigest()
        hashes[relative] = digest
        if expected != original:
            patches.append(relative)
        if relative.startswith(policy["device_root"] + "/"):
            if actual != original:
                raise RuntimeError(f"device source changed: {relative}")
            device_hashes[relative] = digest
    if name == "deep-gemm":
        for include in ("cute", "cutlass"):
            path = root / "deep_gemm/include" / include
            if path.resolve() != (root / "third-party/cutlass/include" / include).resolve():
                raise RuntimeError(f"DeepGEMM include link does not use the frozen CUTLASS: {path}")
    return {
        "tracked_file_sha256": hashes,
        "device_file_sha256": device_hashes,
        "host_patch_files": patches,
        "omitted_files": missing,
        "submodule_revisions": submodules,
        "submodule_symlinks": symlinks,
    }


def deepgemm_build_commands(log: str) -> list[str]:
    """DeepGEMM's pinned setuptools build emits commands directly, without Ninja."""
    compile_commands = [
        line
        for line in log.splitlines()
        if " -c " in line
        and "csrc/python_api.cpp" in line
        and "-std=c++17" in line
        and "-O3" in line
    ]
    link_commands = [
        line
        for line in log.splitlines()
        if " -shared " in line and "deep_gemm/_C." in line and ".so" in line
    ]
    if len(compile_commands) != 1 or len(link_commands) != 1:
        raise RuntimeError("DeepGEMM requires retained actual host compile and link commands")
    return compile_commands + link_commands


def validate_variant(name: str, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    policy = VARIANTS[name]
    if (manifest.get("schema"), manifest.get("name"), manifest.get("cuda_arch")) != (
        1,
        name,
        "sm_110a",
    ):
        raise RuntimeError(f"invalid Thor reference manifest: {manifest_path}")
    if (manifest.get("source_revision"), manifest.get("source_url")) != (
        policy["revision"],
        policy["url"],
    ):
        raise RuntimeError("reference manifest pin mismatch")
    root = Path(manifest["root"]).resolve()
    actual = source_inventory(
        name, root, allow_legacy_omissions=bool(manifest["source_inventory"]["omitted_files"])
    )
    if actual != manifest["source_inventory"]:
        raise RuntimeError("reference source changed after manifest registration")
    if manifest["python_soabi"] != sysconfig.get_config_var("SOABI"):
        raise RuntimeError("reference Python extension ABI mismatch")
    if manifest["torch_version"] != importlib.metadata.version("torch"):
        raise RuntimeError("reference PyTorch extension ABI version mismatch")
    extension = (root / manifest["extension"]["relative_path"]).resolve()
    if not extension.is_relative_to(root) or sha256(extension) != manifest["extension"]["sha256"]:
        raise RuntimeError("reference extension hash or location mismatch")
    build = manifest["build"]
    if sha256(Path(build["log"])) != build["log_sha256"]:
        raise RuntimeError("reference build log changed after registration")
    if name == "flash-mla" and not build["ninja_files"]:
        raise RuntimeError("reference build flags are missing")
    for ninja in build["ninja_files"]:
        if sha256(Path(ninja["path"])) != ninja["sha256"]:
            raise RuntimeError("reference build flags changed after registration")
    if name == "flash-mla":
        flags = "\n".join(Path(ninja["path"]).read_text() for ninja in build["ninja_files"])
        if (
            "arch=compute_110a,code=sm_110a" not in flags
            or "code=sm_100f" in flags
            or "code=sm_90a" in flags
            or build.get("cubin_architectures") != ["sm_110a"]
            or build.get("cubin_count", 0) < 1
        ):
            raise RuntimeError("FlashMLA reference has no verified isolated sm_110a build")
    if name == "deep-gemm":
        commands = deepgemm_build_commands(Path(build["log"]).read_text())
        if commands != build.get("compiler_commands"):
            raise RuntimeError("DeepGEMM host compiler commands do not match the actual build log")
        if build.get("jit_target") != "actual_device_arch":
            raise RuntimeError("DeepGEMM must retain the real device JIT target")
    return manifest


_loaded_manifests: dict[str, dict] = {}


def load_reference(name: str):
    """Import a canonical native source, or a verified explicitly selected Thor build."""
    from tirx_kernels.target import prepare_cuda_arch

    policy = VARIANTS[name]
    selected = os.environ.get(policy["environment"])
    if prepare_cuda_arch() != "sm_110a":
        if selected:
            raise RuntimeError(f"{policy['environment']} is only valid for sm_110a workers")
        return importlib.import_module(policy["import"])
    if not selected:
        raise RuntimeError(
            f"Thor {name} requires {policy['environment']} pointing to a verified sm_110a "
            "source-reference build manifest"
        )
    if name in _loaded_manifests and selected != _loaded_manifests[name]["selected_environment"]:
        raise RuntimeError("reference variant cannot change within a running worker")
    if name not in _loaded_manifests:
        manifest = validate_variant(name, Path(selected))
        spec = importlib.util.find_spec(policy["import"])
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or not origin.is_relative_to(Path(manifest["root"]).resolve()):
            raise RuntimeError(
                f"{name} import does not use the selected variant; set worker PYTHONPATH before startup"
            )
        module = importlib.import_module(policy["import"])
        extension = importlib.import_module(policy["extension"])
        expected = (Path(manifest["root"]) / manifest["extension"]["relative_path"]).resolve()
        if Path(extension.__file__).resolve() != expected:
            raise RuntimeError(f"{name} imported a different extension than its manifest")
        manifest["selected_environment"] = selected
        manifest["manifest_path"] = str(Path(selected).resolve())
        manifest["manifest_sha256"] = sha256(Path(selected))
        _loaded_manifests[name] = manifest
        return module
    return importlib.import_module(policy["import"])


def reference_provenance(name: str) -> dict:
    """Attach the selected artifact identity outside benchmark timing."""
    manifest = _loaded_manifests.get(name)
    if manifest is None:
        return {}
    return {
        "manifest_path": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_revision": manifest["source_revision"],
        "cuda_arch": "sm_110a",
        "root": manifest["root"],
        "extension": manifest["extension"],
        "host_patch_files": manifest["source_inventory"]["host_patch_files"],
        "omitted_files": manifest["source_inventory"]["omitted_files"],
        "submodule_symlinks": manifest["source_inventory"]["submodule_symlinks"],
        "build": manifest["build"],
    }
