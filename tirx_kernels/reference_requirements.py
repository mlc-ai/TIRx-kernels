# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Availability, source identity, and build checks for correctness references.

Native architectures use the original source packages. Thor-specific build
validation applies only when a worker explicitly prepares ``sm_110a``.
Importing this module does not import a reference package or GPU runtime.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sysconfig
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

_IMPORT_NAME_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")
_FULL_GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


@dataclass(frozen=True)
class GitRequirement:
    """Exact source identity required by a correctness reference."""

    url: str
    commit: str


@dataclass(frozen=True)
class ReferenceRequirement:
    """Canonical form of one literal KERNEL_META requirement."""

    package: str
    import_name: str
    specifier: str | None = None
    git: GitRequirement | None = None


def parse_reference_requirements(
    value: object,
) -> tuple[tuple[ReferenceRequirement, ...], list[str]]:
    """Parse the optional literal metadata field without importing its packages."""
    if value is None:
        return (), []
    if not isinstance(value, tuple):
        return (), ["'reference_requirements' must be tuple"]
    if not value:
        return (), ["'reference_requirements' must not be empty when present"]

    requirements: list[ReferenceRequirement] = []
    errors: list[str] = []
    packages: set[str] = set()
    allowed = {"package", "import", "specifier", "git"}
    for index, item in enumerate(value):
        owner = f"reference_requirements[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{owner} must be dict")
            continue
        unknown = set(item) - allowed
        if unknown:
            errors.append(f"{owner} has unsupported field(s): {sorted(unknown)}")

        package = item.get("package")
        import_name = item.get("import")
        specifier = item.get("specifier")
        git_value = item.get("git")
        if not isinstance(package, str) or not package:
            errors.append(f"{owner}.package must be a non-empty string")
        if not isinstance(import_name, str) or _IMPORT_NAME_PATTERN.fullmatch(import_name) is None:
            errors.append(f"{owner}.import must be a canonical Python import name")
        if specifier is None and git_value is None:
            errors.append(f"{owner} requires at least one of 'specifier' or 'git'")

        if specifier is not None:
            if not isinstance(specifier, str) or not specifier:
                errors.append(f"{owner}.specifier must be a non-empty PEP 440 string")
            else:
                try:
                    SpecifierSet(specifier)
                except InvalidSpecifier as error:
                    errors.append(f"{owner}.specifier is invalid: {error}")

        git = None
        if git_value is not None:
            if not isinstance(git_value, dict):
                errors.append(f"{owner}.git must be dict")
            else:
                git_unknown = set(git_value) - {"url", "commit"}
                if git_unknown:
                    errors.append(f"{owner}.git has unsupported field(s): {sorted(git_unknown)}")
                url = git_value.get("url")
                commit = git_value.get("commit")
                if not isinstance(url, str) or not url:
                    errors.append(f"{owner}.git.url must be a non-empty string")
                if (
                    not isinstance(commit, str)
                    or _FULL_GIT_COMMIT_PATTERN.fullmatch(commit) is None
                ):
                    errors.append(f"{owner}.git.commit must be a full lowercase Git SHA")
                if isinstance(url, str) and url and isinstance(commit, str):
                    git = GitRequirement(url=url, commit=commit)

        if isinstance(package, str) and package:
            canonical_package = canonicalize_name(package)
            if canonical_package in packages:
                errors.append(f"{owner}.package duplicates {package!r}")
            packages.add(canonical_package)
        if (
            isinstance(package, str)
            and package
            and isinstance(import_name, str)
            and _IMPORT_NAME_PATTERN.fullmatch(import_name) is not None
        ):
            requirements.append(
                ReferenceRequirement(
                    package=package,
                    import_name=import_name,
                    specifier=specifier if isinstance(specifier, str) and specifier else None,
                    git=git,
                )
            )

    if errors:
        return (), errors
    return tuple(requirements), []


def _normalized_git_url(url: str) -> str:
    value = url.strip()
    if value.startswith("git@") and ":" in value:
        authority, path = value.split(":", 1)
        value = f"ssh://{authority}/{path}"
    parsed = urlparse(value)
    if parsed.scheme and parsed.hostname:
        path = parsed.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"{parsed.hostname.lower()}/{path.lstrip('/')}"
    return value.removesuffix(".git").rstrip("/")


def _git_checkout_info(path: Path) -> tuple[str, str] | None:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        url = subprocess.run(
            ["git", "-C", root, "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return url, commit


def _direct_url(package: str) -> dict | None:
    try:
        payload = importlib.metadata.distribution(package).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _git_identity(requirement: ReferenceRequirement, module_spec) -> tuple[str, str] | None:
    direct = _direct_url(requirement.package)
    if direct is not None:
        vcs = direct.get("vcs_info")
        url = direct.get("url")
        if isinstance(vcs, dict) and isinstance(url, str):
            commit = vcs.get("commit_id")
            if isinstance(commit, str):
                return url, commit
        if isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme == "file":
                checkout = _git_checkout_info(Path(unquote(parsed.path)))
                if checkout is not None:
                    return checkout

    origin = getattr(module_spec, "origin", None)
    candidates = []
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        candidates.append(Path(origin).parent)
    locations = getattr(module_spec, "submodule_search_locations", None)
    if locations is not None:
        candidates.extend(Path(location) for location in locations)
    for candidate in candidates:
        checkout = _git_checkout_info(candidate)
        if checkout is not None:
            return checkout
    return None


@cache
def probe_reference_requirement(requirement: ReferenceRequirement) -> str | None:
    """Return an unmet-requirement reason, or None when it is satisfied."""
    try:
        module_spec = importlib.util.find_spec(requirement.import_name)
    except ModuleNotFoundError as error:
        return f"{requirement.package}: import {requirement.import_name!r} is unavailable ({error})"
    if module_spec is None:
        return f"{requirement.package}: import {requirement.import_name!r} is unavailable"

    if requirement.specifier is not None:
        try:
            installed = importlib.metadata.version(requirement.package)
        except importlib.metadata.PackageNotFoundError:
            return f"{requirement.package}: installed distribution version is unavailable"
        accepted = SpecifierSet(requirement.specifier)
        if not accepted.contains(installed, prereleases=True):
            return (
                f"{requirement.package}: requires {requirement.specifier}, "
                f"installed version is {installed}"
            )

    if requirement.git is not None:
        identity = _git_identity(requirement, module_spec)
        if identity is None:
            return f"{requirement.package}: Git source identity cannot be verified"
        actual_url, actual_commit = identity
        if _normalized_git_url(actual_url) != _normalized_git_url(requirement.git.url):
            return (
                f"{requirement.package}: requires Git URL {requirement.git.url}, "
                f"installed source is {actual_url}"
            )
        if actual_commit.lower() != requirement.git.commit:
            return (
                f"{requirement.package}: requires Git commit {requirement.git.commit}, "
                f"installed source is {actual_commit}"
            )
    return None


def unmet_reference_requirements(value: object) -> tuple[str, ...]:
    """Return every unmet reason for a validated literal metadata value."""
    requirements, errors = parse_reference_requirements(value)
    if errors:
        raise ValueError("invalid reference requirements: " + "; ".join(errors))
    return tuple(
        reason
        for requirement in requirements
        if (reason := probe_reference_requirement(requirement)) is not None
    )


# Explicit frozen builds used by Thor source comparisons.

VARIANTS = {
    "deep-gemm": {
        "import": "deep_gemm",
        "extension": "deep_gemm._C",
        "revision": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
        "url": "https://github.com/deepseek-ai/DeepGEMM.git",
        "environment": "TIRX_DEEP_GEMM_VARIANT_MANIFEST",
        "device_root": "deep_gemm/include/deep_gemm",
        # Omissions of these optional helpers must be recorded in the manifest.
        "legacy_omissions": (
            "third-party/tilelang_ops/__init__.py",
            "third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py",
            "third-party/tilelang_ops/utils.py",
        ),
    }
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
    """Apply the supported host adaptation for a pinned source revision."""
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
        }
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
        if relative == "csrc/jit/device_runtime.hpp":
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
    for ninja in build["ninja_files"]:
        if sha256(Path(ninja["path"])) != ninja["sha256"]:
            raise RuntimeError("reference build flags changed after registration")
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
    from tirx_kernels.runner import prepare_cuda_arch

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
