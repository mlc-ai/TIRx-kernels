#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Prepare/build or register an isolated pinned DeepGEMM/FlashMLA Thor variant.

No package is installed globally. A worker must use the emitted environment
before Python starts. Both new builds and existing-build registration retain
source identities and full tracked-tree differences in a manifest.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tirx_kernels.reference_variants import (  # noqa: E402
    VARIANTS,
    adapted_text,
    deepgemm_build_commands,
    git_output,
    sha256,
    source_inventory,
    validate_variant,
)


@contextmanager
def build_lock():
    # Thor CPU compilation and binary scans share unified DRAM with timed work.
    with open("/tmp/tirx-kernels-gpu.lock", "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def clone_frozen(name: str, source: Path, destination: Path) -> None:
    policy = VARIANTS[name]
    if git_output(source, "rev-parse", "HEAD") != policy["revision"]:
        raise RuntimeError("canonical source is not at its frozen revision")
    if git_output(source, "remote", "get-url", "origin") != policy["url"]:
        raise RuntimeError("canonical source origin mismatch")
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise RuntimeError("canonical source has tracked changes")
    if destination.exists():
        raise FileExistsError(f"refusing to replace an existing variant: {destination}")
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(source), str(destination)], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", policy["revision"]], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "remote", "set-url", "origin", policy["url"]], check=True
    )
    # Copy pinned submodules as real local clones, retaining every tracked file.
    # In particular, the three legacy TileLang Python files are not omitted.
    for line in git_output(source, "submodule", "status", "--recursive").splitlines():
        fields = line.split()
        revision, relative = fields[0], fields[1]
        if revision.startswith(("-", "+", "U")):
            raise RuntimeError(f"canonical submodule is not materialized at its pin: {relative}")
        original, target = source / relative, destination / relative
        if target.exists() and not any(target.iterdir()):
            target.rmdir()
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(original), str(target)], check=True
        )
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
        origin = git_output(original, "remote", "get-url", "origin")
        subprocess.run(
            ["git", "-C", str(target), "remote", "set-url", "origin", origin], check=True
        )
    for relative in ("csrc/jit/device_runtime.hpp", "csrc/api/common.h", "setup.py"):
        path = destination / relative
        if path.is_file():
            original = path.read_text()
            changed = adapted_text(name, relative, original)
            if changed != original:
                path.write_text(changed)
    if name == "deep-gemm":
        for include in ("cute", "cutlass"):
            target = destination / "deep_gemm/include" / include
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"unexpected pre-existing include link: {target}")
            target.symlink_to(
                Path("../../third-party/cutlass/include") / include, target_is_directory=True
            )


def build_environment(name: str, root: Path) -> dict[str, str]:
    # Match the canonical installer's CCCL discovery without installing anything.
    from install_reference_dependencies import source_build_environment

    env = source_build_environment()
    cuda_home = Path(env.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    cccl = cuda_home / "include/cccl"
    if not (cccl / "cuda/std/utility").is_file():
        raise RuntimeError(
            f"CUDA toolkit CCCL headers are required for the Thor source build: {cccl}"
        )
    env["CUDA_HOME"] = str(cuda_home)
    env["CPATH"] = str(cccl) + (os.pathsep + env["CPATH"] if env.get("CPATH") else "")
    env["MAX_JOBS"] = env.get("MAX_JOBS", "2")
    env["NVCC_THREADS"] = env.get("NVCC_THREADS", "1")
    if name == "deep-gemm":
        env.update(DG_FORCE_BUILD="1", DG_SKIP_CUDA_BUILD="0", DG_JIT_USE_RUNTIME_API="0")
        env["DG_JIT_CACHE_DIR"] = str(root.parent / (root.name + "-sm110a-jit"))
    else:
        env.update(FLASH_MLA_DISABLE_SM90="1", FLASH_MLA_DISABLE_SM100="0")
    return env


def build_variant(name: str, root: Path) -> Path:
    env = build_environment(name, root)
    command = [sys.executable, "setup.py", "build_ext", "--inplace"]
    log = root / "tirx-thor-build.log"
    with log.open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        key: env.get(key)
                        for key in (
                            "CUDA_HOME",
                            "CPATH",
                            "CPLUS_INCLUDE_PATH",
                            "CXX",
                            "MAX_JOBS",
                            "NVCC_THREADS",
                            "FLASH_MLA_DISABLE_SM90",
                            "FLASH_MLA_DISABLE_SM100",
                            "DG_FORCE_BUILD",
                            "DG_JIT_USE_RUNTIME_API",
                        )
                    },
                }
            )
            + "\n"
        )
        stream.flush()
        subprocess.run(
            command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    return log


def register_variant(name: str, root: Path, build_log: Path, *, legacy: bool) -> Path:
    policy = VARIANTS[name]
    inventory = source_inventory(name, root, allow_legacy_omissions=legacy)
    package, stem = policy["extension"].rsplit(".", 1)
    extensions = list((root / package.replace(".", "/")).glob(stem + ".*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"expected exactly one built extension, found {extensions}")
    extension = extensions[0]
    if not build_log.is_file():
        raise FileNotFoundError("a retained actual build log is required")
    build = {
        "log": str(build_log.resolve()),
        "log_sha256": sha256(build_log),
        "jit_target": "actual_device_arch" if name == "deep-gemm" else None,
    }
    ninja_files = list((root / "build").rglob("build.ninja"))
    if name == "flash-mla" and not ninja_files:
        raise RuntimeError("retained build.ninja is required to record actual compiler flags")
    build["ninja_files"] = [{"path": str(p.resolve()), "sha256": sha256(p)} for p in ninja_files]
    if name == "deep-gemm":
        build["compiler_commands"] = deepgemm_build_commands(build_log.read_text())
    if name == "flash-mla":
        flags = "\n".join(p.read_text() for p in ninja_files)
        if (
            "arch=compute_110a,code=sm_110a" not in flags
            or "code=sm_100f" in flags
            or "code=sm_90a" in flags
        ):
            raise RuntimeError(
                "FlashMLA actual build flags do not describe an isolated sm_110a variant"
            )
        cuobjdump = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/cuobjdump"
        listing = subprocess.check_output([str(cuobjdump), "--list-elf", str(extension)], text=True)
        build["cubin_tool"] = str(cuobjdump.resolve())
        archs = sorted(set(re.findall(r"sm_[0-9]+[af]?", listing)))
        if archs != ["sm_110a"]:
            raise RuntimeError(f"unexpected FlashMLA cubin architectures: {archs}")
        build.update(
            cubin_architectures=archs, cubin_count=listing.count("ELF file"), cubin_listing=listing
        )
    manifest = {
        "schema": 1,
        "name": name,
        "cuda_arch": "sm_110a",
        "root": str(root.resolve()),
        "source_revision": policy["revision"],
        "source_url": policy["url"],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": importlib.metadata.version("torch"),
        "source_inventory": inventory,
        "extension": {
            "relative_path": str(extension.relative_to(root)),
            "sha256": sha256(extension),
        },
        "build": build,
        "registration": "existing_private_build" if legacy else "frozen_source_build",
    }
    manifest_path = root / "tirx-thor-reference.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    validate_variant(name, manifest_path)
    env = {
        policy["environment"]: str(manifest_path.resolve()),
        "PYTHONPATH": str(root.resolve()) + ":" + os.environ.get("PYTHONPATH", ""),
        "TIRX_PREPARE_CUDA_ARCH": "sm_110a",
    }
    if name == "deep-gemm":
        env["DG_JIT_CACHE_DIR"] = str(root.parent / (root.name + "-sm110a-jit"))
    else:
        env["FLASH_MLA_PATH"] = str(root.resolve())
    (root / "tirx-thor-environment.json").write_text(json.dumps(env, indent=2) + "\n")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / ".reference-deps")
    parser.add_argument("--variant-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--build", action="store_true", help="Compile after preparing a fresh clone."
    )
    action.add_argument(
        "--build-prepared",
        action="store_true",
        help="Compile a previously prepared, unbuilt clone.",
    )
    action.add_argument(
        "--register-existing",
        action="store_true",
        help="Verify and register a retained private build.",
    )
    action.add_argument(
        "--check", action="store_true", help="Check an already registered build; no compilation."
    )
    parser.add_argument("--build-log", type=Path)
    args = parser.parse_args()
    root = args.variant_root.resolve()
    with build_lock():
        if args.check:
            validate_variant(args.name, root / "tirx-thor-reference.json")
            print("Frozen source, extension and target manifest checks passed")
            return
        if args.register_existing:
            if args.build_log is None:
                parser.error("--register-existing requires --build-log")
            print(register_variant(args.name, root, args.build_log, legacy=True))
            return
        if not args.build_prepared:
            clone_frozen(args.name, args.source_root / args.name, root)
        source_inventory(args.name, root)
        if args.build or args.build_prepared:
            log = build_variant(args.name, root)
            print(register_variant(args.name, root, log, legacy=False))
        else:
            print(
                f"Prepared frozen host adaptation at {root}; no build or runtime manifest created"
            )


if __name__ == "__main__":
    main()
