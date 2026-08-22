#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Install exact correctness references and their test runner dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, distributions, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "reference-dependencies.json"
DEFAULT_SOURCE_ROOT = REPO_ROOT / ".reference-deps"


def run(*command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    shown = " ".join(command)
    print(f"+ {shown}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def load_lock() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text())
    if lock.get("schema") != 1:
        raise RuntimeError(f"unsupported reference dependency schema: {lock.get('schema')!r}")
    return lock


def checkout_source(source: dict[str, Any], source_root: Path) -> Path:
    checkout = source_root / source["name"]
    revision = source["revision"]
    if not checkout.exists():
        run("git", "clone", "--filter=blob:none", source["url"], str(checkout))
    elif not (checkout / ".git").exists():
        raise RuntimeError(f"reference source path is not a git checkout: {checkout}")

    try:
        output("git", "rev-parse", "--verify", "HEAD", cwd=checkout)
        has_head = True
    except subprocess.CalledProcessError:
        has_head = False
    materialized = any(path.name != ".git" for path in checkout.iterdir())
    if (
        has_head
        and materialized
        and output(
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules=untracked",
            cwd=checkout,
        )
    ):
        raise RuntimeError(f"reference source checkout has local changes: {checkout}")
    origin = output("git", "remote", "get-url", "origin", cwd=checkout)
    if origin.rstrip("/").removesuffix(".git") != source["url"].rstrip("/").removesuffix(".git"):
        raise RuntimeError(f"unexpected origin for {checkout}: {origin}")

    try:
        output("git", "cat-file", "-e", f"{revision}^{{commit}}", cwd=checkout)
    except subprocess.CalledProcessError:
        run("git", "fetch", "--filter=blob:none", "origin", revision, cwd=checkout)
    run("git", "checkout", "--detach", revision, cwd=checkout)
    run("git", "submodule", "update", "--init", "--recursive", cwd=checkout)
    actual = output("git", "rev-parse", "HEAD", cwd=checkout)
    if actual != revision:
        raise RuntimeError(f"{source['name']} resolved to {actual}, expected {revision}")
    return checkout


def source_build_environment() -> dict[str, str]:
    env = os.environ.copy()
    cccl = distribution("nvidia-cuda-cccl")
    utility = next(
        (
            Path(file.locate())
            for file in cccl.files or ()
            if str(file).endswith("cuda/std/utility")
        ),
        None,
    )
    if utility is None:
        raise RuntimeError("nvidia-cuda-cccl does not contain cuda/std/utility")
    cccl_include = str(utility.parents[2])
    existing = env.get("CPLUS_INCLUDE_PATH")
    env["CPLUS_INCLUDE_PATH"] = (
        f"{cccl_include}{os.pathsep}{existing}" if existing else cccl_include
    )
    return env


def materialize_source_links(source: dict[str, Any], checkout: Path) -> None:
    for link in source.get("source_links", []):
        source_path = (checkout / link["source"]).resolve(strict=True)
        target_path = checkout / link["target"]
        if target_path.is_symlink() and target_path.resolve() == source_path:
            continue
        if target_path.exists() or target_path.is_symlink():
            raise RuntimeError(f"reference install link has unexpected target: {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        relative_source = os.path.relpath(source_path, target_path.parent)
        target_path.symlink_to(relative_source, target_is_directory=True)


def import_paths(import_name: str) -> list[Path]:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return []
    paths = []
    if spec.origin not in (None, "built-in"):
        paths.append(Path(spec.origin).resolve())
    paths.extend(Path(path).resolve() for path in spec.submodule_search_locations or ())
    return paths


def import_uses_checkout(source: dict[str, str], checkout: Path) -> bool:
    install_path = (checkout / source["install_subdirectory"]).resolve()
    return any(path.is_relative_to(install_path) for path in import_paths(source["import_name"]))


def requirement_pin(requirement: str) -> tuple[str, str]:
    package, expected = requirement.split("==", 1)
    return package.split("[", 1)[0], expected


def target_versions(target: Path) -> dict[str, str]:
    return {
        str(item.metadata["Name"]).lower().replace("_", "-"): item.version
        for item in distributions(path=[str(target)])
        if item.metadata["Name"]
    }


def install(lock: dict[str, Any], source_root: Path) -> None:
    requirements = lock["python_requirements"]
    run(sys.executable, "-m", "pip", "install", "--upgrade", *lock["test_requirements"])
    run(sys.executable, "-m", "pip", "install", "--upgrade", "--no-deps", *requirements)
    build_env = source_build_environment()
    source_root.mkdir(parents=True, exist_ok=True)
    for name, isolated in lock.get("isolated_python_requirements", {}).items():
        target = source_root / name
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "--target",
            str(target),
            *isolated,
        )
    for source in lock["sources"]:
        checkout = checkout_source(source, source_root)
        materialize_source_links(source, checkout)
        if import_uses_checkout(source, checkout):
            print(f"{source['name']} already imports from its pinned checkout", flush=True)
            continue
        install_path = checkout / source["install_subdirectory"]
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(install_path),
            env=build_env,
        )


def check(lock: dict[str, Any], source_root: Path) -> None:
    failures: list[str] = []
    for requirement in [*lock["python_requirements"], *lock["test_requirements"]]:
        package, expected = requirement_pin(requirement)
        try:
            actual = version(package)
        except PackageNotFoundError:
            failures.append(f"{package} is not installed")
            continue
        if actual != expected:
            failures.append(f"{package}=={actual}, expected {expected}")

    for name, requirements in lock.get("isolated_python_requirements", {}).items():
        target = source_root / name
        if not target.is_dir():
            failures.append(f"missing isolated dependency directory: {target}")
            continue
        installed = target_versions(target)
        for requirement in requirements:
            package, expected = requirement_pin(requirement)
            actual = installed.get(package.lower().replace("_", "-"))
            if actual is None:
                failures.append(f"{package} is not installed under {target}")
            elif actual != expected:
                failures.append(f"{package}=={actual} under {target}, expected {expected}")

    for source in lock["sources"]:
        checkout = source_root / source["name"]
        if not (checkout / ".git").exists():
            failures.append(f"missing checkout: {checkout}")
            continue
        actual = output("git", "rev-parse", "HEAD", cwd=checkout)
        if actual != source["revision"]:
            failures.append(f"{source['name']} checkout is {actual}, expected {source['revision']}")
        origin = output("git", "remote", "get-url", "origin", cwd=checkout)
        if origin.rstrip("/").removesuffix(".git") != source["url"].rstrip("/").removesuffix(
            ".git"
        ):
            failures.append(f"unexpected origin for {checkout}: {origin}")
        dirty = output(
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules=untracked",
            cwd=checkout,
        )
        if dirty:
            failures.append(f"reference source checkout has local changes: {checkout}")
        try:
            materialize_source_links(source, checkout)
        except (FileNotFoundError, RuntimeError) as error:
            failures.append(str(error))
        resolved_import_paths = import_paths(source["import_name"])
        if not resolved_import_paths:
            failures.append(f"cannot find import {source['import_name']!r}")
            continue
        install_path = (checkout / source["install_subdirectory"]).resolve()
        if not any(path.is_relative_to(install_path) for path in resolved_import_paths):
            rendered = ", ".join(str(path) for path in resolved_import_paths)
            failures.append(
                f"import {source['import_name']!r} resolves outside {install_path}: {rendered}"
            )

    if failures:
        raise RuntimeError("reference dependency check failed:\n- " + "\n- ".join(failures))
    print(f"Reference dependencies match {LOCK_PATH.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"checkout directory (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument("--check", action="store_true", help="verify the installed lock only")
    args = parser.parse_args()
    lock = load_lock()
    source_root = args.source_root.resolve()
    if not args.check:
        install(lock, source_root)
        run(
            sys.executable,
            str(Path(__file__).resolve()),
            "--check",
            "--source-root",
            str(source_root),
        )
        return
    check(lock, source_root)


if __name__ == "__main__":
    main()
