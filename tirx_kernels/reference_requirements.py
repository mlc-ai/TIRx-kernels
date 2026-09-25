# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Validation and availability checks for kernel correctness references."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import re
import subprocess
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
