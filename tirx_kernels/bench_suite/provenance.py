# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Run provenance helpers shared by the bench-suite client and its remote worker shim.

Everything here depends on the standard library only, so the same module runs
inside a kcoral worker (which has no ``git`` binary) and in the local
orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def kernels_repo_root() -> Path:
    """Git root of the tirx-kernels repo (parent of the tirx_kernels package)."""
    return SCRIPT_DIR.parent.parent


def _git_output(args: list[str], *, timeout: float = 5) -> str | None:
    """Run one git command; ``None`` when git is missing, fails, or prints nothing."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_label(repo: Path) -> str | None:
    if not repo.exists():
        return None
    sha = _git_output(["-C", str(repo), "rev-parse", "--short=8", "HEAD"])
    if not sha:
        return None
    dirty = _git_output(["-C", str(repo), "status", "--porcelain"])
    return sha + ("-dirty" if dirty else "")


def module_repo_root(import_name: str) -> Path | None:
    """Git root of an importable package, if it's a local checkout."""
    try:
        mod = __import__(import_name)
    except Exception:
        return None
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        return None
    for p in [Path(pkg_file).resolve().parent, *Path(pkg_file).resolve().parents]:
        if (p / ".git").exists():
            return p
    return None


def tir_repo_root() -> Path | None:
    """TVM git root: TVM_PATH env, else installed tvm package checkout."""
    env = os.environ.get("TVM_PATH")
    if env:
        p = Path(env).resolve()
        if (p / "python" / "tvm").is_dir():
            return p
    return module_repo_root("tvm")


def collect_repo_git() -> dict[str, str | None]:
    """SHAs for the three repos involved: tvm, tirx-kernels, tirx-bench-ci."""
    tir_root = tir_repo_root()
    tirx_root = module_repo_root("tirx_kernels") or kernels_repo_root()
    bench_ci_root: Path | None = None
    for base in (tirx_root, tir_root):
        if base is None:
            continue
        candidate = base.parent / "tirx-bench-ci"
        if (candidate / ".git").exists():
            bench_ci_root = candidate
            break
    return {
        "tir": git_label(tir_root) if tir_root else None,
        "tirx-kernels": git_label(tirx_root) if tirx_root else None,
        "tirx-bench-ci": git_label(bench_ci_root) if bench_ci_root else None,
    }


def git_tree_sha(root: Path | None, path: str) -> str | None:
    """Content-addressed git tree SHA of ``HEAD:<path>`` (merge-stable)."""
    if root is None:
        return None
    return _git_output(["-C", str(root), "rev-parse", f"HEAD:{path}"])


def collect_kernel_fingerprint() -> dict[str, str | None]:
    """Merge-stable content fingerprints (git *tree* SHAs) of the source that
    determines kernel codegen + perf.

    The commit SHAs in ``collect_repo_git`` are rewritten by a squash/rebase
    merge, so a baseline that records only commit SHAs can't be mapped back to a
    mainline commit afterwards. A git tree SHA is content-addressed (Merkle): it
    is identical before and after a merge as long as the directory's content is
    unchanged. Confirm a checkout matches a recorded baseline with
    ``git rev-parse HEAD:<path>``.
    """
    tir_root = tir_repo_root()
    tirx_root = module_repo_root("tirx_kernels") or kernels_repo_root()
    return {
        "tir:python/tvm/tirx": git_tree_sha(tir_root, "python/tvm/tirx"),
        "tirx-kernels:tirx_kernels": git_tree_sha(tirx_root, "tirx_kernels"),
    }


# Packages used as baselines in workloads.yaml — anything our regression
# numbers compare against, so the recorded version pins the comparison.
BASELINE_PACKAGES = [
    "torch",
    "deep_gemm",
    "flashinfer",
    "flash_kda",
    "flash_attn",
    "sglang",
    "cutlass",
]


def package_provenance(import_name: str) -> dict | None:
    """Probe a Python package: version + (if editable git install) repo + SHA.

    Returns None when neither the package nor distribution metadata exists.
    """

    def _record_git(path: Path, info: dict) -> None:
        root = _git_output(["-C", str(path), "rev-parse", "--show-toplevel"])
        if not root:
            return
        sha = _git_output(["-C", root, "rev-parse", "--short=8", "HEAD"])
        if not sha:
            return
        dirty = _git_output(["-C", root, "status", "--porcelain"])
        info["git_dir"] = root
        info["git_sha"] = sha + ("-dirty" if dirty else "")

    dists: list[str] = []
    try:
        from importlib.metadata import distribution as _probe_dist

        _probe_dist(import_name)
        dists.append(import_name)
    except Exception:
        pass
    try:
        from importlib.metadata import packages_distributions

        for dist_name in packages_distributions().get(import_name) or []:
            if dist_name not in dists:
                dists.append(dist_name)
    except Exception:
        pass
    if not dists:
        dists = [import_name]

    mod = None
    try:
        mod = __import__(import_name)
    except Exception:
        pass
    info: dict = {"importable": mod is not None}
    # Heavy optional baselines can fail during their top-level import even when
    # their source checkout is discoverable on PYTHONPATH. Resolve the module
    # spec without executing it so provenance still records that checkout.
    try:
        spec = find_spec(import_name)
    except Exception:
        spec = None
    if spec is not None:
        source_dir = None
        if spec.origin and spec.origin not in ("built-in", "frozen"):
            source_dir = Path(spec.origin).resolve().parent
        elif spec.submodule_search_locations:
            source_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        if source_dir is not None:
            info.setdefault("source_dir", str(source_dir))
            _record_git(source_dir, info)
    # Version: prefer __version__, else importlib.metadata. Top-level import
    # name and the distribution name often disagree (e.g. flash_attn ↔
    # flash-attn-4) — use packages_distributions() to bridge.
    version = getattr(mod, "__version__", None) if mod is not None else None
    if version is None:
        try:
            from importlib.metadata import version as _meta_version

            for d in dists:
                try:
                    version = _meta_version(d)
                    if version is not None:
                        info["dist"] = d
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if version is not None:
        info["version"] = str(version)
    if import_name == "torch":
        cuda = getattr(getattr(mod, "version", None), "cuda", None)
        git_v = getattr(getattr(mod, "version", None), "git_version", None)
        if cuda:
            info["cuda"] = str(cuda)
        if git_v:
            info["torch_git_version"] = str(git_v)
    # PEP 610 direct_url.json: when a package was `pip install -e <path>` or
    # `pip install <path>`, pip writes the source path/URL into the dist-info.
    # This catches the editable case (the package lives outside the repo it
    # was built from, so the __file__ walk below misses it). dist resolution:
    # prefer `info["dist"]` if we set it above, else default to import_name.
    direct_source = False
    try:
        from importlib.metadata import distribution as _meta_dist

        dist = None
        for dist_name in [info.get("dist"), *dists, import_name]:
            if not dist_name:
                continue
            try:
                dist = _meta_dist(dist_name)
                info.setdefault("dist", dist.metadata["Name"])
                break
            except Exception:
                continue
        if dist is not None:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct = json.loads(direct_url_text)
                url = direct.get("url") or ""
                if url.startswith("file://"):
                    src_path = Path(url[len("file://") :]).resolve()
                    direct_source = True
                    info["source_dir"] = str(src_path)
                    info.pop("git_dir", None)
                    info.pop("git_sha", None)
                    if direct.get("dir_info", {}).get("editable"):
                        info["editable"] = True
                    _record_git(src_path, info)
    except Exception:
        pass
    if mod is None:
        return info if "version" in info or "source_dir" in info else None
    # Resolve a directory we can git-probe. Namespace packages and some
    # __init__.py-less namespaces set mod.__file__ to None — fall back to
    # __path__[0] then to a known submodule's file.
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        # Last resort: try to import a likely submodule with a real file.
        for sub in (".cute", ".csrc", ".jit_kernels", ".jit"):
            try:
                submod = __import__(import_name + sub, fromlist=["__file__"])
                if getattr(submod, "__file__", None):
                    pkg_file = submod.__file__
                    break
            except Exception:
                continue
    if pkg_file and not direct_source:
        pkg_dir = Path(pkg_file).resolve().parent
        # Walk up looking for a git repo. .git can be a dir (regular clone)
        # or a file (worktree); both are fine for `git rev-parse`.
        _record_git(pkg_dir, info)
    return info


def collect_baseline_provenance() -> dict:
    return {name: package_provenance(name) or {"installed": False} for name in BASELINE_PACKAGES}


# ── Worker-side helpers (no git binary, no tirx_kernels imports) ─────────────


def read_git_head(repo: Path) -> str | None:
    """Full commit SHA of ``HEAD`` read from the ``.git`` files, without git.

    Handles a detached HEAD, a symbolic ref stored loose or in ``packed-refs``,
    and a ``.git`` *file* pointing at a worktree gitdir.
    """
    git_dir = Path(repo) / ".git"
    try:
        if git_dir.is_file():
            pointer = git_dir.read_text().strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = (Path(repo) / pointer[len("gitdir:") :].strip()).resolve()
        head = (git_dir / "HEAD").read_text().strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head[len("ref:") :].strip()
    common_dir = git_dir
    try:
        common_pointer = (git_dir / "commondir").read_text().strip()
        common_dir = (git_dir / common_pointer).resolve()
    except OSError:
        pass
    for base in (git_dir, common_dir):
        try:
            return (base / ref).read_text().strip() or None
        except OSError:
            pass
    for base in (git_dir, common_dir):
        try:
            for line in (base / "packed-refs").read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
        except OSError:
            continue
    return None


def content_fingerprint(root: Path, *, suffixes: tuple[str, ...] = (".py",)) -> str | None:
    """``sha256:<hex>`` over the sorted (relative path, bytes) of matching files."""
    root = Path(root)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return "sha256:" + digest.hexdigest()


def tvm_identity() -> dict:
    """Identify the TVM that this process imports (file, version, git HEAD, tirx tree)."""
    info: dict = {"importable": False}
    try:
        import tvm
    except Exception as error:  # pragma: no cover - depends on the environment
        info["error"] = f"{type(error).__name__}: {error}"
        return info
    info["importable"] = True
    info["file"] = getattr(tvm, "__file__", None)
    info["version"] = getattr(tvm, "__version__", None)
    root: Path | None = None
    if info["file"]:
        for candidate in Path(info["file"]).resolve().parents:
            if (candidate / "python" / "tvm").is_dir() and (candidate / ".git").exists():
                root = candidate
                break
            if (candidate / ".git").exists():
                root = candidate
                break
    info["root"] = str(root) if root else None
    head = read_git_head(root) if root else None
    info["git_head"] = head
    info["git_label"] = head[:8] if head else None
    tirx_dir = Path(info["file"]).resolve().parent / "tirx" if info["file"] else None
    info["tirx_tree"] = content_fingerprint(tirx_dir) if tirx_dir else None
    return info


def loaded_libraries(needle: str = "libcupti") -> list[str]:
    """Distinct mapped shared-object paths containing ``needle`` (Linux only)."""
    try:
        with open("/proc/self/maps") as maps:
            lines = maps.readlines()
    except OSError:
        return []
    found: set[str] = set()
    for line in lines:
        parts = line.split()
        if len(parts) >= 6 and needle in parts[-1] and parts[-1].startswith("/"):
            found.add(parts[-1])
    return sorted(found)


def python_identity() -> dict:
    return {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
