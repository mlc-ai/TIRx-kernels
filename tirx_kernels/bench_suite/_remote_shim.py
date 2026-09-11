# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Worker-side program uploaded verbatim to a kcoral server by the bench suite.

This module runs inside a kcoral worker process, not in the orchestrator.  It
must import only the standard library at module scope: the worker's own
``tirx_kernels`` is stale, so every request ships the orchestrator's tree as a
tarball and imports it from a temporary directory.

Entry points (see ``tirx_kernels/bench_suite/remote.py`` for the client side):

``probe(tar_bytes, options_json)``
    One GPU-holding request per suite run.  Extracts the tree, proves it imports
    against the worker's TVM, and reports the device, versions, TVM identity,
    libcupti copies and baseline-package provenance.
``prepare(tar_bytes, before_tar_bytes, spec_json)``
    Registered ``cpu_only`` by default so compilation happens off the GPU lease.
    Sets the environment contract, extracts the tree(s), and CPU-prepares one
    workload.  The prepared benchmark is kept in module state for ``run``.
``run(spec_json)``
    GPU-holding.  Times the prepared workload and returns the benchmark result.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback

ROOT_PREFIX = "tirx-bench-suite-"
STALE_ROOT_AGE_S = 6 * 3600
CUPTI_TARGET_RELATIVE = ("nvidia", "cu13", "lib", "libcupti.so.13")

_STATE: dict = {}


# ── helpers ──────────────────────────────────────────────────────────────────


def _sweep_stale_roots() -> list[str]:
    """Remove extraction roots older than ``STALE_ROOT_AGE_S`` (killed workers)."""
    removed: list[str] = []
    now = time.time()
    for path in glob.glob(os.path.join(tempfile.gettempdir(), ROOT_PREFIX + "*")):
        try:
            if now - os.stat(path).st_mtime < STALE_ROOT_AGE_S:
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        except OSError:
            continue
    return removed


def _extract(tar_bytes: bytes, destination: str) -> int:
    os.makedirs(destination, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        archive.extractall(destination, filter="data")
    return len(members)


def _purge_tirx_kernels_modules() -> None:
    for name in list(sys.modules):
        if name == "tirx_kernels" or name.startswith("tirx_kernels."):
            del sys.modules[name]


def _install_tree(tree_root: str) -> None:
    """Make ``tree_root/tirx_kernels`` the package this process (and children) import."""
    _purge_tirx_kernels_modules()
    sys.path.insert(0, tree_root)
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = tree_root if not existing else tree_root + os.pathsep + existing
    importlib.invalidate_caches()


def _cupti_target() -> str | None:
    """The libcupti copy cupti-python (kcoral's cpu_only guard) loads."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except Exception:
        spec = None
    for location in list(getattr(spec, "submodule_search_locations", None) or []):
        candidate = os.path.join(location, *CUPTI_TARGET_RELATIVE[1:])
        if os.path.exists(candidate):
            return candidate
    for path in _loaded_libraries("libcupti"):
        return path
    return None


def _loaded_libraries(needle: str) -> list[str]:
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


def _share_cupti(root: str) -> dict:
    """Point triton's proton at the libcupti instance already loaded in this process.

    kcoral verifies ``cpu_only`` functions through cupti-python, which loads the
    ``nvidia/cu13`` libcupti.  proton would otherwise dlopen triton's bundled
    copy; CUPTI registers with the driver once per process, so the second copy
    fails ``cuptiSubscribe`` with ``CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED``
    even after the first one unsubscribed.  A symlink named ``libcupti.so`` to
    the loaded file makes both users share one instance (glibc dedups by inode).
    """
    info: dict = {"target": None, "triton_cupti_lib_path": os.environ.get("TRITON_CUPTI_LIB_PATH")}
    if info["triton_cupti_lib_path"]:
        info["note"] = "TRITON_CUPTI_LIB_PATH preset; left unchanged"
        return info
    target = _cupti_target()
    info["target"] = target
    if target is None:
        info["note"] = "no libcupti found; proton will load triton's bundled copy"
        return info
    shim_dir = os.path.join(root, "cupti")
    os.makedirs(shim_dir, exist_ok=True)
    os.symlink(target, os.path.join(shim_dir, "libcupti.so"))
    os.environ["TRITON_CUPTI_LIB_PATH"] = shim_dir
    info["triton_cupti_lib_path"] = shim_dir
    return info


def _triton_default_cupti() -> str | None:
    try:
        import triton

        base = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "lib")
        for name in ("cupti-blackwell", "cupti"):
            candidate = os.path.join(base, name, "libcupti.so")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        return None
    return None


def _device_info() -> dict:
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = props.major, props.minor
    return {
        "uuid": str(getattr(props, "uuid", "")) or None,
        "name": props.name,
        "multi_processor_count": int(props.multi_processor_count),
        "capability": [int(major), int(minor)],
        "arch": f"sm_{major}{minor}a",
        "total_memory": int(props.total_memory),
    }


def _versions() -> dict:
    versions: dict = {}
    for name in ("torch", "tvm", "tvm_ffi", "triton", "flashinfer", "cutlass"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception as error:
            versions[name] = f"unavailable: {type(error).__name__}"
    try:
        import torch

        versions["cuda"] = torch.version.cuda
    except Exception:
        versions["cuda"] = None
    return versions


def _link_reference_deps(spec: dict, roots: list[str]) -> str | None:
    """Expose a server-side ``.reference-deps`` checkout to the shipped trees.

    Kernels resolve reference sources as ``<repo root>/.reference-deps/...``
    where the repo root is the directory containing ``tirx_kernels``; here that
    is each extracted tree root, so a symlink there points them at the server's
    copy.  The directory comes from the spec (``--reference-deps-dir``) or the
    worker's ``TIRX_REFERENCE_DEPS`` environment variable.
    """
    target = spec.get("reference_deps_dir") or os.environ.get("TIRX_REFERENCE_DEPS")
    if not target or not os.path.isdir(target):
        return None
    for root in roots:
        link = os.path.join(root, ".reference-deps")
        if not os.path.lexists(link):
            os.symlink(target, link)
    return target


def _apply_environment(spec: dict, *, tree_root: str, after_root: str | None) -> None:
    env = os.environ
    env["TIRX_PREPARE_CUDA_ARCH"] = str(spec["cuda_arch"])
    env["TIRX_PREPARE_NUM_SMS"] = str(spec["num_sms"])
    env["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    # nvrtc calls cuInit, which the cpu_only guard rejects; on-lease prepare
    # leaves the kernel's own choice (some kernels insist on the default).
    env.pop("TVM_CUDA_COMPILE_MODE", None)
    if spec.get("prepare_mode") == "cpu":
        env["TVM_CUDA_COMPILE_MODE"] = str(spec.get("cuda_compile_mode") or "nvcc")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cache_dir = spec.get("cache_dir") or os.path.join(
        env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "tirx-kernels",
        "bench-suite",
    )
    os.makedirs(cache_dir, exist_ok=True)
    env["TIRX_BENCH_CACHE_DIR"] = cache_dir
    env.pop("TIRX_INTERNAL_AB_CURRENT_BENCHMARK_ROOT", None)
    if spec.get("side") == "before":
        if after_root is None:
            raise ValueError("the before side requires the after tree")
        env["TIRX_INTERNAL_AB_CURRENT_BENCHMARK_ROOT"] = after_root
    _install_tree(tree_root)


# ── entry points ─────────────────────────────────────────────────────────────


def probe(tar_bytes: bytes, options_json: str) -> str:
    """Identify the worker and prove the shipped tree imports against its TVM."""
    options = json.loads(options_json) if options_json else {}
    started = time.time()
    out: dict = {"timings": {}, "stale_roots_removed": _sweep_stale_roots()}
    root = tempfile.mkdtemp(prefix=ROOT_PREFIX + "probe-")
    try:
        tree_root = os.path.join(root, "after")
        out["tree_members"] = _extract(tar_bytes, tree_root)
        out["timings"]["extract_s"] = time.time() - started
        _install_tree(tree_root)
        import_started = time.time()
        import tirx_kernels.kern.entry
        import tirx_kernels.runner  # noqa: F401
        from tirx_kernels.bench_suite import provenance

        out["timings"]["framework_import_s"] = time.time() - import_started
        out["tirx_kernels_file"] = sys.modules["tirx_kernels"].__file__
        out["device"] = _device_info()
        out["python"] = provenance.python_identity()
        out["versions"] = _versions()
        out["tvm"] = provenance.tvm_identity()
        out["libcupti"] = {
            "cupti_python": _cupti_target(),
            "triton_default": _triton_default_cupti(),
            "loaded": provenance.loaded_libraries("libcupti"),
        }
        usage = shutil.disk_usage(tempfile.gettempdir())
        out["tmpdir"] = {"path": tempfile.gettempdir(), "free_bytes": int(usage.free)}
        out["environment"] = {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "TVM_SOURCE_DIR",
                "TMPDIR",
                "TIRX_REFERENCE_DEPS",
            )
        }
        reference_deps = options.get("reference_deps_dir") or os.environ.get("TIRX_REFERENCE_DEPS")
        out["reference_deps_dir"] = (
            reference_deps if reference_deps and os.path.isdir(reference_deps) else None
        )
        if options.get("references"):
            provenance_started = time.time()
            out["baselines"] = provenance.collect_baseline_provenance()
            out["timings"]["baselines_s"] = time.time() - provenance_started
        else:
            out["baselines"] = {}
        if options.get("sanity_matmul", True):
            import torch

            a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
            (a @ a).sum().item()
            torch.cuda.synchronize()
            out["sanity"] = {"matmul_ok": True}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    out["timings"]["total_s"] = time.time() - started
    return json.dumps(out, default=str)


def prepare(tar_bytes: bytes, before_tar_bytes: bytes | None, spec_json: str) -> str:
    """CPU-prepare one workload from the shipped tree; keep it for ``run``."""
    spec = json.loads(spec_json)
    started = time.time()
    meta: dict = {"timings": {}, "stale_roots_removed": _sweep_stale_roots()}
    root = tempfile.mkdtemp(prefix=ROOT_PREFIX)
    try:
        after_root = os.path.join(root, "after")
        _extract(tar_bytes, after_root)
        tree_root = after_root
        if before_tar_bytes is not None:
            before_root = os.path.join(root, "before")
            _extract(before_tar_bytes, before_root)
            if spec.get("side") == "before":
                tree_root = before_root
        meta["timings"]["extract_s"] = time.time() - started
        meta["root"] = root
        meta["tree_root"] = tree_root
        meta["cupti"] = _share_cupti(root) if spec.get("cupti_workaround", True) else {}
        roots = [after_root] + (
            [os.path.join(root, "before")] if before_tar_bytes is not None else []
        )
        meta["reference_deps_dir"] = _link_reference_deps(spec, roots)
        _apply_environment(spec, tree_root=tree_root, after_root=after_root)

        import_started = time.time()
        from tirx_kernels.bench.__main__ import _find_bench_config
        from tirx_kernels.registry import load_kernel
        from tirx_kernels.runner import (
            ab_current_benchmark_module,
            cuda_is_initialized,
            prepare_kernel_bench,
            set_external_references_enabled,
        )

        meta["timings"]["framework_import_s"] = time.time() - import_started
        meta["tirx_kernels_file"] = sys.modules["tirx_kernels"].__file__
        set_external_references_enabled(bool(spec.get("references")))
        meta["cuda_initialized_before"] = cuda_is_initialized()

        load_started = time.time()
        module = load_kernel(spec["kernel"], strict=True)
        meta["timings"]["module_load_s"] = time.time() - load_started
        meta["kernel_module"] = module.__name__
        config_started = time.time()
        config = _find_bench_config(ab_current_benchmark_module(module), spec["config"])
        meta["timings"]["config_resolve_s"] = time.time() - config_started
        prepare_started = time.time()
        prepared = prepare_kernel_bench(
            spec["kernel"], config, module=module, require_cuda_uninitialized=False
        )
        meta["timings"]["prepare_s"] = time.time() - prepare_started
        if prepared.required_num_gpus != 1:
            raise ValueError(
                f"workload requires {prepared.required_num_gpus} GPU(s); the remote backend "
                "runs single-GPU workloads only"
            )
        meta["cuda_initialized_after"] = cuda_is_initialized()
    except BaseException:
        meta["traceback"] = traceback.format_exc()
        shutil.rmtree(root, ignore_errors=True)
        raise
    _STATE.clear()
    _STATE.update({"prepared": prepared, "root": root, "spec": spec, "meta": meta})
    meta["timings"]["total_s"] = time.time() - started
    return json.dumps(meta, default=str)


def run(spec_json: str) -> str:
    """Time the prepared workload on the GPU and release everything."""
    spec = json.loads(spec_json)
    if not _STATE:
        raise RuntimeError("run() called without a successful prepare() in this request")
    prepared = _STATE["prepared"]
    root = _STATE["root"]
    started = time.time()
    out: dict = {"timings": {}}
    try:
        from unittest import SkipTest

        from tirx_kernels.runner import close_prepared_kernel_bench, run_prepared_kernel_bench

        bench_kwargs = {
            key: value for key, value in (spec.get("bench") or {}).items() if value is not None
        }
        try:
            try:
                result = run_prepared_kernel_bench(prepared, **bench_kwargs)
            except SkipTest as error:
                result = {
                    "kernel": spec["kernel"],
                    "label": spec["config"],
                    "status": "SKIP",
                    "reason": str(error),
                }
        finally:
            try:
                close_prepared_kernel_bench(prepared)
            except Exception as error:  # pragma: no cover - kernel-owned cleanup
                out["close_error"] = f"{type(error).__name__}: {error}"
        out["timings"]["gpu_s"] = time.time() - started
        out["result"] = result
        try:
            out["device"] = _device_info()
        except Exception as error:  # pragma: no cover
            out["device"] = {"error": f"{type(error).__name__}: {error}"}
    finally:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        if not spec.get("keep_root"):
            shutil.rmtree(root, ignore_errors=True)
        _STATE.clear()
    out["timings"]["total_s"] = time.time() - started
    return json.dumps(out, default=str)
