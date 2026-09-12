# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""kcoral client backend for the bench suite.

Every workload becomes one ``POST /execute`` request: the local ``tirx_kernels``
tree (a deterministic tar.gz, blob-cached on the server by SHA-256) plus the
worker shim ``_remote_shim.py`` are uploaded, the shim CPU-prepares the workload
off the GPU lease and then times it while holding the lease.  The server owns
GPU exclusivity, worker isolation and queueing; this module owns request
construction, retries, and turning outcomes into bench-suite result rows.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import os
import random
import tarfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVER_URL_ENV = "TIRX_BENCH_SERVER"
REFERENCE_DEPS_DIR_ENV = "TIRX_BENCH_REFERENCE_DEPS"
DEFAULT_SERVER_URL = "http://127.0.0.1:8901"
DEFAULT_REQUEST_TIMEOUT_S = 3600.0
MAX_REQUEST_TIMEOUT_S = 3600.0
PROBE_TIMEOUT_S = 600.0
DEFAULT_OUTPUT_LIMIT_BYTES = 4 * 1024 * 1024
MAX_IN_FLIGHT_LIMIT = 32
EXECUTION_MODE = "remote"
PROCESS_MODEL = "kcoral_worker_per_request"
PREPARE_MODES = ("cpu", "gpu")
CUDA_COMPILE_MODE = "nvcc"
TREE_ARCNAME = "tirx_kernels"
# Paths relative to the package directory that never ship to the worker.
TREE_EXCLUDES = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".bench-suite",
    "bench_suite/baseline.json",
    "bench_suite/baseline.md",
)
INSTALL_HINT = (
    "install the bench-suite client extra: pip install -e '.[remote]' "
    "(or: pip install 'kcoral @ git+https://github.com/mlc-ai/kcoral')"
)
SHIM_PATH = Path(__file__).with_name("_remote_shim.py")


def _suite():
    """The orchestrator module, imported lazily (it imports this module's constants)."""
    from tirx_kernels.bench_suite import run

    return run


def now_iso() -> str:
    return _suite().now_iso()


def log(msg: str) -> None:
    _suite().log(msg)


class RemoteBenchError(RuntimeError):
    """A fatal client-side setup error (unreachable server, bad health, missing client)."""


@dataclass(frozen=True)
class KcoralApi:
    """The kcoral client names the backend uses, resolved lazily."""

    Client: Any
    Program: Any
    KCoralError: type
    TransportError: type
    ProtocolError: type


def load_kcoral() -> KcoralApi:
    try:
        import kcoral
    except ImportError as error:
        raise RemoteBenchError(f"the kcoral client is not importable ({error}); {INSTALL_HINT}")
    return KcoralApi(
        Client=kcoral.Client,
        Program=kcoral.Program,
        KCoralError=kcoral.KCoralError,
        TransportError=kcoral.TransportError,
        ProtocolError=kcoral.ProtocolError,
    )


# ── Tree archive ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TreeArchive:
    name: str
    sha256: str
    data: bytes
    file_count: int
    byte_count: int

    def summary(self) -> dict:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "archive_bytes": len(self.data),
        }


def _is_excluded(relative: str, excludes: tuple[str, ...]) -> bool:
    parts = relative.split("/")
    for pattern in excludes:
        if "/" in pattern:
            if relative == pattern or relative.startswith(pattern + "/"):
                return True
        elif any(fnmatch.fnmatchcase(part, pattern) for part in parts):
            return True
    return False


def build_tree_archive(
    package_dir: Path, *, arcname: str = TREE_ARCNAME, excludes: tuple[str, ...] = TREE_EXCLUDES
) -> TreeArchive:
    """Deterministic tar.gz of ``package_dir``: identical content, identical bytes.

    The archive's SHA-256 is the blob key on the server (so a tree is uploaded
    once) and the provenance fingerprint recorded with every row.
    """
    package_dir = Path(package_dir).resolve()
    if not package_dir.is_dir():
        raise RemoteBenchError(f"package directory does not exist: {package_dir}")
    files: list[tuple[str, Path]] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(package_dir).as_posix()
        if _is_excluded(relative, excludes):
            continue
        files.append((relative, path))
    raw = io.BytesIO()
    byte_count = 0
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative, path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{arcname}/{relative}")
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            archive.addfile(info, io.BytesIO(data))
            byte_count += len(data)
    data = gzip.compress(raw.getvalue(), mtime=0)
    return TreeArchive(
        name=arcname,
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
        file_count=len(files),
        byte_count=byte_count,
    )


def shim_source() -> str:
    return SHIM_PATH.read_text()


# ── Server handshake ─────────────────────────────────────────────────────────


@dataclass
class ServerProfile:
    """What one suite run learned about the server before submitting workloads."""

    url: str
    health: dict
    probe: dict
    probe_request_id: str | None = None
    probe_elapsed_ms: float | None = None

    @property
    def arch(self) -> str:
        return str(self.health["target"]["arch"])

    @property
    def worker_count(self) -> int:
        return len(self.health.get("workers") or [])

    @property
    def gpu_count(self) -> int:
        return int(self.health.get("gpu_count") or 0)

    @property
    def versions(self) -> dict:
        return dict(self.probe.get("versions") or self.health.get("versions") or {})

    @property
    def device(self) -> dict:
        return dict(self.probe.get("device") or {})

    @property
    def device_uuid(self) -> str | None:
        return self.device.get("uuid") or None

    @property
    def num_sms(self) -> int:
        return int(self.device["multi_processor_count"])

    @property
    def tvm(self) -> dict:
        return dict(self.probe.get("tvm") or {})

    @property
    def tvm_label(self) -> str:
        """Server TVM identity for ``git.tir``; never empty (ratio_diff requires it)."""
        label = self.tvm.get("git_label")
        if label:
            return str(label)
        return f"tvm-{self.versions.get('tvm') or 'unknown'}"

    @property
    def tirx_tree(self) -> str:
        return str(self.tvm.get("tirx_tree") or self.tvm_label)

    @property
    def baselines(self) -> dict:
        return dict(self.probe.get("baselines") or {})

    def summary(self) -> dict:
        return {
            "url": self.url,
            "arch": self.arch,
            "worker_count": self.worker_count,
            "gpu_count": self.gpu_count,
            "versions": self.versions,
            "device": self.device,
            "tvm": self.tvm,
            "python": self.probe.get("python"),
            "libcupti": self.probe.get("libcupti"),
            "tmpdir": self.probe.get("tmpdir"),
            "health": self.health,
            "probe_request_id": self.probe_request_id,
        }


def validate_health(health: Any, url: str) -> dict:
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise RemoteBenchError(f"{url}: /health did not report status ok: {health!r}")
    if int(health.get("gpu_count") or 0) < 1:
        raise RemoteBenchError(
            f"{url}: server has no GPU workers (gpu_count={health.get('gpu_count')!r})"
        )
    arch = (health.get("target") or {}).get("arch")
    if not isinstance(arch, str) or not arch.startswith("sm_"):
        raise RemoteBenchError(f"{url}: server target arch is unusable: {arch!r}")
    return health


def connect(api: KcoralApi, url: str, *, connect_timeout_s: float = 10.0) -> tuple[Any, dict]:
    """Open a client and validate ``GET /health``."""
    client = api.Client(url, connect_timeout_seconds=connect_timeout_s)
    try:
        health = client.health()
    except Exception as error:
        client.close()
        raise RemoteBenchError(f"cannot reach the benchmark server at {url}: {error}") from error
    try:
        validate_health(health, url)
    except RemoteBenchError:
        client.close()
        raise
    return client, health


def default_max_in_flight(health: dict) -> int:
    workers = len(health.get("workers") or [])
    return max(1, min(workers or 1, MAX_IN_FLIGHT_LIMIT))


def build_probe_program(
    api: KcoralApi,
    *,
    tree: TreeArchive,
    shim: str,
    references_enabled: bool,
    extra_blobs: tuple[TreeArchive, ...] = (),
    reference_deps_dir: str | None = None,
) -> Any:
    program = api.Program()
    tree_ref = program.upload(id="tree", kind="bytes", value=tree.data)
    for index, blob in enumerate(extra_blobs):
        # Warm the server blob cache before the thread pool starts submitting.
        program.upload(id=f"extra_blob_{index}", kind="bytes", value=blob.data)
    module = program.upload(id="shim", kind="module", source=shim)
    probe = program.get_function(id="probe_fn", module=module, name="probe")
    options = json.dumps(
        {
            "references": bool(references_enabled),
            "sanity_matmul": True,
            "reference_deps_dir": reference_deps_dir,
        }
    )
    result = program.run(id="probe", fn=probe, args=[tree_ref, options])
    program.return_(key="probe", value=result)
    return program


def probe_server(
    api: KcoralApi,
    client: Any,
    *,
    url: str,
    health: dict,
    tree: TreeArchive,
    shim: str,
    references_enabled: bool,
    timeout_s: float = PROBE_TIMEOUT_S,
    extra_blobs: tuple[TreeArchive, ...] = (),
    reference_deps_dir: str | None = None,
) -> ServerProfile:
    """One GPU-holding request that identifies the worker and validates the tree."""
    program = build_probe_program(
        api,
        tree=tree,
        shim=shim,
        references_enabled=references_enabled,
        extra_blobs=extra_blobs,
        reference_deps_dir=reference_deps_dir,
    )
    try:
        outcome = client.execute(program, timeout_seconds=timeout_s)
    except Exception as error:
        raise RemoteBenchError(f"server probe request failed: {error}") from error
    if outcome.status != "COMPLETED":
        error = outcome.error or {}
        detail = f"{error.get('kind')}: {error.get('message')}"
        trace = (error.get("traceback") or "").strip()
        raise RemoteBenchError(
            "server probe failed (the shipped tirx_kernels tree does not run on the worker?): "
            f"{detail}\n{trace}\nstderr:\n{(outcome.stderr or '')[-2000:]}"
        )
    probe = json.loads(outcome.results["probe"])
    profile = ServerProfile(
        url=url,
        health=health,
        probe=probe,
        probe_request_id=outcome.request_id,
        probe_elapsed_ms=outcome.elapsed_ms,
    )
    device_arch = profile.device.get("arch")
    if device_arch and device_arch != profile.arch:
        raise RemoteBenchError(
            f"server reports target {profile.arch} but its device computes {device_arch}"
        )
    return profile


# ── Request construction ─────────────────────────────────────────────────────


def workload_spec(
    workload: dict,
    *,
    cuda_arch: str,
    num_sms: int,
    side: str = "after",
    references_enabled: bool,
    prepare_mode: str,
    rounds: int,
    cooldown: float,
    reference_deps_dir: str | None = None,
) -> dict:
    if prepare_mode not in PREPARE_MODES:
        raise ValueError(f"prepare mode must be one of {PREPARE_MODES}, got {prepare_mode!r}")
    bench: dict[str, Any] = {"rounds": rounds, "cooldown": cooldown}
    for key in ("warmup", "repeat", "timer"):
        if workload.get(key) is not None:
            bench[key] = workload[key]
    return {
        "kernel": workload["kernel"],
        "config": workload["config"],
        "side": side,
        "cuda_arch": cuda_arch,
        "num_sms": int(num_sms),
        "references": bool(references_enabled),
        "cuda_compile_mode": CUDA_COMPILE_MODE,
        "cupti_workaround": True,
        "prepare_mode": prepare_mode,
        "reference_deps_dir": reference_deps_dir,
        "cache_dir": None,
        "keep_root": False,
        "bench": bench,
    }


def build_workload_program(
    program: Any, *, tree: TreeArchive, before_tree: TreeArchive | None, shim: str, spec: dict
) -> Any:
    """Populate ``program`` (a ``kcoral.Program`` or a duck-typed recorder)."""
    tree_ref = program.upload(id="tree", kind="bytes", value=tree.data)
    before_ref = None
    if spec.get("side") == "before":
        if before_tree is None:
            raise ValueError("the before side requires a before tree archive")
        before_ref = program.upload(id="before_tree", kind="bytes", value=before_tree.data)
    module = program.upload(id="shim", kind="module", source=shim)
    prepare_fn = program.get_function(
        id="prepare_fn", module=module, name="prepare", cpu_only=spec["prepare_mode"] == "cpu"
    )
    run_fn = program.get_function(id="run_fn", module=module, name="run")
    spec_json = json.dumps(spec, sort_keys=True)
    prepared = program.run(id="prepare", fn=prepare_fn, args=[tree_ref, before_ref, spec_json])
    program.return_(key="prepare", value=prepared)
    result = program.run(id="result", fn=run_fn, args=[spec_json])
    program.return_(key="result", value=result)
    return program


# ── Submission with retries ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryPolicy:
    # A saturated server (HTTP 503) is the normal state of a shared GPU box
    # during a sweep; the old local pool waited for a free card indefinitely,
    # so wait a long time here too before giving a row up.
    busy_total_s: float = 2 * 3600.0
    busy_backoff_s: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
    transport_attempts: int = 3
    transport_backoff_s: float = 2.0
    jitter: float = 0.25


@dataclass
class Submission:
    outcome: Any = None
    error: BaseException | None = None
    attempts: int = 0
    busy_retries: int = 0
    transport_retries: int = 0
    started_at: str = ""
    finished_at: str = ""
    wall_s: float = 0.0
    sleeps: list[float] = field(default_factory=list)
    error_category: str | None = None  # "busy", "timeout", "transport", "protocol", "server"
    prepare_fallback: dict | None = None  # set when cpu_only prepare fell back to on-lease


def _jittered(seconds: float, jitter: float) -> float:
    if jitter <= 0:
        return seconds
    return seconds * (1.0 + random.uniform(-jitter, jitter))


def execute_with_retry(
    api: KcoralApi,
    client: Any,
    program: Any,
    *,
    timeout_s: float,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    policy: RetryPolicy = RetryPolicy(),
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Submission:
    """Execute ``program``; retry a saturated server (503) and transport failures."""
    submission = Submission(started_at=now_iso())
    started = clock()
    deadline = started + policy.busy_total_s
    while True:
        submission.attempts += 1
        try:
            submission.outcome = client.execute(
                program, timeout_seconds=timeout_s, output_limit_bytes=output_limit_bytes
            )
            break
        except api.KCoralError as error:
            status = getattr(error, "status_code", None)
            if status == 503 and clock() < deadline:
                index = min(submission.busy_retries, len(policy.busy_backoff_s) - 1)
                delay = _jittered(policy.busy_backoff_s[index], policy.jitter)
                submission.busy_retries += 1
                submission.sleeps.append(delay)
                sleep(delay)
                continue
            submission.error = error
            kind = getattr(error, "kind", None)
            if status == 504 or kind == "timeout":
                submission.error_category = "timeout"
            elif status == 503 or kind == "busy":
                submission.error_category = "busy"
            else:
                submission.error_category = "server"
            break
        except api.TransportError as error:
            submission.transport_retries += 1
            if submission.transport_retries < policy.transport_attempts:
                delay = _jittered(
                    policy.transport_backoff_s * (2 ** (submission.transport_retries - 1)),
                    policy.jitter,
                )
                submission.sleeps.append(delay)
                sleep(delay)
                continue
            submission.error = error
            submission.error_category = "transport"
            break
        except api.ProtocolError as error:
            submission.error = error
            submission.error_category = "protocol"
            break
    submission.finished_at = now_iso()
    submission.wall_s = clock() - started
    return submission


def prepare_violation(outcome: Any) -> str | None:
    """The failure message when ``outcome`` failed inside the cpu_only prepare stage.

    Any prepare-stage failure is worth one retry with the lease held: the guard
    rejects CUDA entry (``gpu_access``), and some kernels reject the nvcc
    compile mode that off-lease prepare requires.
    """
    if outcome is None or getattr(outcome, "status", None) != "FAILED":
        return None
    error = outcome.error or {}
    if error.get("instruction_id") != "prepare":
        return None
    return f"{error.get('kind')}: {error.get('message') or 'prepare failed'}"


def submit_workload(
    api: KcoralApi,
    client: Any,
    *,
    tree: TreeArchive,
    before_tree: TreeArchive | None,
    shim: str,
    spec: dict,
    timeout_s: float,
    policy: RetryPolicy = RetryPolicy(),
    on_fallback: Callable[[str], None] | None = None,
) -> tuple[Submission, dict]:
    """Execute one workload; retry once with on-lease prepare if cpu_only prepare failed.

    Returns the final submission and the spec it ran with (``prepare_mode`` may
    have changed to ``"gpu"``); the fallback reason is recorded on the submission.
    """
    program = build_workload_program(
        api.Program(), tree=tree, before_tree=before_tree, shim=shim, spec=spec
    )
    submission = execute_with_retry(api, client, program, timeout_s=timeout_s, policy=policy)
    violation = prepare_violation(submission.outcome)
    if violation is None or spec.get("prepare_mode") != "cpu":
        return submission, spec
    if on_fallback is not None:
        on_fallback(violation)
    fallback_spec = dict(spec, prepare_mode="gpu")
    program = build_workload_program(
        api.Program(), tree=tree, before_tree=before_tree, shim=shim, spec=fallback_spec
    )
    retried = execute_with_retry(api, client, program, timeout_s=timeout_s, policy=policy)
    retried.attempts += submission.attempts
    retried.busy_retries += submission.busy_retries
    retried.transport_retries += submission.transport_retries
    retried.started_at = submission.started_at
    retried.wall_s += submission.wall_s
    retried.prepare_fallback = {"from": "cpu", "to": "gpu", "reason": violation}
    return retried, fallback_spec


# ── Outcome → row ────────────────────────────────────────────────────────────


def _loads_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return value


def _error_summary(error: dict | None) -> dict | None:
    if not isinstance(error, dict):
        return None
    keys = (
        "kind",
        "message",
        "instruction_id",
        "instruction_index",
        "instruction_op",
        "cuda_call",
        "location",
    )
    return {key: error.get(key) for key in keys if key in error}


_PHASE_BY_INSTRUCTION = {
    "prepare": "prepare",
    "result": "gpu",
    "tree": "upload",
    "before_tree": "upload",
    "shim": "upload",
    "prepare_fn": "upload",
    "run_fn": "upload",
}


def _failed_error_text(error: dict, *, timeout_s: float) -> str:
    phase = _PHASE_BY_INSTRUCTION.get(str(error.get("instruction_id")), "protocol")
    kind = error.get("kind")
    text = f"{phase}: {kind}: {error.get('message')}"
    if kind == "gpu_access":
        text += (
            f" (cuda_call={error.get('cuda_call')!r}, location={error.get('location')!r}); "
            "prepare entered CUDA under cpu_only; rerun with --prepare gpu"
        )
    trace = (error.get("traceback") or "").strip()
    if trace:
        text += "\n" + trace
    return text


def outcome_to_record(
    workload: dict,
    submission: Submission,
    *,
    profile: ServerProfile,
    tree_sha256: str,
    before_tree_sha256: str | None,
    prepare_mode: str,
    rounds: int,
    cooldown: float,
    references_enabled: bool,
    request_timeout_s: float,
    log_path: Path | None,
    side: str | None = None,
) -> dict:
    """Turn one submission into a bench-suite result row."""
    kernel = workload["kernel"]
    config = workload["config"]
    remote: dict[str, Any] = {
        "server_url": profile.url,
        "request_id": None,
        "attempts": submission.attempts,
        "busy_retries": submission.busy_retries,
        "transport_retries": submission.transport_retries,
        "wall_s": submission.wall_s,
        "prepare_mode": prepare_mode,
        "cuda_compile_mode": CUDA_COMPILE_MODE,
        "tree_sha256": tree_sha256,
        "before_tree_sha256": before_tree_sha256,
        "log": str(log_path) if log_path else None,
    }
    if side is not None:
        remote["side"] = side
    if submission.prepare_fallback is not None:
        remote["prepare_fallback"] = dict(submission.prepare_fallback)
    record: dict[str, Any] = {
        "kernel": kernel,
        "config": config,
        "label": config,
        "num_gpus": 1,
        "attempt": submission.attempts,
        "execution_mode": EXECUTION_MODE,
        "process_model": PROCESS_MODEL,
        "started_at": submission.started_at,
        "remote": remote,
    }
    device_uuid = profile.device_uuid
    outcome = submission.outcome
    if outcome is not None:
        remote.update(
            {
                "request_id": outcome.request_id,
                "status": outcome.status,
                "queue_ms": outcome.queue_ms,
                "elapsed_ms": outcome.elapsed_ms,
                "lease_wait_ms": outcome.lease_wait_ms,
                "lease_held_ms": outcome.lease_held_ms,
                "stdout_truncated": outcome.stdout_truncated,
                "stderr_truncated": outcome.stderr_truncated,
            }
        )
        prepare_meta = _loads_json((outcome.results or {}).get("prepare"))
        if isinstance(prepare_meta, dict):
            remote["prepare"] = prepare_meta
            remote["cupti"] = prepare_meta.get("cupti")

    if submission.error is not None:
        error = submission.error
        status = getattr(error, "status_code", None)
        kind = getattr(error, "kind", None)
        category = submission.error_category
        if category == "timeout" or status == 504:
            text = f"timeout: program exceeded {request_timeout_s:g}s ({error})"
        elif category == "busy" or status == 503:
            text = (
                f"busy: server saturated after {submission.attempts} attempt(s) "
                f"over {submission.wall_s:.0f}s ({error})"
            )
        elif category == "transport":
            text = f"transport: {error}"
        else:
            text = f"protocol: {type(error).__name__}: {error}"
        remote["error"] = {"kind": kind or type(error).__name__, "message": str(error)}
        record.update({"status": "FAIL", "error": text})
    elif outcome is None:
        record.update({"status": "FAIL", "error": "client: submission produced no outcome"})
    elif outcome.status == "COMPLETED":
        payload = _loads_json((outcome.results or {}).get("result"))
        result = payload.get("result") if isinstance(payload, dict) else None
        device = payload.get("device") if isinstance(payload, dict) else None
        if isinstance(device, dict) and device.get("uuid"):
            device_uuid = device["uuid"]
        if isinstance(payload, dict):
            remote["gpu_timings"] = payload.get("timings")
        if not isinstance(result, dict):
            record.update(
                {
                    "status": "FAIL",
                    "error": f"protocol: worker returned no result dict: {payload!r}",
                }
            )
        elif result.get("status") in ("SKIP", "FAIL"):
            record.update(result)
        else:
            _suite()._finalize_bench_record(
                result, rounds=rounds, cooldown=cooldown, references_enabled=references_enabled
            )
            record.update(result)
    elif outcome.status == "FAILED":
        error = outcome.error or {}
        remote["error"] = _error_summary(error)
        record.update(
            {"status": "FAIL", "error": _failed_error_text(error, timeout_s=request_timeout_s)}
        )
    else:
        record.update(
            {"status": "FAIL", "error": f"protocol: unexpected outcome status {outcome.status!r}"}
        )

    record.setdefault("label", config)
    record["gpu"] = device_uuid or ""
    record["gpus"] = [device_uuid] if device_uuid else []
    record["physical_gpu_uuids"] = [device_uuid] if device_uuid else []
    record["finished_at"] = submission.finished_at or now_iso()
    return record


def write_request_log(
    path: Path, *, workload: dict, submission: Submission, side: str | None = None
) -> None:
    outcome = submission.outcome
    lines = [
        f"kernel/config: {workload['kernel']}/{workload['config']}",
        f"side: {side or 'after'}",
        f"started_at: {submission.started_at}",
        f"finished_at: {submission.finished_at}",
        f"attempts: {submission.attempts} (busy_retries={submission.busy_retries}, "
        f"transport_retries={submission.transport_retries})",
    ]
    if submission.error is not None:
        lines.append(f"client_error: {type(submission.error).__name__}: {submission.error}")
    if submission.prepare_fallback is not None:
        lines.append(f"prepare_fallback: {json.dumps(submission.prepare_fallback)}")
    if outcome is not None:
        lines += [
            f"request_id: {outcome.request_id}",
            f"status: {outcome.status}",
            f"queue_ms: {outcome.queue_ms:.1f}",
            f"elapsed_ms: {outcome.elapsed_ms:.1f}",
            f"lease_wait_ms: {outcome.lease_wait_ms:.1f}",
            f"lease_held_ms: {outcome.lease_held_ms:.1f}",
        ]
        if outcome.error:
            lines.append(f"error: {json.dumps(_error_summary(outcome.error))}")
    lines.append("")
    if outcome is not None:
        lines.append(f"--- stdout (truncated={outcome.stdout_truncated}) ---")
        lines.append(outcome.stdout or "")
        lines.append(f"--- stderr (truncated={outcome.stderr_truncated}) ---")
        lines.append(outcome.stderr or "")
        if outcome.error and outcome.error.get("traceback"):
            lines.append("--- traceback ---")
            lines.append(outcome.error["traceback"])
        prepare_meta = (outcome.results or {}).get("prepare")
        if prepare_meta:
            lines.append("--- prepare meta ---")
            lines.append(
                prepare_meta if isinstance(prepare_meta, str) else json.dumps(prepare_meta)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ── Sweep orchestration ──────────────────────────────────────────────────────


class ClientPool:
    """One ``kcoral.Client`` per worker thread (an httpx client is not shared)."""

    def __init__(self, api: KcoralApi, url: str) -> None:
        self._api = api
        self._url = url
        self._local = threading.local()
        self._clients: list[Any] = []
        self._lock = threading.Lock()

    def get(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._api.Client(self._url)
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        return client

    def close(self) -> None:
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            try:
                client.close()
            except Exception:
                pass


def _result_line(record: dict) -> str:
    identity = f"{record['kernel']}/{record.get('config') or record.get('label')}"
    remote = record.get("remote") or {}
    timing = ""
    if remote.get("lease_held_ms") is not None:
        timing = (
            f" (lease {remote['lease_held_ms'] / 1000:.1f}s, "
            f"wait {(remote.get('lease_wait_ms') or 0) / 1000:.1f}s, req {remote.get('request_id')})"
        )
    status = record.get("status")
    if status == "ok":
        impls = ", ".join(f"{name}={value:.3f}us" for name, value in record["impls"].items())
        return f"ok   {identity} {impls}{timing}"
    if status == "SKIP":
        return f"SKIP {identity}: {record.get('reason')}{timing}"
    detail = (record.get("error") or "unknown failure").splitlines()[0][:160]
    return f">>> FAIL {identity} attempt {record.get('attempt')}: {detail} <<<{timing}"


def run_remote_jobs(
    workloads: list[dict],
    *,
    api: KcoralApi,
    url: str,
    tree: TreeArchive,
    shim: str,
    profile: ServerProfile,
    log_dir: Path,
    rounds: int,
    cooldown: float,
    with_references: bool,
    max_in_flight: int,
    request_timeout_s: float,
    prepare_mode: str,
    reference_deps_dir: str | None = None,
    policy: RetryPolicy = RetryPolicy(),
) -> tuple[list[dict], dict]:
    """Submit every workload to the server and collect result rows."""
    clients = ClientPool(api, url)
    records: list[dict] = []
    counters = {"busy": 0, "transport": 0}

    def run_one(workload: dict) -> dict:
        kernel, config = workload["kernel"], workload["config"]
        log_path = log_dir / f"{kernel}__{config}.log"
        spec = workload_spec(
            workload,
            cuda_arch=profile.arch,
            num_sms=profile.num_sms,
            references_enabled=with_references,
            prepare_mode=prepare_mode,
            rounds=rounds,
            cooldown=cooldown,
            reference_deps_dir=reference_deps_dir,
        )
        log(f"[bench-suite] {now_iso()} SUBMIT {kernel}/{config}")
        submission, spec = submit_workload(
            api,
            clients.get(),
            tree=tree,
            before_tree=None,
            shim=shim,
            spec=spec,
            timeout_s=request_timeout_s,
            policy=policy,
            on_fallback=lambda reason: log(
                f"[bench-suite] {now_iso()} retry {kernel}/{config} with --prepare gpu: {reason}"
            ),
        )
        write_request_log(log_path, workload=workload, submission=submission)
        return outcome_to_record(
            workload,
            submission,
            profile=profile,
            tree_sha256=tree.sha256,
            before_tree_sha256=None,
            prepare_mode=spec["prepare_mode"],
            rounds=rounds,
            cooldown=cooldown,
            references_enabled=with_references,
            request_timeout_s=request_timeout_s,
            log_path=log_path,
        )

    def guarded(workload: dict) -> dict:
        try:
            return run_one(workload)
        except Exception as error:  # the sweep never dies on one workload
            return {
                "kernel": workload["kernel"],
                "config": workload["config"],
                "label": workload["config"],
                "num_gpus": 1,
                "attempt": 0,
                "execution_mode": EXECUTION_MODE,
                "process_model": PROCESS_MODEL,
                "status": "FAIL",
                "error": f"client: {type(error).__name__}: {error}",
                "gpu": profile.device_uuid or "",
                "gpus": [profile.device_uuid] if profile.device_uuid else [],
                "physical_gpu_uuids": [profile.device_uuid] if profile.device_uuid else [],
                "started_at": now_iso(),
                "finished_at": now_iso(),
                "remote": {"server_url": url, "tree_sha256": tree.sha256},
            }

    executor = ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="bench-remote")
    try:
        futures = [executor.submit(guarded, workload) for workload in workloads]
        for future in as_completed(futures):
            record = future.result()
            remote = record.get("remote") or {}
            counters["busy"] += int(remote.get("busy_retries") or 0)
            counters["transport"] += int(remote.get("transport_retries") or 0)
            records.append(record)
            log(f"[bench-suite] {now_iso()} {_result_line(record)}")
    except KeyboardInterrupt:
        log(
            "[bench-suite] interrupted; cancelling queued workloads (in-flight requests finish or time out on the server)"
        )
        executor.shutdown(wait=False, cancel_futures=True)
        clients.close()
        raise
    executor.shutdown(wait=True)
    clients.close()
    log(
        f"[bench-suite] remote retry summary: busy={counters['busy']} "
        f"transport={counters['transport']}"
    )
    pipeline = pipeline_metadata(
        profile=profile,
        tree=tree,
        rounds=rounds,
        cooldown=cooldown,
        prepare_mode=prepare_mode,
        max_in_flight=max_in_flight,
        request_timeout_s=request_timeout_s,
        records=records,
        busy_retries=counters["busy"],
        transport_retries=counters["transport"],
    )
    pipeline["reference_deps_dir"] = profile.probe.get("reference_deps_dir")
    return records, pipeline


def pipeline_metadata(
    *,
    profile: ServerProfile,
    tree: TreeArchive,
    rounds: int,
    cooldown: float,
    prepare_mode: str,
    max_in_flight: int,
    request_timeout_s: float,
    records: list[dict],
    busy_retries: int = 0,
    transport_retries: int = 0,
) -> dict:
    from tirx_kernels.runner import DEFAULT_BENCH_COOLDOWN_S, DEFAULT_BENCH_ROUNDS

    return {
        "execution_mode": EXECUTION_MODE,
        "process_model": PROCESS_MODEL,
        "measurement_protocol": {
            "rounds": rounds,
            "cooldown_s": cooldown,
            "default_rounds": DEFAULT_BENCH_ROUNDS,
            "default_cooldown_s": DEFAULT_BENCH_COOLDOWN_S,
            "is_default": rounds == DEFAULT_BENCH_ROUNDS and cooldown == DEFAULT_BENCH_COOLDOWN_S,
        },
        "prepare_mode": prepare_mode,
        "cuda_compile_mode": CUDA_COMPILE_MODE,
        "max_in_flight": max_in_flight,
        "request_timeout_s": request_timeout_s,
        "server": profile.summary(),
        "tree_sha256": tree.sha256,
        "busy_retry_count": busy_retries,
        "transport_retry_count": transport_retries,
        "failure_count": sum(1 for record in records if record.get("status") == "FAIL"),
        "prepare_fallback_count": sum(
            1 for record in records if (record.get("remote") or {}).get("prepare_fallback")
        ),
    }


def merge_provenance(
    profile: ServerProfile, *, local_git: dict, local_tree: dict
) -> tuple[dict, dict]:
    """Provenance for the run JSON: TVM identity from the server, kernels from the checkout."""
    git = dict(local_git)
    git["tir"] = profile.tvm_label
    kernel_tree = dict(local_tree)
    kernel_tree["tir:python/tvm/tirx"] = profile.tirx_tree
    return git, kernel_tree


def probe_summary(profile: ServerProfile, *, local_tvm: dict | None = None) -> dict:
    """Top-level ``probe`` entry of the run JSON (informational)."""
    return {"server": profile.summary(), "local_tvm": local_tvm or {}}
