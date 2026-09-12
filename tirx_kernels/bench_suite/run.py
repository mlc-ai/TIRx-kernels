#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""bench-suite: pre-commit regression benchmark for TIRx kernels.

See README.md in this directory for setup, baseline workflow, and flags.

Quick start:
    python -m tirx_kernels.bench_suite
    python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<id>.json

Exit codes:
    0  no regressions (or no baseline yet)
    1  one or more workloads failed
    2  config error (no workloads / bad YAML)
    3  one or more regressions exceeded the threshold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tirx_kernels.bench_suite.provenance import (  # re-exported for ab.py, tests, tvm
    BASELINE_PACKAGES,  # noqa: F401
    collect_baseline_provenance,
    collect_kernel_fingerprint,
    collect_repo_git,
    git_label,  # noqa: F401
    package_provenance,  # noqa: F401
)
from tirx_kernels.bench_suite.provenance import module_repo_root as _module_repo_root  # noqa: F401
from tirx_kernels.bench_suite.provenance import tir_repo_root as _tir_repo_root  # noqa: F401
from tirx_kernels.bench_suite.remote import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SERVER_URL,
    MAX_REQUEST_TIMEOUT_S,
    PREPARE_MODES,
    REFERENCE_DEPS_DIR_ENV,
    SERVER_URL_ENV,
)
from tirx_kernels.registry import kernel_index
from tirx_kernels.runner import DEFAULT_BENCH_COOLDOWN_S as DEFAULT_COOLDOWN_S
from tirx_kernels.runner import DEFAULT_BENCH_ROUNDS as DEFAULT_ROUNDS

try:
    from tirx_kernels.bench_suite.impls import our_impls
except ModuleNotFoundError:  # Support `python tirx_kernels/bench_suite/run.py`.
    from impls import our_impls

SCRIPT_DIR = Path(__file__).resolve().parent


def _kernels_repo_root() -> Path:
    """Git root of the tirx-kernels repo (parent of the tirx_kernels package)."""
    return SCRIPT_DIR.parent.parent


DEFAULT_OUT_DIR = _kernels_repo_root() / ".bench-suite"
# One file per kernel, every config it can bench, each flagged `default:` for
# whether the pinned regression sweep includes it. With no --workloads, the
# flagged ones are assembled into a generated workload file and benched.
CONFIG_DIR = SCRIPT_DIR / "config"
GENERATED_WORKLOADS_NAME = "workloads.generated.yaml"
MAX_DEFAULT_CONFIGS_PER_KERNEL = 3
DEFAULT_SELECTION_ROLES = ("small", "medium", "large")
# Single pinned baseline: every suite run benches our kernel directly. External
# references run only under the explicit ``--with-references`` diagnostic flag
# (also available on ``python -m tirx_kernels.bench``) and feed the ratio
# report; they never replace the direct before/after verdict.
DEFAULT_BASELINE = SCRIPT_DIR / "baseline.json"
DEFAULT_REGRESSION_THRESHOLD = 1.0


# ── Workload loading ─────────────────────────────────────────────────────────


def _normalize_workload(workload: dict) -> dict:
    """Apply the field rules every loader shares."""
    if "kernel" not in workload or "config" not in workload:
        raise ValueError(f"workload missing kernel/config: {workload}")
    num_gpus = workload.get("num_gpus", 1)
    if type(num_gpus) is not int or num_gpus < 1:
        raise ValueError(f"workload num_gpus must be a positive integer: {workload}")
    workload["num_gpus"] = num_gpus
    if workload.get("timer") == "megamoe" and (
        workload.get("warmup") is not None or workload.get("repeat") is not None
    ):
        raise ValueError(
            "timer='megamoe' uses a fixed DeepGEMM protocol and cannot override "
            f"warmup/repeat: {workload}"
        )
    return workload


def _read_kernel_config(path: Path) -> tuple[str, list[dict], str | None]:
    data = yaml.safe_load(path.read_text()) or {}
    kernel = data.get("kernel")
    if not kernel:
        raise ValueError(f"{path.name}: missing top-level 'kernel'")
    if path.stem != kernel:
        raise ValueError(f"{path.name}: kernel {kernel!r} must match the file stem")
    default_suite = data.get("default_suite", True)
    if type(default_suite) is not bool:
        raise ValueError(f"{path.name}: default_suite must be true or false")
    defaults = data.get("defaults") or {}
    entries = []
    labels: set[str] = set()
    default_count = 0
    for entry in data.get("configs") or []:
        if "config" not in entry:
            raise ValueError(f"{path.name}: config entry missing 'config': {entry}")
        label = entry["config"]
        if not isinstance(label, str) or not label:
            raise ValueError(f"{path.name}: config label must be a non-empty string: {entry}")
        if label in labels:
            raise ValueError(f"{path.name}: duplicate config key {kernel}/{label}")
        labels.add(label)
        if type(entry.get("default")) is not bool:
            raise ValueError(f"{path.name}: config {label!r} must declare default: true or false")
        if entry["default"]:
            default_count += 1
        selection_role = entry.get("selection_role")
        if selection_role is not None and selection_role not in DEFAULT_SELECTION_ROLES:
            raise ValueError(
                f"{path.name}: selection_role must be one of "
                f"{DEFAULT_SELECTION_ROLES}, got {selection_role!r}"
            )
        entries.append({"kernel": kernel, **defaults, **entry})
    if not default_suite and default_count:
        raise ValueError(
            f"{path.name}: default_suite=false requires every config to be non-default"
        )
    if default_suite and not default_count:
        raise ValueError(f"{path.name}: default_suite kernels require at least one default config")
    if default_count > MAX_DEFAULT_CONFIGS_PER_KERNEL:
        raise ValueError(
            f"{path.name}: kernel {kernel!r} has {default_count} default configs; "
            f"maximum is {MAX_DEFAULT_CONFIGS_PER_KERNEL}"
        )
    selection_rationale = data.get("selection_rationale")
    if selection_rationale is not None and (
        not isinstance(selection_rationale, str) or not selection_rationale.strip()
    ):
        raise ValueError(f"{path.name}: selection_rationale must be a non-empty string")
    if len(entries) > MAX_DEFAULT_CONFIGS_PER_KERNEL and default_count == 3:
        if selection_rationale is None:
            raise ValueError(
                f"{path.name}: curated three-point default selection requires selection_rationale"
            )
        default_roles = {
            entry.get("selection_role") for entry in entries if entry.get("default", False)
        }
        if default_roles != set(DEFAULT_SELECTION_ROLES):
            raise ValueError(
                f"{path.name}: curated defaults must have exactly the roles "
                f"{DEFAULT_SELECTION_ROLES}, got {sorted(default_roles, key=str)}"
            )
        nondefault_roles = [
            entry["config"]
            for entry in entries
            if not entry.get("default", False) and entry.get("selection_role") is not None
        ]
        if nondefault_roles:
            raise ValueError(
                f"{path.name}: non-default configs cannot have selection_role: {nondefault_roles}"
            )
    elif selection_rationale is not None:
        raise ValueError(
            f"{path.name}: selection_rationale is only valid for a curated "
            "three-of-many default selection"
        )
    elif any(entry.get("selection_role") is not None for entry in entries):
        raise ValueError(
            f"{path.name}: selection_role is only valid for a curated "
            "three-of-many default selection"
        )
    return kernel, entries, selection_rationale


def load_kernel_configs(kernel: str, config_dir: Path = CONFIG_DIR) -> list[dict]:
    """Every config one kernel can bench, flagged for the pinned sweep or not.

    Entries keep their ``default`` flag; :func:`load_config_dir` is the filtered
    view the regression gate runs.
    """
    matches = sorted(config_dir.rglob(f"{kernel}.yaml"))
    if not matches:
        raise FileNotFoundError(f"no config file for kernel {kernel!r} under {config_dir}")
    if len(matches) > 1:
        raise ValueError(f"kernel {kernel!r} has more than one config file: {matches}")
    path = matches[0]
    _, entries, _selection_rationale = _read_kernel_config(path)
    return [
        _normalize_workload({key: value for key, value in entry.items() if key != "selection_role"})
        for entry in entries
    ]


def load_config_dir(config_dir: Path = CONFIG_DIR) -> list[dict]:
    """The pinned sweep: every `default: true` config across ``config/**/*.yaml``.

    Each file is one kernel's complete benchable matrix, so which configs the
    regression gate covers is a per-line flag rather than a separate file.  The
    files are bucketed to mirror the kernel tree, so the walk is recursive.
    """
    files = sorted(config_dir.rglob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no kernel config files under {config_dir}")
    out: list[dict] = []
    configured: dict[str, Path] = {}
    for path in files:
        kernel, entries, _selection_rationale = _read_kernel_config(path)
        previous = configured.get(kernel)
        if previous is not None:
            raise ValueError(f"kernel {kernel!r} has more than one config file: {previous}, {path}")
        configured[kernel] = path
        for entry in entries:
            if not entry.pop("default", False):
                continue
            entry.pop("selection_role", None)
            workload = _normalize_workload(entry)
            if workload["num_gpus"] != 1:
                raise ValueError(
                    "default measured sweep must remain single-GPU; run multi-GPU "
                    f"workload explicitly instead: {workload}"
                )
            out.append(workload)

    registered = set(kernel_index(strict=True))
    configured_names = set(configured)
    if registered != configured_names:
        missing = sorted(registered - configured_names)
        unknown = sorted(configured_names - registered)
        details = []
        if missing:
            details.append(f"missing config YAML for: {', '.join(missing)}")
        if unknown:
            details.append(f"config YAML has unregistered kernel(s): {', '.join(unknown)}")
        raise ValueError("registry/config mismatch: " + "; ".join(details))
    return out


def write_generated_workloads(workloads: list[dict], path: Path) -> Path:
    """Materialize the assembled sweep so a run's exact input is inspectable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by bench_suite from tirx_kernels/bench_suite/config/**/*.yaml\n"
        "# (every config flagged `default: true`). Do not edit -- rewritten each run.\n"
    )
    path.write_text(
        header + yaml.safe_dump({"defaults": {}, "workloads": workloads}, sort_keys=False)
    )
    return path


def load_workloads(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults") or {}
    return [_normalize_workload({**defaults, **entry}) for entry in data.get("workloads") or []]


# ── Tee stdout → run log ─────────────────────────────────────────────────────


class _Tee:
    """Write to multiple streams; flush on every write so the log is live.

    Locks per write so two threads' simultaneous writes don't interleave
    bytes. For atomic *lines*, callers should still hold _log_lock around
    the full print+flush sequence — see log() below.
    """

    def __init__(self, *streams):
        self._streams = streams
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            for st in self._streams:
                st.write(s)
                st.flush()
        return len(s)

    def flush(self):
        with self._lock:
            for st in self._streams:
                st.flush()


# Thread-safe one-liner emitter. `print()` calls file.write() multiple times
# (once for the message, once for the trailing newline), so without this
# lock concurrent prints from worker threads can interleave halfway through
# a line. Use log() for any [bench-suite] status print from a worker thread.
_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)


# ── Workload architecture filtering ─────────────────────────────────────────


def partition_workloads_by_arch(
    workloads: list[dict], cuda_arch: str
) -> tuple[list[dict], list[dict]]:
    """Partition workloads by an exact registered CUDA architecture."""
    records = kernel_index(strict=True)
    supported: list[dict] = []
    incompatible: list[dict] = []
    for workload in workloads:
        destination = (
            supported
            if cuda_arch in records[workload["kernel"]].runtime_cuda_archs
            else incompatible
        )
        destination.append(workload)
    return supported, incompatible


def validate_workload_archs(workloads: list[dict], cuda_arch: str) -> None:
    """Reject workloads that are not registered for the selected exact architecture."""
    _supported, incompatible_workloads = partition_workloads_by_arch(workloads, cuda_arch)
    incompatible = sorted({workload["kernel"] for workload in incompatible_workloads})
    if incompatible:
        raise ValueError(
            f"selected GPU architecture {cuda_arch} is unsupported by workload kernel(s): "
            + ", ".join(incompatible)
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Output ───────────────────────────────────────────────────────────────────


def write_run(
    out_dir: Path,
    stamp: str,
    results: list[dict],
    label: str | None,
    *,
    selection: dict,
    references_enabled: bool,
    git: dict | None = None,
    kernel_tree: dict | None = None,
    baselines: dict | None = None,
    probe: dict | None = None,
    pipeline: dict | None = None,
) -> Path:
    """Write ``runs/<stamp>.json``.

    Provenance defaults to this process (local git checkouts, locally importable
    baseline packages); the remote backend passes the server's identity instead.
    """
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if git is None:
        git = collect_repo_git()
    if kernel_tree is None:
        kernel_tree = collect_kernel_fingerprint()
    if baselines is None:
        baselines = collect_baseline_provenance() if references_enabled else {}
    payload = {
        "timestamp": stamp,
        "label": label,
        "references_enabled": references_enabled,
        "git": git,
        "kernel_tree": kernel_tree,
        "baselines": baselines,
        "selection": selection,
        "probe": probe or {},
        "pipeline": pipeline or {},
        "results": results,
    }
    path = runs_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


#: Which reference a kernel's ratio is quoted against, for the kernels that
#: benchmark against more than one.  A kernel with a single non-ours impl needs
#: no entry -- `_baseline_impl` derives it from the row.
BASELINE_IMPL_BY_KERNEL = {
    "fp16_bf16_gemm": "torch-cublas",
    "nvfp4_gemm": "flashinfer",
    "flashkda_bf16_fused_m128": "flashinfer_m128",
    "deepgemm_sm100_fp8_paged_mqa_logits": "deepgemm",
    "sparse_flashmla_prefill_head64_phase1": "flashmla",
    "sparse_flashmla_prefill_head128_phase1": "flashmla",
}


def _baseline_impl(kernel: str, impl_names: list[str]) -> str | None:
    """The reference a kernel's ratio is quoted against.

    Most kernels have exactly one non-ours implementation, so naming it in a
    hand-maintained table only creates a row that can be forgotten -- and four
    kernels already had been, printing a bare `ratio` header.  The table now
    only breaks genuine ties.
    """
    pinned = BASELINE_IMPL_BY_KERNEL.get(kernel)
    if pinned:
        return pinned
    ours = set(our_impls(dict.fromkeys(impl_names)))
    others = [name for name in impl_names if name not in ours]
    return others[0] if len(others) == 1 else None


def _our_impl(row_impls: dict) -> str | None:
    """Pick the first TIR/TIRx implementation from a row's impls dict."""
    return next(iter(our_impls(row_impls)), None)


def write_summary(out_dir: Path, current: dict) -> Path:
    """Human-readable per-run report, grouped by kernel.

    Times are in µs to match the existing bench-suite doc convention. Per row:
    config, one column per impl present in that kernel, baseline/ours ratio
    (against the kernel's reference impl from BASELINE_IMPL_BY_KERNEL),
    then attempt + gpu.
    """
    stamp = current["timestamp"]
    reports_dir = out_dir / "reports" / stamp
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# bench-suite run {stamp}")
    lines.append("")
    label = current.get("label") or "-"
    git = current.get("git") or {}
    lines.append(f"- label: `{label}`")
    lines.append(
        f"- git: tir=`{git.get('tir') or '-'}`  "
        f"tirx-kernels=`{git.get('tirx-kernels') or '-'}`  "
        f"tirx-bench-ci=`{git.get('tirx-bench-ci') or '-'}`"
    )
    statuses: dict[str, int] = {}
    for r in current.get("results") or []:
        s = r.get("status") or "?"
        statuses[s] = statuses.get(s, 0) + 1
    status_line = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    lines.append(f"- status: {status_line} (over {sum(statuses.values())} workloads)")
    lines.append("")

    baselines = current.get("baselines") or {}
    if baselines:
        lines.append("## Baseline impl provenance")
        lines.append("")
        for name, info in sorted(baselines.items()):
            if not info or info.get("installed") is False:
                lines.append(f"- `{name}`: not installed")
                continue
            bits = []
            if "version" in info:
                bits.append(f"v{info['version']}")
            if "cuda" in info:
                bits.append(f"cuda={info['cuda']}")
            if "torch_git_version" in info:
                bits.append(f"torch_git={info['torch_git_version'][:12]}")
            if "git_sha" in info:
                bits.append(f"@`{info['git_sha']}`")
            if "git_dir" in info:
                bits.append(f"({info['git_dir']})")
            lines.append(f"- `{name}`: {' '.join(bits) if bits else '?'}")
        lines.append("")

    # Group by kernel
    by_kernel: dict[str, list[dict]] = {}
    for r in current.get("results") or []:
        by_kernel.setdefault(r["kernel"], []).append(r)

    for kernel in sorted(by_kernel):
        rows = sorted(by_kernel[kernel], key=lambda r: r.get("label") or r.get("config") or "")
        # Discover all impl names that appear in this kernel
        impl_names: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for impl in r.get("impls") or {}:
                if impl not in seen:
                    seen.add(impl)
                    impl_names.append(impl)
        impl_names.sort()
        baseline_impl = _baseline_impl(kernel, impl_names)
        # Determine "ours" impl name once for the whole kernel (constant per kernel)
        ours_impl = None
        for r in rows:
            ours_impl = _our_impl(r.get("impls") or {})
            if ours_impl:
                break
        ratio_label = f"{baseline_impl}/{ours_impl}" if baseline_impl and ours_impl else "ratio"
        lines.append(f"## `{kernel}`")
        if baseline_impl and ours_impl:
            lines.append("")
            lines.append(
                f"_baseline impl_: `{baseline_impl}` · _ours_: `{ours_impl}` · "
                f"_ratio_ = baseline/ours · `>1` means ours is faster"
            )
        lines.append("")
        # Table header
        header = ["config", *impl_names, ratio_label, "attempt", "gpus"]
        align = ["---"] + ["---:"] * len(impl_names) + ["---:", "---:", "---:"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(align) + "|")
        for r in rows:
            cfg = r.get("label") or r.get("config") or "?"
            status = r.get("status", "ok")
            impls = r.get("impls") or {}
            row = [cfg]
            for impl in impl_names:
                us = impls.get(impl)
                row.append(f"{us:.2f}" if us is not None else "—")
            # Ratio column
            ratio_cell = "—"
            if baseline_impl and ours_impl:
                base_us = impls.get(baseline_impl)
                ours_us = impls.get(ours_impl)
                if base_us is not None and ours_us is not None and ours_us > 0:
                    ratio = base_us / ours_us
                    # Bold values that flag a regression risk (we're slower)
                    ratio_cell = f"**{ratio:.3f}**" if ratio < 1.0 else f"{ratio:.3f}"
            row.append(ratio_cell)
            if status != "ok":
                row[0] = f"{cfg} **[{status}]**"
            row.append(str(r.get("attempt", 1)))
            row.append(str(r.get("gpu", "-")))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path = reports_dir / "summary.md"
    path.write_text("\n".join(lines))
    return path


def load_baseline(path=None):
    """Load the pinned baseline.json, or None if no baseline exists yet.

    ``path`` (optional) overrides the default baseline location."""
    p = Path(path) if path is not None else DEFAULT_BASELINE
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── Main ─────────────────────────────────────────────────────────────────────


def _finalize_bench_record(
    row: dict, *, rounds: int, cooldown: float, references_enabled: bool
) -> None:
    """Validate in-bench round samples and write aggregated impl times (microseconds)."""
    required_fields = ("round_samples", "errors", "timer", "benchmark_protocol")
    missing_fields = [field for field in required_fields if field not in row]
    if missing_fields:
        row["status"] = "FAIL"
        row["error"] = f"bench result is missing required field(s): {missing_fields}"
        return
    if not isinstance(row["errors"], dict):
        row["status"] = "FAIL"
        row["error"] = "bench result field 'errors' must be a mapping"
        return
    if not isinstance(row["timer"], str) or not row["timer"]:
        row["status"] = "FAIL"
        row["error"] = "bench result field 'timer' must be a non-empty string"
        return
    protocol = row["benchmark_protocol"]
    if not isinstance(protocol, dict):
        row["status"] = "FAIL"
        row["error"] = "bench result field 'benchmark_protocol' must be a mapping"
        return
    if protocol.get("rounds") != rounds:
        row["status"] = "FAIL"
        row["error"] = (
            "benchmark protocol round count does not match suite request: "
            f"{protocol.get('rounds')!r} != {rounds}"
        )
        return
    if protocol.get("round_aggregate") != "mean":
        row["status"] = "FAIL"
        row["error"] = "benchmark protocol must declare round_aggregate='mean'"
        return
    protocol_cooldown = protocol.get("cooldown_s", protocol.get("round_cooldown_s"))
    if (
        not isinstance(protocol_cooldown, int | float)
        or isinstance(protocol_cooldown, bool)
        or not math.isclose(float(protocol_cooldown), cooldown, rel_tol=0.0, abs_tol=1e-9)
    ):
        row["status"] = "FAIL"
        row["error"] = (
            "benchmark protocol cooldown does not match suite request: "
            f"{protocol_cooldown!r} != {cooldown}"
        )
        return

    baseline_errors = row["errors"]
    if baseline_errors:
        details = "; ".join(f"{name}: {error}" for name, error in baseline_errors.items())
        row["status"] = "FAIL"
        row["error"] = f"baseline error(s): {details}"
        return
    samples = row["round_samples"]
    if not isinstance(samples, dict) or not samples:
        row["status"] = "FAIL"
        row["error"] = "bench result field 'round_samples' must be a non-empty mapping"
        return
    if not references_enabled:
        external_impls = [name for name in samples if name not in our_impls(samples)]
        if external_impls:
            row["status"] = "FAIL"
            row["error"] = (
                "reference-disabled benchmark reported external implementation(s): "
                f"{external_impls}"
            )
            return
    bad = {
        impl: len(vals) if isinstance(vals, list) else type(vals).__name__
        for impl, vals in samples.items()
        if not isinstance(vals, list) or len(vals) != rounds
    }
    if bad:
        row["status"] = "FAIL"
        row["error"] = f"expected {rounds} round(s) per impl, got {bad}"
        return
    invalid = {
        impl: value
        for impl, values in samples.items()
        for value in values
        if not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    }
    if invalid:
        row["status"] = "FAIL"
        row["error"] = f"round samples must be finite positive numbers: {invalid}"
        return
    sample_order = list(samples)
    if "order" in protocol:
        if protocol["order"] != sample_order:
            row["status"] = "FAIL"
            row["error"] = (
                "benchmark protocol implementation order does not match round samples: "
                f"{protocol['order']!r} != {sample_order!r}"
            )
            return
    else:
        round_orders = protocol.get("round_orders")
        if (
            not isinstance(round_orders, list)
            or len(round_orders) != rounds
            or any(
                not isinstance(order, list)
                or len(order) != len(sample_order)
                or set(order) != set(sample_order)
                for order in round_orders
            )
        ):
            row["status"] = "FAIL"
            row["error"] = (
                "benchmark protocol must declare one implementation order or one valid "
                "permutation per round"
            )
            return
    row["impls"] = {impl: statistics.mean(vals) for impl, vals in samples.items()}
    row["aggregated"] = {"rounds": rounds, "method": "mean"}
    row["status"] = "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="bench-suite: pre-commit regression benchmark")
    ap.add_argument(
        "--workloads",
        type=Path,
        default=None,
        help="YAML file listing kernels/configs to bench (default: assemble every "
        "`default: true` config from bench_suite/config/**/*.yaml)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to store runs/, logs/, reports/, latest.json "
        "(default: <tirx-kernels>/.bench-suite)",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON to diff against instead of the pinned baseline.json",
    )
    ap.add_argument(
        "--ab-before",
        type=str,
        default=None,
        metavar="REV",
        help="Run REV and the current committed checkout as a paired A/B campaign; "
        "both sides of a workload run back to back on the benchmark server",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help="Regression threshold in percent slowdown (default 1)",
    )
    ap.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only keep workloads whose kernel contains this substring",
    )
    ap.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form label for this run (default: git short sha)",
    )
    ap.add_argument("--no-report", action="store_true", help="Skip regression report generation")
    ap.add_argument(
        "--with-references",
        action="store_true",
        help="Run and benchmark external reference implementations (off by default)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Independent standard-timer samples per workload (default "
        f"{DEFAULT_ROUNDS}). Compile/prepare once; each round cools down and runs "
        "a complete timer call.",
    )
    ap.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help=f"Seconds to sleep before every implementation (default {DEFAULT_COOLDOWN_S})",
    )
    ap.add_argument(
        "--server",
        type=str,
        default=os.environ.get(SERVER_URL_ENV, DEFAULT_SERVER_URL),
        help=f"kcoral benchmark server URL (default: $TIRX_BENCH_SERVER or {DEFAULT_SERVER_URL})",
    )
    ap.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Maximum concurrent requests to the server (default: the server's worker count; "
        "A/B campaigns default to 1 so both sides of a pair run back to back)",
    )
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Per-workload server execution timeout in seconds, excluding queue wait "
        f"(default {DEFAULT_REQUEST_TIMEOUT_S:g}, max {MAX_REQUEST_TIMEOUT_S:g})",
    )
    ap.add_argument(
        "--prepare",
        choices=PREPARE_MODES,
        default="cpu",
        help="Where the worker compiles: 'cpu' releases the GPU lease during prepare "
        "(default), 'gpu' keeps it (fallback if a prepare touches CUDA)",
    )
    ap.add_argument(
        "--reference-deps-dir",
        type=str,
        default=os.environ.get(REFERENCE_DEPS_DIR_ENV),
        metavar="SERVER_PATH",
        help="Directory ON THE SERVER holding the pinned reference checkouts "
        "(scripts/install_reference_dependencies.py output); linked next to the shipped "
        f"tree as .reference-deps (default: ${REFERENCE_DEPS_DIR_ENV})",
    )
    ap.add_argument(
        "--check-imports",
        action="store_true",
        help="Import every unique kernel in --workloads and exit (for CI import gates)",
    )
    args = ap.parse_args()
    if args.rounds < 1:
        print("[bench-suite] --rounds must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.cooldown < 0:
        print("[bench-suite] --cooldown must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.max_in_flight is not None and args.max_in_flight < 1:
        print("[bench-suite] --max-in-flight must be >= 1", file=sys.stderr)
        sys.exit(2)
    if not 0 < args.request_timeout <= MAX_REQUEST_TIMEOUT_S:
        print(
            f"[bench-suite] --request-timeout must be in (0, {MAX_REQUEST_TIMEOUT_S:g}]",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.workloads is None:
        assembled = load_config_dir()
        workloads_path = write_generated_workloads(
            assembled, args.out_dir.resolve() / GENERATED_WORKLOADS_NAME
        )
        print(
            f"[bench-suite] assembled {len(assembled)} default workload(s) "
            f"from {CONFIG_DIR}/*.yaml -> {workloads_path}"
        )
    else:
        workloads_path = args.workloads

    workloads = load_workloads(workloads_path)
    if args.filter:
        workloads = [w for w in workloads if args.filter in w["kernel"]]
    if not workloads:
        print("[bench-suite] no workloads to run.", file=sys.stderr)
        sys.exit(2)
    multi_gpu = [w for w in workloads if w.get("num_gpus", 1) != 1]
    if multi_gpu and not args.check_imports:
        names = sorted({f"{w['kernel']}/{w['config']}" for w in multi_gpu})
        print(
            "[bench-suite] multi-GPU workloads cannot run on the remote backend "
            f"(one GPU per server worker): {', '.join(names)}",
            file=sys.stderr,
        )
        sys.exit(2)

    selection = {
        "mode": "default" if args.workloads is None and args.filter is None else "targeted",
        "keys": [[workload["kernel"], workload["config"]] for workload in workloads],
    }

    if args.check_imports:
        if args.ab_before:
            print(
                "[bench-suite] --check-imports cannot be combined with --ab-before", file=sys.stderr
            )
            sys.exit(2)
        from tirx_kernels.registry import check_workload_imports

        names = check_workload_imports(workloads, strict=True)
        print(f"[bench-suite] import check ok ({len(names)} kernels from {workloads_path})")
        return

    if args.ab_before:
        if args.baseline is not None:
            print("[bench-suite] --baseline cannot be combined with --ab-before", file=sys.stderr)
            sys.exit(2)
        if args.with_references:
            print(
                "[bench-suite] --with-references cannot be combined with --ab-before; "
                "the A/B verdict is direct before/after time on our implementation only",
                file=sys.stderr,
            )
            sys.exit(2)
        from tirx_kernels.bench_suite.ab import run_ab

        try:
            exit_code = run_ab(
                workloads,
                selection=selection,
                before_revision=args.ab_before,
                out_dir=args.out_dir,
                label=args.label,
                threshold=args.threshold,
                rounds=args.rounds,
                cooldown=args.cooldown,
                no_report=args.no_report,
                server_url=args.server,
                max_in_flight=args.max_in_flight,
                request_timeout_s=args.request_timeout,
                prepare_mode=args.prepare,
            )
        except ValueError as error:
            print(f"[bench-suite] invalid A/B configuration: {error}", file=sys.stderr)
            sys.exit(2)
        except RuntimeError as error:
            print(f"[bench-suite] A/B setup failed: {error}", file=sys.stderr)
            sys.exit(1)
        sys.exit(exit_code)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Run id: incrementing integer — one more than the highest existing numeric
    # run in runs/ (runs/7.json, reports/7/, latest -> 7).
    _existing = [int(p.stem) for p in runs_dir.glob("*.json") if p.stem.isdigit()]
    stamp = str(max(_existing, default=0) + 1)
    run_log_path = runs_dir / f"{stamp}.log"
    run_log_fh = open(run_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, run_log_fh)
    sys.stderr = _Tee(sys.stderr, run_log_fh)
    # Repoint `latest.log` symlink immediately so `tail -f .bench-suite/latest.log`
    # picks up this run before any output happens.
    latest_log = out_dir / "latest.log"
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(run_log_path.relative_to(out_dir))

    print(f"[bench-suite] live log: {run_log_path}")
    print(f"[bench-suite]   tail : tail -f {latest_log}")
    print(f"[bench-suite] run id : {stamp}")

    # ── Benchmark server handshake ──
    # The server owns GPU exclusivity: one lease per GPU, a fresh worker process
    # per request. The suite never touches a local GPU.
    from tirx_kernels.bench_suite import remote

    try:
        api = remote.load_kcoral()
        client, health = remote.connect(api, args.server)
    except remote.RemoteBenchError as error:
        print(f"[bench-suite] {error}", file=sys.stderr)
        sys.exit(1)
    cuda_arch = health["target"]["arch"]
    print(
        f"[bench-suite] server {args.server}: arch={cuda_arch} gpus={health.get('gpu_count')} "
        f"workers={len(health.get('workers') or [])} queue={health.get('queue_length')} "
        f"versions={health.get('versions')}",
        flush=True,
    )
    if selection["mode"] == "default":
        workloads, incompatible_workloads = partition_workloads_by_arch(workloads, cuda_arch)
        if not workloads:
            print(
                f"[bench-suite] default roster has no workloads for {cuda_arch}.", file=sys.stderr
            )
            sys.exit(2)
        if incompatible_workloads:
            incompatible_kernels = {workload["kernel"] for workload in incompatible_workloads}
            print(
                f"[bench-suite] selected {len(workloads)} {cuda_arch} default workload(s); "
                f"excluded {len(incompatible_workloads)} workload(s) across "
                f"{len(incompatible_kernels)} incompatible kernel(s)",
                flush=True,
            )
    else:
        try:
            validate_workload_archs(workloads, cuda_arch)
        except ValueError as error:
            print(f"[bench-suite] incompatible workload architecture: {error}", file=sys.stderr)
            sys.exit(2)
    selection["cuda_arch"] = cuda_arch
    selection["keys"] = [[workload["kernel"], workload["config"]] for workload in workloads]

    tree = remote.build_tree_archive(_kernels_repo_root() / "tirx_kernels")
    shim = remote.shim_source()
    print(
        f"[bench-suite] tree archive: {tree.file_count} files, {tree.byte_count} bytes -> "
        f"{len(tree.data)} bytes gz, sha256={tree.sha256[:12]}",
        flush=True,
    )
    try:
        profile = remote.probe_server(
            api,
            client,
            url=args.server,
            health=health,
            tree=tree,
            shim=shim,
            references_enabled=args.with_references,
            timeout_s=min(args.request_timeout, remote.PROBE_TIMEOUT_S),
            reference_deps_dir=args.reference_deps_dir,
        )
    except remote.RemoteBenchError as error:
        print(f"[bench-suite] {error}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()
    max_in_flight = args.max_in_flight or remote.default_max_in_flight(health)
    device = profile.device
    print(
        f"[bench-suite] worker: {device.get('name')} uuid={device.get('uuid')} "
        f"sms={device.get('multi_processor_count')} tvm={profile.tvm_label} "
        f"tirx-tree={profile.tirx_tree[:19]} "
        f"reference-deps={profile.probe.get('reference_deps_dir')}",
        flush=True,
    )

    git = collect_repo_git()
    git, kernel_tree = remote.merge_provenance(
        profile, local_git=git, local_tree=collect_kernel_fingerprint()
    )
    label = args.label or git.get("tirx-kernels") or git.get("tir") or "local"
    agg_note = (
        f", {args.rounds} standard-timer round(s), aggregate=mean, "
        f"cooldown={args.cooldown:g}s before every impl/round"
        if args.rounds > 1 or args.cooldown > 0
        else ""
    )
    print(
        f"[bench-suite] {len(workloads)} workloads, max-in-flight={max_in_flight}, "
        f"prepare={args.prepare}, request-timeout={args.request_timeout:g}s, "
        f"label={label}{agg_note}",
        flush=True,
    )

    results, pipeline_meta = remote.run_remote_jobs(
        workloads,
        api=api,
        url=args.server,
        tree=tree,
        shim=shim,
        profile=profile,
        log_dir=log_dir,
        rounds=args.rounds,
        cooldown=args.cooldown,
        with_references=args.with_references,
        max_in_flight=max_in_flight,
        request_timeout_s=args.request_timeout,
        prepare_mode=args.prepare,
        reference_deps_dir=args.reference_deps_dir,
    )

    results.sort(key=lambda r: (r["kernel"], r.get("label") or r.get("config")))
    local_tvm = {
        "git_label": collect_repo_git().get("tir"),
        "tree": collect_kernel_fingerprint().get("tir:python/tvm/tirx"),
    }
    run_path = write_run(
        out_dir,
        stamp,
        results,
        label,
        selection=selection,
        references_enabled=args.with_references,
        git=git,
        kernel_tree=kernel_tree,
        baselines=profile.baselines,
        probe=remote.probe_summary(profile, local_tvm=local_tvm),
        pipeline=pipeline_meta,
    )
    current = json.loads(run_path.read_text())

    latest = out_dir / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_path.relative_to(out_dir))

    summary_path = write_summary(out_dir, current)
    print(f"[bench-suite] wrote {run_path}")
    print(f"[bench-suite] wrote {summary_path}")

    failures = [record for record in results if record.get("status") == "FAIL"]
    if failures:
        print(f"[bench-suite] workload failure summary: {len(failures)}", file=sys.stderr)
        for failure in failures:
            detail = (failure.get("error") or "unknown workload failure").splitlines()[0]
            print(
                f"[bench-suite]   - {failure['kernel']}/"
                f"{failure.get('config') or failure.get('label')}: {detail}",
                file=sys.stderr,
            )
        sys.exit(1)

    if args.no_report:
        return

    if not args.with_references:
        print(
            "[bench-suite] references are disabled; skipping the reference-ratio "
            "regression report (rerun with --with-references to compare ratios)"
        )
        return

    # Single pinned baseline (baseline.json). Promote a fresh run over it via
    # promote_baseline.py.
    baseline = load_baseline(args.baseline)
    if baseline is None:
        print("[bench-suite] no baseline (baseline.json) — skipping regression report")
        print(f"[bench-suite]   set baseline: promote_baseline.py {run_path}")
        return

    reports_dir = out_dir / "reports" / current["timestamp"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    # keep reports/latest pointing at the most recent run's folder
    reports_latest = out_dir / "reports" / "latest"
    if reports_latest.exists() or reports_latest.is_symlink():
        reports_latest.unlink()
    reports_latest.symlink_to(current["timestamp"])

    sys.path.insert(0, str(SCRIPT_DIR))
    from ratio_diff import build_report as _build_bench_report

    try:
        bench_md, n_regress = _build_bench_report(baseline, current, threshold_pct=args.threshold)
    except Exception as e:
        print(f"[bench-suite] bench report failed: {e}", file=sys.stderr)
        sys.exit(3)

    bench_path = reports_dir / "bench.md"
    bench_path.write_text(bench_md)
    print(f"[bench-suite] wrote {bench_path}\n")
    print(bench_md)

    if n_regress > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
