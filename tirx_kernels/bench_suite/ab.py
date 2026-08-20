# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Same-GPU paired before/after orchestration for the bench suite."""

from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tirx_kernels.bench_suite.impls import our_impls
from tirx_kernels.bench_suite.ratio_diff import build_report
from tirx_kernels.runner import AB_CURRENT_BENCHMARK_ROOT_ENV

_SHARED_HARNESS_PATHS = (
    Path("tirx_kernels/bench"),
    Path("tirx_kernels/bench_suite"),
    Path("tirx_kernels/runner.py"),
    Path("tirx_kernels/low_level_ir.py"),
    Path("tirx_kernels/basic/utils/_runtime.py"),
    # The before child execs the *current* kernel module for the benchmark
    # contract; that module needs the current kern substrate, which old
    # revisions may lack (and old kernels never import it themselves).
    Path("tirx_kernels/kern"),
)


class _InterferenceError(RuntimeError):
    """One side was retried, so the complete pair must be discarded."""


@dataclass(frozen=True)
class _PairResult:
    index: int
    workload: dict[str, Any]
    gpu_index: str
    gpu_uuid: str
    order: tuple[str, str]
    attempt: int
    before: dict[str, Any]
    after: dict[str, Any]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repository_state(repo: Path) -> tuple[str, str, str]:
    """Exact tracked state used to reject edits during a campaign."""

    return (
        _git(repo, "status", "--porcelain", "--untracked-files=all"),
        _git(repo, "diff", "--binary", "HEAD"),
        _git(repo, "diff", "--cached", "--binary", "HEAD"),
    )


def _copy_shared_harness(after_root: Path, before_root: Path) -> None:
    for relative in _SHARED_HARNESS_PATHS:
        source = after_root / relative
        destination = before_root / relative
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _extract_before_tree(repo: Path, revision: str, destination: Path) -> None:
    archive = destination.parent / "before.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", f"--output={archive}", revision],
        check=True,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def _workload_key(workload: dict[str, Any]) -> tuple[str, str]:
    return workload["kernel"], workload["config"]


def _write_workload(path: Path, workload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"defaults": {}, "workloads": [workload]}, sort_keys=False))


def _run_payload(out_dir: Path) -> dict[str, Any]:
    paths = sorted((out_dir / "runs").glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one run JSON under {out_dir}, found {paths}")
    return json.loads(paths[0].read_text())


def _validate_side_payload(
    side: str, payload: dict[str, Any], workload: dict[str, Any], *, gpu_uuid: str
) -> None:
    kernel, config = _workload_key(workload)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"{side} {kernel}/{config}: expected exactly one result")
    row = results[0]
    if (row.get("kernel"), row.get("config") or row.get("label")) != (kernel, config):
        raise RuntimeError(f"{side} {kernel}/{config}: result identity differs")
    if row.get("status") != "ok":
        raise RuntimeError(f"{side} {kernel}/{config}: {row.get('error') or row.get('status')!r}")
    if row.get("physical_gpu_uuids") != [gpu_uuid]:
        raise RuntimeError(
            f"{side} {kernel}/{config}: expected GPU {gpu_uuid}, "
            f"got {row.get('physical_gpu_uuids')!r}"
        )
    samples = row.get("round_samples")
    if not isinstance(samples, dict) or len(our_impls(samples)) != 1:
        raise RuntimeError(f"{side} {kernel}/{config}: expected one TIR/TIRx implementation")
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict):
        raise RuntimeError(f"{side} {kernel}/{config}: missing pipeline provenance")
    if (
        pipeline.get("interference_retry_count")
        or pipeline.get("interference_retries")
        or row.get("interfered")
        or row.get("retry_in_place")
    ):
        raise _InterferenceError(f"{side} {kernel}/{config}: interference retry recorded")


def _side_environment(
    root: Path, gpu_index: str, *, current_benchmark_root: Path | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_index
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("TIRX_INTERNAL_BENCH_REFERENCES", None)
    environment.pop(AB_CURRENT_BENCHMARK_ROOT_ENV, None)
    if current_benchmark_root is not None:
        environment[AB_CURRENT_BENCHMARK_ROOT_ENV] = str(current_benchmark_root)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"
    return environment


def _run_side(
    side: str,
    root: Path,
    workload: dict[str, Any],
    workload_path: Path,
    out_dir: Path,
    *,
    gpu_index: str,
    gpu_uuid: str,
    revision: str,
    rounds: int,
    cooldown: float,
    util_threshold: float,
    mem_threshold: float,
    current_benchmark_root: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "tirx_kernels.bench_suite",
        "--workloads",
        str(workload_path),
        "--out-dir",
        str(out_dir),
        "--label",
        f"ab-{side}-{revision[:8]}",
        "--rounds",
        str(rounds),
        "--cooldown",
        str(cooldown),
        "--util-threshold",
        str(util_threshold),
        "--mem-threshold",
        str(mem_threshold),
        "--max-prepare-processes",
        "1",
        "--ready-backlog",
        "1",
        "--no-probe",
        "--no-report",
    ]
    log_path = out_dir / "side.log"
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=root,
            env=_side_environment(root, gpu_index, current_benchmark_root=current_benchmark_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text().splitlines()[-20:])
        raise RuntimeError(
            f"{side} {_workload_key(workload)[0]}/{_workload_key(workload)[1]} "
            f"exited {completed.returncode}:\n{tail}"
        )
    payload = _run_payload(out_dir)
    _validate_side_payload(side, payload, workload, gpu_uuid=gpu_uuid)
    return payload


def _run_pair(
    index: int,
    workload: dict[str, Any],
    *,
    gpu_index: str,
    gpu_uuid: str,
    campaign_root: Path,
    roots: dict[str, Path],
    revisions: dict[str, str],
    rounds: int,
    cooldown: float,
    util_threshold: float,
    mem_threshold: float,
    rejected: list[dict[str, Any]],
    rejected_lock: threading.Lock,
) -> _PairResult:
    workload_root = campaign_root / "workloads" / f"{index:03d}"
    workload_path = workload_root / "workload.yaml"
    _write_workload(workload_path, workload)
    order = ("before", "after") if index % 2 else ("after", "before")
    attempt = 0
    while True:
        attempt += 1
        payloads: dict[str, dict[str, Any]] = {}
        try:
            for side in order:
                payloads[side] = _run_side(
                    side,
                    roots[side],
                    workload,
                    workload_path,
                    workload_root / f"attempt-{attempt}" / side,
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid,
                    revision=revisions[side],
                    rounds=rounds,
                    cooldown=cooldown,
                    util_threshold=util_threshold,
                    mem_threshold=mem_threshold,
                    current_benchmark_root=roots["after"] if side == "before" else None,
                )
        except _InterferenceError as error:
            with rejected_lock:
                rejected.append(
                    {
                        "index": index,
                        "kernel": workload["kernel"],
                        "config": workload["config"],
                        "gpu_index": gpu_index,
                        "gpu_uuid": gpu_uuid,
                        "attempt": attempt,
                        "reason": str(error),
                    }
                )
            continue
        return _PairResult(
            index=index,
            workload=workload,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            order=order,
            attempt=attempt,
            before=payloads["before"],
            after=payloads["after"],
        )


def _shared_provenance(payload: dict[str, Any]) -> tuple[Any, Any, Any]:
    git = dict(payload.get("git") or {})
    git.pop("tirx-kernels", None)
    tree = dict(payload.get("kernel_tree") or {})
    tree.pop("tirx-kernels:tirx_kernels", None)
    return git, tree, payload.get("baselines")


def _aggregate_side(
    side: str,
    pairs: list[_PairResult],
    *,
    selection: dict[str, Any],
    revision: str,
    tree: str,
    timestamp: str,
    gpu_rows: list[tuple[str, str]],
    probe_enabled: bool,
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    first = pairs[0].before if side == "before" else pairs[0].after
    shared = _shared_provenance(first)
    for pair in pairs[1:]:
        payload = pair.before if side == "before" else pair.after
        if _shared_provenance(payload) != shared:
            raise RuntimeError(f"{side}: shared provenance changed within the A/B campaign")

    aggregate = copy.deepcopy(first)
    aggregate["timestamp"] = timestamp
    aggregate["label"] = f"ab-{side}-{revision[:8]}"
    aggregate["selection"] = copy.deepcopy(selection)
    aggregate["results"] = [
        copy.deepcopy((pair.before if side == "before" else pair.after)["results"][0])
        for pair in pairs
    ]
    aggregate["results"].sort(key=lambda row: (row["kernel"], row.get("config") or row["label"]))
    aggregate.setdefault("git", {})["tirx-kernels"] = revision[:8]
    aggregate.setdefault("kernel_tree", {})["tirx-kernels:tirx_kernels"] = tree
    aggregate["probe"] = {
        "enabled": probe_enabled,
        "usable": [index for index, _uuid in gpu_rows],
        "failed": {},
    }
    aggregate["pipeline"] = {
        "execution_mode": "pipeline",
        "process_model": "one_shot_child_per_workload",
        "max_prepare_processes": 1,
        "ready_backlog": 1,
        "measurement_protocol": copy.deepcopy(first["pipeline"]["measurement_protocol"]),
        "interference_retry_count": 0,
        "interference_retries": [],
        "failure_count": 0,
    }
    aggregate["ab"] = {
        "side": side,
        "revision": revision,
        "pairing": "same physical GPU UUID per workload",
        "pair_order": "alternating by workload index",
        "rejected_pair_attempts": copy.deepcopy(rejected),
    }
    return aggregate


def _available_gpus(
    *, no_probe: bool, probe_timeout: float, util_threshold: float, mem_threshold: float
) -> list[tuple[str, str]]:
    from tirx_kernels.bench_suite.run import GpuPool, _visible_gpu_rows, detect_usable_gpus

    listing = GpuPool(util_threshold=util_threshold, mem_threshold=mem_threshold)
    visible_rows = _visible_gpu_rows(listing._all_gpus())
    occupied = listing._occupied_indices()
    candidates = [(index, uuid) for index, uuid in visible_rows if index not in occupied]
    if not candidates:
        raise RuntimeError("no idle visible GPUs are available for paired A/B")
    if no_probe:
        usable = {index for index, _uuid in candidates}
    else:
        usable, failures = detect_usable_gpus([index for index, _uuid in candidates], probe_timeout)
        for index, error in sorted(failures.items(), key=lambda item: int(item[0])):
            print(f"[bench-suite ab] gpu {index} probe failed: {error}", file=sys.stderr)
    result = [(index, uuid) for index, uuid in candidates if index in usable]
    if not result:
        raise RuntimeError("no idle visible GPU passed the A/B startup probe")
    return sorted(result, key=lambda row: int(row[0]))


def run_ab(
    workloads: list[dict[str, Any]],
    *,
    selection: dict[str, Any],
    before_revision: str,
    out_dir: Path,
    label: str | None,
    threshold: float,
    rounds: int,
    cooldown: float,
    no_probe: bool,
    probe_timeout: float,
    util_threshold: float,
    mem_threshold: float,
    no_report: bool,
) -> int:
    """Run a same-GPU paired campaign and return the bench-suite exit code."""

    if any(workload.get("num_gpus", 1) != 1 for workload in workloads):
        raise ValueError("--ab-before currently requires single-GPU workloads")

    after_root = Path(__file__).resolve().parents[2]
    after_state = _repository_state(after_root)
    if after_state[0]:
        raise RuntimeError("--ab-before requires the after checkout to be clean and committed")
    after_revision = _git(after_root, "rev-parse", "HEAD")
    before_revision = _git(after_root, "rev-parse", "--verify", f"{before_revision}^{{commit}}")
    before_tree = _git(after_root, "rev-parse", f"{before_revision}:tirx_kernels")
    after_tree = _git(after_root, "rev-parse", f"{after_revision}:tirx_kernels")

    from tirx_kernels.bench_suite.run import _tir_repo_root

    tir_root = _tir_repo_root()
    tir_state = _repository_state(tir_root) if tir_root is not None else None
    gpu_rows = _available_gpus(
        no_probe=no_probe,
        probe_timeout=probe_timeout,
        util_threshold=util_threshold,
        mem_threshold=mem_threshold,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    raw_campaign_name = label or f"{before_revision[:8]}-{after_revision[:8]}"
    campaign_name = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in raw_campaign_name
    ).strip("-.")
    if not campaign_name:
        campaign_name = f"{before_revision[:8]}-{after_revision[:8]}"
    campaign_root = out_dir.resolve() / "ab" / f"{stamp}-{campaign_name}"
    campaign_root.mkdir(parents=True, exist_ok=False)

    rejected: list[dict[str, Any]] = []
    rejected_lock = threading.Lock()
    results: list[_PairResult] = []
    failures: list[dict[str, Any]] = []
    result_lock = threading.Lock()
    work: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
    for index, workload in enumerate(workloads, 1):
        work.put((index, workload))

    print(
        f"[bench-suite ab] {len(workloads)} paired workload(s), "
        f"{len(gpu_rows)} GPU worker(s): {[index for index, _uuid in gpu_rows]}"
    )
    print(f"[bench-suite ab] before={before_revision[:8]} after={after_revision[:8]}")
    print(f"[bench-suite ab] artifacts: {campaign_root}")

    with tempfile.TemporaryDirectory(prefix="tirx-bench-ab-") as temporary:
        before_root = Path(temporary) / "before"
        before_root.mkdir()
        _extract_before_tree(after_root, before_revision, before_root)
        _copy_shared_harness(after_root, before_root)
        roots = {"before": before_root, "after": after_root}
        revisions = {"before": before_revision, "after": after_revision}

        def worker(gpu_index: str, gpu_uuid: str) -> None:
            while True:
                try:
                    index, workload = work.get_nowait()
                except queue.Empty:
                    return
                kernel, config = _workload_key(workload)
                print(
                    f"[bench-suite ab] {index:03d}/{len(workloads):03d} "
                    f"gpu={gpu_index} START {kernel}/{config}",
                    flush=True,
                )
                try:
                    pair = _run_pair(
                        index,
                        workload,
                        gpu_index=gpu_index,
                        gpu_uuid=gpu_uuid,
                        campaign_root=campaign_root,
                        roots=roots,
                        revisions=revisions,
                        rounds=rounds,
                        cooldown=cooldown,
                        util_threshold=util_threshold,
                        mem_threshold=mem_threshold,
                        rejected=rejected,
                        rejected_lock=rejected_lock,
                    )
                except Exception as error:
                    with result_lock:
                        failures.append(
                            {
                                "index": index,
                                "kernel": kernel,
                                "config": config,
                                "gpu_index": gpu_index,
                                "gpu_uuid": gpu_uuid,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    print(
                        f"[bench-suite ab] {index:03d}/{len(workloads):03d} "
                        f"gpu={gpu_index} FAIL {kernel}/{config}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    with result_lock:
                        results.append(pair)
                    print(
                        f"[bench-suite ab] {index:03d}/{len(workloads):03d} "
                        f"gpu={gpu_index} OK {kernel}/{config}",
                        flush=True,
                    )
                finally:
                    work.task_done()

        with ThreadPoolExecutor(max_workers=len(gpu_rows)) as executor:
            futures = [executor.submit(worker, index, uuid) for index, uuid in gpu_rows]
            for future in futures:
                future.result()

    campaign = {
        "timestamp": stamp,
        "label": campaign_name,
        "before_revision": before_revision,
        "after_revision": after_revision,
        "rounds": rounds,
        "cooldown_s": cooldown,
        "gpu_workers": [{"physical_index": index, "uuid": uuid} for index, uuid in gpu_rows],
        "selection": selection,
        "rejected_pair_attempts": rejected,
        "failures": failures,
    }
    (campaign_root / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    if _repository_state(after_root) != after_state:
        raise RuntimeError("after checkout changed during the A/B campaign")
    if tir_root is not None and _repository_state(tir_root) != tir_state:
        raise RuntimeError("TVM checkout changed during the A/B campaign")
    if failures:
        print(f"[bench-suite ab] workload failure summary: {len(failures)}", file=sys.stderr)
        return 1

    results.sort(key=lambda pair: pair.index)
    before_payload = _aggregate_side(
        "before",
        results,
        selection=selection,
        revision=before_revision,
        tree=before_tree,
        timestamp=stamp,
        gpu_rows=gpu_rows,
        probe_enabled=not no_probe,
        rejected=rejected,
    )
    after_payload = _aggregate_side(
        "after",
        results,
        selection=selection,
        revision=after_revision,
        tree=after_tree,
        timestamp=stamp,
        gpu_rows=gpu_rows,
        probe_enabled=not no_probe,
        rejected=rejected,
    )
    before_path = campaign_root / "before.json"
    after_path = campaign_root / "after.json"
    before_path.write_text(json.dumps(before_payload, indent=2) + "\n")
    after_path.write_text(json.dumps(after_payload, indent=2) + "\n")
    report, report_failures = build_report(
        before_payload, after_payload, threshold_pct=threshold, paired=True
    )
    if not no_report:
        report_path = campaign_root / "bench.md"
        report_path.write_text(report)
        print(f"[bench-suite ab] wrote {report_path}")
        print(report)
    return 3 if report_failures else 0
