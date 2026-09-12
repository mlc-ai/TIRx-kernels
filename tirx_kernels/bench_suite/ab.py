# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Paired before/after campaigns for the bench suite on the kcoral server.

Both revisions ship as tarballs; every workload is two requests (before and
after) submitted back to back in alternating order.  The server's per-GPU lease
gives both sides the same physical GPU and a fresh worker process each.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tirx_kernels.bench_suite import remote
from tirx_kernels.bench_suite.impls import our_impls
from tirx_kernels.bench_suite.provenance import collect_repo_git
from tirx_kernels.bench_suite.ratio_diff import build_report

_SHARED_HARNESS_PATHS = (
    Path("tirx_kernels/bench"),
    Path("tirx_kernels/bench_suite"),
    Path("tirx_kernels/runner.py"),
    Path("tirx_kernels/basic/utils/_runtime.py"),
)
DEFAULT_AB_MAX_IN_FLIGHT = 1


@dataclass(frozen=True)
class _PairResult:
    index: int
    workload: dict[str, Any]
    order: tuple[str, str]
    gpu_uuid: str
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
    """Materialize ``<revision>:tirx_kernels`` under ``destination/tirx_kernels``."""
    archive = destination.parent / "before.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            f"--output={archive}",
            revision,
            "tirx_kernels",
        ],
        check=True,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def _workload_key(workload: dict[str, Any]) -> tuple[str, str]:
    return workload["kernel"], workload["config"]


def _validate_side_row(side: str, row: dict[str, Any], workload: dict[str, Any]) -> None:
    kernel, config = _workload_key(workload)
    if (row.get("kernel"), row.get("config") or row.get("label")) != (kernel, config):
        raise RuntimeError(f"{side} {kernel}/{config}: result identity differs")
    if row.get("status") != "ok":
        raise RuntimeError(f"{side} {kernel}/{config}: {row.get('error') or row.get('status')!r}")
    samples = row.get("round_samples")
    if not isinstance(samples, dict) or len(our_impls(samples)) != 1:
        raise RuntimeError(f"{side} {kernel}/{config}: expected one TIR/TIRx implementation")
    uuids = row.get("physical_gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != 1 or not uuids[0]:
        raise RuntimeError(f"{side} {kernel}/{config}: expected one physical GPU, got {uuids!r}")


def _submit_side(
    side: str,
    index: int,
    workload: dict[str, Any],
    *,
    api: remote.KcoralApi,
    clients: remote.ClientPool,
    trees: dict[str, remote.TreeArchive],
    shim: str,
    profile: remote.ServerProfile,
    rounds: int,
    cooldown: float,
    request_timeout_s: float,
    prepare_mode: str,
    campaign_root: Path,
    policy: remote.RetryPolicy = remote.RetryPolicy(),
) -> dict[str, Any]:
    """One request for one side of a pair; returns a validated result row."""
    spec = remote.workload_spec(
        workload,
        cuda_arch=profile.arch,
        num_sms=profile.num_sms,
        side=side,
        references_enabled=False,
        prepare_mode=prepare_mode,
        rounds=rounds,
        cooldown=cooldown,
    )
    submission, spec = remote.submit_workload(
        api,
        clients.get(),
        tree=trees["after"],
        before_tree=trees["before"] if side == "before" else None,
        shim=shim,
        spec=spec,
        timeout_s=request_timeout_s,
        policy=policy,
        on_fallback=lambda reason: print(
            f"[bench-suite ab] {index:03d} {side} {workload['kernel']}/{workload['config']}: "
            f"retrying with --prepare gpu: {reason}",
            flush=True,
        ),
    )
    log_path = campaign_root / "workloads" / f"{index:03d}" / f"{side}.log"
    remote.write_request_log(log_path, workload=workload, submission=submission, side=side)
    row = remote.outcome_to_record(
        workload,
        submission,
        profile=profile,
        tree_sha256=trees[side].sha256,
        before_tree_sha256=trees["before"].sha256 if side == "before" else None,
        prepare_mode=spec["prepare_mode"],
        rounds=rounds,
        cooldown=cooldown,
        references_enabled=False,
        request_timeout_s=request_timeout_s,
        log_path=log_path,
        side=side,
    )
    _validate_side_row(side, row, workload)
    return row


def _run_pair(index: int, workload: dict[str, Any], **side_kwargs: Any) -> _PairResult:
    """Run both sides back to back; odd workloads start with before, even with after."""
    order = ("before", "after") if index % 2 else ("after", "before")
    rows: dict[str, dict[str, Any]] = {}
    for side in order:
        rows[side] = _submit_side(side, index, workload, **side_kwargs)
    before_uuids = rows["before"].get("physical_gpu_uuids")
    after_uuids = rows["after"].get("physical_gpu_uuids")
    if before_uuids != after_uuids:
        kernel, config = _workload_key(workload)
        raise RuntimeError(
            f"{kernel}/{config}: before ran on {before_uuids!r} but after on {after_uuids!r}"
        )
    return _PairResult(
        index=index,
        workload=workload,
        order=order,
        gpu_uuid=after_uuids[0],
        before=rows["before"],
        after=rows["after"],
    )


def _aggregate_side(
    side: str,
    pairs: list[_PairResult],
    *,
    selection: dict[str, Any],
    revision: str,
    tree: str,
    timestamp: str,
    git: dict[str, Any],
    kernel_tree: dict[str, Any],
    probe: dict[str, Any],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    rows = [copy.deepcopy(pair.before if side == "before" else pair.after) for pair in pairs]
    rows.sort(key=lambda row: (row["kernel"], row.get("config") or row["label"]))
    return {
        "timestamp": timestamp,
        "label": f"ab-{side}-{revision[:8]}",
        "references_enabled": False,
        "git": {**git, "tirx-kernels": revision[:8]},
        "kernel_tree": {**kernel_tree, "tirx-kernels:tirx_kernels": tree},
        "baselines": {},
        "selection": copy.deepcopy(selection),
        "probe": copy.deepcopy(probe),
        "pipeline": copy.deepcopy(pipeline),
        "ab": {
            "side": side,
            "revision": revision,
            "pairing": "same kcoral GPU per pair",
            "pair_order": "alternating by workload index",
        },
        "results": rows,
    }


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
    no_report: bool,
    server_url: str,
    max_in_flight: int | None,
    request_timeout_s: float,
    prepare_mode: str,
) -> int:
    """Run a paired campaign on the server and return the bench-suite exit code."""

    if any(workload.get("num_gpus", 1) != 1 for workload in workloads):
        raise ValueError("--ab-before requires single-GPU workloads")

    after_root = Path(__file__).resolve().parents[2]
    after_state = _repository_state(after_root)
    if after_state[0]:
        raise RuntimeError("--ab-before requires the after checkout to be clean and committed")
    after_revision = _git(after_root, "rev-parse", "HEAD")
    before_revision = _git(after_root, "rev-parse", "--verify", f"{before_revision}^{{commit}}")
    before_tree = _git(after_root, "rev-parse", f"{before_revision}:tirx_kernels")
    after_tree = _git(after_root, "rev-parse", f"{after_revision}:tirx_kernels")

    api = remote.load_kcoral()
    client, health = remote.connect(api, server_url)
    cuda_arch = health["target"]["arch"]
    from tirx_kernels.bench_suite.run import partition_workloads_by_arch, validate_workload_archs

    selection = copy.deepcopy(selection)
    if selection["mode"] == "default":
        workloads, incompatible_workloads = partition_workloads_by_arch(workloads, cuda_arch)
        if not workloads:
            raise ValueError(f"default roster has no workloads for {cuda_arch}")
        if incompatible_workloads:
            incompatible_kernels = {workload["kernel"] for workload in incompatible_workloads}
            print(
                f"[bench-suite ab] selected {len(workloads)} {cuda_arch} default workload(s); "
                f"excluded {len(incompatible_workloads)} workload(s) across "
                f"{len(incompatible_kernels)} incompatible kernel(s)"
            )
    else:
        validate_workload_archs(workloads, cuda_arch)
    selection["cuda_arch"] = cuda_arch
    selection["keys"] = [[workload["kernel"], workload["config"]] for workload in workloads]

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

    in_flight = max_in_flight or DEFAULT_AB_MAX_IN_FLIGHT
    results: list[_PairResult] = []
    failures: list[dict[str, Any]] = []
    result_lock = threading.Lock()
    all_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="tirx-bench-ab-") as temporary:
        before_root = Path(temporary) / "before"
        before_root.mkdir()
        _extract_before_tree(after_root, before_revision, before_root)
        _copy_shared_harness(after_root, before_root)
        trees = {
            "after": remote.build_tree_archive(after_root / "tirx_kernels"),
            "before": remote.build_tree_archive(before_root / "tirx_kernels"),
        }
    shim = remote.shim_source()
    try:
        profile = remote.probe_server(
            api,
            client,
            url=server_url,
            health=health,
            tree=trees["after"],
            shim=shim,
            references_enabled=False,
            timeout_s=min(request_timeout_s, remote.PROBE_TIMEOUT_S),
            extra_blobs=(trees["before"],),
        )
    finally:
        client.close()

    print(
        f"[bench-suite ab] {len(workloads)} paired workload(s) on {server_url} "
        f"(arch={profile.arch}, gpu={profile.device_uuid}, max-in-flight={in_flight}, "
        f"prepare={prepare_mode})"
    )
    print(
        f"[bench-suite ab] before={before_revision[:8]} (tree {trees['before'].sha256[:12]}) "
        f"after={after_revision[:8]} (tree {trees['after'].sha256[:12]})"
    )
    print(f"[bench-suite ab] artifacts: {campaign_root}")

    clients = remote.ClientPool(api, server_url)
    side_kwargs = {
        "api": api,
        "clients": clients,
        "trees": trees,
        "shim": shim,
        "profile": profile,
        "rounds": rounds,
        "cooldown": cooldown,
        "request_timeout_s": request_timeout_s,
        "prepare_mode": prepare_mode,
        "campaign_root": campaign_root,
    }

    def worker(index: int, workload: dict[str, Any]) -> None:
        kernel, config = _workload_key(workload)
        print(
            f"[bench-suite ab] {index:03d}/{len(workloads):03d} START {kernel}/{config}", flush=True
        )
        try:
            pair = _run_pair(index, workload, **side_kwargs)
        except Exception as error:
            with result_lock:
                failures.append(
                    {
                        "index": index,
                        "kernel": kernel,
                        "config": config,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            print(
                f"[bench-suite ab] {index:03d}/{len(workloads):03d} FAIL {kernel}/{config}: {error}",
                flush=True,
            )
        else:
            with result_lock:
                results.append(pair)
                all_rows.extend([pair.before, pair.after])
            ratio = ""
            try:
                before_us = next(iter(pair.before["impls"].values()))
                after_us = next(iter(pair.after["impls"].values()))
                ratio = f" after/before={after_us / before_us:.4f}"
            except Exception:
                pass
            print(
                f"[bench-suite ab] {index:03d}/{len(workloads):03d} OK {kernel}/{config}{ratio}",
                flush=True,
            )

    try:
        with ThreadPoolExecutor(max_workers=in_flight, thread_name_prefix="bench-ab") as executor:
            futures = [
                executor.submit(worker, index, workload)
                for index, workload in enumerate(workloads, 1)
            ]
            for future in futures:
                future.result()
    finally:
        clients.close()

    pipeline = remote.pipeline_metadata(
        profile=profile,
        tree=trees["after"],
        rounds=rounds,
        cooldown=cooldown,
        prepare_mode=prepare_mode,
        max_in_flight=in_flight,
        request_timeout_s=request_timeout_s,
        records=all_rows,
    )
    pipeline["before_tree_sha256"] = trees["before"].sha256
    campaign = {
        "timestamp": stamp,
        "label": campaign_name,
        "before_revision": before_revision,
        "after_revision": after_revision,
        "rounds": rounds,
        "cooldown_s": cooldown,
        "server": profile.summary(),
        "trees": {side: archive.summary() for side, archive in trees.items()},
        "selection": selection,
        "pair_orders": [
            [pair.index, list(pair.order)] for pair in sorted(results, key=lambda p: p.index)
        ],
        "failures": failures,
    }
    (campaign_root / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    if _repository_state(after_root) != after_state:
        raise RuntimeError("after checkout changed during the A/B campaign")
    if failures:
        print(f"[bench-suite ab] workload failure summary: {len(failures)}")
        for failure in failures:
            print(
                f"[bench-suite ab]   - {failure['kernel']}/{failure['config']}: {failure['error']}"
            )
        return 1

    results.sort(key=lambda pair: pair.index)
    git, kernel_tree = remote.merge_provenance(
        profile,
        local_git=collect_repo_git(),
        local_tree={"tir:python/tvm/tirx": None, "tirx-kernels:tirx_kernels": after_tree},
    )
    probe = remote.probe_summary(profile)
    side_common = {
        "selection": selection,
        "timestamp": stamp,
        "git": git,
        "kernel_tree": kernel_tree,
        "probe": probe,
        "pipeline": pipeline,
    }
    before_payload = _aggregate_side(
        "before", results, revision=before_revision, tree=before_tree, **side_common
    )
    after_payload = _aggregate_side(
        "after", results, revision=after_revision, tree=after_tree, **side_common
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
