# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

from tirx_kernels.bench_suite import run as bench_run

_OUR_CHILD_PID = 4001
_FOREIGN_PID = 9001


def _pool_with_fake_smi(monkeypatch, apps_rows: list[str]) -> bench_run.GpuPool:
    def fake_smi(args: list[str]) -> list[str]:
        query = args[0]
        if query == "--query-gpu=index,utilization.gpu":
            return ["0, 0", "1, 0"]
        if query == "--query-gpu=index,memory.used,memory.total":
            return ["0, 7000, 180000", "1, 400, 180000"]
        if query == "--query-gpu=index,memory.total":
            return ["0, 180000", "1, 180000"]
        if query == "--query-gpu=index,uuid":
            return ["0, GPU-aaa", "1, GPU-bbb"]
        if query == "--query-compute-apps=pid,gpu_uuid,used_memory":
            return apps_rows
        raise AssertionError(f"unexpected nvidia-smi query: {args!r}")

    monkeypatch.setattr(bench_run.GpuPool, "_nvidia_smi", staticmethod(fake_smi))
    monkeypatch.setattr(bench_run, "_our_pids", lambda: {_OUR_CHILD_PID})
    return bench_run.GpuPool(allowed={"0", "1"})


def test_parked_own_child_does_not_occupy_its_affinity_card(monkeypatch):
    """An interference-parked bench child's residual context must not mark the
    card externally occupied, or the child starves waiting to reacquire it."""
    pool = _pool_with_fake_smi(monkeypatch, [f"{_OUR_CHILD_PID}, GPU-aaa, 7000"])
    assert pool._occupied_indices() == set()
    assert pool.try_acquire_exact(("0",)) is None  # occupancy not refreshed yet
    pool.refresh_external_occupancy()
    assert pool.try_acquire_exact(("0",)) == ("0",)


def test_foreign_resident_memory_occupies_card(monkeypatch):
    pool = _pool_with_fake_smi(monkeypatch, [f"{_FOREIGN_PID}, GPU-aaa, 53000"])
    assert pool._occupied_indices() == {"0"}
    pool.refresh_external_occupancy()
    assert pool.try_acquire_exact(("0",)) is None
    assert pool.try_acquire_exact(("1",)) == ("1",)


def test_small_foreign_residual_is_forgiven_by_idle_floor(monkeypatch):
    pool = _pool_with_fake_smi(monkeypatch, [f"{_FOREIGN_PID}, GPU-aaa, 300"])
    assert pool._occupied_indices() == set()


def test_mixed_own_and_foreign_memory_counts_only_foreign(monkeypatch):
    pool = _pool_with_fake_smi(
        monkeypatch,
        [
            f"{_OUR_CHILD_PID}, GPU-aaa, 4000",
            f"{_FOREIGN_PID}, GPU-aaa, 2000",
            f"{_OUR_CHILD_PID}, GPU-bbb, 6000",
        ],
    )
    assert pool._occupied_indices() == {"0"}
