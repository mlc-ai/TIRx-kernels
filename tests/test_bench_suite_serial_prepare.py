# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Exercise the scheduler's child lifecycle without CUDA or subprocess builds."""

import io
import signal
import sys
from types import SimpleNamespace

import pytest

from tirx_kernels.bench_suite import run as suite


class IdlePool:
    def __init__(self, **kwargs):
        self.owned = set()
        self.util_threshold = 0

    def total_visible(self):
        return 2

    def _all_gpus(self):
        return [("0", "GPU-0"), ("1", "GPU-1")]

    def try_acquire_many(self, count):
        available = tuple(gpu for gpu in ("0", "1") if gpu not in self.owned)
        return self.try_acquire_exact(available[:count]) if len(available) >= count else None

    def try_acquire_exact(self, indices):
        if self.owned.intersection(indices):
            return None
        self.owned.update(indices)
        return indices

    def release_many(self, indices):
        self.owned.difference_update(indices)

    def wake(self):
        pass

    def _utils(self):
        return {"0": 0, "1": 0}

    _mem_used_pct = _utils

    def _occupied_indices(self):
        return set()

    _busy_indices = _occupied_indices


@pytest.fixture
def protocol(monkeypatch, tmp_path):
    """Drive real scheduler transitions with deterministic child messages."""
    children, events = [], []

    class Control:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.messages = []
            self.closed = False

        def fileno(self):
            return self.descriptor

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, pid):
            self.pid, self.returncode, self.exit_polls = pid, None, None

        def poll(self):
            if self.exit_polls is not None:
                self.exit_polls -= 1
                if self.exit_polls <= 0:
                    self.wait()
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                events.append(("exit", self.pid))
                self.returncode = 0
            return self.returncode

    def spawn(workload, attempt, log_dir, **kwargs):
        live = [child for child in children if not child.control.closed]
        events.append(("spawn", workload["config"], tuple(child.state for child in live)))
        item = suite._PreparedAttempt(
            workload=workload,
            attempt=attempt,
            process=Process(10000 + len(children)),
            control=Control(1000 + len(children)),
            log_file=io.StringIO(),
            log_path=tmp_path / "child.log",
            workdir=str(tmp_path / f"child-{len(children)}"),
            started_at=0,
        )
        if workload.get("scenario") == "prepare_failure":
            item.control.messages.append({"type": "FAIL", "phase": "prepare", "error": "test"})
        else:
            item.control.messages.append(
                {"type": "READY", "required_num_gpus": workload.get("num_gpus", 1)}
            )
        children.append(item)
        return item

    def send(item, message):
        kind = message["type"]
        events.append((kind, item.workload["config"], tuple(child.state for child in children)))
        if kind == "ASSIGN":
            item.control.messages.append(
                {"type": "RUNNING_GPU", "physical_gpu_uuids": message["gpu_uuids"]}
            )
        elif kind == "ACCEPT_RESULT":
            # Keep RESULT alive across another scheduler iteration to test that
            # serial mode waits for process exit, not merely result acceptance.
            item.process.exit_polls = 2
        else:
            pytest.fail(f"unexpected child command {kind}")

    def receive(item):
        message = item.control.messages.pop(0)
        if message["type"] == "RUNNING_GPU":
            item.control.messages.append(
                {
                    "type": "RESULT_READY",
                    "result": {
                        "impls": {"tirx": 1.0},
                        "round_samples": {"tirx": [1.0]},
                        "errors": {},
                        "timer": "proton",
                        "benchmark_protocol": {
                            "rounds": 1,
                            "cooldown_s": 0.0,
                            "round_aggregate": "mean",
                            "order": ["tirx"],
                        },
                    },
                }
            )
        return [message], False

    def select(readers, *_):
        ready = [reader for reader in readers if getattr(reader, "messages", None)]
        # Process one message per iteration, exposing the ASSIGNED/RUNNING and
        # still-PREPARING overlap that buffering limits alone permit.
        return ready[:1], [], []

    interrupted = set()

    def strangers(item, *_):
        if item.workload.get("scenario") == "interference" and item.process.pid not in interrupted:
            interrupted.add(item.process.pid)
            return {9001: 100.0}
        return {}

    def interrupt(pid, sig):
        assert sig == signal.SIGUSR1
        item = next(child for child in children if child.process.pid == pid)
        item.control.messages[:] = [{"type": "INTERFERED"}]

    monkeypatch.setattr(suite, "_spawn_prepared_attempt", spawn)
    monkeypatch.setattr(suite, "_send_child", send)
    monkeypatch.setattr(suite, "_receive_child_messages", receive)
    monkeypatch.setattr(suite.select, "select", select)
    monkeypatch.setattr(suite, "_active_strangers", lambda *a: {})
    monkeypatch.setattr(suite, "_pid_sm_on_gpus", lambda *a: {})
    monkeypatch.setattr(suite, "_attempt_strangers", strangers)
    monkeypatch.setattr(suite.os, "kill", interrupt)
    monkeypatch.setattr(suite, "_terminate_subprocess", lambda process: process.wait())
    monkeypatch.setattr(suite, "MONITOR_INTERVAL", 0)
    monkeypatch.setattr(
        suite.threading,
        "Thread",
        lambda **kw: SimpleNamespace(start=lambda: None, join=lambda **k: None),
    )
    return SimpleNamespace(children=children, events=events)


@pytest.mark.parametrize("limits", [(1, 1), (4, 4)])
@pytest.mark.parametrize("scenario", ["normal", "prepare_failure", "interference"])
def test_serial_prepare_owns_one_child_through_exit(protocol, tmp_path, limits, scenario):
    pool = IdlePool()
    workloads = [
        {"kernel": "test", "config": "first", "num_gpus": 2, "scenario": scenario},
        {"kernel": "test", "config": "second", "num_gpus": 1},
    ]
    records, retries, pipeline = suite.run_scheduled_jobs(
        workloads,
        pool,
        tmp_path,
        rounds=1,
        cooldown=0.0,
        compile_profile={"cuda_arch": "sm_110a", "num_sms": 20},
        with_references=False,
        max_prepare_processes=limits[0],
        ready_backlog=limits[1],
        serial_prepare=True,
    )
    assert len(records) == 2
    assert [event[2] for event in protocol.events if event[0] == "spawn"] == [(), ()]
    assert all("PREPARING_CPU" not in event[2] for event in protocol.events if event[0] == "ASSIGN")
    assert all(child.control.closed for child in protocol.children)
    assert not pool.owned
    assert pipeline["serial_prepare"] is True
    assert pipeline["max_active_children"] == 1
    assert pipeline["measurement_protocol"]["rounds"] == 1
    assert records[0]["status"] == ("FAIL" if scenario == "prepare_failure" else "ok")
    assert records[1]["status"] == "ok"
    if scenario == "interference":
        assert len(retries) == 1
        assert records[0]["attempt"] == 2
        assert len(protocol.children) == 2  # Retry retains its prepared child.


def test_default_buffer_limits_still_allow_preparation_during_gpu(protocol, tmp_path):
    _, _, pipeline = suite.run_scheduled_jobs(
        [{"kernel": "test", "config": label} for label in ("first", "second")],
        IdlePool(),
        tmp_path,
        rounds=1,
        cooldown=0.0,
        compile_profile={"cuda_arch": "sm_100a", "num_sms": 148},
        with_references=False,
        max_prepare_processes=1,
        ready_backlog=1,
    )
    assert any("RUNNING_GPU" in event[2] for event in protocol.events if event[0] == "spawn")
    assert pipeline["serial_prepare"] is False


@pytest.mark.parametrize("serial", [False, True])
def test_cli_forwards_serial_choice_without_changing_timer(monkeypatch, tmp_path, serial):
    argv = [
        "bench_suite",
        "--workloads",
        "unused.yaml",
        "--out-dir",
        str(tmp_path),
        "--no-probe",
        "--with-references",
        "--timer",
        "proton",
        "--rounds",
        "15",
        "--cooldown",
        "1",
    ]
    if serial:
        argv.append("--serial-prepare")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(suite, "load_workloads", lambda path: [{"kernel": "test", "config": "x"}])
    monkeypatch.setattr(suite, "GpuPool", IdlePool)
    monkeypatch.setattr(
        suite, "gpu_compile_profile", lambda _: {"cuda_arch": "sm_110a", "num_sms": 20}
    )
    monkeypatch.setattr(suite, "validate_workload_archs", lambda *a: None)
    monkeypatch.setattr(suite, "collect_repo_git", lambda: {})

    class SchedulerReached(Exception):
        pass

    def scheduled(workloads, pool, log_dir, **kwargs):
        assert kwargs["serial_prepare"] is serial
        assert kwargs["rounds"] == 15
        assert kwargs["cooldown"] == 1.0
        assert kwargs["with_references"] is True
        assert workloads[0]["timer"] == "proton"
        raise SchedulerReached

    monkeypatch.setattr(suite, "run_scheduled_jobs", scheduled)
    with pytest.raises(SchedulerReached):
        suite.main()


def test_cli_rejects_serial_prepare_for_separate_ab_runner(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_suite", "--workloads", "unused.yaml", "--ab-before", "HEAD", "--serial-prepare"],
    )
    monkeypatch.setattr(suite, "load_workloads", lambda path: [{"kernel": "test", "config": "x"}])
    with pytest.raises(SystemExit) as exit_info:
        suite.main()
    assert exit_info.value.code == 2
    assert "--serial-prepare do not apply" in capsys.readouterr().err
