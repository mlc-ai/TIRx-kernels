# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Unified kernel test and benchmark runner.

Each kernel module must provide ``run_test(**config)`` which handles
compile → run → correctness-check internally.  Optionally, it can
provide ``run_bench(**config, warmup, repeat)`` for profiling.

The helpers ``compile_kernel`` and ``proton_bench`` are exposed for
kernel modules to use.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import tvm

DEFAULT_BENCH_ROUNDS = 5
DEFAULT_BENCH_COOLDOWN_S = 1.0
PREPARE_NUM_SMS_ENV = "TIRX_PREPARE_NUM_SMS"


@runtime_checkable
class PreparedBenchmark(Protocol):
    """Opaque CPU-prepared benchmark retained by its preparing process."""

    def run_gpu(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PreparedKernelBenchmark:
    """Runner-owned identity wrapped around a kernel-owned prepared object."""

    kernel: str
    label: str
    benchmark: PreparedBenchmark
    owner_pid: int = field(default_factory=os.getpid, init=False, repr=False)

    def assert_process_local(self) -> None:
        current_pid = os.getpid()
        if current_pid != self.owner_pid:
            raise RuntimeError(
                "prepared benchmark is process-local: "
                f"created in PID {self.owner_pid}, used in PID {current_pid}"
            )

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("prepared benchmarks cannot be serialized across processes")

    @property
    def required_num_gpus(self) -> int:
        self.assert_process_local()
        value = getattr(self.benchmark, "required_num_gpus", 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"prepared benchmark reported invalid required_num_gpus={value!r}")
        return value


@dataclass
class _CompileReplay:
    prepared: list[tuple[Any, Any]]
    next_index: int = 0


_COMPILE_REPLAY: ContextVar[_CompileReplay | None] = ContextVar(
    "tirx_compile_replay", default=None
)
_NVML_LOCK = threading.Lock()
_NVML_MODULE: Any | None = None
_NVML_UNAVAILABLE = False


@dataclass(frozen=True)
class PreparedRunBench:
    """Generic adapter that replays CPU-precompiled kernels into ``run_bench``.

    The original GPU-stage implementation remains the single authority for
    allocation, correctness, reference construction, implementation order and
    timing. Only its structurally identical ``compile_kernel`` call is replaced
    by the executable compiled during CPU prepare.
    """

    module: ModuleType
    params: dict[str, Any]
    compiled: tuple[tuple[Any, Any], ...]

    def run_gpu(self, **kwargs: Any) -> dict[str, Any]:
        run_bench_fn = getattr(self.module, "run_bench", None)
        if run_bench_fn is None:
            raise TypeError(f"module {self.module.__name__!r} has no run_bench()")
        with replay_compiled_kernels(self.compiled):
            return run_bench_fn(**self.params, **kwargs)


def _current_process_cuda_gpus() -> tuple[int, ...]:
    """Return physical GPUs on which this PID owns a CUDA compute context."""
    global _NVML_MODULE, _NVML_UNAVAILABLE
    if _NVML_UNAVAILABLE:
        return ()
    with _NVML_LOCK:
        if _NVML_MODULE is None:
            try:
                import pynvml

                pynvml.nvmlInit()
            except (ImportError, OSError):
                _NVML_UNAVAILABLE = True
                return ()
            _NVML_MODULE = pynvml
        pynvml = _NVML_MODULE

    pid = os.getpid()
    owned = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        process_query = next(
            getattr(pynvml, name)
            for name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            )
            if hasattr(pynvml, name)
        )
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            if any(int(process.pid) == pid for process in process_query(handle)):
                owned.append(index)
    except Exception:
        # Torch remains an independent oracle. NVML sampling failures must not
        # make ordinary standalone CPU tooling unusable.
        return ()
    return tuple(owned)


def cuda_is_initialized() -> bool:
    """Report framework or driver-level CUDA initialization without creating it."""
    torch = sys.modules.get("torch")
    torch_initialized = bool(torch is not None and torch.cuda.is_initialized())
    return torch_initialized or bool(_current_process_cuda_gpus())


def hardware_num_sms(default: int = 148) -> int:
    """Return the compile-profile SM count without touching CUDA during prepare."""
    configured = os.environ.get(PREPARE_NUM_SMS_ENV)
    if configured is not None:
        value = int(configured)
        if value < 1:
            raise ValueError(f"{PREPARE_NUM_SMS_ENV} must be positive, got {value}")
        return value

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    return default


@contextmanager
def cuda_initialization_guard(*, require_uninitialized: bool = False):
    """Prove a CPU prepare region neither starts nor changes CUDA state."""
    before = cuda_is_initialized()
    if require_uninitialized and before:
        raise RuntimeError("CPU prepare requires a process with CUDA still uninitialized")
    yield
    after = cuda_is_initialized()
    if after != before:
        raise RuntimeError(
            "CPU prepare changed CUDA initialization state "
            f"from {before!r} to {after!r}"
        )


def compile_kernel(func):
    """Compile a single TIR PrimFunc via the tirx pipeline."""
    replay = _COMPILE_REPLAY.get()
    if replay is not None:
        if replay.next_index >= len(replay.prepared):
            raise RuntimeError("GPU stage requested more kernel compilations than CPU prepare")
        expected_func, executable = replay.prepared[replay.next_index]
        if not tvm.ir.structural_equal(func, expected_func, map_free_vars=True):
            raise RuntimeError(
                "GPU-stage compile request does not structurally match the next "
                "CPU-prepared PrimFunc"
            )
        replay.next_index += 1
        return executable
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=target, tir_pipeline="tirx")


def compile_kernel_lazy(builder):
    """Build/compile normally, but skip both operations during prepared replay."""
    replay = _COMPILE_REPLAY.get()
    if replay is not None:
        if replay.next_index >= len(replay.prepared):
            raise RuntimeError("GPU stage requested more kernel compilations than CPU prepare")
        _expected_func, executable = replay.prepared[replay.next_index]
        replay.next_index += 1
        return executable
    return compile_kernel(builder())


@contextmanager
def replay_compiled_kernels(compiled: tuple[tuple[Any, Any], ...]):
    """Make prepared executables available to matching GPU-stage compile calls."""
    replay = _CompileReplay(list(compiled))
    token = _COMPILE_REPLAY.set(replay)
    try:
        yield
        if replay.next_index != len(replay.prepared):
            raise RuntimeError(
                "GPU stage consumed "
                f"{replay.next_index} of {len(replay.prepared)} CPU-prepared kernels"
            )
    finally:
        _COMPILE_REPLAY.reset(token)


def prepare_run_bench(module: ModuleType, params: dict[str, Any]) -> PreparedRunBench:
    """Prepare the common one-PrimFunc ``get_kernel``/``run_bench`` contract."""
    get_kernel_fn = getattr(module, "get_kernel", None)
    if get_kernel_fn is None or getattr(module, "run_bench", None) is None:
        raise TypeError(
            f"module {module.__name__!r} needs an explicit prepare_bench() adapter"
        )
    prim_func = get_kernel_fn(**dict(params))
    executable = compile_kernel(prim_func)
    return PreparedRunBench(
        module=module,
        params=dict(params),
        compiled=((prim_func, executable),),
    )


def prepare_module_bench(module_name: str, params: dict[str, Any]) -> PreparedRunBench:
    """Explicit module adapter for the common one-PrimFunc benchmark shape."""
    return prepare_run_bench(sys.modules[module_name], params)


def prepared_cached_run_bench(module_name: str, params: dict[str, Any]) -> PreparedRunBench:
    """Wrap ``run_bench`` after its custom CPU compile cache was populated."""
    module = sys.modules[module_name]
    return PreparedRunBench(module=module, params=dict(params), compiled=())


def run_kernel_test(kernel_name: str, config: dict[str, Any], *, registry=None):
    """Run a kernel's correctness test.

    Delegates to ``mod.run_test(**params)``.
    """
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        registry = discover_kernels()

    mod = registry[kernel_name]
    params = {k: v for k, v in config.items() if k != "label"}
    mod.run_test(**params)


def run_kernel_bench(
    kernel_name: str,
    config: dict[str, Any],
    *,
    registry=None,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int | None = None,
    cooldown: float | None = None,
):
    """Run a kernel's benchmark.

    Delegates to ``mod.run_bench(**params, ...)`` if available, otherwise runs
    ``run_test`` without timing. warmup/repeat are only forwarded when explicitly
    provided (CLI ``--warmup/--repeat`` or a per-workload override); otherwise each
    timer uses its own Triton-aligned default inside ``tvm.tirx.bench.bench``.
    """
    mod = None if registry is None else registry[kernel_name]
    if mod is None:
        from tirx_kernels.registry import load_kernel

        mod = load_kernel(kernel_name)
    params = {k: v for k, v in config.items() if k != "label"}
    label = config.get("label", "default")

    prepared = prepare_kernel_bench(
        kernel_name,
        config,
        module=mod,
        require_cuda_uninitialized=False,
    )
    return run_prepared_kernel_bench(
        prepared,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown=cooldown,
    )


def prepare_kernel_bench(
    kernel_name: str,
    config: dict[str, Any],
    *,
    module: ModuleType | None = None,
    require_cuda_uninitialized: bool = True,
) -> PreparedKernelBenchmark:
    """Load and CPU-prepare one benchmark without initializing CUDA."""
    label = config.get("label", "default")
    params = {key: value for key, value in config.items() if key != "label"}
    with cuda_initialization_guard(require_uninitialized=require_cuda_uninitialized):
        if module is None:
            from tirx_kernels.registry import load_kernel

            module = load_kernel(kernel_name, strict=True)
        prepare_bench_fn = getattr(module, "prepare_bench", None)
        if prepare_bench_fn is None:
            raise TypeError(
                f"kernel {kernel_name!r} module {module.__name__!r} has no prepare_bench(); "
                "benchable kernels cannot use a one-stage fallback"
            )
        benchmark = prepare_bench_fn(**params)

    if not isinstance(benchmark, PreparedBenchmark):
        raise TypeError(
            f"kernel {kernel_name!r} prepare_bench() must return an object with run_gpu()"
        )
    return PreparedKernelBenchmark(kernel=kernel_name, label=label, benchmark=benchmark)


def run_prepared_kernel_bench(
    prepared: PreparedKernelBenchmark,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int | None = None,
    cooldown: float | None = None,
) -> dict[str, Any]:
    """Run the GPU stage and normalize the existing benchmark result schema."""
    prepared.assert_process_local()
    bench_kwargs: dict[str, Any] = {}
    if warmup is not None:
        bench_kwargs["warmup"] = warmup
    if repeat is not None:
        bench_kwargs["repeat"] = repeat
    if timer is not None:
        bench_kwargs["timer"] = timer
    if rounds is not None:
        bench_kwargs["rounds"] = rounds
    if cooldown is not None:
        bench_kwargs["cooldown_s"] = cooldown

    result = prepared.benchmark.run_gpu(**bench_kwargs)
    if not isinstance(result, dict):
        result = {}
    result.setdefault("kernel", prepared.kernel)
    result.setdefault("label", prepared.label)
    return result
