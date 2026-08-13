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
import shlex
import sys
import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import tvm

DEFAULT_BENCH_ROUNDS = 5
DEFAULT_BENCH_COOLDOWN_S = 1.0
PREPARE_NUM_SMS_ENV = "TIRX_PREPARE_NUM_SMS"
PREPARE_CUDA_ARCH_ENV = "TIRX_PREPARE_CUDA_ARCH"
TVM_FFI_DISABLE_TORCH_C_DLPACK_ENV = "TVM_FFI_DISABLE_TORCH_C_DLPACK"
TVM_COMPILE_FORCE_FALLBACK_ENV = "TVM_COMPILE_FORCE_FALLBACK"


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


@dataclass(frozen=True)
class PreparedCacheEntry:
    """One process-local cached artifact that the GPU stage must consume."""

    namespace: str
    key: Any
    value: Any


@dataclass
class _CacheReplay:
    prepared: list[PreparedCacheEntry]
    next_index: int = 0


_COMPILE_REPLAY: ContextVar[_CompileReplay | None] = ContextVar("tirx_compile_replay", default=None)
_CACHE_REPLAY: ContextVar[_CacheReplay | None] = ContextVar("tirx_cache_replay", default=None)
_NVML_LOCK = threading.Lock()
_NVML_MODULE: Any | None = None
_NVML_UNAVAILABLE = False
_PREPARE_COMPILE_SCOPE = ContextVar("tirx_prepare_compile_scope", default=False)
_PREPARE_COMPILE_LOCK = threading.RLock()
_ORIGINAL_TVM_COMPILE = tvm.compile


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
    cached: tuple[PreparedCacheEntry, ...] = ()

    def run_gpu(self, **kwargs: Any) -> dict[str, Any]:
        run_bench_fn = getattr(self.module, "run_bench", None)
        if run_bench_fn is None:
            raise TypeError(f"module {self.module.__name__!r} has no run_bench()")
        with replay_compiled_kernels(self.compiled):
            with replay_prepared_cache(self.cached):
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
        return int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        )
    return default


def cuda_target(*, arch: str | None = None) -> tvm.target.Target:
    """Construct CUDA target metadata without querying a late-bound device."""
    configured = arch or os.environ.get(PREPARE_CUDA_ARCH_ENV)
    if configured is not None:
        if not configured.startswith("sm_"):
            raise ValueError(f"{PREPARE_CUDA_ARCH_ENV} must start with 'sm_', got {configured!r}")
        return tvm.target.Target({"kind": "cuda", "arch": configured})
    return tvm.target.Target("cuda")


def physical_cuda_uuids(required_num_gpus: int) -> tuple[str, ...]:
    """Resolve visible logical CUDA devices to canonical physical UUIDs."""
    if required_num_gpus < 1:
        raise ValueError("required_num_gpus must be positive")
    from cuda.bindings import driver as driver_api

    (result,) = driver_api.cuInit(0)
    if result != driver_api.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed while validating GPU assignment: {result}")
    result, count = driver_api.cuDeviceGetCount()
    if result != driver_api.CUresult.CUDA_SUCCESS or count != required_num_gpus:
        raise RuntimeError(
            "late GPU assignment exposed "
            f"{count if result == driver_api.CUresult.CUDA_SUCCESS else 'unknown'} device(s), "
            f"expected {required_num_gpus}"
        )
    uuids = []
    for logical_index in range(required_num_gpus):
        result, device = driver_api.cuDeviceGet(logical_index)
        if result != driver_api.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet({logical_index}) failed: {result}")
        result, uuid = driver_api.cuDeviceGetUuid(device)
        if result != driver_api.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGetUuid({logical_index}) failed: {result}")
        hex_uuid = uuid.bytes.hex()
        uuids.append(
            "GPU-"
            f"{hex_uuid[0:8]}-{hex_uuid[8:12]}-{hex_uuid[12:16]}-"
            f"{hex_uuid[16:20]}-{hex_uuid[20:32]}"
        )
    return tuple(uuids)


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _offline_nvcc_arch(arch: str) -> str | list[str]:
    """Preserve family-specific CUDA features through nvcc's virtual-arch stage."""
    if arch.startswith("sm_") and arch.endswith(("a", "f")):
        suffix = arch.removeprefix("sm_")
        return ["-gencode", f"arch=compute_{suffix},code=sm_{suffix}"]
    return arch


def _materialize_cuda_import(module: Any, target: tvm.target.Target) -> Any:
    """Compile one fallback CUDA source offline and rebuild a lazy runtime module."""
    if module.kind != "cuda" or module.is_runnable():
        return module
    if not module.is_binary_serializable():
        raise RuntimeError(
            f"CPU prepare produced non-runnable, non-serializable module kind={module.kind!r}"
        )
    source = module.inspect_source("cuda")
    if not source:
        raise RuntimeError("CPU prepare produced a CUDA fallback module without source")

    from tvm.support.nvcc import compile_cuda

    nvcc_options = shlex.split(os.environ.get("TVM_CUDA_NVRTC_EXTRA_OPTS", ""))
    with target:
        compiled = compile_cuda(
            source,
            target_format="fatbin",
            arch=_offline_nvcc_arch(target.arch),
            options=nvcc_options or None,
            compiler="nvcc",
        )
    serialized = tvm.ir.save_json(module)
    previous_callback = tvm.get_global_func("tvm_callback_cuda_compile")

    @tvm.register_global_func("tvm_callback_cuda_compile", override=True)
    def _reuse_precompiled_cuda(_source):
        return compiled

    try:
        materialized = tvm.ir.load_json(serialized)
    finally:
        tvm.register_global_func("tvm_callback_cuda_compile", previous_callback, override=True)
    if materialized.kind != "cuda" or not materialized.is_runnable():
        raise RuntimeError("offline CUDA materialization did not produce a runnable module")
    return materialized


def _materialize_cuda_import_tree(module: Any, target: tvm.target.Target) -> None:
    imports = list(module.imports)
    if not imports:
        return
    replacements = []
    changed = False
    for imported in imports:
        _materialize_cuda_import_tree(imported, target)
        replacement = _materialize_cuda_import(imported, target)
        replacements.append(replacement)
        changed |= replacement is not imported
    if changed:
        module.clear_imports()
        for replacement in replacements:
            module.import_module(replacement)


def _assert_runnable_cuda_imports(module: Any) -> None:
    for imported in module.imports:
        if imported.kind == "cuda" and not imported.is_runnable():
            raise RuntimeError("READY cannot retain a non-runnable CUDA fallback module")
        _assert_runnable_cuda_imports(imported)


def compile_executable(
    mod: Any, target: Any = None, *, relax_pipeline: Any = "default", tir_pipeline: Any = "default"
):
    """Compile normally, or use driver-safe offline CUDA materialization in prepare."""
    if not _PREPARE_COMPILE_SCOPE.get():
        return _ORIGINAL_TVM_COMPILE(
            mod, target, relax_pipeline=relax_pipeline, tir_pipeline=tir_pipeline
        )
    if target is None:
        target = tvm.target.Target.current(allow_none=True)
    target = tvm.target.Target(target)
    if target.kind.name != "cuda":
        return _ORIGINAL_TVM_COMPILE(
            mod, target, relax_pipeline=relax_pipeline, tir_pipeline=tir_pipeline
        )
    if not getattr(target, "arch", None):
        target = cuda_target()

    previous_fallback = os.environ.get(TVM_COMPILE_FORCE_FALLBACK_ENV)
    os.environ[TVM_COMPILE_FORCE_FALLBACK_ENV] = "1"
    try:
        executable = _ORIGINAL_TVM_COMPILE(
            mod, target, relax_pipeline=relax_pipeline, tir_pipeline=tir_pipeline
        )
        _materialize_cuda_import_tree(executable.mod, target)
        _assert_runnable_cuda_imports(executable.mod)
        return executable
    finally:
        _restore_environment(TVM_COMPILE_FORCE_FALLBACK_ENV, previous_fallback)


@contextmanager
def cpu_prepare_compile_scope():
    """Route every ``tvm.compile`` in one prepare through the driver-safe path."""
    with _PREPARE_COMPILE_LOCK:
        token = _PREPARE_COMPILE_SCOPE.set(True)
        previous_compile = tvm.compile
        tvm.compile = compile_executable
        try:
            yield
        finally:
            tvm.compile = previous_compile
            _PREPARE_COMPILE_SCOPE.reset(token)


@contextmanager
def gpu_stage_compile_guard():
    """Reject TIRx compilation after a prepared workload receives GPU ownership."""

    def reject_compile(*_args: Any, **_kwargs: Any):
        raise RuntimeError(
            "GPU stage attempted tvm.compile; all TIRx specialization and compilation "
            "must be completed before READY"
        )

    with _PREPARE_COMPILE_LOCK:
        previous_compile = tvm.compile
        tvm.compile = reject_compile
        try:
            yield
        finally:
            tvm.compile = previous_compile


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
            f"CPU prepare changed CUDA initialization state from {before!r} to {after!r}"
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
    target = cuda_target()
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


def consume_prepared_cache(namespace: str, key: Any, builder):
    """Build/cache normally, or consume the exact CPU-prepared artifact on replay."""
    replay = _CACHE_REPLAY.get()
    if replay is None:
        return builder()
    if replay.next_index >= len(replay.prepared):
        raise RuntimeError(
            f"GPU stage requested an unprepared cached artifact {namespace!r} key={key!r}"
        )
    expected = replay.prepared[replay.next_index]
    if namespace != expected.namespace or key != expected.key:
        raise RuntimeError(
            "GPU-stage cached artifact request does not match CPU prepare: "
            f"requested {namespace!r} key={key!r}, expected "
            f"{expected.namespace!r} key={expected.key!r}"
        )
    replay.next_index += 1
    return expected.value


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


@contextmanager
def replay_prepared_cache(cached: tuple[PreparedCacheEntry, ...]):
    """Require the GPU stage to consume every declared custom cached artifact."""
    replay = _CacheReplay(list(cached))
    token = _CACHE_REPLAY.set(replay)
    try:
        yield
        if replay.next_index != len(replay.prepared):
            raise RuntimeError(
                "GPU stage consumed "
                f"{replay.next_index} of {len(replay.prepared)} CPU-prepared cached artifacts"
            )
    finally:
        _CACHE_REPLAY.reset(token)


def prepare_run_bench(module: ModuleType, params: dict[str, Any]) -> PreparedRunBench:
    """Prepare the common one-PrimFunc ``get_kernel``/``run_bench`` contract."""
    get_kernel_fn = getattr(module, "get_kernel", None)
    if get_kernel_fn is None or getattr(module, "run_bench", None) is None:
        raise TypeError(f"module {module.__name__!r} needs an explicit prepare_bench() adapter")
    prim_func = get_kernel_fn(**dict(params))
    executable = compile_kernel(prim_func)
    return PreparedRunBench(module=module, params=dict(params), compiled=((prim_func, executable),))


def prepare_module_bench(module_name: str, params: dict[str, Any]) -> PreparedRunBench:
    """Explicit module adapter for the common one-PrimFunc benchmark shape."""
    return prepare_run_bench(sys.modules[module_name], params)


def prepared_cached_run_bench(
    module_name: str, params: dict[str, Any], *, cached: tuple[tuple[str, Any, Any], ...] = ()
) -> PreparedRunBench:
    """Wrap ``run_bench`` after its custom CPU compile cache was populated."""
    module = sys.modules[module_name]
    entries = tuple(PreparedCacheEntry(namespace, key, value) for namespace, key, value in cached)
    return PreparedRunBench(module=module, params=dict(params), compiled=(), cached=entries)


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
        kernel_name, config, module=mod, require_cuda_uninitialized=False
    )
    return run_prepared_kernel_bench(
        prepared, warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown=cooldown
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
        compile_scope = cpu_prepare_compile_scope() if require_cuda_uninitialized else nullcontext()
        with compile_scope:
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

    with gpu_stage_compile_guard():
        result = prepared.benchmark.run_gpu(**bench_kwargs)
    if not isinstance(result, dict):
        result = {}
    result.setdefault("kernel", prepared.kernel)
    result.setdefault("label", prepared.label)
    return result
