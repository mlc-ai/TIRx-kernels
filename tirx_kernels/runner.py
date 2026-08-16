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
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import tvm
from tirx_kernels.low_level_ir import check_low_level_ir

DEFAULT_BENCH_ROUNDS = 5
DEFAULT_BENCH_COOLDOWN_S = 0.0
PREPARE_NUM_SMS_ENV = "TIRX_PREPARE_NUM_SMS"
PREPARE_CUDA_ARCH_ENV = "TIRX_PREPARE_CUDA_ARCH"
TVM_FFI_DISABLE_TORCH_C_DLPACK_ENV = "TVM_FFI_DISABLE_TORCH_C_DLPACK"
TVM_COMPILE_FORCE_FALLBACK_ENV = "TVM_COMPILE_FORCE_FALLBACK"
_EXTERNAL_REFERENCES_ENV = "TIRX_INTERNAL_BENCH_REFERENCES"


def set_external_references_enabled(enabled: bool) -> None:
    """Select explicit diagnostic reference timing for the current process."""
    if not isinstance(enabled, bool):
        raise TypeError("external reference mode must be a bool")
    if enabled:
        os.environ[_EXTERNAL_REFERENCES_ENV] = "1"
    else:
        os.environ.pop(_EXTERNAL_REFERENCES_ENV, None)


def external_references_enabled() -> bool:
    """Whether this process may import, prepare, or launch external references."""
    return os.environ.get(_EXTERNAL_REFERENCES_ENV) == "1"


@runtime_checkable
class PreparedBenchmark(Protocol):
    """Opaque CPU-prepared benchmark retained by its preparing process."""

    def run_gpu(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExplicitPreparedBenchmark:
    """Process-local bridge from a kernel's explicit prepare to its GPU stage."""

    run_gpu_fn: Callable[..., dict[str, Any]]
    state: Any
    required_num_gpus: int = 1
    close_fn: Callable[[], None] | None = None
    owner_pid: int = field(default_factory=os.getpid, init=False, repr=False)

    def _assert_process_local(self) -> None:
        current_pid = os.getpid()
        if current_pid != self.owner_pid:
            raise RuntimeError(
                "prepared benchmark is process-local: "
                f"created in PID {self.owner_pid}, used in PID {current_pid}"
            )

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("prepared benchmarks cannot be serialized across processes")

    def run_gpu(self, **kwargs: Any) -> dict[str, Any]:
        self._assert_process_local()
        return self.run_gpu_fn(self.state, **kwargs)

    def close(self) -> None:
        self._assert_process_local()
        if self.close_fn is not None:
            self.close_fn()


def prepared_gpu_benchmark(
    run_gpu: Callable[..., dict[str, Any]],
    state: Any,
    *,
    required_num_gpus: int = 1,
    close: Callable[[], None] | None = None,
) -> ExplicitPreparedBenchmark:
    """Package kernel-owned prepared state without interpreting or serializing it."""
    if not callable(run_gpu):
        raise TypeError("run_gpu must be callable")
    if not isinstance(required_num_gpus, int) or isinstance(required_num_gpus, bool):
        raise TypeError("required_num_gpus must be an integer")
    if required_num_gpus < 1:
        raise ValueError("required_num_gpus must be positive")
    if close is not None and not callable(close):
        raise TypeError("close must be callable")
    return ExplicitPreparedBenchmark(run_gpu, state, required_num_gpus, close)


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
class _CudaAssignment:
    indices: tuple[int, ...]
    uuids: tuple[str, ...]
    previously_assigned: set[int]


_NVML_LOCK = threading.Lock()
_NVML_MODULE: Any | None = None
_NVML_UNAVAILABLE = False
_PREPARE_COMPILE_SCOPE = ContextVar("tirx_prepare_compile_scope", default=False)
_PREPARE_COMPILE_LOCK = threading.RLock()
_ORIGINAL_TVM_COMPILE = tvm.compile
_CUDA_ASSIGNMENT: _CudaAssignment | None = None


def _current_process_cuda_gpus(*, required: bool = False) -> tuple[int, ...]:
    """Return physical GPUs on which this PID owns a CUDA compute context."""
    global _NVML_MODULE, _NVML_UNAVAILABLE
    if _NVML_UNAVAILABLE:
        if required:
            raise RuntimeError("NVML is unavailable for CUDA assignment validation")
        return ()
    with _NVML_LOCK:
        if _NVML_MODULE is None:
            try:
                import pynvml

                pynvml.nvmlInit()
            except (ImportError, OSError):
                _NVML_UNAVAILABLE = True
                if required:
                    raise RuntimeError(
                        "NVML is unavailable for CUDA assignment validation"
                    ) from None
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
    except Exception as error:
        # Torch remains an independent oracle. NVML sampling failures must not
        # make ordinary standalone CPU tooling unusable.
        if required:
            raise RuntimeError("NVML failed while validating CUDA assignment") from error
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


def physical_cuda_uuids(device_indices: Sequence[int]) -> tuple[str, ...]:
    """Resolve exact all-visible CUDA ordinals to canonical physical UUIDs."""
    indices = tuple(int(index) for index in device_indices)
    if not indices:
        raise ValueError("device_indices must not be empty")
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError(f"invalid CUDA device indices: {device_indices!r}")
    from cuda.bindings import driver as driver_api

    (result,) = driver_api.cuInit(0)
    if result != driver_api.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed while validating GPU assignment: {result}")
    result, count = driver_api.cuDeviceGetCount()
    if result != driver_api.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGetCount failed while validating GPU assignment: {result}")
    invalid = [index for index in indices if index >= count]
    if invalid:
        raise RuntimeError(
            f"late GPU assignment requested device(s) {invalid}, but only {count} are visible"
        )
    uuids = []
    for physical_index in indices:
        result, device = driver_api.cuDeviceGet(physical_index)
        if result != driver_api.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet({physical_index}) failed: {result}")
        result, uuid = driver_api.cuDeviceGetUuid(device)
        if result != driver_api.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGetUuid({physical_index}) failed: {result}")
        hex_uuid = uuid.bytes.hex()
        uuids.append(
            "GPU-"
            f"{hex_uuid[0:8]}-{hex_uuid[8:12]}-{hex_uuid[12:16]}-"
            f"{hex_uuid[16:20]}-{hex_uuid[20:32]}"
        )
    return tuple(uuids)


def bind_cuda_assignment(
    device_indices: Sequence[int], expected_uuids: Sequence[str]
) -> tuple[str, ...]:
    """Select a late-bound physical device and install the process assignment."""
    global _CUDA_ASSIGNMENT
    indices = tuple(int(index) for index in device_indices)
    expected = tuple(str(uuid) for uuid in expected_uuids)
    if not indices or len(indices) != len(expected) or len(set(indices)) != len(indices):
        raise ValueError(
            f"invalid CUDA assignment indices={device_indices!r}, uuids={expected_uuids!r}"
        )
    import torch

    torch.cuda.set_device(indices[0])
    actual = physical_cuda_uuids(indices)
    if actual != expected:
        raise RuntimeError(
            f"late GPU assignment identity mismatch: requested {list(expected)}, "
            f"physical {list(actual)}"
        )
    previously_assigned = set(indices)
    if _CUDA_ASSIGNMENT is not None:
        previously_assigned.update(_CUDA_ASSIGNMENT.previously_assigned)
    _CUDA_ASSIGNMENT = _CudaAssignment(indices, actual, previously_assigned)
    validate_current_cuda_assignment("after ASSIGN")
    return actual


def current_cuda_assignment() -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return the process-local physical claim installed after ASSIGN."""
    if _CUDA_ASSIGNMENT is None:
        raise RuntimeError("no CUDA assignment is active in this process")
    return _CUDA_ASSIGNMENT.indices, _CUDA_ASSIGNMENT.uuids


def validate_current_cuda_assignment(stage: str, *, restore: bool = False) -> tuple[str, ...]:
    """Prove the current device and all observed contexts belong to assigned cards."""
    if _CUDA_ASSIGNMENT is None:
        raise RuntimeError(f"{stage}: no CUDA assignment is active")
    import torch

    primary = _CUDA_ASSIGNMENT.indices[0]
    if restore:
        torch.cuda.set_device(primary)
    current = int(torch.cuda.current_device())
    if current != primary:
        raise RuntimeError(
            f"{stage}: current CUDA device {current} differs from assigned device {primary}"
        )
    actual = physical_cuda_uuids(_CUDA_ASSIGNMENT.indices)
    if actual != _CUDA_ASSIGNMENT.uuids:
        raise RuntimeError(
            f"{stage}: assigned CUDA UUIDs changed from {_CUDA_ASSIGNMENT.uuids!r} to {actual!r}"
        )
    unexpected = set(_current_process_cuda_gpus(required=True)) - (
        _CUDA_ASSIGNMENT.previously_assigned
    )
    if unexpected:
        raise RuntimeError(
            f"{stage}: process owns CUDA context(s) on never-assigned physical GPU(s) "
            f"{sorted(unexpected)}"
        )
    return actual


def bench(*args: Any, references: Mapping[str, Any] | None = None, **kwargs: Any):
    """Run the canonical timer, admitting references only in explicit diagnostic mode."""
    from tvm.tirx.bench import bench as canonical_bench

    if _CUDA_ASSIGNMENT is not None:
        validate_current_cuda_assignment("before benchmark setup", restore=True)

    checked_references = None
    if references is not None and external_references_enabled():
        checked_references = {}
        for name, builder in references.items():

            def checked_builder(builder=builder, name=name):
                try:
                    return builder()
                finally:
                    if _CUDA_ASSIGNMENT is not None:
                        validate_current_cuda_assignment(
                            f"after external reference builder {name!r}", restore=True
                        )

            checked_references[name] = checked_builder

    result = canonical_bench(*args, references=checked_references, **kwargs)
    if _CUDA_ASSIGNMENT is not None:
        validate_current_cuda_assignment("after benchmark timing", restore=True)
    return result


def current_process_cuda_memory_bytes(device_indices: Sequence[int]) -> dict[int, int | None]:
    """Sample this PID's resident compute memory on exact physical GPUs."""
    indices = tuple(int(index) for index in device_indices)
    result: dict[int, int | None] = {index: None for index in indices}
    global _NVML_MODULE, _NVML_UNAVAILABLE
    if _NVML_UNAVAILABLE:
        return result
    with _NVML_LOCK:
        if _NVML_MODULE is None:
            try:
                import pynvml

                pynvml.nvmlInit()
            except (ImportError, OSError):
                _NVML_UNAVAILABLE = True
                return result
            _NVML_MODULE = pynvml
        pynvml = _NVML_MODULE
    try:
        process_query = next(
            getattr(pynvml, name)
            for name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            )
            if hasattr(pynvml, name)
        )
        pid = os.getpid()
        for index in indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = process_query(handle)
            matches = [process for process in processes if int(process.pid) == pid]
            result[index] = sum(int(process.usedGpuMemory) for process in matches)
    except Exception:
        return {index: None for index in indices}
    return result


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
    target = cuda_target()
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=target, tir_pipeline="tirx")


def run_kernel_test(kernel_name: str, config: dict[str, Any], *, registry=None):
    """Run a kernel's correctness test.

    Validates the public pre-lowering IR, then delegates to
    ``mod.run_test(**params)``.
    """
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        registry = discover_kernels()

    mod = registry[kernel_name]
    params = {k: v for k, v in config.items() if k != "label"}
    check_low_level_ir(mod.get_kernel(**params))
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
    try:
        return run_prepared_kernel_bench(
            prepared, warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown=cooldown
        )
    finally:
        close_prepared_kernel_bench(prepared)


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

    if _CUDA_ASSIGNMENT is not None:
        validate_current_cuda_assignment("before GPU stage")
    with gpu_stage_compile_guard():
        result = prepared.benchmark.run_gpu(**bench_kwargs)
    if _CUDA_ASSIGNMENT is not None:
        validate_current_cuda_assignment("after GPU stage")
    if not isinstance(result, dict):
        result = {}
    result.setdefault("kernel", prepared.kernel)
    result.setdefault("label", prepared.label)
    return result


def close_prepared_kernel_bench(prepared: PreparedKernelBenchmark) -> None:
    """Release process-local CPU-prepared resources after the workload exits."""
    prepared.assert_process_local()
    close = getattr(prepared.benchmark, "close", None)
    if close is not None:
        close()
