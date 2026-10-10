# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Kernel entry frame exposed as ``txl.kernel`` and its trace-time session.

The decorator owns an ``IRBuilder`` and traces the decorated function once,
at decoration time. Plain Python control flow in the body is macro expansion;
nothing goes through the TVMScript parser.
"""

from __future__ import annotations

import inspect
import linecache
import os
import sys
import sysconfig
import threading
from functools import cache
from pathlib import Path
from types import MappingProxyType

import tvm
from tvm.ir import SourceName, Span
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import ir_builder as I

# Registers the hardware hands the entry when ptxas pins the allocation with
# __launch_bounds__(nthreads, min_blocks_per_sm). setmaxnreg's direction is
# read off this: a role asking for more must .inc, for less .dec.
REGS_PER_CTA = 65536
_REGS_PER_THREAD_MAX = 255
_REGS_PER_THREAD_GRANULARITY = 8


def _aligned_register_share(warps: int, min_blocks_per_sm: int) -> int:
    """The 8-aligned per-thread share of the resident warpgroup pool."""
    per_thread = REGS_PER_CTA // (min_blocks_per_sm * warps * 32)
    return per_thread & ~(_REGS_PER_THREAD_GRANULARITY - 1)


def entry_regs(warps: int, min_blocks_per_sm: int = 1) -> int:
    """The per-thread register allocation ptxas pins for a *warps*-wide entry.

    The register file split over the threads that must fit on the SM at once,
    rounded down to the 8-register allocation granularity, then capped at the
    architectural per-thread usage maximum.

    ``min_blocks_per_sm`` is part of that division, not a detail: the second
    ``__launch_bounds__`` operand promises *m* CTAs resident, so each one gets
    ``REGS_PER_CTA // m`` and each thread ``REGS_PER_CTA // (m * warps * 32)``.
    Ignoring it overstates the entry allocation by a factor of *m*, and since
    ``setmaxnreg``'s direction is read off this number, the error surfaces as a
    wrong direction: an 8-warp entry at ``min_blocks_per_sm=8`` really gets 32
    registers, so a role asking for 64 must ``.inc``. Modelled as 248 it emits
    ``.dec`` and ptxas rejects the kernel outright — ``(C7406) setmaxnreg.dec
    has register count (64) which is larger than the largest temporal register
    count in the program (32)``.

    The 255 cap applies only after the pool share is aligned. For example, four
    warps at ``min_blocks_per_sm=2`` have a 256-register ownership share but
    ptxas reports at most 255 registers used per thread, so the entry is 255
    and a role asking for the legal ``setmaxnreg`` target 256 must ``.inc``.
    """
    return min(_aligned_register_share(warps, min_blocks_per_sm), _REGS_PER_THREAD_MAX)


def cta_register_pool(warps: int, min_blocks_per_sm: int = 1) -> int:
    """Registers one CTA can redistribute after resident-warpgroup rounding."""
    return _aligned_register_share(warps, min_blocks_per_sm) * warps * 32


class gptr:  # pylint: disable=invalid-name
    """A global buffer parameter with symbolic rank or a fixed shape.

    ``txl.gptr[dtype]`` and ``txl.gptr[dtype, ndim]`` give every dimension a fresh
    symbolic ``int64`` extent. ``txl.gptr[dtype, shape]`` owns the exact fixed
    extents when they are part of the specialized kernel contract.
    ``txl.gptr(dtype, shape=lambda p: (...))`` derives dynamic extents from the
    entry's scalar parameters, preserving one shape fact instead of inventing
    unrelated symbols.
    """

    def __init__(self, dtype: str, ndim: int | tuple[int, ...] = 1, *, shape=None):
        if shape is not None and ndim != 1:
            raise TypeError("gptr shape= cannot be combined with an ndim or fixed-shape argument")
        if shape is not None and not callable(shape):
            raise TypeError(f"gptr shape= must be callable, got {shape!r}")
        self.shape_factory = shape
        self.shape = None
        if shape is not None:
            ndim = None
        elif isinstance(ndim, tuple):
            if not ndim or any(
                not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
                for extent in ndim
            ):
                raise TypeError(f"gptr shape must contain positive integers, got {ndim!r}")
            self.shape = ndim
            ndim = len(ndim)
        elif not isinstance(ndim, int) or isinstance(ndim, bool) or ndim < 1:
            raise TypeError(f"gptr ndim must be a positive integer, got {ndim!r}")
        self.dtype = dtype
        self.ndim = ndim

    def __class_getitem__(cls, dtype):
        if isinstance(dtype, tuple):
            if len(dtype) != 2:
                raise TypeError(
                    "gptr expects txl.gptr[dtype], txl.gptr[dtype, ndim], or txl.gptr[dtype, shape]"
                )
            return cls(*dtype)
        return cls(dtype)

    def __repr__(self):
        if self.shape_factory is not None:
            return f"txl.gptr({self.dtype!r}, shape=<entry-shape>)"
        if self.shape is not None:
            return f"txl.gptr[{self.dtype!r}, {self.shape!r}]"
        if self.ndim != 1:
            return f"txl.gptr[{self.dtype!r}, {self.ndim}]"
        return f"txl.gptr[{self.dtype!r}]"


class TensorMap:  # pylint: disable=invalid-name
    """``txl.TensorMap`` — a ``const __grid_constant__ CUtensorMap`` parameter."""


_TLS = threading.local()


# tirx-lite traces Python directly instead of going through the TVMScript parser.
# Keep the parser's source-location contract by recording the active user line
# while the body is being traced.  Frames in tirx-lite itself, TVM, and Python's
# runtime are implementation details; allowing them to update the active span
# would make diagnostics point into the DSL rather than to the kernel source.
_TXL_SOURCE_ROOT = Path(__file__).resolve().parent
_TVM_SOURCE_ROOT = Path(tvm.__file__).resolve().parent
_PYTHON_STDLIB_ROOT = Path(sysconfig.get_paths()["stdlib"]).resolve()
_PYTHON_SITE_ROOTS = tuple(
    {Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")}
)
_SOURCE_NAME_CACHE = {}
_SOURCE_SPAN_CACHE = {}


@cache
def _is_user_source(filename: str) -> bool:
    """Whether *filename* belongs to kernel author code rather than machinery."""
    if not filename or filename.startswith("<"):
        return False
    try:
        path = Path(filename).resolve()
    except OSError:
        return False
    if path == _TXL_SOURCE_ROOT or _TXL_SOURCE_ROOT in path.parents:
        return False
    if path == _TVM_SOURCE_ROOT or _TVM_SOURCE_ROOT in path.parents:
        return False
    in_stdlib = path == _PYTHON_STDLIB_ROOT or _PYTHON_STDLIB_ROOT in path.parents
    in_site_packages = any(root == path or root in path.parents for root in _PYTHON_SITE_ROOTS)
    if in_stdlib and not in_site_packages:
        return False
    return True


def _source_span(filename: str, line: int) -> Span | None:
    """Return the one-line span for a Python frame, if it is author code."""
    if not _is_user_source(filename):
        return None
    key = (filename, line)
    span = _SOURCE_SPAN_CACHE.get(key)
    if span is not None:
        return span
    source_name = _SOURCE_NAME_CACHE.setdefault(filename, SourceName(filename))
    source_line = linecache.getline(filename, line)
    end_column = max(2, len(source_line.rstrip("\r\n")) + 1)
    span = Span(source_name, line, line, 1, end_column)
    _SOURCE_SPAN_CACHE[key] = span
    return span


def _callable_span(func) -> Span | None:
    """Return the definition-line span used for generated entry scaffolding."""
    code = getattr(func, "__code__", None)
    if code is None:
        return None
    return _source_span(code.co_filename, code.co_firstlineno)


class _SourceSpanTracer:
    """Attach the current Python source line to statements emitted by tirx-lite."""

    def __init__(self, builder):
        self.builder = builder
        self.previous_trace = None
        self.active_context = None
        self.active_frame = None
        self.active_span = None
        self.frame_spans = {}

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, ptype, value, trace):  # pylint: disable=unused-argument
        sys.settrace(self.previous_trace)
        self._restore(None)
        self.frame_spans.clear()

    def _set_active(self, frame, span):
        if self.active_frame is frame and self.active_span is span:
            return
        if self.active_context is not None:
            self.active_context.__exit__(None, None, None)
        self.active_context = self.builder.with_source_span(span)
        self.active_context.__enter__()
        self.active_frame = frame
        self.active_span = span

    def _restore(self, frame):
        """Restore the nearest caller's span after a nested user helper returns."""
        if self.active_context is not None:
            self.active_context.__exit__(None, None, None)
            self.active_context = None
        self.active_frame = None
        self.active_span = None
        while frame is not None:
            span = self.frame_spans.get(frame)
            if span is not None:
                self.active_context = self.builder.with_source_span(span)
                self.active_context.__enter__()
                self.active_frame = frame
                self.active_span = span
                return
            frame = frame.f_back

    def _trace(self, frame, event, arg):  # pylint: disable=unused-argument
        if event == "call":
            if _is_user_source(frame.f_code.co_filename):
                frame.f_trace_lines = True
                return self._trace
            return None
        if event == "line":
            span = _source_span(frame.f_code.co_filename, frame.f_lineno)
            if span is not None:
                self.frame_spans[frame] = span
                self._set_active(frame, span)
            return self._trace
        if event == "return" and frame in self.frame_spans:
            self.frame_spans.pop(frame, None)
            if self.active_frame is frame:
                self._restore(frame.f_back)
        return self._trace


def current(required: bool = True) -> Session | None:
    """The kernel session being traced on this thread."""
    session = getattr(_TLS, "session", None)
    if session is None and required:
        raise RuntimeError(
            "no tirx-lite kernel is being traced; this call is only valid inside a @txl.kernel body"
        )
    return session


class Session:
    """Trace-time state of one ``@txl.kernel`` body.

    Owns the ``IRBuilder``, the bound scope ids, and the single smem pool and
    ``specialize`` object the body is allowed to create.
    """

    def __init__(self, name, warps, arch, min_blocks_per_sm):
        self.name = name
        self.warps = warps
        self.arch = arch
        self.min_blocks_per_sm = min_blocks_per_sm
        # None means the entry is UNPINNED: ptxas is free to choose, so there
        # is no promised occupancy to divide by. Keep the one-CTA model for
        # specialization diagnostics that do not request register transitions;
        # the low-level IR contract rejects setmaxnreg on such an entry.
        self.blocks_per_sm = 1 if min_blocks_per_sm is None else min_blocks_per_sm
        self.entry_regs = entry_regs(warps, self.blocks_per_sm)
        self.nthreads = warps * 32
        self.cta_id = None
        self.warp_scope_id = None
        self.lane_id = None
        self.thread_id = None
        self.params = {}
        self.pool = None
        self.specialize = None

    def warp_id(self):
        """The id the role dispatch compares against, as a **warp-uniform** value.

        ``threadIdx.x >> 5`` is already the same in every lane of a warp, but
        ptxas cannot see that: it keeps the value and everything predicated on
        it in the vector datapath. Broadcasting lane 0 through ``__shfl_sync``
        is a semantic no-op that *tells* the compiler, and the role predicate —
        plus the address math the roles hang off it — then lives in uniform
        registers.

        Measured on the GDN prefill port (7 configs, per-kernel GPU time, both
        kernels in one process): the plain shift runs at 1.014x the frozen
        hand-written kernel, the warp-uniform form at **0.977x** — a 3.7% swing,
        consistent in sign on every config. The frozen kernel does exactly this
        (`_make_warp_uniform`). An earlier revision of this method deliberately
        avoided the broadcast; that was measurably the wrong call.

        This costs nothing: the entry already declares the ``cta->warp`` scope
        id (``I.cuda.warp_id()``, emitted as ``warp_id_in_cta``) and that id
        is *already* the broadcast. Reusing it is one fewer instruction than
        even the plain shift, which computed a second, redundant value.
        """
        return self.warp_scope_id


class Kernel:
    """The traced kernel: a ``Function`` plus the launch metadata."""

    def __init__(self, func, session):
        self.func = func
        self.session = session
        self.name = session.name
        self.warps = session.warps
        self.arch = session.arch
        self.entry_regs = session.entry_regs

    @property
    def mod(self):
        return tvm.IRModule({"main": self.func})

    def target(self):
        # The decorator records the author's native/default target. Prepared
        # runs compile for the GPU that will execute the kernel, including
        # callers using Kernel.compile() instead of runner.compile_kernel().
        arch = os.environ.get("TIRX_PREPARE_CUDA_ARCH", self.arch)
        return tvm.target.Target({"kind": "cuda", "arch": arch})

    def compile(self, target=None):
        """Compile to a runnable module (``tir_pipeline="tirx"``)."""
        return tvm.compile(self.mod, target=target or self.target(), tir_pipeline="tirx")

    def source(self, target=None):
        """The generated CUDA source."""
        return self.compile(target).mod.imports[0].inspect_source()

    def __repr__(self):
        return f"<txl.kernel {self.name} warps={self.warps} arch={self.arch}>"


def _scalar_param(name, ann):
    """Create an unbound scalar entry variable from a dtype annotation."""
    ctor = getattr(I, ann, None)
    if ctor is None:
        if any(ch in ann for ch in (".", "[", "(")):
            raise TypeError(
                f"parameter {name!r}: annotation arrived as the string {ann!r} — "
                "`from __future__ import annotations` (PEP 563) stringifies "
                "annotations before @txl.kernel can read them. Remove that import "
                "from the kernel's module: tirx-lite kernels trace at decoration time "
                "and need live annotation objects."
            )
        raise TypeError(f"parameter {name!r}: unknown dtype token {ann!r}")
    return ctor()


def _derived_gptr_shape(name, ann, scalar_params):
    """Resolve a non-empty integer shape from the entry's scalar parameters."""
    try:
        shape = ann.shape_factory(MappingProxyType(scalar_params))
    except KeyError as error:
        missing = error.args[0]
        raise ValueError(
            f"gptr parameter {name!r} shape refers to unknown scalar parameter {missing!r}"
        ) from error
    if not isinstance(shape, tuple | list) or not shape:
        raise TypeError(f"gptr parameter {name!r} shape must return a non-empty tuple or list")
    for index, extent in enumerate(shape):
        if isinstance(extent, bool):
            raise TypeError(f"gptr parameter {name!r} shape extent {index} is boolean")
        if isinstance(extent, int):
            if extent <= 0:
                raise ValueError(f"gptr parameter {name!r} shape extent {index} must be positive")
            continue
        if not isinstance(extent, tvm.ir.Expr) or not isinstance(extent.ty, tvm.ir.PrimType):
            raise TypeError(
                f"gptr parameter {name!r} shape extent {index} must be an integer expression"
            )
        if not tvm.DataType(str(extent.ty.dtype)).is_integer:
            raise TypeError(
                f"gptr parameter {name!r} shape extent {index} must be integer, "
                f"got {extent.ty.dtype}"
            )
    return shape


def _declare_param(name, ann, scalar_params):
    """Turn one annotation into a Function parameter."""
    if isinstance(ann, gptr):
        if ann.shape_factory is not None:
            shape = _derived_gptr_shape(name, ann, scalar_params)
        else:
            # Symbolic dimensions stay parameter-local unless the annotation
            # explicitly owns a scalar-derived shape contract.
            shape = ann.shape or [tvm.tirx.Var(f"{name}_dim{i}", "int64") for i in range(ann.ndim)]
        return I.arg_(name, I.Tensor(shape, ann.dtype))
    if ann is TensorMap or isinstance(ann, TensorMap):
        return I.arg_(name, I.TensorMap())
    if isinstance(ann, str):
        return I.arg_(name, scalar_params[name])
    raise TypeError(
        f"parameter {name!r} has annotation {ann!r}; expected txl.gptr[dtype], "
        "txl.gptr[dtype, ndim], txl.gptr[dtype, shape], "
        "txl.TensorMap, or a dtype token such as txl.i32"
    )


def kernel(
    *,
    launch,
    options=None,
    arch: str = "sm_100a",
    host_prelude=None,
    allowed_func_calls: tuple[str, ...] = (),
    check_ir: bool = True,
):
    """Declare a kernel with a CUDA LaunchConfig and optional KernelOptions.

    ``launch`` accepts a configuration object or a factory over the bound ABI
    parameter mapping. The block must be a static, one-dimensional multiple
    of 32; grid and cluster may be multidimensional. Launch dimensions are
    independent of the index accessors used by the body.

    ``options.min_blocks_per_sm`` pins the occupancy contract used to validate
    register transitions in specialized warp roles. An omitted option leaves
    that contract unpinned, preserving CUDA's default launch-bounds behavior.

    ``host_prelude`` receives the bound ABI parameters before device entry and
    supplies the decorated function's keyword-only ``host`` argument.
    """
    from tvm.backend.cuda.launch import KernelOptions, LaunchConfig

    if options is not None and not isinstance(options, KernelOptions):
        raise TypeError("options must be a CUDA KernelOptions")

    def decorator(fn):
        sig = inspect.signature(fn)
        host_param = sig.parameters.get("host")
        if host_prelude is None:
            if host_param is not None:
                raise TypeError("keyword-only `host` requires host_prelude=")
        elif host_param is None or host_param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("host_prelude= requires a keyword-only `host` parameter")
        params = {}
        prev = getattr(_TLS, "session", None)
        function_span = _callable_span(fn)
        with IRBuilder() as ib:
            with I.function_():
                I.func_name_(fn.__name__)
                # Keep the authored/default target on the Function itself.
                # Consumers such as NumSim receive ``Kernel.func`` rather than
                # the wrapper, so ``Kernel.arch`` alone loses an ISA fact that
                # affects SM100 versus SM107 descriptor decoding.
                I.func_attr({"global_symbol": fn.__name__, "tirx.cuda_arch": arch})
                args = []
                scalar_params = {
                    pname: _scalar_param(pname, param.annotation)
                    for pname, param in sig.parameters.items()
                    if pname != "host" and isinstance(param.annotation, str)
                }
                for pname, param in sig.parameters.items():
                    if pname == "host" and host_prelude is not None:
                        continue
                    if param.annotation is inspect.Parameter.empty:
                        raise TypeError(f"kernel parameter {pname!r} needs an annotation")
                    value = _declare_param(pname, param.annotation, scalar_params)
                    params[pname] = value
                    args.append(value)
                config = launch(params) if callable(launch) else launch
                if not isinstance(config, LaunchConfig):
                    raise TypeError("launch must be a CUDA LaunchConfig or a factory returning one")
                if len(config.block) != 1:
                    raise ValueError("tirx-lite requires a one-dimensional block")
                from tvm.backend.cuda.launch._impl import _integer

                nthreads = _integer(config.block[0])
                if nthreads is None or nthreads % 32 or not 32 <= nthreads <= 1024:
                    raise ValueError(
                        "tirx-lite block must be a static multiple of 32 between 32 and 1024"
                    )
                min_blocks = options.min_blocks_per_sm if options is not None else None
                session = Session(fn.__name__, nthreads // 32, arch, min_blocks)
                session.params = params
                session.launch = config
                session.options = options
                host = host_prelude(params) if host_prelude is not None else None
                with I.device_entry(launch=config, options=options):

                    def index(name, value):
                        return I.bind(value, var=tvm.ir.Var(name, value.ty))

                    cta_ids = [
                        index("b" + axis, I.cuda.block_idx(axis))
                        for axis in "xyz"[: len(config.grid)]
                    ]
                    if len(cta_ids) == 1:
                        session.cta_id = cta_ids[0]
                    else:
                        from tvm_ffi import convert

                        session.cta_id = convert(cta_ids)
                    session.warp_scope_id = index("warp_id", I.cuda.warp_id())
                    session.lane_id = index("lane_id", I.cuda.lane_id())
                    session.thread_id = index("thread_id", I.cuda.linear_thread_id())
                    _TLS.session = session
                    try:
                        with _SourceSpanTracer(ib):
                            if host_prelude is None:
                                fn(*args)
                            else:
                                fn(*args, host=host)
                        if session.specialize is not None:
                            session.specialize.finalize()
                        if session.pool is not None:
                            session.pool.commit()
                    finally:
                        _TLS.session = prev
        func = ib.get()
        if function_span is not None:
            # FunctionFrame intentionally constructs the function with an
            # undefined span.  Preserve its generated body and attributes while
            # giving the function itself the kernel declaration's location.
            func = func.with_body(func.body, span=function_span)
        if session.specialize is not None:
            # Adjacent role guards become an else-if chain. Done on the built
            # body rather than with nested frames during the trace: a frame
            # left open across sibling `with role:` blocks would swallow any
            # CTA-scope code between them into the previous role's branch.
            func = session.specialize.chain_dispatch(func)
        if check_ir:
            # Every tirx-lite build passes the low-level IR contract by default:
            # direct global/shared buffer accesses and unlisted func_calls are
            # trace-time errors, not something a later test run discovers.
            from .low_level_ir import check_low_level_ir

            check_low_level_ir(func, allowed_func_calls=allowed_func_calls)
        return Kernel(func, session)

    return decorator


def cta_id():
    """Return the entry's CTA index, or its Array of indices for a multidimensional grid."""
    return current().cta_id


def thread_id():
    """Flattened CTA-local thread id owned by the current kernel entry."""
    return current().thread_id


def warp_id():
    """Warp-uniform CTA-local id owned by the current kernel entry."""
    return current().warp_id()


def lane_id():
    """Warp-local lane id owned by the current kernel entry."""
    return current().lane_id
