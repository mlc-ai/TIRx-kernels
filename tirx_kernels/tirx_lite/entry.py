# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Kernel entry frame exposed as ``txl.kernel`` and its trace-time session.

The decorator owns an ``IRBuilder`` and traces the decorated function once,
at decoration time. Plain Python control flow in the body is macro expansion;
nothing goes through the TVMScript parser.
"""

from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path

import tvm
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


_TLS = threading.local()
_TXL_SOURCE_ROOT = Path(__file__).resolve().parent


def current(required: bool = True) -> Session | None:
    """The kernel session being traced on this thread."""
    session = getattr(_TLS, "session", None)
    if session is None and required:
        raise RuntimeError(
            "this call requires an active txl.device_entry inside a @txl.kernel body"
        )
    return session


class Session:
    """Trace-time state of one ``@txl.kernel`` body.

    Owns the ``IRBuilder``, the bound scope ids, and the single smem pool and
    ``specialize`` object the body is allowed to create.
    """

    def __init__(self, name, warps, min_blocks_per_sm):
        self.name = name
        self.warps = warps
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
        self.entry_regs = session.entry_regs

    @property
    def mod(self):
        return tvm.IRModule({"main": self.func})

    def compile(self, target=None):
        """Compile with Target defaults overridden by the device entry settings."""
        if target is None:
            target = tvm.target.Target.current() or "cuda"
        return tvm.compile(self.mod, target=target, tir_pipeline="tirx")

    def source(self, target=None):
        """Return the CUDA source generated for the supplied compile settings."""
        return self.compile(target).mod.imports[0].inspect_source("cuda")

    def __repr__(self):
        return f"<txl.kernel {self.name} warps={self.warps}>"


def _declare_param(name, annotation):
    """Bind an ordinary native type or shared symbolic Var as an entry parameter."""
    if isinstance(annotation, str):
        annotation = tvm.ir.PrimType(annotation)
    return I.arg_(name, annotation)


class _TraceContext:
    """Function-wide tracing state; device state exists only inside its entry."""

    def __init__(self, name, params, builder):
        self.name = name
        self.params = params
        self.builder = builder
        self.function_frame = builder.frames[-1]
        self.entry = None


class _DeviceEntry:
    """One native region, closed by a with block or by the kernel decorator."""

    def __init__(self, context, launch, kernel_attrs, backend_config):
        from tvm.backend.cuda.launch import KernelAttributes, LaunchConfig
        from tvm.backend.cuda.launch._impl import _integer

        if not isinstance(launch, LaunchConfig):
            raise TypeError("device_entry launch must be a CUDA LaunchConfig")
        if kernel_attrs is not None and not isinstance(kernel_attrs, KernelAttributes):
            raise TypeError("kernel_attrs must be a CUDA KernelAttributes")
        if context.entry is not None:
            raise RuntimeError("a txl.kernel must declare exactly one device_entry")
        if not context.builder.frames[-1].same_as(context.function_frame):
            raise RuntimeError(
                "device_entry must be at function scope, outside TIR branches and loops"
            )
        if len(launch.block) != 1:
            raise ValueError("tirx-lite requires a one-dimensional block")
        nthreads = _integer(launch.block[0])
        if nthreads is None or nthreads % 32 or not 32 <= nthreads <= 1024:
            raise ValueError("tirx-lite block must be a static multiple of 32 between 32 and 1024")
        min_blocks = kernel_attrs.min_blocks_per_sm if kernel_attrs is not None else None
        self.session = Session(context.name, nthreads // 32, min_blocks)
        self.session.params = context.params
        self.session.launch = launch
        self.session.kernel_attrs = kernel_attrs
        self.frame = I.device_entry(
            launch=launch, kernel_attrs=kernel_attrs, backend_config=backend_config
        )
        self.closed = False
        self.managed = False
        self.frame.__enter__()
        context.entry = self
        _TLS.session = self.session

        def index(name, value):
            return I.bind(value, var=tvm.ir.Var(name, value.ty))

        cta_ids = [index("b" + axis, I.cuda.block_idx(axis)) for axis in "xyz"[: len(launch.grid)]]
        if len(cta_ids) == 1:
            self.session.cta_id = cta_ids[0]
        else:
            from tvm_ffi import convert

            self.session.cta_id = convert(cta_ids)
        self.session.warp_scope_id = index("warp_id", I.cuda.warp_id())
        self.session.lane_id = index("lane_id", I.cuda.lane_id())
        self.session.thread_id = index("thread_id", I.cuda.linear_thread_id())

    def __enter__(self):
        if self.closed or self.managed:
            raise RuntimeError("device_entry cannot be entered more than once")
        self.managed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.closed:
            return
        try:
            if exc_type is None:
                if self.session.specialize is not None:
                    self.session.specialize.finalize()
                if self.session.pool is not None:
                    self.session.pool.commit()
            self.frame.__exit__(exc_type, exc_value, traceback)
        finally:
            self.closed = True
            _TLS.session = None


def device_entry(*, launch, kernel_attrs=None, backend_config=None):
    """Start the kernel's device region with explicit CUDA launch configuration.

    A flat call covers the remainder of the traced function. A ``with`` block
    closes the region explicitly and permits subsequent host statements. Host
    preparation belongs before entry; launch operands can refer directly to
    function parameters. Exactly one top-level entry is required per kernel.

    The block must be a static, one-dimensional multiple of 32, between 32 and
    1024. Grid and cluster can be multidimensional. ``kernel_attrs`` supplies
    compile-time CUDA attributes, including the occupancy contract used by
    specialized warp roles.
    """
    context = getattr(_TLS, "trace", None)
    if context is None:
        raise RuntimeError("device_entry is only valid inside a @txl.kernel body")
    return _DeviceEntry(context, launch, kernel_attrs, backend_config)


def kernel(*, allowed_func_calls: tuple[str, ...] = (), check_ir: bool = True, **obsolete):
    """Trace a host function containing one explicit ``txl.device_entry``.

    Annotations bind the host entry's ABI. The function prepares host values
    and declares its launch configuration in the body, using those parameters
    directly. Only values used by the device body become device parameters.
    """

    if obsolete:
        raise TypeError(
            f"Unsupported txl.kernel options {tuple(obsolete)}; set BackendConfig on "
            "the compile Target or device_entry(backend_config=...)"
        )

    def decorator(fn):
        sig = inspect.signature(fn)
        params = {}
        previous_trace = getattr(_TLS, "trace", None)
        previous_session = getattr(_TLS, "session", None)
        function_loc = None
        try:
            _TLS.trace = None
            _TLS.session = None
            with IRBuilder() as ib:
                tracer = ib.line_trace_scope(exclude_paths=(_TXL_SOURCE_ROOT,))
                code = getattr(fn, "__code__", None)
                if code is not None:
                    function_loc = tracer.source_loc(code.co_filename, code.co_firstlineno)
                with I.function_():
                    I.func_name_(fn.__name__)
                    I.func_attr({"global_symbol": fn.__name__})
                    args, kwargs = [], {}
                    for pname, param in sig.parameters.items():
                        if param.annotation is inspect.Parameter.empty:
                            raise TypeError(f"kernel parameter {pname!r} needs an annotation")
                        value = _declare_param(pname, param.annotation)
                        params[pname] = value
                        if param.kind is inspect.Parameter.KEYWORD_ONLY:
                            kwargs[pname] = value
                        else:
                            args.append(value)
                    context = _TraceContext(fn.__name__, params, ib)
                    _TLS.trace = context
                    with tracer:
                        try:
                            fn(*args, **kwargs)
                        except BaseException:
                            if context.entry is not None:
                                context.entry.__exit__(*sys.exc_info())
                            raise
                        if context.entry is None:
                            raise RuntimeError("a txl.kernel must declare exactly one device_entry")
                        context.entry.__exit__(None, None, None)
                    session = context.entry.session
            func = ib.get()
        finally:
            _TLS.trace = previous_trace
            _TLS.session = previous_session
        if function_loc is not None:
            func = func.with_body(func.body, loc=function_loc)
        if session.specialize is not None:
            # Keep adjacent role guards mutually exclusive after tracing.
            func = session.specialize.chain_dispatch(func)
        if check_ir:
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
