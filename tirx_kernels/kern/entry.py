# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Kernel entry frame exposed as ``K.kernel`` and its trace-time session.

The decorator owns an ``IRBuilder`` and traces the decorated function once,
at decoration time. Plain Python control flow in the body is macro expansion;
nothing goes through the TVMScript parser.
"""

from __future__ import annotations

import inspect
import threading
from functools import partial

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script.builder import ir as I

# Registers the hardware hands the entry when ptxas pins the allocation with
# __launch_bounds__(nthreads, min_blocks_per_sm). setmaxnreg's direction is
# read off this: a role asking for more must .inc, for less .dec.
REGS_PER_CTA = 65536
_REGS_PER_THREAD_MAX = 255

# The name ScriptMacro._find_parser_def looks for in each caller's module globals.
_PARSER_GLOBAL = "__current_script_parser__"


def entry_regs(warps: int, min_blocks_per_sm: int = 1) -> int:
    """The per-thread register allocation ptxas pins for a *warps*-wide entry.

    The register file split over the threads that must fit on the SM at once,
    capped at the architectural per-thread maximum and rounded down to the
    8-register allocation granularity.

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

    No in-tree kernel currently combines ``min_blocks_per_sm > 1`` with roles
    that set ``regs=``, so the ``m`` term is not fixing a live miscompile. What
    it fixes is a model that lied: the number this returned was wrong whenever
    ``m > 1``, ``Specialize.role`` quoted it in an error message that was
    therefore false, and the first kernel to combine the two would have been
    told its allocation was 248 while ptxas worked to 32.

    The 255 cap matters too: a 256-thread CTA divides out to 256 registers, but
    a thread can only hold 255, so the entry really gets 248 — and a role asking
    for 256 is an *increase*, not a decrease.
    """
    per_thread = REGS_PER_CTA // (min_blocks_per_sm * warps * 32)
    return min(per_thread, _REGS_PER_THREAD_MAX) & ~7


class gptr:  # pylint: disable=invalid-name
    """A global buffer parameter with symbolic rank or a fixed shape.

    ``K.gptr[dtype]`` and ``K.gptr[dtype, ndim]`` give every dimension a fresh
    symbolic ``int64`` extent. ``K.gptr[dtype, shape]`` owns the exact fixed
    extents when they are part of the specialized kernel contract.
    """

    def __init__(self, dtype: str, ndim: int | tuple[int, ...] = 1):
        self.shape = None
        if isinstance(ndim, tuple):
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
                    "gptr expects K.gptr[dtype], K.gptr[dtype, ndim], or K.gptr[dtype, shape]"
                )
            return cls(*dtype)
        return cls(dtype)

    def __repr__(self):
        if self.shape is not None:
            return f"K.gptr[{self.dtype!r}, {self.shape!r}]"
        if self.ndim != 1:
            return f"K.gptr[{self.dtype!r}, {self.ndim}]"
        return f"K.gptr[{self.dtype!r}]"


class TensorMap:  # pylint: disable=invalid-name
    """``K.TensorMap`` — a ``const __grid_constant__ CUtensorMap`` parameter."""


_TLS = threading.local()


def current(required: bool = True) -> Session | None:
    """The kernel session being traced on this thread."""
    session = getattr(_TLS, "session", None)
    if session is None and required:
        raise RuntimeError(
            "no kern kernel is being traced; this call is only valid inside a @K.kernel body"
        )
    return session


class Session:
    """Trace-time state of one ``@K.kernel`` body.

    Owns the ``IRBuilder``, the bound scope ids, and the single smem pool and
    ``specialize`` object the body is allowed to create.
    """

    def __init__(self, name, warps, arch, min_blocks_per_sm):
        self.name = name
        self.warps = warps
        self.arch = arch
        self.min_blocks_per_sm = min_blocks_per_sm
        # None means the entry is UNPINNED: ptxas is free to choose, so there
        # is no promised occupancy to divide by. The one-CTA reading is the
        # only defensible model, and a role with regs= -- the only consumer
        # that needs the number to be real -- is refused outright while it
        # holds (Specialize.role).
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
        id (``I.warp_id([warps])``, emitted as ``warp_id_in_cta``) and that id
        is *already* the broadcast. Reusing it is one fewer instruction than
        even the plain shift, which computed a second, redundant value.
        """
        return self.warp_scope_id


class Kernel:
    """The traced kernel: a ``PrimFunc`` plus the launch metadata."""

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
        return tvm.target.Target({"kind": "cuda", "arch": self.arch})

    def compile(self, target=None):
        """Compile to a runnable module (``tir_pipeline="tirx"``)."""
        return tvm.compile(self.mod, target=target or self.target(), tir_pipeline="tirx")

    def source(self, target=None):
        """The generated CUDA source."""
        return self.compile(target).mod.imports[0].inspect_source()

    def __repr__(self):
        return f"<K.kernel {self.name} warps={self.warps} arch={self.arch}>"


def _declare_param(name, ann):
    """Turn one annotation into a PrimFunc parameter."""
    if isinstance(ann, gptr):
        # Symbolic dimensions stay parameter-local: sharing would impose shape
        # equalities that are not part of the raw-pointer contract.
        shape = ann.shape or [tvm.tirx.Var(f"{name}_dim{i}", "int64") for i in range(ann.ndim)]
        return I.arg(name, I.buffer(shape, ann.dtype))
    if ann is TensorMap or isinstance(ann, TensorMap):
        return I.arg(name, I.TensorMap())
    if isinstance(ann, str):
        ctor = getattr(I, ann, None)
        if ctor is None:
            if any(ch in ann for ch in ".[("):
                raise TypeError(
                    f"parameter {name!r}: annotation arrived as the string {ann!r} — "
                    "`from __future__ import annotations` (PEP 563) stringifies "
                    "annotations before @K.kernel can read them. Remove that import "
                    "from the kernel's module: kern kernels trace at decoration time "
                    "and need live annotation objects."
                )
            raise TypeError(f"parameter {name!r}: unknown dtype token {ann!r}")
        return I.arg(name, ctor())
    raise TypeError(
        f"parameter {name!r} has annotation {ann!r}; expected K.gptr[dtype], "
        "K.gptr[dtype, ndim], K.gptr[dtype, shape], "
        "K.TensorMap, or a dtype token such as K.i32"
    )


def _make_host_parser():
    """A parser object for the in-tree ``@T.inline`` helpers to run against.

    ``Pipeline`` / ``MBarrier`` / ``PipelineState`` — the protocol layer a kern
    kernel uses raw — are built out of ``@T.inline`` methods, and an inline
    re-parses its own source into whatever IRBuilder is current. To find the
    parser it walks the call stack for a module global named
    ``__current_script_parser__``. A traced body has no parser at all, so the
    session lends it one: the source and var table are replaced by the inline
    itself, leaving only the dispatch tokens to set (``tirx``, the token the
    ``T.prim_func`` parser pushes around a function body).

    The global lives in *this* module because ``kernel``'s call into the user
    function is the frame that keeps it on the stack for the whole trace.
    """
    from tvm.script.parser.core.diagnostics import Source
    from tvm.script.parser.core.parser import Parser

    parser = Parser(Source("pass"), {})
    parser.dispatch_tokens = ["default", "tirx"]
    return parser


def _flat_frame(frame):
    """Open *frame* so it wraps the rest of the body without a ``with`` block.

    ``T.attr`` returns a frame the parser would have entered from an
    expression statement; a traced body has to do it by hand.
    """
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


def _resolve_grid(grid, params):
    if grid is None:
        return params["num_sms"] if "num_sms" in params else 1
    if isinstance(grid, (list, tuple)):
        if not grid:
            raise ValueError("grid must contain at least one CTA dimension")
        dimensions = [_resolve_grid(dimension, params) for dimension in grid]
        if any(isinstance(dimension, list) for dimension in dimensions):
            raise ValueError("grid dimensions cannot be nested")
        return dimensions
    if isinstance(grid, str):
        if grid not in params:
            raise ValueError(f"grid={grid!r} does not name a kernel parameter")
        return params[grid]
    # TIR expressions implement ``__call__`` for function-call construction;
    # they are values here, not grid callbacks.
    if callable(grid) and not isinstance(grid, tvm.ir.Expr):
        return _resolve_grid(grid(params), params)
    return grid


def _resolve_grid_dimensions(grid, params):
    """Resolve and validate the one-to-three CTA extents owned by K."""
    dimensions = _resolve_grid(grid, params)
    if not isinstance(dimensions, list):
        dimensions = [dimensions]
    if len(dimensions) > 3:
        raise ValueError(f"grid supports at most three CTA dimensions, got {len(dimensions)}")
    for index, extent in enumerate(dimensions):
        if isinstance(extent, bool):
            raise TypeError(f"grid dimension {index} must be an integer extent, got {extent!r}")
        if isinstance(extent, int):
            if extent <= 0:
                raise ValueError(f"grid dimension {index} must be positive, got {extent}")
            continue
        if not isinstance(extent, tvm.ir.Expr) or not isinstance(extent.ty, tvm.ir.PrimType):
            raise TypeError(f"grid dimension {index} must be an integer extent, got {extent!r}")
        dtype = tvm.DataType(str(extent.ty.dtype))
        if not dtype.is_integer:
            raise TypeError(
                f"grid dimension {index} must have an integer dtype, got {extent.ty.dtype}"
            )
        if isinstance(extent, tvm.tirx.IntImm) and int(extent) <= 0:
            raise ValueError(f"grid dimension {index} must be positive, got {int(extent)}")
    return dimensions


def kernel(
    warps: int,
    arch: str = "sm_100a",
    min_blocks_per_sm: int | None = None,
    grid=None,
    thread_layout: str | bool = "flat",
    host_prelude=None,
):
    """Declare a kernel entry.

    Parameters
    ----------
    warps : int
        CTA width in warps. ``flat`` uses ``blockDim.x = warps * 32``;
        ``lane_warp`` uses ``blockDim = (32, warps, 1)``.
    arch : str
        CUDA arch for the default compile target.
    min_blocks_per_sm : int, optional
        Second ``__launch_bounds__`` operand. **Defaults to None, which omits
        it**, emitting ``__launch_bounds__(nthreads)`` — the spelling in-tree
        kernels use. The operand is a floor on occupancy, so pinning it also
        pins the entry's register allocation *down*: on an rmsnorm kernel the
        forced ``, 1`` cost 32 -> 54 registers (the kernel's measured
        ``-Xptxas -v`` table; an earlier 53 here was a transcription slip), and
        the ``None`` default matches the original at every configured size —
        including one (hs=128, 34 regs) where a pin of ``8`` forced 32 vs the
        original's 34. Pass an int when the kernel wants that
        trade, and note that pinning is what makes the entry allocation — and
        therefore the ``setmaxnreg`` direction a role must use — well defined:
        a role declared with ``regs=`` requires it (see
        :meth:`Specialize.role`).

        A high value and warp roles pull against each other, and the pull is
        sharp. The whole register file is a CTA's only at one block per SM;
        promising *m* leaves ``REGS_PER_CTA // m`` for every role to share.
        An 8-warp CTA at ``m=8`` has 8192 registers total — 32 a thread — so
        its roles can afford, say, 40 and 24 but not 64 and 64, which
        :meth:`Specialize.finalize` refuses. The rmsnorm-style
        ``min_blocks_per_sm=8`` is advice for occupancy-bound kernels without
        roles; a warp-specialized kernel usually wants 1.
    grid : int | str | list | tuple | callable, optional
        CTA extents: one constant or kernel-parameter name for ``blockIdx.x``,
        or a sequence for ``blockIdx.x/y/z``. A callable over the bound
        parameters may return either form. Defaults to the parameter named
        ``num_sms`` when the kernel has one, else 1. Pass ``False`` to leave
        the kernel-to-CTA scope to the body, which can then declare the
        original kernel's dimensions directly with the TIRx IRBuilder. This
        is an ownership opt-out, not a second grid representation in K.
    thread_layout : {"flat", "lane_warp", False}
        ``flat`` binds a single ``warps * 32`` ``threadIdx.x`` axis.
        ``lane_warp`` binds ``threadIdx.x`` to the lane and ``threadIdx.y``
        to the warp while preserving the same flattened ``K.thread_id()``.
        Pass ``False`` to leave the thread scope to the body, matching
        ``grid=False`` for kernels whose original contract owns its axes.
    host_prelude : callable, optional
        Emit host-only setup in the same traced PrimFunc before its device
        entry.  The callable receives the ABI parameter mapping and returns
        the trace-time value passed to the decorated function's required
        keyword-only ``host`` parameter.  This is for real host/device
        contracts such as runtime TensorMap encoding; the returned values are
        IR objects owned by this one function, not a second body to splice.
    """

    def decorator(fn):
        if thread_layout not in ("flat", "lane_warp", False):
            raise ValueError(f"unsupported thread_layout={thread_layout!r}")
        sig = inspect.signature(fn)
        host_param = sig.parameters.get("host")
        if host_prelude is None:
            if host_param is not None:
                raise TypeError("keyword-only `host` requires host_prelude=")
        elif host_param is None or host_param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("host_prelude= requires a keyword-only `host` parameter")
        session = Session(fn.__name__, warps, arch, min_blocks_per_sm)
        prev = getattr(_TLS, "session", None)
        with IRBuilder() as ib:
            with I.prim_func():
                I.func_name(fn.__name__)
                I.func_attr({"global_symbol": fn.__name__})
                args = []
                for pname, param in sig.parameters.items():
                    if pname == "host" and host_prelude is not None:
                        continue
                    if param.annotation is inspect.Parameter.empty:
                        raise TypeError(f"kernel parameter {pname!r} needs an annotation")
                    value = _declare_param(pname, param.annotation)
                    session.params[pname] = value
                    args.append(value)
                host = host_prelude(session.params) if host_prelude is not None else None
                I.device_entry()
                # Omitted entirely when None: the codegen writes the second
                # __launch_bounds__ operand iff this attribute is present.
                if min_blocks_per_sm is not None:
                    _flat_frame(I.attr({"tirx.launch_bounds_min_blocks_per_sm": min_blocks_per_sm}))
                # K binds the CTA scope only when requested. Otherwise the body
                # declares the original kernel's scope directly; the warp/thread
                # siblings below remain available to infer deferred ids used by
                # user closures.
                if grid is not False:
                    session.cta_id = I.cta_id(_resolve_grid_dimensions(grid, session.params))
                if thread_layout == "flat":
                    session.warp_scope_id = I.warp_id([warps])
                    session.lane_id = I.lane_id([32])
                    session.thread_id = I.thread_id([session.nthreads])
                elif thread_layout == "lane_warp":
                    session.lane_id, session.warp_scope_id = I.thread_id([32, warps])
                    session.thread_id = session.warp_scope_id * 32 + session.lane_id
                _TLS.session = session
                prev_parser = globals().get(_PARSER_GLOBAL)
                globals()[_PARSER_GLOBAL] = _make_host_parser()
                try:
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
                    if prev_parser is None:
                        globals().pop(_PARSER_GLOBAL, None)
                    else:
                        globals()[_PARSER_GLOBAL] = prev_parser
        func = ib.get()
        if session.specialize is not None:
            # Adjacent role guards become an else-if chain. Done on the built
            # body rather than with nested frames during the trace: a frame
            # left open across sibling `with role:` blocks would swallow any
            # CTA-scope code between them into the previous role's branch.
            func = session.specialize.chain_dispatch(func)
        return Kernel(func, session)

    return decorator


def cta_id(extents=None, preferred=None, dtype="int32"):
    """CTA id owned by ``@K.kernel``.

    A one-dimensional grid returns its scalar scope id; a multidimensional
    grid returns the corresponding TVM ``Array`` of scope ids. Passing
    ``extents`` declares the scope directly for a ``grid=False`` kernel.
    """
    if extents is not None:
        return I.cta_id(extents, preferred=preferred, dtype=dtype)
    value = current().cta_id
    if value is None:
        raise RuntimeError(
            "K.cta_id() is unavailable because @K.kernel set grid=False. "
            "Declare the kernel-to-CTA scope directly with the TIRx IRBuilder "
            "and use the returned scope id(s)."
        )
    return value


def thread_id(extents=None, dtype="int32"):
    """Flattened CTA-local thread id, or an explicitly declared thread scope."""
    if extents is not None:
        return I.thread_id(extents, dtype=dtype)
    value = current().thread_id
    if value is None:
        raise RuntimeError(
            "K.thread_id() is unavailable because @K.kernel set thread_layout=False. "
            "Declare the thread scope directly with the TIRx IRBuilder and use its id."
        )
    return value


def warp_id():
    """Warp-uniform CTA-local id owned by the current kernel entry."""
    value = current().warp_id()
    if value is None:
        raise RuntimeError(
            "K.warp_id() is unavailable because @K.kernel set thread_layout=False. "
            "Declare the thread scope directly with the TIRx IRBuilder and derive its warp id."
        )
    return value


def lane_id():
    """Warp-local lane id owned by the current kernel entry."""
    value = current().lane_id
    if value is None:
        raise RuntimeError(
            "K.lane_id() is unavailable because @K.kernel set thread_layout=False. "
            "Declare the thread scope directly with the TIRx IRBuilder and derive its lane id."
        )
    return value
