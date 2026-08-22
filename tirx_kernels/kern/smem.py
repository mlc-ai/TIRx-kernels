# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""Shared-memory allocation exposed as ``K.smem_pool`` plus address constructors.

``swizzle=None`` is the identity: a plain row-major tirx buffer. A swizzle
mode makes the allocation a *composed* layout (the in-tree
``mma_shared_layout``: the last two dims tiled into ``[8, atom_bytes/elem]``
atoms, xor inside the atom).

Addresses inside a swizzled tile are named through a :class:`KTileView` and
nothing else. A staged (3-D) allocation hands out one per stage --
``tile[stage]`` -- and an unstaged (2-D) allocation *is* one; either way the
view carries the four constructors ``ptr_to`` / ``m8n8`` / ``m8n8x4`` /
``mma_desc``, all taking ``(row, col)`` inside the stage. There is no flat
``tile.ptr_to(stage, row, col)``: the stage lives in the view, so there is
exactly one spelling for an address rather than two.

The constructors emit nothing, except ``mma_desc``, which emits one
``encode_matrix_descriptor``.
"""

from __future__ import annotations

import warnings

from tvm import DataType
from tvm.backend.cuda.lang import SMEMPool
from tvm.backend.cuda.lang.alloc_pool import _validate_mma_alloc_shape
from tvm.backend.cuda.tile_primitive.layout_utils import strip_swizzle_to_tile
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.tirx.analysis import undefined_vars
from tvm.tirx.script.builder import ir as _I

from . import entry as _entry

_ATOM_BYTES = {
    SwizzleMode.SWIZZLE_32B_ATOM: 32,
    SwizzleMode.SWIZZLE_64B_ATOM: 64,
    SwizzleMode.SWIZZLE_128B_ATOM: 128,
}


def smem_desc_add_16B_offset(desc_val, offset):
    """Add a 16-byte-unit offset to the low half of an SMEM descriptor."""
    # Descriptor offsets wrap in the low 32 bits without carrying into the
    # encoded layout fields in the high half.
    desc_lo = T.alloc_local((1,), "uint32")
    desc_hi = T.alloc_local((1,), "uint32")
    result = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mov.b64(desc_lo[0], desc_hi[0], desc_val))
    T.evaluate(T.ptx.add.u32(desc_lo[0], desc_lo[0], T.cast(offset, "uint32")))
    T.evaluate(T.ptx.mov.b64(result[0], desc_lo[0], desc_hi[0]))
    return result[0]


@T.meta_class
class SmemDescriptor:
    """Raw tcgen05 shared-memory descriptor with native offset arithmetic."""

    def __init__(self):
        self._buf = T.alloc_local((1,), "uint64")

    @property
    def desc(self):
        return self._buf[0]

    def init(self, smem_ptr, ldo, sdo, swizzle):
        _I.evaluate(
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(self._buf[0]), smem_ptr, ldo, sdo, swizzle
            )
        )

    def make_lo_uniform(self):
        """Broadcast the descriptor's low half without a device helper call.

        The TVM ``lang.smem_desc`` helper emits a ``tirx.cuda.func_call`` for
        this operation.  Kern descriptors stay inside the low-level IR
        contract by spelling the same operation as the two descriptor moves
        and one warp shuffle directly.
        """
        desc_lo = T.alloc_local((1,), "uint32")
        desc_hi = T.alloc_local((1,), "uint32")
        T.evaluate(T.ptx.mov.b64(desc_lo[0], desc_hi[0], self._buf[0]))
        _I.buffer_store(
            desc_lo,
            T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), desc_lo[0], T.uint32(0), T.uint32(32)),
            [0],
        )
        T.evaluate(T.ptx.mov.b64(self._buf[0], desc_lo[0], desc_hi[0]))

    def add_16B_offset(self, offset):
        return smem_desc_add_16B_offset(self._buf[0], offset)


class KStep:
    """The per-k-tile descriptor step of a :class:`KDesc`.

    A typed step rather than an ``int`` because the offset is *not* linear in
    the k-tile index: a K-major walk moves inside one swizzle atom for the
    first ``atom_cols / mma_k`` tiles and then jumps a whole atom. Writing
    ``d + kp * d.k_step`` with a plain int would be silently wrong from that
    tile on.

    ``__rmul__`` returns the exact offset for a trace-time-constant ``kp``; a
    symbolic ``kp`` is only accepted where the walk really is linear.

    (A variant that also emits the exact two-term walk for a *runtime* ``kp``,
    letting a kernel keep its k-loop rolled the way the frozen GDN kernel does,
    was prototyped during the GDN port and measured no faster -- the issuer
    warps are latency-bound on the matrix engine, not on instruction issue.)
    """

    def __init__(self, offset_of, n_steps, linear, why_not_linear=""):
        self._offset_of = offset_of
        self._n_steps = n_steps
        self._linear = linear
        self._why = why_not_linear
        self.base = offset_of(1)

    def of(self, kp):
        """The 16B-unit offset of k-tile *kp*."""
        return self.__rmul__(kp)

    def __rmul__(self, kp):
        if isinstance(kp, int):
            if not 0 <= kp < self._n_steps:
                raise ValueError(f"k-tile index {kp} is outside the tile's {self._n_steps} k-tiles")
            return self._offset_of(kp)
        if not self._linear:
            raise TypeError("k_step cannot be scaled by a runtime expression here: " + self._why)
        return kp * self.base

    __mul__ = __rmul__

    def __int__(self):
        return self.base

    def __repr__(self):
        return f"<K.k_step base={self.base} steps={self._n_steps} linear={self._linear}>"


def _lint_invariant_encode(stage):
    """Warn when a ``mma_desc`` encode inside a runtime loop cannot vary with it.

    ``encode_matrix_descriptor`` is a real instruction sequence, so one left in
    a hot loop that recomputes the same descriptor every iteration is pure
    overhead. Detecting that is cheap in the one case where it is *provable*:
    if the encode's only runtime input -- ``stage`` -- refers to no variable at
    all, the descriptor is a constant and every iteration recomputes it.

    The check is deliberately not smarter than that, and it never dedups.
    The pipelined case that *looks* invariant is the common correct one: a
    kernel encodes from ``pipe_state.stage``, which is a **mutable Var**
    advanced by ``advance()``. Two such encodes are structurally equal while
    denoting different stages, so structural equality lies here and silently
    reusing an earlier descriptor would emit a kernel that reads the wrong
    ring stage. Referring to any variable is therefore treated as varying, and
    the diagnostic is a warning the author can act on rather than a rewrite.
    """
    try:
        from tvm.script.ir_builder import IRBuilder
        from tvm.tirx.script.builder.frame import ForFrame

        builder = IRBuilder.current()
    except Exception:  # pylint: disable=broad-except
        return
    if builder is None or not any(isinstance(f, ForFrame) for f in builder.frames):
        return
    if not isinstance(stage, int) and undefined_vars(stage):
        return
    warnings.warn(
        "mma_desc: encode is invariant w.r.t. the enclosing loop; hoist it. "
        f"stage={stage!r} refers to no loop-varying value, so this descriptor is "
        "re-encoded identically on every iteration. If the stage is meant to "
        "advance, pass the PipelineState's `.stage`.",
        stacklevel=3,
    )


class KDesc:
    """A tcgen05 smem matrix descriptor, encoded once.

    ``desc + off16`` is one IADD on the low half (``smem_desc_add_16B_offset``);
    ``desc.k_step`` supplies the offsets.
    """

    def __init__(self, buf, k_step, major="k"):
        self._buf = buf
        self.k_step = k_step
        #: which axis this descriptor was encoded to contract along; a chain
        #: built from a different transpose than the encode is a silent
        #: wrong-operand bug, so the encode records its own answer.
        self.major = major

    @property
    def value(self):
        """The raw ``uint64`` descriptor (offset 0)."""
        return self._buf[0]

    def __add__(self, offset):
        if isinstance(offset, int) and offset == 0:
            return self._buf[0]
        return smem_desc_add_16B_offset(self._buf[0], offset)

    __radd__ = __add__


class KTile:
    """A **staged** swizzled shared-memory tile. It names no address itself.

    The only thing a 3-D tile does is hand out a stage: ``tile[stage]`` returns
    a :class:`KTileView`, and the view carries the four operand constructors.
    There is deliberately no flat ``tile.ptr_to(stage, row, col)`` spelling —
    one way to name an address, not two.
    """

    def __init__(self, buf, shape, dtype, swizzle):
        self.buf = buf
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.swizzle = swizzle
        self.stages = self.shape[0] if len(self.shape) == 3 else None
        self.rows = self.shape[-2]
        self.cols = self.shape[-1]
        self.bits = DataType(dtype).bits
        self.atom_bytes = _ATOM_BYTES[swizzle]
        # One swizzle atom is [8, atom_cols]; addresses step in 16B units.
        self.atom_cols = self.atom_bytes * 8 // self.bits
        self.elem_per_16b = 128 // self.bits
        self._tile_layout = strip_swizzle_to_tile(buf.ty.layout, lambda: self.shape)

    # -- physical (pre-xor) offsets, straight off the allocated layout --------

    def _phys(self, coord):
        linear = 0
        for value, extent in zip(coord, self.shape):
            linear = linear * extent + value
        return int(self._tile_layout.apply(linear)["m"])

    def _coord(self, stage, row, col):
        return (stage, row, col) if self.stages is not None else (row, col)

    def __getitem__(self, stage):
        """``tile[stage]`` — the view of one ring stage.

        The stage may be any expression, including a ``PipelineState.stage``
        var; see :class:`KTileView` for the lifetime rule that comes with that.
        """
        if self.stages is None:
            raise TypeError(
                "this tile has no stage dimension; it is already a view -- call "
                "ptr_to/m8n8/m8n8x4/mma_desc on it directly"
            )
        return KTileView(self, stage)

    def __repr__(self):
        return f"<K.tile {self.shape} {self.dtype} {self.swizzle.name}>"


class KTileView:
    """One stage of a swizzled tile: the four operand constructors, 2-D coords.

    Obtained as ``tile[stage]`` for a staged tile; an unstaged (2-D) swizzled
    allocation *is* one of these already. All coordinates are ``(row, col)``
    inside the stage — the stage is in the view, never in the coordinate list.

    The constructors emit nothing except ``mma_desc``, which emits exactly one
    ``encode_matrix_descriptor``.

    **Lifetime.** A view captures the stage *expression*, not its value at
    construction. ``PipelineState.stage`` is a mutable ``Var`` that
    ``advance()`` rewrites, so a view held across an ``advance()`` silently
    starts naming a different stage. Views are ephemeral operands: take one
    after the wait, use it before the release. To hold a stage across an
    advance, copy it into an int local first and index with that -- which is
    exactly what the GDN kernel's ``a2[j] = st_ainv.stage`` discipline does.
    """

    def __init__(self, tile, stage=None):
        self.tile = tile
        self.stage = stage

    # -- the tile's own attributes, so a view is usable wherever a tile was ---

    @property
    def buf(self):
        return self.tile.buf

    @property
    def dtype(self):
        return self.tile.dtype

    @property
    def bits(self):
        return self.tile.bits

    @property
    def rows(self):
        return self.tile.rows

    @property
    def cols(self):
        return self.tile.cols

    @property
    def swizzle(self):
        return self.tile.swizzle

    @property
    def shape(self):
        return self.tile.shape

    def _coord(self, row, col):
        return self.tile._coord(self.stage, row, col)

    # -- the four constructors ------------------------------------------------

    def ptr_to(self, row, col):
        """Address of element ``(row, col)`` of this stage, swizzle applied."""
        return self.tile.buf.ptr_to(list(self._coord(row, col)))

    def m8n8(self, row, col):
        """``ldmatrix``/``stmatrix`` ``.m8n8`` operand — the caller's own lane math."""
        return self.ptr_to(row, col)

    def m8n8x4(self, row, col, lane):
        """``.x4`` operand: the fixed lane distribution over four 8x8 matrices.

        ``m8n8x4(r, c, lane)`` is ``ptr_to(r + (lane & 15), c + 8 * (lane >> 4))``.
        """
        return self.ptr_to(row + (lane & 15), col + 8 * (lane >> 4))

    def mma_desc(self, major="k", mma_k=None):
        """Encode this stage's tcgen05 matrix descriptor once.

        ``major="k"`` contracts along the tile's contiguous (last) axis;
        ``major="mn"`` contracts along the row axis instead — the physical tile
        is the same, only which axis is K differs.

        This stays public for kernels that issue ``tcgen05.mma`` through bare
        ``K.ptx`` rather than ``K.idioms.mma_chain``; the chain takes views and
        encodes internally, so chain callers never call this.
        """
        return self.encode(major=major, mma_k=mma_k)[0]

    # -- descriptor derivation, shared by mma_desc and K.idioms.mma_chain -----

    def encode(self, major="k", mma_k=None, k0=0):
        """``(KDesc encoded at k == k0, off16_of)``.

        ``off16_of(kp)`` is the 16-byte-unit offset of the ``kp``-th k-tile
        *after* ``k0`` — a trace-time ``int``, so ``desc + off16_of(kp)`` is one
        IADD on the descriptor's low half.
        """
        tile = self.tile
        if major not in ("k", "mn"):
            raise ValueError(f"major must be 'k' or 'mn', got {major!r}")
        mma_k = mma_k if mma_k is not None else 256 // tile.bits
        _lint_invariant_encode(self.stage)

        # Closed form of the in-tree gemm_async derivation (_atom_off over the
        # mma_shared_layout tiler): SBO steps one 8-row atom group, LBO steps
        # one whole atom along the tiled axis.
        sdo = 8 * tile.atom_cols // tile.elem_per_16b
        ldo = tile.rows * tile.atom_cols // tile.elem_per_16b
        # ...and _atom_off returns **0** when the dim it tiles has extent 1.
        # That rule governs *both* offsets, because both are strides to a
        # neighbour that a degenerate tile does not have:
        #
        #   LBO steps to the next atom along the row -- absent when the tile is
        #   one atom wide (fp8 + SW128B at BLK_K=128; also any 64-column f16
        #   tile, which GDN allocates).
        #   SBO steps to the next 8-row atom group -- absent when the tile is
        #   only one group tall (rows == 8).
        #
        # Emitting the non-degenerate formula in either case disagrees with the
        # canonical encoder. Benign so far -- the MMA does not read an offset
        # whose neighbour does not exist, measured bit-identical both ways --
        # but these are exactly the cases the measurement below cannot see (it
        # needs a second atom or a ninth row to measure against), so they have
        # to be right by derivation instead of by check.
        one_atom_wide = tile.cols <= tile.atom_cols  # LBO's tiled dim, extent 1
        one_group_tall = tile.rows <= 8  # SBO's tiled dim, extent 1
        if major == "k":
            if one_atom_wide:
                ldo = 0
            if one_group_tall:
                sdo = 0

        # A ring stage is a constant translation of the same tile, so every
        # number below -- SBO, LBO and the k-tile offsets, which are all
        # *differences* from the base -- is the same for every stage. Deriving
        # them on stage 0 is what lets the view's stage be the runtime
        # PipelineState index a pipelined kernel actually holds; only the
        # encoded start address uses it.
        geo = 0
        if major == "k":
            per_atom = tile.atom_cols // mma_k
            if per_atom < 1:
                raise ValueError(
                    f"mma_k={mma_k} exceeds the {tile.atom_cols}-element swizzle atom "
                    f"of {tile.swizzle.name}"
                )
            extent = tile.cols
            base_row, base_col = 0, k0

            def step_coord(kp):
                return tile._coord(geo, 0, k0 + kp * mma_k)

            linear = one_atom_wide
            why = (
                f"a K-major walk over {tile.cols} columns crosses the "
                f"{tile.atom_cols}-element swizzle atom, so the descriptor offset "
                f"jumps by {ldo} instead of {mma_k // tile.elem_per_16b} every "
                f"{per_atom} k-tiles; index the k-tile with a Python int"
            )
        else:
            extent = tile.rows
            base_row, base_col = k0, 0

            def step_coord(kp):
                return tile._coord(geo, k0 + kp * mma_k, 0)

            linear = True
            why = ""

        if not 0 <= k0 < extent:
            raise ValueError(f"k origin {k0} is outside this tile's {extent}-element k axis")
        n_steps = (extent - k0) // mma_k
        base = tile._phys(tile._coord(geo, base_row, base_col))

        def offset_of(kp):
            return (tile._phys(step_coord(kp)) - base) // tile.elem_per_16b

        # Measure the closed forms against the layout that was actually
        # allocated rather than trusting them: SBO is the stride of one 8-row
        # atom group, LBO the stride of one whole atom along the tiled axis.
        # Each is measurable only when the neighbour it steps to exists -- a
        # ninth row for SBO, a second atom for LBO. The degenerate tiles fall
        # to the `elif`s, which are **regression guards, not verification**:
        # they restate the rule the derivation above just applied, so they
        # cannot catch an error in it. What they catch is a future edit that
        # reintroduces the non-degenerate formula here without noticing the
        # degenerate case -- which is how this bug arrived the first time.
        origin = tile._phys(tile._coord(geo, 0, 0))
        if tile.rows > 8:
            measured_sdo = (tile._phys(tile._coord(geo, 8, 0)) - origin) // tile.elem_per_16b
            if measured_sdo != sdo:
                raise ValueError(
                    f"derived SBO {sdo} disagrees with the allocated layout's 8-row "
                    f"stride {measured_sdo}"
                )
        elif major == "k" and sdo != 0:
            raise ValueError(
                f"this tile is {tile.rows} rows, one 8-row atom group, so its SBO "
                f"steps to a group that does not exist and must be 0; derived {sdo}"
            )
        if tile.cols > tile.atom_cols:
            measured_ldo = (
                tile._phys(tile._coord(geo, 0, tile.atom_cols)) - origin
            ) // tile.elem_per_16b
            if measured_ldo != ldo:
                raise ValueError(
                    f"derived LBO {ldo} disagrees with the allocated layout's atom "
                    f"stride {measured_ldo}"
                )
        elif major == "k" and ldo != 0:
            raise ValueError(
                f"this tile is one {tile.atom_cols}-element swizzle atom wide, so its "
                f"LBO steps to an atom that does not exist and must be 0; derived {ldo}"
            )
        # Every k-tile's offset must be a whole number of 16B units, or no
        # descriptor can name it.
        for kp in range(n_steps):
            delta = tile._phys(step_coord(kp)) - base
            if delta % tile.elem_per_16b != 0:
                raise ValueError(
                    f"k-tile {kp} of this tile starts at element offset {delta}, which "
                    f"is not 16B-aligned; a matrix descriptor cannot name it"
                )
        if linear and n_steps > 1:
            base_step = offset_of(1)
            if any(offset_of(kp) != kp * base_step for kp in range(n_steps)):
                linear = False
                why = "the measured k-tile offsets of this tile are not linear in kp"
        desc = T.alloc_local([1], "uint64")
        T.evaluate(
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc[0]),
                self.ptr_to(base_row, base_col),
                ldo=ldo,
                sdo=sdo,
                swizzle=self.swizzle.value,
            )
        )
        return KDesc(desc, KStep(offset_of, max(n_steps, 1), linear, why), major), offset_of

    def __repr__(self):
        return f"<K.tile.view stage={self.stage!r} of {self.tile!r}>"


class SmemPool:
    """Session-bound wrapper over the in-tree :class:`SMEMPool`."""

    def __init__(self, session):
        self.session = session
        self.pool = SMEMPool()
        self._committed = False

    def alloc(self, shape, dtype="float32", swizzle=None, align=0):
        """Allocate a tile.

        ``swizzle=None`` (or ``SWIZZLE_NONE``) returns the raw tirx buffer —
        scalar indexing and ``T.address_of`` work as usual.

        Any other swizzle mode returns a staged :class:`KTile` (index it,
        ``tile[stage]``, to get the :class:`KTileView` that names addresses)
        for a 3-D shape, or a :class:`KTileView` directly for a 2-D one.
        """
        if swizzle is None or swizzle == SwizzleMode.SWIZZLE_NONE:
            return self.pool.alloc(tuple(shape), dtype, align=align)
        if isinstance(swizzle, int):
            swizzle = SwizzleMode(swizzle)
        if len(shape) < 2:
            raise ValueError(f"swizzled alloc shape={tuple(shape)} must have at least two dims")
        _validate_mma_alloc_shape(list(shape), dtype, swizzle)
        layout = mma_shared_layout(dtype, swizzle, list(shape))
        buf = self.pool.alloc(tuple(shape), dtype, align=align or 1024, layout=layout)
        if len(shape) > 3:
            return buf
        tile = KTile(buf, shape, dtype, swizzle)
        # A staged tile hands out views via tile[stage]; an unstaged one has
        # nothing to hand out, so it *is* the view.
        return tile if tile.stages is not None else KTileView(tile)

    def commit(self, size=None):
        """Emit the dynamic-smem size annotation. Called for you at trace end."""
        if self._committed:
            return
        self._committed = True
        self.pool.commit(size)

    @property
    def bytes(self):
        """High-water mark of the pool so far."""
        return self.pool.max_offset


def smem_pool():
    """The kernel's shared-memory pool. One per kernel; committed at trace end."""
    session = _entry.current()
    if session.pool is not None:
        raise RuntimeError("K.smem_pool() was already called for this kernel")
    session.pool = SmemPool(session)
    return session.pool
