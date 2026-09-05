# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""The ``K.idioms`` multi-instruction shapes a PTX-level kernel keeps
re-deriving, written once.

These are **not** wrappers that rename instructions. ``K.ptx`` already spells
every instruction the way the ISA does, and nothing here hides that: each
function below expands to a sequence of bare ``K.ptx`` / ``K.cuda`` calls that
its docstring writes out in full, so a reader can predict the emitted
instructions 1:1 without opening this file.

**Structure earns an API; spelling does not.** The bar this module was pruned
to: a function belongs here only when the *structure* is what is hard — a walk
derived from a tile's layout rather than from any value in a register
(:func:`mma_chain`), a warp-collective whose placement relative to a guard is
the difference between working and deadlocking (:func:`warp_scan_add`). A
sequence that is merely *worth knowing how to spell* is not an idiom: that
knowledge belongs in a doc note and the spelling belongs at the call site.
Several functions were removed on exactly that test — packed-f32x2 multiply
chains, f16x2 packing, division by a constant — and live as closures in the one
kernel that uses them, with their evidence in the comment beside them. What
survived is the short list above.

The one value-level operation here, :func:`sigmoid_tanh_approx_f32`, earns its
place by naming an explicit **approximate numerical contract** and
materialization boundary, not by renaming ``tanh``. Its result is deliberately
written before it is returned so a caller's scale cannot reassociate through
the sigmoid and silently select a different instruction schedule.

The one shape that turned out to need neither a wrapper nor a doc note is
``cp.async``: its three spellings differ in what a *skipped* lane's destination
holds, so instead of a wrapper the ``K.ptx`` proxy makes the distinguishing
operand mandatory and explains the choice when it is missing. See
``_CP_ASYNC_HELP`` in this package's ``__init__``.

Every "why this form" note records its measured evidence. Where the frozen GDN
kernel and a simpler spelling disagree, the note says which won and on what
instrument, including the f16x2 widening case corrected by
:func:`cast_f16x2_to_f32x2`.

Style note: like ``entry.py`` and ``smem.py`` this module talks to ``tirx``
directly (``from tvm.script import tirx as T``) and wraps void instruction
calls in ``T.evaluate`` by hand. The ``K.ptx`` / ``K.cuda`` statement proxies
live in the package ``__init__``, which imports *this* module; going through
them here would be a cycle, and ``T.evaluate(T.ptx...)`` is exactly what the
proxy does anyway.
"""

from __future__ import annotations

from tvm.backend.cuda.cpp.descriptors import _INSTR_DESC_FORMAT_MAP, _TCGEN05_MMA_K
from tvm.backend.cuda.ptx.table import TABLE as _PTX_TABLE
from tvm.backend.cuda.ptx.table import _tcgen05_mma_mask_lanes
from tvm.script import tirx as T
from tvm.tirx.script.builder import ir as _I

from .smem import KTileView

FULL_MASK = 0xFFFFFFFF


def _dtype_of(expr):
    """The dtype string of a traced expression."""
    return str(expr.ty.dtype)


# ---------------------------------------------------------------------------
# tcgen05 MMA k-loops
# ---------------------------------------------------------------------------

# Bit layout of the **dense** tcgen05 instruction descriptor, mirroring
# ``encode_instr_descriptor_dense_uint32`` in
# ``python/tvm/backend/cuda/cpp/descriptors.py`` (itself a port of
# ``InstrDescriptor`` in the codegen header). Read off that function's
# ``desc |= (field & mask) << shift`` chain, not hand-derived from the ISA doc,
# so the two cannot drift silently -- and every decode re-packs and checks
# itself, so a descriptor carrying a bit this table does not model is rejected
# rather than half-understood.
#
# Two of the header union's named fields are deliberately absent: ``sparse_id2``
# (bits 0-1) and ``max_shift`` (bits 30-31, .ws-only). The in-tree encoder
# cannot produce either, so a descriptor carrying one came from somewhere else
# and gets refused rather than decoded with a field this module's callers
# would then ignore.
_IDESC_FIELDS_DENSE = (
    ("is_sparse", 2, 0x1),
    ("sat_d", 3, 0x1),
    ("d_format", 4, 0x3),
    ("a_format", 7, 0x7),
    ("b_format", 10, 0x7),
    ("neg_a", 13, 0x1),
    ("neg_b", 14, 0x1),
    ("trans_a", 15, 0x1),
    ("trans_b", 16, 0x1),
    ("n_over_8", 17, 0x3F),
    ("m_over_16", 24, 0x1F),
)

# Bit layout of the **block-scaled** descriptor, mirroring
# ``union InstrDescriptorBlockScaled`` in
# ``python/tvm/backend/cuda/intrinsics/header.py`` (the ``instr_descriptor_
# block_scaled`` header tag). Transcribed from that union's field widths and
# bit comments; there is no in-tree encoder for this family to read off, so the
# header *is* the source. Every named field of the union appears here -- the
# only unmodelled bits are its three unnamed pad slots (3, 6, 31).
#
# This is NOT an extension of the dense layout. Four slots are reassigned:
# bit 3 (dense ``sat_d``) is pad, bits 4-5 (dense ``d_format``) are ``b_sf_id``,
# bit 23 (dense pad) is ``scale_format``, and bits 29-30 -- the low half of
# dense ``max_shift`` -- are ``a_sf_id``.
_IDESC_FIELDS_BLOCK_SCALED = (
    ("sparse_id2", 0, 0x3),  # bit [ 0, 2) : sparse meta data id2
    ("is_sparse", 2, 0x1),  # bit [ 2, 3) : 0 = dense, 1 = sparse
    ("b_sf_id", 4, 0x3),  # bit [ 4, 6) : matrix B scale factor ID
    ("a_format", 7, 0x7),  # bit [ 7,10) : A operand format
    ("b_format", 10, 0x7),  # bit [10,13) : B operand format
    ("neg_a", 13, 0x1),  # bit [13,14) : negate A
    ("neg_b", 14, 0x1),  # bit [14,15) : negate B
    ("trans_a", 15, 0x1),  # bit [15,16) : 0 = K-major, 1 = MN-major
    ("trans_b", 16, 0x1),  # bit [16,17) : 0 = K-major, 1 = MN-major
    ("n_over_8", 17, 0x3F),  # bit [17,23) : N, 3 LSBs not included
    ("scale_format", 23, 0x1),  # bit [23,24) : 0 = E4M3, 1 = E8M0
    ("m_over_16", 24, 0x1F),  # bit [24,29) : M, 4 LSBs not included
    ("a_sf_id", 29, 0x3),  # bit [29,31) : matrix A scale factor ID
)


def decode_instr_descriptor(idesc, *, block_scaled=False):
    """Unpack a tcgen05 instruction descriptor into its named fields.

    Returns a dict with ``trans_a`` / ``trans_b`` / ``M`` / ``N`` /
    ``a_format`` / ``b_format`` and the rest of the selected layout --
    :data:`_IDESC_FIELDS_DENSE` (the default) or
    :data:`_IDESC_FIELDS_BLOCK_SCALED` when ``block_scaled=True``.

    Raises if the value has any bit set that the selected table does not model:
    the descriptor is the one operand of a ``tcgen05.mma`` with no instruction
    to build it and no runtime check on it, so guessing at an unrecognised
    encoding is worse than refusing.

    Two layouts, and the caller must say which
    ---------------------------------------------
    The hardware has **two** 32-bit descriptor unions, and they are not
    extensions of each other -- the block-scaled one reassigns four of the
    dense one's slots (``sat_d`` -> pad, ``d_format`` -> ``b_sf_id``, pad ->
    ``scale_format``, the low half of ``max_shift`` -> ``a_sf_id``). They are
    **bit-indistinguishable**: no value, and no bit within a value, says which
    union built it. ``0x082016A0`` is a well-formed member of both, meaning
    ``b_sf_id=2`` under one and ``d_format=2`` under the other.

    So the discriminator has to come from the caller, which knows because it
    knows which family of MMA it is building for -- the block-scaled unions go
    with ``kind::mxf8f6f4`` / ``kind::mxf4`` / ``kind::mxf4nvf4``, i.e. the
    ``.block_scale`` instructions. Defaulting to dense and hoping is exactly
    the failure this parameter exists to prevent: without it, a block-scaled
    descriptor's ``b_sf_id`` decodes *silently* as ``d_format``, a foreign
    field landing inside a modelled one, which the re-pack check cannot catch
    because every bit is accounted for.

    Checked against the three constants the frozen GDN kernel hardcodes
    (dense)::

        0x04100010  M=64  N=64   trans_a=0 trans_b=0   (KK / QK)
        0x08100010  M=128 N=64   trans_a=0 trans_b=0   (the ts-form chains)
        0x08210010  M=128 N=128  trans_a=0 trans_b=1   (the K^T state update)

    which is how ``ID_KV``'s "B col-major" stops being folklore in a comment
    and becomes a bit this code reads.
    """
    if isinstance(idesc, bool) or not isinstance(idesc, int):
        raise TypeError(
            f"idesc must be a trace-time Python int, got {idesc!r}. The transpose "
            "bits are decoded from it at trace time to pick each operand's "
            "descriptor orientation, which a runtime expression cannot answer."
        )
    if not 0 <= idesc <= 0xFFFFFFFF:
        raise ValueError(f"idesc must fit in a uint32, got {idesc!r}")
    layout = _IDESC_FIELDS_BLOCK_SCALED if block_scaled else _IDESC_FIELDS_DENSE
    fields = {name: (idesc >> shift) & mask for name, shift, mask in layout}
    repacked = 0
    for name, shift, mask in layout:
        repacked |= (fields[name] & mask) << shift
    if repacked != idesc:
        which = "block-scaled" if block_scaled else "dense"
        other = "" if block_scaled else " (a block-scaled descriptor? pass block_scaled=True)"
        raise ValueError(
            f"instruction descriptor 0x{idesc:08X} has bits set outside the known "
            f"fields of the {which} layout (re-packing the decoded fields gives "
            f"0x{repacked:08X}){other}"
        )
    fields["M"] = fields.pop("m_over_16") << 4
    fields["N"] = fields.pop("n_over_8") << 3
    return fields


# Operand element widths a ``kind::`` token pins down, used only as a
# cross-check on the tiles a chain is handed. The mixed-format kinds
# (``f8f6f4``, ``mxf8f6f4``) are absent on purpose: they admit 8-, 6- and
# 4-bit operands under one K, which is precisely why the width cannot drive
# the k-step. ``tf32`` operands live in 32-bit containers.
_KIND_ELEM_BITS = {
    "f16": 16,  # f16 and bf16
    "tf32": 32,
    "i8": 8,
    "mxf4": 4,  # e2m1 only; block-scaled, so unreachable from mma_chain today
    "mxf4nvf4": 4,
}


def _mma_kind(mma):
    """The ``kind::<name>`` token of a tcgen05 MMA mnemonic, e.g. ``"f16"``."""
    tokens = [tok for tok in str(mma).split(".") if tok.startswith("kind::")]
    if len(tokens) != 1:
        raise ValueError(
            f"mma mnemonic {mma!r} carries {len(tokens)} 'kind::' tokens, expected "
            "exactly one. The k-step of a tcgen05 MMA is a property of its kind "
            "(e.g. 'tcgen05.mma.cta_group::1.kind::f16'), so a mnemonic without "
            "one cannot drive a chain."
        )
    return tokens[0][len("kind::") :]


def _mma_k_from_mnemonic(mma):
    """The contraction step, in **elements**, of one MMA of this mnemonic.

    Straight out of ``_TCGEN05_MMA_K`` (imported, not copied), keyed by the
    mnemonic's ``kind::`` token -- which is the only thing that determines it.
    In particular it is *not* ``256 // element_bits``: that identity holds for
    ``f16`` (16), ``tf32`` (8), ``f8f6f4``/``i8`` (32) and coincidentally for
    ``mxf4`` (64), and breaks for every sub-byte operand of a mixed-format
    kind -- an fp6 operand of ``kind::f8f6f4`` would give 42, and an fp4 one 64,
    where the hardware K is 32 either way.

    The kind cannot be recovered from the descriptor: ``_INSTR_DESC_FORMAT_MAP``
    collides across kinds (``a_format=0`` is F16 under ``kind::f16`` and E4M3
    under ``kind::mxf8f6f4``), which is why this reads the mnemonic the caller
    passed rather than the bits it also passed.
    """
    kind = _mma_kind(mma)
    k_pair = _TCGEN05_MMA_K.get(kind)
    if k_pair is None:
        raise ValueError(
            f"mma mnemonic {mma!r} has kind::{kind}, which is not a tcgen05 MMA kind "
            f"(known: {', '.join(sorted(_TCGEN05_MMA_K))}); its k-step is unknown"
        )
    k_dense, _k_sparse = k_pair
    return k_dense


def _mma_cta_group(mma):
    """The ``cta_group::<n>`` token of a tcgen05 MMA mnemonic.

    Same shape as :func:`_mma_kind`, and for the same reason: the token is a
    required modifier of every ``tcgen05.mma`` entry in the table, so a mnemonic
    carrying none — or two — is malformed rather than defaultable.
    """
    tokens = [tok for tok in str(mma).split(".") if tok.startswith("cta_group::")]
    if len(tokens) != 1:
        raise ValueError(
            f"mma mnemonic {mma!r} carries {len(tokens)} 'cta_group::' tokens, expected "
            "exactly one. The disable-output-lane vector's width is a property of the "
            "cta_group (e.g. 'tcgen05.mma.cta_group::1.kind::f16'), so a mnemonic "
            "without one cannot drive a chain."
        )
    return tokens[0]


def _dol_lanes(mma):
    """How many ``disable_output_lane`` masks this mnemonic's MMA takes.

    ``cta_group::1`` takes 4 and ``cta_group::2`` takes 8 (ISA 9.7.17.10.9.1):
    the accumulator of a 2-CTA MMA spans both CTAs of the pair, so the mask
    vector is twice as wide.

    The count is **derived, never passed**. It is fixed by a token the caller
    already spells in ``mma``, so a ``dol_count=`` argument would be a second
    place to say the same thing and therefore a second place to say it wrong.
    The rule itself is imported from the instruction table's own
    ``_tcgen05_mma_mask_lanes`` — the same function the table uses to size the
    operand slot — rather than transcribed here, so the two cannot drift; only
    the token extraction is local, and ``test_idioms_dol_lane_count_matches_the_
    table`` pins that against the table function directly.

    Orthogonal to ``kind``: ``cta_group`` and ``kind`` are independent modifier
    slots, and the lane count reads only the former.
    """
    return _tcgen05_mma_mask_lanes({"cta_group": _mma_cta_group(mma)})


def _zeros(n):
    """The default ``dol``: *n* zero ``uint32`` **immediates**.

    Deliberately not a zeroed local. Immediates allocate nothing, emit nothing,
    and land in the call as ``(uint)0`` — so there is no per-call-site cost to
    amortise and, more importantly, no local whose scope has to be reasoned
    about when chains appear in several different role blocks.
    """
    return (T.uint32(0),) * n


# The kinds that exist ONLY on the block-scaled entries, read off the table
# rather than transcribed, so a kind added there is refused here automatically.
_BLOCK_SCALED_KINDS = frozenset(
    choice[len("kind::") :]
    for slot in _PTX_TABLE["tcgen05_mma_block_scale_ss"].slots
    if getattr(slot, "name", None) == "kind"
    for choice in slot.choices
)


def mma_chain(mma, d, *, a, b, idesc, pred, accumulate, guard, dol=None, k_range=None):
    """One tcgen05 MMA k-loop: exactly ``n_k`` ``K.ptx[mma](...)`` calls.

    Expansion. Each operand's descriptor is encoded **once**, at the start of
    ``k_range``; every k-phase is then that descriptor plus a trace-time
    constant 16-byte offset, i.e. one IADD on its low half (and phase 0 is the
    base itself, no IADD)::

        a_desc = a.mma_desc(major=<from idesc trans_a>)   # once, if a is a view
        b_desc = b.mma_desc(major=<from idesc trans_b>)   # once
        # guard="branch": ONE `If(pred != 0)` opens HERE -- between the
        # encodes above and the MMAs below -- and no per-MMA predicate exists
        for kp in range(n_k):
            K.ptx[mma](
                K.Cast("uint32", d),           # accumulator, tmem
                a_desc + off16_a(kp),          # or Cast(u32, a + kp*a_step) for tmem A
                b_desc + off16_b(kp),
                K.uint32(idesc),
                *dol,                          # 4 masks, or 8 under cta_group::2
                K.ptx.pred(accumulate if kp == 0 else 1),   # phase 0: the flag
                pred=pred,   # guard="pred" only; omitted entirely when pred=None
            )

    ``n_k = (hi - lo) // mma_k``, and ``mma_k`` is read off the mnemonic's
    ``kind::`` token (see :func:`_mma_k_from_mnemonic`). The
    ``disable_output_lane`` vector's width is read off the same mnemonic's
    ``cta_group::`` token (see :func:`_dol_lanes`) — 4 for ``cta_group::1``, 8
    for ``cta_group::2``, orthogonal to the kind. Nothing else is emitted — in
    particular this function never emits a warp collective (see ``pred``).

    Out of scope: the other two operand shapes
    ------------------------------------------
    ``tcgen05.mma`` is not one instruction with optional extras. It is **three
    operand shapes**, and this function emits exactly one of them::

        dense non-ws   d, a, b, idesc, disable_output_lane x4 (x8), enable_input_d
        block-scaled   d, a, b, idesc, sfa_tmem, sfb_tmem,         enable_input_d
        .ws            d, a, b, idesc, enable_input_d, zero_col_mask

    The loop below writes the first line — nine operands under
    ``cta_group::1``, thirteen under ``cta_group::2``, the difference being the
    mask vector's width alone. A ``.block_scale`` or ``.ws`` mnemonic is
    **rejected** rather than handled: reaching the emission it would build a
    call whose operands do not match the instruction it names. That is a
    structural limit, not a missing feature, and the two are out of scope for
    different reasons. Block-scaled differs structurally — it carries the second
    descriptor bit union (see :func:`decode_instr_descriptor`), whose per-phase
    scale-factor ids are patched at runtime rather than being the trace-time
    constant this function decodes. ``.ws`` differs only in the operand list,
    but differs there by more than an append: its six operands are the dense
    nine with the ``disable_output_lane`` masks **removed**, ``enable_input_d``
    moved out of the tail, and ``zero_col_mask`` appended — and it is the one
    form for which the dense union's ``max_shift`` field means anything.

    Both are kernel-local closures in the kernels that need them today. A chain
    form for either is a separate proposal, to be argued on its own evidence
    through the idiom reviewer rather than bolted onto this function —
    "structure earns an API" applies to each shape on its own.

    Parameters
    ----------
    mma : str
        The **real table spelling**, passed through verbatim — e.g.
        ``"tcgen05.mma.cta_group::1.kind::f16"``. This function never builds,
        completes or alters a mnemonic. It is *read*, though: the ``kind::``
        token is what fixes ``mma_k``, since the descriptor's format bits
        cannot say which kind they belong to.
    d : expr
        Accumulator address in **tmem**. That is the only form: a tcgen05
        accumulator lives in tmem.
    a : KTileView | expr
        A shared-memory stage view (``tile[stage]``), or a raw **tmem** address
        for the ts-form. The ts a-step is derived, not passed: one k-tile is
        ``mma_k * elem_bits // 32`` tmem columns (8 for f16).
    b : KTileView
        A shared-memory stage view. B has no tmem form.
    idesc : int
        The instruction descriptor, a trace-time Python int. Its transpose bits
        are **decoded** (see :func:`decode_instr_descriptor`) and choose each
        view's descriptor orientation, so there are no ``trans_`` arguments to
        disagree with it. ``trans_a`` set with a tmem ``a`` is rejected: no such
        hardware form.
    pred : expr | None
        Issue predicate — **required and keyword-only**, with two legal modes
        matching the ISA's two legal issue forms (one elected lane, or all 32
        lanes of the issuer warp — orig:L443, port notes §3.4):

        * an expression — every call carries ``@p`` on that local;
        * ``None`` — **no predicate operand is emitted at all**. Correctness
          then rests on the call site being either single-lane under the
          caller's own elect *branch* (the original kernels' form, and the
          bit-identical port of it) or fully convergent. ``None`` must be
          typed at every call site: presence or absence of the ``@p`` operand
          is a semantic fork and is never reachable by default.

        Either way this function emits **no warp collective**. That — not a
        mandatory ``@p`` — is what the G3 rationale requires: materialising an
        ``elect_sync`` *here* would put a collective wherever this function is
        called, including inside a guard, where the excluded lanes never reach
        it and the CTA deadlocks with no diagnostic (port notes §4 G3). The
        caller elects once, at a point it knows is convergent, and passes the
        local — or guards with the branch and passes ``None``.
    accumulate : bool | expr
        **Required.** ``False`` — phase 0 overwrites the accumulator (its
        ``enable_input_d`` is 0) and later phases accumulate. ``True`` — every
        phase accumulates, i.e. ``+=`` into what is already there. No default:
        it is a real per-chain semantic choice and defaulting it silently
        produces wrong numbers rather than an error.

        A **traced expression** is the common case, not the exception: of the
        in-tree chains, FA4's PV (``should_accumulate``), FA-bwd's dSdK and
        dV, and fp8_blockwise's main chain all gate phase 0 on a runtime
        flag, semantically ``(kp != 0) OR flag`` in each (FA-bwd spells it by
        parameterizing the whole operand); GDN is the outlier for knowing the
        value at trace time. Pass the **flag alone** — phase 0
        takes it and phases 1+ take the constant 1, which *is* that
        disjunction, built structurally. A caller-spelled
        ``(kp != 0) or flag`` cannot be detected here and would double the
        condition. A Python ``bool`` emits byte-identical IR to the narrow
        form (pinned by test).
    guard : str
        **Required.** How the chain is issued — ``"pred"`` or ``"branch"``,
        the ISA's two legal issue forms. ``"pred"``: every MMA carries ``@p``
        on *pred* (or, with ``pred=None``, no predicate operand at all).
        ``"branch"``: the descriptors are encoded first, then **one**
        ``If(pred != 0)`` wraps the whole MMA loop, with no per-MMA predicate.
        The distinction is WHERE the control-flow edge sits relative to this
        function's own descriptor encodes — and that placement is producible
        only here: a caller's own ``If`` around the call puts the edge ABOVE
        the encodes, which measured with the losing cell (1.22 on FA4 s4096),
        not the winning one.

        Measured (FA4's QK chain, every arm hand-written and bit-identical to
        the frozen kernel before timing)::

                                 s4096_h32kv32          s8192_h32kv32
                                 branch     pred        branch     pred
            hoisted encode       1.0000   0.9942        1.0000   1.0097
            per-call encode      0.9991   1.2687        1.0124   1.2560

        Either factor alone stays within ±1.3%; per-call encode TOGETHER with
        per-MMA ``@p`` is **+25%** — same MMA count, same registers, zero
        spills, and the fast arm's SASS is 24 instructions *larger*. The
        mechanism is a scheduling effect, and the following account is the
        **current best, not settled**: the ``@p`` form leaves the chain's
        async MMAs in straight-line code with their 64-bit descriptor
        operands hanging off freshly computed values; the branch edge between
        encode and issue is what lets them land first. A live competitor
        attributes the effect to descriptor **lo-uniformity** instead: one
        kernel (flashkda) occupies the same structural cell at parity, and
        the variable that differs is its hoisted warp-uniform broadcast of
        each descriptor's low word — the on-record discriminating prediction
        is that removing that broadcast reproduces the interaction. The
        REQUIREMENT does not depend on which account wins: per-acquisition
        uniformity measures +2.4-5.7% (so this function cannot buy it
        cheaply), and the edge between its own encode and its own MMAs is
        producible only here.

        There is **no default** because the same structural cell disagrees
        across kernels: GDN sits in per-call+``@p`` and measures 0.975 against
        a frozen reference that itself predicates all 60 sites — ``"pred"``
        is what keeps it faithful — while FA4-shaped chains need
        ``"branch"``. Two disagreeing kernels do not establish a default; if
        a third and fourth land on one value, re-open that question.

        ``guard="branch"`` requires ``pred`` to be an expression —
        ``pred=None`` under ``"branch"`` raises (there would be nothing to
        branch on). Neither mode emits a warp collective.
    dol : tuple, optional
        The ``disable_output_lane`` masks. Defaults to that many zero u32
        immediates (no lane disabled). The count is **not** a parameter: it is
        4 under ``cta_group::1`` and 8 under ``cta_group::2``, derived from the
        mnemonic's own token (:func:`_dol_lanes`), and a tuple of the other
        length is rejected. Passing the width separately would be a second
        place to state something ``mma`` already states.
    k_range : (int, int), optional
        The contraction range in **element** units, ``[lo, hi)``. Defaults to
        B's whole k axis — its columns when B is k-major, its rows when B is
        mn-major — which is what every chain in the GDN kernel walks.
        ``hi - lo`` must be a whole number of ``mma_k``.

    Why views and not a descriptor
    ------------------------------
    An encoded descriptor is a **runtime** ``uint64`` whose fields are not
    readable at trace time, and the per-phase walk is not a property of that
    value — it is a property of the tile's *layout*. For a 2-atom-wide f16 tile
    the k-tile offsets run ``0, 2, 4, 6, 512, 514, 516, 518``: four steps
    inside one 128-byte swizzle atom and then a jump of a whole atom
    (``rows * atom_cols / 8``). Deriving that needs the rows, the atom width
    and the element size — the view's metadata. So a bare ``uint64`` cannot
    drive a chain, and any object that could would be carrying the view's
    metadata under another name. Passing the view says what is actually
    required.

    The degenerate case is real but does not earn a second operand form: a tile
    exactly one atom wide never leaves the atom, so its walk *is* linear and a
    plain stride would do. Adding an operand form for it would mean the same
    call site is correct or silently wrong depending on the tile's width, which
    is precisely the bug the typed step exists to prevent.

    Passing the *same view object* as both ``a`` and ``b`` (an ``A @ A^T``
    chain) encodes it once — an object-identity test, never a structural one.

    One consequence: two chains over the same stage encode that stage's
    descriptor twice (the frozen kernel shares one encode between its KK and QK
    chains). That is a handful of integer ops per chunk on a single-warp
    issuer role that is latency-bound on the matrix engine, and it measured
    inside noise.

    Scope
    -----
    This function encodes each operand's descriptor **per call**, because a
    view carries its stage in the base address. That per-call encode is **not
    itself a cost**: measured on FA4, hoisted-versus-per-call encode is free
    on its own (0.9991-1.0124). What costs is its *interaction with the issue
    guard* - see ``guard``, whose entry carries the 2x2. That interaction is
    why ``guard`` is required.

    What a chain function still does not model is the rest of a kernel's
    descriptor *lifecycle*: FA4 pre-encodes eight kernel-scope descriptors,
    including specialization-branch variants chosen at trace time, and
    broadcasts each one's low word warp-uniform — which is free at kernel
    scope and measures +2.4% to +5.7% worse per acquisition. A kernel whose
    descriptors have that shape keeps its own chain for the *lifecycle*, not
    for the encode count (idiom-review ruling (a): no second chain member).
    """
    if guard not in ("pred", "branch"):
        raise ValueError(
            f"guard must be 'pred' or 'branch', got {guard!r}. The two spell the "
            "ISA's two issue forms, and their difference is a measured +25% "
            "interaction (see the guard entry); there is no default because the "
            "same cell is catastrophic in one kernel and free in another."
        )
    if guard == "branch" and pred is None:
        raise ValueError(
            "guard='branch' emits `If(pred != 0)` around the MMA loop, so pred "
            "must be an expression; pred=None (no predicate operand, caller "
            "manages convergence) is a guard='pred' mode."
        )
    # tcgen05.mma is three operand shapes; the loop below writes one of them.
    # The other two are refused here rather than emitted as a call whose
    # operands do not match the instruction it names.
    tokens = str(mma).split(".")
    for token, shape, instead in (
        ("block_scale", "block-scaled", "sfa/sfb tmem addresses"),
        ("ws", "weight-stationary (.ws)", "a zero-column-mask descriptor"),
    ):
        if token in tokens:
            raise ValueError(
                f"mma_chain emits only the dense non-ws operand shape, and {mma!r} is "
                f"the {shape} one: its MMA takes {instead} where the dense one takes "
                "disable_output_lane. Spell that chain at the call site -- a chain "
                "form for it is a separate proposal, not an argument to this one."
            )
    # A block-scaled KIND names the block-scaled shape whether or not the
    # `.block_scale` token was written. Keying the refusal on the token alone
    # let `kind::mxf8f6f4` bare through to a later, less teaching PTX-table
    # error about operand counts. The kind set comes from the table's own
    # block-scaled entry (_BLOCK_SCALED_KINDS).
    if _mma_kind(mma) in _BLOCK_SCALED_KINDS:
        raise ValueError(
            f"mma_chain emits only the dense non-ws operand shape, and {mma!r} names a "
            f"block-scaled kind (kind::{_mma_kind(mma)} exists only on "
            "tcgen05.mma.block_scale): its MMA takes sfa/sfb tmem addresses where the "
            "dense one takes disable_output_lane, and its descriptor is the other bit "
            "union. Spell that chain at the call site -- a chain form for it is a "
            "separate proposal, not an argument to this one."
        )
    mma_k = _mma_k_from_mnemonic(mma)

    fields = decode_instr_descriptor(idesc)
    trans_a, trans_b = fields["trans_a"], fields["trans_b"]
    if fields["is_sparse"]:
        raise ValueError(
            f"idesc 0x{idesc:08X} sets the sparse bit. A sparse chain walks A at half "
            "B's k-step through a metadata operand this function does not emit, and "
            "'tcgen05.mma.sp' is not in the instruction table."
        )

    if not isinstance(b, KTileView):
        raise TypeError(
            "mma_chain: b must be a KTileView -- index the tile with its stage, "
            f"`tile[stage]`. Got {type(b).__name__}."
        )
    a_is_view = isinstance(a, KTileView)
    if not a_is_view and trans_a:
        raise ValueError(
            f"idesc 0x{idesc:08X} sets trans_a, but a is a tmem address; the "
            "ts-form A operand has no transposed variant"
        )

    bits = b.bits
    if a_is_view and a.bits != bits:
        raise ValueError(
            f"a and b have different element widths ({a.bits} vs {bits} bits). The "
            "mixed-format kinds (f8f6f4, mxf8f6f4) do allow that in hardware -- each "
            "view already walks in its own bytes -- but no in-tree chain exercises it, "
            "so it is refused rather than emitted untested."
        )
    # The kind fixes the k-step; the tile only has to be consistent with it.
    want_bits = _KIND_ELEM_BITS.get(_mma_kind(mma))
    if want_bits is not None and want_bits != bits:
        raise ValueError(
            f"{mma!r} takes {want_bits}-bit operands, but the b tile is {b.dtype} "
            f"({bits} bits). mma_k comes from the kind ({mma_k}), so a mismatched "
            "tile would walk the wrong k-step rather than fail."
        )
    a_major = "mn" if trans_a else "k"
    b_major = "mn" if trans_b else "k"

    # The descriptor names its operand dtypes too; check the tiles agree with
    # what the caller's hardcoded constant claims.
    for label, operand in (("b", b), ("a", a if a_is_view else None)):
        if operand is None:
            continue
        want = _INSTR_DESC_FORMAT_MAP.get(operand.dtype)
        got = fields[f"{label}_format"]
        if want is not None and want != got:
            raise ValueError(
                f"idesc 0x{idesc:08X} declares {label}_format={got}, but the {label} "
                f"tile is {operand.dtype} (format {want})"
            )

    b_extent = b.cols if b_major == "k" else b.rows
    lo, hi = (0, b_extent) if k_range is None else tuple(k_range)
    if not 0 <= lo < hi <= b_extent:
        raise ValueError(
            f"k_range=({lo}, {hi}) is outside b's {b_extent}-element "
            f"{'column' if b_major == 'k' else 'row'} axis"
        )
    if (hi - lo) % mma_k:
        raise ValueError(
            f"k_range spans {hi - lo} elements, which is not a whole number of "
            f"mma_k={mma_k} (kind::{_mma_kind(mma)})"
        )
    n_k = (hi - lo) // mma_k

    b_desc, b_off = b.encode(major=b_major, mma_k=mma_k, k0=lo)
    if a_is_view:
        a_extent = a.cols if a_major == "k" else a.rows
        if hi > a_extent:
            raise ValueError(f"k_range=({lo}, {hi}) exceeds a's {a_extent}-element k axis")
        if a is b and a_major == b_major:
            # The same view object on both sides (A @ A^T) with the same
            # orientation is the same descriptor: encode it once. This is an
            # *identity* test, not a structural one -- two views of the same
            # tile built from a mutable PipelineState var can be structurally
            # equal while denoting different stages, which is exactly why
            # `mma_desc` refuses to dedup on structure.
            a_desc, a_off = b_desc, b_off
        else:
            a_desc, a_off = a.encode(major=a_major, mma_k=mma_k, k0=lo)
    else:
        # One k-tile of a tmem A operand is mma_k elements wide, packed into
        # 32-bit tmem columns.
        a_step = mma_k * bits // 32
        a_base = lo // mma_k

    # Both the default and the check follow the derived width; a count fixed at
    # 4 made every cta_group::2 chain unspellable -- the default failed in the
    # PTX table ("expects 13 operand(s)") and an explicit 8 failed here.
    n_dol = _dol_lanes(mma)
    if dol is None:
        dol = _zeros(n_dol)
    if len(dol) != n_dol:
        raise ValueError(
            f"{mma} takes {n_dol} disable-output-lane masks ({_mma_cta_group(mma)}), got {len(dol)}"
        )

    # Phase 0's enable_input_d is the flag; phases 1+ are the constant 1. That
    # per-phase split IS the disjunction `(kp != 0) OR flag`, built
    # structurally, which is why callers pass the flag alone -- a
    # caller-spelled disjunction cannot be detected here and would double the
    # condition. A Python bool keeps the int() constant path byte-for-byte; a
    # traced expression rides the operand as-is (T.ptx.pred takes either).
    acc0 = int(accumulate) if isinstance(accumulate, bool | int) else accumulate

    def _emit(issue_pred):
        for kp in range(n_k):
            if a_is_view:
                a_op = a_desc + a_off(kp)
            else:
                a_op = T.Cast("uint32", a + (a_base + kp) * a_step)
            T.evaluate(
                T.ptx[mma](
                    T.Cast("uint32", d),
                    a_op,
                    b_desc + b_off(kp),
                    T.uint32(idesc),
                    *dol,
                    T.ptx.pred(acc0 if kp == 0 else 1),
                    pred=issue_pred,
                )
            )

    if guard == "branch":
        # The one edge that earns this mode: it sits BETWEEN the descriptor
        # encodes (already emitted above) and the MMAs. A refactor that lifts
        # this If above the encode calls silently restores the losing cell --
        # the placement is pinned by test.
        with T.If(pred != 0), T.Then():
            _emit(None)
    else:
        _emit(pred)


# ---------------------------------------------------------------------------
# approximate sigmoid
# ---------------------------------------------------------------------------


def sigmoid_tanh_approx_f32(value=None, *, tanh_input=None):
    """Return materialized ``0.5 * tanh.approx(value * 0.5) + 0.5`` in f32.

    Expansion -- two explicit DPS PTX calls and two local f32 values; the
    input half-scale remains an ordinary f32 expression::

        tanh_value = K.local_scalar("float32")
        result = K.local_scalar("float32")
        K.ptx.tanh.approx.f32(tanh_value, value * 0.5)
        K.ptx.fma.rn.f32(result, tanh_value, 0.5, 0.5)

    Pass ``tanh_input=`` instead of ``value`` when the caller already has the
    ``value * 0.5`` operand consumed by ``tanh.approx``. This preserves
    schedules that precompute a shared scale so each sigmoid input remains one
    FMA; exactly one input is required.

    This is an opt-in approximation, not an implementation of an exact math
    ``sigmoid``. The name exposes both the tanh realization and its f32
    precision so a call site states the numerical contract it accepts.

    Why this form
    -------------
    KDA forward v39 replaced ``ex2.approx`` + ``rcp.approx`` with this identity:
    pass 1 fell from about 0.72 us to 0.41 us and paired official runs retained
    a small end-to-end win on B200. The result is materialized on purpose. A
    later form that folded a caller-provided scale through the sigmoid removed
    instructions but lost 0.56% in a controlled A-B-A run, so scaling remains
    at the call site after this function returns.
    """
    if (value is None) == (tanh_input is None):
        raise ValueError("pass exactly one of value or tanh_input")
    if tanh_input is None:
        tanh_input = value * T.float32(0.5)
    tanh_value = T.alloc_local([1], "float32")
    result = T.alloc_local([1], "float32")
    T.evaluate(T.ptx.tanh.approx.f32(tanh_value[0], tanh_input))
    T.evaluate(T.ptx.fma.rn.f32(result[0], tanh_value[0], T.float32(0.5), T.float32(0.5)))
    return result[0]


# ---------------------------------------------------------------------------
# f16x2 -> f32x2
# ---------------------------------------------------------------------------


def cast_f16x2_to_f32x2(dst, i, word):
    """Widen a packed ``half2`` in *word* to ``dst[2i]``, ``dst[2i+1]`` as f32.

    Expansion — three instructions, which is what ``__half22float2`` compiles
    to::

        u = K.alloc_local([2], "uint16")
        K.ptx.mov.b32(u[0], u[1], word)    # mov.b32 {lo, hi}, word
        K.ptx.cvt.f32.f16(dst[2 * i], u[0])
        K.ptx.cvt.f32.f16(dst[2 * i + 1], u[1])

    Why this form
    -------------
    This is the instruction sequence, spelled as instructions. An earlier
    revision used a chain of pure expressions instead (mask/shift →
    ``reinterpret("float16", ...)`` → widen) on the belief -- recorded in
    ``gdn_port_NOTES.md`` §3.3 -- that ``mov.b32`` + ``cvt.f32.f16`` "does not
    type-check". **That was wrong**, and the note is corrected here: the
    two-destination unpack form of ``mov.b32`` exists in the table and the
    table's ``cvt.f32.f16`` is satisfied by a ``uint16`` source. The earlier
    attempt must have fed it something else.

    The two forms are equivalent all the way down. Measured on this toolchain
    (CUDA 13.2, sm_100a), a kernel doing eight of these compiles to **byte-
    identical SASS** either way -- 46 instructions, same opcode histogram, the
    widening done by 16 ``HADD2.F32`` in both. So this is a spelling choice
    with no code consequence, and the spelling that names the instructions is
    the one that belongs in a PTX-level DSL.

    Destination-passing rather than value-returning because ``mov`` and ``cvt``
    write registers; the caller supplies the two f32 slots.
    """
    halves = T.alloc_local([2], "uint16")
    T.evaluate(T.ptx.mov.b32(halves[0], halves[1], word))
    T.evaluate(T.ptx.cvt.f32.f16(dst[2 * i], halves[0]))
    T.evaluate(T.ptx.cvt.f32.f16(dst[2 * i + 1], halves[1]))


# ---------------------------------------------------------------------------
# low-precision conversion compatibility
# ---------------------------------------------------------------------------


def cvt_rs_f16x2_f32(dst, a, b, rbits):
    """Stochastically convert two f32 values to a packed f16 pair.

    ``cvt.rs.f16x2.f32`` is architecture-specific: PTX exposes it on
    ``sm_100a`` and ``sm_103a``, but CUDA 13.1 rejects it for Thor's
    ``sm_110a``.  Preserve the native instruction on the two supported
    targets and otherwise use FlashInfer's integer emulation: add each
    13-bit random field to the discarded f32 mantissa bits, then truncate and
    rebias the result to f16.

    The operand and random-bit layout follows PTX directly: ``a`` becomes the
    high half and consumes bits 28:16 of *rbits*; ``b`` becomes the low half
    and consumes bits 12:0.
    """
    from tirx_kernels.runner import prepare_cuda_arch

    if prepare_cuda_arch("sm_100a") in {"sm_100a", "sm_103a"}:
        T.evaluate(T.ptx.cvt.rs.f16x2.f32(dst, a, b, rbits))
        return

    def cvt_rs_f16_sw(value, random13):
        materialized = T.alloc_local([1], "float32")
        _I.buffer_store(materialized, value, [0])
        bits = T.reinterpret("uint32", materialized[0])
        sign = T.bitwise_and(bits, T.uint32(0x80000000))
        abs_bits = T.bitwise_and(bits, T.uint32(0x7FFFFFFF)) + T.bitwise_and(
            random13, T.uint32(0x1FFF)
        )
        f32_exp = T.bitwise_and(T.shift_right(abs_bits, T.uint32(23)), T.uint32(0xFF))
        f32_mantissa = T.bitwise_and(abs_bits, T.uint32(0x7FFFFF))
        normal = T.bitwise_or(
            T.shift_left(f32_exp - T.uint32(112), T.uint32(10)),
            T.shift_right(f32_mantissa, T.uint32(13)),
        )
        magnitude = T.if_then_else(
            f32_exp == T.uint32(0xFF),
            T.if_then_else(f32_mantissa != T.uint32(0), T.uint32(0x7E00), T.uint32(0x7C00)),
            T.if_then_else(
                f32_exp > T.uint32(142),
                T.uint32(0x7C00),
                T.if_then_else(f32_exp < T.uint32(113), T.uint32(0), normal),
            ),
        )
        return T.bitwise_or(T.shift_right(sign, T.uint32(16)), magnitude)

    lo = cvt_rs_f16_sw(b, T.bitwise_and(rbits, T.uint32(0x1FFF)))
    hi = cvt_rs_f16_sw(a, T.bitwise_and(T.shift_right(rbits, T.uint32(16)), T.uint32(0x1FFF)))
    packed = T.bitwise_or(lo, T.shift_left(hi, T.uint32(16)))
    T.evaluate(T.ptx.mov.b32(dst, packed))


# ---------------------------------------------------------------------------
# warp reduction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# warp scan
# ---------------------------------------------------------------------------


def warp_scan_add(vals, n, lane, *, width=32, chain=True):
    """Inclusive ``+``-scan of ``vals[0:n]`` across the warp, in place.

    Each lane holds *n* consecutive elements, so the warp scans ``n * width``
    elements: sub-array ``i`` is the ``i``-th contiguous block.

    Expansion — ``log2(width)`` rounds of ``n`` shuffles and one guarded add
    block per round::

        for step in range(log2(width)):
            delta = 1 << step
            prior = K.alloc_local([n], <dtype>)
            for i in range(n):                                  # ALL shuffles first
                prior[i] = K.tvm_warp_shuffle_up(
                    K.uint32(0xFFFFFFFF), vals[i], delta, width, width)
            with K.If(lane >= delta), K.Then():                 # then the guarded adds
                for i in range(n):
                    vals[i] = vals[i] + prior[i]

    then, when ``chain=True`` (the default), the blocks are joined by
    carrying each sub-array's warp total into the next::

        for i in range(1, n):
            carry = K.alloc_local([1], <dtype>)
            carry[0] = K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), vals[i-1], width-1, width)
            vals[i] = vals[i] + carry[0]

    Why this form
    -------------
    The shuffles are hoisted **out** of the ``lane >= delta`` guard, and that
    placement is the whole reason this is an idiom rather than a loop anyone
    can write. ``shfl.up`` is a warp collective; a traced body emits a
    value-returning intrinsic at its **use** site, so writing the natural
    ``vals[i] = vals[i] + shfl_up(...)`` inside the guard puts the collective
    in divergent code, the excluded lanes never reach it, and the CTA
    deadlocks with no diagnostic (port notes §4 G3 — this is the exact bug that
    cost the port its first deadlock, found with ``cuda-gdb`` after a
    bounded-spin trace had cleared every mbarrier).

    The guard is still needed: lanes below ``delta`` have no predecessor and
    must not add. Hoisting the shuffle and keeping the add guarded is the only
    arrangement that is both correct and deadlock-free.

    ``chain`` defaults to **True**: holding ``n`` elements per lane in the
    block-interleaved layout almost always means "one sequence longer than the
    warp" — that is the reason the layout exists — so the joined scan is the
    common case and ``chain=False`` (``n`` unrelated ``width``-long sequences
    sharing one warp) is the variant that has to be asked for.
    The carry is sequential on purpose — by the time sub-array ``i`` is read it
    already carries every earlier block's total, so one shuffle per block
    suffices (the frozen kernel's ``n == 2`` case, orig:L1223-1230, is
    literally its last line).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive Python int (elements per lane), got {n!r}")
    if width & (width - 1):
        raise ValueError(f"width must be a power of two, got {width}")
    dtype = _dtype_of(vals[0])

    for step in range(width.bit_length() - 1):
        delta = 1 << step
        prior = T.alloc_local([n], dtype)
        for i in range(n):
            _I.buffer_store(
                prior, T.tvm_warp_shuffle_up(T.uint32(FULL_MASK), vals[i], delta, width, width), [i]
            )
        with T.If(lane >= delta), T.Then():
            for i in range(n):
                _I.buffer_store(vals, vals[i] + prior[i], [i])

    if chain:
        for i in range(1, n):
            carry = T.alloc_local([1], dtype)
            _I.buffer_store(
                carry, T.cuda._shfl_sync(T.uint32(FULL_MASK), vals[i - 1], width - 1, width), [0]
            )
            _I.buffer_store(vals, vals[i] + carry[0], [i])


__all__ = [
    "cast_f16x2_to_f32x2",
    "cvt_rs_f16x2_f32",
    "decode_instr_descriptor",
    "mma_chain",
    "sigmoid_tanh_approx_f32",
    "warp_scan_add",
]
