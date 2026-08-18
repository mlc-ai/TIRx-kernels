# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""``FilteredTopKUnifiedKernel`` building blocks (``include/flashinfer/topk.cuh``).

``FilteredTopKTraits`` (``:2272-2337``) and the row-scan helper the kernel funnels
every full-row pass through (``for_each_score``, ``:2463-2481``).

The traits are **not** interchangeable with ``RadixTopKTraits``
(``topk_common.cuh:35-49``).  Both make a float's bit pattern monotone, but the
filtered ones are written as ``(bits & sign) ? ~bits : (bits | sign)`` while the
radix ones use the XOR form, whose ``FromOrdered`` inverse
``StableSortTopKByValueKernel`` (``:3094``) depends on.  Keeping them separate is
deliberate.

The two are also lowered differently even though the source writes one
expression: on 16 bits nvcc recognises the shape as a sign-broadcast XOR
(``shr.s16`` + ``xor.b16``), and on 32 bits it routes it through
``abs``/``neg`` (``not.b32`` + ``abs.ftz.f32`` + ``neg.ftz.f32`` + ``setp.lt.s32``
+ ``selp.b32``).

Everything here that emits statements is a ``T.macro``: a plain Python helper
called from a traced body can build expressions but silently drops ``if``
statements and cannot write a local buffer.
"""

from typing import NamedTuple

from tirx_kernels.flashinfer.utils.topk_radix import (
    atom_shared_add_u32,
    bar_sync,
    ld_shared_u32,
    st_global_u16,
    st_global_u32,
    st_shared_u32,
    warp_inclusive_sum_u32,
)
from tvm.script import tirx as T

# Traits constants (:2279-2280, :2303-2304, :2323-2324).
NUM_REFINE_ROUNDS = {"float32": 4, "float16": 1, "bfloat16": 1}
FIRST_REFINE_SHIFT = {"float32": 24, "float16": 0, "bfloat16": 0}


def to_ordered_filtered_u32(bits):
    """``FilteredTopKTraits<float>::ToOrdered`` (``:2291-2294``).

    ``(bits & 0x80000000) ? ~bits : (bits | 0x80000000)``.
    """
    return T.Select(
        T.bitwise_and(bits, T.uint32(0x80000000)) != T.uint32(0),
        T.bitwise_xor(bits, T.uint32(0xFFFFFFFF)),
        T.bitwise_or(bits, T.uint32(0x80000000)),
    )


def to_ordered_filtered_u16(bits):
    """``FilteredTopKTraits<half|nv_bfloat16>::ToOrdered`` (``:2313-2316``, ``:2333-2336``)."""
    return T.Select(
        T.bitwise_and(bits, T.uint16(0x8000)) != T.uint16(0),
        T.bitwise_xor(bits, T.uint16(0xFFFF)),
        T.bitwise_or(bits, T.uint16(0x8000)),
    )


def to_coarse_key_u16(bits):
    """The shared tail of every ``ToCoarseKey``: monotone flip, then ``>> 8``.

    ``:2286-2288`` / ``:2308-2310`` / ``:2328-2330``.  The flip is the same one
    ``ToOrdered`` performs on 16 bits, so the coarse key is the ordered key's
    high byte.
    """
    return T.cast(T.shift_right(to_ordered_filtered_u16(bits), T.uint16(8)), "int32")


def to_coarse_key_f32(bits):
    """``FilteredTopKTraits<float>::ToCoarseKey`` (``:2282-2289``).

    Rounds through fp16 first (``__float2half_rn``, ``cvt.rn.f16.f32``), so the
    coarse key is **lossy** -- distinct floats can share a coarse bin.  That is
    fine because every phase re-derives it the same way, so the partition stays
    consistent; it is also why the refine rounds exist at all.
    """
    half_bits = T.reinterpret("uint16", T.cast(T.reinterpret("float32", bits), "float16"))
    return to_coarse_key_u16(half_bits)


def coarse_key(bits, is32):
    return to_coarse_key_f32(bits) if is32 else to_coarse_key_u16(bits)


def ordered_key(bits, is32):
    return to_ordered_filtered_u32(bits) if is32 else to_ordered_filtered_u16(bits)


# --- non-coherent global loads ----------------------------------------------
# Every input pointer of FilteredTopKUnifiedKernel is `__restrict__ const`
# (:2364-2373), so nvcc routes all of its global reads through the read-only
# path: the export carries 1476 `ld.global.nc.*` and no plain unified-kernel
# load.  These wrappers are local to this port on purpose -- the finalize
# kernel's pointers are NOT `__restrict__`, and its loads must stay plain
# `ld.global.b32`, as must the two merged radix ports that share
# `utils/topk_radix.py`.
def ld_global_nc_u32(buffer, index):
    """``ld.global.nc.b32``."""
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.global_.nc.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def ld_global_nc_bits(buf, elem_index, is32):
    """One scalar element's raw bits: ``ld.global.nc.b32`` | ``ld.global.nc.b16``."""
    if is32:
        return ld_global_nc_u32(buf, elem_index)
    out16 = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.nc.b16(out16[0], buf.ptr_to([elem_index])))
    return out16[0]


def ld_global_nc_words(buf, elem_index, load_bytes):
    """One vector load of ``load_bytes`` bytes, returned as 32-bit words."""
    if load_bytes == 16:
        w = T.alloc_local((4,), "uint32", align=16)
        T.evaluate(T.ptx["ld.global.nc.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index])))
        return [w[0], w[1], w[2], w[3]]
    if load_bytes == 8:
        w = T.alloc_local((2,), "uint32", align=8)
        T.evaluate(T.ptx["ld.global.nc.v2.b32"](w[0], w[1], buf.ptr_to([elem_index])))
        return [w[0], w[1]]
    w = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.global_.nc.b32(w[0], buf.ptr_to([elem_index])))
    return [w[0]]


def ld_global_nc_pair_u16(buffer, index):
    """``ld.global.nc.v2.b16``; both 16-bit lanes land in 16-bit registers.

    The source's ``vec_t<DType, 2>::cast_load`` keeps the monotone key flip on
    16-bit operands with no extract or repack arithmetic.
    """
    out = T.alloc_local((2,), "uint16")
    T.evaluate(T.ptx["ld.global.nc.v2.b16"](out[0], out[1], buffer.ptr_to([index])))
    return out[0], out[1]


def st_global_bits(buf, elem_index, bits, is32):
    """Store a raw 32- or 16-bit value; the mirror of ``ld_global_bits``."""
    if is32:
        st_global_u32(buf, elem_index, bits)
    else:
        st_global_u16(buf, elem_index, bits)


def atom_shared_or_b32(buffer, index, value):
    """``atom.shared.or.b32`` with the result discarded (``:2624``, ``:2667``).

    The source writes ``atomicOr(&s_refine_overflow, 1)`` and ignores the return
    value, yet nvcc still emits the ``atom`` form rather than the ``red``
    reduction form -- the export shows ``atom.shared.or.b32`` 20/29 times and
    ``red.shared.or.b32`` zero times, so the return operand is kept here too.
    """
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.atom.shared.or_.b32(out[0], buffer.ptr_to([index]), value))
    return out[0]


# --- scalar slots -----------------------------------------------------------
# The source declares these as separate __shared__ scalars (:2433-2441); they
# live in one small buffer here, which is the same bytes with the offsets folded
# into the address.
SC_COUNTER = 0  # s_counter
SC_THRESH_BIN = 1  # s_threshold_bin_id
SC_REFINE_OVERFLOW = 2  # s_refine_overflow
SC_LAST_REMAIN = 3  # s_last_remain
SC_NUM_INPUT = 4  # s_num_input[2]
SC_REFINE_TH = 6  # s_refine_thresholds[4]
# DeterministicContiguousCollect's chunk-walk state (:313-315); live only across
# that collector, but allocated with the rest so the layout stays static.
SC_EMITTED = 10
SC_CHUNK_BASE = 11
SC_CHUNK_TAKE = 12
NUM_SCALARS = 13


class FilteredCfg(NamedTuple):
    """Compile-time specialization carried into the macros below.

    Read fields **inline** as ``cfg.field``; never unpack them into locals first.
    Assigning a scalar inside a ``T.macro`` body turns it into a TIR expression,
    after which any Python-level use of it -- an ``if``, a ternary, ``not``, a
    ``range()`` bound, a list index -- fails or silently misbehaves, and the
    damage propagates to every macro the value is forwarded to.  Buffers are
    unaffected and may be unpacked normally.
    """

    top_k: int
    vec: int
    load_bytes: int
    is32: bool
    det: bool
    tie_break: int
    num_rounds: int
    first_shift: int
    smem_input: int
    hist_stride: int
    # `#pragma unroll 2` on the row-scan vector loop (:2469) is how many
    # independent global loads the source wants in flight.  This toolchain can
    # extract less memory parallelism from the same factor, so the value is
    # derived from the compile-time per-thread trip count rather than pinned to
    # the source's literal.
    scan_unroll: int
    basic: bool
    page_table: bool
    ragged: bool
    block: int = 1024
    radix: int = 256


def _emit_word_fanout(body, ctx, words, base, is32):
    """Hand each element of one vector load to the body, in index order.

    A plain Python function because the fan-out width is a compile-time count
    over a Python list: inside a macro, ``range(len(words))`` would become a TIR
    loop whose variable cannot index that list.
    """
    if is32:
        for w in range(len(words)):
            body(ctx, words[w], base + w)
    else:
        # Each loaded word carries two 16-bit lanes, low lane first.
        for w in range(len(words)):
            body(ctx, T.cast(T.bitwise_and(words[w], T.uint32(0xFFFF)), "uint16"), base + 2 * w)
            body(ctx, T.cast(T.shift_right(words[w], T.uint32(16)), "uint16"), base + 2 * w + 1)


@T.macro
def for_each_score(inp, row_in, tx, row_len, body, ctx, cfg):
    """``for_each_score`` / ``for_each_score_full`` (``:2463-2481``).

    A ``#pragma unroll 2`` vector loop over ``aligned_length = length / VEC_SIZE *
    VEC_SIZE`` followed by a scalar tail.  ``body(ctx, raw_bits, index)`` is a
    macro, matching the C++ lambdas the source hands this helper; ``ctx`` is an
    opaque tuple of the buffers and values that body needs, standing in for the
    lambdas' capture list.  Every load is ``.nc``-qualified because the kernel's
    inputs are ``__restrict__ const``.

    The kernel re-runs this in full on every phase that rescans the row, so the
    same load widths reappear in the histogram, the filter, and both fallbacks.
    """
    aligned: T.int32 = row_len // cfg.vec * cfg.vec
    for i in T.serial(tx * cfg.vec, aligned, step=cfg.block * cfg.vec, unroll=cfg.scan_unroll):
        if cfg.vec == 1:
            body(ctx, ld_global_nc_bits(inp, row_in + T.cast(i, "int64"), cfg.is32), i)
        else:
            if cfg.load_bytes == 4:
                if cfg.is32:
                    _emit_word_fanout(
                        body,
                        ctx,
                        ld_global_nc_words(inp, row_in + T.cast(i, "int64"), cfg.load_bytes),
                        i,
                        True,
                    )
                else:
                    # VEC_SIZE == 2 on a 16-bit dtype: the source's native
                    # ld.global.nc.v2.b16 pair, kept in 16-bit registers rather
                    # than routed through one 32-bit word and unpacked.
                    lo, hi = ld_global_nc_pair_u16(inp, row_in + T.cast(i, "int64"))
                    body(ctx, lo, i)
                    body(ctx, hi, i + 1)
            else:
                _emit_word_fanout(
                    body,
                    ctx,
                    ld_global_nc_words(inp, row_in + T.cast(i, "int64"), cfg.load_bytes),
                    i,
                    cfg.is32,
                )
    # Scalar tail (:2477-2480); empty when VEC_SIZE divides the row length.
    for j in T.serial(aligned + tx, row_len, step=cfg.block):
        body(ctx, ld_global_nc_bits(inp, row_in + T.cast(j, "int64"), cfg.is32), j)


@T.macro
def collect_gt_and_nondet_eq(ctx, value, threshold, idx, allow_eq):
    """``collect_gt_and_nondet_eq_threshold`` (``:2567-2580``).

    Strict winners are appended at ``s_counter``.  Equal-to-threshold elements
    only matter when ``!DETERMINISTIC``: they race for the remaining slots by
    counting ``s_last_remain`` **down** and writing from the back of
    ``s_indices``, so which ties win is genuinely racy.  Under ``DETERMINISTIC``
    this ``else if constexpr`` branch does not exist at all; those ties are
    collected later by ``collect_det_eq_pivot``.

    ``ctx = (s_scal, s_indices, cfg)``.
    """
    s_scal = ctx[0]
    s_indices = ctx[1]
    if value > threshold:
        pos: T.int32 = T.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, T.uint32(1)))
        st_shared_u32(s_indices, pos, T.reinterpret("uint32", idx))
    else:
        if not ctx[2].det:
            if allow_eq:
                if value == threshold:
                    back: T.int32 = T.reinterpret(
                        "int32", atom_shared_add_u32(s_scal, SC_LAST_REMAIN, T.uint32(0xFFFFFFFF))
                    )
                    if back > 0:
                        st_shared_u32(s_indices, ctx[2].top_k - back, T.reinterpret("uint32", idx))


@T.macro
def body_coarse_hist(ctx, bits, index):
    """``accumulate_coarse_hist`` (``:2482-2485``).  ``ctx = (s_hist2, cfg)``."""
    atom_shared_add_u32(ctx[0], coarse_key(bits, ctx[1].is32), T.uint32(1))


@T.macro
def body_collect_coarse_gt(ctx, bits, index):
    """``collect_coarse_gt`` on the coarse fast exit (``:2551-2557``).

    ``ctx = (s_scal, s_indices, threshold_bin, cfg)``.
    """
    if coarse_key(bits, ctx[3].is32) > ctx[2]:
        pos: T.int32 = T.reinterpret("int32", atom_shared_add_u32(ctx[0], SC_COUNTER, T.uint32(1)))
        st_shared_u32(ctx[1], pos, T.reinterpret("uint32", index))


@T.macro
def body_filter(ctx, bits, index):
    """``filter_and_add_to_histogram`` (``:2611-2627``) -- the step the algorithm is named for.

    Strict winners go straight out; only threshold-bin candidates are compacted
    into ``s_input_idx[0]``, and each one also bumps the first refine byte's
    histogram.  Past ``SMEM_INPUT_SIZE`` the compacted buffer is truncated and
    unusable, so the overflow flag is raised and a fallback rebuilds from the
    row.  The source's ``__builtin_expect(pos < SMEM_INPUT_SIZE, 1)`` marks that
    arm cold.

    ``ctx = (s_hist2, s_scal, s_indices, s_input, threshold_bin, cfg)``.
    """
    s_hist2 = ctx[0]
    s_scal = ctx[1]
    s_indices = ctx[2]
    s_input = ctx[3]
    bin_id: T.int32 = coarse_key(bits, ctx[5].is32)
    if bin_id > ctx[4]:
        pos: T.int32 = T.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, T.uint32(1)))
        st_shared_u32(s_indices, pos, T.reinterpret("uint32", index))
    else:
        if bin_id == ctx[4]:
            slot: T.int32 = T.reinterpret(
                "int32", atom_shared_add_u32(s_scal, SC_NUM_INPUT, T.uint32(1))
            )
            if slot < ctx[5].smem_input:
                st_shared_u32(s_input, slot, T.reinterpret("uint32", index))
                sub: T.int32 = T.cast(
                    T.bitwise_and(
                        T.shift_right(
                            T.cast(ordered_key(bits, ctx[5].is32), "uint32"),
                            T.uint32(ctx[5].first_shift),
                        ),
                        T.uint32(0xFF),
                    ),
                    "int32",
                )
                atom_shared_add_u32(s_hist2, sub, T.uint32(1))
            else:
                atom_shared_or_b32(s_scal, SC_REFINE_OVERFLOW, T.uint32(1))


@T.macro
def run_cumsum(s_hist2, tx, cfg):
    """Hillis-Steele inclusive **suffix** scan, eight ping-pong steps (``:2490-2504``).

    Eight is even, so the result lands back in buffer 0, which is the alias the
    rest of the kernel reads as ``s_histogram``.  ``s_histogram[RADIX]`` must stay
    zero throughout: it is the exclusive-suffix sentinel every threshold test
    reads at ``tx + 1``, and the scan only ever writes indices below ``RADIX``.
    """
    for i in T.unroll(8):
        if tx < cfg.radix:
            j: T.int32 = T.shift_left(T.int32(1), i)
            src: T.int32 = T.bitwise_and(i, T.int32(1)) * cfg.hist_stride
            dst: T.int32 = T.bitwise_xor(T.bitwise_and(i, T.int32(1)), T.int32(1)) * cfg.hist_stride
            value: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, src + tx))
            if tx < cfg.radix - j:
                value = value + T.reinterpret("int32", ld_shared_u32(s_hist2, src + tx + j))
            st_shared_u32(s_hist2, dst + tx, T.reinterpret("uint32", value))
        bar_sync()


# --- cub::BlockScan<uint32_t, 1024, BLOCK_SCAN_RAKING_MEMOIZE> geometry -------
# The tie collectors share one instance of this scan (:2584-2585).  It is the
# same collective the radix single-CTA port already carries, so the raking layout
# matches: one warp rakes 32-element segments with a padded stride.
RAKING_THREADS = 32
RAKING_SEGMENT = 32
RAKING_STRIDE = RAKING_SEGMENT + 1  # cub BlockRakingLayout segment padding
RAKING_ELEMENTS = RAKING_THREADS * RAKING_STRIDE  # 1056
DET_ITEMS_PER_THREAD = 4  # DeterministicContiguousCollect (:311)


def raking_offset(tx):
    """Padded placement of a thread's cell in the raking grid."""
    return tx // RAKING_SEGMENT * RAKING_STRIDE + tx % RAKING_SEGMENT


def scan_elements():
    """Raking grid plus one slot for the block aggregate."""
    return RAKING_ELEMENTS + 1


@T.macro
def block_exclusive_sum_raking(s_scan, out, total_out, tx, value):
    """``BlockScan::ExclusiveSum`` with ``BLOCK_SCAN_RAKING_MEMOIZE``.

    Place into the padded raking grid; one warp serially reduces its 32-element
    segment while memoizing it in registers, a warp shuffle scan runs over the
    segment totals, then the same warp scatters the prefixes back.  ``out[0]``
    receives this thread's exclusive prefix and ``total_out[0]`` the block
    aggregate, which the contiguous collector needs for its quota walk.
    """
    off: T.int32 = raking_offset(tx)
    st_shared_u32(s_scan, off, value)
    bar_sync()
    if tx < RAKING_THREADS:
        base: T.int32 = tx * RAKING_STRIDE
        cache = T.alloc_local((RAKING_SEGMENT,), "uint32")
        total: T.uint32 = T.uint32(0)
        for j in T.unroll(RAKING_SEGMENT):
            cache[j] = ld_shared_u32(s_scan, base + j)
            total = total + cache[j]
        incl: T.uint32 = warp_inclusive_sum_u32(total, tx)
        run: T.uint32 = incl - total
        for j2 in T.unroll(RAKING_SEGMENT):
            st_shared_u32(s_scan, base + j2, run)
            run = run + cache[j2]
        if tx == RAKING_THREADS - 1:
            st_shared_u32(s_scan, RAKING_ELEMENTS, incl)
    bar_sync()
    out[0] = ld_shared_u32(s_scan, off)
    total_out[0] = ld_shared_u32(s_scan, RAKING_ELEMENTS)


@T.macro
def det_thread_strided_collect(inp, s_scan, s_indices, tx, row_in, row_len, cfg, pivot, eq_needed):
    """``DeterministicThreadStridedCollect`` (``:255-286``), the TIE_BREAK=None collector.

    Count matches per thread over a thread-strided walk, block-scan the counts,
    then re-walk and emit.  The emit target is
    ``s_indices[top_k - eq_needed + local_pos]`` (``:2587-2590``); the predicate
    is ``ToOrdered(score[idx]) == pivot``.
    """
    count = T.alloc_local((1,), "uint32")
    count[0] = T.uint32(0)
    for i in T.serial(tx, row_len, step=cfg.block):
        cur: T.uint32 = T.cast(
            ordered_key(ld_global_nc_bits(inp, row_in + T.cast(i, "int64"), cfg.is32), cfg.is32),
            "uint32",
        )
        if cur == pivot:
            count[0] = count[0] + T.uint32(1)
    prefix = T.alloc_local((1,), "uint32")
    total = T.alloc_local((1,), "uint32")
    block_exclusive_sum_raking(s_scan, prefix, total, tx, count[0])
    if count[0] > T.uint32(0):
        if prefix[0] < T.cast(eq_needed, "uint32"):
            pos = T.alloc_local((1,), "uint32")
            pos[0] = prefix[0]
            end: T.uint32 = T.min(prefix[0] + count[0], T.cast(eq_needed, "uint32"))
            done = T.alloc_local((1,), "int32")
            done[0] = T.int32(0)
            for i2 in T.serial(tx, row_len, step=cfg.block):
                if done[0] == 0:
                    cur2: T.uint32 = T.cast(
                        ordered_key(
                            ld_global_nc_bits(inp, row_in + T.cast(i2, "int64"), cfg.is32), cfg.is32
                        ),
                        "uint32",
                    )
                    if cur2 == pivot:
                        st_shared_u32(
                            s_indices,
                            cfg.top_k - eq_needed + T.reinterpret("int32", pos[0]),
                            T.reinterpret("uint32", i2),
                        )
                        pos[0] = pos[0] + T.uint32(1)
                        if pos[0] == end:
                            done[0] = T.int32(1)
    bar_sync()


@T.macro
def det_contiguous_collect(
    inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivot, eq_needed, reverse
):
    """``DeterministicContiguousCollect`` (``:298-377``), the TIE_BREAK=Small/Large collector.

    Walks the row in **contiguous** index order across the CTA in
    ``BLOCK_THREADS * 4``-element chunks, so equal-valued candidates are claimed
    smallest index first, or largest first when ``REVERSE``.  ``s_emitted`` /
    ``s_chunk_base`` / ``s_chunk_take`` carry the quota between chunks and the
    walk stops once it is met.
    """
    if tx == 0:
        st_shared_u32(s_scal, SC_EMITTED, T.uint32(0))
        st_shared_u32(s_scal, SC_CHUNK_BASE, T.uint32(0))
        st_shared_u32(s_scal, SC_CHUNK_TAKE, T.uint32(0))
    bar_sync()
    chunk_items: T.int32 = cfg.block * DET_ITEMS_PER_THREAD
    num_chunks: T.int32 = (row_len + chunk_items - 1) // chunk_items
    stop = T.alloc_local((1,), "int32")
    stop[0] = T.int32(0)
    for chunk in T.serial(0, num_chunks):
        if stop[0] == 0:
            rows_of = T.alloc_local((DET_ITEMS_PER_THREAD,), "int32")
            sel_of = T.alloc_local((DET_ITEMS_PER_THREAD,), "uint32")
            cnt = T.alloc_local((1,), "uint32")
            cnt[0] = T.uint32(0)
            for item in T.unroll(DET_ITEMS_PER_THREAD):
                linear: T.int32 = chunk * chunk_items + tx * DET_ITEMS_PER_THREAD + item
                rows_of[item] = T.int32(0)
                sel_of[item] = T.uint32(0)
                if linear < row_len:
                    if reverse:
                        rows_of[item] = row_len - 1 - linear
                    else:
                        rows_of[item] = linear
                    curc: T.uint32 = T.cast(
                        ordered_key(
                            ld_global_nc_bits(
                                inp, row_in + T.cast(rows_of[item], "int64"), cfg.is32
                            ),
                            cfg.is32,
                        ),
                        "uint32",
                    )
                    if curc == pivot:
                        sel_of[item] = T.uint32(1)
                        cnt[0] = cnt[0] + T.uint32(1)
            prefix = T.alloc_local((1,), "uint32")
            blocksel = T.alloc_local((1,), "uint32")
            block_exclusive_sum_raking(s_scan, prefix, blocksel, tx, cnt[0])
            if tx == 0:
                emitted: T.uint32 = ld_shared_u32(s_scal, SC_EMITTED)
                st_shared_u32(s_scal, SC_CHUNK_BASE, emitted)
                remaining: T.uint32 = T.uint32(0)
                if emitted < T.cast(eq_needed, "uint32"):
                    remaining = T.cast(eq_needed, "uint32") - emitted
                take: T.uint32 = T.min(remaining, blocksel[0])
                st_shared_u32(s_scal, SC_CHUNK_TAKE, take)
                st_shared_u32(s_scal, SC_EMITTED, emitted + take)
            bar_sync()
            chunk_take: T.uint32 = ld_shared_u32(s_scal, SC_CHUNK_TAKE)
            chunk_base: T.uint32 = ld_shared_u32(s_scal, SC_CHUNK_BASE)
            if cnt[0] > T.uint32(0):
                if prefix[0] < chunk_take:
                    epos = T.alloc_local((1,), "uint32")
                    epos[0] = prefix[0]
                    eend: T.uint32 = T.min(prefix[0] + cnt[0], chunk_take)
                    fin = T.alloc_local((1,), "int32")
                    fin[0] = T.int32(0)
                    for item2 in T.unroll(DET_ITEMS_PER_THREAD):
                        if fin[0] == 0:
                            if sel_of[item2] == T.uint32(1):
                                st_shared_u32(
                                    s_indices,
                                    cfg.top_k
                                    - eq_needed
                                    + T.reinterpret("int32", chunk_base + epos[0]),
                                    T.reinterpret("uint32", rows_of[item2]),
                                )
                                epos[0] = epos[0] + T.uint32(1)
                                if epos[0] == eend:
                                    fin[0] = T.int32(1)
            bar_sync()
            if ld_shared_u32(s_scal, SC_EMITTED) >= T.cast(eq_needed, "uint32"):
                stop[0] = T.int32(1)
    bar_sync()


@T.macro
def collect_det_eq_pivot(
    inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivot, eq_needed
):
    """``collect_det_eq_pivot`` (``:2582-2608``).

    Picks the collector by ``TIE_BREAK``: contiguous ascending for ``Small``,
    contiguous descending for ``Large``, thread-strided for plain determinism.
    All three share one BlockScan instance and the same
    ``ToOrdered(score[idx]) == pivot`` predicate.
    """
    if eq_needed > 0:
        if cfg.tie_break == 1:
            det_contiguous_collect(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivot, eq_needed, False
            )
        elif cfg.tie_break == 2:
            det_contiguous_collect(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivot, eq_needed, True
            )
        else:
            det_thread_strided_collect(
                inp, s_scan, s_indices, tx, row_in, row_len, cfg, pivot, eq_needed
            )


@T.macro
def update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, next_idx, reset_next):
    """``update_refine_threshold`` (``:2505-2516``).

    ``run_cumsum`` plus the same predicated pick as the first one, except that it
    also publishes ``s_last_remain`` and never touches ``s_counter`` -- only the
    first pick (``:2522``) resets that.  ``RESET_NEXT_INPUT`` is false at exactly
    one call site, the 16-bit overflow fallback (``:2733``).
    """
    run_cumsum(s_hist2, tx, cfg)
    if tx < cfg.radix:
        cur: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, tx))
        if cur > topk[0]:
            nxt: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
            if nxt <= topk[0]:
                st_shared_u32(s_scal, SC_THRESH_BIN, T.reinterpret("uint32", tx))
                if reset_next:
                    st_shared_u32(s_scal, SC_NUM_INPUT + next_idx, T.uint32(0))
                st_shared_u32(s_scal, SC_LAST_REMAIN, T.reinterpret("uint32", topk[0] - nxt))
    bar_sync()


@T.macro
def run_refine_round(
    inp,
    s_hist2,
    s_indices,
    s_scal,
    s_input,
    tx,
    row_in,
    cfg,
    topk,
    resolved,
    r_idx,
    offset,
    is_last,
):
    """``run_refine_round`` (``:2675-2709``).

    ``r_idx`` is the ping-pong **buffer** index (``round % 2``, ``:2775``), never
    the round number: ``s_input_idx`` and ``s_num_input`` are two deep, so
    indexing them by the round would run off the end from round 2 on.  Sets
    ``resolved[0]`` when the round fully resolves the pivot.
    """
    raw: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_NUM_INPUT + r_idx))
    num_input: T.int32 = T.min(raw, cfg.smem_input)

    update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, r_idx ^ 1, True)

    threshold: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
    if cfg.det:
        if tx == 0:
            st_shared_u32(
                s_scal,
                SC_REFINE_TH + (cfg.first_shift - offset) // 8,
                T.reinterpret("uint32", threshold),
            )
    topk[0] = topk[0] - T.reinterpret("int32", ld_shared_u32(s_hist2, threshold + 1))
    if topk[0] == 0:
        # Pivot resolved: only bins strictly greater than the threshold remain.
        for i in T.serial(tx, num_input, step=cfg.block):
            idx: T.int32 = T.reinterpret(
                "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i)
            )
            bin_id: T.int32 = T.cast(
                T.bitwise_and(
                    T.shift_right(
                        T.cast(
                            ordered_key(
                                ld_global_nc_bits(inp, row_in + T.cast(idx, "int64"), cfg.is32),
                                cfg.is32,
                            ),
                            "uint32",
                        ),
                        T.uint32(offset),
                    ),
                    T.uint32(0xFF),
                ),
                "int32",
            )
            if bin_id > threshold:
                pos: T.int32 = T.reinterpret(
                    "int32", atom_shared_add_u32(s_scal, SC_COUNTER, T.uint32(1))
                )
                st_shared_u32(s_indices, pos, T.reinterpret("uint32", idx))
        bar_sync()
        resolved[0] = T.int32(1)
    else:
        if is_last:
            # collect_with_threshold_last_round (:2635-2645): one barrier.
            for i2 in T.serial(tx, num_input, step=cfg.block):
                idx2: T.int32 = T.reinterpret(
                    "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i2)
                )
                bin2: T.int32 = T.cast(
                    T.bitwise_and(
                        T.shift_right(
                            T.cast(
                                ordered_key(
                                    ld_global_nc_bits(
                                        inp, row_in + T.cast(idx2, "int64"), cfg.is32
                                    ),
                                    cfg.is32,
                                ),
                                "uint32",
                            ),
                            T.uint32(offset),
                        ),
                        T.uint32(0xFF),
                    ),
                    "int32",
                )
                collect_gt_and_nondet_eq((s_scal, s_indices, cfg), bin2, threshold, idx2, True)
            bar_sync()
        else:
            # collect_with_threshold_non_last_round (:2646-2672): three barriers,
            # ping-ponging the survivors into s_input_idx[r_idx ^ 1] together with
            # the next byte's histogram.
            bar_sync()
            if tx < cfg.radix + 1:
                st_shared_u32(s_hist2, tx, T.uint32(0))
            bar_sync()
            for i3 in T.serial(tx, num_input, step=cfg.block):
                idx3: T.int32 = T.reinterpret(
                    "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i3)
                )
                ord3: T.uint32 = T.cast(
                    ordered_key(
                        ld_global_nc_bits(inp, row_in + T.cast(idx3, "int64"), cfg.is32), cfg.is32
                    ),
                    "uint32",
                )
                bin3: T.int32 = T.cast(
                    T.bitwise_and(T.shift_right(ord3, T.uint32(offset)), T.uint32(0xFF)), "int32"
                )
                if bin3 > threshold:
                    pos3: T.int32 = T.reinterpret(
                        "int32", atom_shared_add_u32(s_scal, SC_COUNTER, T.uint32(1))
                    )
                    st_shared_u32(s_indices, pos3, T.reinterpret("uint32", idx3))
                else:
                    if bin3 == threshold:
                        slot3: T.int32 = T.reinterpret(
                            "int32",
                            atom_shared_add_u32(s_scal, SC_NUM_INPUT + (r_idx ^ 1), T.uint32(1)),
                        )
                        if slot3 < cfg.smem_input:
                            st_shared_u32(
                                s_input,
                                (r_idx ^ 1) * cfg.smem_input + slot3,
                                T.reinterpret("uint32", idx3),
                            )
                            sub3: T.int32 = T.cast(
                                T.bitwise_and(
                                    T.shift_right(ord3, T.uint32(offset - 8)), T.uint32(0xFF)
                                ),
                                "int32",
                            )
                            atom_shared_add_u32(s_hist2, sub3, T.uint32(1))
                        else:
                            atom_shared_or_b32(s_scal, SC_REFINE_OVERFLOW, T.uint32(1))
            bar_sync()


@T.macro
def body_rehist_threshold_bin(ctx, bits, index):
    """16-bit fallback re-histogram (``:2715-2724``): low byte, threshold bin only.

    ``ctx = (s_hist2, threshold_bin, cfg)``.
    """
    if coarse_key(bits, ctx[2].is32) == ctx[1]:
        atom_shared_add_u32(
            ctx[0],
            T.cast(
                T.bitwise_and(T.cast(ordered_key(bits, ctx[2].is32), "uint32"), T.uint32(0xFF)),
                "int32",
            ),
            T.uint32(1),
        )


@T.macro
def body_recollect_threshold_bin(ctx, bits, index):
    """16-bit fallback re-collect (``:2740-2748``).

    The ``coarse_bin != threshold_bin`` guard at ``:2741-2744`` is load-bearing:
    without it the pass would compare out-of-bin elements by their low byte and
    re-collect every strict winner the filter stage already appended.

    ``ctx = (s_scal, s_indices, threshold_bin, cfg)``.
    """
    if coarse_key(bits, ctx[3].is32) == ctx[2]:
        sub: T.int32 = T.cast(
            T.bitwise_and(T.cast(ordered_key(bits, ctx[3].is32), "uint32"), T.uint32(0xFF)), "int32"
        )
        threshold: T.int32 = T.reinterpret("int32", ld_shared_u32(ctx[0], SC_THRESH_BIN))
        collect_gt_and_nondet_eq((ctx[0], ctx[1], ctx[3]), sub, threshold, index, True)


@T.macro
def _prefix_match_one(match, ordered, threshold_bytes, prev):
    """One already-fixed byte of the fallback's prefix test (``:2823-2831``)."""
    got: T.int32 = T.cast(
        T.bitwise_and(T.shift_right(ordered, T.uint32(24 - prev * 8)), T.uint32(0xFF)), "int32"
    )
    if got != T.cast(threshold_bytes[prev], "int32"):
        match[0] = T.int32(0)


def _prefix_match(match, ordered, threshold_bytes, rnd):
    """Compare every byte fixed by an earlier round; a Python loop, so it unrolls."""
    for prev in range(rnd):
        _prefix_match_one(match, ordered, threshold_bytes, prev)


@T.macro
def body_fallback_rehist(ctx, bits, index):
    """fp32 fallback per-round re-histogram (``:2819-2837``).

    Only threshold-bin elements whose ordered key still matches the bytes fixed
    by earlier rounds contribute to this round's byte histogram.
    ``threshold_bytes`` is the per-thread register array of ``:2805``.

    ``ctx = (s_hist2, threshold_bytes, threshold_bin, cfg, round)``.
    """
    if coarse_key(bits, ctx[3].is32) == ctx[2]:
        ordered: T.uint32 = T.cast(ordered_key(bits, ctx[3].is32), "uint32")
        match = T.alloc_local((1,), "int32")
        match[0] = T.int32(1)
        _prefix_match(match, ordered, ctx[1], ctx[4])
        if match[0] == 1:
            atom_shared_add_u32(
                ctx[0],
                T.cast(
                    T.bitwise_and(
                        T.shift_right(ordered, T.uint32(24 - ctx[4] * 8)), T.uint32(0xFF)
                    ),
                    "int32",
                ),
                T.uint32(1),
            )


@T.macro
def body_collect_by_pivot(ctx, bits, index):
    """fp32 fallback re-collect (``:2883-2895``) -- a three-way dispatch.

    Coarse-bin winners are compared on the **coarse** key with no eq claim;
    out-of-bin elements are dropped; threshold-bin elements are compared on the
    full 32-bit ordered key against the rebuilt pivot.

    ``ctx = (s_scal, s_indices, threshold_bin, cfg, pivot, eq_needed)``.
    """
    bin_id: T.int32 = coarse_key(bits, ctx[3].is32)
    if bin_id > ctx[2]:
        collect_gt_and_nondet_eq((ctx[0], ctx[1], ctx[3]), bin_id, ctx[2], index, False)
    else:
        if bin_id == ctx[2]:
            collect_gt_and_nondet_eq(
                (ctx[0], ctx[1], ctx[3]),
                T.cast(ordered_key(bits, ctx[3].is32), "uint32"),
                ctx[4],
                index,
                ctx[5] > 0,
            )


@T.macro
def _fp32_refine_one_round(
    inp, s_hist2, s_indices, s_scal, s_input, tx, row_in, cfg, topk, resolved, stop, det_stop, rnd
):
    """One iteration of the source's ``#pragma unroll`` refine loop (``:2774-2790``)."""
    if stop[0] == 0:
        run_refine_round(
            inp,
            s_hist2,
            s_indices,
            s_scal,
            s_input,
            tx,
            row_in,
            cfg,
            topk,
            resolved,
            rnd % 2,  # ping-pong BUFFER index (:2775), not the round
            cfg.first_shift - rnd * 8,
            rnd == 3,
        )
        if resolved[0] == 1:
            det_stop[0] = T.int32(rnd)
            stop[0] = T.int32(1)
        else:
            if T.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0:
                stop[0] = T.int32(1)


@T.macro
def _fp32_fallback_one_round(
    inp,
    s_hist2,
    s_scal,
    tx,
    row_in,
    row_len,
    cfg,
    threshold_bin,
    bytes_reg,
    remain,
    stop_round,
    halt,
    rnd,
):
    """One iteration of the fallback's ``#pragma unroll`` rebuild loop (``:2812-2856``)."""
    if halt[0] == 0:
        if tx < cfg.radix + 1:
            st_shared_u32(s_hist2, tx, T.uint32(0))
        bar_sync()
        for_each_score(
            inp,
            row_in,
            tx,
            row_len,
            body_fallback_rehist,
            (s_hist2, bytes_reg, threshold_bin, cfg, rnd),
            cfg,
        )
        bar_sync()
        run_cumsum(s_hist2, tx, cfg)
        if tx < cfg.radix:
            curf: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, tx))
            if curf > remain[0]:
                nxtf: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
                if nxtf <= remain[0]:
                    # Only the bin id here; s_num_input and s_last_remain stay
                    # untouched (:2842-2845).
                    st_shared_u32(s_scal, SC_THRESH_BIN, T.reinterpret("uint32", tx))
        bar_sync()
        thrf: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
        # threshold_bytes is a per-thread register array in the source (:2805);
        # every thread reads the same s_threshold_bin_id, so no publication step
        # and no extra barrier is needed.
        bytes_reg[rnd] = T.cast(thrf, "uint32")
        remain[0] = remain[0] - T.reinterpret("int32", ld_shared_u32(s_hist2, thrf + 1))
        bar_sync()
        if remain[0] == 0:
            stop_round[0] = T.int32(rnd)
            halt[0] = T.int32(1)


@T.macro
def _fp32_build_det_pivot(s_scal, piv, det_stop, rnd):
    """One byte of ``build_det_pivot`` (``:2539-2544``)."""
    byte0: T.uint32 = T.Select(
        T.int32(rnd) <= det_stop[0], ld_shared_u32(s_scal, SC_REFINE_TH + rnd), T.uint32(0xFF)
    )
    piv[0] = T.bitwise_or(piv[0], T.shift_left(byte0, T.uint32(24 - rnd * 8)))


@T.macro
def _fp32_assemble_pivot(pivf, bytes_reg, remain, stop_round, rnd):
    """One byte of the fallback pivot (``:2860-2867``).

    A byte is forced to ``0xFF`` only once the quota is met and the round is past
    ``stop_round``.
    """
    bytef = T.alloc_local((1,), "uint32")
    bytef[0] = bytes_reg[rnd]
    if remain[0] == 0:
        if T.int32(rnd) > stop_round[0]:
            bytef[0] = T.uint32(0xFF)
    pivf[0] = T.bitwise_or(pivf[0], T.shift_left(bytef[0], T.uint32(24 - rnd * 8)))


def _fp32_rounds_unrolled(
    inp, s_hist2, s_indices, s_scal, s_input, tx, row_in, cfg, topk, resolved, stop, det_stop
):
    for rnd in range(4):
        _fp32_refine_one_round(
            inp,
            s_hist2,
            s_indices,
            s_scal,
            s_input,
            tx,
            row_in,
            cfg,
            topk,
            resolved,
            stop,
            det_stop,
            rnd,
        )


def _fp32_det_pivot_bytes(s_scal, piv, det_stop):
    for rnd in range(4):
        _fp32_build_det_pivot(s_scal, piv, det_stop, rnd)


def _fp32_fallback_rounds(
    inp,
    s_hist2,
    s_scal,
    tx,
    row_in,
    row_len,
    cfg,
    threshold_bin,
    bytes_reg,
    remain,
    stop_round,
    halt,
):
    for rnd in range(4):
        _fp32_fallback_one_round(
            inp,
            s_hist2,
            s_scal,
            tx,
            row_in,
            row_len,
            cfg,
            threshold_bin,
            bytes_reg,
            remain,
            stop_round,
            halt,
            rnd,
        )


def _fp32_pivot_bytes(pivf, bytes_reg, remain, stop_round):
    for rnd in range(4):
        _fp32_assemble_pivot(pivf, bytes_reg, remain, stop_round, rnd)


@T.macro
def _fp32_refine_prologue(
    s_scal, tx, det_stop, stop, remain, stop_round, halt, bytes_reg, topk_after_coarse
):
    """Register state the fp32 refine path carries across its unrolled rounds."""
    det_stop[0] = T.int32(3)  # NUM_ROUNDS - 1 (:2771)
    stop[0] = T.int32(0)
    remain[0] = topk_after_coarse  # (:2804)
    stop_round[0] = T.int32(3)
    halt[0] = T.int32(0)
    for r in T.unroll(4):
        bytes_reg[r] = T.uint32(0xFF)  # (:2805-2810)


def emit_fp32_refine(
    inp,
    s_hist2,
    s_indices,
    s_scal,
    s_input,
    s_scan,
    tx,
    row_in,
    row_len,
    cfg,
    topk,
    resolved,
    det_stop,
    stop,
    remain,
    stop_round,
    halt,
    bytes_reg,
    piv,
    pivf,
    threshold_bin,
    topk_after_coarse,
):
    """fp32's four refine rounds and the 32-bit pivot-rebuild fallback (``:2767-2902``).

    A plain Python function so the ``#pragma unroll`` loops over ``NUM_ROUNDS``
    stay compile-time: inside a macro ``range(4)`` becomes a TIR loop, which would
    make the ping-pong index, the byte offset, and ``IS_LAST_ROUND`` runtime
    values and collapse the source's four distinct round bodies into one.
    """
    _fp32_refine_prologue(
        s_scal, tx, det_stop, stop, remain, stop_round, halt, bytes_reg, topk_after_coarse
    )
    _fp32_guarded_rounds(
        inp, s_hist2, s_indices, s_scal, s_input, tx, row_in, cfg, topk, resolved, stop, det_stop
    )
    _fp32_det_collect(inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, topk, det_stop, piv)
    _fp32_fallback(
        inp,
        s_hist2,
        s_indices,
        s_scal,
        s_scan,
        tx,
        row_in,
        row_len,
        cfg,
        threshold_bin,
        bytes_reg,
        remain,
        stop_round,
        halt,
        pivf,
    )


@T.macro
def _fp32_guarded_rounds(
    inp, s_hist2, s_indices, s_scal, s_input, tx, row_in, cfg, topk, resolved, stop, det_stop
):
    """The whole round loop is guarded on the flag the filter stage may have set (``:2772``)."""
    if T.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) == 0:
        _fp32_rounds_unrolled(
            inp,
            s_hist2,
            s_indices,
            s_scal,
            s_input,
            tx,
            row_in,
            cfg,
            topk,
            resolved,
            stop,
            det_stop,
        )


@T.macro
def _fp32_det_collect(
    inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, topk, det_stop, piv
):
    """The deterministic collect sits OUTSIDE the guard and re-checks the flag itself.

    ``run_refine_round`` can raise ``s_refine_overflow`` mid-loop through the
    ``atomicOr`` at ``:2667``; the source spells this out at ``:2798-2799``.
    """
    if cfg.det:
        if T.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) == 0:
            piv[0] = T.uint32(0)
            _fp32_det_pivot_bytes(s_scal, piv, det_stop)
            collect_det_eq_pivot(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, piv[0], topk[0]
            )


@T.macro
def _fp32_fallback(
    inp,
    s_hist2,
    s_indices,
    s_scal,
    s_scan,
    tx,
    row_in,
    row_len,
    cfg,
    threshold_bin,
    bytes_reg,
    remain,
    stop_round,
    halt,
    pivf,
):
    """32-bit pivot rebuild after an overflow (``:2800-2900``).

    Overflow can follow partial writes to ``s_indices`` / ``s_counter``, so the
    selection is rebuilt from scratch.
    """
    if T.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0:
        _fp32_fallback_rounds(
            inp,
            s_hist2,
            s_scal,
            tx,
            row_in,
            row_len,
            cfg,
            threshold_bin,
            bytes_reg,
            remain,
            stop_round,
            halt,
        )
        pivf[0] = T.uint32(0)
        _fp32_pivot_bytes(pivf, bytes_reg, remain, stop_round)
        if tx == 0:
            st_shared_u32(s_scal, SC_COUNTER, T.uint32(0))
            st_shared_u32(s_scal, SC_LAST_REMAIN, T.reinterpret("uint32", remain[0]))
        bar_sync()
        for_each_score(
            inp,
            row_in,
            tx,
            row_len,
            body_collect_by_pivot,
            (s_scal, s_indices, threshold_bin, cfg, pivf[0], remain[0]),
            cfg,
        )
        bar_sync()
        if cfg.det:
            collect_det_eq_pivot(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivf[0], remain[0]
            )


@T.macro
def emit_filtered_topk_main(
    inp,
    out_idx,
    out_val,
    aux,
    s_hist2,
    s_indices,
    s_scal,
    s_input,
    s_scan,
    tx,
    row_in,
    row_out,
    row_len,
    batch_idx,
    page_start,
    offset_val,
    aux_stride,
    cfg,
):
    """``FilteredTopKUnifiedKernel``'s non-trivial path (``:2431-2919``)."""
    topk = T.alloc_local((1,), "int32")
    topk[0] = cfg.top_k

    # --- init (:2450-2458) -------------------------------------------------
    if tx == 0:
        st_shared_u32(s_scal, SC_REFINE_OVERFLOW, T.uint32(0))
    if cfg.det:
        if tx < 4:
            st_shared_u32(s_scal, SC_REFINE_TH + tx, T.uint32(0xFF))
    if tx < cfg.radix + 1:
        st_shared_u32(s_hist2, tx, T.uint32(0))
    bar_sync()

    # --- Stage 1: coarse histogram over the whole row (:2482-2487) ---------
    for_each_score(inp, row_in, tx, row_len, body_coarse_hist, (s_hist2, cfg), cfg)
    bar_sync()

    # --- first threshold pick (:2518-2524) ---------------------------------
    # Three short-circuit guards, not a fused predicate: the export shows three
    # `@p bra` and zero `and.pred`.  Exactly one thread satisfies all three, and
    # the counter resets ride on that same predicate rather than on tx == 0.
    run_cumsum(s_hist2, tx, cfg)
    if tx < cfg.radix:
        cur0: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, tx))
        if cur0 > topk[0]:
            nxt0: T.int32 = T.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
            if nxt0 <= topk[0]:
                st_shared_u32(s_scal, SC_THRESH_BIN, T.reinterpret("uint32", tx))
                st_shared_u32(s_scal, SC_NUM_INPUT, T.uint32(0))
                st_shared_u32(s_scal, SC_COUNTER, T.uint32(0))
    bar_sync()
    threshold_bin: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
    topk[0] = topk[0] - T.reinterpret("int32", ld_shared_u32(s_hist2, threshold_bin + 1))
    topk_after_coarse: T.int32 = topk[0]

    if topk[0] == 0:
        # The coarse pass already resolved k (:2549-2559).
        for_each_score(
            inp,
            row_in,
            tx,
            row_len,
            body_collect_coarse_gt,
            (s_scal, s_indices, threshold_bin, cfg),
            cfg,
        )
        bar_sync()
    else:
        # --- Stage 2: the filter (:2561-2629) ------------------------------
        bar_sync()
        if tx < cfg.radix + 1:
            st_shared_u32(s_hist2, tx, T.uint32(0))
        bar_sync()
        for_each_score(
            inp,
            row_in,
            tx,
            row_len,
            body_filter,
            (s_hist2, s_scal, s_indices, s_input, threshold_bin, cfg),
            cfg,
        )
        bar_sync()

        # --- Stage 3: refine (:2710-2902) ----------------------------------
        resolved = T.alloc_local((1,), "int32")
        resolved[0] = T.int32(0)
        if cfg.num_rounds == 1:
            # 16-bit: a single refine round, with a full-row slow path on overflow.
            if T.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0:
                if tx < cfg.radix + 1:
                    st_shared_u32(s_hist2, tx, T.uint32(0))
                bar_sync()
                for_each_score(
                    inp,
                    row_in,
                    tx,
                    row_len,
                    body_rehist_threshold_bin,
                    (s_hist2, threshold_bin, cfg),
                    cfg,
                )
                bar_sync()
                if tx == 0:
                    st_shared_u32(s_scal, SC_THRESH_BIN, T.uint32(0))
                    st_shared_u32(s_scal, SC_LAST_REMAIN, T.uint32(0))
                bar_sync()
                # The only RESET_NEXT_INPUT=false call in the kernel (:2733).
                update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, 0, False)
                for_each_score(
                    inp,
                    row_in,
                    tx,
                    row_len,
                    body_recollect_threshold_bin,
                    (s_scal, s_indices, threshold_bin, cfg),
                    cfg,
                )
                bar_sync()
                if cfg.det:
                    thr_f: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
                    eq_f: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_LAST_REMAIN))
                    collect_det_eq_pivot(
                        inp,
                        s_scan,
                        s_indices,
                        s_scal,
                        tx,
                        row_in,
                        row_len,
                        cfg,
                        T.bitwise_or(
                            T.shift_left(T.cast(threshold_bin, "uint32"), T.uint32(8)),
                            T.cast(thr_f, "uint32"),
                        ),
                        eq_f,
                    )
            else:
                run_refine_round(
                    inp,
                    s_hist2,
                    s_indices,
                    s_scal,
                    s_input,
                    tx,
                    row_in,
                    cfg,
                    topk,
                    resolved,
                    0,
                    cfg.first_shift,
                    True,
                )
                if cfg.det:
                    # build_det_pivot(0) on 16 bits (:2535-2537).
                    th0: T.int32 = T.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_TH))
                    collect_det_eq_pivot(
                        inp,
                        s_scan,
                        s_indices,
                        s_scal,
                        tx,
                        row_in,
                        row_len,
                        cfg,
                        T.bitwise_or(
                            T.shift_left(T.cast(threshold_bin, "uint32"), T.uint32(8)),
                            T.cast(th0, "uint32"),
                        ),
                        topk[0],
                    )
        else:
            det_stop = T.alloc_local((1,), "int32")
            stop = T.alloc_local((1,), "int32")
            remain = T.alloc_local((1,), "int32")
            stop_round = T.alloc_local((1,), "int32")
            halt = T.alloc_local((1,), "int32")
            # threshold_bytes: a per-thread register array in the source (:2805),
            # not shared state.
            bytes_reg = T.alloc_local((4,), "uint32")
            piv = T.alloc_local((1,), "uint32")
            pivf = T.alloc_local((1,), "uint32")
            emit_fp32_refine(
                inp,
                s_hist2,
                s_indices,
                s_scal,
                s_input,
                s_scan,
                tx,
                row_in,
                row_len,
                cfg,
                topk,
                resolved,
                det_stop,
                stop,
                remain,
                stop_round,
                halt,
                bytes_reg,
                piv,
                pivf,
                threshold_bin,
                topk_after_coarse,
            )

    # --- Stage 5: output (:2905-2918) --------------------------------------
    # Strict winners plus tie fillers sum to exactly top_k on this path, so
    # nothing is padded here; only the trivial path emits -1.
    for base in T.serial(tx, cfg.top_k, step=cfg.block, unroll=2):
        sel: T.int32 = T.reinterpret("int32", ld_shared_u32(s_indices, base))
        slot: T.int64 = row_out + T.cast(base, "int64")
        if cfg.basic:
            st_global_u32(out_idx, slot, T.reinterpret("uint32", sel))
            st_global_bits(
                out_val,
                slot,
                ld_global_nc_bits(inp, row_in + T.cast(sel, "int64"), cfg.is32),
                cfg.is32,
            )
        elif cfg.det:
            # Local index; the transform is deferred to the finalize kernel.
            st_global_u32(out_idx, slot, T.reinterpret("uint32", sel))
        elif cfg.page_table:
            st_global_u32(
                out_idx,
                slot,
                ld_global_nc_u32(
                    aux, T.cast(batch_idx, "int64") * aux_stride + T.cast(page_start + sel, "int64")
                ),
            )
        else:
            st_global_u32(out_idx, slot, T.reinterpret("uint32", sel + offset_val))
