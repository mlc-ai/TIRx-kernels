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

Everything here is a plain Python function emitting into the traced ``@K.kernel``
body that calls it: runtime control flow is spelled with ``K.If`` / ``K.Then`` /
``K.Else`` and ``K.serial`` / ``K.unroll``, a Python ``if`` or ``for`` is
compile-time expansion, and mutable per-thread state is a ``K.local_scalar`` or a
``K.alloc_local`` array written through ``K.assign``.  The per-element bodies the
row scan drives are passed as Python closures, which is what the C++ lambdas the
source hands ``for_each_score`` are.
"""

from typing import NamedTuple

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils.topk_radix import (
    atom_shared_add_u32,
    bar_sync,
    ld_shared_u32,
    st_global_u16,
    st_global_u32,
    st_shared_u32,
    warp_inclusive_sum_u32,
)

# Traits constants (:2279-2280, :2303-2304, :2323-2324).
NUM_REFINE_ROUNDS = {"float32": 4, "float16": 1, "bfloat16": 1}
FIRST_REFINE_SHIFT = {"float32": 24, "float16": 0, "bfloat16": 0}


def to_ordered_filtered_u32(bits):
    """``FilteredTopKTraits<float>::ToOrdered`` (``:2291-2294``).

    ``(bits & 0x80000000) ? ~bits : (bits | 0x80000000)``.
    """
    return K.Select(
        K.bitwise_and(bits, K.uint32(0x80000000)) != K.uint32(0),
        K.bitwise_xor(bits, K.uint32(0xFFFFFFFF)),
        K.bitwise_or(bits, K.uint32(0x80000000)),
    )


def to_ordered_filtered_u16(bits):
    """``FilteredTopKTraits<half|nv_bfloat16>::ToOrdered`` (``:2313-2316``, ``:2333-2336``)."""
    return K.Select(
        K.bitwise_and(bits, K.uint16(0x8000)) != K.uint16(0),
        K.bitwise_xor(bits, K.uint16(0xFFFF)),
        K.bitwise_or(bits, K.uint16(0x8000)),
    )


def to_coarse_key_u16(bits):
    """The shared tail of every ``ToCoarseKey``: monotone flip, then ``>> 8``.

    ``:2286-2288`` / ``:2308-2310`` / ``:2328-2330``.  The flip is the same one
    ``ToOrdered`` performs on 16 bits, so the coarse key is the ordered key's
    high byte.
    """
    return K.cast(K.shift_right(to_ordered_filtered_u16(bits), K.uint16(8)), "int32")


def to_coarse_key_f32(bits):
    """``FilteredTopKTraits<float>::ToCoarseKey`` (``:2282-2289``).

    Rounds through fp16 first (``__float2half_rn``, ``cvt.rn.f16.f32``), so the
    coarse key is **lossy** -- distinct floats can share a coarse bin.  That is
    fine because every phase re-derives it the same way, so the partition stays
    consistent; it is also why the refine rounds exist at all.
    """
    half_bits = K.reinterpret("uint16", K.cast(K.reinterpret("float32", bits), "float16"))
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
    out = K.local_scalar("uint32")
    K.ptx.ld.global_.nc.b32(out, buffer.ptr_to([index]))
    return out


def ld_global_nc_bits(buf, elem_index, is32):
    """One scalar element's raw bits: ``ld.global.nc.b32`` | ``ld.global.nc.b16``."""
    if is32:
        return ld_global_nc_u32(buf, elem_index)
    out16 = K.local_scalar("uint16")
    K.ptx.ld.global_.nc.b16(out16, buf.ptr_to([elem_index]))
    return out16


def ld_global_nc_words(buf, elem_index, load_bytes):
    """One vector load of ``load_bytes`` bytes, returned as 32-bit words."""
    if load_bytes == 16:
        w = K.alloc_local([4], "uint32", align=16)
        K.ptx["ld.global.nc.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index]))
        return [w[0], w[1], w[2], w[3]]
    if load_bytes == 8:
        w = K.alloc_local([2], "uint32", align=8)
        K.ptx["ld.global.nc.v2.b32"](w[0], w[1], buf.ptr_to([elem_index]))
        return [w[0], w[1]]
    w = K.alloc_local([1], "uint32")
    K.ptx.ld.global_.nc.b32(w[0], buf.ptr_to([elem_index]))
    return [w[0]]


def ld_global_nc_pair_u16(buffer, index):
    """``ld.global.nc.v2.b16``; both 16-bit lanes land in 16-bit registers.

    The source's ``vec_t<DType, 2>::cast_load`` keeps the monotone key flip on
    16-bit operands with no extract or repack arithmetic.
    """
    out = K.alloc_local([2], "uint16")
    K.ptx["ld.global.nc.v2.b16"](out[0], out[1], buffer.ptr_to([index]))
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
    out = K.local_scalar("uint32")
    K.ptx.atom.shared.or_.b32(out, buffer.ptr_to([index]), value)
    return out


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
    """Compile-time specialization carried into the emitters below.

    Every field is a Python value read at trace time: it selects which
    instructions are emitted, and never becomes part of the emitted IR.
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


def _emit_word_fanout(body, words, base, is32):
    """Hand each element of one vector load to the body, in index order."""
    if is32:
        for w in range(len(words)):
            body(words[w], base + w)
    else:
        # Each loaded word carries two 16-bit lanes, low lane first.
        for w in range(len(words)):
            body(K.cast(K.bitwise_and(words[w], K.uint32(0xFFFF)), "uint16"), base + 2 * w)
            body(K.cast(K.shift_right(words[w], K.uint32(16)), "uint16"), base + 2 * w + 1)


def for_each_score(inp, row_in, tx, row_len, body, cfg):
    """``for_each_score`` / ``for_each_score_full`` (``:2463-2481``).

    A ``#pragma unroll 2`` vector loop over ``aligned_length = length / VEC_SIZE *
    VEC_SIZE`` followed by a scalar tail.  ``body(raw_bits, index)`` is a Python
    closure, matching the C++ lambdas the source hands this helper.  Every load
    is ``.nc``-qualified because the kernel's inputs are ``__restrict__ const``.

    The kernel re-runs this in full on every phase that rescans the row, so the
    same load widths reappear in the histogram, the filter, and both fallbacks.
    """
    # A loop bound is re-evaluated by the loop condition on every iteration, so a
    # lazy one puts its whole expression inside the loop.  On Plain this folds to
    # a literal; on the transform modes `row_len` is a runtime read and the bound
    # would otherwise carry a shift and a multiply per trip.
    aligned = row_len // cfg.vec * cfg.vec
    if not isinstance(aligned, int):
        aligned = K.local_scalar("int32", init=aligned)
    with K.serial(tx * cfg.vec, aligned, step=cfg.block * cfg.vec, unroll=cfg.scan_unroll) as i:
        if cfg.vec == 1:
            body(ld_global_nc_bits(inp, row_in + K.cast(i, "int64"), cfg.is32), i)
        elif cfg.load_bytes == 4 and not cfg.is32:
            # VEC_SIZE == 2 on a 16-bit dtype: the source's native
            # ld.global.nc.v2.b16 pair, kept in 16-bit registers rather than
            # routed through one 32-bit word and unpacked.
            lo, hi = ld_global_nc_pair_u16(inp, row_in + K.cast(i, "int64"))
            body(lo, i)
            body(hi, i + 1)
        else:
            _emit_word_fanout(
                body,
                ld_global_nc_words(inp, row_in + K.cast(i, "int64"), cfg.load_bytes),
                i,
                cfg.is32,
            )
    # Scalar tail (:2477-2480); empty when VEC_SIZE divides the row length.
    with K.serial(aligned + tx, row_len, step=cfg.block) as j:
        body(ld_global_nc_bits(inp, row_in + K.cast(j, "int64"), cfg.is32), j)


def _backfill_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx):
    """The racing tie claim of ``collect_gt_and_nondet_eq_threshold`` (``:2573-2579``).

    Equal-to-threshold elements count ``s_last_remain`` **down** and write from
    the back of ``s_indices``, so which ties win is genuinely racy.
    """
    with K.If(value == threshold), K.Then():
        back = K.reinterpret(
            "int32", atom_shared_add_u32(s_scal, SC_LAST_REMAIN, K.uint32(0xFFFFFFFF))
        )
        with K.If(back > 0), K.Then():
            st_shared_u32(s_indices, cfg.top_k - back, K.reinterpret("uint32", idx))


def collect_gt_and_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx, allow_eq):
    """``collect_gt_and_nondet_eq_threshold`` (``:2567-2580``).

    Strict winners are appended at ``s_counter``.  Equal-to-threshold elements
    only matter when ``!DETERMINISTIC``; under ``DETERMINISTIC`` this ``else if
    constexpr`` branch does not exist at all and those ties are collected later
    by ``collect_det_eq_pivot``.

    ``allow_eq`` is the source's own compile-time flag at every call site but
    one, where it arrives as the runtime ``eq_needed > 0`` (``:2891``); a Python
    bool folds the arm away, anything else becomes a real guard.
    """
    with K.If(value > threshold):
        with K.Then():
            pos = K.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, K.uint32(1)))
            st_shared_u32(s_indices, pos, K.reinterpret("uint32", idx))
        if not cfg.det and allow_eq is not False:
            with K.Else():
                if allow_eq is True:
                    _backfill_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx)
                else:
                    with K.If(allow_eq), K.Then():
                        _backfill_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx)


def body_coarse_hist(s_hist2, cfg, bits, index):
    """``accumulate_coarse_hist`` (``:2482-2485``)."""
    atom_shared_add_u32(s_hist2, coarse_key(bits, cfg.is32), K.uint32(1))


def body_collect_coarse_gt(s_scal, s_indices, threshold_bin, cfg, bits, index):
    """``collect_coarse_gt`` on the coarse fast exit (``:2551-2557``)."""
    with K.If(coarse_key(bits, cfg.is32) > threshold_bin), K.Then():
        pos = K.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, K.uint32(1)))
        st_shared_u32(s_indices, pos, K.reinterpret("uint32", index))


def body_filter(s_hist2, s_scal, s_indices, s_input, threshold_bin, cfg, bits, index):
    """``filter_and_add_to_histogram`` (``:2611-2627``) -- the step the algorithm is named for.

    Strict winners go straight out; only threshold-bin candidates are compacted
    into ``s_input_idx[0]``, and each one also bumps the first refine byte's
    histogram.  Past ``SMEM_INPUT_SIZE`` the compacted buffer is truncated and
    unusable, so the overflow flag is raised and a fallback rebuilds from the
    row.  The source's ``__builtin_expect(pos < SMEM_INPUT_SIZE, 1)`` marks that
    arm cold.
    """
    # Compared against the threshold twice, once per element of the whole row.
    bin_id = K.local_scalar("int32", init=coarse_key(bits, cfg.is32))
    with K.If(bin_id > threshold_bin):
        with K.Then():
            pos = K.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, K.uint32(1)))
            st_shared_u32(s_indices, pos, K.reinterpret("uint32", index))
        with K.Else():
            with K.If(bin_id == threshold_bin), K.Then():
                slot = K.reinterpret(
                    "int32", atom_shared_add_u32(s_scal, SC_NUM_INPUT, K.uint32(1))
                )
                with K.If(slot < cfg.smem_input):
                    with K.Then():
                        st_shared_u32(s_input, slot, K.reinterpret("uint32", index))
                        sub = K.cast(
                            K.bitwise_and(
                                K.shift_right(
                                    K.cast(ordered_key(bits, cfg.is32), "uint32"),
                                    K.uint32(cfg.first_shift),
                                ),
                                K.uint32(0xFF),
                            ),
                            "int32",
                        )
                        atom_shared_add_u32(s_hist2, sub, K.uint32(1))
                    with K.Else():
                        atom_shared_or_b32(s_scal, SC_REFINE_OVERFLOW, K.uint32(1))


def run_cumsum(s_hist2, tx, cfg):
    """Hillis-Steele inclusive **suffix** scan, eight ping-pong steps (``:2490-2504``).

    Eight is even, so the result lands back in buffer 0, which is the alias the
    rest of the kernel reads as ``s_histogram``.  ``s_histogram[RADIX]`` must stay
    zero throughout: it is the exclusive-suffix sentinel every threshold test
    reads at ``tx + 1``, and the scan only ever writes indices below ``RADIX``.
    """
    with K.unroll(8) as i:
        with K.If(tx < cfg.radix), K.Then():
            j = K.shift_left(K.int32(1), i)
            src = K.bitwise_and(i, K.int32(1)) * cfg.hist_stride
            dst = K.bitwise_xor(K.bitwise_and(i, K.int32(1)), K.int32(1)) * cfg.hist_stride
            value = K.local_scalar(
                "int32", init=K.reinterpret("int32", ld_shared_u32(s_hist2, src + tx))
            )
            with K.If(tx < cfg.radix - j), K.Then():
                K.assign(
                    value, value + K.reinterpret("int32", ld_shared_u32(s_hist2, src + tx + j))
                )
            st_shared_u32(s_hist2, dst + tx, K.reinterpret("uint32", value))
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


def block_exclusive_sum_raking(out, total_out, s_scan, tx, value):
    """``BlockScan::ExclusiveSum`` with ``BLOCK_SCAN_RAKING_MEMOIZE``.

    Place into the padded raking grid; one warp serially reduces its 32-element
    segment while memoizing it in registers, a warp shuffle scan runs over the
    segment totals, then the same warp scatters the prefixes back.  ``out``
    receives this thread's exclusive prefix and ``total_out`` the block
    aggregate, which the contiguous collector needs for its quota walk.

    The shuffle scan runs inside ``tx < RAKING_THREADS``, which is cub's own
    shape and is warp-aligned: warp 0 enters whole, so the collective stays
    convergent and must not be hoisted out of the guard.
    """
    # Read by both the scatter and the gather of every scan instance.
    off = K.local_scalar("int32", init=raking_offset(tx))
    st_shared_u32(s_scan, off, value)
    bar_sync()
    with K.If(tx < RAKING_THREADS), K.Then():
        base = tx * RAKING_STRIDE
        cache = K.alloc_local([RAKING_SEGMENT], "uint32")
        total = K.local_scalar("uint32", init=K.uint32(0))
        with K.unroll(RAKING_SEGMENT) as j:
            K.assign(cache[j], ld_shared_u32(s_scan, base + j))
            K.assign(total, total + cache[j])
        incl = K.local_scalar("uint32", init=warp_inclusive_sum_u32(total, tx))
        run = K.local_scalar("uint32", init=incl - total)
        with K.unroll(RAKING_SEGMENT) as j2:
            st_shared_u32(s_scan, base + j2, run)
            K.assign(run, run + cache[j2])
        with K.If(tx == RAKING_THREADS - 1), K.Then():
            st_shared_u32(s_scan, RAKING_ELEMENTS, incl)
    bar_sync()
    K.assign(out, ld_shared_u32(s_scan, off))
    K.assign(total_out, ld_shared_u32(s_scan, RAKING_ELEMENTS))


def det_thread_strided_collect(inp, s_scan, s_indices, tx, row_in, row_len, cfg, pivot, eq_needed):
    """``DeterministicThreadStridedCollect`` (``:255-286``), the TIE_BREAK=None collector.

    Count matches per thread over a thread-strided walk, block-scan the counts,
    then re-walk and emit.  The emit target is
    ``s_indices[top_k - eq_needed + local_pos]`` (``:2587-2590``); the predicate
    is ``ToOrdered(score[idx]) == pivot``.
    """
    count = K.local_scalar("uint32", init=K.uint32(0))
    with K.serial(tx, row_len, step=cfg.block) as i:
        cur = K.cast(
            ordered_key(ld_global_nc_bits(inp, row_in + K.cast(i, "int64"), cfg.is32), cfg.is32),
            "uint32",
        )
        with K.If(cur == pivot), K.Then():
            K.assign(count, count + K.uint32(1))
    prefix = K.local_scalar("uint32")
    total = K.local_scalar("uint32")
    block_exclusive_sum_raking(prefix, total, s_scan, tx, count)
    with K.If(count > K.uint32(0)), K.Then():
        with K.If(prefix < K.cast(eq_needed, "uint32")), K.Then():
            pos = K.local_scalar("uint32", init=prefix)
            # Loop-invariant but read inside the walk below: a plain binding
            # would sink the min back into the loop body.
            end = K.local_scalar("uint32", init=K.min(prefix + count, K.cast(eq_needed, "uint32")))
            done = K.local_scalar("int32", init=K.int32(0))
            with K.serial(tx, row_len, step=cfg.block) as i2:
                with K.If(done == 0), K.Then():
                    cur2 = K.cast(
                        ordered_key(
                            ld_global_nc_bits(inp, row_in + K.cast(i2, "int64"), cfg.is32), cfg.is32
                        ),
                        "uint32",
                    )
                    with K.If(cur2 == pivot), K.Then():
                        st_shared_u32(
                            s_indices,
                            cfg.top_k - eq_needed + K.reinterpret("int32", pos),
                            K.reinterpret("uint32", i2),
                        )
                        K.assign(pos, pos + K.uint32(1))
                        with K.If(pos == end), K.Then():
                            K.assign(done, K.int32(1))
    bar_sync()


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
    with K.If(tx == 0), K.Then():
        st_shared_u32(s_scal, SC_EMITTED, K.uint32(0))
        st_shared_u32(s_scal, SC_CHUNK_BASE, K.uint32(0))
        st_shared_u32(s_scal, SC_CHUNK_TAKE, K.uint32(0))
    bar_sync()
    chunk_items = cfg.block * DET_ITEMS_PER_THREAD
    num_chunks = (row_len + chunk_items - 1) // chunk_items
    if not isinstance(num_chunks, int):
        num_chunks = K.local_scalar("int32", init=num_chunks)
    stop = K.local_scalar("int32", init=K.int32(0))
    with K.serial(0, num_chunks) as chunk:
        with K.If(stop == 0), K.Then():
            rows_of = K.alloc_local([DET_ITEMS_PER_THREAD], "int32")
            sel_of = K.alloc_local([DET_ITEMS_PER_THREAD], "uint32")
            cnt = K.local_scalar("uint32", init=K.uint32(0))
            with K.unroll(DET_ITEMS_PER_THREAD) as item:
                linear = K.local_scalar(
                    "int32", init=chunk * chunk_items + tx * DET_ITEMS_PER_THREAD + item
                )
                K.assign(rows_of[item], K.int32(0))
                K.assign(sel_of[item], K.uint32(0))
                with K.If(linear < row_len), K.Then():
                    if reverse:
                        K.assign(rows_of[item], row_len - 1 - linear)
                    else:
                        K.assign(rows_of[item], linear)
                    curc = K.cast(
                        ordered_key(
                            ld_global_nc_bits(
                                inp, row_in + K.cast(rows_of[item], "int64"), cfg.is32
                            ),
                            cfg.is32,
                        ),
                        "uint32",
                    )
                    with K.If(curc == pivot), K.Then():
                        K.assign(sel_of[item], K.uint32(1))
                        K.assign(cnt, cnt + K.uint32(1))
            prefix = K.local_scalar("uint32")
            blocksel = K.local_scalar("uint32")
            block_exclusive_sum_raking(prefix, blocksel, s_scan, tx, cnt)
            with K.If(tx == 0), K.Then():
                emitted = ld_shared_u32(s_scal, SC_EMITTED)
                st_shared_u32(s_scal, SC_CHUNK_BASE, emitted)
                remaining = K.local_scalar("uint32", init=K.uint32(0))
                with K.If(emitted < K.cast(eq_needed, "uint32")), K.Then():
                    K.assign(remaining, K.cast(eq_needed, "uint32") - emitted)
                take = K.min(remaining, blocksel)
                st_shared_u32(s_scal, SC_CHUNK_TAKE, take)
                st_shared_u32(s_scal, SC_EMITTED, emitted + take)
            bar_sync()
            chunk_take = ld_shared_u32(s_scal, SC_CHUNK_TAKE)
            chunk_base = ld_shared_u32(s_scal, SC_CHUNK_BASE)
            with K.If(cnt > K.uint32(0)), K.Then():
                with K.If(prefix < chunk_take), K.Then():
                    epos = K.local_scalar("uint32", init=prefix)
                    # Same: invariant across the item walk that reads it.
                    eend = K.local_scalar("uint32", init=K.min(prefix + cnt, chunk_take))
                    fin = K.local_scalar("int32", init=K.int32(0))
                    with K.unroll(DET_ITEMS_PER_THREAD) as item2:
                        with K.If(fin == 0), K.Then():
                            with K.If(sel_of[item2] == K.uint32(1)), K.Then():
                                st_shared_u32(
                                    s_indices,
                                    cfg.top_k
                                    - eq_needed
                                    + K.reinterpret("int32", chunk_base + epos),
                                    K.reinterpret("uint32", rows_of[item2]),
                                )
                                K.assign(epos, epos + K.uint32(1))
                                with K.If(epos == eend), K.Then():
                                    K.assign(fin, K.int32(1))
            bar_sync()
            with K.If(ld_shared_u32(s_scal, SC_EMITTED) >= K.cast(eq_needed, "uint32")), K.Then():
                K.assign(stop, K.int32(1))
    bar_sync()


def collect_det_eq_pivot(
    inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivot, eq_needed
):
    """``collect_det_eq_pivot`` (``:2582-2608``).

    Picks the collector by ``TIE_BREAK``: contiguous ascending for ``Small``,
    contiguous descending for ``Large``, thread-strided for plain determinism.
    All three share one BlockScan instance and the same
    ``ToOrdered(score[idx]) == pivot`` predicate.
    """
    with K.If(eq_needed > 0), K.Then():
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


def update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, next_idx, reset_next):
    """``update_refine_threshold`` (``:2505-2516``).

    ``run_cumsum`` plus the same predicated pick as the first one, except that it
    also publishes ``s_last_remain`` and never touches ``s_counter`` -- only the
    first pick (``:2522``) resets that.  ``RESET_NEXT_INPUT`` is false at exactly
    one call site, the 16-bit overflow fallback (``:2733``).
    """
    run_cumsum(s_hist2, tx, cfg)
    with K.If(tx < cfg.radix), K.Then():
        cur = K.reinterpret("int32", ld_shared_u32(s_hist2, tx))
        with K.If(cur > topk), K.Then():
            nxt = K.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
            with K.If(nxt <= topk), K.Then():
                st_shared_u32(s_scal, SC_THRESH_BIN, K.reinterpret("uint32", tx))
                if reset_next:
                    st_shared_u32(s_scal, SC_NUM_INPUT + next_idx, K.uint32(0))
                st_shared_u32(s_scal, SC_LAST_REMAIN, K.reinterpret("uint32", topk - nxt))
    bar_sync()


def _refine_bin(inp, row_in, idx, offset, cfg):
    """One candidate's byte at ``offset`` of its ordered key (``:2679``, ``:2637``)."""
    return K.cast(
        K.bitwise_and(
            K.shift_right(
                K.cast(
                    ordered_key(
                        ld_global_nc_bits(inp, row_in + K.cast(idx, "int64"), cfg.is32), cfg.is32
                    ),
                    "uint32",
                ),
                K.uint32(offset),
            ),
            K.uint32(0xFF),
        ),
        "int32",
    )


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
    ``resolved`` when the round fully resolves the pivot.
    """
    raw = K.reinterpret("int32", ld_shared_u32(s_scal, SC_NUM_INPUT + r_idx))
    num_input = K.local_scalar("int32", init=K.min(raw, cfg.smem_input))

    update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, r_idx ^ 1, True)

    threshold = K.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
    if cfg.det:
        with K.If(tx == 0), K.Then():
            st_shared_u32(
                s_scal,
                SC_REFINE_TH + (cfg.first_shift - offset) // 8,
                K.reinterpret("uint32", threshold),
            )
    K.assign(topk, topk - K.reinterpret("int32", ld_shared_u32(s_hist2, threshold + 1)))
    with K.If(topk == 0):
        with K.Then():
            # Pivot resolved: only bins strictly greater than the threshold remain.
            with K.serial(tx, num_input, step=cfg.block) as i:
                idx = K.reinterpret("int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i))
                with K.If(_refine_bin(inp, row_in, idx, offset, cfg) > threshold), K.Then():
                    pos = K.reinterpret(
                        "int32", atom_shared_add_u32(s_scal, SC_COUNTER, K.uint32(1))
                    )
                    st_shared_u32(s_indices, pos, K.reinterpret("uint32", idx))
            bar_sync()
            K.assign(resolved, K.int32(1))
        with K.Else():
            if is_last:
                # collect_with_threshold_last_round (:2635-2645): one barrier.
                with K.serial(tx, num_input, step=cfg.block) as i2:
                    idx2 = K.reinterpret(
                        "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i2)
                    )
                    bin2 = K.local_scalar("int32", init=_refine_bin(inp, row_in, idx2, offset, cfg))
                    collect_gt_and_nondet_eq(s_scal, s_indices, cfg, bin2, threshold, idx2, True)
                bar_sync()
            else:
                # collect_with_threshold_non_last_round (:2646-2672): three
                # barriers, ping-ponging the survivors into s_input_idx[r_idx ^ 1]
                # together with the next byte's histogram.
                bar_sync()
                with K.If(tx < cfg.radix + 1), K.Then():
                    st_shared_u32(s_hist2, tx, K.uint32(0))
                bar_sync()
                with K.serial(tx, num_input, step=cfg.block) as i3:
                    idx3 = K.reinterpret(
                        "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i3)
                    )
                    ord3 = K.cast(
                        ordered_key(
                            ld_global_nc_bits(inp, row_in + K.cast(idx3, "int64"), cfg.is32),
                            cfg.is32,
                        ),
                        "uint32",
                    )
                    bin3 = K.cast(
                        K.bitwise_and(K.shift_right(ord3, K.uint32(offset)), K.uint32(0xFF)),
                        "int32",
                    )
                    with K.If(bin3 > threshold):
                        with K.Then():
                            pos3 = K.reinterpret(
                                "int32", atom_shared_add_u32(s_scal, SC_COUNTER, K.uint32(1))
                            )
                            st_shared_u32(s_indices, pos3, K.reinterpret("uint32", idx3))
                        with K.Else():
                            with K.If(bin3 == threshold), K.Then():
                                slot3 = K.reinterpret(
                                    "int32",
                                    atom_shared_add_u32(
                                        s_scal, SC_NUM_INPUT + (r_idx ^ 1), K.uint32(1)
                                    ),
                                )
                                with K.If(slot3 < cfg.smem_input):
                                    with K.Then():
                                        st_shared_u32(
                                            s_input,
                                            (r_idx ^ 1) * cfg.smem_input + slot3,
                                            K.reinterpret("uint32", idx3),
                                        )
                                        sub3 = K.cast(
                                            K.bitwise_and(
                                                K.shift_right(ord3, K.uint32(offset - 8)),
                                                K.uint32(0xFF),
                                            ),
                                            "int32",
                                        )
                                        atom_shared_add_u32(s_hist2, sub3, K.uint32(1))
                                    with K.Else():
                                        atom_shared_or_b32(s_scal, SC_REFINE_OVERFLOW, K.uint32(1))
                bar_sync()


def body_rehist_threshold_bin(s_hist2, threshold_bin, cfg, bits, index):
    """16-bit fallback re-histogram (``:2715-2724``): low byte, threshold bin only."""
    with K.If(coarse_key(bits, cfg.is32) == threshold_bin), K.Then():
        atom_shared_add_u32(
            s_hist2,
            K.cast(
                K.bitwise_and(K.cast(ordered_key(bits, cfg.is32), "uint32"), K.uint32(0xFF)),
                "int32",
            ),
            K.uint32(1),
        )


def body_recollect_threshold_bin(s_scal, s_indices, threshold_bin, cfg, bits, index):
    """16-bit fallback re-collect (``:2740-2748``).

    The ``coarse_bin != threshold_bin`` guard at ``:2741-2744`` is load-bearing:
    without it the pass would compare out-of-bin elements by their low byte and
    re-collect every strict winner the filter stage already appended.
    """
    with K.If(coarse_key(bits, cfg.is32) == threshold_bin), K.Then():
        sub = K.local_scalar(
            "int32",
            init=K.cast(
                K.bitwise_and(K.cast(ordered_key(bits, cfg.is32), "uint32"), K.uint32(0xFF)),
                "int32",
            ),
        )
        threshold = K.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
        collect_gt_and_nondet_eq(s_scal, s_indices, cfg, sub, threshold, index, True)


def _prefix_match(match, ordered, threshold_bytes, rnd):
    """Compare every byte fixed by an earlier round (``:2823-2831``).

    A Python loop over the compile-time round index, so it unrolls into the
    source's straight-line chain of byte compares.
    """
    for prev in range(rnd):
        got = K.cast(
            K.bitwise_and(K.shift_right(ordered, K.uint32(24 - prev * 8)), K.uint32(0xFF)), "int32"
        )
        with K.If(got != K.cast(threshold_bytes[prev], "int32")), K.Then():
            K.assign(match, K.int32(0))


def body_fallback_rehist(s_hist2, threshold_bytes, threshold_bin, cfg, rnd, bits, index):
    """fp32 fallback per-round re-histogram (``:2819-2837``).

    Only threshold-bin elements whose ordered key still matches the bytes fixed
    by earlier rounds contribute to this round's byte histogram.
    ``threshold_bytes`` is the per-thread register array of ``:2805``.
    """
    with K.If(coarse_key(bits, cfg.is32) == threshold_bin), K.Then():
        # Read once per already-fixed byte plus once for this round's bump, on
        # every element of the row, on each of the four rebuild rounds.
        ordered = K.local_scalar("uint32", init=K.cast(ordered_key(bits, cfg.is32), "uint32"))
        match = K.local_scalar("int32", init=K.int32(1))
        _prefix_match(match, ordered, threshold_bytes, rnd)
        with K.If(match == 1), K.Then():
            atom_shared_add_u32(
                s_hist2,
                K.cast(
                    K.bitwise_and(K.shift_right(ordered, K.uint32(24 - rnd * 8)), K.uint32(0xFF)),
                    "int32",
                ),
                K.uint32(1),
            )


def body_collect_by_pivot(s_scal, s_indices, threshold_bin, cfg, pivot, eq_needed, bits, index):
    """fp32 fallback re-collect (``:2883-2895``) -- a three-way dispatch.

    Coarse-bin winners are compared on the **coarse** key with no eq claim;
    out-of-bin elements are dropped; threshold-bin elements are compared on the
    full 32-bit ordered key against the rebuilt pivot.
    """
    # Both keys are read twice by the collector they are handed to.
    bin_id = K.local_scalar("int32", init=coarse_key(bits, cfg.is32))
    with K.If(bin_id > threshold_bin):
        with K.Then():
            collect_gt_and_nondet_eq(s_scal, s_indices, cfg, bin_id, threshold_bin, index, False)
        with K.Else():
            with K.If(bin_id == threshold_bin), K.Then():
                ordered = K.local_scalar(
                    "uint32", init=K.cast(ordered_key(bits, cfg.is32), "uint32")
                )
                collect_gt_and_nondet_eq(
                    s_scal, s_indices, cfg, ordered, pivot, index, eq_needed > 0
                )


def _fp32_refine_rounds(
    inp, s_hist2, s_indices, s_scal, s_input, tx, row_in, cfg, topk, resolved, stop, det_stop
):
    """The source's ``#pragma unroll`` refine loop (``:2774-2790``).

    A Python loop so the ping-pong index, the byte offset and ``IS_LAST_ROUND``
    stay compile-time and the four rounds are emitted as four distinct bodies.
    """
    for rnd in range(4):
        with K.If(stop == 0), K.Then():
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
            with K.If(resolved == 1):
                with K.Then():
                    K.assign(det_stop, K.int32(rnd))
                    K.assign(stop, K.int32(1))
                with K.Else():
                    with (
                        K.If(
                            K.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0
                        ),
                        K.Then(),
                    ):
                        K.assign(stop, K.int32(1))


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
    """The fallback's ``#pragma unroll`` rebuild loop (``:2812-2856``)."""
    for rnd in range(4):
        with K.If(halt == 0), K.Then():
            with K.If(tx < cfg.radix + 1), K.Then():
                st_shared_u32(s_hist2, tx, K.uint32(0))
            bar_sync()
            for_each_score(
                inp,
                row_in,
                tx,
                row_len,
                lambda bits, index, r=rnd: body_fallback_rehist(
                    s_hist2, bytes_reg, threshold_bin, cfg, r, bits, index
                ),
                cfg,
            )
            bar_sync()
            run_cumsum(s_hist2, tx, cfg)
            with K.If(tx < cfg.radix), K.Then():
                curf = K.reinterpret("int32", ld_shared_u32(s_hist2, tx))
                with K.If(curf > remain), K.Then():
                    nxtf = K.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
                    with K.If(nxtf <= remain), K.Then():
                        # Only the bin id here; s_num_input and s_last_remain stay
                        # untouched (:2842-2845).
                        st_shared_u32(s_scal, SC_THRESH_BIN, K.reinterpret("uint32", tx))
            bar_sync()
            thrf = K.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
            # threshold_bytes is a per-thread register array in the source (:2805);
            # every thread reads the same s_threshold_bin_id, so no publication step
            # and no extra barrier is needed.
            K.assign(bytes_reg[rnd], K.cast(thrf, "uint32"))
            K.assign(remain, remain - K.reinterpret("int32", ld_shared_u32(s_hist2, thrf + 1)))
            bar_sync()
            with K.If(remain == 0), K.Then():
                K.assign(stop_round, K.int32(rnd))
                K.assign(halt, K.int32(1))


def _fp32_det_pivot_bytes(s_scal, piv, det_stop):
    """``build_det_pivot`` (``:2539-2544``), one byte per refine round."""
    for rnd in range(4):
        byte0 = K.Select(
            K.int32(rnd) <= det_stop, ld_shared_u32(s_scal, SC_REFINE_TH + rnd), K.uint32(0xFF)
        )
        K.assign(piv, K.bitwise_or(piv, K.shift_left(byte0, K.uint32(24 - rnd * 8))))


def _fp32_pivot_bytes(pivf, bytes_reg, remain, stop_round):
    """The fallback pivot (``:2860-2867``), one byte per rebuild round.

    A byte is forced to ``0xFF`` only once the quota is met and the round is past
    ``stop_round``.
    """
    for rnd in range(4):
        bytef = K.local_scalar("uint32", init=bytes_reg[rnd])
        with K.If(remain == 0), K.Then():
            with K.If(K.int32(rnd) > stop_round), K.Then():
                K.assign(bytef, K.uint32(0xFF))
        K.assign(pivf, K.bitwise_or(pivf, K.shift_left(bytef, K.uint32(24 - rnd * 8))))


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
    threshold_bin,
    topk_after_coarse,
):
    """fp32's four refine rounds and the 32-bit pivot-rebuild fallback (``:2767-2902``)."""
    det_stop = K.local_scalar("int32", init=K.int32(3))  # NUM_ROUNDS - 1 (:2771)
    stop = K.local_scalar("int32", init=K.int32(0))
    remain = K.local_scalar("int32", init=topk_after_coarse)  # (:2804)
    stop_round = K.local_scalar("int32", init=K.int32(3))
    halt = K.local_scalar("int32", init=K.int32(0))
    # threshold_bytes: a per-thread register array in the source (:2805), not
    # shared state.
    bytes_reg = K.alloc_local([4], "uint32")
    with K.unroll(4) as r:
        K.assign(bytes_reg[r], K.uint32(0xFF))  # (:2805-2810)

    # The whole round loop is guarded on the flag the filter stage may have set
    # (:2772).
    with K.If(K.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) == 0), K.Then():
        _fp32_refine_rounds(
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

    # The deterministic collect sits OUTSIDE that guard and re-checks the flag
    # itself: run_refine_round can raise s_refine_overflow mid-loop through the
    # atomicOr at :2667, which the source spells out at :2798-2799.
    if cfg.det:
        with K.If(K.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) == 0), K.Then():
            piv = K.local_scalar("uint32", init=K.uint32(0))
            _fp32_det_pivot_bytes(s_scal, piv, det_stop)
            collect_det_eq_pivot(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, piv, topk
            )

    # 32-bit pivot rebuild after an overflow (:2800-2900).  Overflow can follow
    # partial writes to s_indices / s_counter, so the selection is rebuilt from
    # scratch.
    with K.If(K.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0), K.Then():
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
        pivf = K.local_scalar("uint32", init=K.uint32(0))
        _fp32_pivot_bytes(pivf, bytes_reg, remain, stop_round)
        with K.If(tx == 0), K.Then():
            st_shared_u32(s_scal, SC_COUNTER, K.uint32(0))
            st_shared_u32(s_scal, SC_LAST_REMAIN, K.reinterpret("uint32", remain))
        bar_sync()
        for_each_score(
            inp,
            row_in,
            tx,
            row_len,
            lambda bits, index: body_collect_by_pivot(
                s_scal, s_indices, threshold_bin, cfg, pivf, remain, bits, index
            ),
            cfg,
        )
        bar_sync()
        if cfg.det:
            collect_det_eq_pivot(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, pivf, remain
            )


def _emit_16bit_refine(
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
    threshold_bin,
):
    """The 16-bit dtypes' single refine round and its full-row slow path (``:2710-2765``)."""
    with K.If(K.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0):
        with K.Then():
            with K.If(tx < cfg.radix + 1), K.Then():
                st_shared_u32(s_hist2, tx, K.uint32(0))
            bar_sync()
            for_each_score(
                inp,
                row_in,
                tx,
                row_len,
                lambda bits, index: body_rehist_threshold_bin(
                    s_hist2, threshold_bin, cfg, bits, index
                ),
                cfg,
            )
            bar_sync()
            with K.If(tx == 0), K.Then():
                st_shared_u32(s_scal, SC_THRESH_BIN, K.uint32(0))
                st_shared_u32(s_scal, SC_LAST_REMAIN, K.uint32(0))
            bar_sync()
            # The only RESET_NEXT_INPUT=false call in the kernel (:2733).
            update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, 0, False)
            for_each_score(
                inp,
                row_in,
                tx,
                row_len,
                lambda bits, index: body_recollect_threshold_bin(
                    s_scal, s_indices, threshold_bin, cfg, bits, index
                ),
                cfg,
            )
            bar_sync()
            if cfg.det:
                thr_f = K.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
                eq_f = K.reinterpret("int32", ld_shared_u32(s_scal, SC_LAST_REMAIN))
                collect_det_eq_pivot(
                    inp,
                    s_scan,
                    s_indices,
                    s_scal,
                    tx,
                    row_in,
                    row_len,
                    cfg,
                    K.bitwise_or(
                        K.shift_left(K.cast(threshold_bin, "uint32"), K.uint32(8)),
                        K.cast(thr_f, "uint32"),
                    ),
                    eq_f,
                )
        with K.Else():
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
                th0 = K.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_TH))
                collect_det_eq_pivot(
                    inp,
                    s_scan,
                    s_indices,
                    s_scal,
                    tx,
                    row_in,
                    row_len,
                    cfg,
                    K.bitwise_or(
                        K.shift_left(K.cast(threshold_bin, "uint32"), K.uint32(8)),
                        K.cast(th0, "uint32"),
                    ),
                    topk,
                )


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
    topk = K.local_scalar("int32", init=K.int32(cfg.top_k))

    # --- init (:2450-2458) -------------------------------------------------
    with K.If(tx == 0), K.Then():
        st_shared_u32(s_scal, SC_REFINE_OVERFLOW, K.uint32(0))
    if cfg.det:
        with K.If(tx < 4), K.Then():
            st_shared_u32(s_scal, SC_REFINE_TH + tx, K.uint32(0xFF))
    with K.If(tx < cfg.radix + 1), K.Then():
        st_shared_u32(s_hist2, tx, K.uint32(0))
    bar_sync()

    # --- Stage 1: coarse histogram over the whole row (:2482-2487) ---------
    for_each_score(
        inp,
        row_in,
        tx,
        row_len,
        lambda bits, index: body_coarse_hist(s_hist2, cfg, bits, index),
        cfg,
    )
    bar_sync()

    # --- first threshold pick (:2518-2524) ---------------------------------
    # Three short-circuit guards, not a fused predicate: the export shows three
    # `@p bra` and zero `and.pred`.  Exactly one thread satisfies all three, and
    # the counter resets ride on that same predicate rather than on tx == 0.
    run_cumsum(s_hist2, tx, cfg)
    with K.If(tx < cfg.radix), K.Then():
        cur0 = K.reinterpret("int32", ld_shared_u32(s_hist2, tx))
        with K.If(cur0 > topk), K.Then():
            nxt0 = K.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
            with K.If(nxt0 <= topk), K.Then():
                st_shared_u32(s_scal, SC_THRESH_BIN, K.reinterpret("uint32", tx))
                st_shared_u32(s_scal, SC_NUM_INPUT, K.uint32(0))
                st_shared_u32(s_scal, SC_COUNTER, K.uint32(0))
    bar_sync()
    threshold_bin = K.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
    K.assign(topk, topk - K.reinterpret("int32", ld_shared_u32(s_hist2, threshold_bin + 1)))
    # A SNAPSHOT of `topk` before the refine rounds walk it down (:2804); a plain
    # binding would re-read the counter after those writes.
    topk_after_coarse = K.local_scalar("int32", init=topk)

    with K.If(topk == 0):
        with K.Then():
            # The coarse pass already resolved k (:2549-2559).
            for_each_score(
                inp,
                row_in,
                tx,
                row_len,
                lambda bits, index: body_collect_coarse_gt(
                    s_scal, s_indices, threshold_bin, cfg, bits, index
                ),
                cfg,
            )
            bar_sync()
        with K.Else():
            # --- Stage 2: the filter (:2561-2629) --------------------------
            bar_sync()
            with K.If(tx < cfg.radix + 1), K.Then():
                st_shared_u32(s_hist2, tx, K.uint32(0))
            bar_sync()
            for_each_score(
                inp,
                row_in,
                tx,
                row_len,
                lambda bits, index: body_filter(
                    s_hist2, s_scal, s_indices, s_input, threshold_bin, cfg, bits, index
                ),
                cfg,
            )
            bar_sync()

            # --- Stage 3: refine (:2710-2902) ------------------------------
            resolved = K.local_scalar("int32", init=K.int32(0))
            if cfg.num_rounds == 1:
                _emit_16bit_refine(
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
                    threshold_bin,
                )
            else:
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
                    threshold_bin,
                    topk_after_coarse,
                )

    # --- Stage 5: output (:2905-2918) --------------------------------------
    # Strict winners plus tie fillers sum to exactly top_k on this path, so
    # nothing is padded here; only the trivial path emits -1.
    with K.serial(tx, cfg.top_k, step=cfg.block, unroll=2) as base:
        sel = K.reinterpret("int32", ld_shared_u32(s_indices, base))
        # Materialized: a lazy 64-bit slot address is re-narrowed per store.
        slot = K.local_scalar("int64", init=row_out + K.cast(base, "int64"))
        if cfg.basic:
            st_global_u32(out_idx, slot, K.reinterpret("uint32", sel))
            st_global_bits(
                out_val,
                slot,
                ld_global_nc_bits(inp, row_in + K.cast(sel, "int64"), cfg.is32),
                cfg.is32,
            )
        elif cfg.det:
            # Local index; the transform is deferred to the finalize kernel.
            st_global_u32(out_idx, slot, K.reinterpret("uint32", sel))
        elif cfg.page_table:
            st_global_u32(
                out_idx,
                slot,
                ld_global_nc_u32(
                    aux, K.cast(batch_idx, "int64") * aux_stride + K.cast(page_start + sel, "int64")
                ),
            )
        else:
            st_global_u32(out_idx, slot, K.reinterpret("uint32", sel + offset_val))
