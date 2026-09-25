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

Everything here is a plain Python function emitting into the traced ``@txl.kernel``
body that calls it: runtime control flow is spelled with ``txl.If`` / ``txl.Then`` /
``txl.Else`` and ``txl.serial`` / ``txl.unroll``, a Python ``if`` or ``for`` is
compile-time expansion, and mutable per-thread state is a ``txl.local_scalar`` or a
``txl.alloc_local`` array written through ``txl.assign``.  The per-element bodies the
row scan drives are passed as Python closures, which is what the C++ lambdas the
source hands ``for_each_score`` are.
"""

from typing import NamedTuple

import tirx_kernels.tirx_lite as txl
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
    return txl.Select(
        txl.bitwise_and(bits, txl.uint32(0x80000000)) != txl.uint32(0),
        txl.bitwise_xor(bits, txl.uint32(0xFFFFFFFF)),
        txl.bitwise_or(bits, txl.uint32(0x80000000)),
    )


def to_ordered_filtered_u16(bits):
    """``FilteredTopKTraits<half|nv_bfloat16>::ToOrdered`` (``:2313-2316``, ``:2333-2336``)."""
    return txl.Select(
        txl.bitwise_and(bits, txl.uint16(0x8000)) != txl.uint16(0),
        txl.bitwise_xor(bits, txl.uint16(0xFFFF)),
        txl.bitwise_or(bits, txl.uint16(0x8000)),
    )


def to_coarse_key_u16(bits):
    """The shared tail of every ``ToCoarseKey``: monotone flip, then ``>> 8``.

    ``:2286-2288`` / ``:2308-2310`` / ``:2328-2330``.  The flip is the same one
    ``ToOrdered`` performs on 16 bits, so the coarse key is the ordered key's
    high byte.
    """
    return txl.cast(txl.shift_right(to_ordered_filtered_u16(bits), txl.uint16(8)), "int32")


def to_coarse_key_f32(bits):
    """``FilteredTopKTraits<float>::ToCoarseKey`` (``:2282-2289``).

    Rounds through fp16 first (``__float2half_rn``, ``cvt.rn.f16.f32``), so the
    coarse key is **lossy** -- distinct floats can share a coarse bin.  That is
    fine because every phase re-derives it the same way, so the partition stays
    consistent; it is also why the refine rounds exist at all.
    """
    half_bits = txl.reinterpret("uint16", txl.cast(txl.reinterpret("float32", bits), "float16"))
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
    out = txl.local_scalar("uint32")
    txl.ptx.ld.global_.nc.b32(out, buffer.ptr_to([index]))
    return out


def ld_global_nc_bits(buf, elem_index, is32):
    """One scalar element's raw bits: ``ld.global.nc.b32`` | ``ld.global.nc.b16``."""
    if is32:
        return ld_global_nc_u32(buf, elem_index)
    out16 = txl.local_scalar("uint16")
    txl.ptx.ld.global_.nc.b16(out16, buf.ptr_to([elem_index]))
    return out16


def ld_global_nc_words(buf, elem_index, load_bytes):
    """One vector load of ``load_bytes`` bytes, returned as 32-bit words."""
    if load_bytes == 16:
        w = txl.alloc_local([4], "uint32", align=16)
        txl.ptx["ld.global.nc.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index]))
        return [w[0], w[1], w[2], w[3]]
    if load_bytes == 8:
        w = txl.alloc_local([2], "uint32", align=8)
        txl.ptx["ld.global.nc.v2.b32"](w[0], w[1], buf.ptr_to([elem_index]))
        return [w[0], w[1]]
    w = txl.alloc_local([1], "uint32")
    txl.ptx.ld.global_.nc.b32(w[0], buf.ptr_to([elem_index]))
    return [w[0]]


def ld_global_nc_pair_u16(buffer, index):
    """``ld.global.nc.v2.b16``; both 16-bit lanes land in 16-bit registers.

    The source's ``vec_t<DType, 2>::cast_load`` keeps the monotone key flip on
    16-bit operands with no extract or repack arithmetic.
    """
    out = txl.alloc_local([2], "uint16")
    txl.ptx["ld.global.nc.v2.b16"](out[0], out[1], buffer.ptr_to([index]))
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
    out = txl.local_scalar("uint32")
    txl.ptx.atom.shared.or_.b32(out, buffer.ptr_to([index]), value)
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
            body(txl.cast(txl.bitwise_and(words[w], txl.uint32(0xFFFF)), "uint16"), base + 2 * w)
            body(txl.cast(txl.shift_right(words[w], txl.uint32(16)), "uint16"), base + 2 * w + 1)


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
        aligned = txl.local_scalar("int32", init=aligned)
    with txl.serial(tx * cfg.vec, aligned, step=cfg.block * cfg.vec, unroll=cfg.scan_unroll) as i:
        if cfg.vec == 1:
            body(ld_global_nc_bits(inp, row_in + txl.cast(i, "int64"), cfg.is32), i)
        elif cfg.load_bytes == 4 and not cfg.is32:
            # VEC_SIZE == 2 on a 16-bit dtype: the source's native
            # ld.global.nc.v2.b16 pair, kept in 16-bit registers rather than
            # routed through one 32-bit word and unpacked.
            lo, hi = ld_global_nc_pair_u16(inp, row_in + txl.cast(i, "int64"))
            body(lo, i)
            body(hi, i + 1)
        else:
            _emit_word_fanout(
                body,
                ld_global_nc_words(inp, row_in + txl.cast(i, "int64"), cfg.load_bytes),
                i,
                cfg.is32,
            )
    # Scalar tail (:2477-2480); empty when VEC_SIZE divides the row length.
    with txl.serial(aligned + tx, row_len, step=cfg.block) as j:
        body(ld_global_nc_bits(inp, row_in + txl.cast(j, "int64"), cfg.is32), j)


def _backfill_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx):
    """The racing tie claim of ``collect_gt_and_nondet_eq_threshold`` (``:2573-2579``).

    Equal-to-threshold elements count ``s_last_remain`` **down** and write from
    the back of ``s_indices``, so which ties win is genuinely racy.
    """
    with txl.If(value == threshold), txl.Then():
        back = txl.reinterpret(
            "int32", atom_shared_add_u32(s_scal, SC_LAST_REMAIN, txl.uint32(0xFFFFFFFF))
        )
        with txl.If(back > 0), txl.Then():
            st_shared_u32(s_indices, cfg.top_k - back, txl.reinterpret("uint32", idx))


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
    with txl.If(value > threshold):
        with txl.Then():
            pos = txl.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, txl.uint32(1)))
            st_shared_u32(s_indices, pos, txl.reinterpret("uint32", idx))
        if not cfg.det and allow_eq is not False:
            with txl.Else():
                if allow_eq is True:
                    _backfill_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx)
                else:
                    with txl.If(allow_eq), txl.Then():
                        _backfill_nondet_eq(s_scal, s_indices, cfg, value, threshold, idx)


def body_coarse_hist(s_hist2, cfg, bits, index):
    """``accumulate_coarse_hist`` (``:2482-2485``)."""
    atom_shared_add_u32(s_hist2, coarse_key(bits, cfg.is32), txl.uint32(1))


def body_collect_coarse_gt(s_scal, s_indices, threshold_bin, cfg, bits, index):
    """``collect_coarse_gt`` on the coarse fast exit (``:2551-2557``)."""
    with txl.If(coarse_key(bits, cfg.is32) > threshold_bin), txl.Then():
        pos = txl.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, txl.uint32(1)))
        st_shared_u32(s_indices, pos, txl.reinterpret("uint32", index))


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
    bin_id = txl.local_scalar("int32", init=coarse_key(bits, cfg.is32))
    with txl.If(bin_id > threshold_bin):
        with txl.Then():
            pos = txl.reinterpret("int32", atom_shared_add_u32(s_scal, SC_COUNTER, txl.uint32(1)))
            st_shared_u32(s_indices, pos, txl.reinterpret("uint32", index))
        with txl.Else():
            with txl.If(bin_id == threshold_bin), txl.Then():
                slot = txl.reinterpret(
                    "int32", atom_shared_add_u32(s_scal, SC_NUM_INPUT, txl.uint32(1))
                )
                with txl.If(slot < cfg.smem_input):
                    with txl.Then():
                        st_shared_u32(s_input, slot, txl.reinterpret("uint32", index))
                        sub = txl.cast(
                            txl.bitwise_and(
                                txl.shift_right(
                                    txl.cast(ordered_key(bits, cfg.is32), "uint32"),
                                    txl.uint32(cfg.first_shift),
                                ),
                                txl.uint32(0xFF),
                            ),
                            "int32",
                        )
                        atom_shared_add_u32(s_hist2, sub, txl.uint32(1))
                    with txl.Else():
                        atom_shared_or_b32(s_scal, SC_REFINE_OVERFLOW, txl.uint32(1))


def run_cumsum(s_hist2, tx, cfg):
    """Hillis-Steele inclusive **suffix** scan, eight ping-pong steps (``:2490-2504``).

    Eight is even, so the result lands back in buffer 0, which is the alias the
    rest of the kernel reads as ``s_histogram``.  ``s_histogram[RADIX]`` must stay
    zero throughout: it is the exclusive-suffix sentinel every threshold test
    reads at ``tx + 1``, and the scan only ever writes indices below ``RADIX``.
    """
    with txl.unroll(8) as i:
        with txl.If(tx < cfg.radix), txl.Then():
            j = txl.shift_left(txl.int32(1), i)
            src = txl.bitwise_and(i, txl.int32(1)) * cfg.hist_stride
            dst = txl.bitwise_xor(txl.bitwise_and(i, txl.int32(1)), txl.int32(1)) * cfg.hist_stride
            value = txl.local_scalar(
                "int32", init=txl.reinterpret("int32", ld_shared_u32(s_hist2, src + tx))
            )
            with txl.If(tx < cfg.radix - j), txl.Then():
                txl.assign(
                    value, value + txl.reinterpret("int32", ld_shared_u32(s_hist2, src + tx + j))
                )
            st_shared_u32(s_hist2, dst + tx, txl.reinterpret("uint32", value))
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
    off = txl.local_scalar("int32", init=raking_offset(tx))
    st_shared_u32(s_scan, off, value)
    bar_sync()
    with txl.If(tx < RAKING_THREADS), txl.Then():
        base = tx * RAKING_STRIDE
        cache = txl.alloc_local([RAKING_SEGMENT], "uint32")
        total = txl.local_scalar("uint32", init=txl.uint32(0))
        with txl.unroll(RAKING_SEGMENT) as j:
            txl.assign(cache[j], ld_shared_u32(s_scan, base + j))
            txl.assign(total, total + cache[j])
        incl = txl.local_scalar("uint32", init=warp_inclusive_sum_u32(total, tx))
        run = txl.local_scalar("uint32", init=incl - total)
        with txl.unroll(RAKING_SEGMENT) as j2:
            st_shared_u32(s_scan, base + j2, run)
            txl.assign(run, run + cache[j2])
        with txl.If(tx == RAKING_THREADS - 1), txl.Then():
            st_shared_u32(s_scan, RAKING_ELEMENTS, incl)
    bar_sync()
    txl.assign(out, ld_shared_u32(s_scan, off))
    txl.assign(total_out, ld_shared_u32(s_scan, RAKING_ELEMENTS))


def det_thread_strided_collect(inp, s_scan, s_indices, tx, row_in, row_len, cfg, pivot, eq_needed):
    """``DeterministicThreadStridedCollect`` (``:255-286``), the TIE_BREAK=None collector.

    Count matches per thread over a thread-strided walk, block-scan the counts,
    then re-walk and emit.  The emit target is
    ``s_indices[top_k - eq_needed + local_pos]`` (``:2587-2590``); the predicate
    is ``ToOrdered(score[idx]) == pivot``.
    """
    count = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.serial(tx, row_len, step=cfg.block) as i:
        cur = txl.cast(
            ordered_key(ld_global_nc_bits(inp, row_in + txl.cast(i, "int64"), cfg.is32), cfg.is32),
            "uint32",
        )
        with txl.If(cur == pivot), txl.Then():
            txl.assign(count, count + txl.uint32(1))
    prefix = txl.local_scalar("uint32")
    total = txl.local_scalar("uint32")
    block_exclusive_sum_raking(prefix, total, s_scan, tx, count)
    with txl.If(count > txl.uint32(0)), txl.Then():
        with txl.If(prefix < txl.cast(eq_needed, "uint32")), txl.Then():
            pos = txl.local_scalar("uint32", init=prefix)
            # Loop-invariant but read inside the walk below: a plain binding
            # would sink the min back into the loop body.
            end = txl.local_scalar("uint32", init=txl.min(prefix + count, txl.cast(eq_needed, "uint32")))
            done = txl.local_scalar("int32", init=txl.int32(0))
            with txl.serial(tx, row_len, step=cfg.block) as i2:
                with txl.If(done == 0), txl.Then():
                    cur2 = txl.cast(
                        ordered_key(
                            ld_global_nc_bits(inp, row_in + txl.cast(i2, "int64"), cfg.is32), cfg.is32
                        ),
                        "uint32",
                    )
                    with txl.If(cur2 == pivot), txl.Then():
                        st_shared_u32(
                            s_indices,
                            cfg.top_k - eq_needed + txl.reinterpret("int32", pos),
                            txl.reinterpret("uint32", i2),
                        )
                        txl.assign(pos, pos + txl.uint32(1))
                        with txl.If(pos == end), txl.Then():
                            txl.assign(done, txl.int32(1))
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
    with txl.If(tx == 0), txl.Then():
        st_shared_u32(s_scal, SC_EMITTED, txl.uint32(0))
        st_shared_u32(s_scal, SC_CHUNK_BASE, txl.uint32(0))
        st_shared_u32(s_scal, SC_CHUNK_TAKE, txl.uint32(0))
    bar_sync()
    chunk_items = cfg.block * DET_ITEMS_PER_THREAD
    num_chunks = (row_len + chunk_items - 1) // chunk_items
    if not isinstance(num_chunks, int):
        num_chunks = txl.local_scalar("int32", init=num_chunks)
    stop = txl.local_scalar("int32", init=txl.int32(0))
    with txl.serial(0, num_chunks) as chunk:
        with txl.If(stop == 0), txl.Then():
            rows_of = txl.alloc_local([DET_ITEMS_PER_THREAD], "int32")
            sel_of = txl.alloc_local([DET_ITEMS_PER_THREAD], "uint32")
            cnt = txl.local_scalar("uint32", init=txl.uint32(0))
            with txl.unroll(DET_ITEMS_PER_THREAD) as item:
                linear = txl.local_scalar(
                    "int32", init=chunk * chunk_items + tx * DET_ITEMS_PER_THREAD + item
                )
                txl.assign(rows_of[item], txl.int32(0))
                txl.assign(sel_of[item], txl.uint32(0))
                with txl.If(linear < row_len), txl.Then():
                    if reverse:
                        txl.assign(rows_of[item], row_len - 1 - linear)
                    else:
                        txl.assign(rows_of[item], linear)
                    curc = txl.cast(
                        ordered_key(
                            ld_global_nc_bits(
                                inp, row_in + txl.cast(rows_of[item], "int64"), cfg.is32
                            ),
                            cfg.is32,
                        ),
                        "uint32",
                    )
                    with txl.If(curc == pivot), txl.Then():
                        txl.assign(sel_of[item], txl.uint32(1))
                        txl.assign(cnt, cnt + txl.uint32(1))
            prefix = txl.local_scalar("uint32")
            blocksel = txl.local_scalar("uint32")
            block_exclusive_sum_raking(prefix, blocksel, s_scan, tx, cnt)
            with txl.If(tx == 0), txl.Then():
                emitted = ld_shared_u32(s_scal, SC_EMITTED)
                st_shared_u32(s_scal, SC_CHUNK_BASE, emitted)
                remaining = txl.local_scalar("uint32", init=txl.uint32(0))
                with txl.If(emitted < txl.cast(eq_needed, "uint32")), txl.Then():
                    txl.assign(remaining, txl.cast(eq_needed, "uint32") - emitted)
                take = txl.min(remaining, blocksel)
                st_shared_u32(s_scal, SC_CHUNK_TAKE, take)
                st_shared_u32(s_scal, SC_EMITTED, emitted + take)
            bar_sync()
            chunk_take = ld_shared_u32(s_scal, SC_CHUNK_TAKE)
            chunk_base = ld_shared_u32(s_scal, SC_CHUNK_BASE)
            with txl.If(cnt > txl.uint32(0)), txl.Then():
                with txl.If(prefix < chunk_take), txl.Then():
                    epos = txl.local_scalar("uint32", init=prefix)
                    # Same: invariant across the item walk that reads it.
                    eend = txl.local_scalar("uint32", init=txl.min(prefix + cnt, chunk_take))
                    fin = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.unroll(DET_ITEMS_PER_THREAD) as item2:
                        with txl.If(fin == 0), txl.Then():
                            with txl.If(sel_of[item2] == txl.uint32(1)), txl.Then():
                                st_shared_u32(
                                    s_indices,
                                    cfg.top_k
                                    - eq_needed
                                    + txl.reinterpret("int32", chunk_base + epos),
                                    txl.reinterpret("uint32", rows_of[item2]),
                                )
                                txl.assign(epos, epos + txl.uint32(1))
                                with txl.If(epos == eend), txl.Then():
                                    txl.assign(fin, txl.int32(1))
            bar_sync()
            with txl.If(ld_shared_u32(s_scal, SC_EMITTED) >= txl.cast(eq_needed, "uint32")), txl.Then():
                txl.assign(stop, txl.int32(1))
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
    with txl.If(eq_needed > 0), txl.Then():
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
    with txl.If(tx < cfg.radix), txl.Then():
        cur = txl.reinterpret("int32", ld_shared_u32(s_hist2, tx))
        with txl.If(cur > topk), txl.Then():
            nxt = txl.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
            with txl.If(nxt <= topk), txl.Then():
                st_shared_u32(s_scal, SC_THRESH_BIN, txl.reinterpret("uint32", tx))
                if reset_next:
                    st_shared_u32(s_scal, SC_NUM_INPUT + next_idx, txl.uint32(0))
                st_shared_u32(s_scal, SC_LAST_REMAIN, txl.reinterpret("uint32", topk - nxt))
    bar_sync()


def _refine_bin(inp, row_in, idx, offset, cfg):
    """One candidate's byte at ``offset`` of its ordered key (``:2679``, ``:2637``)."""
    return txl.cast(
        txl.bitwise_and(
            txl.shift_right(
                txl.cast(
                    ordered_key(
                        ld_global_nc_bits(inp, row_in + txl.cast(idx, "int64"), cfg.is32), cfg.is32
                    ),
                    "uint32",
                ),
                txl.uint32(offset),
            ),
            txl.uint32(0xFF),
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
    raw = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_NUM_INPUT + r_idx))
    num_input = txl.local_scalar("int32", init=txl.min(raw, cfg.smem_input))

    update_refine_threshold(s_hist2, s_scal, tx, cfg, topk, r_idx ^ 1, True)

    threshold = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
    if cfg.det:
        with txl.If(tx == 0), txl.Then():
            st_shared_u32(
                s_scal,
                SC_REFINE_TH + (cfg.first_shift - offset) // 8,
                txl.reinterpret("uint32", threshold),
            )
    txl.assign(topk, topk - txl.reinterpret("int32", ld_shared_u32(s_hist2, threshold + 1)))
    with txl.If(topk == 0):
        with txl.Then():
            # Pivot resolved: only bins strictly greater than the threshold remain.
            with txl.serial(tx, num_input, step=cfg.block) as i:
                idx = txl.reinterpret("int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i))
                with txl.If(_refine_bin(inp, row_in, idx, offset, cfg) > threshold), txl.Then():
                    pos = txl.reinterpret(
                        "int32", atom_shared_add_u32(s_scal, SC_COUNTER, txl.uint32(1))
                    )
                    st_shared_u32(s_indices, pos, txl.reinterpret("uint32", idx))
            bar_sync()
            txl.assign(resolved, txl.int32(1))
        with txl.Else():
            if is_last:
                # collect_with_threshold_last_round (:2635-2645): one barrier.
                with txl.serial(tx, num_input, step=cfg.block) as i2:
                    idx2 = txl.reinterpret(
                        "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i2)
                    )
                    bin2 = txl.local_scalar("int32", init=_refine_bin(inp, row_in, idx2, offset, cfg))
                    collect_gt_and_nondet_eq(s_scal, s_indices, cfg, bin2, threshold, idx2, True)
                bar_sync()
            else:
                # collect_with_threshold_non_last_round (:2646-2672): three
                # barriers, ping-ponging the survivors into s_input_idx[r_idx ^ 1]
                # together with the next byte's histogram.
                bar_sync()
                with txl.If(tx < cfg.radix + 1), txl.Then():
                    st_shared_u32(s_hist2, tx, txl.uint32(0))
                bar_sync()
                with txl.serial(tx, num_input, step=cfg.block) as i3:
                    idx3 = txl.reinterpret(
                        "int32", ld_shared_u32(s_input, r_idx * cfg.smem_input + i3)
                    )
                    ord3 = txl.cast(
                        ordered_key(
                            ld_global_nc_bits(inp, row_in + txl.cast(idx3, "int64"), cfg.is32),
                            cfg.is32,
                        ),
                        "uint32",
                    )
                    bin3 = txl.cast(
                        txl.bitwise_and(txl.shift_right(ord3, txl.uint32(offset)), txl.uint32(0xFF)),
                        "int32",
                    )
                    with txl.If(bin3 > threshold):
                        with txl.Then():
                            pos3 = txl.reinterpret(
                                "int32", atom_shared_add_u32(s_scal, SC_COUNTER, txl.uint32(1))
                            )
                            st_shared_u32(s_indices, pos3, txl.reinterpret("uint32", idx3))
                        with txl.Else():
                            with txl.If(bin3 == threshold), txl.Then():
                                slot3 = txl.reinterpret(
                                    "int32",
                                    atom_shared_add_u32(
                                        s_scal, SC_NUM_INPUT + (r_idx ^ 1), txl.uint32(1)
                                    ),
                                )
                                with txl.If(slot3 < cfg.smem_input):
                                    with txl.Then():
                                        st_shared_u32(
                                            s_input,
                                            (r_idx ^ 1) * cfg.smem_input + slot3,
                                            txl.reinterpret("uint32", idx3),
                                        )
                                        sub3 = txl.cast(
                                            txl.bitwise_and(
                                                txl.shift_right(ord3, txl.uint32(offset - 8)),
                                                txl.uint32(0xFF),
                                            ),
                                            "int32",
                                        )
                                        atom_shared_add_u32(s_hist2, sub3, txl.uint32(1))
                                    with txl.Else():
                                        atom_shared_or_b32(s_scal, SC_REFINE_OVERFLOW, txl.uint32(1))
                bar_sync()


def body_rehist_threshold_bin(s_hist2, threshold_bin, cfg, bits, index):
    """16-bit fallback re-histogram (``:2715-2724``): low byte, threshold bin only."""
    with txl.If(coarse_key(bits, cfg.is32) == threshold_bin), txl.Then():
        atom_shared_add_u32(
            s_hist2,
            txl.cast(
                txl.bitwise_and(txl.cast(ordered_key(bits, cfg.is32), "uint32"), txl.uint32(0xFF)),
                "int32",
            ),
            txl.uint32(1),
        )


def body_recollect_threshold_bin(s_scal, s_indices, threshold_bin, cfg, bits, index):
    """16-bit fallback re-collect (``:2740-2748``).

    The ``coarse_bin != threshold_bin`` guard at ``:2741-2744`` is load-bearing:
    without it the pass would compare out-of-bin elements by their low byte and
    re-collect every strict winner the filter stage already appended.
    """
    with txl.If(coarse_key(bits, cfg.is32) == threshold_bin), txl.Then():
        sub = txl.local_scalar(
            "int32",
            init=txl.cast(
                txl.bitwise_and(txl.cast(ordered_key(bits, cfg.is32), "uint32"), txl.uint32(0xFF)),
                "int32",
            ),
        )
        threshold = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
        collect_gt_and_nondet_eq(s_scal, s_indices, cfg, sub, threshold, index, True)


def _prefix_match(match, ordered, threshold_bytes, rnd):
    """Compare every byte fixed by an earlier round (``:2823-2831``).

    A Python loop over the compile-time round index, so it unrolls into the
    source's straight-line chain of byte compares.
    """
    for prev in range(rnd):
        got = txl.cast(
            txl.bitwise_and(txl.shift_right(ordered, txl.uint32(24 - prev * 8)), txl.uint32(0xFF)), "int32"
        )
        with txl.If(got != txl.cast(threshold_bytes[prev], "int32")), txl.Then():
            txl.assign(match, txl.int32(0))


def body_fallback_rehist(s_hist2, threshold_bytes, threshold_bin, cfg, rnd, bits, index):
    """fp32 fallback per-round re-histogram (``:2819-2837``).

    Only threshold-bin elements whose ordered key still matches the bytes fixed
    by earlier rounds contribute to this round's byte histogram.
    ``threshold_bytes`` is the per-thread register array of ``:2805``.
    """
    with txl.If(coarse_key(bits, cfg.is32) == threshold_bin), txl.Then():
        # Read once per already-fixed byte plus once for this round's bump, on
        # every element of the row, on each of the four rebuild rounds.
        ordered = txl.local_scalar("uint32", init=txl.cast(ordered_key(bits, cfg.is32), "uint32"))
        match = txl.local_scalar("int32", init=txl.int32(1))
        _prefix_match(match, ordered, threshold_bytes, rnd)
        with txl.If(match == 1), txl.Then():
            atom_shared_add_u32(
                s_hist2,
                txl.cast(
                    txl.bitwise_and(txl.shift_right(ordered, txl.uint32(24 - rnd * 8)), txl.uint32(0xFF)),
                    "int32",
                ),
                txl.uint32(1),
            )


def body_collect_by_pivot(s_scal, s_indices, threshold_bin, cfg, pivot, eq_needed, bits, index):
    """fp32 fallback re-collect (``:2883-2895``) -- a three-way dispatch.

    Coarse-bin winners are compared on the **coarse** key with no eq claim;
    out-of-bin elements are dropped; threshold-bin elements are compared on the
    full 32-bit ordered key against the rebuilt pivot.
    """
    # Both keys are read twice by the collector they are handed to.
    bin_id = txl.local_scalar("int32", init=coarse_key(bits, cfg.is32))
    with txl.If(bin_id > threshold_bin):
        with txl.Then():
            collect_gt_and_nondet_eq(s_scal, s_indices, cfg, bin_id, threshold_bin, index, False)
        with txl.Else():
            with txl.If(bin_id == threshold_bin), txl.Then():
                ordered = txl.local_scalar(
                    "uint32", init=txl.cast(ordered_key(bits, cfg.is32), "uint32")
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
        with txl.If(stop == 0), txl.Then():
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
            with txl.If(resolved == 1):
                with txl.Then():
                    txl.assign(det_stop, txl.int32(rnd))
                    txl.assign(stop, txl.int32(1))
                with txl.Else():
                    with (
                        txl.If(
                            txl.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0
                        ),
                        txl.Then(),
                    ):
                        txl.assign(stop, txl.int32(1))


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
        with txl.If(halt == 0), txl.Then():
            with txl.If(tx < cfg.radix + 1), txl.Then():
                st_shared_u32(s_hist2, tx, txl.uint32(0))
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
            with txl.If(tx < cfg.radix), txl.Then():
                curf = txl.reinterpret("int32", ld_shared_u32(s_hist2, tx))
                with txl.If(curf > remain), txl.Then():
                    nxtf = txl.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
                    with txl.If(nxtf <= remain), txl.Then():
                        # Only the bin id here; s_num_input and s_last_remain stay
                        # untouched (:2842-2845).
                        st_shared_u32(s_scal, SC_THRESH_BIN, txl.reinterpret("uint32", tx))
            bar_sync()
            thrf = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
            # threshold_bytes is a per-thread register array in the source (:2805);
            # every thread reads the same s_threshold_bin_id, so no publication step
            # and no extra barrier is needed.
            txl.assign(bytes_reg[rnd], txl.cast(thrf, "uint32"))
            txl.assign(remain, remain - txl.reinterpret("int32", ld_shared_u32(s_hist2, thrf + 1)))
            bar_sync()
            with txl.If(remain == 0), txl.Then():
                txl.assign(stop_round, txl.int32(rnd))
                txl.assign(halt, txl.int32(1))


def _fp32_det_pivot_bytes(s_scal, piv, det_stop):
    """``build_det_pivot`` (``:2539-2544``), one byte per refine round."""
    for rnd in range(4):
        byte0 = txl.Select(
            txl.int32(rnd) <= det_stop, ld_shared_u32(s_scal, SC_REFINE_TH + rnd), txl.uint32(0xFF)
        )
        txl.assign(piv, txl.bitwise_or(piv, txl.shift_left(byte0, txl.uint32(24 - rnd * 8))))


def _fp32_pivot_bytes(pivf, bytes_reg, remain, stop_round):
    """The fallback pivot (``:2860-2867``), one byte per rebuild round.

    A byte is forced to ``0xFF`` only once the quota is met and the round is past
    ``stop_round``.
    """
    for rnd in range(4):
        bytef = txl.local_scalar("uint32", init=bytes_reg[rnd])
        with txl.If(remain == 0), txl.Then():
            with txl.If(txl.int32(rnd) > stop_round), txl.Then():
                txl.assign(bytef, txl.uint32(0xFF))
        txl.assign(pivf, txl.bitwise_or(pivf, txl.shift_left(bytef, txl.uint32(24 - rnd * 8))))


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
    det_stop = txl.local_scalar("int32", init=txl.int32(3))  # NUM_ROUNDS - 1 (:2771)
    stop = txl.local_scalar("int32", init=txl.int32(0))
    remain = txl.local_scalar("int32", init=topk_after_coarse)  # (:2804)
    stop_round = txl.local_scalar("int32", init=txl.int32(3))
    halt = txl.local_scalar("int32", init=txl.int32(0))
    # threshold_bytes: a per-thread register array in the source (:2805), not
    # shared state.
    bytes_reg = txl.alloc_local([4], "uint32")
    with txl.unroll(4) as r:
        txl.assign(bytes_reg[r], txl.uint32(0xFF))  # (:2805-2810)

    # The whole round loop is guarded on the flag the filter stage may have set
    # (:2772).
    with txl.If(txl.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) == 0), txl.Then():
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
        with txl.If(txl.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) == 0), txl.Then():
            piv = txl.local_scalar("uint32", init=txl.uint32(0))
            _fp32_det_pivot_bytes(s_scal, piv, det_stop)
            collect_det_eq_pivot(
                inp, s_scan, s_indices, s_scal, tx, row_in, row_len, cfg, piv, topk
            )

    # 32-bit pivot rebuild after an overflow (:2800-2900).  Overflow can follow
    # partial writes to s_indices / s_counter, so the selection is rebuilt from
    # scratch.
    with txl.If(txl.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0), txl.Then():
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
        pivf = txl.local_scalar("uint32", init=txl.uint32(0))
        _fp32_pivot_bytes(pivf, bytes_reg, remain, stop_round)
        with txl.If(tx == 0), txl.Then():
            st_shared_u32(s_scal, SC_COUNTER, txl.uint32(0))
            st_shared_u32(s_scal, SC_LAST_REMAIN, txl.reinterpret("uint32", remain))
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
    with txl.If(txl.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_OVERFLOW)) != 0):
        with txl.Then():
            with txl.If(tx < cfg.radix + 1), txl.Then():
                st_shared_u32(s_hist2, tx, txl.uint32(0))
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
            with txl.If(tx == 0), txl.Then():
                st_shared_u32(s_scal, SC_THRESH_BIN, txl.uint32(0))
                st_shared_u32(s_scal, SC_LAST_REMAIN, txl.uint32(0))
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
                thr_f = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
                eq_f = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_LAST_REMAIN))
                collect_det_eq_pivot(
                    inp,
                    s_scan,
                    s_indices,
                    s_scal,
                    tx,
                    row_in,
                    row_len,
                    cfg,
                    txl.bitwise_or(
                        txl.shift_left(txl.cast(threshold_bin, "uint32"), txl.uint32(8)),
                        txl.cast(thr_f, "uint32"),
                    ),
                    eq_f,
                )
        with txl.Else():
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
                th0 = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_REFINE_TH))
                collect_det_eq_pivot(
                    inp,
                    s_scan,
                    s_indices,
                    s_scal,
                    tx,
                    row_in,
                    row_len,
                    cfg,
                    txl.bitwise_or(
                        txl.shift_left(txl.cast(threshold_bin, "uint32"), txl.uint32(8)),
                        txl.cast(th0, "uint32"),
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
    topk = txl.local_scalar("int32", init=txl.int32(cfg.top_k))

    # --- init (:2450-2458) -------------------------------------------------
    with txl.If(tx == 0), txl.Then():
        st_shared_u32(s_scal, SC_REFINE_OVERFLOW, txl.uint32(0))
    if cfg.det:
        with txl.If(tx < 4), txl.Then():
            st_shared_u32(s_scal, SC_REFINE_TH + tx, txl.uint32(0xFF))
    with txl.If(tx < cfg.radix + 1), txl.Then():
        st_shared_u32(s_hist2, tx, txl.uint32(0))
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
    with txl.If(tx < cfg.radix), txl.Then():
        cur0 = txl.reinterpret("int32", ld_shared_u32(s_hist2, tx))
        with txl.If(cur0 > topk), txl.Then():
            nxt0 = txl.reinterpret("int32", ld_shared_u32(s_hist2, tx + 1))
            with txl.If(nxt0 <= topk), txl.Then():
                st_shared_u32(s_scal, SC_THRESH_BIN, txl.reinterpret("uint32", tx))
                st_shared_u32(s_scal, SC_NUM_INPUT, txl.uint32(0))
                st_shared_u32(s_scal, SC_COUNTER, txl.uint32(0))
    bar_sync()
    threshold_bin = txl.reinterpret("int32", ld_shared_u32(s_scal, SC_THRESH_BIN))
    txl.assign(topk, topk - txl.reinterpret("int32", ld_shared_u32(s_hist2, threshold_bin + 1)))
    # A SNAPSHOT of `topk` before the refine rounds walk it down (:2804); a plain
    # binding would re-read the counter after those writes.
    topk_after_coarse = txl.local_scalar("int32", init=topk)

    with txl.If(topk == 0):
        with txl.Then():
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
        with txl.Else():
            # --- Stage 2: the filter (:2561-2629) --------------------------
            bar_sync()
            with txl.If(tx < cfg.radix + 1), txl.Then():
                st_shared_u32(s_hist2, tx, txl.uint32(0))
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
            resolved = txl.local_scalar("int32", init=txl.int32(0))
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
    with txl.serial(tx, cfg.top_k, step=cfg.block, unroll=2) as base:
        sel = txl.reinterpret("int32", ld_shared_u32(s_indices, base))
        # Materialized: a lazy 64-bit slot address is re-narrowed per store.
        slot = txl.local_scalar("int64", init=row_out + txl.cast(base, "int64"))
        if cfg.basic:
            st_global_u32(out_idx, slot, txl.reinterpret("uint32", sel))
            st_global_bits(
                out_val,
                slot,
                ld_global_nc_bits(inp, row_in + txl.cast(sel, "int64"), cfg.is32),
                cfg.is32,
            )
        elif cfg.det:
            # Local index; the transform is deferred to the finalize kernel.
            st_global_u32(out_idx, slot, txl.reinterpret("uint32", sel))
        elif cfg.page_table:
            st_global_u32(
                out_idx,
                slot,
                ld_global_nc_u32(
                    aux, txl.cast(batch_idx, "int64") * aux_stride + txl.cast(page_start + sel, "int64")
                ),
            )
        else:
            st_global_u32(out_idx, slot, txl.reinterpret("uint32", sel + offset_val))
