# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
#
# The collective transcribed here is NVIDIA CCCL/CUB's `cub::BlockRadixSort`, which
# FlashInfer's topk kernels instantiate and which ships in the cccl tree its JIT
# compiles (flashinfer/data/cccl/cub). That code is BSD-3, so its notice travels
# with this file and the SPDX expression below is the pair.
#
# Copyright (c) 2011, Duane Merrill. All rights reserved.
# Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""``cub::BlockRadixSort`` emitters for the filtered top-k finalize kernel.

``FinalizeTopKIndicesKernel`` (``include/flashinfer/topk.cuh:2993``) instantiates
``cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, DType>`` -- key
plus value for ``Plain``, and keys-only (``DType = uint8_t`` with null values) for
the transform modes.  These helpers reproduce that collective's per-pass
structure so the finalize kernel can call it the way the source does.

Template defaults that fix the shape (``cub/block/block_radix_sort.cuh:221-231``):
``RadixBits = 4`` for **every** ``(BLOCK_THREADS, ITEMS_PER_THREAD)`` pair,
``MemoizeOuterScan = true``, ``InnerScanAlgorithm = BLOCK_SCAN_WARP_SCANS``.  The
ranker is the basic ``BlockRadixRank`` (``:250-255``), so there is no
``match.any``, ballot, or ``PRMT`` on any path, and no arch-conditional fork --
SM100 compiles the classic algorithm.

With ``RadixBits = 4`` and ``uint16``/``uint32`` counters
(``block_radix_rank.cuh:229-262``): ``PACKING_RATIO = 2``, ``COUNTER_LANES = 8``,
``PADDED_COUNTER_LANES = RAKING_SEGMENT = 9``, and ``BINS_TRACKED_PER_THREAD = 1``
for every block size used here.  The counter grid is ``9 * BLOCK_THREADS`` words
addressed both as ``uint32`` (packed raking adds) and as ``uint16`` (per-digit
read-modify-write); each thread owns column ``tid``, so the RMW needs **no**
atomics.

Digit extraction is a shift plus a mask, not ``bfe``: ``BFEDigitExtractor``
(``radix_rank_sort_operations.cuh:102-105``) calls ``cuda::bitfield_extract``,
whose ``NV_PROVIDES_SM_70`` dispatch arm is empty (``cuda/__bit/bitfield.h:107``)
so control falls through to the shift-and-mask return at ``:114``.
"""

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils.topk_radix import (
    bar_sync,
    ld_shared_pair_u32,
    ld_shared_quad_u32,
    ld_shared_u16,
    ld_shared_u32,
    shfl_up_u32,
    st_shared_u16,
    st_shared_u32,
)

# Every loop in this module that walks a *trace-time* count -- the digit-pass
# walk, the warp-fold chain -- is a Python loop, and every loop that must survive
# into the TIR as a real ``For`` node is a ``K.unroll`` context.  A Python loop
# emits the same straight-line TIR that cub's ``_CCCL_PRAGMA_UNROLL_FULL``
# produces, and the digit-pass loop is unrolled because ``end_bit`` is static per
# config.

# --- cub::BlockRadixRank geometry for RadixBits == 4 -------------------------
RADIX_BITS = 4
RADIX_DIGITS = 1 << RADIX_BITS  # 16
PACKING_RATIO = 2  # sizeof(uint32) / sizeof(uint16)
LOG_COUNTER_LANES = RADIX_BITS - 1  # 3
COUNTER_LANES = 1 << LOG_COUNTER_LANES  # 8
PADDED_COUNTER_LANES = COUNTER_LANES + 1  # 9
RAKING_SEGMENT = PADDED_COUNTER_LANES  # 9
WARP_THREADS = 32
LOG_SMEM_BANKS = 5


def sort_passes(end_bit: int) -> list[tuple[int, int]]:
    """``[begin_bit, end_bit)`` split into ``RADIX_BITS``-wide digit passes.

    ``SortBlocked`` (``block_radix_sort.cuh:377-429``) walks ``begin_bit`` up by
    ``RADIX_BITS`` and clamps the last pass, so a non-multiple ``end_bit`` gets a
    narrow final digit rather than an extra full one.
    """
    passes = []
    begin = 0
    while begin < end_bit:
        passes.append((begin, min(RADIX_BITS, end_bit - begin)))
        begin += RADIX_BITS
    return passes


def insert_padding(items_per_thread: int) -> bool:
    """``BlockExchange::INSERT_PADDING`` (``block_exchange.cuh:139``).

    ``ItemsPerThread > 4 && is_power_of_two(ItemsPerThread)`` -- true only for
    ``IPT == 8`` among the six finalize configs; ``IPT == 4`` fails the first
    clause and ``IPT == 9`` the second.
    """
    return items_per_thread > 4 and (items_per_thread & (items_per_thread - 1)) == 0


def exchange_elements(block_threads: int, items_per_thread: int) -> int:
    """``TIME_SLICED_ITEMS + PADDING_ITEMS`` (``block_exchange.cuh:140-145``)."""
    items = block_threads * items_per_thread
    pad = (items >> LOG_SMEM_BANKS) if insert_padding(items_per_thread) else 0
    return items + pad


def rank_words(block_threads: int) -> int:
    """``uint32`` words in the counter grid: ``PADDED_COUNTER_LANES * BLOCK_THREADS``."""
    return RAKING_SEGMENT * block_threads


def warps(block_threads: int) -> int:
    return (block_threads + WARP_THREADS - 1) // WARP_THREADS


def scan_words(block_threads: int) -> int:
    """``BlockScanWarpScans::_TempStorage``: ``warp_aggregates[WARPS]`` + ``block_prefix``.

    ``WarpScanShfl::TempStorage`` is an empty struct
    (``warp_scan_shfl.cuh:74-77``), so the shuffle scan itself needs no storage.
    """
    return warps(block_threads) + 1


def union_bytes(block_threads: int, items_per_thread: int, value_bytes: int) -> int:
    """``BlockRadixSort::_TempStorage`` (``block_radix_sort.cuh:268-274``).

    A union of the ranking storage with the key and value exchange buffers.
    """
    items = exchange_elements(block_threads, items_per_thread)
    return max(rank_words(block_threads) * 4, items * 4, items * value_bytes)


def alloc_sort_smem_static(block_threads, items_per_thread):
    """Static `__shared__` layout for the u32-key/u32-value sort.

    The source declares its `BlockRadixSort::TempStorage` as a plain
    `__shared__` array, so every offset into it is a compile-time constant.  The
    dynamic-shared pool instead hands out offsets from a runtime `extern
    __shared__` base, which puts an address computation on the critical path of
    every shared access -- visible as a fixed cost that short kernels cannot
    amortize.

    The union is expressed as views over one allocation rather than through
    `move_base_to`: the rank grid dominates at every rung, so the key and value
    exchange buffers fit inside it exactly as in cub's union.
    """
    words = rank_words(block_threads)
    assert exchange_elements(block_threads, items_per_thread) <= words, (
        "exchange must fit inside the rank grid for the union to hold"
    )
    counters32 = K.alloc_buffer((words,), "uint32", scope="shared")
    counters16 = counters32.view("uint16")
    # Both exchange buffers alias the rank grid, exactly as cub's inner union
    # does: a barrier separates the rank phase from each scatter, and the ranks
    # themselves live in registers by then, so nothing live is overwritten.
    xchg_keys = counters32
    xchg_values = counters32
    # The block-scan scratch sits outside that union, as in cub -- a separate
    # `__shared__` array rather than an offset view, so its indices stay
    # zero-based for the emitters.
    scan = K.alloc_buffer((scan_words(block_threads),), "uint32", scope="shared")
    return counters32, counters16, xchg_keys, xchg_values, scan


def alloc_sort_smem(pool, block_threads, items_per_thread, value_dtype, value_bytes):
    """Lay out the sort's shared union plus the block-scan scratch.

    The counter grid is addressed as ``uint32`` and as ``uint16`` over the same
    bytes, and the key/value exchange buffers alias it, exactly as cub's nested
    unions do.  Every offset passed to ``move_base_to`` is a Python literal:
    reading ``pool.offset`` inside a traced body turns it into a TIR variable and
    the ``max()`` inside ``move_base_to`` then fails.
    """
    base = 0
    xchg_items = exchange_elements(block_threads, items_per_thread)
    total = union_bytes(block_threads, items_per_thread, value_bytes)

    pool.move_base_to(base)
    counters32 = pool.alloc((rank_words(block_threads),), "uint32", align=16)
    pool.move_base_to(base)
    counters16 = pool.alloc((rank_words(block_threads) * PACKING_RATIO,), "uint16", align=16)
    pool.move_base_to(base)
    xchg_keys = pool.alloc((xchg_items,), "uint32", align=16)
    pool.move_base_to(base)
    xchg_values = pool.alloc((xchg_items,), value_dtype, align=16)

    pool.move_base_to(base + total)
    scan = pool.alloc((scan_words(block_threads),), "uint32", align=16)
    return counters32, counters16, xchg_keys, xchg_values, scan


def emit_digit(key, begin_bit, num_bits: int):
    """``BFEDigitExtractor::Digit`` -- shift and mask, never ``bfe`` on SM >= 70.

    ``begin_bit`` may be a Python int (unrolled pass loop) or a TIR value (rolled
    pass loop, which is the shape the source compiles to).
    """
    shift = K.uint32(begin_bit) if isinstance(begin_bit, int) else K.cast(begin_bit, "uint32")
    return K.bitwise_and(K.shift_right(key, shift), K.uint32((1 << num_bits) - 1))


def padded_offset(off, items_per_thread: int):
    """``BlockExchange``'s bank-conflict padding: ``x + (x >> 5)`` when enabled."""
    if not insert_padding(items_per_thread):
        return off
    return off + K.shift_right(off, K.int32(LOG_SMEM_BANKS))


def _scan_warp_inclusive(out, incl, tx, value):
    """``WarpScanShfl`` inclusive sum plus cub's integer ``Update`` shortcut.

    Five ``shfl.sync.up.b32`` steps give the inclusive scan; for ``plus<>`` on an
    integer type cub takes ``exclusive = inclusive - input``
    (``warp_scan_shfl.cuh:700-706``) instead of a sixth shuffle, which is why the
    export shows exactly five shuffles per instantiation.
    """
    lane = K.local_scalar("int32", init=tx % WARP_THREADS)
    K.assign(incl[0], value)
    with K.unroll(5) as step:
        peer = K.local_scalar("uint32", init=shfl_up_u32(incl[0], K.shift_left(K.int32(1), step)))
        K.assign(incl[0], K.Select(lane >= K.shift_left(K.int32(1), step), incl[0] + peer, incl[0]))
    K.assign(out[0], incl[0] - value)


def _scan_publish_warp_aggregate(scan, incl, tx):
    """Last lane of each warp shares its warp aggregate (``:169-172``)."""
    with K.If(tx % WARP_THREADS == WARP_THREADS - 1), K.Then():
        st_shared_u32(scan, tx // WARP_THREADS, incl[0])
    bar_sync()


def _scan_seed_aggregate(scan, agg, wpre):
    """``block_aggregate = warp_aggregates[0]`` (``:179``)."""
    K.assign(agg[0], ld_shared_u32(scan, 0))
    K.assign(wpre[0], K.uint32(0))


def _scan_fold_warp(scan, agg, wpre, tx, w):
    """One step of ``ApplyWarpAggregates`` (``:129-134``)."""
    with K.If(tx // WARP_THREADS == w), K.Then():
        K.assign(wpre[0], agg[0])
    K.assign(agg[0], agg[0] + ld_shared_u32(scan, w))


def _scan_fold_warp_value(agg, wpre, tx, w, value):
    """``ApplyWarpAggregates`` over an aggregate already held in a register."""
    with K.If(tx // WARP_THREADS == w), K.Then():
        K.assign(wpre[0], agg[0])
    K.assign(agg[0], agg[0] + value)


def _scan_seed_and_fold_pair(scan, agg, wpre, tx):
    """``ApplyWarpAggregates`` at two warps, reading both with one vector load.

    ``warp_aggregates[]`` is contiguous and the source reads it whole --
    ``ld.shared.v2.b32`` at ``BLOCK_THREADS == 64`` and ``ld.shared.v4.b32`` at
    128 and 256 (``.loc 13 178``) -- rather than one scalar load per warp.
    """
    pair = ld_shared_pair_u32(scan, 0)
    K.assign(agg[0], pair[0])
    K.assign(wpre[0], K.uint32(0))
    with K.If(tx // WARP_THREADS == 1), K.Then():
        K.assign(wpre[0], agg[0])
    K.assign(agg[0], agg[0] + pair[1])


def _scan_seed_and_fold_quad(scan, agg, wpre, tx):
    """The same fold at four warps, over one ``ld.shared.v4.b32``."""
    quad = ld_shared_quad_u32(scan, 0)
    K.assign(agg[0], quad[0])
    K.assign(wpre[0], K.uint32(0))
    with K.If(tx // WARP_THREADS == 1), K.Then():
        K.assign(wpre[0], agg[0])
    K.assign(agg[0], agg[0] + quad[1])
    with K.If(tx // WARP_THREADS == 2), K.Then():
        K.assign(wpre[0], agg[0])
    K.assign(agg[0], agg[0] + quad[2])
    with K.If(tx // WARP_THREADS == 3), K.Then():
        K.assign(wpre[0], agg[0])
    K.assign(agg[0], agg[0] + quad[3])


def _scan_seed_and_fold_octet(scan, agg, wpre, tx):
    """The same fold at eight warps, over two ``ld.shared.v4.b32``.

    ``BLOCK_THREADS == 256`` has eight aggregates and cub reads them as two
    vectors, not eight scalars.
    """
    lo = ld_shared_quad_u32(scan, 0)
    hi = ld_shared_quad_u32(scan, 4)
    K.assign(agg[0], lo[0])
    K.assign(wpre[0], K.uint32(0))
    _scan_fold_warp_value(agg, wpre, tx, 1, lo[1])
    _scan_fold_warp_value(agg, wpre, tx, 2, lo[2])
    _scan_fold_warp_value(agg, wpre, tx, 3, lo[3])
    _scan_fold_warp_value(agg, wpre, tx, 4, hi[0])
    _scan_fold_warp_value(agg, wpre, tx, 5, hi[1])
    _scan_fold_warp_value(agg, wpre, tx, 6, hi[2])
    _scan_fold_warp_value(agg, wpre, tx, 7, hi[3])


def _scan_apply_prefix(scan, out, agg, wpre, tx, n_warps):
    """Warp prefix, the packed block-prefix callback, and the broadcast back.

    ``BlockRadixRank::PrefixCallBack`` returns ``aggregate << 16``
    (``block_radix_rank.cuh:359-374``); the packing loop runs only for
    ``PACKED == 1`` because ``PACKING_RATIO == 2``.
    """
    warp_id = K.local_scalar("int32", init=tx // WARP_THREADS)
    lane = K.local_scalar("int32", init=tx % WARP_THREADS)
    # Apply the warp prefix; lane0 of a non-zero warp takes it outright (:308-317).
    with K.If(warp_id != 0), K.Then():
        K.assign(out[0], wpre[0] + out[0])
        with K.If(lane == 0), K.Then():
            K.assign(out[0], wpre[0])
    # Warp 0 evaluates the prefix callback and shares it (:387-398).
    with K.If(warp_id == 0), K.Then():
        with K.If(lane == 0), K.Then():
            st_shared_u32(scan, n_warps, K.shift_left(agg[0], K.uint32(16)))
            K.assign(out[0], K.shift_left(agg[0], K.uint32(16)))
    bar_sync()
    bp = K.local_scalar("uint32", init=ld_shared_u32(scan, n_warps))
    with K.If(tx > 0), K.Then():
        K.assign(out[0], bp + out[0])


def emit_block_exclusive_sum_packed(scan, out, incl, agg, wpre, tx, block_threads, value):
    """``BlockScanWarpScans::ExclusiveScan`` with a block-prefix callback.

    Mirrors ``block_scan_warp_scans.cuh:299-408``.  Two ``bar.sync`` per scan, as
    in the source.  The warp-fold walk is a Python-level loop here so it emits
    cub's straight-line ``ApplyWarpAggregates`` chain and simply disappears when
    the block is a single warp; ``out[0]`` receives the exclusive result.
    """
    n_warps = warps(block_threads)
    _scan_warp_inclusive(out, incl, tx, value)
    _scan_publish_warp_aggregate(scan, incl, tx)
    # `warp_aggregates[]` is contiguous and the source reads it whole rather than
    # one scalar load per warp: `ld.shared.v2.b32` at BT == 64 and `v4` at 128 and
    # 256.  Measured on the shapes that reproduce to +/-0.001, the two forms are
    # the same speed, so this follows the source.  At one warp there is nothing to
    # vectorize and the fold chain disappears entirely.
    if n_warps == 2:
        _scan_seed_and_fold_pair(scan, agg, wpre, tx)
    elif n_warps == 4:
        _scan_seed_and_fold_quad(scan, agg, wpre, tx)
    elif n_warps == 8:
        _scan_seed_and_fold_octet(scan, agg, wpre, tx)
    else:
        _scan_seed_aggregate(scan, agg, wpre)
        for w in range(1, n_warps):
            _scan_fold_warp(scan, agg, wpre, tx, w)
    _scan_apply_prefix(scan, out, agg, wpre, tx, n_warps)


def emit_rank_keys(
    counters32,
    counters16,
    scan,
    keys,
    ranks,
    tx,
    block_threads,
    items_per_thread,
    begin_bit,
    num_bits,
):
    """``BlockRadixRank::RankKeys`` (``block_radix_rank.cuh:436-493``).

    Reset the nine padded lanes, take each item's ``uint16`` counter slot and
    read back its thread-exclusive prefix while bumping it, scan the packed
    counters, then add the block-exclusive prefix back in.  Each thread owns
    column ``tid``, so the read-modify-write needs no atomics -- two
    ``ld.shared.b16`` plus one ``st.shared.b16`` per item, which is exactly what
    the export shows at every ``ITEMS_PER_THREAD``.
    """
    prefixes = K.alloc_local((items_per_thread,), "uint32")
    slots = K.alloc_local((items_per_thread,), "int32")

    # ResetCounters: one packed zero per padded lane (:346-355).
    with K.unroll(PADDED_COUNTER_LANES) as lane:
        st_shared_u32(counters32, lane * block_threads + tx, K.uint32(0))
        # counter view: &digit_counters[LANE][tid][0] as one packed word.

    with K.unroll(items_per_thread) as i:
        digit = K.local_scalar("uint32", init=emit_digit(keys[i], begin_bit, num_bits))
        sub_counter = K.local_scalar(
            "int32", init=K.cast(K.shift_right(digit, K.uint32(LOG_COUNTER_LANES)), "int32")
        )
        counter_lane = K.local_scalar(
            "int32", init=K.cast(K.bitwise_and(digit, K.uint32(COUNTER_LANES - 1)), "int32")
        )
        # &digit_counters[counter_lane][tid][sub_counter] over the uint16 view.
        K.assign(slots[i], (counter_lane * block_threads + tx) * PACKING_RATIO + sub_counter)
        K.assign(prefixes[i], K.cast(ld_shared_u16(counters16, slots[i]), "uint32"))
        st_shared_u16(counters16, slots[i], K.cast(prefixes[i] + K.uint32(1), "uint16"))

    bar_sync()

    # ScanCounters (:376-392): memoized raking upsweep, packed block scan, then
    # an exclusive downsweep back into the same words.
    # raking_grid[linear_tid][RAKING_SEGMENT] -- each thread rakes a CONTIGUOUS
    # nine-word segment, so the block scan runs over the flat array in index
    # order.  In the counter view that same flat index is `lane * BT + tid`, i.e.
    # digit-major and thread-minor, which is what makes the ranks group by digit.
    # Striding the segment instead would order the prefix by thread first.
    cached = K.alloc_local((RAKING_SEGMENT,), "uint32")
    partial = K.local_scalar("uint32", init=K.uint32(0))
    with K.unroll(RAKING_SEGMENT) as j:
        K.assign(cached[j], ld_shared_u32(counters32, tx * RAKING_SEGMENT + j))
        K.assign(partial, partial + cached[j])

    exclusive = K.alloc_local((1,), "uint32")
    incl = K.alloc_local((1,), "uint32")
    agg = K.alloc_local((1,), "uint32")
    wpre = K.alloc_local((1,), "uint32")
    emit_block_exclusive_sum_packed(scan, exclusive, incl, agg, wpre, tx, block_threads, partial)

    with K.unroll(RAKING_SEGMENT) as j:
        st_shared_u32(counters32, tx * RAKING_SEGMENT + j, exclusive[0])
        K.assign(exclusive[0], exclusive[0] + cached[j])

    bar_sync()

    with K.unroll(items_per_thread) as i:
        K.assign(
            ranks[i],
            K.cast(prefixes[i] + K.cast(ld_shared_u16(counters16, slots[i]), "uint32"), "int32"),
        )


def emit_scatter_to_blocked_u32(buf, items_reg, ranks, tx, items_per_thread):
    """``BlockExchange::ScatterToBlocked`` for the 32-bit keys (``:600-629``).

    The scatter is by rank and stays scalar, but the gather reads this thread's
    own blocked run, which is contiguous whenever ``INSERT_PADDING`` is off.  At
    ``IPT == 4`` that is four adjacent words at a 16-byte-aligned offset and the
    source issues a single ``ld.shared.v4.b32`` for them (``block_exchange.cuh:627``,
    one for the keys and one for the values in every pass).  Reading them scalar
    instead puts three extra shared-memory round trips on the critical path of a
    kernel that runs one warp per CTA, which is where the ``(32, 4)`` rung loses
    its time.
    """
    with K.unroll(items_per_thread) as i:
        st_shared_u32(buf, padded_offset(ranks[i], items_per_thread), items_reg[i])
    bar_sync()
    if items_per_thread == 4 and not insert_padding(items_per_thread):
        quad = ld_shared_quad_u32(buf, tx * items_per_thread)
        K.assign(items_reg[0], quad[0])
        K.assign(items_reg[1], quad[1])
        K.assign(items_reg[2], quad[2])
        K.assign(items_reg[3], quad[3])
    else:
        with K.unroll(items_per_thread) as i:
            K.assign(
                items_reg[i],
                ld_shared_u32(buf, padded_offset(tx * items_per_thread + i, items_per_thread)),
            )


def emit_scatter_to_blocked_u16(buf, items_reg, ranks, tx, items_per_thread):
    """The same exchange for 16-bit paired values."""
    with K.unroll(items_per_thread) as i:
        st_shared_u16(
            buf, padded_offset(ranks[i], items_per_thread), K.cast(items_reg[i], "uint16")
        )
    bar_sync()
    with K.unroll(items_per_thread) as i:
        K.assign(
            items_reg[i],
            K.cast(
                ld_shared_u16(buf, padded_offset(tx * items_per_thread + i, items_per_thread)),
                "uint32",
            ),
        )


def emit_block_radix_sort(
    counters32,
    counters16,
    scan,
    xchg_keys,
    xchg_values,
    keys,
    values,
    ranks,
    tx,
    block_threads,
    items_per_thread,
    end_bit,
    sort_values,
    value_is32,
):
    """``BlockRadixSort::SortBlocked`` (``block_radix_sort.cuh:377-429``).

    One rank plus exchange per digit pass, ascending, blocked in and blocked out.
    The trailing barrier is skipped after the final pass, giving cub's
    ``7P - 1`` / ``9P - 1`` barrier counts for keys-only and key-value.  The pass
    loop is unrolled here because ``end_bit`` is static per config, which is also
    why no ``clz.b32`` appears.
    """
    for p, (begin, nbits) in enumerate(sort_passes(end_bit)):
        emit_rank_keys(
            counters32,
            counters16,
            scan,
            keys,
            ranks,
            tx,
            block_threads,
            items_per_thread,
            begin,
            nbits,
        )
        bar_sync()
        emit_scatter_to_blocked_u32(xchg_keys, keys, ranks, tx, items_per_thread)
        if sort_values:
            bar_sync()
            if value_is32:
                emit_scatter_to_blocked_u32(xchg_values, values, ranks, tx, items_per_thread)
            else:
                emit_scatter_to_blocked_u16(xchg_values, values, ranks, tx, items_per_thread)
        if p != len(sort_passes(end_bit)) - 1:
            bar_sync()


def emit_block_radix_sort_rolled(
    counters32,
    counters16,
    scan,
    xchg_keys,
    xchg_values,
    keys,
    values,
    ranks,
    tx,
    block_threads,
    items_per_thread,
    num_passes,
    sort_values,
    value_is32,
):
    """``SortBlocked`` with the digit-pass loop rolled, in the source's own shape.

    ``block_radix_sort.cuh:377-430`` is a ``while (true)`` that advances
    ``begin_bit`` and breaks before the trailing barrier once the pass is the
    last one, and nvcc leaves it rolled: the export carries 9 static ``bar.sync``
    and 258 instructions for ``<32, 4, int, __half>``, whatever the pass count.

    A counted ``K.serial`` loop does **not** reproduce that.  Its trip count is a
    compile-time constant, so nvcc fully unrolls it and the same entry grows to
    35 ``bar.sync`` and 837 instructions -- 3.2x the source's code for identical
    work.  TVM emits ``#pragma unroll 1`` ahead of a ``While`` node
    (``codegen_c.cc:1382``) and nothing ahead of a ``For``, so the loop is written
    here the way the source writes it: advance the bit offset in the body and
    take the trailing barrier only when another pass follows, which also gives
    the ``9P - 1`` barrier count directly.
    """
    begin = K.local_scalar("int32", init=0)
    with K.While(begin < num_passes * RADIX_BITS):
        emit_rank_keys(
            counters32,
            counters16,
            scan,
            keys,
            ranks,
            tx,
            block_threads,
            items_per_thread,
            begin,
            RADIX_BITS,
        )
        bar_sync()
        emit_scatter_to_blocked_u32(xchg_keys, keys, ranks, tx, items_per_thread)
        if sort_values:
            bar_sync()
            if value_is32:
                emit_scatter_to_blocked_u32(xchg_values, values, ranks, tx, items_per_thread)
            else:
                emit_scatter_to_blocked_u16(xchg_values, values, ranks, tx, items_per_thread)
        K.assign(begin, begin + RADIX_BITS)
        # The source breaks out before this barrier on the final pass
        # (block_radix_sort.cuh:415-421), giving 9P - 1 rather than 9P.
        with K.If(begin < num_passes * RADIX_BITS), K.Then():
            bar_sync()
