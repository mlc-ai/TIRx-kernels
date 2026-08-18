# This file is a TIRx port of code from NVIDIA CCCL/CUB
# (https://github.com/NVIDIA/cccl), Copyright (c) 2011, Duane Merrill; 2011-2018, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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

from tirx_kernels.flashinfer.utils.topk_radix import (
    bar_sync,
    ld_shared_u16,
    ld_shared_u32,
    shfl_up_u32,
    st_shared_u16,
    st_shared_u32,
)
from tvm.script import tirx as T

# Every loop in this module is a Python loop, not ``T.unroll``/``T.serial``.
# These emitters are called from inside a traced prim_func body but are not
# themselves parsed by TVMScript, so a ``T.unroll`` frame here is never turned
# into a loop -- it just fails to iterate.  A Python loop emits the same
# straight-line TIR that cub's ``_CCCL_PRAGMA_UNROLL_FULL`` produces, and the
# digit-pass loop is likewise unrolled because ``end_bit`` is static per config.

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


def emit_digit(key, begin_bit: int, num_bits: int):
    """``BFEDigitExtractor::Digit`` -- shift and mask, never ``bfe`` on SM >= 70."""
    return T.bitwise_and(T.shift_right(key, T.uint32(begin_bit)), T.uint32((1 << num_bits) - 1))


def padded_offset(off, items_per_thread: int):
    """``BlockExchange``'s bank-conflict padding: ``x + (x >> 5)`` when enabled."""
    if not insert_padding(items_per_thread):
        return off
    return off + T.shift_right(off, T.int32(LOG_SMEM_BANKS))


@T.macro
def _scan_warp_inclusive(out, incl, tx, value):
    """``WarpScanShfl`` inclusive sum plus cub's integer ``Update`` shortcut.

    Five ``shfl.sync.up.b32`` steps give the inclusive scan; for ``plus<>`` on an
    integer type cub takes ``exclusive = inclusive - input``
    (``warp_scan_shfl.cuh:700-706``) instead of a sixth shuffle, which is why the
    export shows exactly five shuffles per instantiation.
    """
    lane: T.int32 = tx % WARP_THREADS
    incl[0] = value
    for step in T.unroll(5):
        peer: T.uint32 = shfl_up_u32(incl[0], T.shift_left(T.int32(1), step))
        incl[0] = T.Select(lane >= T.shift_left(T.int32(1), step), incl[0] + peer, incl[0])
    out[0] = incl[0] - value


@T.macro
def _scan_publish_warp_aggregate(scan, incl, tx):
    """Last lane of each warp shares its warp aggregate (``:169-172``)."""
    if tx % WARP_THREADS == WARP_THREADS - 1:
        st_shared_u32(scan, tx // WARP_THREADS, incl[0])
    bar_sync()


@T.macro
def _scan_seed_aggregate(scan, agg, wpre):
    """``block_aggregate = warp_aggregates[0]`` (``:179``)."""
    agg[0] = ld_shared_u32(scan, 0)
    wpre[0] = T.uint32(0)


@T.macro
def _scan_fold_warp(scan, agg, wpre, tx, w):
    """One step of ``ApplyWarpAggregates`` (``:129-134``)."""
    if tx // WARP_THREADS == w:
        wpre[0] = agg[0]
    agg[0] = agg[0] + ld_shared_u32(scan, w)


@T.macro
def _scan_apply_prefix(scan, out, agg, wpre, tx, n_warps):
    """Warp prefix, the packed block-prefix callback, and the broadcast back.

    ``BlockRadixRank::PrefixCallBack`` returns ``aggregate << 16``
    (``block_radix_rank.cuh:359-374``); the packing loop runs only for
    ``PACKED == 1`` because ``PACKING_RATIO == 2``.
    """
    warp_id: T.int32 = tx // WARP_THREADS
    lane: T.int32 = tx % WARP_THREADS
    # Apply the warp prefix; lane0 of a non-zero warp takes it outright (:308-317).
    if warp_id != 0:
        out[0] = wpre[0] + out[0]
        if lane == 0:
            out[0] = wpre[0]
    # Warp 0 evaluates the prefix callback and shares it (:387-398).
    if warp_id == 0:
        if lane == 0:
            st_shared_u32(scan, n_warps, T.shift_left(agg[0], T.uint32(16)))
            out[0] = T.shift_left(agg[0], T.uint32(16))
    bar_sync()
    bp: T.uint32 = ld_shared_u32(scan, n_warps)
    if tx > 0:
        out[0] = bp + out[0]


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
    _scan_seed_aggregate(scan, agg, wpre)
    for w in range(1, n_warps):
        _scan_fold_warp(scan, agg, wpre, tx, w)
    _scan_apply_prefix(scan, out, agg, wpre, tx, n_warps)


@T.macro
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
    prefixes = T.alloc_local((items_per_thread,), "uint32")
    slots = T.alloc_local((items_per_thread,), "int32")

    # ResetCounters: one packed zero per padded lane (:346-355).
    for lane in T.unroll(PADDED_COUNTER_LANES):
        st_shared_u32(counters32, lane * block_threads + tx, T.uint32(0))
        # counter view: &digit_counters[LANE][tid][0] as one packed word.

    for i in T.unroll(items_per_thread):
        digit: T.uint32 = emit_digit(keys[i], begin_bit, num_bits)
        sub_counter: T.int32 = T.cast(T.shift_right(digit, T.uint32(LOG_COUNTER_LANES)), "int32")
        counter_lane: T.int32 = T.cast(T.bitwise_and(digit, T.uint32(COUNTER_LANES - 1)), "int32")
        # &digit_counters[counter_lane][tid][sub_counter] over the uint16 view.
        slots[i] = (counter_lane * block_threads + tx) * PACKING_RATIO + sub_counter
        prefixes[i] = T.cast(ld_shared_u16(counters16, slots[i]), "uint32")
        st_shared_u16(counters16, slots[i], T.cast(prefixes[i] + T.uint32(1), "uint16"))

    bar_sync()

    # ScanCounters (:376-392): memoized raking upsweep, packed block scan, then
    # an exclusive downsweep back into the same words.
    # raking_grid[linear_tid][RAKING_SEGMENT] -- each thread rakes a CONTIGUOUS
    # nine-word segment, so the block scan runs over the flat array in index
    # order.  In the counter view that same flat index is `lane * BT + tid`, i.e.
    # digit-major and thread-minor, which is what makes the ranks group by digit.
    # Striding the segment instead would order the prefix by thread first.
    cached = T.alloc_local((RAKING_SEGMENT,), "uint32")
    partial = T.alloc_local((1,), "uint32")
    partial[0] = T.uint32(0)
    for j in T.unroll(RAKING_SEGMENT):
        cached[j] = ld_shared_u32(counters32, tx * RAKING_SEGMENT + j)
        partial[0] = partial[0] + cached[j]

    exclusive = T.alloc_local((1,), "uint32")
    incl = T.alloc_local((1,), "uint32")
    agg = T.alloc_local((1,), "uint32")
    wpre = T.alloc_local((1,), "uint32")
    emit_block_exclusive_sum_packed(scan, exclusive, incl, agg, wpre, tx, block_threads, partial[0])

    for j in T.unroll(RAKING_SEGMENT):
        st_shared_u32(counters32, tx * RAKING_SEGMENT + j, exclusive[0])
        exclusive[0] = exclusive[0] + cached[j]

    bar_sync()

    for i in T.unroll(items_per_thread):
        ranks[i] = T.cast(
            prefixes[i] + T.cast(ld_shared_u16(counters16, slots[i]), "uint32"), "int32"
        )


@T.macro
def emit_scatter_to_blocked_u32(buf, items_reg, ranks, tx, items_per_thread):
    """``BlockExchange::ScatterToBlocked`` for the 32-bit keys (``:600-629``)."""
    for i in T.unroll(items_per_thread):
        st_shared_u32(buf, padded_offset(ranks[i], items_per_thread), items_reg[i])
    bar_sync()
    for i in T.unroll(items_per_thread):
        items_reg[i] = ld_shared_u32(
            buf, padded_offset(tx * items_per_thread + i, items_per_thread)
        )


@T.macro
def emit_scatter_to_blocked_u16(buf, items_reg, ranks, tx, items_per_thread):
    """The same exchange for 16-bit paired values."""
    for i in T.unroll(items_per_thread):
        st_shared_u16(
            buf, padded_offset(ranks[i], items_per_thread), T.cast(items_reg[i], "uint16")
        )
    bar_sync()
    for i in T.unroll(items_per_thread):
        items_reg[i] = T.cast(
            ld_shared_u16(buf, padded_offset(tx * items_per_thread + i, items_per_thread)), "uint32"
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
