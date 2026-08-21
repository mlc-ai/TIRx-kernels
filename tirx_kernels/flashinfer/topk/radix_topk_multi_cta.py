# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer radix top-k multi-CTA port.

Ports ``RadixTopKKernel_Unified<1024, VEC_SIZE, SINGLE_CTA=false, DETERMINISTIC,
MODE, DType, int32_t>`` (``include/flashinfer/topk.cuh:1170``) -- the cross-CTA
specialization the launchers select when one CTA's shared memory cannot hold a
whole row, i.e. ``ctas_per_group = ceil_div(length, max_chunk_elements) >= 2``.

A group of ``ctas_per_group`` CTAs cooperates on one row: each CTA stages its own
chunk as monotone ordered keys, and the radix rounds are reduced across the group
through a per-group ``RadixRowState`` in global memory (``:139-146``) --
triple-buffered 256-bin histograms so each round costs a single group barrier,
plus an arrival counter and an output counter.  The group barrier
(``AdvanceRadixGroupBarrier``, ``:174-182``) is a monotonically increasing counter
with absolute phase targets: one fenced ``red.relaxed.gpu.global.add.s32`` arrival
per CTA, then every CTA spins on ``ld.global.acquire.gpu.b32`` until
``(phase + 1) * ctas_per_group``.  The last CTA to leave resets the state for the
next launch (``RadixGroupResetStateLastCTA``, ``:204-240``, issue #3610), which is
what makes the cached workspace safe to reuse across launches and across
deterministic/non-deterministic transitions.

The deterministic collect exchanges per-CTA ``gt``/``eq`` counts through a
``RadixDeterministicCollectScratch`` (``:147-152``) placed immediately after the
row states, and derives every output position from a fixed CTA-order prefix, so it
is bit-reproducible.  The non-deterministic collect claims output slots with plain
global atomics on ``output_counter``, so both the order and -- under ties -- the
selected set may differ run to run.

In-scope specialization: every shape the launchers map onto two or more CTAs per
row, i.e. ``length`` above the single-CTA chunk bound, for all three epilogue
modes, both collect paths, fp32/fp16/bf16, every reachable ``VEC_SIZE``, and the
runtime ``row_starts`` / ``page_table_row_starts`` / ``row_to_batch`` branches.
Out of scope, with the dispatch predicate that excludes each: per-row
``top_k_arr`` (``csrc/topk.cu`` always passes ``nullptr``); deterministic runs
needing more than ``RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP == 256`` CTAs per
group, which the launcher rejects with ``cudaErrorInvalidConfiguration`` and which
would need roughly 14.2M fp32 elements per row; groups larger than the SM count
(the ``num_groups == 0 -> 1`` clamp), which needs about 8.5M elements per row; and
the single-CTA specialization, which is ported separately as
``radix_topk_single_cta``.
"""

import math
from typing import Any

from tirx_kernels.flashinfer.topk.radix_topk_single_cta import (
    BLOCK_THREADS,
    DTYPES,
    MODES,
    RADIX,
    RAKING_SEGMENT,
    RAKING_STRIDE,
    RAKING_THREADS,
    _raking_offset,
    assert_reference_is_top_k,
    aux_elements,
    build_reference_launch,
    dtype_bytes,
    hardware_max_smem_optin,
    launch_plan,
    max_chunk_elements,
    num_rounds,
    ordered_dtype,
    page_table_batch,
    run_reference,
    scan_scratch_elements,
)
from tirx_kernels.flashinfer.utils.topk_harness import (
    alloc_outputs,
    assert_device_matches_compile_profile,
    compare_outputs,
    torch_dtype,
)
from tirx_kernels.flashinfer.utils.topk_radix import (
    atom_add_release_gpu_s32,
    atom_global_add_u32,
    atom_shared_add_u32,
    bar_sync,
    emit_selected,
    ld_acquire_gpu_s32,
    ld_global_bits,
    ld_global_u32,
    ld_shared_pair_u32,
    ld_shared_quad_u32,
    ld_shared_u16,
    ld_shared_u32,
    ld_shared_u64,
    red_release_gpu_add_s32,
    shfl_down_u32,
    st_global_u16,
    st_global_u32,
    st_release_gpu_s32,
    st_shared_pair_u32,
    st_shared_quad_u32,
    st_shared_u16,
    st_shared_u32,
    stage_vector,
    to_ordered_u16,
    to_ordered_u32,
    u64_hi,
    u64_lo,
    warp_inclusive_sum_u32,
)
from tirx_kernels.runner import bench
from tvm.script import tirx as T

# This specialization declares four shared scalars (:1194-1200), so the ordered
# keys start at byte 2064 while the launcher's header is sized for five (2080).
# ``shared_scalars[4]`` therefore aliases ``shared_ordered[0]`` and is never
# written on any multi-CTA branch.
NUM_SCALARS_MULTI_CTA = 4

# Staging iterations per thread above which the deeper unroll pays for itself.
STAGING_DEEP_UNROLL_TRIPS = 16

KERNEL_META = {"name": "radix_topk_multi_cta", "category": "flashinfer", "compute_capability": 10}

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# ---------------------------------------------------------------------------
# Per-group global state (topk.cuh:139-152), as uint32 element offsets.
#
#   struct RadixRowState {          offset (u32 words)
#     uint32 histogram[3][256];     0   .. 767
#     uint32 remaining_k;           768          (unused by this kernel)
#     uint32 prefix;                769          (unused by this kernel)
#     int    arrival_counter;       770
#     int    output_counter;        771
#     float  sum_topk;              772          (unused by this kernel)
#   };                              stride 773 words = 3092 bytes
#
#   struct RadixDeterministicCollectScratch {
#     uint32 gt_count[256];         0   .. 255
#     uint32 eq_count[256];         256 .. 511
#   };                              stride 512 words = 2048 bytes
#
# The scratch array starts immediately after the `num_groups` row states
# (MaybeGetRadixDeterministicCollectScratchBuffer, :155-160).
# ---------------------------------------------------------------------------
ROW_STATE_WORDS = 3 * RADIX + 5  # 773
HIST_WORDS = 3 * RADIX  # 768
ARRIVAL_COUNTER_WORD = 3 * RADIX + 2  # 770
OUTPUT_COUNTER_WORD = 3 * RADIX + 3  # 771
DET_SCRATCH_WORDS = 2 * RADIX  # 512
MAX_DETERMINISTIC_CTAS_PER_GROUP = 256

# The Python wrapper hands the kernel a cached 1 MiB zero-initialized buffer
# (flashinfer/topk.py:623-628); mirror that so the workspace is never the reason a
# shape behaves differently from the source.
WORKSPACE_BYTES = 1024 * 1024
WORKSPACE_WORDS = WORKSPACE_BYTES // 4


def workspace_words(num_groups: int, deterministic: bool) -> int:
    """uint32 words the group state and optional collect scratch occupy."""
    words = num_groups * ROW_STATE_WORDS
    if deterministic:
        words += num_groups * DET_SCRATCH_WORDS
    return words


def det_scratch_base_word(num_groups: int) -> int:
    """First word of the deterministic collect scratch array."""
    return num_groups * ROW_STATE_WORDS


def _validate(
    dtype: str,
    mode: str,
    num_rows: int,
    length: int,
    k: int,
    deterministic: bool,
    scalar_addressing: bool,
) -> dict[str, Any]:
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if num_rows < 1 or length < 1 or k < 1:
        raise ValueError(f"num_rows={num_rows}, length={length}, k={k} must be positive")
    if k > length:
        raise ValueError(f"k={k} exceeds length={length}")
    plan = launch_plan(dtype, mode, length, num_rows, deterministic, scalar_addressing)
    if plan["single_cta"]:
        raise ValueError(
            f"({dtype}, len={length}, det={deterministic}) maps onto one CTA per row: "
            "outside the multi-CTA dispatch domain"
        )
    if deterministic and plan["ctas_per_group"] > MAX_DETERMINISTIC_CTAS_PER_GROUP:
        raise ValueError(
            f"ctas_per_group={plan['ctas_per_group']} exceeds the deterministic cap "
            f"{MAX_DETERMINISTIC_CTAS_PER_GROUP}; the launcher returns "
            "cudaErrorInvalidConfiguration here"
        )
    needed = workspace_words(plan["num_groups"], deterministic)
    if needed > WORKSPACE_WORDS:
        raise ValueError(f"workspace needs {needed} words, buffer holds {WORKSPACE_WORDS}")
    return plan


# ---------------------------------------------------------------------------
# Domain helpers: pick lengths that land on a chosen ctas_per_group and VEC_SIZE.
# ---------------------------------------------------------------------------
def chunk_bound(dtype: str, deterministic: bool) -> int:
    """``max_chunk_elements`` for this (dtype, det); vec-independent in range."""
    widest = 16 // dtype_bytes(dtype)
    probe = widest * 16384
    return max_chunk_elements(dtype, probe, deterministic, False)


def length_for(dtype: str, deterministic: bool, ctas: int, vec: int) -> int | None:
    """Smallest length giving exactly ``ctas`` CTAs per group at this VEC_SIZE."""
    bound = chunk_bound(dtype, deterministic)
    widest = 16 // dtype_bytes(dtype)
    lo = bound * (ctas - 1) + 1
    hi = bound * ctas
    for length in range(lo, min(lo + 8 * widest, hi) + 1):
        if math.gcd(widest, length) != vec:
            continue
        plan = launch_plan(dtype, "basic", length, 1, deterministic)
        if plan["ctas_per_group"] == ctas and not plan["single_cta"]:
            return length
    return None


def reachable_vecs(dtype: str) -> list[int]:
    widest = 16 // dtype_bytes(dtype)
    out, v = [], widest
    while v >= 1:
        out.append(v)
        v //= 2
    return out


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
def get_kernel(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 8,
    length: int = 65536,
    k: int = 256,
    deterministic: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    short_rows: bool = False,
    reuse_workspace: bool = False,
    **kwargs,
):
    """Return the TIRx specialization for one multi-CTA launcher dispatch cell."""
    scalar_addressing = row_starts and mode != "basic"
    plan = _validate(dtype, mode, num_rows, length, k, deterministic, scalar_addressing)
    ordered = ordered_dtype(dtype)
    is32 = ordered == "uint32"
    obits = 32 if is32 else 16
    rounds = num_rounds(dtype)
    vec = plan["vec_size"]
    chunk = plan["chunk_size"]
    grid = plan["grid"]
    ctas_per_group = plan["ctas_per_group"]
    num_groups = plan["num_groups"]
    load_bytes = vec * dtype_bytes(dtype)
    aux_elems = aux_elements(mode, num_rows, length, row_to_batch)
    scan_elems = scan_scratch_elements(deterministic)
    basic = mode == "basic"
    page_table = mode == "page_table"
    ragged = mode == "ragged"
    basic_trivial = basic and k >= length
    # The source marks the staging loop `#pragma unroll 2` to keep independent
    # global loads in flight.  Under this toolchain an unroll of 2 leaves that
    # latency exposed once a thread has enough iterations to amortize a deeper
    # one: on f16_rag_r4_l115186_k256_vec2 (28.1 staging iterations per thread)
    # it measured 7.19 warp-cycles of `long_scoreboard` per issued instruction
    # against the source's 3.74, and unrolling by 4 cut that to 2.89 -- 33.5us
    # against the source's 39.0us, a 0.915x shape becoming 1.21x.  Unrolling by 6
    # or 8 regresses again on register pressure.
    #
    # Short staging loops go the other way: at 6.8 iterations per thread
    # (f32_basic_det_r1_l55468_k256) the remainder dominates and unroll 4
    # measured 0.95x against 1.11x for unroll 2.  Pick the factor from the
    # compile-time iteration count rather than from any named shape; the two
    # regimes are far apart (nothing in the domain falls between 12.8 and 28.1).
    staging_trips = chunk // (BLOCK_THREADS * vec)
    stage_unroll = 4 if staging_trips >= STAGING_DEEP_UNROLL_TRIPS else 2
    total_iterations = (num_rows + num_groups - 1) // num_groups
    det_scratch_base = det_scratch_base_word(num_groups)

    def to_ordered(bits):
        return to_ordered_u32(bits) if is32 else to_ordered_u16(bits)

    def ld_key(buf, i):
        return ld_shared_u32(buf, i) if is32 else ld_shared_u16(buf, i)

    def st_key(buf, i, v):
        st_shared_u32(buf, i, v) if is32 else st_shared_u16(buf, i, v)

    @T.prim_func
    def radix_topk_multi_cta(
        input_h: T.handle,
        output_indices_h: T.handle,
        output_values_h: T.handle,
        aux_data_h: T.handle,
        lengths_h: T.handle,
        row_starts_h: T.handle,
        page_table_row_starts_h: T.handle,
        row_to_batch_h: T.handle,
        row_states_h: T.handle,
        aux_stride: T.int64,
    ):
        T.func_attr({"global_symbol": "radix_topk_multi_cta"})
        inp = T.match_buffer(input_h, (num_rows * length,), dtype, scope="global")
        out_idx = T.match_buffer(output_indices_h, (num_rows * k,), "int32", scope="global")
        out_val = T.match_buffer(output_values_h, (num_rows * k,), dtype, scope="global")
        aux = T.match_buffer(aux_data_h, (aux_elems,), "int32", scope="global")
        lengths_g = T.match_buffer(lengths_h, (num_rows,), "int32", scope="global")
        row_starts_g = T.match_buffer(row_starts_h, (num_rows,), "int32", scope="global")
        pt_starts_g = T.match_buffer(page_table_row_starts_h, (num_rows,), "int32", scope="global")
        row_to_batch_g = T.match_buffer(row_to_batch_h, (num_rows,), "int32", scope="global")
        state = T.match_buffer(row_states_h, (WORKSPACE_WORDS,), "uint32", scope="global")
        T.device_entry()

        cta = T.cta_id([grid])
        tx = T.thread_id([BLOCK_THREADS])

        pool = T.SMEMPool()
        s_hist = pool.alloc((RADIX,), "uint32")
        s_suffix = pool.alloc((RADIX,), "uint32")
        s_scalars = pool.alloc((NUM_SCALARS_MULTI_CTA,), "uint32")
        s_ordered = pool.alloc((chunk,), ordered, align=16)
        if deterministic:
            s_scan = pool.alloc((scan_elems,), "uint32", align=16)
        s_last = pool.alloc((1,), "uint32")
        pool.commit()

        # --- group identity and per-group state (:1187-1201) --------------
        group_id: T.int32 = T.truncdiv(cta, T.int32(ctas_per_group))
        cta_in_group: T.int32 = cta - group_id * T.int32(ctas_per_group)
        st_base: T.int32 = group_id * T.int32(ROW_STATE_WORDS)
        arrival_word: T.int32 = st_base + T.int32(ARRIVAL_COUNTER_WORD)
        output_word: T.int32 = st_base + T.int32(OUTPUT_COUNTER_WORD)
        det_base: T.int32 = T.int32(det_scratch_base) + group_id * T.int32(DET_SCRATCH_WORDS)
        barrier_phase: T.int32 = T.int32(0)

        # --- persistent row loop (:1218-1220) -----------------------------
        for it in T.serial(0, total_iterations):
            # `T.serial` rather than a `while`: the trip count is static, so a
            # single-iteration grid (num_rows <= num_groups, i.e. every shape but
            # the fully-occupied ones) folds `it` to 0 and turns `global_round`
            # and the trivial branch's next-buffer index into constants.  A
            # mutable while-loop counter blocks that folding.
            row_idx: T.int32 = group_id + it * T.int32(num_groups)
            if row_idx < num_rows:
                # --- per-row header (:1221-1240) --------------------------
                row_start: T.int32 = T.int32(0)
                page_start: T.int32 = T.int32(0)
                row_len = length
                if not basic:
                    if row_starts:
                        row_start = T.reinterpret("int32", ld_global_u32(row_starts_g, row_idx))
                    if page_table:
                        if page_table_row_starts:
                            page_start = T.reinterpret("int32", ld_global_u32(pt_starts_g, row_idx))
                        else:
                            page_start = row_start
                    row_len_rt: T.int32 = T.reinterpret("int32", ld_global_u32(lengths_g, row_idx))
                    row_len = row_len_rt
                row_in: T.int64 = T.cast(row_idx, "int64") * T.int64(length) + T.cast(
                    row_start, "int64"
                )
                row_out: T.int64 = T.cast(row_idx, "int64") * T.int64(k)

                # --- mode trivial early-out (:1243-1307) ------------------
                if not (basic and not basic_trivial):
                    take_main: T.int32 = T.int32(1)
                if basic:
                    if basic_trivial:
                        # Basic SPLITS the identity emit across the group (:1246-1255).
                        take_main = T.int32(0)
                        t_start: T.int32 = cta_in_group * T.int32(chunk)
                        t_actual: T.int32 = T.int32(0)
                        if t_start < row_len:
                            t_actual = T.min(T.int32(chunk), row_len - t_start)
                        for i0 in T.serial(tx, t_actual, step=BLOCK_THREADS):
                            g0: T.int32 = t_start + i0
                            if g0 < k:
                                slot0: T.int64 = row_out + T.cast(g0, "int64")
                                st_global_u32(out_idx, slot0, T.reinterpret("uint32", g0))
                                bits0 = ld_global_bits(
                                    inp,
                                    T.cast(row_idx, "int64") * T.int64(length)
                                    + T.cast(g0, "int64"),
                                    is32,
                                )
                                if is32:
                                    st_global_u32(out_val, slot0, bits0)
                                else:
                                    st_global_u16(out_val, slot0, bits0)
                batch_idx: T.int32 = row_idx
                offset: T.int32 = T.int32(0)
                if page_table:
                    if row_to_batch:
                        batch_idx = T.reinterpret("int32", ld_global_u32(row_to_batch_g, row_idx))
                    if row_len <= k:
                        # NOT split: every CTA writes the whole [0, k) range (:1272-1276).
                        take_main = T.int32(0)
                        src0: T.int64 = T.cast(batch_idx, "int64") * aux_stride
                        for i1 in T.serial(tx, k, step=BLOCK_THREADS):
                            page_id: T.int32 = T.int32(-1)
                            if i1 < row_len:
                                page_id = T.reinterpret(
                                    "int32",
                                    ld_global_u32(aux, src0 + T.cast(page_start + i1, "int64")),
                                )
                            st_global_u32(
                                out_idx,
                                row_out + T.cast(i1, "int64"),
                                T.reinterpret("uint32", page_id),
                            )
                if ragged:
                    offset = T.reinterpret("int32", ld_global_u32(aux, T.cast(row_idx, "int64")))
                    if row_len <= k:
                        take_main = T.int32(0)
                        for i2 in T.serial(tx, k, step=BLOCK_THREADS):
                            val2: T.int32 = T.int32(-1)
                            if i2 < row_len:
                                val2 = i2 + offset
                            st_global_u32(
                                out_idx,
                                row_out + T.cast(i2, "int64"),
                                T.reinterpret("uint32", val2),
                            )
                if not (basic and not basic_trivial):
                    # Common tail of the trivial branches (:1257-1266, :1277-1286,
                    # :1295-1304): CTA 0 clears the buffer the next iteration opens
                    # on, so a skipped row cannot leave it dirty.
                    if take_main == T.int32(0):
                        next_first: T.int32 = T.truncmod(
                            (it + T.int32(1)) * T.int32(rounds), T.int32(3)
                        )
                        if cta_in_group == 0:
                            for bz in T.serial(tx, RADIX, step=BLOCK_THREADS):
                                st_global_u32(
                                    state, st_base + next_first * T.int32(RADIX) + bz, T.uint32(0)
                                )

                main_guard = True if (basic and not basic_trivial) else (take_main == 1)
                if main_guard:
                    chunk_start: T.int32 = cta_in_group * T.int32(chunk)
                    actual: T.int32 = T.int32(0)
                    if chunk_start < row_len:
                        actual = T.min(T.int32(chunk), row_len - chunk_start)
                    chunk_in: T.int64 = row_in + T.cast(chunk_start, "int64")

                    # === Stage 1: stage this CTA's chunk (:602-623) ========
                    aligned_rt: T.int32 = T.truncdiv(actual, T.int32(vec)) * T.int32(vec)
                    for iv in T.serial(
                        tx * vec, aligned_rt, step=BLOCK_THREADS * vec, unroll=stage_unroll
                    ):
                        stage_vector(
                            inp, s_ordered, chunk_in, iv, vec, load_bytes, is32, to_ordered, st_key
                        )
                    for it_tail in T.serial(aligned_rt + tx, actual, step=BLOCK_THREADS):
                        st_key(
                            s_ordered,
                            it_tail,
                            to_ordered(
                                ld_global_bits(inp, chunk_in + T.cast(it_tail, "int64"), is32)
                            ),
                        )
                    bar_sync()

                    # scalar caches (:670-677); no shared output counter here
                    if tx == 0:
                        st_shared_pair_u32(s_scalars, 0, T.uint32(0), T.uint32(k))
                    bar_sync()

                    # === initial group rendezvous (:679-687) ===============
                    # --- group barrier (AdvanceRadixGroupBarrier, :174-182) ---------
                    # One fenced arrival from thread 0 (:176), an acquire spin to the
                    # absolute target (:179), then TWO CTA barriers: wait_ge ends with
                    # one (:133) and the phase bump is followed by another (:182).
                    # Complete every CTA-local state update before thread 0
                    # publishes the group arrival.  The following acquire
                    # spin is the inter-CTA handoff; without this CTA
                    # rendezvous, another warp could still be folding its
                    # histogram when the consumer observes the arrival.
                    bar_sync()
                    if tx == 0:
                        red_release_gpu_add_s32(state, arrival_word, T.int32(1))
                    bar_target0: T.int32 = (barrier_phase + T.int32(1)) * T.int32(ctas_per_group)
                    if tx == 0:
                        spin0: T.int32 = ld_acquire_gpu_s32(state, arrival_word)
                        while spin0 < bar_target0:
                            spin0 = ld_acquire_gpu_s32(state, arrival_word)
                    bar_sync()
                    barrier_phase = barrier_phase + T.int32(1)
                    bar_sync()
                    if T.And(cta_in_group == 0, tx == 0):
                        st_release_gpu_s32(state, output_word, T.int32(0))

                    # === Stage 2: NUM_ROUNDS rounds, reduced group-wide ====
                    for rnd in T.serial(0, rounds):
                        global_round: T.int32 = it * T.int32(rounds) + rnd
                        cur_buf: T.int32 = T.truncmod(global_round, T.int32(3))
                        next_buf: T.int32 = T.truncmod(global_round + T.int32(1), T.int32(3))
                        cur_hist: T.int32 = st_base + cur_buf * T.int32(RADIX)
                        next_hist: T.int32 = st_base + next_buf * T.int32(RADIX)
                        shift: T.int32 = T.int32(obits) - (rnd + 1) * 8
                        mask: T.uint32 = T.uint32(0)
                        if rnd != 0:
                            mask = T.shift_left(
                                T.uint32(0xFFFFFFFF), T.cast(T.int32(obits) - rnd * 8, "uint32")
                            )
                        packed: T.uint64 = ld_shared_u64(s_scalars, 0)
                        prefix: T.uint32 = u64_lo(packed)
                        remaining_k: T.uint32 = u64_hi(packed)

                        for bh in T.serial(tx, RADIX, step=BLOCK_THREADS):
                            st_shared_u32(s_hist, bh, T.uint32(0))
                        bar_sync()
                        for ih in T.serial(tx, actual, step=BLOCK_THREADS, unroll=2):
                            key: T.uint32 = T.cast(ld_key(s_ordered, ih), "uint32")
                            if T.bitwise_and(key, mask) == prefix:
                                bucket: T.uint32 = T.bitwise_and(
                                    T.shift_right(key, T.cast(shift, "uint32")), T.uint32(0xFF)
                                )
                                T.evaluate(
                                    atom_shared_add_u32(
                                        s_hist, T.cast(bucket, "int32"), T.uint32(1)
                                    )
                                )
                        bar_sync()

                        # fold this CTA's histogram into the group's buffer (:727-736)
                        for ba in T.serial(tx, RADIX, step=BLOCK_THREADS):
                            hval: T.uint32 = ld_shared_u32(s_hist, ba)
                            if hval > T.uint32(0):
                                T.evaluate(atom_global_add_u32(state, cur_hist + ba, hval))
                        if cta_in_group == 0:
                            for bc in T.serial(tx, RADIX, step=BLOCK_THREADS):
                                st_global_u32(state, next_hist + bc, T.uint32(0))
                        # --- group barrier (AdvanceRadixGroupBarrier, :174-182) ---------
                        # One fenced arrival from thread 0 (:176), an acquire spin to the
                        # absolute target (:179), then TWO CTA barriers: wait_ge ends with
                        # one (:133) and the phase bump is followed by another (:182).
                        bar_sync()
                        if tx == 0:
                            red_release_gpu_add_s32(state, arrival_word, T.int32(1))
                        bar_target1: T.int32 = (barrier_phase + T.int32(1)) * T.int32(
                            ctas_per_group
                        )
                        if tx == 0:
                            spin1: T.int32 = ld_acquire_gpu_s32(state, arrival_word)
                            while spin1 < bar_target1:
                                spin1 = ld_acquire_gpu_s32(state, arrival_word)
                        bar_sync()
                        barrier_phase = barrier_phase + T.int32(1)
                        bar_sync()
                        for bs in T.serial(tx, RADIX, step=BLOCK_THREADS):
                            st_shared_u32(s_suffix, bs, ld_global_u32(state, cur_hist + bs))
                        bar_sync()

                        # RadixSuffixSum (:389-406)
                        for step in T.unroll(8):
                            stride: T.int32 = T.shift_left(T.int32(1), step)
                            acc: T.uint32 = T.uint32(0)
                            if tx < RADIX:
                                acc = ld_shared_u32(s_suffix, tx)
                                if tx + stride < RADIX:
                                    acc = acc + ld_shared_u32(s_suffix, tx + stride)
                            bar_sync()
                            if tx < RADIX:
                                st_shared_u32(s_suffix, tx, acc)
                            bar_sync()

                        # threshold bucket (:753-767)
                        if tx == 0:
                            st_shared_pair_u32(s_scalars, 2, T.uint32(0), remaining_k)
                        bar_sync()
                        if tx < RADIX:
                            count_ge: T.uint32 = ld_shared_u32(s_suffix, tx)
                            count_gt: T.uint32 = T.uint32(0)
                            if tx + 1 < RADIX:
                                count_gt = ld_shared_u32(s_suffix, tx + 1)
                            if T.And(count_ge >= remaining_k, count_gt < remaining_k):
                                st_shared_pair_u32(
                                    s_scalars, 2, T.cast(tx, "uint32"), remaining_k - count_gt
                                )
                        bar_sync()
                        if tx == 0:
                            found = ld_shared_pair_u32(s_scalars, 2)
                            st_shared_pair_u32(
                                s_scalars,
                                0,
                                T.bitwise_or(
                                    prefix, T.shift_left(found[0], T.cast(shift, "uint32"))
                                ),
                                found[1],
                            )
                        bar_sync()

                    pivot: T.uint32 = ld_shared_u32(s_scalars, 0)
                    if not is32:
                        pivot = T.bitwise_and(pivot, T.uint32(0xFFFF))

                    # === Stage 3: CTA-local gt (and eq) counts (:777-830) ==
                    if tx == 0:
                        if deterministic:
                            st_shared_pair_u32(s_suffix, 0, T.uint32(0), T.uint32(0))
                        else:
                            st_shared_u32(s_suffix, 0, T.uint32(0))
                    bar_sync()
                    my_gt: T.uint32 = T.uint32(0)
                    my_eq: T.uint32 = T.uint32(0)
                    for ic in T.serial(tx, actual, step=BLOCK_THREADS, unroll=2):
                        key2: T.uint32 = T.cast(ld_key(s_ordered, ic), "uint32")
                        my_gt = my_gt + T.Select(key2 > pivot, T.uint32(1), T.uint32(0))
                        if deterministic:
                            my_eq = my_eq + T.Select(key2 == pivot, T.uint32(1), T.uint32(0))
                    for step in T.unroll(5):
                        delta: T.int32 = T.shift_right(T.int32(16), step)
                        my_gt = my_gt + shfl_down_u32(my_gt, delta)
                        if deterministic:
                            my_eq = my_eq + shfl_down_u32(my_eq, delta)
                    lane: T.int32 = T.bitwise_and(tx, T.int32(31))
                    if T.And(lane == 0, my_gt > T.uint32(0)):
                        T.evaluate(atom_shared_add_u32(s_suffix, 0, my_gt))
                    if deterministic:
                        if T.And(lane == 0, my_eq > T.uint32(0)):
                            T.evaluate(atom_shared_add_u32(s_suffix, 1, my_eq))
                    bar_sync()
                    gt_count: T.uint32 = ld_shared_u32(s_suffix, 0)

                    # === Stage 3b: epilogue-scope aux (:1344-1345, :1373) ==
                    src_base: T.int64 = T.int64(0)
                    if page_table:
                        if row_to_batch:
                            batch_idx = T.reinterpret(
                                "int32", ld_global_u32(row_to_batch_g, row_idx)
                            )
                        src_base = T.cast(batch_idx, "int64") * aux_stride
                    if ragged:
                        offset = T.reinterpret(
                            "int32", ld_global_u32(aux, T.cast(row_idx, "int64"))
                        )

                    if not deterministic:
                        # === Stage 4a: non-deterministic collect (:906-963) =
                        if tx == 0:
                            st_shared_u32(s_hist, 0, T.uint32(0))
                            if gt_count > T.uint32(0):
                                st_shared_u32(
                                    s_hist, 1, atom_global_add_u32(state, output_word, gt_count)
                                )
                        bar_sync()
                        for i in T.serial(tx, actual, step=BLOCK_THREADS, unroll=2):
                            key3: T.uint32 = T.cast(ld_key(s_ordered, i), "uint32")
                            if key3 > pivot:
                                local_pos: T.uint32 = atom_shared_add_u32(s_hist, 0, T.uint32(1))
                                pos: T.int32 = T.cast(ld_shared_u32(s_hist, 1) + local_pos, "int32")
                                emit_selected(
                                    out_idx,
                                    out_val,
                                    row_out,
                                    chunk_start + i,
                                    key3,
                                    pos,
                                    offset,
                                    basic,
                                    ragged,
                                    is32,
                                    dtype,
                                )
                        # --- group barrier (AdvanceRadixGroupBarrier, :174-182) ---------
                        # One fenced arrival from thread 0 (:176), an acquire spin to the
                        # absolute target (:179), then TWO CTA barriers: wait_ge ends with
                        # one (:133) and the phase bump is followed by another (:182).
                        bar_sync()
                        if tx == 0:
                            red_release_gpu_add_s32(state, arrival_word, T.int32(1))
                        bar_target2: T.int32 = (barrier_phase + T.int32(1)) * T.int32(
                            ctas_per_group
                        )
                        if tx == 0:
                            spin2: T.int32 = ld_acquire_gpu_s32(state, arrival_word)
                            while spin2 < bar_target2:
                                spin2 = ld_acquire_gpu_s32(state, arrival_word)
                        bar_sync()
                        barrier_phase = barrier_phase + T.int32(1)
                        bar_sync()
                        for i in T.serial(tx, actual, step=BLOCK_THREADS, unroll=2):
                            key4: T.uint32 = T.cast(ld_key(s_ordered, i), "uint32")
                            if key4 == pivot:
                                pos2: T.int32 = T.cast(
                                    atom_global_add_u32(state, output_word, T.uint32(1)), "int32"
                                )
                                if pos2 < k:
                                    emit_selected(
                                        out_idx,
                                        out_val,
                                        row_out,
                                        chunk_start + i,
                                        pivot,
                                        pos2,
                                        offset,
                                        basic,
                                        ragged,
                                        is32,
                                        dtype,
                                    )
                    else:
                        # === Stage 4b: deterministic collect (:1048-1138) ==
                        eq_count: T.uint32 = ld_shared_u32(s_suffix, 1)
                        if tx == 0:
                            st_shared_u32(s_hist, 1, T.uint32(0))
                            st_shared_u32(s_hist, 4, T.uint32(0))
                            st_global_u32(state, det_base + cta_in_group, gt_count)
                            st_global_u32(state, det_base + T.int32(RADIX) + cta_in_group, eq_count)
                        # --- group barrier (AdvanceRadixGroupBarrier, :174-182) ---------
                        # One fenced arrival from thread 0 (:176), an acquire spin to the
                        # absolute target (:179), then TWO CTA barriers: wait_ge ends with
                        # one (:133) and the phase bump is followed by another (:182).
                        bar_sync()
                        if tx == 0:
                            red_release_gpu_add_s32(state, arrival_word, T.int32(1))
                        bar_target3: T.int32 = (barrier_phase + T.int32(1)) * T.int32(
                            ctas_per_group
                        )
                        if tx == 0:
                            spin3: T.int32 = ld_acquire_gpu_s32(state, arrival_word)
                            while spin3 < bar_target3:
                                spin3 = ld_acquire_gpu_s32(state, arrival_word)
                        bar_sync()
                        barrier_phase = barrier_phase + T.int32(1)
                        bar_sync()
                        if tx == 0:
                            gt_prefix: T.uint32 = T.uint32(0)
                            row_total_gt: T.uint32 = T.uint32(0)
                            eq_prefix: T.uint32 = T.uint32(0)
                            for c in T.serial(0, ctas_per_group):
                                c_gt: T.uint32 = ld_global_u32(state, det_base + c)
                                c_eq: T.uint32 = ld_global_u32(state, det_base + T.int32(RADIX) + c)
                                if c < cta_in_group:
                                    gt_prefix = gt_prefix + c_gt
                                    eq_prefix = eq_prefix + c_eq
                                row_total_gt = row_total_gt + c_gt
                            eq_needed: T.uint32 = T.uint32(0)
                            if T.uint32(k) > row_total_gt:
                                eq_needed = T.uint32(k) - row_total_gt
                            st_shared_quad_u32(
                                s_hist, 0, gt_prefix, eq_prefix, row_total_gt, eq_needed
                            )
                            st_shared_u32(s_hist, 4, T.uint32(0))
                            if eq_needed > eq_prefix:
                                st_shared_u32(s_hist, 4, T.min(eq_count, eq_needed - eq_prefix))
                        bar_sync()
                        plan_v = ld_shared_quad_u32(s_hist, 0)
                        gt_base: T.uint32 = plan_v[0]
                        eq_base: T.uint32 = plan_v[2] + plan_v[1]
                        eq_limit: T.uint32 = ld_shared_u32(s_hist, 4)
                        gt_limit: T.uint32 = T.uint32(0)
                        if T.uint32(k) > gt_base:
                            gt_limit = T.uint32(k) - gt_base

                        if eq_limit == T.uint32(0):
                            sel: T.uint32 = T.uint32(0)
                            for id1 in T.serial(tx, actual, step=BLOCK_THREADS):
                                key5: T.uint32 = T.cast(ld_key(s_ordered, id1), "uint32")
                                sel = sel + T.Select(key5 > pivot, T.uint32(1), T.uint32(0))
                            rake_off: T.int32 = _raking_offset(tx)
                            st_shared_u32(s_scan, rake_off, sel)
                            bar_sync()
                            if tx < RAKING_THREADS:
                                rake_base: T.int32 = tx * RAKING_STRIDE
                                cache = T.alloc_local((RAKING_SEGMENT,), "uint32")
                                total: T.uint32 = T.uint32(0)
                                for j in T.unroll(RAKING_SEGMENT):
                                    cache[j] = ld_shared_u32(s_scan, rake_base + j)
                                    total = total + cache[j]
                                run: T.uint32 = warp_inclusive_sum_u32(total, tx) - total
                                for j in T.unroll(RAKING_SEGMENT):
                                    st_shared_u32(s_scan, rake_base + j, run)
                                    run = run + cache[j]
                            bar_sync()
                            pre: T.uint32 = ld_shared_u32(s_scan, rake_off)
                            if T.And(sel > T.uint32(0), pre < gt_limit):
                                emit_pos: T.uint32 = pre
                                emit_end: T.uint32 = pre + sel
                                if emit_end > gt_limit:
                                    emit_end = gt_limit
                                done: T.int32 = 0
                                for id3 in T.serial(tx, actual, step=BLOCK_THREADS):
                                    if done == 0:
                                        key6: T.uint32 = T.cast(ld_key(s_ordered, id3), "uint32")
                                        if key6 > pivot:
                                            emit_selected(
                                                out_idx,
                                                out_val,
                                                row_out,
                                                chunk_start + id3,
                                                key6,
                                                T.cast(gt_base + emit_pos, "int32"),
                                                offset,
                                                basic,
                                                ragged,
                                                is32,
                                                dtype,
                                            )
                                            emit_pos = emit_pos + T.uint32(1)
                                            if emit_pos == emit_end:
                                                done = 1
                            bar_sync()
                        else:
                            sel_gt: T.uint32 = T.uint32(0)
                            sel_eq: T.uint32 = T.uint32(0)
                            for id4 in T.serial(tx, actual, step=BLOCK_THREADS):
                                key7: T.uint32 = T.cast(ld_key(s_ordered, id4), "uint32")
                                sel_gt = sel_gt + T.Select(key7 > pivot, T.uint32(1), T.uint32(0))
                                sel_eq = sel_eq + T.Select(key7 == pivot, T.uint32(1), T.uint32(0))
                            rake_off2: T.int32 = _raking_offset(tx) * 2
                            st_shared_pair_u32(s_scan, rake_off2, sel_gt, sel_eq)
                            bar_sync()
                            if tx < RAKING_THREADS:
                                rake_base2: T.int32 = tx * RAKING_STRIDE * 2
                                cache_gt = T.alloc_local((RAKING_SEGMENT,), "uint32")
                                cache_eq = T.alloc_local((RAKING_SEGMENT,), "uint32")
                                total_gt: T.uint32 = T.uint32(0)
                                total_eq: T.uint32 = T.uint32(0)
                                for j in T.unroll(RAKING_SEGMENT):
                                    seg = ld_shared_pair_u32(s_scan, rake_base2 + j * 2)
                                    cache_gt[j] = seg[0]
                                    cache_eq[j] = seg[1]
                                    total_gt = total_gt + cache_gt[j]
                                    total_eq = total_eq + cache_eq[j]
                                run_gt: T.uint32 = warp_inclusive_sum_u32(total_gt, tx) - total_gt
                                run_eq: T.uint32 = warp_inclusive_sum_u32(total_eq, tx) - total_eq
                                for j in T.unroll(RAKING_SEGMENT):
                                    st_shared_pair_u32(s_scan, rake_base2 + j * 2, run_gt, run_eq)
                                    run_gt = run_gt + cache_gt[j]
                                    run_eq = run_eq + cache_eq[j]
                            bar_sync()
                            seg_out = ld_shared_pair_u32(s_scan, rake_off2)
                            seg_gt: T.uint32 = seg_out[0]
                            seg_eq: T.uint32 = seg_out[1]
                            cur_gt: T.uint32 = seg_gt
                            cur_eq: T.uint32 = seg_eq
                            for i8 in T.serial(tx, actual, step=BLOCK_THREADS):
                                key8: T.uint32 = T.cast(ld_key(s_ordered, i8), "uint32")
                                if T.And(key8 > pivot, cur_gt < gt_limit):
                                    emit_selected(
                                        out_idx,
                                        out_val,
                                        row_out,
                                        chunk_start + i8,
                                        key8,
                                        T.cast(gt_base + cur_gt, "int32"),
                                        offset,
                                        basic,
                                        ragged,
                                        is32,
                                        dtype,
                                    )
                                    cur_gt = cur_gt + T.uint32(1)
                                elif T.And(key8 == pivot, cur_eq < eq_limit):
                                    emit_selected(
                                        out_idx,
                                        out_val,
                                        row_out,
                                        chunk_start + i8,
                                        key8,
                                        T.cast(eq_base + cur_eq, "int32"),
                                        offset,
                                        basic,
                                        ragged,
                                        is32,
                                        dtype,
                                    )
                                    cur_eq = cur_eq + T.uint32(1)
                            bar_sync()

                    # === Stage 5: PageTable gather, split (:1359-1371) =====
                    if page_table:
                        # --- group barrier (AdvanceRadixGroupBarrier, :174-182) ---------
                        # One fenced arrival from thread 0 (:176), an acquire spin to the
                        # absolute target (:179), then TWO CTA barriers: wait_ge ends with
                        # one (:133) and the phase bump is followed by another (:182).
                        bar_sync()
                        if tx == 0:
                            red_release_gpu_add_s32(state, arrival_word, T.int32(1))
                        bar_target4: T.int32 = (barrier_phase + T.int32(1)) * T.int32(
                            ctas_per_group
                        )
                        if tx == 0:
                            spin4: T.int32 = ld_acquire_gpu_s32(state, arrival_word)
                            while spin4 < bar_target4:
                                spin4 = ld_acquire_gpu_s32(state, arrival_word)
                        bar_sync()
                        barrier_phase = barrier_phase + T.int32(1)
                        bar_sync()
                        elems: T.int32 = T.truncdiv(
                            T.int32(k) + T.int32(ctas_per_group) - 1, T.int32(ctas_per_group)
                        )
                        my_start: T.int32 = cta_in_group * elems
                        my_end: T.int32 = T.min(my_start + elems, T.int32(k))
                        for ig in T.serial(my_start + tx, my_end, step=BLOCK_THREADS):
                            out_slot: T.int64 = row_out + T.cast(ig, "int64")
                            gidx: T.int32 = T.reinterpret("int32", ld_global_u32(out_idx, out_slot))
                            gathered: T.uint32 = ld_global_u32(
                                aux, src_base + T.cast(page_start + gidx, "int64")
                            )
                            st_global_u32(out_idx, out_slot, gathered)

        # === exit: the last CTA of the group resets the state (:203-238) ===
        bar_sync()
        if tx == 0:
            old: T.int32 = atom_add_release_gpu_s32(state, arrival_word, T.int32(1))
            st_shared_u32(
                s_last,
                0,
                T.Select(
                    old + T.int32(1) == (barrier_phase + T.int32(1)) * ctas_per_group,
                    T.uint32(1),
                    T.uint32(0),
                ),
            )
        bar_sync()
        if ld_shared_u32(s_last, 0) != T.uint32(0):
            for bufi in T.serial(0, 3):
                for br in T.serial(tx, RADIX, step=BLOCK_THREADS):
                    st_global_u32(state, st_base + bufi * T.int32(RADIX) + br, T.uint32(0))
            if deterministic:
                for dw in T.serial(tx, DET_SCRATCH_WORDS, step=BLOCK_THREADS):
                    st_global_u32(state, det_base + dw, T.uint32(0))
            bar_sync()
            if tx == 0:
                st_release_gpu_s32(state, arrival_word, T.int32(0))

    return radix_topk_multi_cta.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


# ---------------------------------------------------------------------------
# Config matrix: every launcher dispatch cell that maps a row onto >= 2 CTAs.
# ---------------------------------------------------------------------------
_DT_TAG = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}
_MODE_TAG = {"basic": "basic", "page_table": "pt", "ragged": "rag"}

# Upstream benchmark row extents (benchmarks/bench_topk.py) that stay multi-CTA.
_LARGE_LENGTHS = (262144, 524288)
# Upstream k sweep; k > FILTERED_TOPK_MAX_K (2048) always dispatches radix.
_K_SWEEP = (64, 256, 512, 1024, 2048, 4096)


def _cfg(
    dtype: str,
    mode: str,
    num_rows: int,
    length: int,
    k: int,
    deterministic: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    short_rows: bool = False,
    reuse_workspace: bool = False,
    tag: str = "",
) -> dict[str, Any]:
    label = f"{_DT_TAG[dtype]}_{_MODE_TAG[mode]}"
    if deterministic:
        label += "_det"
    label += f"_r{num_rows}_l{length}_k{k}"
    for flag, suffix in (
        (row_starts, "rs"),
        (page_table_row_starts, "pts"),
        (row_to_batch, "r2b"),
        (trivial, "trivial"),
        (short_rows, "short"),
        (reuse_workspace, "reuse"),
    ):
        if flag:
            label += f"_{suffix}"
    if tag:
        label += f"_{tag}"
    return {
        "label": label,
        "dtype": dtype,
        "mode": mode,
        "num_rows": num_rows,
        "length": length,
        "k": k,
        "deterministic": deterministic,
        "row_starts": row_starts,
        "page_table_row_starts": page_table_row_starts,
        "row_to_batch": row_to_batch,
        "trivial": trivial,
        "short_rows": short_rows,
        "reuse_workspace": reuse_workspace,
    }


def _build_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for mode in MODES:
            for deterministic in (False, True):
                base = length_for(dtype, deterministic, 2, 16 // dtype_bytes(dtype))
                assert base is not None

                # VEC_SIZE dispatch: every reachable instantiation, at the minimal
                # multi-CTA length for that vec.
                for vec in reachable_vecs(dtype):
                    length = length_for(dtype, deterministic, 2, vec)
                    if length is None:
                        continue
                    configs.append(
                        _cfg(dtype, mode, 4, length, 256, deterministic, tag=f"vec{vec}")
                    )

                # A group whose chunk division leaves a partially filled tail CTA.
                three = length_for(dtype, deterministic, 3, 16 // dtype_bytes(dtype))
                if three is not None:
                    configs.append(_cfg(dtype, mode, 4, three, 256, deterministic, tag="ctas3"))

                # The upstream benchmark row extents (5 and 10 CTAs per group for
                # fp32, 3 and 5 for the 16-bit types).
                for length in _LARGE_LENGTHS:
                    configs.append(_cfg(dtype, mode, 2, length, 256, deterministic, tag="large"))

                # k sweep, including k > FILTERED_TOPK_MAX_K.
                for k in _K_SWEEP:
                    configs.append(_cfg(dtype, mode, 4, base, k, deterministic))

                # Group topology: single row; groups bounded by rows; groups at the
                # SM-count cap; and more rows than groups (persistent row loop).
                for rows in (1, 32, 74, 148):
                    configs.append(_cfg(dtype, mode, rows, base, 256, deterministic))

                # Trivial early-out. Basic needs k >= length, which in this domain
                # means k above the single-CTA chunk bound; the other two modes take
                # it whenever the per-row length is at most k.
                if mode == "basic":
                    configs.append(_cfg(dtype, mode, 2, base, base, deterministic, trivial=True))
                else:
                    configs.append(_cfg(dtype, mode, 4, base, 256, deterministic, trivial=True))

                # Rows short enough that trailing CTAs get an empty chunk while the
                # row still takes the main path.
                if mode != "basic":
                    configs.append(
                        _cfg(dtype, mode, 4, three or base, 256, deterministic, short_rows=True)
                    )

                # Runtime pointer branches.
                if mode == "page_table":
                    configs.append(_cfg(dtype, mode, 4, base, 256, deterministic, row_starts=True))
                    configs.append(
                        _cfg(
                            dtype,
                            mode,
                            4,
                            base,
                            256,
                            deterministic,
                            row_starts=True,
                            page_table_row_starts=True,
                        )
                    )
                    configs.append(
                        _cfg(dtype, mode, 4, base, 256, deterministic, row_to_batch=True)
                    )
                elif mode == "ragged":
                    configs.append(_cfg(dtype, mode, 4, base, 256, deterministic, row_starts=True))

                # Workspace reuse: two launches on one buffer, crossing the
                # deterministic/non-deterministic transition, exercising the
                # last-CTA reset protocol.
                configs.append(_cfg(dtype, mode, 4, base, 256, deterministic, reuse_workspace=True))
    # Deduplicate on the actual launch parameters, not just the label: the
    # VEC_SIZE sweep and the k sweep both anchor on the same base length, and
    # that length's natural vec is already the maximum, so their first entries
    # describe the identical dispatch cell under two names.  Keying on the label
    # alone would keep both and inflate the matrix with 18 redundant runs.
    # First occurrence wins, which keeps the more descriptive `_vec*` label.
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cfg in configs:
        params = tuple(sorted((k, v) for k, v in cfg.items() if k != "label"))
        seen.setdefault(params, cfg)
    deduped = list(seen.values())
    labels = [cfg["label"] for cfg in deduped]
    # Labels must still be globally unique (tests/test_correctness.py
    # parametrizes on them, and the bench-suite yaml keys on them).
    assert len(set(labels)) == len(labels), "duplicate config label"
    return deduped


CONFIGS = _build_configs()


def _build_bench_configs() -> list[dict[str, Any]]:
    """Perf matrix: the same domain minus the cases that bypass the radix path.

    The trivial early-outs measure a copy loop, and the workspace-reuse cases
    time two launches; neither is this kernel's performance domain.
    """
    return [
        cfg
        for cfg in CONFIGS
        if not cfg["trivial"] and not cfg["reuse_workspace"] and not cfg["short_rows"]
    ]


BENCH_CONFIGS = _build_bench_configs()


# ---------------------------------------------------------------------------
# Data preparation, correctness and benchmark entry points.
# ---------------------------------------------------------------------------
def prepare_data(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 8,
    length: int = 65536,
    k: int = 256,
    deterministic: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    short_rows: bool = False,
    reuse_workspace: bool = False,
    **kwargs,
):
    """Logical inputs for one config.

    Same seeded-normal pattern as the single-CTA port; ties are expected and
    wanted, since they are what exercises the ``== pivot`` collect pass.
    ``short_rows`` mixes very short rows into a long-row batch so that trailing
    CTAs of a group receive ``actual_chunk_size == 0`` and must still run every
    group barrier.
    """
    import torch

    device = "cuda"
    rows = num_rows
    generator = torch.Generator(device=device).manual_seed(1234)
    scores = torch.randn(rows, length, dtype=torch.float32, device=device, generator=generator)
    scores = scores.to(torch_dtype(dtype)).contiguous()

    data: dict[str, Any] = {"scores": scores}
    if mode != "basic":
        if trivial:
            lengths = torch.full((rows,), min(k, length), dtype=torch.int32, device=device)
        elif short_rows:
            # Rows far shorter than one chunk leave the group's later CTAs with
            # an empty chunk (:1309-1311) while the barrier count stays uniform.
            short = torch.full((rows,), max(k + 1, 4000), dtype=torch.int32, device=device)
            lengths = torch.where(
                torch.arange(rows, device=device) % 2 == 0,
                short,
                torch.full((rows,), length, dtype=torch.int32, device=device),
            ).to(torch.int32)
        else:
            lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
        if row_starts:
            starts = torch.arange(rows, dtype=torch.int32, device=device) % 8
            lengths = torch.clamp(lengths - starts, min=k)
            data["row_starts"] = starts
        data["lengths"] = lengths
    if mode == "page_table":
        batch = page_table_batch(rows, row_to_batch)
        if row_to_batch:
            data["row_to_batch"] = (
                torch.arange(rows, dtype=torch.int32, device=device) % batch
            ).contiguous()
        # Injective page ids so a returned page id inverts to the row-local index.
        data["src_page_table"] = (
            torch.arange(batch * length, dtype=torch.int32, device=device)
            .reshape(batch, length)
            .contiguous()
        )
        data["page_table_batch"] = batch
        if page_table_row_starts:
            data["page_table_row_starts"] = (
                torch.arange(rows, dtype=torch.int32, device=device) % 4
            ).contiguous()
    if mode == "ragged":
        data["offsets"] = (
            torch.arange(rows, dtype=torch.int32, device=device) * length
        ).contiguous()
    return data


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "dtype": "float32",
        "mode": "basic",
        "num_rows": 8,
        "length": 65536,
        "k": 256,
        "deterministic": False,
        "row_starts": False,
        "page_table_row_starts": False,
        "row_to_batch": False,
        "trivial": False,
        "short_rows": False,
        "reuse_workspace": False,
    }
    cfg.update({key: value for key, value in config.items() if key != "label"})
    return cfg


def build_tirx_args(cfg: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]):
    """Materialize the flat launch ABI once, outside any timed region.

    Adds the cross-CTA workspace the single-CTA ABI does not carry: the same
    byte buffer the reference receives, viewed as ``uint32`` words.
    """
    import torch

    rows = cfg["num_rows"]
    length = cfg["length"]
    empty_i32 = torch.zeros(rows, dtype=torch.int32, device="cuda")
    aux = data.get("src_page_table")
    if aux is None:
        aux = data.get("offsets")
    if aux is None:
        aux = torch.zeros(1, dtype=torch.int32, device="cuda")
    lengths = data.get("lengths")
    if lengths is None:
        lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    aux_stride = int(aux.stride(0)) if aux.dim() == 2 else 0
    expected_aux = aux_elements(cfg["mode"], rows, length, cfg["row_to_batch"])
    assert aux.numel() == expected_aux, (aux.numel(), expected_aux)
    return (
        data["scores"].view(-1),
        outputs["indices"].view(-1),
        outputs["values"].view(-1),
        aux.reshape(-1),
        lengths,
        data.get("row_starts", empty_i32),
        data.get("page_table_row_starts", empty_i32),
        data.get("row_to_batch", empty_i32),
        outputs["row_states"].view(torch.uint32),
        aux_stride,
    )


def _launch_tirx(ex, cfg: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]) -> None:
    ex(*build_tirx_args(cfg, data, outputs))


def run_test(**config):
    """Compile, launch, and validate one config against the FlashInfer source."""
    import unittest

    import torch

    from tirx_kernels.runner import compile_kernel

    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"flashinfer unavailable: {exc}") from exc
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    cfg = _normalize_config(config)
    assert_device_matches_compile_profile(hardware_max_smem_optin(), "multi-CTA")
    _validate(
        cfg["dtype"],
        cfg["mode"],
        cfg["num_rows"],
        cfg["length"],
        cfg["k"],
        cfg["deterministic"],
        cfg["row_starts"] and cfg["mode"] != "basic",
    )

    data = prepare_data(**cfg)
    ref_out = alloc_outputs(cfg)
    run_reference(cfg, data, ref_out)
    assert_reference_is_top_k(cfg, data, ref_out)

    ex = compile_kernel(get_kernel(**cfg))
    tirx_out = alloc_outputs(cfg)
    _launch_tirx(ex, cfg, data, tirx_out)
    torch.cuda.synchronize()
    compare_outputs(cfg, data, ref_out, tirx_out)

    if cfg["reuse_workspace"]:
        # The exit reset (:203-238) is what makes the cached workspace safe to
        # reuse.  Relaunch on the SAME dirty buffer, and flip `deterministic` so
        # the second launch crosses the transition the upstream regression test
        # covers -- a buffer left dirty would desynchronize the group barrier or
        # corrupt the first round's histogram.
        _launch_tirx(ex, cfg, data, tirx_out)
        torch.cuda.synchronize()
        compare_outputs(cfg, data, ref_out, tirx_out)

        flipped = dict(cfg)
        flipped["deterministic"] = not cfg["deterministic"]
        try:
            _validate(
                flipped["dtype"],
                flipped["mode"],
                flipped["num_rows"],
                flipped["length"],
                flipped["k"],
                flipped["deterministic"],
                flipped["row_starts"] and flipped["mode"] != "basic",
            )
        except ValueError:
            return  # the flip leaves the multi-CTA domain; the reuse check above stands
        flipped_ref = {
            "indices": ref_out["indices"].clone(),
            "values": ref_out["values"].clone(),
            "row_states": tirx_out["row_states"],
        }
        run_reference(flipped, data, flipped_ref)
        ex2 = compile_kernel(get_kernel(**flipped))
        flipped_out = {
            "indices": torch.empty_like(ref_out["indices"]),
            "values": torch.empty_like(ref_out["values"]),
            "row_states": tirx_out["row_states"],
        }
        _launch_tirx(ex2, flipped, data, flipped_out)
        torch.cuda.synchronize()
        compare_outputs(flipped, data, flipped_ref, flipped_out)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    cfg = _normalize_config(kwargs)
    state = {"config": cfg, "executable": compile_kernel(get_kernel(**cfg))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Kernel-only comparison against the FlashInfer source launch."""
    cfg = dict(prepared["config"])
    ex = prepared["executable"]

    data = prepare_data(**cfg)
    tirx_out = alloc_outputs(cfg)
    tirx_args = build_tirx_args(cfg, data, tirx_out)

    def tirx_launch():
        ex(*tirx_args)

    def build_reference():
        ref_out = alloc_outputs(cfg)
        return build_reference_launch(cfg, data, ref_out)

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    prepared = prepare_bench(**config)
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )
