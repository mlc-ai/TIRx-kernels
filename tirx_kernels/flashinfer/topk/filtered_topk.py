# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer filtered top-k port.

Ports the **FilteredTopK** pipeline that ``TopKDispatch`` selects when a row fits
one CTA's candidate budget: ``FilteredTopKUnifiedKernel<DType, int32_t, VEC_SIZE,
DETERMINISTIC, MODE, TIE_BREAK>`` (``include/flashinfer/topk.cuh:2362``) plus its
companion ``FinalizeTopKIndicesKernel`` (``:2958``).

The unified kernel runs one CTA of 1024 threads per row (``:3183-3184``) with a
fixed 128 KiB dynamic shared arena (``FILTERED_TOPK_SMEM_DYNAMIC``, ``:2343-2344``)
and no global workspace at all -- everything is shared-memory atomics and
``__syncthreads()``.  It builds a 256-bin coarse histogram over a lossy
``ToCoarseKey`` (fp32 rounds through fp16, ``:2282-2289``), suffix-scans it to pick
the threshold bin, then *filters*: only the candidates landing in that bin are
compacted into a 16384-entry shared index buffer and refined byte-by-byte
(``:2611-2627``, ``:2674-2709``).  That is the whole point of the algorithm --
later passes re-read at most 16K scattered elements instead of the whole row.
When a coarse bin holds more than 16384 candidates the kernel sets
``s_refine_overflow`` and falls back to full-row histogram passes (``:2711-2757``
for 16-bit, ``:2800-2901`` for fp32).

``DETERMINISTIC`` is ``deterministic || tie_break != None`` (``:3181``).  It swaps
the racing back-fill tie claim (``:2567-2580``) for a CUB-scan collector -- thread
strided for plain determinism (``:255-286``), contiguous ascending/descending for
``TopKTieBreak::Small``/``Large`` (``:298-377``) -- and defers the PageTable/Ragged
index transform to the finalize kernel.

**The finalize kernel is part of this port, not an optional extra.**  For
``deterministic`` (and for ``tie_break`` on the transform modes) the dispatchers
launch it after the main kernel (``:3460-3464`` Plain, ``:3392-3402`` PageTable,
``:3428-3437`` Ragged) to block-radix-sort each row's indices ascending and apply
the deferred transform.  Without it "deterministic" would yield a deterministic
*set* in a nondeterministic *order*, so both the semantics and an honest benchmark
require the pair.

FilteredTopK is the **only** implementation of ``tie_break != None``
(``:3327-3329``) and of ``dsa_graph_safe`` (``:3323-3325``); both radix ports
excluded them because ``ShouldUseFilteredTopK`` returns true unconditionally for
those flags.

In-scope specialization: ``k <= FILTERED_TOPK_MAX_K == 2048`` with ``max_len > k``
(or either forcing flag set), all three dtypes, all three modes, every reachable
``VEC_SIZE``, both collect paths, all three tie-break modes, and the runtime
``row_starts`` / ``page_table_row_starts`` / ``row_to_batch`` branches.  Out of
scope, with the dispatch predicate that excludes each: ``k > 2048`` (radix only,
``:3333``); ``sorted_output`` (``StableSortTopKByValueKernel`` ``:3090``, which both
radix ports also leave out); fp8 (never instantiated, ``:2277-2337``); and the
SM100 cluster kernel, which the ``FLASHINFER_TOPK_ALGO=filtered`` pin keeps out of
the reference path.
"""

from typing import Any

from tirx_kernels.flashinfer.topk.radix_topk_single_cta import (
    DTYPES,
    MODES,
    aux_elements,
    dtype_bytes,
    hardware_max_smem_optin,
    page_table_batch,
    vec_size,
)
from tirx_kernels.flashinfer.utils.block_radix_sort import alloc_sort_smem, emit_block_radix_sort
from tirx_kernels.flashinfer.utils.filtered_topk_ops import (
    FIRST_REFINE_SHIFT,
    NUM_REFINE_ROUNDS,
    NUM_SCALARS,
    FilteredCfg,
    emit_filtered_topk_main,
    ld_global_nc_bits,
    ld_global_nc_u32,
    scan_elements,
    st_global_bits,
)
from tirx_kernels.flashinfer.utils.topk_harness import (
    SOURCE_ALGO_FILTERED,
    alloc_outputs,
    assert_device_matches_compile_profile,
    pin_source_algo,
    row_local_indices,
    selected_values,
    source_module,
    torch_dtype,
)
from tirx_kernels.flashinfer.utils.topk_radix import (
    ld_global_bits,
    ld_global_u32,
    st_global_u16,
    st_global_u32,
)
from tirx_kernels.runner import bench
from tvm.script import tirx as T

# Patterns whose whole purpose is to overflow the candidate arena; prepare_data
# asserts on the host that they still do.
_OVERFLOW_PATTERNS = ("tie_heavy", "quantized")
# One coarse bin must be able to exceed FILTERED_TOPK_SMEM_INPUT_SIZE for the
# fallback to be reachable at all, so the overflow patterns get a long row.
_OVERFLOW_LENGTH = 131072

KERNEL_META = {"name": "filtered_topk", "category": "flashinfer", "compute_capability": 10}

# The unified kernel takes the 128 KiB candidate arena as dynamic shared memory;
# the finalize kernel's BlockRadixSort scratch is static-sized per (BT, IPT).
LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")
FINALIZE_LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# --- source constants (topk.cuh:2339-2347) ---------------------------------
FILTERED_TOPK_MAX_K = 2048
FILTERED_TOPK_BLOCK_THREADS = 1024
FILTERED_TOPK_SMEM_INPUT_SIZE = 16 * 1024
FILTERED_TOPK_SMEM_DYNAMIC = 4 * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE  # 131072
RADIX = 256

# Row-scan unroll: per-thread trips at or above this get the deeper factor.
SCAN_DEEP_UNROLL_TRIPS = 16
SCAN_DEEP_UNROLL = 4

# `CanImplementFilteredTopK` (:3285-3294) gates on
# cudaDevAttrMaxSharedMemoryPerMultiprocessor >= FILTERED_TOPK_SMEM_DYNAMIC.
MIN_SMEM_PER_SM = FILTERED_TOPK_SMEM_DYNAMIC

# TopKTieBreak (topk.cuh:36-40); the FFI takes the int64 form (csrc/topk.cu:25-39).
TIE_NONE = 0
TIE_SMALL = 1
TIE_LARGE = 2
TIE_BREAKS = (TIE_NONE, TIE_SMALL, TIE_LARGE)

# Input distributions mirroring benchmarks/bench_topk.py --input-pattern
# (:982-996).  `tie_heavy`/`quantized` are the only way to reach the
# `s_refine_overflow` fallbacks, which are data- not shape-dependent.
PATTERNS = ("random", "quantized", "tie_heavy", "pivot_tie")


def finalize_block_config(k: int) -> tuple[int, int]:
    """`LaunchFinalizeTopKIndices`'s block/items ladder (topk.cuh:3061-3079)."""
    if k <= 128:
        return (32, 4)
    if k <= 256:
        return (32, 8)
    if k <= 512:
        return (64, 8)
    if k <= 576:
        return (64, 9)
    if k <= 1024:
        return (128, 8)
    return (256, 8)


def finalize_end_bit(max_len: int) -> int:
    """``end_bit = 32 - __clz(max_len)`` (topk.cuh:2994), static per config."""
    return max_len.bit_length()


def effective_page_table_row_starts(mode: str, page_table_row_starts: bool, row_starts: bool):
    """`TopKPageTableTransformDispatch` substitutes `row_starts` for a null pointer.

    ``:3379-3380`` runs ``if (page_table_row_starts == nullptr) page_table_row_starts
    = row_starts;`` before dispatching, so **both** launches see a non-null
    pointer whenever ``row_starts`` is non-null.  It is invisible in the unified
    kernel, whose own ternary (``:2387-2390``) falls back to ``row_start`` and so
    computes the same value either way, but it is load-bearing in the finalize
    kernel, which falls back to ``0`` instead (``:3008``) and would otherwise drop
    the row offset from the deferred page-table transform.
    """
    if mode != "page_table":
        return page_table_row_starts
    return page_table_row_starts or row_starts


def finalize_plan(mode: str, k: int, deterministic: bool, tie_break: int):
    """Which finalize launch (if any) the dispatcher issues after the main kernel.

    Verified in all three dispatchers: Plain :3460-3464, PageTable :3392-3402,
    Ragged :3428-3437.  ``sort_local_indices`` distinguishes the deterministic
    launch (block-radix-sort the row, then transform) from the tie-break-only
    launch on the transform modes (transform, no sort).
    """
    if k == 0:
        return None
    if deterministic:
        if mode == "basic" and k <= 1:
            return None  # LaunchFinalizeTopKIndices early-out (:3043-3047)
        return {"sort_local_indices": True}
    if tie_break != TIE_NONE and mode != "basic":
        return {"sort_local_indices": False}
    return None


def filtered_vec_size(
    dtype: str, max_len: int, scalar_addressing: bool, dsa_graph_safe: bool
) -> int:
    """`ComputeFilteredTopKVecSize` (:2934-2943) plus the launcher's row_starts clause.

    ``dsa_graph_safe`` pins 1 so the instantiation cannot depend on a runtime
    ``max_len`` (graph-capture stable); a non-null ``row_starts`` on a transform
    mode pins 1 because the row base pointer can then be misaligned (:3190-3192).
    """
    if dsa_graph_safe:
        return 1
    return vec_size(dtype, max_len, scalar_addressing=scalar_addressing)


def _validate(
    dtype: str,
    mode: str,
    num_rows: int,
    length: int,
    k: int,
    deterministic: bool,
    tie_break: int,
    dsa_graph_safe: bool,
) -> dict[str, Any]:
    """Reject anything outside the FilteredTopK dispatch domain.

    Mirrors `ShouldUseFilteredTopK` (:3318-3369) under the
    ``FLASHINFER_TOPK_ALGO=filtered`` override: the override skips the
    num_rows/max_len heuristics (:3346-3368) but not the hard preconditions, and
    the dispatchers' guard (:3382-3385) still errors when a forcing flag meets
    ``k > FILTERED_TOPK_MAX_K``.
    """
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"Unsupported tie_break: {tie_break}")
    if num_rows < 1 or length < 1 or k < 1:
        raise ValueError(f"num_rows={num_rows}, length={length}, k={k} must be positive")
    if k > FILTERED_TOPK_MAX_K:
        raise ValueError(
            f"k={k} exceeds FILTERED_TOPK_MAX_K={FILTERED_TOPK_MAX_K}: radix-only shape"
        )
    forced = dsa_graph_safe or tie_break != TIE_NONE
    if not forced and length <= k:
        raise ValueError(
            f"length={length} <= k={k} only reaches FilteredTopK through "
            "dsa_graph_safe or a tie-break mode (:3333)"
        )
    scalar_addressing = False  # set by the caller through row_starts on non-basic modes
    vec = filtered_vec_size(dtype, length, scalar_addressing, dsa_graph_safe)
    return {
        "vec_size": vec,
        "block_threads": FILTERED_TOPK_BLOCK_THREADS,
        "grid": num_rows,
        "smem_bytes": FILTERED_TOPK_SMEM_DYNAMIC,
        "finalize": finalize_plan(mode, k, deterministic, tie_break),
        "finalize_block": finalize_block_config(k),
        "end_bit": finalize_end_bit(length),
    }


# ---------------------------------------------------------------------------
# Target entries.
# ---------------------------------------------------------------------------
def get_kernel(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 16,
    length: int = 8192,
    k: int = 256,
    deterministic: bool = False,
    tie_break: int = TIE_NONE,
    dsa_graph_safe: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    pattern: str = "random",
    **kwargs,
):
    """Return the TIRx specialization of `FilteredTopKUnifiedKernel` for one cell."""
    scalar_addressing = row_starts and mode != "basic"
    plan = _validate(dtype, mode, num_rows, length, k, deterministic, tie_break, dsa_graph_safe)
    vec = filtered_vec_size(dtype, length, scalar_addressing, dsa_graph_safe)
    aux_elems = aux_elements(mode, num_rows, length, row_to_batch)
    grid = plan["grid"]

    basic = mode == "basic"
    page_table = mode == "page_table"
    ragged = mode == "ragged"
    is32 = dtype == "float32"
    # DETERMINISTIC is not the user's `deterministic` flag: it is
    # `deterministic || tie_break != None` (:3181), and it is what swaps the
    # racing tie back-fill for a CUB-scan collector and defers the transform.
    det = deterministic or tie_break != TIE_NONE
    num_rounds = NUM_REFINE_ROUNDS[dtype]
    first_shift = FIRST_REFINE_SHIFT[dtype]
    load_bytes = vec * dtype_bytes(dtype)
    smem_input = FILTERED_TOPK_SMEM_INPUT_SIZE
    hist_stride = RADIX + 128  # s_histogram_buf[2][RADIX + 128] (:2432)
    # `length <= top_k` (:2409) tests the *per-row* length.  On the transform
    # modes that is a runtime value, so the branch must always be emitted there;
    # only Plain, whose row length is the static kernel stride, can fold it away.
    trivial_path = (not basic) or length <= k
    # Per-thread trips through the row-scan vector loop.  The source's literal
    # `#pragma unroll 2` (:2469) encodes an intent -- keep independent loads in
    # flight -- not a portable constant, so the factor is derived from the trip
    # count and swept rather than assumed.
    scan_trips = max(1, (length // vec) // FILTERED_TOPK_BLOCK_THREADS)
    scan_unroll = SCAN_DEEP_UNROLL if scan_trips >= SCAN_DEEP_UNROLL_TRIPS else 2
    cfg = FilteredCfg(
        top_k=k,
        vec=vec,
        load_bytes=load_bytes,
        is32=is32,
        det=det,
        tie_break=tie_break,
        num_rounds=num_rounds,
        first_shift=first_shift,
        smem_input=smem_input,
        hist_stride=hist_stride,
        scan_unroll=scan_unroll,
        basic=basic,
        page_table=page_table,
        ragged=ragged,
        block=FILTERED_TOPK_BLOCK_THREADS,
        radix=RADIX,
    )

    @T.prim_func
    def filtered_topk(
        input_h: T.handle,
        output_indices_h: T.handle,
        output_values_h: T.handle,
        aux_data_h: T.handle,
        lengths_h: T.handle,
        row_starts_h: T.handle,
        page_table_row_starts_h: T.handle,
        row_to_batch_h: T.handle,
        aux_stride: T.int64,
    ):
        T.func_attr({"global_symbol": "filtered_topk"})
        inp = T.match_buffer(input_h, (num_rows * length,), dtype, scope="global")
        out_idx = T.match_buffer(output_indices_h, (num_rows * k,), "int32", scope="global")
        out_val = T.match_buffer(output_values_h, (num_rows * k,), dtype, scope="global")
        aux = T.match_buffer(aux_data_h, (aux_elems,), "int32", scope="global")
        lengths_g = T.match_buffer(lengths_h, (num_rows,), "int32", scope="global")
        row_starts_g = T.match_buffer(row_starts_h, (num_rows,), "int32", scope="global")
        pt_starts_g = T.match_buffer(page_table_row_starts_h, (num_rows,), "int32", scope="global")
        row_to_batch_g = T.match_buffer(row_to_batch_h, (num_rows,), "int32", scope="global")
        T.device_entry()
        row = T.cta_id([grid])
        tx = T.thread_id([FILTERED_TOPK_BLOCK_THREADS])

        # --- shared layout (:2431-2446) --------------------------------------
        pool = T.SMEMPool()
        s_hist2 = pool.alloc((2 * hist_stride,), "uint32", align=128)
        s_indices = pool.alloc((FILTERED_TOPK_MAX_K,), "uint32", align=128)
        s_scal = pool.alloc((NUM_SCALARS,), "uint32")
        if det:
            s_scan = pool.alloc((scan_elements(),), "uint32", align=16)
        # The candidate arena is a fixed 128 KiB, never scaled by max_len or k
        # (:2343-2344); that constant is exactly what CanImplementFilteredTopK
        # gates on (:3285-3294).
        s_input = pool.alloc((2 * smem_input,), "uint32", align=128)
        pool.commit()

        # --- per-row header (:2384-2406) --------------------------------------
        # Basic's row length is the static kernel stride, so it stays a Python
        # constant and `aligned_length` folds to a literal exactly as in the
        # source.  Binding it to a name inside the traced body would make it a TIR
        # variable and turn the row-scan bound into a runtime read.
        row_len_rt: T.int32 = T.int32(0)
        row_start: T.int32 = T.int32(0)
        page_start: T.int32 = T.int32(0)
        if not basic:
            row_len_rt = T.reinterpret("int32", ld_global_nc_u32(lengths_g, row))
            if row_starts:
                row_start = T.reinterpret("int32", ld_global_nc_u32(row_starts_g, row))
            if page_table:
                if page_table_row_starts:
                    page_start = T.reinterpret("int32", ld_global_nc_u32(pt_starts_g, row))
                else:
                    page_start = row_start
        row_in: T.int64 = T.cast(row, "int64") * T.int64(length) + T.cast(row_start, "int64")
        row_out: T.int64 = T.cast(row, "int64") * T.int64(k)

        batch_idx: T.int32 = row
        offset_val: T.int32 = T.int32(0)
        if page_table:
            if row_to_batch:
                batch_idx = T.reinterpret("int32", ld_global_nc_u32(row_to_batch_g, row))
        if ragged:
            offset_val = T.reinterpret("int32", ld_global_nc_u32(aux, T.cast(row, "int64")))

        # --- trivial early-out, length <= top_k (:2409-2429) ------------------
        take_main: T.int32 = T.int32(1)
        if trivial_path:
            if (length if basic else row_len_rt) <= k:
                take_main = T.int32(0)
                for i0 in T.serial(tx, k, step=FILTERED_TOPK_BLOCK_THREADS):
                    slot0: T.int64 = row_out + T.cast(i0, "int64")
                    if basic:
                        if i0 < (length if basic else row_len_rt):
                            st_global_u32(out_idx, slot0, T.reinterpret("uint32", i0))
                            st_global_bits(
                                out_val,
                                slot0,
                                ld_global_nc_bits(inp, row_in + T.cast(i0, "int64"), is32),
                                is32,
                            )
                        else:
                            # A literal -1 is written as its bit pattern: reinterpreting
                            # an immediate lowers to an address-of-literal, which is
                            # not an lvalue.
                            st_global_u32(out_idx, slot0, T.uint32(0xFFFFFFFF))
                            st_global_bits(
                                out_val, slot0, T.uint32(0) if is32 else T.uint16(0), is32
                            )
                    elif det:
                        # Local index; the transform is deferred to the finalizer.
                        st_global_u32(
                            out_idx,
                            slot0,
                            T.reinterpret(
                                "uint32",
                                T.Select(i0 < (length if basic else row_len_rt), i0, T.int32(-1)),
                            ),
                        )
                    elif page_table:
                        page0: T.int32 = T.int32(-1)
                        if i0 < (length if basic else row_len_rt):
                            page0 = T.reinterpret(
                                "int32",
                                ld_global_nc_u32(
                                    aux,
                                    T.cast(batch_idx, "int64") * aux_stride
                                    + T.cast(page_start + i0, "int64"),
                                ),
                            )
                        st_global_u32(out_idx, slot0, T.reinterpret("uint32", page0))
                    else:
                        st_global_u32(
                            out_idx,
                            slot0,
                            T.reinterpret(
                                "uint32",
                                T.Select(
                                    i0 < (length if basic else row_len_rt),
                                    i0 + offset_val,
                                    T.int32(-1),
                                ),
                            ),
                        )

        if take_main == 1:
            emit_filtered_topk_main(
                inp,
                out_idx,
                out_val,
                aux,
                s_hist2,
                s_indices,
                s_scal,
                s_input,
                s_scan if det else s_scal,
                tx,
                row_in,
                row_out,
                length if basic else row_len_rt,
                batch_idx,
                page_start,
                offset_val,
                aux_stride,
                cfg,
            )

    return filtered_topk.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


def get_finalize_kernel(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 16,
    length: int = 8192,
    k: int = 256,
    deterministic: bool = False,
    tie_break: int = TIE_NONE,
    dsa_graph_safe: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    pattern: str = "random",
    **kwargs,
):
    """Return the TIRx specialization of `FinalizeTopKIndicesKernel`, or None.

    ``None`` means the dispatcher issues no second launch for this config, in
    which case the harness must launch only the unified kernel.
    """
    plan = _validate(dtype, mode, num_rows, length, k, deterministic, tie_break, dsa_graph_safe)
    fin = plan["finalize"]
    if fin is None:
        return None
    block_threads, items_per_thread = plan["finalize_block"]
    end_bit = plan["end_bit"]
    do_sort = fin["sort_local_indices"]
    basic = mode == "basic"
    page_table = mode == "page_table"
    ragged = mode == "ragged"
    is32 = dtype == "float32"
    # Plain keeps the scores paired with the keys through the sort; the transform
    # modes instantiate the keys-only specialization (:2993-2999).
    sort_values = do_sort and basic
    val_bytes = dtype_bytes(dtype)
    aux_elems = aux_elements(mode, num_rows, length, row_to_batch)

    @T.prim_func
    def filtered_topk_finalize(
        output_indices_h: T.handle,
        output_values_h: T.handle,
        aux_data_h: T.handle,
        page_table_row_starts_h: T.handle,
        row_to_batch_h: T.handle,
        aux_stride: T.int64,
    ):
        T.func_attr({"global_symbol": "filtered_topk_finalize"})
        out_idx = T.match_buffer(output_indices_h, (num_rows * k,), "int32", scope="global")
        out_val = T.match_buffer(output_values_h, (num_rows * k,), dtype, scope="global")
        aux = T.match_buffer(aux_data_h, (aux_elems,), "int32", scope="global")
        pt_starts_g = T.match_buffer(page_table_row_starts_h, (num_rows,), "int32", scope="global")
        row_to_batch_g = T.match_buffer(row_to_batch_h, (num_rows,), "int32", scope="global")
        T.device_entry()

        row = T.cta_id([num_rows])
        tx = T.thread_id([block_threads])

        pool = T.SMEMPool()
        if do_sort:
            c32, c16, xk, xv, sscan = alloc_sort_smem(
                pool, block_threads, items_per_thread, dtype, val_bytes
            )
        pool.commit()

        row_out: T.int64 = T.cast(row, "int64") * T.int64(k)
        keys = T.alloc_local((items_per_thread,), "uint32")
        values = T.alloc_local((items_per_thread,), "uint32")

        # --- blocked load with ~0u padding (:2976-2991) --------------------
        for i in T.unroll(items_per_thread):
            pos: T.int32 = tx * items_per_thread + i
            keys[i] = T.uint32(0xFFFFFFFF)
            values[i] = T.uint32(0)
            if pos < k:
                slot: T.int64 = row_out + T.cast(pos, "int64")
                idx: T.int32 = T.reinterpret("int32", ld_global_u32(out_idx, slot))
                # `(idx >= 0) ? idx : ~0u` -- nvcc folds the clamp to a single
                # max.s32 against immediate -1 (:2981).
                keys[i] = T.reinterpret("uint32", T.max(idx, T.int32(-1)))
                if sort_values:
                    values[i] = ld_global_bits(out_val, slot, is32)

        # --- ascending block radix sort over [0, end_bit) (:2993-2999) -----
        # `~0u` truncated to end_bit bits is 2**end_bit - 1 while every real
        # index is <= max_len - 1 < 2**end_bit - 1, so padding is strictly
        # maximal in every digit pass and lands at the tail.
        if do_sort:
            ranks = T.alloc_local((items_per_thread,), "int32")
            emit_block_radix_sort(
                c32,
                c16,
                sscan,
                xk,
                xv,
                keys,
                values,
                ranks,
                tx,
                block_threads,
                items_per_thread,
                end_bit,
                sort_values,
                is32,
            )

        # --- deferred transform + writeback (:3002-3029) -------------------
        batch_idx: T.int32 = row
        page_start: T.int32 = T.int32(0)
        offset: T.int32 = T.int32(0)
        if page_table:
            if row_to_batch:
                batch_idx = T.reinterpret("int32", ld_global_u32(row_to_batch_g, row))
            # `:3008` falls back to 0, but the dispatcher already substituted
            # row_starts for a null pointer at `:3379-3380`.
            if effective_page_table_row_starts(mode, page_table_row_starts, row_starts):
                page_start = T.reinterpret("int32", ld_global_u32(pt_starts_g, row))
        if ragged:
            offset = T.reinterpret("int32", ld_global_u32(aux, T.cast(row, "int64")))

        for i in T.unroll(items_per_thread):
            pos2: T.int32 = tx * items_per_thread + i
            if pos2 < k:
                slot2: T.int64 = row_out + T.cast(pos2, "int64")
                key: T.uint32 = keys[i]
                if basic:
                    # `~0u` reinterprets to -1 for free, so both stores are
                    # unguarded (:3010-3014).
                    st_global_u32(out_idx, slot2, key)
                    if sort_values:
                        # The exchange keeps values in uint32 registers; narrow
                        # them back on the way out for the 16-bit dtypes.
                        if is32:
                            st_global_u32(out_val, slot2, values[i])
                        else:
                            st_global_u16(out_val, slot2, T.cast(values[i], "uint16"))
                elif page_table:
                    page_id: T.int32 = T.int32(-1)
                    if key != T.uint32(0xFFFFFFFF):
                        src: T.int64 = T.cast(batch_idx, "int64") * aux_stride
                        page_id = T.reinterpret(
                            "int32",
                            ld_global_u32(
                                aux, src + T.cast(page_start + T.reinterpret("int32", key), "int64")
                            ),
                        )
                    st_global_u32(out_idx, slot2, T.reinterpret("uint32", page_id))
                else:
                    val2: T.int32 = T.int32(-1)
                    if key != T.uint32(0xFFFFFFFF):
                        val2 = T.reinterpret("int32", key) + offset
                    st_global_u32(out_idx, slot2, T.reinterpret("uint32", val2))

    return filtered_topk_finalize.with_attr("tirx.kernel_launch_params", list(FINALIZE_LAUNCH_TAGS))


_ = (
    bench,
    dtype_bytes,
    hardware_max_smem_optin,
    page_table_batch,
    PATTERNS,
    MIN_SMEM_PER_SM,
    RADIX,
    TIE_SMALL,
    TIE_LARGE,
)  # wired up with the config matrix and harness


def prepare_data(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 16,
    length: int = 8192,
    k: int = 256,
    pattern: str = "random",
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    **kwargs,
):
    """Logical inputs for one config.

    ``pattern`` mirrors ``benchmarks/bench_topk.py --input-pattern`` (``:982-996``).
    It is part of the config domain rather than a testing nicety: the
    ``s_refine_overflow`` fallbacks (``:2711-2757`` for 16-bit, ``:2800-2901`` for
    fp32) are reached only when one coarse bin holds more than
    ``FILTERED_TOPK_SMEM_INPUT_SIZE`` candidates, which seeded normals never do at
    these lengths (about ``length / 256`` per bin).  ``tie_heavy`` and
    ``quantized`` collapse the row onto few distinct values so a single bin
    overflows; ``pivot_tie`` piles ties exactly at the pivot.

    Configs tagged with an overflow-reaching pattern are checked on the host
    below, so the coverage cannot silently rot if the traits or the arena size
    ever change.
    """
    import torch

    device = "cuda"
    rows = num_rows
    generator = torch.Generator(device=device).manual_seed(1234)
    if pattern == "random":
        scores = torch.randn(rows, length, dtype=torch.float32, device=device, generator=generator)
    elif pattern == "quantized":
        # A few distinct magnitudes: many rows share a coarse bin.
        levels = torch.randint(
            0, 6, (rows, length), dtype=torch.int32, device=device, generator=generator
        )
        scores = levels.to(torch.float32)
    elif pattern == "tie_heavy":
        # Two values, so the winning coarse bin holds a third of the row.
        pick = torch.randint(
            0, 3, (rows, length), dtype=torch.int32, device=device, generator=generator
        )
        scores = torch.where(
            pick == 0,
            torch.full((rows, length), 2.0, device=device),
            torch.full((rows, length), 1.0, device=device),
        )
    elif pattern == "pivot_tie":
        # Distinct values above the pivot, then a wide plateau exactly at it.
        scores = torch.randn(rows, length, dtype=torch.float32, device=device, generator=generator)
        plateau = max(1, k // 2)
        scores = torch.sort(scores, dim=-1, descending=True).values
        scores[:, plateau : plateau + 4 * k] = scores[:, plateau : plateau + 1]
        perm = torch.randperm(length, device=device, generator=generator)
        scores = scores[:, perm].contiguous()
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    scores = scores.to(torch_dtype(dtype)).contiguous()

    data: dict[str, Any] = {"scores": scores, "pattern": pattern}
    if mode != "basic":
        if trivial:
            lengths = torch.full((rows,), min(k, length), dtype=torch.int32, device=device)
        else:
            lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
        if row_starts:
            starts = torch.arange(rows, dtype=torch.int32, device=device) % 8
            lengths = torch.clamp(lengths - starts, min=min(k, length))
            data["row_starts"] = starts
        data["lengths"] = lengths
    if mode == "page_table":
        batch = page_table_batch(rows, row_to_batch)
        if row_to_batch:
            data["row_to_batch"] = (
                torch.arange(rows, dtype=torch.int32, device=device) % batch
            ).contiguous()
        # Injective page ids (batch-major) so a returned page id inverts back to
        # the row-local index the kernel selected: id == batch * length + slot.
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


def assert_pattern_reaches_overflow(cfg: dict[str, Any], data: dict[str, Any]) -> None:
    """Check on the host that an overflow-tagged config really overflows.

    Recomputes ``ToCoarseKey`` and the suffix scan exactly as the kernel does,
    finds the threshold bin, and requires its population to exceed the 16384-entry
    candidate arena -- the condition that sets ``s_refine_overflow`` (``:2618-2625``).
    Without this the fallback coverage could vanish silently.
    """
    import torch

    if cfg["pattern"] not in _OVERFLOW_PATTERNS:
        return
    scores = data["scores"]
    lengths = data.get("lengths")
    bins = _coarse_key_host(scores)
    reached = False
    for r in range(scores.size(0)):
        n = int(lengths[r]) if lengths is not None else scores.size(1)
        hist = torch.bincount(bins[r, :n], minlength=RADIX).to(torch.int64)
        suffix = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
        topk = cfg["k"]
        above = torch.cat([suffix[1:], suffix.new_zeros(1)])
        hit = ((suffix > topk) & (above <= topk)).nonzero()
        if hit.numel() == 0:
            continue
        if int(hist[int(hit[0])]) > FILTERED_TOPK_SMEM_INPUT_SIZE:
            reached = True
            break
    assert reached, (
        f"pattern {cfg['pattern']!r} at length={cfg['length']} k={cfg['k']} no longer "
        f"overflows the {FILTERED_TOPK_SMEM_INPUT_SIZE}-entry candidate arena, so the "
        "s_refine_overflow fallback is untested by this config"
    )


def _coarse_key_host(scores):
    """``Traits::ToCoarseKey`` on the host: fp16 round, monotone flip, high byte."""
    import torch

    if scores.dtype == torch.float32:
        bits = scores.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    else:
        bits = scores.view(torch.int16).to(torch.int32) & 0xFFFF
    flipped = torch.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits | 0x8000)
    return (flipped >> 8).to(torch.int64)


def run_reference(config: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]) -> None:
    """One launch of the FlashInfer source pipeline with preallocated outputs.

    The filtered path never touches ``row_states``, so the optional workspace is
    passed as ``None``: an accidental dispatch back to a radix kernel would fail
    loudly instead of silently producing a correct answer by another route.
    """
    import torch

    pin_source_algo(SOURCE_ALGO_FILTERED)
    module = source_module()
    fn, args = _reference_args(config, data, outputs)
    fn(*args)
    torch.cuda.synchronize()


def _reference_args(config: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]):
    mode = config["mode"]
    k = config["k"]
    det = config["deterministic"]
    tie = config["tie_break"]
    dsa = config["dsa_graph_safe"]
    indices = outputs["indices"]
    scores = data["scores"]
    if mode == "basic":
        return module_fn(mode), (
            scores,
            indices,
            outputs["values"],
            None,  # row_states: unused on the filtered path
            k,
            False,  # sorted_output
            det,
            tie,
            dsa,
        )
    if mode == "page_table":
        return module_fn(mode), (
            scores,
            indices,
            data["src_page_table"],
            data.get("row_to_batch"),
            data["lengths"],
            None,
            k,
            det,
            tie,
            dsa,
            data.get("row_starts"),
            data.get("page_table_row_starts"),
        )
    return module_fn(mode), (
        scores,
        indices,
        data["offsets"],
        data["lengths"],
        None,
        k,
        det,
        tie,
        dsa,
        data.get("row_starts"),
    )


def module_fn(mode: str):
    module = source_module()
    if mode == "basic":
        return module.radix_topk
    if mode == "page_table":
        return module.radix_topk_page_table_transform
    return module.radix_topk_ragged_transform


def build_reference_launch(config: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]):
    """Resolve the module and bind every argument once; return a bare launch.

    The timed closure must enqueue exactly what the dispatcher enqueues -- which
    for a deterministic or tie-break config is the unified kernel *and* the
    finalize kernel -- and nothing else: no env writes, no module lookup, no host
    synchronize.
    """
    pin_source_algo(SOURCE_ALGO_FILTERED)
    fn, args = _reference_args(config, data, outputs)

    def launch():
        fn(*args)

    return launch


def build_tirx_args(cfg: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]):
    """Materialize the flat launch ABI once, outside any timed region.

    Every tensor the kernels read must exist before the launch closure is built:
    allocating placeholders inside the closure would enqueue extra fill kernels
    that a per-kernel timer attributes to this workload.
    """
    import torch

    rows = cfg["num_rows"]
    length = cfg["length"]
    mode = cfg["mode"]
    device = "cuda"
    empty = torch.zeros(rows, dtype=torch.int32, device=device)
    lengths = data.get("lengths")
    if lengths is None:
        lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
    row_starts = data.get("row_starts", empty)
    # Mirror the dispatcher's null-pointer substitution (:3379-3380) so both
    # launches see what the reference's launches see.
    pt_starts = data.get("page_table_row_starts")
    if pt_starts is None and mode == "page_table":
        pt_starts = data.get("row_starts", empty)
    if pt_starts is None:
        pt_starts = empty
    r2b = data.get("row_to_batch", empty)
    if mode == "page_table":
        aux = data["src_page_table"].reshape(-1)
        aux_stride = length
    elif mode == "ragged":
        aux = data["offsets"]
        aux_stride = 1
    else:
        aux = torch.zeros(1, dtype=torch.int32, device=device)
        aux_stride = 1
    scores = data["scores"].reshape(-1)
    indices = outputs["indices"].reshape(-1)
    values = outputs["values"].reshape(-1)
    return {
        "main": (scores, indices, values, aux, lengths, row_starts, pt_starts, r2b, aux_stride),
        "finalize": (indices, values, aux, pt_starts, r2b, aux_stride),
    }


def _launch_tirx(ex, ex_finalize, args) -> None:
    """Enqueue the ported pipeline: unified kernel, then finalize when dispatched.

    Nothing but the launches -- no allocation, no host synchronize.
    """
    ex(*args["main"])
    if ex_finalize is not None:
        ex_finalize(*args["finalize"])


def run_test(**config):
    """Compile, launch, and validate one config against the FlashInfer source."""
    import unittest

    import torch

    from tirx_kernels.runner import check_low_level_ir, compile_kernel

    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"flashinfer unavailable: {exc}") from exc
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    cfg = _normalize_config(config)
    assert_device_matches_compile_profile(hardware_max_smem_optin(), "filtered")
    _validate(
        cfg["dtype"],
        cfg["mode"],
        cfg["num_rows"],
        cfg["length"],
        cfg["k"],
        cfg["deterministic"],
        cfg["tie_break"],
        cfg["dsa_graph_safe"],
    )

    data = prepare_data(**cfg)
    assert_pattern_reaches_overflow(cfg, data)

    ref_out = alloc_outputs(cfg)
    run_reference(cfg, data, ref_out)
    assert_reference_is_top_k(cfg, data, ref_out)
    assert_reference_tie_break(cfg, data, ref_out)

    kernel = get_kernel(**cfg)
    finalize = get_finalize_kernel(**cfg)
    # The runner only contract-checks `get_kernel`; the finalize kernel is a
    # second entry point, so check it here too.
    if finalize is not None:
        check_low_level_ir(finalize)
    ex = compile_kernel(kernel)
    ex_finalize = compile_kernel(finalize) if finalize is not None else None

    tirx_out = alloc_outputs(cfg)
    _launch_tirx(ex, ex_finalize, build_tirx_args(cfg, data, tirx_out))
    torch.cuda.synchronize()

    compare_filtered_outputs(cfg, data, ref_out, tirx_out)


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "dtype": "float32",
        "mode": "basic",
        "num_rows": 16,
        "length": 8192,
        "k": 256,
        "deterministic": False,
        "tie_break": TIE_NONE,
        "dsa_graph_safe": False,
        "row_starts": False,
        "page_table_row_starts": False,
        "row_to_batch": False,
        "trivial": False,
        "pattern": "random",
    }
    cfg.update({key: value for key, value in config.items() if key != "label"})
    return cfg


def assert_reference_is_top_k(
    cfg: dict[str, Any], data: dict[str, Any], ref: dict[str, Any]
) -> None:
    """Independent oracle on the reference launch itself (tie-robust)."""
    import torch

    if cfg["trivial"]:
        return
    scores = data["scores"].to(torch.float32)
    row_starts = data.get("row_starts")
    lengths = data.get("lengths")
    if row_starts is None and lengths is None:
        window = scores
    else:
        rows, length = cfg["num_rows"], cfg["length"]
        starts = (
            torch.zeros(rows, dtype=torch.int64, device=scores.device)
            if row_starts is None
            else row_starts.to(torch.int64)
        )
        lens = (
            torch.full((rows,), length, dtype=torch.int64, device=scores.device)
            if lengths is None
            else lengths.to(torch.int64)
        )
        cols = torch.arange(length, device=scores.device).unsqueeze(0)
        inside = (cols >= starts.unsqueeze(-1)) & (cols < (starts + lens).unsqueeze(-1))
        window = torch.where(inside, scores, torch.full_like(scores, float("-inf")))
    expect = torch.sort(torch.topk(window, cfg["k"], dim=-1).values, dim=-1, descending=True).values
    actual = selected_values(cfg, data, ref["indices"])
    torch.testing.assert_close(actual, expect, rtol=0, atol=0)


def assert_reference_tie_break(
    cfg: dict[str, Any], data: dict[str, Any], ref: dict[str, Any]
) -> None:
    """Oracle for ``tie_break``: the claimed ties must be the extremal indices.

    A plain top-k check cannot see this -- every tie subset has the same values.
    ``Small`` must claim the smallest row-local indices among the elements equal
    to the pivot, ``Large`` the largest (``:2591-2600``).
    """
    import torch

    if cfg["tie_break"] == TIE_NONE or cfg["trivial"]:
        return
    scores = data["scores"].to(torch.float32)
    idx, pad = row_local_indices(cfg, data, ref["indices"])
    row_starts = data.get("row_starts")
    lengths = data.get("lengths")
    for r in range(cfg["num_rows"]):
        start = int(row_starts[r]) if row_starts is not None else 0
        span = int(lengths[r]) if lengths is not None else cfg["length"]
        keep = ~pad[r]
        chosen = (idx[r][keep] + start).tolist()
        if not chosen:
            continue
        pivot = min(scores[r][c].item() for c in chosen)
        window = scores[r][start : start + span]
        eq_all = (window == pivot).nonzero().flatten() + start
        eq_chosen = sorted(c for c in chosen if scores[r][c].item() == pivot)
        n = len(eq_chosen)
        want = eq_all[:n] if cfg["tie_break"] == TIE_SMALL else eq_all[-n:]
        assert eq_chosen == sorted(want.tolist()), (
            f"row {r}: tie_break={cfg['tie_break']} claimed {eq_chosen[:6]}... but the "
            f"{'smallest' if cfg['tie_break'] == TIE_SMALL else 'largest'} equal-valued "
            f"indices are {sorted(want.tolist())[:6]}..."
        )


def compare_filtered_outputs(
    cfg: dict[str, Any], data: dict[str, Any], ref: dict[str, Any], got: dict[str, Any]
) -> None:
    """Compare a TIRx pipeline launch against the FlashInfer pipeline.

    Three regimes, matching what the source actually guarantees:

    * ``DETERMINISTIC`` (``deterministic``, or ``tie_break`` on a transform mode):
      the reference finalize-sorts each row ascending, so the outputs must match
      **element for element**.
    * ``tie_break`` without ``deterministic`` on Plain: no finalize launch, so the
      order is racy, but the selected *set* is unique because ties are broken by
      extremal index.  Compare sorted index sets -- strictly stronger than the
      value check.
    * otherwise: which ties win is genuinely racy on both sides, so compare the
      multiset of selected score values, which every valid top-k shares.
    """
    import torch

    plan = finalize_plan(cfg["mode"], cfg["k"], cfg["deterministic"], cfg["tie_break"])
    if plan is not None and plan["sort_local_indices"]:
        torch.testing.assert_close(got["indices"], ref["indices"], rtol=0, atol=0)
        if cfg["mode"] == "basic":
            torch.testing.assert_close(got["values"], ref["values"], rtol=0, atol=0)
        return

    if cfg["tie_break"] != TIE_NONE:
        ref_idx = torch.sort(ref["indices"].to(torch.int64), dim=-1).values
        got_idx = torch.sort(got["indices"].to(torch.int64), dim=-1).values
        torch.testing.assert_close(got_idx, ref_idx, rtol=0, atol=0)
        return

    ref_vals = selected_values(cfg, data, ref["indices"])
    got_vals = selected_values(cfg, data, got["indices"])
    torch.testing.assert_close(got_vals, ref_vals, rtol=0, atol=0)
    if cfg["mode"] == "basic":
        ref_out = torch.sort(ref["values"].to(torch.float32), dim=-1, descending=True).values
        got_out = torch.sort(got["values"].to(torch.float32), dim=-1, descending=True).values
        torch.testing.assert_close(got_out, ref_out, rtol=0, atol=0)
        # The emitted values must be the scores at the emitted indices.  On the
        # trivial path the padding slots carry a literal 0 (:2417) that is not a
        # score, and `selected_values` sorts padding to -inf, so the two orderings
        # only line up over the real entries; restrict the check to those.
        _, pad = row_local_indices(cfg, data, got["indices"])
        real = torch.where(
            pad,
            torch.full_like(got["values"].to(torch.float32), float("-inf")),
            got["values"].to(torch.float32),
        )
        real = torch.sort(real, dim=-1, descending=True).values
        torch.testing.assert_close(real, got_vals, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------
def prepare_bench(**kwargs: Any):
    """Specialize and compile both executables before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    cfg = _normalize_config(kwargs)
    finalize = get_finalize_kernel(**cfg)
    state = {
        "config": cfg,
        "executable": compile_kernel(get_kernel(**cfg)),
        # None where `finalize_plan` says the dispatcher issues no second launch.
        "finalize": compile_kernel(finalize) if finalize is not None else None,
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Timed comparison against the FlashInfer source pipeline.

    Both sides enqueue the same number of kernels: one where ``finalize_plan``
    returns ``None``, two where it does not.  Every tensor is allocated before the
    closure is built -- allocating inside it would enqueue fill kernels that a
    per-kernel timer attributes to this workload.
    """
    cfg = dict(prepared["config"])
    ex = prepared["executable"]
    ex_finalize = prepared["finalize"]

    data = prepare_data(**cfg)
    tirx_out = alloc_outputs(cfg)
    tirx_args = build_tirx_args(cfg, data, tirx_out)

    def tirx_launch():
        _launch_tirx(ex, ex_finalize, tirx_args)

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


# ---------------------------------------------------------------------------
# Config matrix.
# ---------------------------------------------------------------------------
_DT_TAG = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}
_MODE_TAG = {"basic": "plain", "page_table": "pt", "ragged": "rag"}
_TIE_TAG = {TIE_NONE: "", TIE_SMALL: "_tsmall", TIE_LARGE: "_tlarge"}

# Every finalize (BLOCK_THREADS, ITEMS_PER_THREAD) rung of the ladder at
# :3061-3079, plus k == 1 for the Plain finalize early-out at :3043-3047.
_K_SWEEP = (1, 128, 256, 512, 576, 1024, 2048)
# end_bit = bit_length(max_len), giving 2..5 digit passes in the finalize sort.
_END_BIT_LENGTHS = (200, 4096, 65536, 524288)
# grid is one CTA per row, against 148 SMs on B200.
_ROW_SWEEP = (1, 16, 64, 256)
_VEC_BASE = 8192
# Offsets from _VEC_BASE whose gcd with 16/sizeof(DType) walks every reachable
# VEC_SIZE: f32 gives 4/2/1, the 16-bit dtypes 8/4/2/1 (ComputeFilteredTopKVecSize
# :2934-2943).
_VEC_OFFSETS = {4: (0, -2, -3), 8: (0, -4, -2, -3)}


def _cfg(
    dtype,
    mode,
    num_rows,
    length,
    k,
    deterministic=False,
    tie_break=TIE_NONE,
    dsa_graph_safe=False,
    row_starts=False,
    page_table_row_starts=False,
    row_to_batch=False,
    trivial=False,
    pattern="random",
    tag="",
):
    label = f"{_DT_TAG[dtype]}_{_MODE_TAG[mode]}"
    if deterministic:
        label += "_det"
    label += _TIE_TAG[tie_break]
    label += f"_r{num_rows}_l{length}_k{k}"
    if dsa_graph_safe:
        label += "_dsa"
    if row_starts:
        label += "_rs"
    if page_table_row_starts:
        label += "_pts"
    if row_to_batch:
        label += "_r2b"
    if trivial:
        label += "_trivial"
    if pattern != "random":
        label += f"_{pattern}"
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
        "tie_break": tie_break,
        "dsa_graph_safe": dsa_graph_safe,
        "row_starts": row_starts,
        "page_table_row_starts": page_table_row_starts,
        "row_to_batch": row_to_batch,
        "trivial": trivial,
        "pattern": pattern,
    }


def _reachable_vec_lengths(dtype):
    """(length, vec) for every VEC_SIZE the launcher can instantiate."""
    max_vec = 16 // dtype_bytes(dtype)
    out = []
    for off in _VEC_OFFSETS[max_vec]:
        length = _VEC_BASE + off
        out.append((length, vec_size(dtype, length)))
    seen = set()
    uniq = []
    for length, vec in out:
        if vec not in seen:
            seen.add(vec)
            uniq.append((length, vec))
    return uniq


# Cells that carry the axis sweeps.  Every cell of the 54-cell grid still gets a
# base config, so no template instantiation or finalize behavior goes untested;
# the sweeps that are orthogonal to the cell -- k, end_bit, VEC_SIZE, num_rows --
# run only where they are structurally distinct, instead of being repeated in all
# 54 cells.
_CARRIER_NONDET = ("float32", "basic", False, TIE_NONE)  # base dispatch, no finalize
_CARRIER_DET = ("float32", "basic", True, TIE_NONE)  # sorting finalize, keys+values
_CARRIER_XFORM = ("float32", "page_table", False, TIE_SMALL)  # transform-only finalize


def _build_configs():
    """Representative cover of the FilteredTopK dispatch domain.

    Retained in full:

    * all **54 cells** -- dtype x mode x (deterministic, tie_break) -- one base
      config each.  These pin every distinct kernel instantiation and all three
      finalize behaviors (no launch / sorting / transform-only).
    * every reachable **VEC_SIZE** per dtype (f32 4/2/1, 16-bit 8/4/2/1).
    * the whole **k ladder**, so each of the six finalize ``(BLOCK_THREADS,
      ITEMS_PER_THREAD)`` rungs is exercised, plus ``k == 1`` for the Plain
      finalize early-out.
    * the **end_bit** sweep covering 2..5 digit passes in the finalize sort.
    * both data-dependent **overflow fallbacks**, on both dtype families, since
      the 16-bit and fp32 fallbacks are different code.
    * ``dsa_graph_safe``, the in-kernel **trivial path**, every runtime pointer
      branch, and the ``num_rows`` occupancy axis.

    Reduced: those sweeps used to be repeated inside every cell, which is what a
    cross of all axes costs.  They now run on the carrier cells above, where the
    behavior they exercise actually differs; other cells inherit coverage of the
    swept axis from the carrier that shares its code path.
    """
    configs = []
    base = _VEC_BASE

    # 1. Cell grid: one config per (dtype, mode, deterministic, tie_break).
    for dtype in DTYPES:
        for mode in MODES:
            for deterministic in (False, True):
                for tie_break in TIE_BREAKS:
                    configs.append(_cfg(dtype, mode, 4, base, 256, deterministic, tie_break))

    # 2. VEC_SIZE dispatch, per dtype.
    for dtype in DTYPES:
        for length, _vec in _reachable_vec_lengths(dtype):
            configs.append(_cfg(dtype, "basic", 4, length, 256, tag="vec"))

    # 3. k ladder: the six finalize rungs, with and without the sorting finalize.
    for carrier in (_CARRIER_NONDET, _CARRIER_DET):
        dtype, mode, det, tie = carrier
        for k in _K_SWEEP:
            configs.append(_cfg(dtype, mode, 4, base, k, det, tie))

    # 4. end_bit sweep -- only the sorting finalize depends on the pass count.
    dtype, mode, det, tie = _CARRIER_DET
    for length in _END_BIT_LENGTHS:
        configs.append(
            _cfg(dtype, mode, 2, length, 128 if length < 1024 else 256, det, tie, tag="endbit")
        )

    # 5. Occupancy: grid is one CTA per row against 148 SMs.
    dtype, mode, det, tie = _CARRIER_NONDET
    for rows in _ROW_SWEEP:
        configs.append(_cfg(dtype, mode, rows, base, 256, det, tie))

    # 6. Data-dependent paths.  The 16-bit slow path (:2711-2757) and the fp32
    #    32-bit pivot rebuild (:2800-2901) are different code, so both dtype
    #    families carry the overflow patterns; `prepare_data` asserts each one
    #    still overflows the candidate arena.
    for dtype in ("float32", "float16"):
        for det in (False, True):
            for pattern in ("quantized", "tie_heavy"):
                configs.append(_cfg(dtype, "basic", 2, _OVERFLOW_LENGTH, 256, det, pattern=pattern))
        for tie in (TIE_SMALL, TIE_LARGE):
            configs.append(_cfg(dtype, "basic", 4, base, 256, False, tie, pattern="pivot_tie"))

    # 7. dsa_graph_safe: unconditional filtered dispatch and VEC_SIZE forced to 1.
    for mode in MODES:
        configs.append(_cfg("float32", mode, 4, base, 256, dsa_graph_safe=True))

    # 8. Runtime pointer branches (transform modes only).
    for dtype, mode, det, tie in (_CARRIER_XFORM, ("float32", "ragged", False, TIE_NONE)):
        configs.append(_cfg(dtype, mode, 4, base, 256, det, tie, row_starts=True))
    configs.append(
        _cfg(
            *_CARRIER_XFORM[:2],
            4,
            base,
            256,
            *_CARRIER_XFORM[2:],
            row_starts=True,
            page_table_row_starts=True,
        )
    )
    configs.append(_cfg(*_CARRIER_XFORM[:2], 4, base, 256, *_CARRIER_XFORM[2:], row_to_batch=True))

    # 9. In-kernel trivial path (length <= k, :2409-2429): reachable on the
    #    transform modes through short per-row lengths, and on Plain only when a
    #    forcing flag bypasses the `max_len > k` precondition.
    configs.append(_cfg("float32", "page_table", 4, base, 256, trivial=True))
    configs.append(_cfg("float32", "ragged", 4, base, 256, trivial=True))
    configs.append(_cfg("float32", "basic", 4, 256, 256, False, TIE_SMALL, trivial=True))
    configs.append(_cfg("float16", "basic", 4, 256, 256, True, TIE_LARGE, trivial=True))
    # `length < k` (not just `==`) is the only shape that reaches the `:2416-2417`
    # padding arm of the Plain trivial path; a forcing flag is what makes it
    # dispatchable at all.
    configs.append(_cfg("float16", "basic", 4, 200, 256, True, TIE_LARGE, trivial=True))
    configs.append(_cfg("float32", "basic", 4, 200, 256, dsa_graph_safe=True, trivial=True))

    # Deduplicate on launch parameters (labels alone would keep near-twins).
    seen: dict[tuple, dict[str, Any]] = {}
    for cfg in configs:
        params = tuple(sorted((key, value) for key, value in cfg.items() if key != "label"))
        seen.setdefault(params, cfg)
    deduped = list(seen.values())
    labels = [cfg["label"] for cfg in deduped]
    assert len(set(labels)) == len(labels), "duplicate config label"
    return deduped


def _build_bench_configs():
    """Timed matrix: drop the trivial-path cases, keep everything else.

    The overflow patterns stay in -- the fallbacks are real performance regimes
    and both sides consume identical seeded inputs.
    """
    return [cfg for cfg in _build_configs() if not cfg["trivial"]]


CONFIGS = _build_configs()
BENCH_CONFIGS = _build_bench_configs()
