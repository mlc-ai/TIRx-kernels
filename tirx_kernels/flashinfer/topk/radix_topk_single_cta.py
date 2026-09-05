# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer radix top-k single-CTA port.

Ports ``RadixTopKKernel_Unified<1024, VEC_SIZE, SINGLE_CTA=true, DETERMINISTIC,
MODE, DType, int32_t>`` (``include/flashinfer/topk.cuh:1170``) together with the
device helpers it inlines: ``LoadToSharedOrdered`` (:602),
``RadixSelectFindPivot`` (:851) / ``RadixSelectFromSharedMemory`` (:655),
``RadixSuffixSum`` (:389), ``RadixCollectIndices`` (:901),
``RadixCollectIndicesDeterministic`` (:1017),
``DeterministicThreadStridedCollect`` (:256) and ``RadixTopKTraits``
(``include/flashinfer/topk_common.cuh``).  Host launchers:
``RadixTopKMultiCTA`` (:2172), ``RadixTopKPageTableTransformMultiCTA`` (:1939),
``RadixTopKRaggedTransformMultiCTA`` (:2055); bindings ``csrc/topk.cu``;
Python entries ``flashinfer.top_k`` / ``flashinfer.top_k_page_table_transform``
/ ``flashinfer.top_k_ragged_transform`` (``flashinfer/topk.py``).

One 1024-thread CTA owns a whole row: the row is staged into dynamic shared
memory as monotone ordered integers, an 8-bit radix select walks 4 (fp32) or 2
(fp16/bf16) rounds of a 256-bin shared histogram plus suffix sum to the exact
k-th value, and a single collect pass emits the winners through a
mode-specific epilogue.  Grid is persistent: ``min(num_sms, num_rows)`` CTAs
stride over the rows.

In-scope specialization: the ``SINGLE_CTA=true`` template, i.e. every shape the
launchers map onto one CTA per row (``ceil_div(length, max_chunk_elements) ==
1``).  All three epilogue modes, both collect paths, fp32/fp16/bf16, every
reachable ``VEC_SIZE``, and the runtime ``row_starts`` /
``page_table_row_starts`` / ``row_to_batch`` branches are covered.  Out of
scope: per-row ``top_k_arr`` (``csrc/topk.cu`` always passes ``nullptr``), the
multi-CTA cross-group protocol, and the FilteredTopK and cluster algorithms
that ``TopKDispatch`` selects for other shapes.
"""

import math
import os
from contextlib import nullcontext
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils.topk_radix import (
    atom_shared_add_u32,
    bar_sync,
    from_ordered_u16,
    from_ordered_u32,
    ld_global_u32,
    ld_shared_pair_u32,
    ld_shared_quad_u32,
    ld_shared_u16,
    ld_shared_u32,
    ld_shared_u64,
    shfl_down_u32,
    st_global_u16,
    st_global_u32,
    st_shared_pair_u32,
    st_shared_quad_u32,
    st_shared_u16,
    st_shared_u32,
    to_ordered_u16,
    to_ordered_u32,
    u64_hi,
    u64_lo,
    warp_inclusive_sum_u32,
)
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "radix_topk_single_cta",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}
LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# ---------------------------------------------------------------------------
# Source constants (topk.cuh:2172-2270 launcher, :1192-1201 smem layout).
# ---------------------------------------------------------------------------
BLOCK_THREADS = 1024
RADIX = 256
RADIX_BITS = 8
NUM_SCALARS_SINGLE_CTA = 5  # prefix, remaining_k, found_bucket, found_rk, out_counter

# fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 5), 16B aligned (:1194-1200)
_FIXED_SMEM_BYTES = 4 * (RADIX + RADIX + NUM_SCALARS_SINGLE_CTA)
_FIXED_SMEM_ALIGNED = (_FIXED_SMEM_BYTES + 15) // 16 * 16

# sizeof(cub::BlockScan<uint32_t, 1024, BLOCK_SCAN_RAKING_MEMOIZE>::TempStorage),
# doubled into RADIX_TOPK_LAUNCH_SMEM_HEADROOM (topk.cuh:46-52).  The launcher
# reserves it only when `deterministic` is set, which shrinks the chunk budget
# and therefore the single-CTA shape domain.
_DET_BLOCK_SCAN_TEMP_BYTES = 4256
_DET_LAUNCH_HEADROOM = 2 * _DET_BLOCK_SCAN_TEMP_BYTES

# cudaDevAttrMaxSharedMemoryPerBlockOptin on the accepted SM100 target (B200).
# Kept as a compile-profile constant so module import and CPU prepare never
# touch CUDA; `run_test` cross-checks it against the live device.
DEFAULT_MAX_SMEM_OPTIN = 232448
PREPARE_MAX_SMEM_OPTIN_ENV = "TIRX_PREPARE_MAX_SMEM_OPTIN"

DTYPES = ("float32", "float16", "bfloat16")
MODES = ("basic", "page_table", "ragged")

_TIE_BREAK_NONE = 0
_SOURCE_ALGO_ENV = "FLASHINFER_TOPK_ALGO"
_SOURCE_ALGO_VALUE = "multi_cta"


def hardware_max_smem_optin(default: int = DEFAULT_MAX_SMEM_OPTIN) -> int:
    """Compile-profile ``cudaDevAttrMaxSharedMemoryPerBlockOptin``."""
    configured = os.environ.get(PREPARE_MAX_SMEM_OPTIN_ENV)
    if configured is not None:
        value = int(configured)
        if value < 1:
            raise ValueError(f"{PREPARE_MAX_SMEM_OPTIN_ENV} must be positive, got {value}")
        return value
    return default


def ordered_dtype(dtype: str) -> str:
    """``RadixTopKTraits<DType>::OrderedType`` (topk_common.cuh:26/53/79)."""
    return "uint32" if dtype == "float32" else "uint16"


def ordered_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def dtype_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def num_rounds(dtype: str) -> int:
    return ordered_bytes(dtype) * 8 // RADIX_BITS


def vec_size(dtype: str, length: int, scalar_addressing: bool = False) -> int:
    """Launcher vec-size dispatch (topk.cuh:2179, :1947, :2064).

    ``scalar_addressing`` mirrors the ``row_starts != nullptr`` clause the
    page-table and ragged launchers apply before the gcd.
    """
    if scalar_addressing:
        return 1
    return math.gcd(16 // dtype_bytes(dtype), length)


def max_chunk_elements(
    dtype: str, length: int, deterministic: bool, scalar_addressing: bool
) -> int:
    """``max_chunk_elements`` after round-down and the min-chunk floor (:2199-2203)."""
    headroom = _DET_LAUNCH_HEADROOM if deterministic else 0
    available = hardware_max_smem_optin() - _FIXED_SMEM_ALIGNED - headroom
    if available <= 0:
        raise ValueError("no shared memory available for the ordered chunk")
    vec = vec_size(dtype, length, scalar_addressing)
    chunk = (available // ordered_bytes(dtype)) // vec * vec
    return max(chunk, vec * BLOCK_THREADS)


def launch_plan(
    dtype: str,
    mode: str,
    length: int,
    num_rows: int,
    deterministic: bool,
    scalar_addressing: bool = False,
    num_sms: int | None = None,
) -> dict[str, Any]:
    """Replicate the host launcher's chunk/grid/smem arithmetic (:2179-2231)."""
    if num_sms is None:
        from tirx_kernels.runner import hardware_num_sms

        num_sms = hardware_num_sms()
    vec = vec_size(dtype, length, scalar_addressing)
    max_chunk = max_chunk_elements(dtype, length, deterministic, scalar_addressing)
    ctas_per_group = (length + max_chunk - 1) // max_chunk
    chunk = (length + ctas_per_group - 1) // ctas_per_group
    chunk = (chunk + vec - 1) // vec * vec
    chunk = min(chunk, max_chunk)
    num_groups = min(num_sms // ctas_per_group, num_rows)
    if num_groups == 0:
        num_groups = 1
    return {
        "vec_size": vec,
        "max_chunk_elements": max_chunk,
        "ctas_per_group": ctas_per_group,
        "chunk_size": chunk,
        "single_cta": ctas_per_group == 1,
        "num_groups": num_groups,
        "grid": num_groups * ctas_per_group,
        "smem_bytes": _FIXED_SMEM_ALIGNED + chunk * ordered_bytes(dtype),
        "num_rounds": num_rounds(dtype),
    }


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
    if not plan["single_cta"]:
        raise ValueError(
            f"({dtype}, len={length}, det={deterministic}) needs "
            f"{plan['ctas_per_group']} CTAs per row: outside the SINGLE_CTA dispatch domain"
        )
    return plan


def page_table_batch(num_rows: int, row_to_batch: bool) -> int:
    """Rows map identity->batch unless ``row_to_batch`` folds them onto fewer."""
    return max(1, num_rows // 2) if row_to_batch else num_rows


def aux_elements(mode: str, num_rows: int, length: int, row_to_batch: bool) -> int:
    """Elements of the mode-specific ``aux_data`` buffer."""
    if mode == "page_table":
        return page_table_batch(num_rows, row_to_batch) * length
    if mode == "ragged":
        return num_rows
    return 1


def _ld_global_bits(buf, elem_index, is32):
    """One scalar element's raw bits (``ld.global.b32`` | ``ld.global.b16``)."""
    if is32:
        out = K.local_scalar("uint32")
        K.ptx.ld.global_.b32(out, buf.ptr_to([elem_index]))
        return out
    out16 = K.local_scalar("uint16")
    K.ptx.ld.global_.b16(out16, buf.ptr_to([elem_index]))
    return out16


def _ld_global_words(buf, elem_index, load_bytes):
    """One vector load of ``load_bytes`` bytes, returned as 32-bit words."""
    if load_bytes == 16:
        w = K.alloc_local((4,), "uint32", align=16)
        K.ptx["ld.global.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index]))
        return [w[0], w[1], w[2], w[3]]
    if load_bytes == 8:
        w = K.alloc_local((2,), "uint32", align=8)
        K.ptx["ld.global.v2.b32"](w[0], w[1], buf.ptr_to([elem_index]))
        return [w[0], w[1]]
    w = K.alloc_local((1,), "uint32")
    K.ptx.ld.global_.b32(w[0], buf.ptr_to([elem_index]))
    return [w[0]]


def _stage_vector(buf, s_ordered, row_in, i, vec, load_bytes, is32, to_ordered, st_key):
    """Load and convert one vector, then store it at the load's width."""
    base = row_in + K.cast(i, "int64")
    if load_bytes == 2:
        st_key(s_ordered, i, to_ordered(_ld_global_bits(buf, base, False)))
        return
    words = _ld_global_words(buf, base, load_bytes)
    if is32:
        keys = [to_ordered(word) for word in words]
    else:
        keys = []
        for word in words:
            lo = to_ordered(K.cast(K.bitwise_and(word, K.uint32(0xFFFF)), "uint16"))
            hi = to_ordered(K.cast(K.shift_right(word, K.uint32(16)), "uint16"))
            keys.append(
                K.bitwise_or(K.cast(lo, "uint32"), K.shift_left(K.cast(hi, "uint32"), K.uint32(16)))
            )
    if len(keys) == 4:
        st_shared_quad_u32(s_ordered, i, keys[0], keys[1], keys[2], keys[3])
    elif len(keys) == 2:
        st_shared_pair_u32(s_ordered, i, keys[0], keys[1])
    else:
        st_shared_u32(s_ordered, i, keys[0])


def _emit(out_idx, out_val, row_out, i, key, pos, offset, basic, ragged, is32, dtype):
    """The mode epilogue the collect passes call per selected element (:1339-1375)."""
    slot = row_out + K.cast(pos, "int64")
    if basic:
        st_global_u32(out_idx, slot, K.reinterpret("uint32", i))
        if is32:
            st_global_u32(out_val, slot, from_ordered_u32(key))
        else:
            st_global_u16(out_val, slot, from_ordered_u16(K.cast(key, "uint16")))
    elif ragged:
        st_global_u32(out_idx, slot, K.reinterpret("uint32", i + offset))
    else:
        st_global_u32(out_idx, slot, K.reinterpret("uint32", i))


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
RAKING_THREADS = 32
RAKING_SEGMENT = BLOCK_THREADS // RAKING_THREADS  # 32
RAKING_STRIDE = RAKING_SEGMENT + 1  # cub BlockRakingLayout segment padding
RAKING_ELEMENTS = RAKING_THREADS * RAKING_STRIDE  # 1056


def _raking_offset(tx):
    """cub ``BlockRakingLayout::PlacementPtr``: ``tid + tid / SEGMENT_LENGTH``."""
    return tx + K.shift_right(tx, K.int32(5))


def scan_scratch_elements(deterministic: bool) -> int:
    """uint32 slots for the collect scan scratch (0 when non-deterministic)."""
    return RAKING_ELEMENTS * 2 if deterministic else 0


def get_kernel(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 16,
    length: int = 32000,
    k: int = 256,
    deterministic: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    **kwargs,
):
    """Return the TIRx specialization for one launcher dispatch cell."""
    scalar_addressing = row_starts and mode != "basic"
    plan = _validate(dtype, mode, num_rows, length, k, deterministic, scalar_addressing)
    ordered = ordered_dtype(dtype)
    is32 = ordered == "uint32"
    obits = 32 if is32 else 16
    rounds = num_rounds(dtype)
    vec = plan["vec_size"]
    chunk = plan["chunk_size"]
    grid = plan["grid"]
    load_bytes = vec * dtype_bytes(dtype)
    aux_elems = aux_elements(mode, num_rows, length, row_to_batch)
    scan_elems = scan_scratch_elements(deterministic)
    basic = mode == "basic"
    page_table = mode == "page_table"
    ragged = mode == "ragged"
    # Basic mode's `length` is the static kernel stride, so `k >= length` folds.
    basic_trivial = basic and k >= length

    def to_ordered(bits):
        return to_ordered_u32(bits) if is32 else to_ordered_u16(bits)

    def from_ordered(key):
        return from_ordered_u32(key) if is32 else from_ordered_u16(key)

    def ld_key(buf, i):
        return ld_shared_u32(buf, i) if is32 else ld_shared_u16(buf, i)

    def st_key(buf, i, v):
        st_shared_u32(buf, i, v) if is32 else st_shared_u16(buf, i, v)

    def ld_i32(buf, i):
        return K.reinterpret("int32", ld_global_u32(buf, i))

    def st_i32(buf, i, v):
        st_global_u32(buf, i, K.reinterpret("uint32", v))

    def copy_value(dst, dst_i, src, src_i):
        bits = _ld_global_bits(src, src_i, is32)
        if is32:
            st_global_u32(dst, dst_i, bits)
        else:
            st_global_u16(dst, dst_i, bits)

    @K.kernel(warps=BLOCK_THREADS // 32, arch="sm_100a", grid=grid)
    def radix_topk_single_cta(
        inp: K.gptr[dtype, (num_rows * length,)],
        out_idx: K.gptr[K.i32, (num_rows * k,)],
        out_val: K.gptr[dtype, (num_rows * k,)],
        aux: K.gptr[K.i32, (aux_elems,)],
        lengths_g: K.gptr[K.i32, (num_rows,)],
        row_starts_g: K.gptr[K.i32, (num_rows,)],
        pt_starts_g: K.gptr[K.i32, (num_rows,)],
        row_to_batch_g: K.gptr[K.i32, (num_rows,)],
        aux_stride: K.i64,
    ):
        group_id = K.cta_id()
        tx = K.thread_id()

        pool = K.smem_pool()
        s_hist = pool.alloc((RADIX,), "uint32")
        s_suffix = pool.alloc((RADIX,), "uint32")
        s_scalars = pool.alloc((NUM_SCALARS_SINGLE_CTA,), "uint32")
        s_ordered = pool.alloc((chunk,), ordered, align=16)
        if deterministic:
            s_scan = pool.alloc((scan_elems,), "uint32", align=16)

        # --- persistent row loop (:1218-1220) -----------------------------
        row_idx = K.local_scalar("int32", init=group_id)
        with K.While(row_idx < num_rows):
            # --- per-row header (:1221-1247) ------------------------------
            row_start = K.int32(0)
            page_start = K.int32(0)
            row_len = length
            if not basic:
                if row_starts:
                    row_start = ld_i32(row_starts_g, row_idx)
                if page_table:
                    if page_table_row_starts:
                        page_start = ld_i32(pt_starts_g, row_idx)
                    else:
                        page_start = row_start
                row_len = ld_i32(lengths_g, row_idx)
            row_in = K.cast(row_idx, "int64") * K.int64(length) + K.cast(row_start, "int64")
            row_out = K.cast(row_idx, "int64") * K.int64(k)

            # --- mode trivial early-out (:1243-1307) ----------------------
            take_main = None
            if not (basic and not basic_trivial):
                take_main = K.alloc_local([1], "int32")
                K.assign(take_main[0], K.int32(1))
            if basic:
                if basic_trivial:
                    K.assign(take_main[0], K.int32(0))
                    with K.serial(tx, row_len, step=BLOCK_THREADS) as i0:
                        with K.If(i0 < k), K.Then():
                            out_i = row_out + K.cast(i0, "int64")
                            st_i32(out_idx, out_i, i0)
                            copy_value(
                                out_val,
                                out_i,
                                inp,
                                K.cast(row_idx, "int64") * K.int64(length) + K.cast(i0, "int64"),
                            )
            batch_idx = row_idx
            offset = K.int32(0)
            if page_table:
                if row_to_batch:
                    batch_idx = ld_i32(row_to_batch_g, row_idx)
                with K.If(row_len <= k), K.Then():
                    K.assign(take_main[0], K.int32(0))
                    src0 = K.cast(batch_idx, "int64") * aux_stride
                    with K.serial(tx, k, step=BLOCK_THREADS) as i1:
                        page_id = K.local_scalar("int32", init=K.int32(-1))
                        with K.If(i1 < row_len), K.Then():
                            K.assign(page_id, ld_i32(aux, src0 + K.cast(page_start + i1, "int64")))
                        st_i32(out_idx, row_out + K.cast(i1, "int64"), page_id)
            if ragged:
                offset = ld_i32(aux, K.cast(row_idx, "int64"))
                with K.If(row_len <= k), K.Then():
                    K.assign(take_main[0], K.int32(0))
                    with K.serial(tx, k, step=BLOCK_THREADS) as i2:
                        val2 = K.local_scalar("int32", init=K.int32(-1))
                        with K.If(i2 < row_len), K.Then():
                            K.assign(val2, i2 + offset)
                        st_i32(out_idx, row_out + K.cast(i2, "int64"), val2)

            static_main = basic and not basic_trivial
            with (
                nullcontext() if static_main else K.If(take_main[0] == 1),
                nullcontext() if static_main else K.Then(),
            ):
                # === Stage 1: stage the row as monotone keys (:605-623) ====
                if basic:
                    aligned = length // vec * vec
                else:
                    aligned = K.truncdiv(row_len, K.int32(vec)) * K.int32(vec)
                with K.serial(tx * vec, aligned, step=BLOCK_THREADS * vec, unroll=2) as iv:
                    _stage_vector(
                        inp, s_ordered, row_in, iv, vec, load_bytes, is32, to_ordered, st_key
                    )
                with K.serial(aligned + tx, row_len, step=BLOCK_THREADS) as it_tail:
                    st_key(
                        s_ordered,
                        it_tail,
                        to_ordered(_ld_global_bits(inp, row_in + K.cast(it_tail, "int64"), is32)),
                    )
                bar_sync()

                # scalar caches (:669-677)
                with K.If(tx == 0), K.Then():
                    st_shared_pair_u32(s_scalars, 0, K.uint32(0), K.uint32(k))
                    st_shared_u32(s_scalars, 4, K.uint32(0))
                bar_sync()

                # === Stage 2: NUM_ROUNDS radix-select rounds (:690-774) ====
                with K.serial(0, rounds) as rnd:
                    shift = K.int32(obits) - (rnd + 1) * 8
                    mask = K.local_scalar("uint32", init=K.uint32(0))
                    with K.If(rnd != 0), K.Then():
                        K.assign(
                            mask,
                            K.shift_left(
                                K.uint32(0xFFFFFFFF), K.cast(K.int32(obits) - rnd * 8, "uint32")
                            ),
                        )
                    packed = ld_shared_u64(s_scalars, 0)
                    prefix = u64_lo(packed)
                    remaining_k = u64_hi(packed)

                    with K.serial(tx, RADIX, step=BLOCK_THREADS) as bh:
                        st_shared_u32(s_hist, bh, K.uint32(0))
                    bar_sync()

                    with K.serial(tx, row_len, step=BLOCK_THREADS, unroll=2) as ih:
                        key = K.cast(ld_key(s_ordered, ih), "uint32")
                        with K.If(K.bitwise_and(key, mask) == prefix), K.Then():
                            bucket = K.bitwise_and(
                                K.shift_right(key, K.cast(shift, "uint32")), K.uint32(0xFF)
                            )
                            atom_shared_add_u32(s_hist, K.cast(bucket, "int32"), K.uint32(1))
                    bar_sync()

                    with K.serial(tx, RADIX, step=BLOCK_THREADS) as bs:
                        st_shared_u32(s_suffix, bs, ld_shared_u32(s_hist, bs))
                    bar_sync()

                    # RadixSuffixSum (:389-406)
                    with K.unroll(8) as step:
                        stride = K.shift_left(K.int32(1), step)
                        acc = K.local_scalar("uint32", init=K.uint32(0))
                        with K.If(tx < RADIX), K.Then():
                            K.assign(acc, ld_shared_u32(s_suffix, tx))
                            with K.If(tx + stride < RADIX), K.Then():
                                K.assign(acc, acc + ld_shared_u32(s_suffix, tx + stride))
                        bar_sync()
                        with K.If(tx < RADIX), K.Then():
                            st_shared_u32(s_suffix, tx, acc)
                        bar_sync()

                    # threshold bucket (:753-767)
                    with K.If(tx == 0), K.Then():
                        st_shared_pair_u32(s_scalars, 2, K.uint32(0), remaining_k)
                    bar_sync()
                    with K.If(tx < RADIX), K.Then():
                        count_ge = ld_shared_u32(s_suffix, tx)
                        count_gt = K.local_scalar("uint32", init=K.uint32(0))
                        with K.If(tx + 1 < RADIX), K.Then():
                            K.assign(count_gt, ld_shared_u32(s_suffix, tx + 1))
                        with K.If(K.And(count_ge >= remaining_k, count_gt < remaining_k)), K.Then():
                            st_shared_pair_u32(
                                s_scalars, 2, K.cast(tx, "uint32"), remaining_k - count_gt
                            )
                    bar_sync()
                    with K.If(tx == 0), K.Then():
                        found = ld_shared_pair_u32(s_scalars, 2)
                        st_shared_pair_u32(
                            s_scalars,
                            0,
                            K.bitwise_or(prefix, K.shift_left(found[0], K.cast(shift, "uint32"))),
                            found[1],
                        )
                    bar_sync()

                pivot = ld_shared_u32(s_scalars, 0)
                if not is32:
                    pivot = K.bitwise_and(pivot, K.uint32(0xFFFF))

                # === Stage 3: row-wide gt (and eq) counts (:782-830) =======
                with K.If(tx == 0), K.Then():
                    if deterministic:
                        st_shared_pair_u32(s_suffix, 0, K.uint32(0), K.uint32(0))
                    else:
                        st_shared_u32(s_suffix, 0, K.uint32(0))
                bar_sync()
                my_gt = K.local_scalar("uint32")
                my_eq = K.local_scalar("uint32")
                K.assign(my_gt, K.uint32(0))
                K.assign(my_eq, K.uint32(0))
                with K.serial(tx, row_len, step=BLOCK_THREADS, unroll=2) as ic:
                    key2 = K.cast(ld_key(s_ordered, ic), "uint32")
                    K.assign(my_gt, my_gt + K.Select(key2 > pivot, K.uint32(1), K.uint32(0)))
                    if deterministic:
                        K.assign(my_eq, my_eq + K.Select(key2 == pivot, K.uint32(1), K.uint32(0)))
                with K.unroll(5) as step:
                    delta = K.shift_right(K.int32(16), step)
                    K.assign(my_gt, my_gt + shfl_down_u32(my_gt, delta))
                    if deterministic:
                        K.assign(my_eq, my_eq + shfl_down_u32(my_eq, delta))
                lane = K.bitwise_and(tx, K.int32(31))
                with K.If(K.And(lane == 0, my_gt > K.uint32(0))), K.Then():
                    atom_shared_add_u32(s_suffix, 0, my_gt)
                if deterministic:
                    with K.If(K.And(lane == 0, my_eq > K.uint32(0))), K.Then():
                        atom_shared_add_u32(s_suffix, 1, my_eq)
                bar_sync()
                gt_count = ld_shared_u32(s_suffix, 0)

                # === Stage 3b: epilogue-scope aux, per row (:1344-1345, :1373)
                src_base = K.int64(0)
                if page_table:
                    if row_to_batch:
                        batch_idx = ld_i32(row_to_batch_g, row_idx)
                    src_base = K.cast(batch_idx, "int64") * aux_stride
                if ragged:
                    offset = ld_i32(aux, K.cast(row_idx, "int64"))

                if not deterministic:
                    # === Stage 4a: non-deterministic collect (:906-963) ====
                    with K.If(tx == 0), K.Then():
                        st_shared_u32(s_hist, 0, K.uint32(0))
                        with K.If(gt_count > K.uint32(0)), K.Then():
                            st_shared_u32(s_hist, 1, atom_shared_add_u32(s_scalars, 4, gt_count))
                    bar_sync()
                    with K.serial(tx, row_len, step=BLOCK_THREADS, unroll=2) as i:
                        key3 = K.cast(ld_key(s_ordered, i), "uint32")
                        with K.If(key3 > pivot), K.Then():
                            local_pos = atom_shared_add_u32(s_hist, 0, K.uint32(1))
                            pos = K.cast(ld_shared_u32(s_hist, 1) + local_pos, "int32")
                            _emit(
                                out_idx,
                                out_val,
                                row_out,
                                i,
                                key3,
                                pos,
                                offset,
                                basic,
                                ragged,
                                is32,
                                dtype,
                            )
                    bar_sync()
                    with K.serial(tx, row_len, step=BLOCK_THREADS, unroll=2) as i:
                        key4 = K.cast(ld_key(s_ordered, i), "uint32")
                        with K.If(key4 == pivot), K.Then():
                            pos2 = K.cast(atom_shared_add_u32(s_scalars, 4, K.uint32(1)), "int32")
                            with K.If(pos2 < k), K.Then():
                                _emit(
                                    out_idx,
                                    out_val,
                                    row_out,
                                    i,
                                    pivot,
                                    pos2,
                                    offset,
                                    basic,
                                    ragged,
                                    is32,
                                    dtype,
                                )
                else:
                    # === Stage 4b: deterministic collect (:1023-1138) ======
                    eq_count = ld_shared_u32(s_suffix, 1)
                    with K.If(tx == 0), K.Then():
                        need = K.local_scalar("uint32", init=K.uint32(0))
                        with K.If(K.uint32(k) > gt_count), K.Then():
                            K.assign(need, K.uint32(k) - gt_count)
                        st_shared_quad_u32(s_hist, 0, K.uint32(0), K.uint32(0), gt_count, need)
                        st_shared_u32(s_hist, 4, K.uint32(0))
                    bar_sync()
                    collect_plan = ld_shared_quad_u32(s_hist, 0)
                    gt_base = collect_plan[0]
                    eq_base = collect_plan[2]
                    eq_limit = collect_plan[3]
                    gt_limit = K.local_scalar("uint32", init=K.uint32(0))
                    with K.If(K.uint32(k) > gt_base), K.Then():
                        K.assign(gt_limit, K.uint32(k) - gt_base)

                    with K.If(eq_limit == K.uint32(0)):
                        with K.Then():
                            sel = K.local_scalar("uint32", init=K.uint32(0))
                            with K.serial(tx, row_len, step=BLOCK_THREADS) as id1:
                                key5 = K.cast(ld_key(s_ordered, id1), "uint32")
                                K.assign(
                                    sel, sel + K.Select(key5 > pivot, K.uint32(1), K.uint32(0))
                                )
                            # cub BLOCK_SCAN_RAKING_MEMOIZE exclusive sum (:268-270):
                            # place into the padded raking grid, one warp serially
                            # reduces its 32-element segment while memoizing it in
                            # registers, a warp shuffle scan runs over the segment
                            # totals, then the same warp scatters the prefixes back.
                            rake_off = _raking_offset(tx)
                            st_shared_u32(s_scan, rake_off, sel)
                            bar_sync()
                            with K.If(tx < RAKING_THREADS), K.Then():
                                rake_base = tx * RAKING_STRIDE
                                cache = K.alloc_local([RAKING_SEGMENT], "uint32")
                                total = K.local_scalar("uint32", init=K.uint32(0))
                                with K.unroll(RAKING_SEGMENT) as j:
                                    K.ptx.mov.b32(cache[j], ld_shared_u32(s_scan, rake_base + j))
                                    K.assign(total, total + cache[j])
                                scan_run = K.local_scalar("uint32")
                                K.assign(scan_run, warp_inclusive_sum_u32(total, tx) - total)
                                with K.unroll(RAKING_SEGMENT) as j:
                                    st_shared_u32(s_scan, rake_base + j, scan_run)
                                    K.assign(scan_run, scan_run + cache[j])
                            bar_sync()
                            pre = ld_shared_u32(s_scan, rake_off)
                            with K.If(K.And(sel > K.uint32(0), pre < gt_limit)), K.Then():
                                emit_pos = K.local_scalar("uint32")
                                emit_end = K.local_scalar("uint32")
                                done = K.local_scalar("int32")
                                K.assign(emit_pos, pre)
                                K.assign(emit_end, pre + sel)
                                with K.If(emit_end > gt_limit), K.Then():
                                    K.assign(emit_end, gt_limit)
                                K.assign(done, 0)
                                with K.serial(tx, row_len, step=BLOCK_THREADS) as id3:
                                    with K.If(done == 0), K.Then():
                                        key6 = K.cast(ld_key(s_ordered, id3), "uint32")
                                        with K.If(key6 > pivot), K.Then():
                                            _emit(
                                                out_idx,
                                                out_val,
                                                row_out,
                                                id3,
                                                key6,
                                                K.cast(gt_base + emit_pos, "int32"),
                                                offset,
                                                basic,
                                                ragged,
                                                is32,
                                                dtype,
                                            )
                                            K.assign(emit_pos, emit_pos + K.uint32(1))
                                            with K.If(emit_pos == emit_end), K.Then():
                                                K.assign(done, 1)
                            bar_sync()
                        with K.Else():
                            sel_gt = K.local_scalar("uint32")
                            sel_eq = K.local_scalar("uint32")
                            K.assign(sel_gt, K.uint32(0))
                            K.assign(sel_eq, K.uint32(0))
                            with K.serial(tx, row_len, step=BLOCK_THREADS) as id4:
                                key7 = K.cast(ld_key(s_ordered, id4), "uint32")
                                K.assign(
                                    sel_gt,
                                    sel_gt + K.Select(key7 > pivot, K.uint32(1), K.uint32(0)),
                                )
                                K.assign(
                                    sel_eq,
                                    sel_eq + K.Select(key7 == pivot, K.uint32(1), K.uint32(0)),
                                )
                            # The same raking scan over the {gt, eq} pair (:1122-1125).
                            rake_off2 = _raking_offset(tx) * 2
                            st_shared_pair_u32(s_scan, rake_off2, sel_gt, sel_eq)
                            bar_sync()
                            with K.If(tx < RAKING_THREADS), K.Then():
                                rake_base2 = tx * RAKING_STRIDE * 2
                                cache_gt = K.alloc_local([RAKING_SEGMENT], "uint32")
                                cache_eq = K.alloc_local([RAKING_SEGMENT], "uint32")
                                total_gt = K.local_scalar("uint32")
                                total_eq = K.local_scalar("uint32")
                                K.assign(total_gt, K.uint32(0))
                                K.assign(total_eq, K.uint32(0))
                                with K.unroll(RAKING_SEGMENT) as j:
                                    seg = ld_shared_pair_u32(s_scan, rake_base2 + j * 2)
                                    K.ptx.mov.b32(cache_gt[j], seg[0])
                                    K.ptx.mov.b32(cache_eq[j], seg[1])
                                    K.assign(total_gt, total_gt + cache_gt[j])
                                    K.assign(total_eq, total_eq + cache_eq[j])
                                run_gt = K.local_scalar("uint32")
                                run_eq = K.local_scalar("uint32")
                                K.assign(run_gt, warp_inclusive_sum_u32(total_gt, tx) - total_gt)
                                K.assign(run_eq, warp_inclusive_sum_u32(total_eq, tx) - total_eq)
                                with K.unroll(RAKING_SEGMENT) as j:
                                    st_shared_pair_u32(s_scan, rake_base2 + j * 2, run_gt, run_eq)
                                    K.assign(run_gt, run_gt + cache_gt[j])
                                    K.assign(run_eq, run_eq + cache_eq[j])
                            bar_sync()
                            seg_out = ld_shared_pair_u32(s_scan, rake_off2)
                            cur_gt = K.local_scalar("uint32")
                            cur_eq = K.local_scalar("uint32")
                            K.assign(cur_gt, seg_out[0])
                            K.assign(cur_eq, seg_out[1])
                            with K.serial(tx, row_len, step=BLOCK_THREADS) as i8:
                                key8 = K.cast(ld_key(s_ordered, i8), "uint32")
                                with K.If(K.And(key8 > pivot, cur_gt < gt_limit)):
                                    with K.Then():
                                        _emit(
                                            out_idx,
                                            out_val,
                                            row_out,
                                            i8,
                                            key8,
                                            K.cast(gt_base + cur_gt, "int32"),
                                            offset,
                                            basic,
                                            ragged,
                                            is32,
                                            dtype,
                                        )
                                        K.assign(cur_gt, cur_gt + K.uint32(1))
                                    with K.Else():
                                        with (
                                            K.If(K.And(key8 == pivot, cur_eq < eq_limit)),
                                            K.Then(),
                                        ):
                                            _emit(
                                                out_idx,
                                                out_val,
                                                row_out,
                                                i8,
                                                key8,
                                                K.cast(eq_base + cur_eq, "int32"),
                                                offset,
                                                basic,
                                                ragged,
                                                is32,
                                                dtype,
                                            )
                                            K.assign(cur_eq, cur_eq + K.uint32(1))
                            bar_sync()

                # === Stage 5: PageTable in-place gather (:1352-1358) =======
                if page_table:
                    bar_sync()
                    with K.serial(tx, k, step=BLOCK_THREADS) as ig:
                        out_i = row_out + K.cast(ig, "int64")
                        gidx = ld_i32(out_idx, out_i)
                        st_i32(
                            out_idx,
                            out_i,
                            ld_i32(aux, src_base + K.cast(page_start + gidx, "int64")),
                        )

            K.assign(row_idx, row_idx + grid)

    return radix_topk_single_cta.func.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


# ---------------------------------------------------------------------------
# Config matrix: every launcher dispatch cell reachable with one CTA per row.
# ---------------------------------------------------------------------------
_DT_TAG = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}
_MODE_TAG = {"basic": "basic", "page_table": "pt", "ragged": "rag"}


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
    tag: str = "",
) -> dict[str, Any]:
    label = f"{_DT_TAG[dtype]}_{_MODE_TAG[mode]}"
    if deterministic:
        label += "_det"
    label += f"_r{num_rows}_l{length}_k{k}"
    if row_starts:
        label += "_rs"
    if page_table_row_starts:
        label += "_pts"
    if row_to_batch:
        label += "_r2b"
    if trivial:
        label += "_trivial"
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
    }


def _max_single_cta_length(dtype: str, deterministic: bool) -> int:
    """Largest ``length`` the launcher still maps onto one CTA per row."""
    headroom = _DET_LAUNCH_HEADROOM if deterministic else 0
    available = hardware_max_smem_optin() - _FIXED_SMEM_ALIGNED - headroom
    best = available // ordered_bytes(dtype)
    # Walk down to the first length whose own gcd-derived vec size keeps it
    # inside its own max_chunk_elements (the launcher recomputes both).
    for candidate in range(best, best - 64, -1):
        if candidate < 1:
            break
        if candidate <= max_chunk_elements(dtype, candidate, deterministic, False):
            return candidate
    raise RuntimeError("no single-CTA length found")


def _vec_lengths(dtype: str, deterministic: bool) -> list[tuple[int, int]]:
    """One small length per reachable VEC_SIZE, as ``(vec, length)``."""
    widest = 16 // dtype_bytes(dtype)
    out: list[tuple[int, int]] = []
    vec = widest
    while vec >= 1:
        if vec == widest:
            length = 32768
        elif vec == 1:
            length = 32001
        else:
            length = 32768 + vec  # gcd(widest, 32768 + vec) == vec
        assert vec_size(dtype, length) == vec, (dtype, vec, length)
        out.append((vec, length))
        vec //= 2
    return out


def _build_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for mode in MODES:
            for deterministic in (False, True):
                # VEC_SIZE dispatch: every reachable instantiation.
                for vec, length in _vec_lengths(dtype, deterministic):
                    configs.append(
                        _cfg(dtype, mode, 8, length, 256, deterministic, tag=f"vec{vec}")
                    )
                # Largest chunk the smem budget still keeps on one CTA.
                big = _max_single_cta_length(dtype, deterministic)
                configs.append(_cfg(dtype, mode, 4, big, 256, deterministic, tag="maxchunk"))
                # k sweep inside one shape.
                for k in (64, 512, 1024):
                    configs.append(_cfg(dtype, mode, 8, 32768, k, deterministic))
                # Launch topology: single row, below num_sms, above num_sms.
                for rows in (1, 64, 256):
                    configs.append(_cfg(dtype, mode, rows, 32768, 256, deterministic))
                # Trivial early-out branch (k >= length / length <= top_k).
                configs.append(_cfg(dtype, mode, 8, 2048, 2048, deterministic, trivial=True))
                # Runtime pointer branches.
                if mode == "page_table":
                    configs.append(_cfg(dtype, mode, 8, 32768, 256, deterministic, row_starts=True))
                    configs.append(
                        _cfg(
                            dtype,
                            mode,
                            8,
                            32768,
                            256,
                            deterministic,
                            row_starts=True,
                            page_table_row_starts=True,
                        )
                    )
                    configs.append(
                        _cfg(dtype, mode, 8, 32768, 256, deterministic, row_to_batch=True)
                    )
                elif mode == "ragged":
                    configs.append(_cfg(dtype, mode, 8, 32768, 256, deterministic, row_starts=True))
    return configs


CONFIGS = _build_configs()


def _build_bench_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for mode in MODES:
            for deterministic in (False, True):
                configs.append(_cfg(dtype, mode, 8, 8192, 256, deterministic))
                configs.append(_cfg(dtype, mode, 64, 32768, 512, deterministic))
                configs.append(
                    _cfg(
                        dtype,
                        mode,
                        256,
                        _max_single_cta_length(dtype, deterministic),
                        1024,
                        deterministic,
                        tag="maxchunk",
                    )
                )
    return configs


BENCH_CONFIGS = _build_bench_configs()


# ---------------------------------------------------------------------------
# Data preparation and the FlashInfer reference path.
# ---------------------------------------------------------------------------
def _torch_dtype(dtype: str):
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def prepare_data(
    dtype: str = "float32",
    mode: str = "basic",
    num_rows: int = 16,
    length: int = 32000,
    k: int = 256,
    deterministic: bool = False,
    row_starts: bool = False,
    page_table_row_starts: bool = False,
    row_to_batch: bool = False,
    trivial: bool = False,
    **kwargs,
):
    """Logical inputs for one config.

    Scores follow the source benchmark's ``random`` pattern: seeded fp32 normals
    cast to the target dtype.  Ties are expected and wanted -- fp16 has only
    63487 distinct finite values while this kernel's single-CTA domain reaches
    115184 elements per row, so a tie-free input does not exist at the large
    shapes, and ties are what exercise the ``== pivot`` collect pass.  See
    ``_compare`` for the tie-robust correctness criterion.
    """
    import torch

    device = "cuda"
    rows = num_rows
    generator = torch.Generator(device=device).manual_seed(1234)
    scores = torch.randn(rows, length, dtype=torch.float32, device=device, generator=generator)
    scores = scores.to(_torch_dtype(dtype)).contiguous()

    data: dict[str, Any] = {"scores": scores}
    if mode != "basic":
        if trivial:
            lengths = torch.full((rows,), min(k, length), dtype=torch.int32, device=device)
        else:
            lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
        if row_starts:
            # Non-zero row starts shrink the usable span; keep k <= span.
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


def _source_module():
    """Raw FlashInfer topk FFI module (no allocating Python wrapper)."""
    from flashinfer.jit.topk import gen_topk_module

    return gen_topk_module().build_and_load()


def _pin_source_algo() -> None:
    """Force ``TopKDispatch`` onto the radix path this port targets."""
    os.environ[_SOURCE_ALGO_ENV] = _SOURCE_ALGO_VALUE


def _row_states_buffer():
    import torch

    return torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda")


def run_reference(config: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]) -> None:
    """One launch of the FlashInfer source kernel with preallocated outputs."""
    import torch

    _pin_source_algo()
    module = _source_module()
    mode = config["mode"]
    k = config["k"]
    det = config["deterministic"]
    row_states = outputs["row_states"]
    if mode == "basic":
        module.radix_topk(
            data["scores"],
            outputs["indices"],
            outputs["values"],
            row_states,
            k,
            False,  # sorted_output
            det,
            _TIE_BREAK_NONE,
            False,  # dsa_graph_safe
        )
    elif mode == "page_table":
        module.radix_topk_page_table_transform(
            data["scores"],
            outputs["indices"],
            data["src_page_table"],
            data.get("row_to_batch"),
            data["lengths"],
            row_states,
            k,
            det,
            _TIE_BREAK_NONE,
            False,
            data.get("row_starts"),
            data.get("page_table_row_starts"),
        )
    else:
        module.radix_topk_ragged_transform(
            data["scores"],
            outputs["indices"],
            data["offsets"],
            data["lengths"],
            row_states,
            k,
            det,
            _TIE_BREAK_NONE,
            False,
            data.get("row_starts"),
        )
    torch.cuda.synchronize()


def build_reference_launch(config: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]):
    """Resolve the module and bind every argument once; return a bare launch.

    The timed closure must enqueue exactly one kernel and nothing else -- no env
    writes, no module lookup, no host synchronize.
    """
    _pin_source_algo()
    module = _source_module()
    mode = config["mode"]
    k = config["k"]
    det = config["deterministic"]
    row_states = outputs["row_states"]
    scores = data["scores"]
    indices = outputs["indices"]
    if mode == "basic":
        values = outputs["values"]
        fn = module.radix_topk
        args = (scores, indices, values, row_states, k, False, det, _TIE_BREAK_NONE, False)
    elif mode == "page_table":
        fn = module.radix_topk_page_table_transform
        args = (
            scores,
            indices,
            data["src_page_table"],
            data.get("row_to_batch"),
            data["lengths"],
            row_states,
            k,
            det,
            _TIE_BREAK_NONE,
            False,
            data.get("row_starts"),
            data.get("page_table_row_starts"),
        )
    else:
        fn = module.radix_topk_ragged_transform
        args = (
            scores,
            indices,
            data["offsets"],
            data["lengths"],
            row_states,
            k,
            det,
            _TIE_BREAK_NONE,
            False,
            data.get("row_starts"),
        )
    return lambda: fn(*args)


def _alloc_outputs(config: dict[str, Any]):
    import torch

    rows = config["num_rows"]
    k = config["k"]
    return {
        "indices": torch.empty(rows, k, dtype=torch.int32, device="cuda"),
        "values": torch.empty(rows, k, dtype=_torch_dtype(config["dtype"]), device="cuda"),
        "row_states": _row_states_buffer(),
    }


def _assert_device_matches_compile_profile() -> None:
    import torch

    optin = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin
    expected = hardware_max_smem_optin()
    if optin != expected:
        raise AssertionError(
            f"device optin shared memory {optin} != compile profile {expected}; "
            "the single-CTA dispatch domain would differ from the configured matrix"
        )


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
    _assert_device_matches_compile_profile()
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
    ref_out = _alloc_outputs(cfg)
    run_reference(cfg, data, ref_out)

    assert_reference_is_top_k(cfg, data, ref_out)

    ex = compile_kernel(get_kernel(**cfg))
    tirx_out = _alloc_outputs(cfg)
    _launch_tirx(ex, cfg, data, tirx_out)
    torch.cuda.synchronize()

    _compare(cfg, data, ref_out, tirx_out)


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "dtype": "float32",
        "mode": "basic",
        "num_rows": 16,
        "length": 32000,
        "k": 256,
        "deterministic": False,
        "row_starts": False,
        "page_table_row_starts": False,
        "row_to_batch": False,
        "trivial": False,
    }
    cfg.update({key: value for key, value in config.items() if key != "label"})
    return cfg


def build_tirx_args(cfg: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]):
    """Materialize the flat launch ABI once, outside any timed region.

    Every tensor the kernel reads must exist before the launch closure is built:
    allocating placeholders inside the closure would launch extra fill kernels
    that a per-kernel timer attributes to this kernel.
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
        aux_stride,
    )


def _launch_tirx(ex, cfg: dict[str, Any], data: dict[str, Any], outputs: dict[str, Any]) -> None:
    ex(*build_tirx_args(cfg, data, outputs))


def _row_local_indices(cfg: dict[str, Any], data: dict[str, Any], out_indices):
    """Map one launch's raw output back to row-local score offsets.

    Basic returns the index directly; Ragged adds a per-row offset; PageTable
    returns an injective page id (``batch * length + slot``).  ``-1`` marks the
    padding the trivial branches write when a row is shorter than ``k``.
    """
    import torch

    rows = cfg["num_rows"]
    length = cfg["length"]
    idx = out_indices.to(torch.int64)
    pad = idx < 0
    if cfg["mode"] == "ragged":
        idx = idx - data["offsets"].to(torch.int64).unsqueeze(-1)
    elif cfg["mode"] == "page_table":
        row_to_batch = data.get("row_to_batch")
        if row_to_batch is None:
            batch = torch.arange(rows, dtype=torch.int64, device=idx.device)
        else:
            batch = row_to_batch.to(torch.int64)
        idx = idx - batch.unsqueeze(-1) * length
        pt_starts = data.get("page_table_row_starts")
        if pt_starts is None:
            pt_starts = data.get("row_starts")
        if pt_starts is not None:
            idx = idx - pt_starts.to(torch.int64).unsqueeze(-1)
    return idx, pad


def _selected_values(cfg: dict[str, Any], data: dict[str, Any], out_indices):
    """Sorted-descending scores of the selected elements, padding removed."""
    import torch

    idx, pad = _row_local_indices(cfg, data, out_indices)
    row_starts = data.get("row_starts")
    if row_starts is not None:
        idx = idx + row_starts.to(torch.int64).unsqueeze(-1)
    scores = data["scores"].to(torch.float32)
    safe = idx.clamp_(0, scores.size(1) - 1)
    vals = torch.gather(scores, 1, safe)
    # Padding slots carry no score; sort them to the bottom deterministically.
    vals = torch.where(pad, torch.full_like(vals, float("-inf")), vals)
    return torch.sort(vals, dim=-1, descending=True).values


def _compare(
    cfg: dict[str, Any], data: dict[str, Any], ref: dict[str, Any], got: dict[str, Any]
) -> None:
    """Compare a TIRx launch against the FlashInfer launch.

    Ties make the selected *index set* ambiguous, so the criterion is the
    multiset of selected score values, which every valid top-k shares.  For
    deterministic configs both sides run the same fixed-order collect, so the
    raw outputs must additionally match element for element.
    """
    import torch

    from tirx_kernels.flashinfer.utils.topk_harness import assert_valid_outputs

    assert_valid_outputs(cfg, data, ref)
    assert_valid_outputs(cfg, data, got)
    if cfg["deterministic"]:
        torch.testing.assert_close(got["indices"], ref["indices"], rtol=0, atol=0)
        if cfg["mode"] == "basic":
            torch.testing.assert_close(got["values"], ref["values"], rtol=0, atol=0)
        return

    ref_vals = _selected_values(cfg, data, ref["indices"])
    got_vals = _selected_values(cfg, data, got["indices"])
    torch.testing.assert_close(got_vals, ref_vals, rtol=0, atol=0)


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
        rows, length, k = cfg["num_rows"], cfg["length"], cfg["k"]
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
        del k
    expect = torch.topk(window, cfg["k"], dim=-1).values
    expect = torch.sort(expect, dim=-1, descending=True).values
    actual = _selected_values(cfg, data, ref["indices"])
    torch.testing.assert_close(actual, expect, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------
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
    tirx_out = _alloc_outputs(cfg)
    tirx_args = build_tirx_args(cfg, data, tirx_out)

    def tirx_launch():
        ex(*tirx_args)

    def build_reference():
        ref_out = _alloc_outputs(cfg)
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
