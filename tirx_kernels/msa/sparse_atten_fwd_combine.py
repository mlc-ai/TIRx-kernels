# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MSA sparse attention forward combine, the split-reduction half of the pair.

Ports ``SparseAttentionForwardCombine``: the K2 kernel that reduces the
per-split partials the forward kernel in this package writes. For every query
row it merges the split log-sum-exps, rescales and accumulates the split
outputs in fp32, and un-permutes the forward's STG.128 fake column order back
to real head-dim order on the way out.

The forward writes ``O_partial`` in a permuted column order so its epilogue can
store 128 bits at a time; this kernel is where that permutation is undone, so
the two kernels share a layout contract that neither one states alone.

Upstream source: python/fmha_sm100/cute/src/sm100/fwd/combine.py:37.
"""

import math
from functools import lru_cache
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "msa_sparse_atten_fwd_combine_sm100",
    "category": "msa",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "msa",
            "git": {
                "url": "https://github.com/MiniMax-AI/MSA.git",
                "commit": "80434d7f67877c6570ca19cac444b84bc9855dac",
            },
            "import": "fmha_sm100",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.5.3", "import": "cutlass"},
        {"package": "quack-kernels", "specifier": "==0.5.0", "import": "quack"},
    ),
}

# The host entry fixes every one of these for the production dispatch domain:
# `k_block_size = 128 if D > 64 else 64` and `tile_m = 64` (combine.py:1330),
# `stages=2` (:1374), `num_threads=256` (the constructor default).
HEAD_DIM = 128
K_BLOCK_SIZE = 128
TILE_M = 64
STAGES = 2
NUM_THREADS = 256
WARP_SIZE = 32
WARPS = NUM_THREADS // WARP_SIZE

# `sO_perm` pads each row by 16 output elements to break the bank-conflict
# pattern the permuted scatter would otherwise hit (combine.py:374-382).
PERM_ROW_PAD = 16

LOG_2_E = math.log2(math.e)
# `log(x)` under fastmath is `lg2(x) * ln(2)`, folded with the running
# maximum into one `fma.rn.f32` (combine.py:818).
LN_2 = math.log(2.0)

# `use_pdl=True` at both production call sites (interface.py:972, :1532), so the
# kernel always waits on the forward's launch-dependents signal and the launch
# always carries the programmatic-dependent-launch attribute. The partial
# staging buffer alone is 64 KB for fp32 partials, so shared memory is dynamic.
LAUNCH_TAGS = (
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
    "tirx.use_dyn_shared_memory",
)

_TORCH_DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float8_e4m3": "float8_e4m3fn",
}

# CUDA codegen has no fp8 scalar type and MSA does not want one: its fp8 tiles
# are moved as raw bytes and converted with PTX on the bit pattern. An fp8
# partial buffer therefore binds as `uint8`, exactly as the forward port and
# `flashmla/sparse_decode_head64.py` do.
_KERN_DTYPES = {"bfloat16": K.bf16, "float16": K.f16, "float32": K.f32, "float8_e4m3": K.u8}


def _torch_dtype(name: str):
    import torch

    return getattr(torch, _TORCH_DTYPES[name])


def fast_divmod(divisor: int) -> tuple[int, int, int]:
    """The `(multiplier, shift_1, shift_2)` triple CUTLASS's `FastDivmod` builds.

    The source divides the flat `(q, head)` row index by the head count through
    a host-built `FastDivmodDivisor`, and the export keeps that as a magic
    multiply rather than a division (`combine.py:463`, `mul.hi.u32` + two
    `shr.u32` + `add` + `mul.lo.s32` + `sub`). Passing the divisor alone and
    dividing in the kernel would put a `div.s32` in the hottest index path, so
    the port precomputes the same operands the reference's launcher does.

    The lowering the kernel then reproduces is the round-up form::

        q0   = mulhi(idx, multiplier)
        quot = (((idx - q0) >> shift_1) + q0) >> shift_2
        rem  = idx - quot * divisor
    """
    if divisor < 1:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if divisor == 1:
        return 1, 0, 0
    shift = max(divisor - 1, 1).bit_length()
    multiplier = ((1 << (32 + shift)) + divisor - 1) // divisor - (1 << 32)
    return multiplier, 1, shift - 1


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _partial_bytes(name: str) -> int:
    return {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3": 1}[name]


# ---------------------------------------------------------------------------
# Target entry.
#
# Optional buffers stay in the signature and are simply never read when their
# axis is off, which is what the kern sparse-decode combine does too
# (`flashmla/sparse_decode_head64.py:1958`). The device code is unchanged
# either way -- an unread pointer costs parameter space, not instructions --
# and it keeps one launch ABI across the presence axes.
#
# Extents and strides are explicit runtime scalars rather than shapes baked
# into the buffer declarations: a split-combine kernel that declared a
# multi-dimensional alias with a runtime split stride is exactly the case the
# codegen field notes record as `split_buffer_misaddress`.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def make_kernel(
    *,
    topk: int,
    partial_dtype: str = "float32",
    temperature: bool = False,
    output_scale: bool = False,
    seqused: bool = False,
):
    """Trace one specialization of the combine kernel.

    The axes here are the compile-cache key the host entry uses
    (combine.py:1339-1352) minus the values production pins: head dim 128, bf16
    output, varlen, split counts present, LSE out present, ``use_pdl`` true.
    """
    if (TILE_M * topk) % NUM_THREADS != 0:
        raise ValueError(f"topk={topk} violates can_implement: (tile_m*topk) % num_threads != 0")
    partial_ty = _KERN_DTYPES[partial_dtype]
    pbytes = _partial_bytes(partial_dtype)
    # 128-bit copies: 4 fp32, 8 bf16/fp16, 16 fp8 elements per transaction, so
    # the thread grid over a k-block row and the rows per thread follow the
    # partial dtype (combine.py:112-134).
    o_elems = 16 // pbytes
    o_threads_per_row = K_BLOCK_SIZE // o_elems
    o_rows = TILE_M // (NUM_THREADS // o_threads_per_row)
    # The row group is `tidx // o_threads_per_row`: the column threads are
    # contiguous, so the shift is log2 of the threads per row (5 for fp32,
    # 4 for bf16/fp16, 3 for fp8), not log2 of the rows per group.
    o_row_shift = o_threads_per_row.bit_length() - 1
    # The output store partitions independently: 8 bf16 per 128-bit store
    # (combine.py:136-163), which is why 7a and 7b use different maps.
    out_elems = 16 // 2
    out_rows = TILE_M // (NUM_THREADS // (K_BLOCK_SIZE // out_elems))
    splits_pt = topk // (NUM_THREADS // TILE_M)
    # Byte map of SharedStorage (combine.py:383-397). sLSETemperature occupies
    # its slot unconditionally; every later offset depends on that.
    off_slse, off_slse_t = 0, topk * TILE_M * 4
    off_maxvalid = off_slse_t + topk * TILE_M * 4
    off_so = off_maxvalid + _align_up(TILE_M * 4, 128)
    off_sperm = off_so + _align_up(STAGES * TILE_M * K_BLOCK_SIZE * pbytes, 128)
    smem_total = off_sperm + _align_up(TILE_M * (K_BLOCK_SIZE + PERM_ROW_PAD) * 2, 128)
    # `min_blocks_per_mp = 3 if has_output_scale and use_pdl else 0`
    # (combine.py:1337).
    min_blocks_per_sm = 3 if output_scale else None

    OFF_SLSE, OFF_SLSE_T = off_slse, off_slse_t
    O_ELEMS, O_ROW_SHIFT, O_COL_MASK = o_elems, o_row_shift, o_threads_per_row - 1
    OUT_ELEMS = out_elems
    O_ROW_STEP = NUM_THREADS // o_threads_per_row
    OUT_ROW_STEP = NUM_THREADS // (K_BLOCK_SIZE // out_elems)
    PERM_ROW = K_BLOCK_SIZE + PERM_ROW_PAD
    SMEM_TOTAL = smem_total
    NUM_ROWS, OUT_ROWS, SPLITS_PT = o_rows, out_rows, splits_pt
    NUM_VALS = o_elems

    @K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=min_blocks_per_sm, grid=False)
    def msa_sparse_atten_fwd_combine(
        o_partial: K.gptr[partial_ty],
        lse_partial: K.gptr[K.f32],
        o_out: K.gptr[K.bf16],
        lse_out: K.gptr[K.f32],
        lse_temperature_partial: K.gptr[K.f32],
        lse_temperature_out: K.gptr[K.f32],
        cu_seqlens: K.gptr[K.i32],
        seqused_q: K.gptr[K.i32],
        split_counts: K.gptr[K.i32],
        output_scale_ptr: K.gptr[K.f32],
        stride_op_split: K.i32,
        stride_op_q: K.i32,
        stride_op_h: K.i32,
        stride_lp_split: K.i32,
        stride_lp_q: K.i32,
        stride_o_q: K.i32,
        stride_o_h: K.i32,
        stride_l_q: K.i32,
        stride_sc_q: K.i32,
        qhead_per_kv: K.i32,
        total_q: K.i32,
        head_q: K.i32,
        num_batches: K.i32,
        head_div_mul: K.i32,
        head_div_s1: K.i32,
        head_div_s2: K.i32,
    ):
        # ===== MSA COMBINE TRANSCRIPTION START =====
        # Grid: (ceil(seqlen*num_head / tile_m), ceil(head_dim / k_block), batch)
        # with the head axis innermost inside the flattened row index
        # (combine.py:401-418).
        m_block, k_block, batch = K.cta_id(
            [(total_q * head_q + (TILE_M - 1)) // TILE_M, HEAD_DIM // K_BLOCK_SIZE, num_batches]
        )
        tidx = K.thread_id()

        # -------------------------------------------------------------------
        # Storage. One dynamic allocation carved into the source's five fields
        # at their exact byte offsets (combine.py:383-397). `sLSETemperature`
        # is allocated whether or not the temperature axis is on -- the source
        # declares it unconditionally, and every later offset depends on it.
        # -------------------------------------------------------------------
        pool = K.smem_pool()
        raw = pool.alloc((SMEM_TOTAL,), K.u8, align=1024)
        pool.commit(SMEM_TOTAL)

        def _view(shape, dtype, byte_offset):
            return K.decl_buffer(
                shape, dtype, data=raw.data, scope="shared.dyn", byte_offset=byte_offset, align=1024
            )

        s_lse = _view((topk * TILE_M,), K.f32, OFF_SLSE)
        s_maxvalid = _view((TILE_M,), K.i32, off_maxvalid)
        s_o = _view((STAGES * TILE_M * K_BLOCK_SIZE,), partial_ty, off_so)
        s_perm = _view((TILE_M * PERM_ROW,), K.bf16, off_sperm)
        s_perm_i32 = _view((TILE_M * PERM_ROW // 2,), K.u32, off_sperm)
        if temperature:
            s_lse_t = _view((topk * TILE_M,), K.f32, OFF_SLSE_T)

        # The thread decomposition the source's four tiled copies imply. These
        # are four distinct maps over the same 256 threads and they are never
        # interchanged: the staging pair walks (split, row) with order=(1, 0),
        # the read-back pair walks it with order=(0, 1), and the two epilogue
        # pairs differ again in row stride and vector width. Conflating any two
        # of them addresses real memory and fails only on some shapes.
        lse_row = K.local_scalar("int32", init=K.bitwise_and(tidx, 63))
        lse_split0 = K.local_scalar("int32", init=K.shift_right(tidx, 6))
        s2r_row = K.local_scalar("int32", init=K.shift_right(tidx, 2))
        s2r_split0 = K.local_scalar("int32", init=K.bitwise_and(tidx, 3))
        o_row0 = K.local_scalar("int32", init=K.shift_right(tidx, O_ROW_SHIFT))
        o_col = K.local_scalar("int32", init=K.bitwise_and(tidx, O_COL_MASK) * O_ELEMS)
        st_row0 = K.local_scalar("int32", init=K.shift_right(tidx, 4))
        st_col = K.local_scalar("int32", init=K.bitwise_and(tidx, 15) * OUT_ELEMS)

        def _ld_u32(buf, index):
            """``ld.global.u32``, the form the reference emits for every i32
            field it reads. The reinterpret is a bitcast and costs nothing."""
            raw = K.local_scalar("uint32")
            K.ptx.ld.global_.u32(raw, buf.ptr_to([index]))
            return K.reinterpret("int32", raw)

        # -------------------------------------------------------------------
        # Prologue: varlen bounds and the K1 dependency (combine.py:531-550).
        # -------------------------------------------------------------------
        q_offset = K.local_scalar("int32", init=_ld_u32(cu_seqlens, batch))
        seqlen = K.local_scalar("int32")
        if seqused:
            K.assign(seqlen, _ld_u32(seqused_q, batch))
        else:
            K.assign(seqlen, _ld_u32(cu_seqlens, batch + 1) - q_offset)
        max_idx = K.local_scalar("int32", init=seqlen * head_q)

        with K.If(m_block * TILE_M < max_idx):
            with K.Then():
                # The source opens this block at :547 and closes it at :1125:
                # the guard covers the WHOLE body, not just the wait. That is
                # what retires a tail CTA before it reaches any barrier, so
                # narrowing it to the wait would hang the remaining CTAs.
                K.ptx.griddepcontrol.wait()

                def _divmod_row(idx):
                    """`decode_flat_row_idx` (combine.py:457-464).

                    The host precomputes the magic operands, so this is the reference's
                    seven-instruction sequence -- mul.hi + sub + shr + add + shr for the
                    quotient, mul.lo + sub for the remainder -- and not a `div.s32`.
                    """
                    q0 = K.local_scalar("uint32")
                    K.ptx.mul.hi.u32(
                        q0, K.reinterpret("uint32", idx), K.reinterpret("uint32", head_div_mul)
                    )
                    t = K.local_scalar("uint32", init=K.reinterpret("uint32", idx) - q0)
                    t2 = K.local_scalar(
                        "uint32", init=K.shift_right(t, K.reinterpret("uint32", head_div_s1)) + q0
                    )
                    quot = K.local_scalar(
                        "int32",
                        init=K.reinterpret(
                            "int32", K.shift_right(t2, K.reinterpret("uint32", head_div_s2))
                        ),
                    )
                    rem = K.local_scalar("int32", init=idx - quot * head_q)
                    return quot, rem

                def _split_count(m_idx, head_idx, floor_div):
                    """`mSplitCounts[offset + m_idx, head_idx // qhead_per_kvhead]`.

                    Step 1's dividend is a fresh divmod remainder, known non-negative,
                    and the reference emits a bare `div.s32` there; step 2 reads its
                    dividend back from a register tensor, so the signed-floor fixup
                    survives (combine.py:642 vs :709).
                    """
                    group = (
                        K.local_scalar("int32", init=head_idx // qhead_per_kv)
                        if floor_div
                        else K.local_scalar("int32", init=K.truncdiv(head_idx, qhead_per_kv))
                    )
                    return _ld_u32(split_counts, (q_offset + m_idx) * stride_sc_q + group)

                NEG_INF = K.reinterpret("float32", K.uint32(0xFF800000))

                def _widen_words(dst, m, words):
                    """Widen a 128-bit staged partial to fp32 lanes.

                    bf16/fp16 carry 8 values per transaction and fp8 carries 16, so the
                    four staged words expand to NUM_VALS accumulator lanes. MSA moves
                    its fp8 tiles as raw bytes and converts through fp16, which is why
                    the fp8 path goes via a byte permute rather than a direct cvt
                    (combine.py:112-134 for the widths).
                    """
                    if partial_dtype in ("bfloat16", "float16"):
                        lo_op = "cvt.f32.bf16" if partial_dtype == "bfloat16" else "cvt.f32.f16"
                        for w in range(4):
                            for half in range(2):
                                half_bits = K.local_scalar("uint16")
                                K.ptx.mov.b16(
                                    half_bits,
                                    K.cast(
                                        K.bitwise_and(K.shift_right(words[w], 16 * half), 0xFFFF),
                                        "uint16",
                                    ),
                                )
                                K.ptx[lo_op](dst[m * NUM_VALS + 2 * w + half], half_bits)
                    else:
                        # float8_e4m3: `cvt.rn.f16x2.e4m3x2` takes a byte PAIR and
                        # yields two halves, so four staged words are 8 conversions,
                        # not 16 with the high half thrown away.
                        for w in range(4):
                            for pair in range(2):
                                bp = K.local_scalar("uint16")
                                K.ptx.mov.b16(
                                    bp,
                                    K.cast(
                                        K.bitwise_and(K.shift_right(words[w], 16 * pair), 0xFFFF),
                                        "uint16",
                                    ),
                                )
                                h2 = K.local_scalar("uint32")
                                K.ptx["cvt.rn.f16x2.e4m3x2"](h2, bp)
                                for half in range(2):
                                    hb = K.local_scalar("uint16")
                                    K.ptx.mov.b16(
                                        hb,
                                        K.cast(
                                            K.bitwise_and(K.shift_right(h2, 16 * half), 0xFFFF),
                                            "uint16",
                                        ),
                                    )
                                    K.ptx["cvt.f32.f16"](
                                        dst[m * NUM_VALS + 4 * w + 2 * pair + half], hb
                                    )

                def _load_stage_vec(dst, m, stage, row):
                    """`autovec_copy(tOsO_partial[..., stage], tOrO_partial)`
                    (combine.py:939): one 128-bit shared read per staged row, widened
                    to fp32 when the partial dtype is narrower."""
                    ptr = s_o.ptr_to([stage * (TILE_M * K_BLOCK_SIZE) + row * K_BLOCK_SIZE + o_col])
                    w = [K.local_scalar("uint32") for _ in range(4)]
                    K.ptx.ld.shared.v4.b32(w[0], w[1], w[2], w[3], ptr)
                    if partial_dtype == "float32":
                        for v in range(4):
                            K.assign(dst[m * NUM_VALS + v], K.reinterpret("float32", w[v]))
                    else:
                        _widen_words(dst, m, w)

                def _store_zero_vec(ptr):
                    """`tOsO_partial_cur[...].fill(0)` (combine.py:1161).

                    One 128-bit shared store of a splatted zero. The reference spells
                    it `st.shared.v4.f32` for fp32 partials and the same width for the
                    narrower ones, so the bit pattern is what varies, not the width.
                    """
                    zero = K.local_scalar("uint32", init=K.uint32(0))
                    K.ptx.st.shared.v4.b32(ptr, zero, zero, zero, zero)

                # -------------------------------------------------------------------
                # Step 1: stage LSE_partial, -inf past a row's split count
                # (combine.py:636-673). One row per thread; four split slots strided
                # by four, which land 1024 bytes apart in the swizzled buffer.
                # -------------------------------------------------------------------
                def _slse_base(split_base, row):
                    """The swizzled element index of `sLSE[split_base, row]`.

                    `make_swizzle(3, 2, 3)` composed over a `(min(topk,8), 64)` atom
                    (combine.py:208-222): an XOR on bits 2..4 of the row-major index,
                    which the export computes as
                    `(((tid >> 3) & 28) ^ (tid & 252)) | (tid & 3)` for the staging map.
                    """
                    flat = split_base * TILE_M + row
                    # `offset ^= ((offset >> 5) & 7) << 2`. The export spells this as
                    # `(flat & 252) ^ ((flat >> 3) & 28)` because its two uses have
                    # `flat < 256`; written that way it silently drops bits 8 and above,
                    # which is wrong for step 6, whose split index is absolute and
                    # reaches topk-1. The XOR only touches bits 2..4, so this form
                    # agrees with the export everywhere the export applies.
                    return K.bitwise_xor(
                        flat, K.shift_left(K.bitwise_and(K.shift_right(flat, 5), 7), 2)
                    )

                idx1 = K.local_scalar("int32", init=m_block * TILE_M + lse_row)
                with K.If(idx1 < max_idx):
                    with K.Then():
                        m_idx1, head_idx1 = _divmod_row(idx1)
                        row_count = K.local_scalar(
                            "int32", init=_split_count(m_idx1, head_idx1, False)
                        )
                        lse_row_base = K.local_scalar(
                            "int64",
                            init=K.cast((q_offset + m_idx1), "int64") * K.cast(stride_lp_q, "int64")
                            + K.cast(head_idx1, "int64"),
                        )
                        for i in range(SPLITS_PT):
                            si = lse_split0 + 4 * i
                            dst = s_lse.ptr_to([_slse_base(lse_split0, lse_row) + i * (4 * TILE_M)])
                            with K.If(si < row_count):
                                with K.Then():
                                    K.ptx["cp.async.ca.shared.global"](
                                        dst,
                                        lse_partial.ptr_to(
                                            [
                                                lse_row_base
                                                + K.cast(si, "int64")
                                                * K.cast(stride_lp_split, "int64")
                                            ]
                                        ),
                                        4,
                                        4,
                                    )
                                    if temperature:
                                        K.ptx["cp.async.ca.shared.global"](
                                            s_lse_t.ptr_to(
                                                [_slse_base(lse_split0, lse_row) + i * (4 * TILE_M)]
                                            ),
                                            lse_temperature_partial.ptr_to(
                                                [
                                                    lse_row_base
                                                    + K.cast(si, "int64")
                                                    * K.cast(stride_lp_split, "int64")
                                                ]
                                            ),
                                            4,
                                            4,
                                        )
                                with K.Else():
                                    K.ptx.st.shared.u32(dst, K.uint32(0xFF800000))
                                    if temperature:
                                        K.ptx.st.shared.u32(
                                            s_lse_t.ptr_to(
                                                [_slse_base(lse_split0, lse_row) + i * (4 * TILE_M)]
                                            ),
                                            K.uint32(0xFF800000),
                                        )
                    with K.Else():
                        for i in range(SPLITS_PT):
                            K.ptx.st.shared.u32(
                                s_lse.ptr_to([_slse_base(lse_split0, lse_row) + i * (4 * TILE_M)]),
                                K.uint32(0xFF800000),
                            )
                            if temperature:
                                K.ptx.st.shared.u32(
                                    s_lse_t.ptr_to(
                                        [_slse_base(lse_split0, lse_row) + i * (4 * TILE_M)]
                                    ),
                                    K.uint32(0xFF800000),
                                )
                K.ptx.cp.async_.commit_group()

                # -------------------------------------------------------------------
                # Step 2: per-row indices, split counts and raw GMEM element pointers,
                # then prime the partial pipeline stages-1 deep (combine.py:679-739).
                # The out-of-range rows take a real predicated branch per row, with
                # the -1/0/0 defaults materialized first.
                # -------------------------------------------------------------------
                hidx = K.alloc_local((NUM_ROWS,), "int32")
                midx = K.alloc_local((NUM_ROWS,), "int32")
                scount = K.alloc_local((NUM_ROWS,), "int32")
                rowptr = K.alloc_local((NUM_ROWS,), "int64")

                for m in range(NUM_ROWS):
                    K.assign(hidx[m], K.int32(-1))
                    K.assign(midx[m], K.int32(0))
                    K.assign(scount[m], K.int32(0))
                    K.assign(rowptr[m], K.int64(0))
                    idx2 = K.local_scalar("int32", init=m_block * TILE_M + o_row0 + m * O_ROW_STEP)
                    with K.If(idx2 < max_idx):
                        with K.Then():
                            q2, h2 = _divmod_row(idx2)
                            K.assign(midx[m], q2)
                            K.assign(hidx[m], h2)
                            K.assign(scount[m], _split_count(q2, h2, True))
                            # `(q_offset + m_idx)*head_q*D + h*D` is identically
                            # `q_offset*head_q*D + idx*D`, because idx is exactly
                            # `m_idx*head_q + h` -- the divmod result is multiplied
                            # straight back out. Forming the address from idx keeps
                            # the seven-instruction divmod off the chain that gates
                            # this row's cp.async issue; the decomposition is still
                            # computed, because split_counts and the output store
                            # genuinely need m_idx and head_idx.
                            K.assign(
                                rowptr[m],
                                K.cast(q_offset, "int64") * K.cast(stride_op_q, "int64")
                                + K.cast(idx2, "int64") * K.cast(HEAD_DIM, "int64")
                                + K.cast(k_block * K_BLOCK_SIZE, "int64"),
                            )

                def _load_o_partial(split, stage):
                    """One pipeline stage (combine.py:1141-1161).

                    A live slot issues the 16-byte `cp.async.cg`; a dead one stores a
                    zero vector instead. The zero fill is what makes a ragged split
                    count safe, and it is a separate store in the reference -- not
                    cp.async's own ignore-src form.
                    """
                    for m in range(NUM_ROWS):
                        dst = s_o.ptr_to(
                            [
                                stage * (TILE_M * K_BLOCK_SIZE)
                                + (o_row0 + m * O_ROW_STEP) * K_BLOCK_SIZE
                                + o_col
                            ]
                        )
                        # The reference nests these: `if tOhidx[m] >= 0` wraps BOTH
                        # arms (combine.py:1143-1161), so a dead row gets neither the
                        # copy nor the fill. Collapsing them into a single `and`
                        # would make a dead row write 16 zero bytes the reference
                        # never writes.
                        with K.If(hidx[m] >= 0):
                            with K.Then():
                                with K.If(split < scount[m]):
                                    with K.Then():
                                        K.ptx["cp.async.cg.shared.global"](
                                            dst,
                                            o_partial.ptr_to(
                                                [
                                                    rowptr[m]
                                                    + K.cast(split, "int64")
                                                    * K.cast(stride_op_split, "int64")
                                                    + K.cast(o_col, "int64")
                                                ]
                                            ),
                                            16,
                                            16,
                                        )
                                    with K.Else():
                                        _store_zero_vec(dst)

                for stage in range(STAGES - 1):
                    _load_o_partial(K.int32(stage), stage)
                    K.ptx.cp.async_.commit_group()

                # -------------------------------------------------------------------
                # Step 3: publish the staged LSE and read it back on the transposed
                # map (combine.py:746-757). The barrier is required because step 1
                # partitions sLSE by (split, row) and this reads it by (row, split).
                # -------------------------------------------------------------------
                K.ptx.cp.async_.wait_group(STAGES - 1)
                K.ptx.bar.sync(K.uint32(0))

                lse_reg = K.alloc_local((SPLITS_PT,), "float32")
                for i in range(SPLITS_PT):
                    K.ptx.ld.shared.f32(
                        lse_reg[i],
                        s_lse.ptr_to([_slse_base(s2r_split0, s2r_row) + i * (4 * TILE_M)]),
                    )

                # -------------------------------------------------------------------
                # Step 4: the split reduction (combine.py:779-873). Every reduction is
                # over the four lanes t..t^3 that share a row, so each is two butterfly
                # shuffles -- laneMask 2 then 1.
                # -------------------------------------------------------------------
                def _shfl_bfly_f32(value, lane_mask):
                    out = K.local_scalar("uint32")
                    K.ptx.shfl_sync.bfly.b32(
                        out,
                        K.reinterpret("uint32", value),
                        K.uint32(lane_mask),
                        K.uint32(31),
                        K.uint32(0xFFFFFFFF),
                    )
                    return K.reinterpret("float32", out)

                def _shfl_bfly_i32(value, lane_mask):
                    out = K.local_scalar("uint32")
                    K.ptx.shfl_sync.bfly.b32(
                        out,
                        K.reinterpret("uint32", value),
                        K.uint32(lane_mask),
                        K.uint32(31),
                        K.uint32(0xFFFFFFFF),
                    )
                    return K.reinterpret("int32", out)

                # The tree over the thread's four register values is NaN-propagating.
                def _max_tree(vals):
                    """NaN-propagating pairwise max over the thread's register slots.

                    `ts2rrLSE[...].reduce(ReductionOp.MAX)` (combine.py:781-786)
                    reduces ALL of mode[1], which is `splits_pt` slots -- 1, 2, 4 or
                    8 as topk moves 4, 8, 16, 32. A tree written for four would
                    silently drop the upper half at topk 32, and only a shape with
                    more than 16 live splits would ever show it.
                    """
                    cur = list(vals)
                    while len(cur) > 1:
                        half = len(cur) // 2
                        nxt = []
                        for a in range(half):
                            out = K.local_scalar("float32")
                            K.ptx["max.NaN.f32"](out, cur[a], cur[a + half])
                            nxt.append(out)
                        if len(cur) % 2:
                            nxt.append(cur[-1])
                        cur = nxt
                    return cur[0]

                acc_max = _max_tree([lse_reg[i] for i in range(SPLITS_PT)])
                other2 = _shfl_bfly_f32(acc_max, 2)
                le2 = K.local_scalar("uint32")
                K.ptx.setp.le.f32(le2, acc_max, other2)
                nan2 = K.local_scalar("uint32")
                K.ptx.setp.nan.f32(nan2, other2, other2)
                pick = K.local_scalar("float32")
                K.ptx.selp.f32(pick, other2, acc_max, K.ptx.pred(le2))
                step2v = K.local_scalar("float32")
                K.ptx.selp.f32(step2v, other2, pick, K.ptx.pred(nan2))
                other1 = _shfl_bfly_f32(step2v, 1)
                lse_max = K.local_scalar("float32")
                K.ptx["max.f32"](lse_max, step2v, other1)

                # The index of the last live split, reduced through the same butterfly
                # shape but as a plain signed max.
                cand = K.local_scalar("int32", init=K.int32(-1))
                for i in range(SPLITS_PT):
                    live = K.local_scalar("uint32")
                    K.ptx.setp.neu.f32(live, lse_reg[i], NEG_INF)
                    coord = s2r_split0 + 4 * i if i else s2r_split0
                    sel = K.local_scalar("int32")
                    K.ptx.selp.b32(sel, coord, cand, K.ptx.pred(live))
                    K.assign(cand, sel)
                mv1 = _shfl_bfly_i32(cand, 2)
                mvt = K.local_scalar("int32")
                K.ptx["max.s32"](mvt, cand, mv1)
                mv2 = _shfl_bfly_i32(mvt, 2 - 1)
                max_valid = K.local_scalar("int32")
                K.ptx["max.s32"](max_valid, mvt, mv2)

                # exp2 scales. `lse_max * LOG2_E` is hoisted above the -inf select, so
                # the site carries five multiplies and not four.
                is_ninf = K.local_scalar("uint32")
                K.ptx.setp.eq.f32(is_ninf, lse_max, NEG_INF)
                scaled_max = K.local_scalar("float32")
                K.ptx["mul.f32"](scaled_max, lse_max, K.float32(LOG_2_E))
                max_term = K.local_scalar("float32")
                K.ptx.selp.f32(max_term, K.float32(0.0), scaled_max, K.ptx.pred(is_ninf))

                lse_sum = K.local_scalar("float32", init=K.float32(0.0))
                for i in range(SPLITS_PT):
                    scaled = K.local_scalar("float32")
                    K.ptx["mul.f32"](scaled, lse_reg[i], K.float32(LOG_2_E))
                    arg = K.local_scalar("float32")
                    K.ptx["sub.f32"](arg, scaled, max_term)
                    K.ptx.ex2.approx.ftz.f32(lse_reg[i], arg)
                    nxt = K.local_scalar("float32")
                    K.ptx["add.f32"](nxt, lse_sum, lse_reg[i])
                    K.assign(lse_sum, nxt)
                sum1 = _shfl_bfly_f32(lse_sum, 2)
                sumt = K.local_scalar("float32")
                K.ptx["add.f32"](sumt, lse_sum, sum1)
                sum2 = _shfl_bfly_f32(sumt, 1)
                lse_sum_all = K.local_scalar("float32")
                K.ptx["add.f32"](lse_sum_all, sumt, sum2)

                # The source's three conditions become two compares: `sum == 0.0` and
                # `sum != sum` fuse into one unordered-equal.
                final_lse = K.local_scalar("float32", init=NEG_INF)
                inv_sum = K.local_scalar("float32", init=K.float32(0.0))
                bad_idx = K.local_scalar("uint32")
                K.ptx.setp.lt.s32(bad_idx, max_valid, K.int32(0))
                bad_sum = K.local_scalar("uint32")
                K.ptx.setp.equ.f32(bad_sum, lse_sum_all, K.float32(0.0))
                degenerate = K.local_scalar("uint32")
                K.ptx.or_.pred(degenerate, K.ptx.pred(bad_idx), K.ptx.pred(bad_sum))
                with K.If(degenerate == K.uint32(0)):
                    with K.Then():
                        lg = K.local_scalar("float32")
                        K.ptx.lg2.approx.ftz.f32(lg, lse_sum_all)
                        K.ptx["fma.rn.f32"](final_lse, lg, K.float32(LN_2), lse_max)
                        K.ptx["rcp.rn.f32"](inv_sum, lse_sum_all)

                if temperature:
                    t_reg = K.alloc_local((SPLITS_PT,), "float32")
                    for i in range(SPLITS_PT):
                        K.ptx.ld.shared.f32(
                            t_reg[i],
                            s_lse_t.ptr_to([_slse_base(s2r_split0, s2r_row) + i * (4 * TILE_M)]),
                        )
                    t_max = _max_tree([t_reg[i] for i in range(SPLITS_PT)])
                    to2 = _shfl_bfly_f32(t_max, 2)
                    tle = K.local_scalar("uint32")
                    K.ptx.setp.le.f32(tle, t_max, to2)
                    tnan = K.local_scalar("uint32")
                    K.ptx.setp.nan.f32(tnan, to2, to2)
                    tpick = K.local_scalar("float32")
                    K.ptx.selp.f32(tpick, to2, t_max, K.ptx.pred(tle))
                    tstep = K.local_scalar("float32")
                    K.ptx.selp.f32(tstep, to2, tpick, K.ptx.pred(tnan))
                    to1 = _shfl_bfly_f32(tstep, 1)
                    t_max_all = K.local_scalar("float32")
                    K.ptx["max.f32"](t_max_all, tstep, to1)

                    t_ninf = K.local_scalar("uint32")
                    K.ptx.setp.eq.f32(t_ninf, t_max_all, NEG_INF)
                    t_scaled = K.local_scalar("float32")
                    K.ptx["mul.f32"](t_scaled, t_max_all, K.float32(LOG_2_E))
                    t_term = K.local_scalar("float32")
                    K.ptx.selp.f32(t_term, K.float32(0.0), t_scaled, K.ptx.pred(t_ninf))
                    t_sum = K.local_scalar("float32", init=K.float32(0.0))
                    for i in range(SPLITS_PT):
                        sc = K.local_scalar("float32")
                        K.ptx["mul.f32"](sc, t_reg[i], K.float32(LOG_2_E))
                        ar = K.local_scalar("float32")
                        K.ptx["sub.f32"](ar, sc, t_term)
                        ev = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(ev, ar)
                        nx = K.local_scalar("float32")
                        K.ptx["add.f32"](nx, t_sum, ev)
                        K.assign(t_sum, nx)
                    ts1 = _shfl_bfly_f32(t_sum, 2)
                    tst = K.local_scalar("float32")
                    K.ptx["add.f32"](tst, t_sum, ts1)
                    ts2 = _shfl_bfly_f32(tst, 1)
                    t_sum_all = K.local_scalar("float32")
                    K.ptx["add.f32"](t_sum_all, tst, ts2)

                    final_lse_t = K.local_scalar("float32", init=NEG_INF)
                    t_bad = K.local_scalar("uint32")
                    K.ptx.setp.equ.f32(t_bad, t_sum_all, K.float32(0.0))
                    t_degen = K.local_scalar("uint32")
                    K.ptx.or_.pred(t_degen, K.ptx.pred(bad_idx), K.ptx.pred(t_bad))
                    with K.If(t_degen == K.uint32(0)):
                        with K.Then():
                            tlg = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(tlg, t_sum_all)
                            K.ptx["fma.rn.f32"](final_lse_t, tlg, K.float32(LN_2), t_max_all)

                for i in range(SPLITS_PT):
                    K.ptx["mul.f32"](lse_reg[i], lse_reg[i], inv_sum)
                # The scales overwrite the staged log-sum-exps in place; step 6 reads
                # them back on the O-partial map, which is why :907 is a barrier.
                for i in range(SPLITS_PT):
                    K.ptx.st.shared.f32(
                        s_lse.ptr_to([_slse_base(s2r_split0, s2r_row) + i * (4 * TILE_M)]),
                        lse_reg[i],
                    )
                with K.If(s2r_split0 == 0):
                    with K.Then():
                        K.ptx.st.shared.u32(
                            s_maxvalid.ptr_to([s2r_row]), K.reinterpret("uint32", max_valid)
                        )

                # -------------------------------------------------------------------
                # Step 5: the authoritative LSE_out store (combine.py:880-898). The two
                # zero-tests are OR-fused into one compare, as at :815.
                # -------------------------------------------------------------------
                with K.If(K.bitwise_or(s2r_split0, k_block) == 0):
                    with K.Then():
                        idx5 = K.local_scalar("int32", init=m_block * TILE_M + s2r_row)
                        with K.If(idx5 < max_idx):
                            with K.Then():
                                # `(q_offset + m_idx)*head_q + h` is identically
                                # `q_offset*head_q + idx`, and these stores are the
                                # only consumers of the decomposition here, so the
                                # divmod drops out entirely.
                                K.ptx.st.global_.f32(
                                    lse_out.ptr_to([q_offset * stride_l_q + idx5]), final_lse
                                )
                                if temperature:
                                    K.ptx.st.global_.f32(
                                        lse_temperature_out.ptr_to([q_offset * stride_l_q + idx5]),
                                        final_lse_t,
                                    )

                # -------------------------------------------------------------------
                # Step 6: the accumulation loop over live splits (combine.py:907-953).
                # The barrier publishes both step-4 writes: the scales, which this
                # thread reads on a different row map, and sMaxValidSplit.
                # -------------------------------------------------------------------
                K.ptx.bar.sync(K.uint32(0))

                thr_max_valid = K.local_scalar("int32")
                first_mv = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(first_mv, s_maxvalid.ptr_to([o_row0]))
                K.assign(thr_max_valid, K.reinterpret("int32", first_mv))
                for m in range(1, NUM_ROWS):
                    mv = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(mv, s_maxvalid.ptr_to([o_row0 + m * O_ROW_STEP]))
                    nxt = K.local_scalar("int32")
                    K.ptx["max.s32"](nxt, thr_max_valid, K.reinterpret("int32", mv))
                    K.assign(thr_max_valid, nxt)

                # The accumulator is zeroed in the preheader, ahead of the zero-trip
                # test, so a thread with no live split still carries zeros into step 7.
                acc = K.alloc_local((NUM_ROWS * NUM_VALS,), "float32")
                for m in range(NUM_ROWS):
                    for v in range(NUM_VALS):
                        K.assign(acc[m * NUM_VALS + v], K.float32(0.0))

                stage_load = K.local_scalar("int32", init=K.int32(STAGES - 1))
                stage_compute = K.local_scalar("int32", init=K.int32(0))
                scale = K.alloc_local((NUM_ROWS,), "float32")
                part = K.alloc_local((NUM_ROWS * NUM_VALS,), "float32")

                with K.serial(thr_max_valid + 1) as sp:
                    # Issue the next stage BEFORE the shared scale reads. The two
                    # are independent -- the copy is indexed by the ring cursor and
                    # the scales by the absolute split -- so putting the global
                    # transfers in flight first lets them overlap the shared-load
                    # latency instead of queueing behind it.
                    with K.If(sp + (STAGES - 1) <= thr_max_valid):
                        with K.Then():
                            _load_o_partial(sp + (STAGES - 1), stage_load)
                    K.ptx.cp.async_.commit_group()
                    for m in range(NUM_ROWS):
                        K.ptx.ld.shared.f32(
                            scale[m], s_lse.ptr_to([_slse_base(sp, o_row0 + m * O_ROW_STEP)])
                        )
                    # Advance the ring cursor branchlessly: STAGES is a power of two
                    # and the cursor is non-negative, so `(x + 1) & (STAGES-1)`
                    # is a single instruction (combine.py:933, :940).
                    # A signed truncmod needs sign correction and a select emits
                    # a branch; both put a chain on the cursor that gates the
                    # next iteration's shared addresses, and the branch costs
                    # most on short loops (topk=4 runs at most four iterations).
                    K.assign(stage_load, K.bitwise_and(stage_load + 1, K.int32(STAGES - 1)))
                    K.ptx.cp.async_.wait_group(STAGES - 1)
                    for m in range(NUM_ROWS):
                        _load_stage_vec(part, m, stage_compute, o_row0 + m * O_ROW_STEP)
                    # Advance the ring cursor branchlessly: STAGES is a power of two
                    # and the cursor is non-negative, so `(x + 1) & (STAGES-1)`
                    # is a single instruction (combine.py:933, :940).
                    # A signed truncmod needs sign correction and a select emits
                    # a branch; both put a chain on the cursor that gates the
                    # next iteration's shared addresses, and the branch costs
                    # most on short loops (topk=4 runs at most four iterations).
                    K.assign(stage_compute, K.bitwise_and(stage_compute + 1, K.int32(STAGES - 1)))
                    for m in range(NUM_ROWS):
                        # `setp.leu.f32` is unordered, so a NaN scale skips the row too;
                        # the `hidx >= 0` half reuses the predicate step 2 computed.
                        with K.If(K.And(hidx[m] >= 0, scale[m] > K.float32(0.0))):
                            with K.Then():
                                for v in range(NUM_VALS):
                                    K.ptx["fma.rn.f32"](
                                        acc[m * NUM_VALS + v],
                                        part[m * NUM_VALS + v],
                                        scale[m],
                                        acc[m * NUM_VALS + v],
                                    )

                K.ptx.cp.async_.wait_group(0)
                K.ptx.bar.sync(K.uint32(0))

                # -------------------------------------------------------------------
                # Step 7a: scatter the accumulator into sO_perm in REAL column order
                # (combine.py:1011-1049). This inverts the permutation K1's STG.128
                # epilogue applied. Adjacent fake columns map to adjacent real ones, so
                # a converted bf16 pair is one 32-bit shared store.
                # -------------------------------------------------------------------
                if output_scale:
                    out_scale = K.local_scalar("float32")
                    K.ptx.ld.global_.f32(out_scale, output_scale_ptr.ptr_to([0]))
                    # The packed operand is loop-invariant; the reference
                    # broadcasts it once (scale_fp32p_t8 PTX:1578), not once
                    # per pair.
                    scale_pair = K.local_scalar("uint64")
                    K.ptx.mov.b64(scale_pair, out_scale, out_scale)

                def _fake_to_real(fake_col):
                    """Invert K1's STG.128 column permutation (copy_utils.py:861-906).

                    Three maps, selected by the PARTIAL dtype -- K1 chose the fake order
                    to get 128-bit stores out of its TMEM fragment, and the fragment
                    width is what differs. All three are the same shape: split off a
                    block, then swap a lane index with a group index inside it.
                    """
                    if partial_dtype in ("bfloat16", "float16"):
                        block, lane_w = 32, 8  # stg128_half_fake_col_to_real_col (:878)
                    elif partial_dtype == "float8_e4m3":
                        block, lane_w = 64, 16  # stg128_fp8_fake_col_to_real_col (:897)
                    else:
                        block, lane_w = 16, 4  # stg128_fake_col_to_real_col (:861)
                    nt = K.bitwise_and(fake_col, -block)
                    inner = K.bitwise_and(fake_col, block - 1)
                    lane = K.truncdiv(inner, lane_w)
                    slot = K.bitwise_and(inner, lane_w - 1)
                    group = K.shift_right(slot, 1)
                    elem = K.bitwise_and(slot, 1)
                    return nt + group * 8 + lane * 2 + elem

                for m in range(NUM_ROWS):
                    with K.If(hidx[m] >= 0):
                        with K.Then():
                            for vp in range(NUM_VALS // 2):
                                v = 2 * vp
                                # real_col for the pair; the two halves differ only in
                                # bit 0, which is what makes them one 32-bit word.
                                real_col = _fake_to_real(o_col + v)
                                hi = acc[m * NUM_VALS + v + 1]
                                lo = acc[m * NUM_VALS + v]
                                if output_scale:
                                    # `cute.arch.mul_packed_f32x2` (combine.py:1030):
                                    # ONE packed multiply per pair. A packed f32x2
                                    # operand is a single 64-bit value, so the pair
                                    # is packed with mov.b64 and unpacked after.
                                    pk = K.local_scalar("uint64")
                                    K.ptx.mov.b64(pk, lo, hi)
                                    K.ptx.mul.f32x2(pk, pk, scale_pair)
                                    scaled_hi = K.local_scalar("float32")
                                    scaled_lo = K.local_scalar("float32")
                                    K.ptx.mov.b64(scaled_lo, scaled_hi, pk)
                                    hi, lo = scaled_hi, scaled_lo
                                word = K.local_scalar("uint32")
                                K.ptx["cvt.rn.bf16x2.f32"](word, hi, lo)
                                K.ptx.st.shared.u32(
                                    s_perm_i32.ptr_to(
                                        [
                                            (o_row0 + m * O_ROW_STEP) * (PERM_ROW // 2)
                                            + K.truncdiv(real_col, 2)
                                        ]
                                    ),
                                    word,
                                )

                # The re-partition point: 7a wrote sO_perm on the O-partial map, 7b
                # reads it on the output map, so every thread reads other threads' words.
                K.ptx.bar.sync(K.uint32(0))

                # -------------------------------------------------------------------
                # Step 7b: sO_perm -> registers -> GMEM, 128 bits at a time
                # (combine.py:1108-1125).
                # -------------------------------------------------------------------
                for m in range(OUT_ROWS):
                    row_out = st_row0 + m * OUT_ROW_STEP
                    w = [K.local_scalar("uint32") for _ in range(4)]
                    K.ptx.ld.shared.v4.b32(
                        w[0],
                        w[1],
                        w[2],
                        w[3],
                        s_perm_i32.ptr_to([row_out * (PERM_ROW // 2) + K.truncdiv(st_col, 2)]),
                    )
                    idx7 = K.local_scalar("int32", init=m_block * TILE_M + row_out)
                    with K.If(idx7 < max_idx):
                        with K.Then():
                            # Same identity as the partial address and the LSE
                            # store: the decomposition is multiplied straight back
                            # out and has no other consumer here, so four divmods
                            # per thread leave the store path.
                            K.ptx.st.global_.v4.b32(
                                o_out.ptr_to(
                                    [
                                        q_offset * stride_o_q
                                        + idx7 * HEAD_DIM
                                        + k_block * K_BLOCK_SIZE
                                        + st_col
                                    ]
                                ),
                                w[0],
                                w[1],
                                w[2],
                                w[3],
                            )
        # ===== MSA COMBINE TRANSCRIPTION END =====

    return msa_sparse_atten_fwd_combine


def get_kernel(**config):
    """Return the TIRx specialization selected by one config."""
    config.pop("label", None)
    kernel = make_kernel(
        topk=int(config["topk"]),
        partial_dtype=str(config.get("partial_dtype", "float32")),
        temperature=bool(config.get("temperature", False)),
        output_scale=bool(config.get("output_scale", False)),
        seqused=bool(config.get("seqused", False)),
    )
    return kernel.func.with_attr("global_symbol", KERNEL_META["name"]).with_attr(
        "tirx.kernel_launch_params", list(LAUNCH_TAGS)
    )


# ---------------------------------------------------------------------------
# Config matrix.
#
# The shapes mirror the forward port's benchmark rows, so a combine row and the
# forward row that produces its input describe the same production launch. The
# presence axes are the ones the two host call sites actually reach: the
# standard path passes temperature partials and no output scale
# (interface.py:1532), the NVFP4 path adds `output_scale=v_global_scale`
# (interface.py:972), and neither passes `seqused`. Partial dtype is the axis
# with genuinely distinct device code: it selects one of three fake-column
# maps and the width of both the staging copy and the epilogue store.
# ---------------------------------------------------------------------------
def _case(
    *,
    batch: int,
    seqlen_q: int,
    head_kv: int,
    qhead_per_kv: int,
    topk: int,
    label: str,
    partial_dtype: str = "float32",
    temperature: bool = False,
    output_scale: bool = False,
    seqused: bool = False,
    seqlen_k: int | None = None,
    seqlen_pattern: str = "uniform",
    blk_kv: int = 128,
) -> dict:
    return {
        "label": label,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_q if seqlen_k is None else seqlen_k,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "topk": topk,
        "partial_dtype": partial_dtype,
        "temperature": temperature,
        "output_scale": output_scale,
        "seqused": seqused,
        "seqlen_pattern": seqlen_pattern,
        "blk_kv": blk_kv,
    }


BENCH_CONFIGS = [
    # MSA's own ring-attention benchmark shape, carried over from the forward
    # port; the `sparse_fmha_adapter` layer pins fp32 partials on this path.
    _case(
        label="ring48k_fp32p_qh16_t16", batch=1, seqlen_q=49152, head_kv=1, qhead_per_kv=16, topk=16
    ),
    # Ulysses-style sequence parallelism: the longest row, with topk lowered so
    # the partials still fit, exactly as the forward port sizes it.
    _case(
        label="ulysses384k_fp32p_qh2_t4",
        batch=1,
        seqlen_q=393216,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
    ),
    _case(
        label="long96k_fp32p_qh2_t16", batch=1, seqlen_q=98304, head_kv=1, qhead_per_kv=2, topk=16
    ),
    _case(
        label="varlen_b3_s8192_fp32p_qh4_t16",
        batch=3,
        seqlen_q=8192,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        seqlen_pattern="varlen",
    ),
    # The forward's fp8 row emits bf16 partials and a temperature LSE, so this
    # is the half fake-column map, the packed `f16x2` staging store, and the
    # second reduction, all on the marquee geometry.
    _case(
        label="ring48k_bf16p_temp_qh16_t16",
        batch=1,
        seqlen_q=49152,
        head_kv=1,
        qhead_per_kv=16,
        topk=16,
        partial_dtype="bfloat16",
        temperature=True,
    ),
    _case(
        label="fp8p_s16384_qh8_t8",
        batch=1,
        seqlen_q=16384,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        partial_dtype="float8_e4m3",
    ),
    # The NVFP4 production combine: V's tensor scale is deferred to this kernel
    # and it is the only configuration that raises `min_blocks_per_mp` to 3.
    _case(
        label="scale_bf16p_temp_s16384_qh4_t16",
        batch=1,
        seqlen_q=16384,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        partial_dtype="bfloat16",
        temperature=True,
        output_scale=True,
    ),
    # The scale applied on the fp32 partial path, where the epilogue multiplies
    # scalars instead of packed pairs.
    _case(
        label="scale_fp32p_s8192_qh8_t8",
        batch=1,
        seqlen_q=8192,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        output_scale=True,
    ),
    _case(
        label="edge_b1_s1024_fp32p_qh2_t4",
        batch=1,
        seqlen_q=1024,
        head_kv=1,
        qhead_per_kv=2,
        topk=4,
    ),
]

# Correctness adds the axes production never benchmarks: fp16 partials (a
# dispatch-legal dtype upstream has no benchmark for), `seqused`, the largest
# legal split count, and the smallest one on a ragged multi-batch shape.
CONFIGS = [
    *BENCH_CONFIGS,
    _case(
        label="corr_fp16p_s4096_qh8_t8",
        batch=1,
        seqlen_q=4096,
        head_kv=2,
        qhead_per_kv=8,
        topk=8,
        partial_dtype="float16",
    ),
    _case(
        label="corr_seqused_s4096_qh4_t16",
        batch=2,
        seqlen_q=4096,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        seqused=True,
        seqlen_pattern="varlen",
    ),
    _case(
        label="corr_topk32_s2048_qh16_t32",
        batch=1,
        seqlen_q=2048,
        head_kv=1,
        qhead_per_kv=16,
        topk=32,
        temperature=True,
    ),
    # The topk=32 row above has seqlen_k 2048, i.e. 16 KV blocks, so a query can
    # never carry more than 16 live splits and the upper half of each thread's
    # register slots stays -inf. This shape has 64 blocks, so it is the only
    # config that exercises `SPLITS_PT == 8` end to end.
    _case(
        label="corr_topk32_s8192_qh4_t32",
        batch=1,
        seqlen_q=8192,
        head_kv=2,
        qhead_per_kv=4,
        topk=32,
    ),
    _case(
        label="corr_tiny_b2_s512_qh1_t4",
        batch=2,
        seqlen_q=512,
        head_kv=1,
        qhead_per_kv=1,
        topk=4,
        seqlen_pattern="varlen",
    ),
]


# ---------------------------------------------------------------------------
# Data preparation.
#
# This kernel's inputs are the forward's outputs, so the split counts come from
# the same CSR schedule the forward port builds -- that is what makes the split
# occupancy ragged in the way production is, with rows whose degree falls short
# of `topk` and rows that use every slot. The partial values themselves are
# synthetic: the reduction is agnostic to where they came from, and generating
# them directly keeps a combine correctness run from also paying for a forward.
#
# Slots at or past a row's degree are filled with NaN rather than left
# undefined. The kernel is specified never to read them -- the LSE staging
# fills `-inf` past the row's count (combine.py:650-672), the partial loader
# zero-fills the stage instead of issuing the copy (:1154-1161), and the
# accumulate loop is guarded on a positive scale (:944) -- so a single read of
# a dead slot poisons the output and the bitwise gate catches it.
# ---------------------------------------------------------------------------
_CSR_CONFIG_KEYS = (
    "batch",
    "seqlen_q",
    "seqlen_k",
    "head_kv",
    "qhead_per_kv",
    "topk",
    "blk_kv",
    "seqlen_pattern",
)

# Fraction of live slots forced to `-inf`, which is the value the forward
# writes for a split whose rows are entirely masked. It exercises the
# `lse_max == -inf` fallback and, on a low-degree row, the all-dead row whose
# output is defined to be zero with an `-inf` LSE (combine.py:815-820).
_DEAD_SPLIT_FRACTION = 0.02


def prepare_data(*, seed: int = 0, **config) -> dict[str, Any]:
    """Build the split partials, the schedule's split counts and the outputs."""
    import torch

    from tirx_kernels.msa.sparse_prepare_fwd_split_atomic import prepare_data as prepare_csr

    config.pop("label", None)
    csr_config = {key: config[key] for key in _CSR_CONFIG_KEYS if key in config}
    csr = prepare_csr(seed=seed, **csr_config)

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(5417 + seed)
    head_kv = config["head_kv"]
    qhead_per_kv = config["qhead_per_kv"]
    head_q = head_kv * qhead_per_kv
    topk = config["topk"]
    total_q = csr["total_q"]
    partial_dtype = config.get("partial_dtype", "float32")
    temperature = bool(config.get("temperature", False))
    has_output_scale = bool(config.get("output_scale", False))

    split_counts = csr["degrees"].to(torch.int32).contiguous()
    if int(split_counts.max()) > topk:
        raise ValueError("the CSR schedule produced a degree above topk")

    # `split < split_counts[q, kv_head]`, broadcast to the partial shape. The
    # split-atomic schedule packs each group's slots into `0 .. degree-1`, so
    # the degree alone decides which slots are live.
    per_head_q = split_counts.repeat_interleave(qhead_per_kv, dim=1)
    splits = torch.arange(topk, device=device).view(-1, 1, 1)
    live = splits < per_head_q.unsqueeze(0)

    shape = (topk, total_q, head_q)
    # A wide spread so the max selection, the exponent magnitudes and the
    # reciprocal of the sum all see non-degenerate values.
    lse_partial = torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * 2.0
    dead = torch.rand(shape, device=device, generator=generator) < _DEAD_SPLIT_FRACTION
    lse_partial = torch.where(dead, torch.full_like(lse_partial, float("-inf")), lse_partial)
    lse_partial = torch.where(live, lse_partial, torch.full_like(lse_partial, float("nan")))

    o_partial = torch.randn(
        (*shape, HEAD_DIM), dtype=torch.float32, device=device, generator=generator
    )
    o_partial = torch.where(
        live.unsqueeze(-1), o_partial, torch.full_like(o_partial, float("nan"))
    ).to(_torch_dtype(partial_dtype))
    if partial_dtype == "float8_e4m3":
        # `nan` survives the cast to e4m3 as `0x7f`; assert it rather than
        # assume it, because a dead slot that quietly became a finite value
        # would make the mask check vacuous.
        dead_bytes = o_partial.view(torch.uint8)[~live.unsqueeze(-1).expand_as(o_partial)]
        if dead_bytes.numel() and int(dead_bytes.min()) != 0x7F:
            raise AssertionError("fp8 dead-slot poison did not encode as 0x7f")

    lse_temperature_partial = None
    if temperature:
        lse_temperature_partial = (
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * 1.5
        )
        lse_temperature_partial = torch.where(
            live, lse_temperature_partial, torch.full_like(lse_temperature_partial, float("nan"))
        )

    output_scale = None
    if has_output_scale:
        # `v_global_scale` is a one-element fp32 tensor (interface.py:972).
        output_scale = torch.tensor([0.7315], dtype=torch.float32, device=device)

    seqused_q = None
    if config.get("seqused", False):
        seqlens_q = torch.tensor(csr["seqlens_q"], dtype=torch.int32, device=device)
        trims = torch.tensor(
            [(37 * (i + 1)) % 64 for i in range(len(csr["seqlens_q"]))],
            dtype=torch.int32,
            device=device,
        )
        seqused_q = torch.clamp(seqlens_q - trims, min=1).contiguous()

    return {
        "config": dict(config),
        "o_partial": o_partial.contiguous(),
        "lse_partial": lse_partial.contiguous(),
        "lse_temperature_partial": lse_temperature_partial,
        "split_counts": split_counts,
        "output_scale": output_scale,
        "seqused_q": seqused_q,
        "cu_seqlens_q": csr["cu_seqlens_q"],
        "seqlens_q": csr["seqlens_q"],
        "live": live,
        "topk": topk,
        "total_q": total_q,
        "head_q": head_q,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "num_batches": len(csr["seqlens_q"]),
        "partial_dtype": partial_dtype,
        "temperature": temperature,
    }


def make_outputs(data: dict[str, Any]) -> dict[str, Any]:
    """Fresh output buffers, NaN-filled so an unwritten row stays visible.

    The host allocates these with `torch.empty` (interface.py:1470-1484). Both
    sides get the same deterministic prefill instead, which costs nothing and
    turns "the kernel wrote every row it owns" into something the comparison
    can actually check.
    """
    import torch

    outputs = {
        "o_out": torch.full(
            (data["total_q"], data["head_q"], HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        "lse_out": torch.full(
            (data["total_q"], data["head_q"]), float("nan"), dtype=torch.float32, device="cuda"
        ),
    }
    if data["temperature"]:
        outputs["lse_temperature_out"] = torch.full(
            (data["total_q"], data["head_q"]), float("nan"), dtype=torch.float32, device="cuda"
        )
    return outputs


def tirx_args(data: dict[str, Any], outputs: dict[str, Any]) -> tuple:
    """The launch ABI, bound once outside any timed region."""
    import torch

    def as_bits(t):
        """fp8 reaches the kernel as raw ``uint8``; see ``_KERN_DTYPES``."""
        return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t

    topk = data["topk"]
    total_q = data["total_q"]
    head_q = data["head_q"]
    head_kv = data["head_kv"]
    # An absent optional buffer still occupies its parameter slot; the
    # specialization simply never reads it.
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device="cuda")
    dummy_i32 = torch.zeros(1, dtype=torch.int32, device="cuda")
    temperature_partial = data["lse_temperature_partial"]
    temperature_out = outputs.get("lse_temperature_out")
    return (
        as_bits(data["o_partial"]).reshape(-1),
        data["lse_partial"].reshape(-1),
        outputs["o_out"].reshape(-1),
        outputs["lse_out"].reshape(-1),
        dummy_f32 if temperature_partial is None else temperature_partial.reshape(-1),
        dummy_f32 if temperature_out is None else temperature_out.reshape(-1),
        data["cu_seqlens_q"],
        dummy_i32 if data["seqused_q"] is None else data["seqused_q"],
        data["split_counts"].reshape(-1),
        dummy_f32 if data["output_scale"] is None else data["output_scale"],
        total_q * head_q * HEAD_DIM,
        head_q * HEAD_DIM,
        HEAD_DIM,
        total_q * head_q,
        head_q,
        head_q * HEAD_DIM,
        HEAD_DIM,
        head_q,
        head_kv,
        data["qhead_per_kv"],
        total_q,
        head_q,
        data["num_batches"],
        *fast_divmod(head_q),
    )


def reference_case(data: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """The argument bundle MSA's own compiled combine takes."""
    return {
        "o_partial": data["o_partial"],
        "lse_partial": data["lse_partial"],
        "o_out": outputs["o_out"],
        "lse_out": outputs["lse_out"],
        "lse_temperature_partial": data["lse_temperature_partial"],
        "lse_temperature_out": outputs.get("lse_temperature_out"),
        "cu_seqlens_q": data["cu_seqlens_q"],
        "seqused": data["seqused_q"],
        "split_counts": data["split_counts"],
        "output_scale": data["output_scale"],
        "qhead_per_kv": data["qhead_per_kv"],
        "head_dim": HEAD_DIM,
        "topk": data["topk"],
    }


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
def written_row_mask(data: dict[str, Any]):
    """The `(q, head)` rows this launch is defined to write.

    Every row is written unless `seqused` shortens a batch, in which case the
    kernel's own bound stops the grid short (combine.py:545-547) and the tail
    rows keep whatever the allocation held.
    """
    import torch

    total_q = data["total_q"]
    head_q = data["head_q"]
    if data["seqused_q"] is None:
        return torch.ones((total_q, head_q), dtype=torch.bool, device="cuda")

    mask = torch.zeros((total_q, head_q), dtype=torch.bool, device="cuda")
    cu = data["cu_seqlens_q"].tolist()
    used = data["seqused_q"].tolist()
    for batch, count in enumerate(used):
        start = cu[batch]
        mask[start : start + int(count)] = True
    return mask


def assert_outputs_match(
    data: dict[str, Any],
    outputs: dict[str, Any],
    expected: dict[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Compare every written row, and hold the rest to the prefill.

    Bitwise by default. The reduction runs a fixed-order serial loop per thread
    with fp32 fused multiply-adds, its reductions are fixed-topology warp
    shuffles, and no atomic appears anywhere -- so a port that selects the same
    instructions reproduces the reference exactly, and a tolerance here would
    only hide the cases where it does not.
    """
    import torch

    mask = written_row_mask(data)
    o_mask = mask.unsqueeze(-1).expand_as(outputs["o_out"])
    torch.testing.assert_close(
        outputs["o_out"][o_mask].float(),
        expected["o_out"][o_mask].float(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    torch.testing.assert_close(
        outputs["lse_out"][mask], expected["lse_out"][mask], rtol=rtol, atol=atol, equal_nan=True
    )
    if "lse_temperature_out" in outputs:
        torch.testing.assert_close(
            outputs["lse_temperature_out"][mask],
            expected["lse_temperature_out"][mask],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    if not bool(mask.all()):
        untouched = ~mask
        if not bool(torch.isnan(outputs["lse_out"][untouched]).all()):
            raise AssertionError("the kernel wrote rows past seqused")


def run_test(**config):
    """Compile, launch, and validate one config against MSA's own kernel."""
    import unittest

    import torch

    from tirx_kernels.runner import compile_kernel

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    config.pop("label", None)
    data = prepare_data(**config)

    try:
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_combine
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc

    expected = make_outputs(data)
    try:
        compiled_sparse_atten_combine(reference_case(data, expected))()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    torch.cuda.synchronize()

    executable = compile_kernel(get_kernel(**config))
    outputs = make_outputs(data)
    executable(*tirx_args(data, outputs))
    torch.cuda.synchronize()
    assert_outputs_match(data, outputs, expected)


def prepare_bench(**config):
    """Compile the TIRx specialization without initializing CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config.pop("label", None)
    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


# ---------------------------------------------------------------------------
# Benchmark entry points.
#
# No rotation is needed between iterations: the kernel reads its partials
# without touching them and overwrites -- never accumulates into -- the outputs
# it owns, so the hundredth launch does exactly the work the first one did.
# ---------------------------------------------------------------------------
def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    """Kernel-only comparison against MSA's compiled combine launch."""
    from tirx_kernels.runner import bench

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    executable = prepared["executable"]

    tirx_outputs = make_outputs(data)
    tirx_bound = tirx_args(data, tirx_outputs)

    def tirx_launch():
        executable(*tirx_bound)

    def build_reference():
        from tirx_kernels.msa.utils._msa_bench import compiled_sparse_atten_combine

        launch = compiled_sparse_atten_combine(reference_case(data, make_outputs(data)))
        launch()  # pay the CuTeDSL compile and first-launch cost outside timing
        return launch

    return bench(
        {"tirx": tirx_launch},
        references={"msa": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
