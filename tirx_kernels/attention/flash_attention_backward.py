# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""SM100 two-CTA FlashAttention backward kernel using raw CUDA/PTX wrappers.

This is a TIRx port of Dao-AILab/flash-attention's
``flash_attn/cute/flash_bwd_sm100.py`` at commit
``d7e4dba3e568106b0f1b6323b07c1272f53679b3`` (2026-08-05).  The retained
specialization is dense MHA with equal query/key lengths, fp16 inputs,
head-dimension 128, and causal or noncausal masking.

The core uses public ``T.cuda`` and ``T.ptx`` wrappers for Tensor Maps, TMA,
DSMEM, mbarriers, tcgen05/TMEM, and scalar/vector PTX.  It intentionally does
not depend on TIRx tile primitives.
"""

import copy
import math

import torch

import tvm
from tvm.script import tirx as T
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.layout import ComposeLayout, S, TileLayout

IKET_EVENT_NAMES = (
    "dq-reduce",
    "ds-exchange",
    "dkv-epilogue",
    "mma-dk",
    "mma-dp",
    "mma-dq-alias-wait",
    "mma-dq-issue",
    "mma-dq-ready-wait",
    "mma-dv",
    "mma-s",
    "softmax-ds",
    "softmax-p",
    "tma-wait-a",
    "tma-wait-dpsum",
    "tma-wait-lse",
    "tma-wait-q",
    "tma-wait-qcol",
    "tma-prefetch",
    "dq-reduce-stage",
)


def tma_shared_layout(dtype, shape):
    """Construct a public 128-byte-swizzled tcgen05/TMA SMEM layout."""
    bits = tvm.DataType(dtype).bits
    per_element = (128 // bits).bit_length() - 1
    swizzle_len = 3
    atom_len = 3
    period = 1 << (per_element + swizzle_len + atom_len)
    atom_shape = [1] * (len(shape) - 2) + [8, 1024 // bits]
    layout = ComposeLayout(per_element, swizzle_len, atom_len, TileLayout(S[(period,)]))
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


# ---------------------------------------------------------------------------
# Preprocessing kernels
# ---------------------------------------------------------------------------


def build_preprocess(B, S, H, D):
    """Build dPsum/LSE-log2 and clear dQ accumulation in one pass.

    A 256-thread block covers 128 rows. Sixteen lanes cooperatively load each
    row, reduce with width-16 shuffles, and clear the matching dQ slice.  This
    matches the official kernel's copy topology: sixteen rows are processed in
    parallel and each thread owns 64 elements from each input tile.
    """
    if S % 128:
        raise ValueError("the SM100 backward preprocess requires seq_len divisible by 128")
    THREADS_PER_ROW = 16
    ELEMS_PER_THREAD = D // THREADS_PER_ROW
    BLOCK = 256
    ROWS_PER_WAVE = BLOCK // THREADS_PER_ROW
    ROWS_PER_BLOCK = 128
    ROW_ITERS = ROWS_PER_BLOCK // ROWS_PER_WAVE
    NBLK = S // ROWS_PER_BLOCK
    LOG2_E = math.log2(math.e)

    dot_f16x8_source_code = R"""
__forceinline__ __device__ float tvm_builtin_dot_f16x8(
    const half* lhs, const half* rhs) {
    uint4 lhs_bits = *reinterpret_cast<const uint4*>(lhs);
    uint4 rhs_bits = *reinterpret_cast<const uint4*>(rhs);
    const half2* lhs2 = reinterpret_cast<const half2*>(&lhs_bits);
    const half2* rhs2 = reinterpret_cast<const half2*>(&rhs_bits);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 l = __half22float2(lhs2[i]);
        float2 r = __half22float2(rhs2[i]);
        sum = fmaf(l.x, r.x, sum);
        sum = fmaf(l.y, r.y, sum);
    }
    return sum;
}
"""

    try:
        from tvm.script import tir as T
    except ImportError:
        from tvm.script import tirx as T

    @T.prim_func
    def preprocess_kernel(
        dO_buf: T.Buffer((B, S, H, D), "float16"),
        O_buf: T.Buffer((B, S, H, D), "float16"),
        LSE_buf: T.Buffer((B, H, S), "float32"),
        dpsum_out: T.Buffer((B, H, S), "float32"),
        LSE_log2_out: T.Buffer((B, H, S), "float32"),
        dQ_accum: T.Buffer((B, H, S, D), "float32"),
    ):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        for bx in T.thread_binding(NBLK, thread="blockIdx.x"):
            for by in T.thread_binding(H, thread="blockIdx.y"):
                for bz in T.thread_binding(B, thread="blockIdx.z"):
                    for tx in T.thread_binding(BLOCK, thread="threadIdx.x"):
                        col_in_row: T.int32 = tx % THREADS_PER_ROW
                        row_in_wave: T.int32 = tx // THREADS_PER_ROW
                        d_start: T.int32 = col_in_row * ELEMS_PER_THREAD

                        # Match the official dependency shape: start the
                        # independent LSE load before the O/dO dot products
                        # and retain it until the final scalar conversion.
                        # This lets its GMEM latency overlap the eight row
                        # iterations below.
                        lse_for_log2: T.float32 = T.float32(0)
                        if tx < ROWS_PER_BLOCK:
                            lse_s: T.int32 = bx * ROWS_PER_BLOCK + tx
                            lse_for_log2 = LSE_buf[bz, by, lse_s]

                        for row_iter in T.unroll(ROW_ITERS):
                            s: T.int32 = (
                                bx * ROWS_PER_BLOCK + row_iter * ROWS_PER_WAVE + row_in_wave
                            )
                            acc: T.float32 = T.cuda.func_call(
                                "tvm_builtin_dot_f16x8",
                                T.address_of(dO_buf[bz, s, by, d_start]),
                                T.address_of(O_buf[bz, s, by, d_start]),
                                source_code=dot_f16x8_source_code,
                                return_type="float32",
                            )
                            for chunk in T.unroll(ELEMS_PER_THREAD // 4):
                                for d in T.vectorized(4):
                                    dQ_accum[bz, by, s, d_start + chunk * 4 + d] = T.float32(0)

                            acc = acc + T.cuda.__shfl_xor_sync(
                                T.uint32(0xFFFFFFFF), acc, 8, THREADS_PER_ROW
                            )
                            acc = acc + T.cuda.__shfl_xor_sync(
                                T.uint32(0xFFFFFFFF), acc, 4, THREADS_PER_ROW
                            )
                            acc = acc + T.cuda.__shfl_xor_sync(
                                T.uint32(0xFFFFFFFF), acc, 2, THREADS_PER_ROW
                            )
                            acc = acc + T.cuda.__shfl_xor_sync(
                                T.uint32(0xFFFFFFFF), acc, 1, THREADS_PER_ROW
                            )
                            if col_in_row == 0:
                                dpsum_out[bz, by, s] = acc

                        # LSE conversion has one independent scalar per row;
                        # spread the 128 rows across 128 threads instead of
                        # serializing eight conversions on each row leader.
                        if tx < ROWS_PER_BLOCK:
                            lse_s: T.int32 = bx * ROWS_PER_BLOCK + tx
                            LSE_log2_out[bz, by, lse_s] = T.if_then_else(
                                lse_for_log2 == T.float32(-float("inf")),
                                T.float32(0),
                                lse_for_log2 * T.float32(LOG2_E),
                            )

    mod = tvm.IRModule({"main": preprocess_kernel})
    return tvm.compile(mod, target=tvm.target.Target("cuda"))


def build_cast_f32_to_f16(B, S, H, D, scale):
    """Scale and transpose the head-major dQ accumulation to fp16."""
    try:
        from tvm.script import tir as T
    except ImportError:
        from tvm.script import tirx as T

    GROUP_WIDTH = 4
    GROUPS_PER_THREAD = 4
    BLOCK = 256
    GROUPS_PER_BLOCK = BLOCK * GROUPS_PER_THREAD
    num_groups = B * S * H * (D // GROUP_WIDTH)
    NBLK = (num_groups + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK

    scale_cast_f32x4_f16x4_source_code = R"""
__forceinline__ __device__ void tvm_builtin_scale_cast_f32x4_f16x4(
    half* dst, const float* src, float scale) {
    float4 value = *reinterpret_cast<const float4*>(src);
    half2 lo = __floats2half2_rn(value.x * scale, value.y * scale);
    half2 hi = __floats2half2_rn(value.z * scale, value.w * scale);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t*>(&lo);
    packed.y = *reinterpret_cast<const uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(dst) = packed;
}
"""

    @T.prim_func
    def cast_kernel(src: T.Buffer((B, H, S, D), "float32"), dst: T.Buffer((B, S, H, D), "float16")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        for bx in T.thread_binding(NBLK, thread="blockIdx.x"):
            for tx in T.thread_binding(BLOCK, thread="threadIdx.x"):
                for e in T.unroll(GROUPS_PER_THREAD):
                    group: T.int32 = bx * GROUPS_PER_BLOCK + e * BLOCK + tx
                    if group < num_groups:
                        d_group: T.int32 = group % (D // GROUP_WIDTH)
                        h: T.int32 = group // (D // GROUP_WIDTH) % H
                        s: T.int32 = group // (D // GROUP_WIDTH * H) % S
                        b: T.int32 = group // (D // GROUP_WIDTH * H * S)
                        d: T.int32 = d_group * GROUP_WIDTH
                        # The raw tcgen05 dQ accumulator uses the physical
                        # 128x128 C-fragment bit layout.  Decode that internal
                        # layout while producing the public sequence-major dQ.
                        s_in_block: T.int32 = s % 128
                        src_s: T.int32 = (
                            s // 128 * 128
                            + ((s_in_block >> 5) & 1)
                            + (((d >> 6) & 1) << 1)
                            + (((d >> 2) & 15) << 2)
                            + (((s_in_block >> 6) & 1) << 6)
                        )
                        src_d: T.int32 = (s_in_block & 31) << 2
                        T.cuda.func_call(
                            "tvm_builtin_scale_cast_f32x4_f16x4",
                            T.address_of(dst[b, s, h, d]),
                            T.address_of(src[b, h, src_s, src_d]),
                            T.float32(scale),
                            source_code=scale_cast_f32x4_f16x4_source_code,
                            return_type="void",
                        )

    mod = tvm.IRModule({"main": cast_kernel})
    return tvm.compile(mod, target=tvm.target.Target("cuda"))


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


def build_kernel(
    BATCH: int,
    HEADS_PER_BATCH: int,
    SEQ_LEN: int,
    HEAD_DIM: int = 128,
    *,
    causal: bool = False,
    attention_scale: float | None = None,
    sm_count: int = 148,
):
    if HEAD_DIM != 128:
        raise ValueError("the SM100 2-CTA backward kernel currently requires head_dim=128")
    if SEQ_LEN % 256:
        raise ValueError("the SM100 2-CTA backward kernel requires seq_len divisible by 256")
    if sm_count < 2:
        raise ValueError("the SM100 2-CTA backward kernel requires at least two SMs")
    f16 = tvm.DataType("float16")
    f32 = tvm.DataType("float32")

    # Leave the first KiB to the barriers allocated before the matrix payloads.
    # TCGEN descriptors derive their address field from an allocated shared view
    # below; this is a pool-relative choice, not an architectural shared address.
    POOL_Q_ROW = 1024
    MATRIX_DESC_F16_SS_LDO_1024 = 0x4000404004000000
    MATRIX_DESC_F16_SS_LDO_512 = 0x4000404002000000
    MATRIX_DESC_F16_TS = 0x4000404000000000

    def matrix_desc_from_anchor(layout_bits, anchor_start, byte_delta):
        # layout_bits deliberately has a zero start-address field.  Authority
        # comes from anchor_start, which is derived from a real shared view.
        start = T.bitwise_and(anchor_start + T.cast(byte_delta // 16, "uint64"), T.uint64(0x3FFF))
        return T.bitwise_or(T.uint64(layout_bits), start)

    @T.inline
    def copy_128b(dst, value):
        # The dialect takes the payload as one 128-bit register; callers pass a
        # uint128 view element rather than a pointer into their local buffer.
        T.ptx.st.weak.shared__cta.b128(dst, value)

    def pointer_offset(ptr, offset):
        return T.ptr_byte_offset(ptr, offset * 2, "float16")

    def pointer_offset_f32(ptr, offset):
        return T.ptr_byte_offset(ptr, offset * 4, "float32")

    # kind::f16 = fp16 A/B into an fp32 accumulator. The .ss and .ts table
    # entries share this chain and are told apart by the A operand's dtype:
    # u64 is a shared-memory descriptor, u32 a TMEM address.
    _MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
    # cta_group::2 takes an 8-lane disable-output-lane vector; nothing is
    # disabled here, but the operands are not optional.
    _MMA_KEEP_ALL_LANES = (0, 0, 0, 0, 0, 0, 0, 0)

    @T.inline
    def mma_ss_one(d, accumulate, a_desc, b_desc, instruction):
        T.ptx[_MMA_F16](
            T.uint32(d),
            T.uint64(a_desc),
            T.uint64(b_desc),
            T.uint32(instruction),
            *_MMA_KEEP_ALL_LANES,
            T.ptx.pred(accumulate),
        )

    @T.inline
    def mma_ss8(d, accumulate, a_base, b_base):
        mma_ss_one(d, accumulate, a_base, b_base, 270532624)
        mma_ss_one(d, 1, a_base + 2, b_base + 2, 270532624)
        mma_ss_one(d, 1, a_base + 4, b_base + 4, 270532624)
        mma_ss_one(d, 1, a_base + 6, b_base + 6, 270532624)
        mma_ss_one(d, 1, a_base + 1024, b_base + 512, 270532624)
        mma_ss_one(d, 1, a_base + 1026, b_base + 514, 270532624)
        mma_ss_one(d, 1, a_base + 1028, b_base + 516, 270532624)
        mma_ss_one(d, 1, a_base + 1030, b_base + 518, 270532624)

    @T.inline
    def mma_ts_one(d, a, accumulate, b_desc):
        T.ptx[_MMA_F16](
            T.uint32(d),
            T.uint32(a),
            T.uint64(b_desc),
            T.uint32(270598160),
            *_MMA_KEEP_ALL_LANES,
            T.ptx.pred(accumulate),
        )

    @T.inline
    def mma_ts8(d, a, accumulate, b_base):
        mma_ts_one(d, a, accumulate, b_base)
        mma_ts_one(d, a + 8, 1, b_base + 128)
        mma_ts_one(d, a + 16, 1, b_base + 256)
        mma_ts_one(d, a + 24, 1, b_base + 384)
        mma_ts_one(d, a + 32, 1, b_base + 512)
        mma_ts_one(d, a + 40, 1, b_base + 640)
        mma_ts_one(d, a + 48, 1, b_base + 768)
        mma_ts_one(d, a + 56, 1, b_base + 896)

    @T.inline
    def mma_s(d, accumulate, desc_k_row, desc_q_row):
        mma_ss8(d, accumulate, desc_k_row, desc_q_row)

    @T.inline
    def mma_dp(d, accumulate, desc_v_row, desc_do_row):
        mma_ss8(d, accumulate, desc_v_row, desc_do_row)

    @T.inline
    def mma_dv(d, a, accumulate, desc_do_col):
        mma_ts8(d, a, accumulate, desc_do_col)

    @T.inline
    def mma_dk(d, a, accumulate, desc_q_col):
        mma_ts8(d, a, accumulate, desc_q_col)

    @T.inline
    def mma_dq(d, accumulate, desc_ds_exch, desc_k_col):
        mma_ss_one(d, accumulate, desc_ds_exch, desc_k_col, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 128, desc_k_col + 128, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 256, desc_k_col + 256, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 384, desc_k_col + 384, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 512, desc_k_col + 512, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 640, desc_k_col + 640, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 768, desc_k_col + 768, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 896, desc_k_col + 896, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1024, desc_k_col + 1024, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1152, desc_k_col + 1152, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1280, desc_k_col + 1280, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1408, desc_k_col + 1408, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1536, desc_k_col + 1536, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1664, desc_k_col + 1664, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1792, desc_k_col + 1792, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1920, desc_k_col + 1920, 136413200)

    @T.inline
    def cast_f32x2_to_f16x2(dst, src):
        T.cuda.float22half2(dst, src)

    fma_scale_sub_f32x2_source_code = R"""
__forceinline__ __device__ unsigned long long tvm_builtin_fma_scale_sub_f32x2(
    unsigned long long scores,
    unsigned long long scale,
    unsigned long long lse) {
    float2 score_pair = *reinterpret_cast<float2*>(&scores);
    float2 scale_pair = *reinterpret_cast<float2*>(&scale);
    float2 lse_pair = *reinterpret_cast<float2*>(&lse);
    float2 result = make_float2(
        fmaf(score_pair.x, scale_pair.x, -lse_pair.x),
        fmaf(score_pair.y, scale_pair.y, -lse_pair.y));
    return *reinterpret_cast<unsigned long long*>(&result);
}
"""

    def fma_scale_sub_f32x2(scores, scale, lse):
        return T.cuda.func_call(
            "tvm_builtin_fma_scale_sub_f32x2",
            scores,
            scale,
            lse,
            source_code=fma_scale_sub_f32x2_source_code,
            return_type="uint64",
        )

    # The dialect takes one composed TMEM address instead of a base plus
    # row/col; get_tmem_addr packs them the same way the old operands did.
    def _tmem_addr(base, row, col):
        return T.cuda.get_tmem_addr(T.uint32(base), row, col)

    def tmem_load_64(dst, dst_offset, base, row, col):
        return T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"](
            *[dst[dst_offset + i] for i in range(64)], _tmem_addr(base, row, col)
        )

    def tmem_load_32(dst, dst_offset, base, row, col):
        return T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
            *[dst[dst_offset + i] for i in range(32)], _tmem_addr(base, row, col)
        )

    def tmem_store_32(src, src_offset, base, row, col):
        return T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
            _tmem_addr(base, row, col), *[src[src_offset + i] for i in range(32)]
        )

    # 2-SM cluster TMA load: unicast (no .multicast::cluster) and no cache
    # policy, so only the completion and cta_group tokens ride the chain. Note
    # the mbarrier operand now follows the coordinates.
    _TMA_G2S_2SM = (
        "cp.async.bulk.tensor.{dim}d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
    )
    # Stores put the tensor map and coordinates first and the shared source last.
    _TMA_S2G = "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"
    _TMA_S2G_REDUCE = (
        "cp.reduce.async.bulk.tensor.{dim}d.global.shared::cta.{redop}.tile.bulk_group"
    )
    # The destination is this CTA's own shared memory.
    _BULK_G2S_CTA = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    # shared::cta -> peer CTA's shared::cluster window (DSMEM push).
    _BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
    # pair_mask names both CTAs of the pair, so the commit is the multicast form.
    _TCGEN05_COMMIT = (
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
    )

    def tma_g2s(dim, dst_ptr, mbar, tensormap_addr, *coords):
        return T.ptx[_TMA_G2S_2SM.format(dim=dim)](dst_ptr, tensormap_addr, *coords, mbar)

    def tma_s2g(dim, src_ptr, tensormap_addr, *coords):
        return T.ptx[_TMA_S2G.format(dim=dim)](tensormap_addr, *coords, src_ptr)

    def tma_s2g_reduce(dim, src_ptr, tensormap_addr, redop, *coords):
        chain = _TMA_S2G_REDUCE.format(dim=dim, redop=redop)
        return T.ptx[chain](tensormap_addr, *coords, src_ptr)

    def bulk_g2s_cta(dst_ptr, src_ptr, num_bytes, mbar):
        return T.ptx[_BULK_G2S_CTA](dst_ptr, src_ptr, T.uint32(num_bytes), mbar)

    def tcgen05_commit(mbar_ptr, cta_mask):
        return T.ptx[_TCGEN05_COMMIT](mbar_ptr, T.Cast("uint16", cta_mask))

    def tmem_store_16(src, src_offset, base, row, col):
        return T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
            _tmem_addr(base, row, col), *[src[src_offset + i] for i in range(16)]
        )

    @T.meta_class
    class RowiseSwizzleOffset:
        def __init__(self, swizzle_len, atom_len, per_element, row_base, prefix="row_sw"):
            self.swizzle_len = swizzle_len
            self.atom_len = atom_len
            self.per_element = per_element
            self.row_base = row_base
            self.signed_strides = T.alloc_buffer([self.atom_len], "int32", scope="local")
            self.n_dim = self.swizzle_len + 1
            self.shape = [2] * self.n_dim
            self.shape[-1] = 1 << self.per_element

        @T.inline
        def init(self):
            for i in T.unroll(self.swizzle_len):
                y_i = T.meta_var(self.row_base & (1 << i))
                stride_i = T.meta_var(1 << (i + self.per_element))
                self.signed_strides[i] = -stride_i if y_i > 0 else stride_i
            for i in T.unroll(self.swizzle_len, self.atom_len):
                stride_i = T.meta_var(1 << (i + self.per_element))
                self.signed_strides[i] = stride_i

        def apply(self, offset):
            offset_layout = TileLayout(
                S[
                    self.shape : [
                        self.signed_strides[self.swizzle_len - 1 - i] for i in range(self.atom_len)
                    ]
                    + [0]
                ]
            )
            return offset_layout.apply(offset)["m"]

    NUM_HEADS = BATCH * HEADS_PER_BATCH

    CTA_GROUP = 2
    CLUSTER_M, CLUSTER_N = 2, 1
    CLUSTER_SIZE = CLUSTER_M * CLUSTER_N
    BLK_M = 128
    BLK_N = 256  # doubled: 2 CTAs x 128 rows each
    CTA_N = BLK_N // CTA_GROUP  # 128 per CTA
    MMA_N = 128  # TMEM output cols (unchanged)
    B_N = BLK_M // CTA_GROUP  # 64: per-CTA B rows for row-split
    B_N_COL = HEAD_DIM // CTA_GROUP  # 64: per-CTA B cols for col-split
    EPI_N = 64
    STRIP_SIZE = 64
    TMEM_LD_N = 64
    DQ_RED_N = 8
    DQ_STAGES = 4
    DQ_M_PER_CTA = 64  # Phase E Layout B: 64 rows per CTA
    DQ_ROWS_PER_STAGE = DQ_RED_N
    DQ_REDUCE_ITERS = DQ_M_PER_CTA // DQ_ROWS_PER_STAGE

    NUM_M_TILES = SEQ_LEN // BLK_M
    NUM_N_TILES = SEQ_LEN // BLK_N  # halved vs v0
    STRIPS = HEAD_DIM // STRIP_SIZE  # 2

    softmax_scale = 1.0 / math.sqrt(HEAD_DIM) if attention_scale is None else float(attention_scale)
    log2e = 1.4426950408889634
    scale_log2 = softmax_scale * log2e  # precomputed compile-time constant

    WG_NUMBER = 4
    DTYPE_SIZE = 2
    # Keep the public specialization argument even though the current upstream
    # kernel uses SingleTileScheduler for both causal and dense launches.
    _ = sm_count

    # TMA byte counts
    CTA_N_BYTES = CTA_N * HEAD_DIM * DTYPE_SIZE  # 32KB per CTA's K or V load
    Q_ROW_BYTES = B_N * HEAD_DIM * DTYPE_SIZE  # 16KB per CTA's Q row-split
    Q_COL_BYTES = BLK_M * B_N_COL * DTYPE_SIZE  # 16KB per CTA's Q col-split
    LSE_BYTES = BLK_M * 4  # 512 bytes (fp32)
    DPSUM_BYTES = BLK_M * 4  # 512 bytes (fp32)

    # SMEM layouts
    kv_layout = tma_shared_layout(f16, (CTA_N, HEAD_DIM))
    q_row_layout = tma_shared_layout(f16, (1, B_N, HEAD_DIM))
    q_col_layout = tma_shared_layout(f16, (BLK_M, B_N_COL))
    do_row_layout = tma_shared_layout(f16, (B_N, HEAD_DIM))
    do_col_layout = tma_shared_layout(f16, (BLK_M, B_N_COL))
    # dS for Phase E after DSMEM exchange: [BLK_N, B_N] per CTA = [256, 64]
    ds_exchange_layout = tma_shared_layout(f16, (BLK_N, B_N))
    # dS staging buffer: holds local N half x peer's M strip, for DSMEM send
    ds_stage_layout = tma_shared_layout(f16, (CTA_N, B_N))
    # K col-split for Phase E after DSMEM exchange: [BLK_N, B_N_COL] = [256, 64]
    k_col_layout = tma_shared_layout(f16, (BLK_N, B_N_COL))
    epi_layout_2 = tma_shared_layout(f16, (2, CTA_N, EPI_N))
    dq_red_layout = TileLayout(S[(DQ_STAGES, BLK_M, DQ_RED_N) : (BLK_M * DQ_RED_N, DQ_RED_N, 1)])

    # Current FA4 D=128 TMEM packing.  dQ aliases the upper half of S/P;
    # dP and dS alias each other after the compute warps drain dP.
    TMEM_OFF_A = 0  # S/P
    TMEM_OFF_DQ = MMA_N // 2  # dQ (64), aliases S/P
    TMEM_OFF_B = MMA_N  # dV accumulator (128)
    TMEM_OFF_DP = 2 * MMA_N  # dP/dS (256)
    TMEM_OFF_C = 3 * MMA_N  # dK accumulator (384)
    iket = IketProfiler()

    # fmt: off
    @T.prim_func
    def kernel(
        Q_g:      T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16),
        K_g:      T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16),
        V_g:      T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16),
        dO_g:     T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16),
        LSE_g:    T.Buffer((BATCH, HEADS_PER_BATCH, SEQ_LEN), f32),
        dpsum_g:  T.Buffer((BATCH, HEADS_PER_BATCH, SEQ_LEN), f32),
        dK_g:     T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16),
        dV_g:     T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16),
        dQ_acc_g: T.Buffer((BATCH, HEADS_PER_BATCH, SEQ_LEN, HEAD_DIM), f32),
    ):
        q_row_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", q_row_tensormap, "float16", 5, Q_g.data,
            HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            HEAD_DIM // 2, B_N, 2, 1, 1,
            1, 1, 1, 1, 1, 0, 3, 2, 0,
        )
        q_col_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", q_col_tensormap, "float16", 4, Q_g.data,
            HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            B_N_COL, BLK_M, 1, 1,
            1, 1, 1, 1, 0, 3, 2, 0,
        )
        k_row_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", k_row_tensormap, "float16", 5, K_g.data,
            HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            HEAD_DIM // 2, CTA_N, 2, 1, 1,
            1, 1, 1, 1, 1, 0, 3, 2, 0,
        )
        k_col_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", k_col_tensormap, "float16", 4, K_g.data,
            HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            B_N_COL, BLK_N, 1, 1,
            1, 1, 1, 1, 0, 3, 2, 0,
        )
        v_row_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", v_row_tensormap, "float16", 5, V_g.data,
            HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            HEAD_DIM // 2, CTA_N, 2, 1, 1,
            1, 1, 1, 1, 1, 0, 3, 2, 0,
        )
        do_row_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", do_row_tensormap, "float16", 5, dO_g.data,
            HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            HEAD_DIM // 2, B_N, 2, 1, 1,
            1, 1, 1, 1, 1, 0, 3, 2, 0,
        )
        do_col_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", do_col_tensormap, "float16", 4, dO_g.data,
            HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            B_N_COL, BLK_M, 1, 1,
            1, 1, 1, 1, 0, 3, 2, 0,
        )
        lse_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", lse_tensormap, "float32", 3, LSE_g.data,
            SEQ_LEN, HEADS_PER_BATCH, BATCH,
            SEQ_LEN * 4, HEADS_PER_BATCH * SEQ_LEN * 4,
            BLK_M, 1, 1,
            1, 1, 1, 0, 0, 2, 0,
        )
        dpsum_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", dpsum_tensormap, "float32", 3, dpsum_g.data,
            SEQ_LEN, HEADS_PER_BATCH, BATCH,
            SEQ_LEN * 4, HEADS_PER_BATCH * SEQ_LEN * 4,
            BLK_M, 1, 1,
            1, 1, 1, 0, 0, 2, 0,
        )
        dk_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", dk_tensormap, "float16", 4, dK_g.data,
            HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2,
            SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            EPI_N, CTA_N, 1, 1,
            1, 1, 1, 1, 0, 3, 2, 0,
        )
        dv_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", dv_tensormap, "float16", 4, dV_g.data,
            HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH,
            HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2,
            SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2,
            EPI_N, CTA_N, 1, 1,
            1, 1, 1, 1, 0, 3, 2, 0,
        )
        dq_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled", dq_tensormap, "float32", 4, dQ_acc_g.data,
            HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH,
            HEAD_DIM * 4, SEQ_LEN * HEAD_DIM * 4,
            HEADS_PER_BATCH * SEQ_LEN * HEAD_DIM * 4,
            HEAD_DIM, DQ_ROWS_PER_STAGE, 1, 1,
            1, 1, 1, 1, 0, 0, 2, 0,
        )
        T.device_entry()
        cluster_rank_ = T.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE])
        bx, by = T.cta_id([NUM_N_TILES * CLUSTER_SIZE, NUM_HEADS])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        id_in_pair: T.let = cluster_rank_ % CTA_GROUP
        pair_leader_rank: T.let = cluster_rank_ - id_in_pair

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tmem_dealloc_mbar = MBarrier(
            pool,
            1,
            leader=(wg_id == 3) & (warp_id == 0) & (lane_id == 0),
        )

        # Barriers
        tma_kv         = TMABar(pool, 1)
        tma_a          = TMABar(pool, 1)        # depth 1: single buf_A
        tma_q          = TMABar(pool, 1)        # single-stage in the current 2-CTA schedule
        tma_lse        = TMABar(pool, 1)
        tma_dpsum      = TMABar(pool, 1)
        tma_qcol       = TMABar(pool, 1)     # depth 1: Q_col single-buf
        mma2wg0_s      = TCGen05Bar(pool, 1)
        mma2wg0_dp     = TCGen05Bar(pool, 1)
        # mma2wg0_dvdk removed — buf_a_consumed/q_consumed signaled directly by tcgen05.commit
        mma2wg0_dq     = TCGen05Bar(pool, 1)
        ds_exch_mbar   = MBarrier(pool, 1)  # DSMEM exchange completion
        ds_exch_consumed = MBarrier(pool, 1)  # dQ MMA released the single DSMEM buffer
        wg02mma        = MBarrier(pool, 1)        # softmax fully done (incl DSMEM) → Phase E
        wg02mma_tmem   = MBarrier(pool, 1)   # softmax dS in TMEM only → Phase D (early signal)
        strip_ready    = MBarrier(pool, 1)
        s_tmem_consumed = MBarrier(pool, 1)  # next S read drained before aliased dQ write
        buf_a_consumed = MBarrier(pool, 1)  # depth 1: single buf_A
        q_consumed     = MBarrier(pool, 1)     # single-stage Q_row release
        lse_consumed   = MBarrier(pool, 1)
        dpsum_consumed = MBarrier(pool, 1)
        qcol_consumed  = MBarrier(pool, 1)  # depth 1: Q_col release
        dq_tmem_free   = MBarrier(pool, 1)
        dv_done        = TCGen05Bar(pool, 1)  # dV accumulator ready → epilogue stage 0
        dk_done        = TCGen05Bar(pool, 1)  # dK accumulator ready → epilogue stage 1
        pool.move_base_to(POOL_Q_ROW)


        # Match SharedStorage's physical order in the upstream 2-CTA D=128
        # specialization: sQ, sK, sV, sdO, sQt, sdOt, sdS_xchg, sKt, sdS.
        Q_row     = pool.alloc((1, B_N, HEAD_DIM), f16, layout=q_row_layout)           # sQ: 16KB
        K_smem    = pool.alloc((CTA_N, HEAD_DIM), f16, layout=kv_layout)               # sK: 32KB
        V_smem    = pool.alloc((CTA_N, HEAD_DIM), f16, layout=kv_layout)               # sV: 32KB
        dO_row    = pool.alloc((B_N, HEAD_DIM), f16, layout=do_row_layout)             # sdO: 16KB
        Q_col     = pool.alloc((BLK_M, B_N_COL), f16, layout=q_col_layout)             # sQt: 16KB
        dO_col    = pool.alloc((BLK_M, B_N_COL), f16, layout=do_col_layout)            # sdOt: 16KB
        dS_send   = pool.alloc((CTA_N, B_N), f16, layout=ds_stage_layout)              # sdS_xchg: 16KB
        K_col     = pool.alloc((BLK_N, B_N_COL), f16, layout=k_col_layout)             # sKt: 32KB
        dS_exch   = pool.alloc((BLK_N, B_N), f16, layout=ds_exchange_layout)           # sdS: 32KB
        sLSE      = pool.alloc((1, BLK_M), f32, layout=TileLayout(S[(1, BLK_M) : (BLK_M, 1)]))  # 512B
        sDPsum    = pool.alloc((BLK_M,), f32, layout=TileLayout(S[(BLK_M,) : (1,)]))          # 512B
        dQ_smem   = pool.alloc(
            (DQ_STAGES, BLK_M, DQ_RED_N), f32,
            layout=dq_red_layout, align=1024,
        )
        # Upstream's two-stage dKV epilogue reuses sV for dV and sK for dK.
        # Both compute WGs collaborate on each output, one 64-column half per
        # WG, while the MMA warp overlaps the dV epilogue with its dK/dQ tail.
        pool.move_base_to(K_smem.elem_offset * DTYPE_SIZE)
        dK_epi = pool.alloc((2, CTA_N, EPI_N), f16, layout=epi_layout_2)
        pool.move_base_to(V_smem.elem_offset * DTYPE_SIZE)
        dV_epi = pool.alloc((2, CTA_N, EPI_N), f16, layout=epi_layout_2)
        pool.commit()

        desc_k_row: T.uint64
        desc_q_row: T.uint64
        desc_v_row: T.uint64
        desc_do_row: T.uint64
        desc_q_col: T.uint64
        desc_do_col: T.uint64
        desc_k_col: T.uint64
        desc_ds_exch: T.uint64

        # Total: 32+32+32+32+16+16+16+32+1+0.5+8 = 217.5KB

        tma_kv.init(1)
        tma_a.init(1)
        tma_q.init(1)
        tma_lse.init(1)
        tma_dpsum.init(1)
        tma_qcol.init(1)
        mma2wg0_s.init(1)
        mma2wg0_dp.init(1)
        # mma2wg0_dvdk removed
        mma2wg0_dq.init(1)
        wg02mma.init(CTA_GROUP)
        wg02mma_tmem.init(8 * CTA_GROUP)  # one elected thread per compute warp
        # PipelineUmmaAsync's empty side: one elected thread per compute warp,
        # across both compute WGs and both CTAs.
        strip_ready.init(8 * CTA_GROUP)
        s_tmem_consumed.init(8 * CTA_GROUP)
        buf_a_consumed.init(1)  # Phase C commit releases dO/dOt
        q_consumed.init(1)     # signaled by tcgen05.commit
        # LSE/dPsum are CTA-local streams.  One elected lane in each of the
        # eight compute warps releases the single buffer after its warp has
        # consumed the statistic, matching the current FA4 producer/consumer
        # topology without tying either statistic to the Q/dO MMA lifetime.
        lse_consumed.init(8)
        dpsum_consumed.init(8)
        qcol_consumed.init(1)  # signaled by tcgen05.commit
        dq_tmem_free.init(4 * CTA_GROUP)  # one elected thread per reducer warp
        ds_exch_mbar.init(1)
        ds_exch_consumed.init(1)
        dv_done.init(1)
        dk_done.init(1)
        tmem_dealloc_mbar.init(32)

        pair_mask: T.int32
        pair_mask = (1 << pair_leader_rank) | (1 << (pair_leader_rank + 1))

        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.barrier.cluster.arrive.relaxed()
        T.ptx.barrier.cluster.wait()

        # TMA barrier remote view for leader's mbar
        tma_kv_cta0 = tma_kv.remote_view(pair_leader_rank)
        tma_q_cta0 = tma_q.remote_view(pair_leader_rank)
        tma_qcol_cta0 = tma_qcol.remote_view(pair_leader_rank)
        tma_a_cta0 = tma_a.remote_view(pair_leader_rank)

        # Match the upstream SingleTileScheduler: every physical 2-CTA
        # cluster owns exactly one logical KV tile for the kernel lifetime.
        n_tile_idx = T.meta_var(bx // CLUSTER_SIZE)
        head_flat = T.meta_var(by)
        b_idx = T.meta_var(head_flat // HEADS_PER_BATCH)
        h_idx = T.meta_var(head_flat % HEADS_PER_BATCH)
        n_st  = T.meta_var(n_tile_idx * BLK_N)
        n_st_cta = T.meta_var(n_st + id_in_pair * CTA_N)  # per-CTA N offset
        # In causal mode, K/V tile n can only receive gradients from Q rows
        # at or below its first row.  BLK_N is exactly two BLK_M tiles, so
        # skipping these all-zero leading tiles removes the triangular half
        # of the dense schedule while keeping the pipeline trip count even.
        m_tile_start = T.meta_var(n_tile_idx * (BLK_N // BLK_M) if causal else 0)
        num_m_tiles_this_n = T.meta_var(NUM_M_TILES - m_tile_start)
        m_st_first = T.meta_var(m_tile_start * BLK_M)

        # ==============================================================
        # WG3: MMA (warp0) + TMA (warp1), matching current FA4's physical
        # warp placement.  The other two warps remain idle.
        # ==============================================================
        if wg_id == 3:
            # setmaxnreg.sync.aligned is a four-warp collective.  The idle
            # warp must request the same target as the MMA/load/relay warps;
            # CUDA already DCE'd its isolated 24-register request in the
            # profiled build, leaving this exact WG-wide dec-104 instruction.
            T.ptx.setmaxnreg.dec.sync.aligned.u32(104)
            if warp_id == 0:
                # The physical MMA warp owns TMEM allocation in both CTAs.
                # MMA, compute, and dQ-reduce warps rendezvous on the same
                # 13-warp named barrier used by the upstream allocator.
                T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(512)
                )
                T.ptx.bar.sync(T.uint32(5), 416)
            if warp_id == 1:
                # ---- TMA warp: 2 loads per M-tile (Q, dO) ----
                if T.cuda.elect_sync():
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_row_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_col_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_row_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_col_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(v_row_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(do_row_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(do_col_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(dk_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(dv_tensormap)))
                    T.evaluate(T.ptx.prefetch.tensormap(T.address_of(dq_tensormap)))
                q_cons_ph = PipelineState(1)
                q_cons_ph.init(1)
                lse_cons_ph = PipelineState(1)
                lse_cons_ph.init(1)
                qcol_cons_ph = PipelineState(1)
                qcol_cons_ph.init(1)
                a_cons_ph = PipelineState(1)
                a_cons_ph.init(1)
                dpsum_cons_ph = PipelineState(1)
                dpsum_cons_ph.init(1)

                # Byte counts for TMA barriers (both CTAs arrive at leader's mbar)
                K_COL_BYTES = T.meta_var(BLK_N * B_N_COL * DTYPE_SIZE)  # 256x64x2 = 32KB
                KV_TOTAL_BYTES = T.meta_var((CTA_N_BYTES * 2 + K_COL_BYTES) * CTA_GROUP)  # K+V+K_col from both CTAs
                Q_BATCH_BYTES = T.meta_var(Q_ROW_BYTES * CTA_GROUP)
                QCOL_BATCH_BYTES = T.meta_var(Q_COL_BYTES * CTA_GROUP)  # Q_col separate
                DO_BATCH_BYTES = T.meta_var((Q_ROW_BYTES + Q_COL_BYTES) * CTA_GROUP)

                @T.inline
                def tma_n_tile():
                    # K/V: each CTA loads its CTA_N=128 rows
                    if T.cuda.elect_sync():
                        tma_g2s(
                            5, K_smem.ptr_to([0, 0]), tma_kv_cta0.ptr_to([0]),
                            T.address_of(k_row_tensormap),
                            0, n_st_cta, 0, h_idx, b_idx,
                        )
                        tma_g2s(
                            5, V_smem.ptr_to([0, 0]), tma_kv_cta0.ptr_to([0]),
                            T.address_of(v_row_tensormap),
                            0, n_st_cta, 0, h_idx, b_idx,
                        )
                        # K_col for Phase E: all BLK_N=256 rows, per-CTA 64 cols
                        k_col_col_st = T.meta_var(id_in_pair * B_N_COL)
                        tma_g2s(
                            4, K_col.ptr_to([0, 0]), tma_kv_cta0.ptr_to([0]),
                            T.address_of(k_col_tensormap),
                            k_col_col_st, n_st, h_idx, b_idx,
                        )
                        if id_in_pair == 0:
                            tma_kv.arrive(0, KV_TOTAL_BYTES)

                    # Prologue: Q/Q_col/dO use the cluster MMA pipelines;
                    # LSE and dPsum use independent CTA-local bulk-copy
                    # pipelines consumed directly by the compute warps.
                    tma_prefetch_token = iket.range_start("tma-prefetch")
                    tma_wait_q_token = iket.range_start("tma-wait-q")
                    q_consumed.wait(0, q_cons_ph.phase)
                    iket.range_end(tma_wait_q_token)
                    q_row_st = T.meta_var(m_st_first + id_in_pair * B_N)
                    if T.cuda.elect_sync():
                        tma_g2s(
                            5, Q_row.ptr_to([0, 0, 0]), tma_q_cta0.ptr_to([0]),
                            T.address_of(q_row_tensormap),
                            0, q_row_st, 0, h_idx, b_idx,
                        )
                        if id_in_pair == 0:
                            tma_q.arrive(0, Q_BATCH_BYTES)
                    q_cons_ph.advance()
                    tma_wait_lse_token = iket.range_start("tma-wait-lse")
                    lse_consumed.wait(0, lse_cons_ph.phase)
                    iket.range_end(tma_wait_lse_token)
                    if T.cuda.elect_sync():
                        tma_lse.arrive(0, LSE_BYTES)
                        bulk_g2s_cta(
                            sLSE.ptr_to([0, 0]),
                            LSE_g.ptr_to([b_idx, h_idx, m_st_first]),
                            LSE_BYTES, tma_lse.ptr_to([0]),
                        )
                    lse_cons_ph.advance()
                    tma_wait_a_token = iket.range_start("tma-wait-a")
                    buf_a_consumed.wait(0, a_cons_ph.phase)
                    iket.range_end(tma_wait_a_token)
                    do_col_st = T.meta_var(id_in_pair * B_N_COL)
                    if T.cuda.elect_sync():
                        tma_g2s(
                            5, dO_row.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]),
                            T.address_of(do_row_tensormap),
                            0, q_row_st, 0, h_idx, b_idx,
                        )
                        tma_g2s(
                            4, dO_col.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]),
                            T.address_of(do_col_tensormap),
                            do_col_st, m_st_first, h_idx, b_idx,
                        )
                        if id_in_pair == 0:
                            tma_a.arrive(0, DO_BATCH_BYTES)
                    a_cons_ph.advance()
                    tma_wait_dpsum_token = iket.range_start("tma-wait-dpsum")
                    dpsum_consumed.wait(0, dpsum_cons_ph.phase)
                    iket.range_end(tma_wait_dpsum_token)
                    if T.cuda.elect_sync():
                        tma_dpsum.arrive(0, DPSUM_BYTES)
                        bulk_g2s_cta(
                            sDPsum.ptr_to([0]),
                            dpsum_g.ptr_to([b_idx, h_idx, m_st_first]),
                            DPSUM_BYTES, tma_dpsum.ptr_to([0]),
                        )
                    dpsum_cons_ph.advance()
                    iket.range_end(tma_prefetch_token)

                    # Loop: every stream is single-stage and independently
                    # released at the earliest observable consumption point.
                    for i_m in T.serial(num_m_tiles_this_n - 1, annotations={"disable_unroll": True}):
                        m_st_next = T.meta_var((m_tile_start + i_m + 1) * BLK_M)
                        q_row_st_next = T.meta_var(m_st_next + id_in_pair * B_N)
                        q_col_st_next = T.meta_var(id_in_pair * B_N_COL)
                        tma_prefetch_token = iket.range_start("tma-prefetch")

                        # Qt trails the row-major streams by one M tile in the
                        # official D=128 schedule.  Issue the previous tile at
                        # the front of this producer trip so dK can consume it
                        # while Q/LSE/dO/dPsum for the next tile are loading.
                        m_st_qcol = T.meta_var(m_st_next - BLK_M)
                        tma_wait_qcol_token = iket.range_start("tma-wait-qcol")
                        qcol_consumed.wait(0, qcol_cons_ph.phase)
                        iket.range_end(tma_wait_qcol_token)
                        if T.cuda.elect_sync():
                            tma_g2s(
                                4, Q_col.ptr_to([0, 0]), tma_qcol_cta0.ptr_to([0]),
                                T.address_of(q_col_tensormap),
                                q_col_st_next, m_st_qcol, h_idx, b_idx,
                            )
                            if id_in_pair == 0:
                                tma_qcol.arrive(0, QCOL_BATCH_BYTES)
                        qcol_cons_ph.advance()

                        tma_wait_q_token = iket.range_start("tma-wait-q")
                        q_consumed.wait(0, q_cons_ph.phase)
                        iket.range_end(tma_wait_q_token)
                        if T.cuda.elect_sync():
                            tma_g2s(
                                5, Q_row.ptr_to([0, 0, 0]), tma_q_cta0.ptr_to([0]),
                                T.address_of(q_row_tensormap),
                                0, q_row_st_next, 0, h_idx, b_idx,
                            )
                            if id_in_pair == 0:
                                tma_q.arrive(0, Q_BATCH_BYTES)
                        q_cons_ph.advance()
                        tma_wait_lse_token = iket.range_start("tma-wait-lse")
                        lse_consumed.wait(0, lse_cons_ph.phase)
                        iket.range_end(tma_wait_lse_token)
                        if T.cuda.elect_sync():
                            tma_lse.arrive(0, LSE_BYTES)
                            bulk_g2s_cta(
                                sLSE.ptr_to([0, 0]),
                                LSE_g.ptr_to([b_idx, h_idx, m_st_next]),
                                LSE_BYTES, tma_lse.ptr_to([0]),
                            )
                        lse_cons_ph.advance()
                        tma_wait_a_token = iket.range_start("tma-wait-a")
                        buf_a_consumed.wait(0, a_cons_ph.phase)
                        iket.range_end(tma_wait_a_token)
                        do_col_st_next = T.meta_var(id_in_pair * B_N_COL)
                        if T.cuda.elect_sync():
                            tma_g2s(
                                5, dO_row.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]),
                                T.address_of(do_row_tensormap),
                                0, q_row_st_next, 0, h_idx, b_idx,
                            )
                            tma_g2s(
                                4, dO_col.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]),
                                T.address_of(do_col_tensormap),
                                do_col_st_next, m_st_next, h_idx, b_idx,
                            )
                            if id_in_pair == 0:
                                tma_a.arrive(0, DO_BATCH_BYTES)
                        a_cons_ph.advance()
                        tma_wait_dpsum_token = iket.range_start("tma-wait-dpsum")
                        dpsum_consumed.wait(0, dpsum_cons_ph.phase)
                        iket.range_end(tma_wait_dpsum_token)
                        if T.cuda.elect_sync():
                            tma_dpsum.arrive(0, DPSUM_BYTES)
                            bulk_g2s_cta(
                                sDPsum.ptr_to([0]),
                                dpsum_g.ptr_to([b_idx, h_idx, m_st_next]),
                                DPSUM_BYTES, tma_dpsum.ptr_to([0]),
                            )
                        dpsum_cons_ph.advance()

                        iket.range_end(tma_prefetch_token)

                    # The producer loop issues Qt for tiles [0, N-2].  Fill
                    # the final tile for the dK tail after its predecessor has
                    # released the single shared-memory buffer.
                    tma_prefetch_token = iket.range_start("tma-prefetch")
                    tma_wait_qcol_token = iket.range_start("tma-wait-qcol")
                    qcol_consumed.wait(0, qcol_cons_ph.phase)
                    iket.range_end(tma_wait_qcol_token)
                    m_st_qcol_tail = T.meta_var(
                        (m_tile_start + num_m_tiles_this_n - 1) * BLK_M
                    )
                    q_col_st_tail = T.meta_var(id_in_pair * B_N_COL)
                    if T.cuda.elect_sync():
                        tma_g2s(
                            4, Q_col.ptr_to([0, 0]), tma_qcol_cta0.ptr_to([0]),
                            T.address_of(q_col_tensormap),
                            q_col_st_tail, m_st_qcol_tail, h_idx, b_idx,
                        )
                        if id_in_pair == 0:
                            tma_qcol.arrive(0, QCOL_BATCH_BYTES)
                    qcol_cons_ph.advance()
                    iket.range_end(tma_prefetch_token)

                tma_n_tile()

            elif warp_id == 0 and id_in_pair == 0:
                # ---- MMA warp (leader CTA only, TMEM accumulation for dV/dK) ----
                # One real shared pointer anchors the whole SMEMPool backing;
                # every other start field is the corresponding view offset
                # relative to that anchor.  No shared.dyn base is assumed.
                q_row_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                    Q_row.ptr_to([0, 0, 0])
                )
                q_row_start: T.uint64 = T.cast(
                    T.shift_right(q_row_addr, T.uint32(4)), "uint64"
                )
                desc_k_row = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_SS_LDO_1024,
                    q_row_start,
                    (K_smem.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                desc_q_row = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_SS_LDO_512, q_row_start, 0
                )
                desc_v_row = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_SS_LDO_1024,
                    q_row_start,
                    (V_smem.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                desc_do_row = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_SS_LDO_512,
                    q_row_start,
                    (dO_row.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                desc_q_col = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_TS,
                    q_row_start,
                    (Q_col.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                desc_do_col = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_TS,
                    q_row_start,
                    (dO_col.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                desc_k_col = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_TS,
                    q_row_start,
                    (K_col.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                desc_ds_exch = matrix_desc_from_anchor(
                    MATRIX_DESC_F16_TS,
                    q_row_start,
                    (dS_exch.elem_offset - Q_row.elem_offset) * DTYPE_SIZE,
                )
                kv_ph = PipelineState(1)
                kv_ph.init(0)
                q_ph = PipelineState(1)
                q_ph.init(0)
                qcol_ph = PipelineState(1)
                qcol_ph.init(0)
                a_ph = PipelineState(1)
                a_ph.init(0)
                wg0_ph = PipelineState(1)         # wg02mma_tmem: dS in TMEM → Phase D
                wg0_ph.init(0)               # first Phase D blocks
                wg0_smem_ph = PipelineState(1)  # peer DSMEM arrival → Phase E
                wg0_smem_ph.init(0)          # first Phase E blocks
                strip_ready_ph = PipelineState(1)
                strip_ready_ph.init(0)
                s_tmem_consumed_ph = PipelineState(1)
                s_tmem_consumed_ph.init(0)
                dq_tmem_free_ph = PipelineState(1)
                dq_tmem_free_ph.init(1)

                accum_var: T.int32
                accum_dv: T.int32
                accum_dk: T.int32

                @T.inline
                def mma_n_tile():
                    tma_kv.wait(0, kv_ph.phase)
                    kv_ph.advance()
                    accum_dv = 0
                    accum_dk = 0

                    # ---- Special first M-tile (i=0): A, B, C only ----
                    # Phase A[0]: S = K @ Q_row^T, M=256 (128/CTA)
                    mma_s_token = iket.range_start("mma-s")
                    tma_q.wait(0, q_ph.phase)
                    q_ph.advance()
                    accum_var = 0
                    if T.cuda.elect_sync():
                        mma_s(TMEM_OFF_A, accum_var, desc_k_row, desc_q_row)
                    if T.cuda.elect_sync():
                        mma2wg0_s.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                        tcgen05_commit(
                            q_consumed.ptr_to([0]),
                            pair_mask,
                        )
                    iket.range_end(mma_s_token)

                    # Phase B[0]: dP = V @ dO_row^T
                    mma_dp_token = iket.range_start("mma-dp")
                    tma_a.wait(0, a_ph.phase)
                    a_ph.advance()
                    accum_var = 0
                    if T.cuda.elect_sync():
                        mma_dp(TMEM_OFF_DP, accum_var, desc_v_row, desc_do_row)
                    if T.cuda.elect_sync():
                        mma2wg0_dp.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                    iket.range_end(mma_dp_token)

                    # Phase C[0]: dV += P^T @ dO_col, wait P strips ready
                    mma_dv_token = iket.range_start("mma-dv")
                    strip_ready.wait(0, strip_ready_ph.phase)
                    strip_ready_ph.advance()
                    if T.cuda.elect_sync():
                        mma_dv(TMEM_OFF_B, TMEM_OFF_A * 2, accum_dv, desc_do_col)
                    accum_dv = 1
                    if T.cuda.elect_sync():
                        tcgen05_commit(buf_a_consumed.ptr_to([0]), pair_mask)
                    iket.range_end(mma_dv_token)

                    # ---- Main loop: M-tiles 1..N-1 ----
                    for i_m_inner in T.serial(num_m_tiles_this_n - 1, annotations={"disable_unroll": True}):
                        i_m = T.meta_var(m_tile_start + i_m_inner + 1)

                        # Phase A[i_m]: S = K @ Q_row^T.  From the second
                        # trip onward, this waits until the reducer has drained
                        # the previous dQ tile that aliases the upper half of S.
                        mma_s_token = iket.range_start("mma-s")
                        tma_q.wait(0, q_ph.phase)
                        q_ph.advance()
                        dq_tmem_free.wait(0, dq_tmem_free_ph.phase)
                        dq_tmem_free_ph.advance()
                        accum_var = 0
                        if T.cuda.elect_sync():
                            mma_s(TMEM_OFF_A, accum_var, desc_k_row, desc_q_row)
                        if T.cuda.elect_sync():
                            mma2wg0_s.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                            tcgen05_commit(
                                q_consumed.ptr_to([0]),
                            pair_mask,
                            )
                        iket.range_end(mma_s_token)

                        # Phase D[i_m-1]: dK += dS_tmem @ Q_col^T.  Once this
                        # instruction is issued, ordered UMMA execution permits
                        # the following dP to reuse the dS/dP TMEM region.
                        mma_dk_token = iket.range_start("mma-dk")
                        wg02mma_tmem.wait(0, wg0_ph.phase)
                        wg0_ph.advance()
                        tma_qcol.wait(0, qcol_ph.phase)
                        qcol_ph.advance()
                        if T.cuda.elect_sync():
                            mma_dk(TMEM_OFF_C, TMEM_OFF_DP, accum_dk, desc_q_col)
                        if T.cuda.elect_sync():
                            tcgen05_commit(qcol_consumed.ptr_to([0]), pair_mask)
                        accum_dk = 1
                        iket.range_end(mma_dk_token)

                        # Phase B[i_m]: dP = V @ dO_row^T.
                        mma_dp_token = iket.range_start("mma-dp")
                        tma_a.wait(0, a_ph.phase)
                        a_ph.advance()
                        accum_var = 0
                        if T.cuda.elect_sync():
                            mma_dp(TMEM_OFF_DP, accum_var, desc_v_row, desc_do_row)
                        if T.cuda.elect_sync():
                            mma2wg0_dp.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                        iket.range_end(mma_dp_token)

                        # Phase E[i_m-1]: dQ = dS_exch^T @ K_col.
                        mma_dq_ready_token = iket.range_start("mma-dq-ready-wait")
                        wg02mma.wait(0, wg0_smem_ph.phase)
                        wg0_smem_ph.advance()
                        iket.range_end(mma_dq_ready_token)
                        # dQ aliases the upper half of S.  As in FA4's
                        # delayed dS producer commit, require both the current
                        # dS exchange and the next tile's drained S loads.
                        mma_dq_alias_token = iket.range_start("mma-dq-alias-wait")
                        s_tmem_consumed.wait(0, s_tmem_consumed_ph.phase)
                        s_tmem_consumed_ph.advance()
                        iket.range_end(mma_dq_alias_token)
                        mma_dq_issue_token = iket.range_start("mma-dq-issue")
                        accum_var = 0
                        if T.cuda.elect_sync():
                            mma_dq(TMEM_OFF_DQ, accum_var, desc_ds_exch, desc_k_col)
                        if T.cuda.elect_sync():
                            mma2wg0_dq.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                            tcgen05_commit(
                                ds_exch_consumed.ptr_to([0]),
                            pair_mask,
                            )
                        iket.range_end(mma_dq_issue_token)

                        # Phase C[i_m]: dV += P_tmem @ dO_col^T
                        mma_dv_token = iket.range_start("mma-dv")
                        strip_ready.wait(0, strip_ready_ph.phase)
                        strip_ready_ph.advance()
                        if T.cuda.elect_sync():
                            mma_dv(TMEM_OFF_B, TMEM_OFF_A * 2, accum_dv, desc_do_col)
                        if T.cuda.elect_sync():
                            tcgen05_commit(buf_a_consumed.ptr_to([0]), pair_mask)
                        iket.range_end(mma_dv_token)

                    # dV is complete before the remaining dK/dQ tail.  Match
                    # sdKVaccum stage 0 by releasing all compute warps now so
                    # its epilogue overlaps the tail below.
                    if T.cuda.elect_sync():
                        tcgen05_commit(
                            dv_done.ptr_to([0]),
                            pair_mask,
                        )

                    # ---- After loop: Phase D[N-1], Phase E[N-1] ----
                    mma_dk_token = iket.range_start("mma-dk")
                    wg02mma_tmem.wait(0, wg0_ph.phase)
                    wg0_ph.advance()
                    tma_qcol.wait(0, qcol_ph.phase)
                    qcol_ph.advance()
                    if T.cuda.elect_sync():
                        mma_dk(TMEM_OFF_C, TMEM_OFF_DP, accum_dk, desc_q_col)
                    if T.cuda.elect_sync():
                        tcgen05_commit(qcol_consumed.ptr_to([0]), pair_mask)
                        # sdKVaccum stage 1: release dK independently after its
                        # final update, while the final dQ path remains live.
                        tcgen05_commit(
                            dk_done.ptr_to([0]),
                            pair_mask,
                        )
                    iket.range_end(mma_dk_token)

                    # The final dQ uses the same TMEM destination as the
                    # preceding iteration.  Unlike the next loop trip, there
                    # is no following Phase-A prologue to carry this release
                    # wait, so consume it explicitly before the last write.
                    dq_tmem_free.wait(0, dq_tmem_free_ph.phase)
                    dq_tmem_free_ph.advance()
                    mma_dq_ready_token = iket.range_start("mma-dq-ready-wait")
                    wg02mma.wait(0, wg0_smem_ph.phase)
                    wg0_smem_ph.advance()
                    iket.range_end(mma_dq_ready_token)
                    mma_dq_issue_token = iket.range_start("mma-dq-issue")
                    accum_var = 0
                    if T.cuda.elect_sync():
                        mma_dq(TMEM_OFF_DQ, accum_var, desc_ds_exch, desc_k_col)
                    if T.cuda.elect_sync():
                        mma2wg0_dq.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask)
                        tcgen05_commit(
                            ds_exch_consumed.ptr_to([0]),
                            pair_mask,
                        )
                    iket.range_end(mma_dq_issue_token)

                    # Consume the final reducer release before TMEM teardown.
                    dq_tmem_free.wait(0, dq_tmem_free_ph.phase)

                mma_n_tile()

            elif warp_id == 2:
                # Relay the per-CTA DSMEM completion to one leader-local
                # barrier, keeping the MMA warp off the cross-CTA wait path.
                relay_ph = PipelineState(1)
                relay_ph.init(0)

                @T.inline
                def relay_n_tile():
                    for _ in T.serial(num_m_tiles_this_n, annotations={"disable_unroll": True}):
                        ds_exch_mbar.wait(0, relay_ph.phase)
                        relay_ph.advance()
                        if T.cuda.elect_sync():
                            wg02mma.arrive(0, remote=0, pred=True)

                relay_n_tile()

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
                T.ptx.bar.sync(T.uint32(5), 416)
                tmem_dealloc_mbar.arrive(
                    0, remote=1 - id_in_pair, pred=True
                )
                tmem_dealloc_mbar.wait(0, 0)
                T.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](
                    tmem_addr[0], T.uint32(512)
                )

        # ==============================================================
        # WG1+WG2: softmax grad + dS + split epilogue
        # ==============================================================
        elif (wg_id >= 1) & (wg_id <= 2):
            T.ptx.setmaxnreg.inc.sync.aligned.u32(136)
            T.ptx.bar.sync(T.uint32(5), 416)
            compute_wg = T.meta_var(wg_id - 1)
            gemm_s_ph = PipelineState(1)
            gemm_s_ph.init(0)
            lse_ph = PipelineState(1)
            lse_ph.init(0)
            ds_exch_ph = PipelineState(1)
            ds_exch_ph.init(0)  # first wait blocks until DSMEM arrives
            gemm_dp_ph = PipelineState(1)
            gemm_dp_ph.init(0)
            dpsum_ph = PipelineState(1)
            dpsum_ph.init(0)
            dv_done_ph = PipelineState(1)
            dv_done_ph.init(0)
            dk_done_ph = PipelineState(1)
            dk_done_ph.init(0)
            ds_exch_consumed_ph = PipelineState(1)
            ds_exch_consumed_ph.init(1)

            # Swizzle offsets for dS_smem and epilogue D_smem writes
            dS_sw = RowiseSwizzleOffset(3, 3, 3, warp_id * 32 + lane_id, prefix="dS_sw")
            dS_sw.init()
            epi_sw = RowiseSwizzleOffset(3, 3, 3, warp_id * 32 + lane_id, prefix="epi_sw")
            epi_sw.init()

            # Strip offset: WG0 → strip 0 (cols 0:64), WG1 → strip 1 (cols 64:128)
            strip_off = T.meta_var(compute_wg * STRIP_SIZE)

            @T.inline
            def softmax_n_tile():
                for i_m_inner in T.serial(num_m_tiles_this_n, annotations={"disable_unroll": True}):
                    i_m = T.meta_var(m_tile_start + i_m_inner)
                    m_st_val = T.meta_var(i_m * BLK_M)
                    row_local = T.meta_var(warp_id * 32 + lane_id)

                    # LSE is an independent CTA-local stream: it may be
                    # consumed and released as soon as P has been formed,
                    # while Q_row is released by the Phase-A MMA commit.
                    softmax_p_token = iket.range_start("softmax-p")
                    tma_lse.wait(0, lse_ph.phase)

                    # ---- Wait Phase A: S^T ready in TMEM ----
                    mma2wg0_s.wait(0, gemm_s_ph.phase)
                    gemm_s_ph.advance()

                    S_strip = T.alloc_local((STRIP_SIZE,), f32)

                    # Read strip: S^T cols [strip_off : strip_off+64]
                    tmem_s_col = T.meta_var(TMEM_OFF_A + strip_off)
                    for stage in T.unroll(STRIP_SIZE // 32):
                        tmem_load_32(
                            S_strip, stage * 32, TMEM_OFF_A, 0,
                            tmem_s_col + stage * 32,
                        )
                    # D=128 aliases dQ with the upper half of S.  Match FA4's
                    # delayed dS commit: publish the previous tile's dQ-safe
                    # token immediately after this warp has drained the next
                    # tile's S load, before doing any P arithmetic.
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    if (i_m_inner > 0) & (lane_id == 0):
                        s_tmem_consumed.arrive(0, remote=0, pred=True)

                    # Compute P^T = exp2(S^T * scale_log2 - LSE_log2[m]).
                    # The base conversion is hoisted to preprocess.
                    P_f16 = T.alloc_local((STRIP_SIZE,), f16)
                    P_f16_u32 = P_f16.view("uint32")
                    tmem_p_col = T.meta_var(TMEM_OFF_A * 2 + strip_off)
                    for stage in T.unroll(STRIP_SIZE // 32):
                        for j_inner in T.unroll(32 // 2):
                            j = T.meta_var(stage * (32 // 2) + j_inner)
                            scaled_pair: T.let = fma_scale_sub_f32x2(
                                T.cuda.make_float2(S_strip[2 * j], S_strip[2 * j + 1]),
                                T.cuda.make_float2(T.float32(scale_log2), T.float32(scale_log2)),
                                T.cuda.make_float2(
                                    sLSE[0, strip_off + 2 * j],
                                    sLSE[0, strip_off + 2 * j + 1],
                                ),
                            )
                            S_strip[2 * j] = T.cuda.float2_x(scaled_pair)
                            S_strip[2 * j + 1] = T.cuda.float2_y(scaled_pair)
                            T.ptx.ex2.approx.ftz.f32(S_strip[2 * j], S_strip[2 * j])
                            T.ptx.ex2.approx.ftz.f32(S_strip[2 * j + 1], S_strip[2 * j + 1])
                        # Only the first two Q tiles that overlap this
                        # 256-row K/V cluster can intersect the causal
                        # diagonal.  Hoist the uniform branch outside the
                        # unrolled element loop so later tiles pay one branch
                        # per 32-column stage rather than 16 branches.
                        if causal:
                            if i_m < m_tile_start + BLK_N // BLK_M:
                                key_idx = T.meta_var(n_st_cta + row_local)
                                for j_inner in T.unroll(32 // 2):
                                    j = T.meta_var(stage * (32 // 2) + j_inner)
                                    query_idx_0 = T.meta_var(
                                        m_st_val + strip_off + 2 * j
                                    )
                                    query_idx_1 = T.meta_var(query_idx_0 + 1)
                                    S_strip[2 * j] = T.if_then_else(
                                        query_idx_0 >= key_idx,
                                        S_strip[2 * j],
                                        T.float32(0),
                                    )
                                    S_strip[2 * j + 1] = T.if_then_else(
                                        query_idx_1 >= key_idx,
                                        S_strip[2 * j + 1],
                                        T.float32(0),
                                    )
                        for j_inner in T.unroll(32 // 2):
                            j = T.meta_var(stage * (32 // 2) + j_inner)
                            cast_f32x2_to_f16x2(
                                T.address_of(P_f16[2 * j]),
                                T.address_of(S_strip[2 * j]),
                            )

                        # P aliases S in TMEM.  Once the first 32-column
                        # register stage is ready, drain both S loads and
                        # synchronize the compute warps before the first store.
                        if stage == 0:
                            T.ptx.bar.sync(T.uint32(8), 256)

                        tmem_store_16(
                            P_f16_u32, stage * 16, TMEM_OFF_A, 0,
                            (tmem_p_col + stage * 32) // 2,
                        )
                    lse_ph.advance()
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx.bar.sync(T.uint32(8), 256)
                    # Bridge the generic LSE reads into the async proxy, then
                    # publish the eight compute warps' release only after the
                    # existing rendezvous has ordered every reader.
                    if lane_id == 0:
                        lse_consumed.arrive(0)

                    # One elected thread per warp contributes to the shared
                    # P-ready barrier in the leader CTA.
                    if T.cuda.elect_sync():
                        strip_ready.arrive(0, remote=0, pred=True)
                    iket.range_end(softmax_p_token)

                    # ---- Phase B: wait for dPsum and dP^T, read strip ----
                    softmax_ds_token = iket.range_start("softmax-ds")
                    tma_dpsum.wait(0, dpsum_ph.phase)
                    mma2wg0_dp.wait(0, gemm_dp_ph.phase)
                    gemm_dp_ph.advance()

                    dP_strip = T.alloc_local((STRIP_SIZE,), f32)
                    # .f32x2 writes its destination as one packed 64-bit
                    # register, so address the same storage in pairs.
                    dP_pairs = dP_strip.view("uint64")
                    tmem_dp_col = T.meta_var(TMEM_OFF_DP + strip_off)
                    for stage in T.unroll(STRIP_SIZE // 32):
                        tmem_load_32(
                            dP_strip, stage * 32, TMEM_OFF_A, 0,
                            tmem_dp_col + stage * 32,
                        )

                    # dS^T[n,m] = P^T[n,j] * (dP^T[n,j] - dpsum[m]).
                    for j in T.unroll(STRIP_SIZE // 2):
                        T.ptx.sub.rn.ftz.f32x2(
                            dP_pairs[j],
                            T.cuda.make_float2(dP_strip[2 * j], dP_strip[2 * j + 1]),
                            T.cuda.make_float2(
                                sDPsum[strip_off + 2 * j],
                                sDPsum[strip_off + 2 * j + 1],
                            ),
                        )
                        T.ptx.mul.rn.ftz.f32x2(
                            dP_pairs[j],
                            T.cuda.make_float2(S_strip[2 * j], S_strip[2 * j + 1]),
                            T.cuda.make_float2(dP_strip[2 * j], dP_strip[2 * j + 1]),
                        )

                    dpsum_ph.advance()

                    T.ptx.tcgen05.wait__ld.sync.aligned()

                    dS_full_f16 = T.alloc_local((STRIP_SIZE,), f16)
                    for j in T.unroll(STRIP_SIZE // 2):
                        cast_f32x2_to_f16x2(
                            T.address_of(dS_full_f16[2 * j]),
                            T.address_of(dP_strip[2 * j]),
                        )

                    T.ptx.bar.sync(T.uint32(8), 256)

                    tmem_ds_col = T.meta_var(TMEM_OFF_DP * 2 + strip_off)
                    dS_f16_u32 = dS_full_f16.view("uint32")
                    tmem_store_32(
                        dS_f16_u32, 0, TMEM_OFF_A, 0,
                        tmem_ds_col // 2,
                    )

                    T.ptx.tcgen05.wait__st.sync.aligned()

                    # The dK MMA only consumes dS from TMEM.  Release that
                    # dependency before the independent SMEM visibility,
                    # cross-WG rendezvous, and DSMEM exchange path.
                    if T.cuda.elect_sync():
                        wg02mma_tmem.arrive(0, remote=0, pred=True)
                    iket.range_end(softmax_ds_token)

                    # The DSMEM path is independent of dK.  Materialize its
                    # local/outbound half only after the TMEM consumer has
                    # been released.  The buffer itself is single-stage, so
                    # wait until the preceding dQ MMA has stopped reading it
                    # before publishing the next tile.
                    ds_exchange_token = iket.range_start("ds-exchange")
                    ds_exch_consumed.wait(0, ds_exch_consumed_ph.phase)
                    # The empty barrier orders TCGEN completion, while this
                    # proxy fence bridges the completed async-proxy read to
                    # the generic stores that reuse dS_exch.
                    T.ptx.fence.proxy.async_.shared__cta()
                    ds_exch_consumed_ph.advance()
                    if compute_wg == id_in_pair:
                        ds_row_base = T.meta_var(id_in_pair * CTA_N)
                        ds_row_st: T.int32
                        ds_row_st = (
                            (ds_row_base + row_local) * B_N
                            + ((ds_row_base + row_local) & 7) * 8
                        )
                        for ni in T.unroll(STRIP_SIZE // 8):
                            copy_128b(
                                pointer_offset(
                                    dS_exch.ptr_to([0, 0]),
                                    ds_row_st + dS_sw.apply(ni * 8),
                                ),
                                dS_full_f16.view("uint128")[ni],
                            )
                    else:
                        stage_row_st: T.int32
                        stage_row_st = row_local * B_N + (row_local & 7) * 8
                        for ni in T.unroll(STRIP_SIZE // 8):
                            copy_128b(
                                pointer_offset(
                                    dS_send.ptr_to([0, 0]),
                                    stage_row_st + dS_sw.apply(ni * 8),
                                ),
                                dS_full_f16.view("uint128")[ni],
                            )

                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx.bar.sync(T.uint32(8), 256)

                    # This fence/rendezvous bridges both the dS stores used by
                    # the DSMEM async copy and the completed generic dPsum
                    # reads.  Release the CTA-local dPsum buffer only after all
                    # eight compute warps have crossed it.
                    if lane_id == 0:
                        dpsum_consumed.arrive(0)

                    # The sender WG starts the peer copy directly.  Its source
                    # has independent storage, so the load warp can reuse dO
                    # as soon as Phase C commits, without a relay-warp drain.
                    if (compute_wg != id_in_pair) & (warp_id == 0) & (lane_id == 0):
                        peer_cta: T.int32
                        peer_cta = 1 - id_in_pair
                        ds_copy_bytes = T.meta_var(CTA_N * B_N * DTYPE_SIZE)
                        # mapa writes its result, so the peer-window addresses
                        # are computed into registers first; both the arrival
                        # and the copy then name the peer's shared window.
                        remote_mbar = T.alloc_local([1], "uint32")
                        T.ptx.mapa.shared__cluster.u32(
                            remote_mbar[0],
                            T.cuda.cvta_generic_to_shared(ds_exch_mbar.ptr_to([0])),
                            T.uint32(peer_cta),
                        )
                        remote_dst = T.alloc_local([1], "uint32")
                        T.ptx.mapa.shared__cluster.u32(
                            remote_dst[0],
                            T.cuda.cvta_generic_to_shared(
                                dS_exch.ptr_to([id_in_pair * CTA_N, 0])
                            ),
                            T.uint32(peer_cta),
                        )
                        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
                            remote_mbar[0], T.uint32(ds_copy_bytes), pred=True
                        )
                        T.ptx[_BULK_S2C](
                            remote_dst[0],
                            dS_send.ptr_to([0, 0]),
                            T.uint32(ds_copy_bytes),
                            remote_mbar[0],
                        )
                    iket.range_end(ds_exchange_token)

                # ---- Two-stage dKV epilogue ----
                # Both compute WGs first split dV by 64-column halves, then
                # split dK the same way.  Stage 0 overlaps the MMA dK/dQ tail.
                dkv_epilogue_token = iket.range_start("dkv-epilogue")
                dv_done.wait(0, dv_done_ph.phase)
                dv_done_ph.advance()
                dv_epi_strip = T.alloc_local((EPI_N,), f32)
                dv_epi_f16 = T.alloc_local((EPI_N,), f16)
                tmem_load_64(
                    dv_epi_strip, 0, TMEM_OFF_A, 0,
                    TMEM_OFF_B + compute_wg * EPI_N,
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                for j in T.unroll(EPI_N // 2):
                    cast_f32x2_to_f16x2(
                        T.address_of(dv_epi_f16[2 * j]),
                        T.address_of(dv_epi_strip[2 * j]),
                    )
                epi_row_st: T.int32
                epi_row_st = (
                    warp_id * 32 + lane_id
                ) * EPI_N + ((warp_id * 32 + lane_id) & 7) * 8
                for ni in T.unroll(EPI_N // 8):
                    copy_128b(
                        pointer_offset(
                            dV_epi.ptr_to([compute_wg, 0, 0]),
                            epi_row_st + epi_sw.apply(ni * 8),
                        ),
                        dv_epi_f16.view("uint128")[ni],
                    )
                T.ptx.fence.proxy.async_.shared__cta()
                T.ptx.bar.sync(T.uint32(wg_id + 10), 128)
                if (warp_id == 0) & (lane_id == 0):
                    tma_s2g(
                        4,
                        dV_epi.ptr_to([compute_wg, 0, 0]),
                        T.address_of(dv_tensormap),
                        compute_wg * EPI_N,
                        n_st_cta,
                        h_idx,
                        b_idx,
                    )
                if warp_id == 0:
                    T.ptx.bar.arrive(T.uint32(wg_id + 10), 160)
                T.ptx.fence.proxy.async_.shared__cta()
                T.ptx.bar.sync(T.uint32(wg_id + 10), 160)

                dk_done.wait(0, dk_done_ph.phase)
                dk_done_ph.advance()
                dk_epi_strip = T.alloc_local((EPI_N,), f32)
                dk_epi_pairs = dk_epi_strip.view("uint64")
                dk_epi_f16 = T.alloc_local((EPI_N,), f16)
                tmem_load_64(
                    dk_epi_strip, 0, TMEM_OFF_A, 0,
                    TMEM_OFF_C + compute_wg * EPI_N,
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                for j in T.unroll(EPI_N // 2):
                    T.ptx.mul.rn.ftz.f32x2(
                        dk_epi_pairs[j],
                        T.cuda.make_float2(dk_epi_strip[2 * j], dk_epi_strip[2 * j + 1]),
                        T.cuda.make_float2(
                            T.float32(softmax_scale), T.float32(softmax_scale)
                        ),
                    )
                    cast_f32x2_to_f16x2(
                        T.address_of(dk_epi_f16[2 * j]),
                        T.address_of(dk_epi_strip[2 * j]),
                    )
                for ni in T.unroll(EPI_N // 8):
                    copy_128b(
                        pointer_offset(
                            dK_epi.ptr_to([compute_wg, 0, 0]),
                            epi_row_st + epi_sw.apply(ni * 8),
                        ),
                        dk_epi_f16.view("uint128")[ni],
                    )
                T.ptx.fence.proxy.async_.shared__cta()
                T.ptx.bar.sync(T.uint32(wg_id + 10), 128)
                if (warp_id == 0) & (lane_id == 0):
                    tma_s2g(
                        4,
                        dK_epi.ptr_to([compute_wg, 0, 0]),
                        T.address_of(dk_tensormap),
                        compute_wg * EPI_N,
                        n_st_cta,
                        h_idx,
                        b_idx,
                    )
                if warp_id == 0:
                    T.ptx.bar.arrive(T.uint32(wg_id + 10), 160)
                T.ptx.fence.proxy.async_.shared__cta()
                T.ptx.bar.sync(T.uint32(wg_id + 10), 160)
                # Keep the terminal dV/dK stores asynchronous, but place both
                # issues in one explicit bulk group so their lifetime remains
                # visible to NumSim/checkers.  Neither source tile is reused.
                if (warp_id == 0) & (lane_id == 0):
                    T.ptx.cp.async_.bulk.commit_group()
                iket.range_end(dkv_epilogue_token)

            softmax_n_tile()
            T.ptx.bar.arrive(T.uint32(5), 416)


        # ==============================================================
        # WG0: dQ reduce (TMEM → SMEM → TMA reduce)
        # ==============================================================
        else:
            T.ptx.setmaxnreg.inc.sync.aligned.u32(136)
            T.ptx.bar.sync(T.uint32(5), 416)
            gemm_dq_ph_wg3 = PipelineState(1)
            gemm_dq_ph_wg3.init(0)

            @T.inline
            def wg3_n_tile():
                for i_m_inner in T.serial(num_m_tiles_this_n, annotations={"disable_unroll": True}):
                    i_m = T.meta_var(m_tile_start + i_m_inner)
                    m_st_val = T.meta_var(i_m * BLK_M)
                    row_local = T.meta_var(warp_id * 32 + lane_id)

                    dq_reduce_token = iket.range_start("dq-reduce")
                    mma2wg0_dq.wait(0, gemm_dq_ph_wg3.phase)
                    gemm_dq_ph_wg3.advance()

                    # Datapath-B readback is a physical (128, 64) image of the
                    # logical (64, 128) dQ tile.
                    dQ_full = T.alloc_local((64,), f32)
                    tmem_load_32(
                        dQ_full, 32, TMEM_OFF_DQ, warp_id * 32, 0
                    )
                    tmem_load_32(
                        dQ_full, 0, TMEM_OFF_DQ, warp_id * 32, 32
                    )

                    # Drain dQ before releasing the shared dP/dQ region.
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    T.cuda.warp_sync()
                    if T.cuda.elect_sync():
                        dq_tmem_free.arrive(0, remote=0, pred=True)

                    m_st_cta = T.meta_var(m_st_val + id_in_pair * DQ_M_PER_CTA)
                    dq_reduce_elected: T.let = T.cuda.elect_sync()

                    for stage in T.unroll(DQ_REDUCE_ITERS):
                        dq_reduce_stage_token = iket.range_start("dq-reduce-stage")
                        smem_slot = T.meta_var(stage % DQ_STAGES)
                        dq_stage_st = smem_slot * BLK_M * DQ_RED_N
                        dq_reg_st = T.meta_var(
                            (
                                (stage + DQ_REDUCE_ITERS // 2)
                                % DQ_REDUCE_ITERS
                            )
                            * DQ_RED_N
                        )
                        for chunk in T.unroll(DQ_RED_N // 4):
                            copy_128b(
                                pointer_offset_f32(
                                    dQ_smem.ptr_to([0, 0, 0]),
                                    dq_stage_st
                                    + chunk * BLK_M * 4
                                    + row_local * 4,
                                ),
                                dQ_full.view("uint128")[(dq_reg_st + chunk * 4) // 4],
                            )
                        T.ptx.fence.proxy.async_.shared__cta()
                        T.cuda.warpgroup_sync(4)

                        if warp_id == 0:
                            if dq_reduce_elected:
                                tma_s2g_reduce(
                                    4,
                                    dQ_smem.ptr_to([smem_slot, 0, 0]),
                                    T.address_of(dq_tensormap),
                                    "add",
                                    0,
                                    m_st_cta + stage * DQ_ROWS_PER_STAGE,
                                    h_idx,
                                    b_idx,
                                )
                            T.ptx.cp.async_.bulk.commit_group()
                            T.ptx.cp.async_.bulk.wait_group(DQ_STAGES - 1)
                        T.cuda.warpgroup_sync(4)
                        iket.range_end(dq_reduce_stage_token)
                    iket.range_end(dq_reduce_token)

                # Wait for all pending TMA reduces
                if warp_id == 0:
                    T.ptx.cp.async_.bulk.wait_group(0)
                T.cuda.warpgroup_sync(4)

            wg3_n_tile()
            T.ptx.bar.arrive(T.uint32(5), 416)
    # fmt: on
    return kernel


# ---------------------------------------------------------------------------
# setup() — compile all kernels, run once, return kernel_fn
# ---------------------------------------------------------------------------


def setup(data, B, H, S, D):
    """Compile backward kernels, prepare data, run once. Return no-arg kernel_fn."""
    Q = data["Q"]
    K = data["K"]
    V = data["V"]
    O = data["O"]
    dO = data["dO"]
    LSE = data["LSE"]
    causal = bool(data.get("causal", False))
    attention_scale = float(data.get("softmax_scale", 1.0 / math.sqrt(D)))

    dpsum = torch.empty(B, H, S, dtype=torch.float32, device="cuda")
    LSE_log2 = torch.empty_like(LSE)
    dQ_acc = torch.empty(B, H, S, D, dtype=torch.float32, device="cuda")
    dK = data["dK"]
    dV = data["dV"]
    dQ = data["dQ"]
    target = tvm.target.Target("cuda")

    preprocess_ex = build_preprocess(B, S, H, D)
    cast_ex = build_cast_f32_to_f16(B, S, H, D, attention_scale)

    with target:
        kernel_func = build_kernel(B, H, S, D, causal=causal, attention_scale=attention_scale)
        kernel_mod = tvm.IRModule({"main": kernel_func})
        kernel_ex = tvm.compile(kernel_mod, target=target, tir_pipeline="tirx")

    def run_all():
        preprocess_ex(dO, O, LSE, dpsum, LSE_log2, dQ_acc)
        kernel_ex(Q, K, V, dO, LSE_log2, dpsum, dK, dV, dQ_acc)
        cast_ex(dQ_acc, dQ)

    run_all()

    data["dQ"] = dQ
    data["dK"] = dK
    data["dV"] = dV

    def kernel_fn():
        run_all()

    return kernel_fn


KERNEL_META = {
    "name": "flash_attention_backward_sm100",
    "category": "attention",
    "compute_capability": 10,
}

CONFIGS = [
    {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": 16,
        "head_dim": 128,
        "is_causal": is_causal,
        "label": (f"b{batch_size}_s{seq_len}_h16_{'causal' if is_causal else 'noncausal'}"),
    }
    for batch_size, seq_len, is_causal in (
        (1, 2048, True),
        (1, 4096, True),
        (2, 4096, True),
        (1, 8192, True),
        (1, 8192, False),
        # flash-attn's own sweep has no fixed config list -- benchmark_attn.py
        # takes every shape from the command line -- so "the official shape" is
        # its default: --headdim (128,128), --seqlen 8192, --nheads unset (16
        # for head_dim <= 192), --total-seqlen 32k giving batch 32k // 8192,
        # and --causal both. These two rows are that default. We still generate
        # fp16 where it generates bf16.
        (4, 8192, True),
        (4, 8192, False),
    )
]


def get_kernel(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Return the raw backward core PrimFunc for the registry configuration."""
    return build_kernel(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        causal=is_causal,
        attention_scale=1.0 / math.sqrt(head_dim),
    )


def _prepare_official_workload(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int, is_causal: bool
):
    """Create saved forward tensors and the current FA4 backward reference."""
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    torch.manual_seed(0)
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    k = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    v = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    dout = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.25).half()
    scale = 1.0 / math.sqrt(head_dim)
    with torch.no_grad():
        out, lse = _flash_attn_fwd(
            q=q, k=k, v=v, softmax_scale=scale, causal=is_causal, return_lse=True
        )[:2]
        expected = _flash_attn_bwd(
            q=q, k=k, v=v, out=out, dout=dout, lse=lse, softmax_scale=scale, causal=is_causal
        )
    data = {
        "Q": q,
        "K": k,
        "V": v,
        "O": out,
        "dO": dout,
        "LSE": lse,
        "dQ": torch.empty_like(q),
        "dK": torch.empty_like(k),
        "dV": torch.empty_like(v),
        "causal": is_causal,
        "softmax_scale": scale,
    }
    return data, expected


def run_test(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Compile the three-kernel pipeline and compare it with current FA4."""
    data, expected = _prepare_official_workload(batch_size, seq_len, num_heads, head_dim, is_causal)
    setup(data, batch_size, num_heads, seq_len, head_dim)
    torch.cuda.synchronize()
    for name, actual, reference in zip(
        ("dQ", "dK", "dV"), (data["dQ"], data["dK"], data["dV"]), expected, strict=True
    ):
        if not torch.isfinite(actual).all():
            raise AssertionError(f"{name} contains a non-finite value")
        matched = torch.isclose(actual, reference, rtol=0.1, atol=0.1).float().mean()
        if matched.item() < 0.995:
            raise AssertionError(f"{name} matched ratio {matched.item():.6f} is below 0.995")


def run_bench(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    warmup=None,
    repeat=None,
    timer=None,
    **kwargs,
):
    """Benchmark the full preprocess/core/cast pipeline against current FA4."""
    from flash_attn.cute.interface import _flash_attn_bwd

    from tvm.tirx.bench import bench

    data, _ = _prepare_official_workload(batch_size, seq_len, num_heads, head_dim, is_causal)
    candidate = setup(data, batch_size, num_heads, seq_len, head_dim)

    def official_factory():
        def run():
            return _flash_attn_bwd(
                q=data["Q"],
                k=data["K"],
                v=data["V"],
                out=data["O"],
                dout=data["dO"],
                lse=data["LSE"],
                softmax_scale=data["softmax_scale"],
                causal=data["causal"],
            )

        return run

    return bench(
        {"tir": candidate},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": official_factory},
        **kwargs,
    )
