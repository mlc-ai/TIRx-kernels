# This file is a TIRx port of code from flash-attention-fp4
# (https://github.com/hao-ai-lab/flash-attention-fp4 @ 5aa37a9680f7b76a11799b5f4846100ed5a3e6d8),
# Copyright (c) 2022, the respective contributors, as shown by
# licenses/AUTHORS.flash-attention.txt
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Block-scaled FP4/MXFP8 FlashAttention-4 forward (SM103) with K-owned launch, roles,
storage, and synchronization.

Upstream source: flash_attn/cute/flash_fwd_sm100_fp4.py (class
``FlashAttentionForwardSm100``), with the helpers it dispatches to:
flash_attn/cute/blackwell_helpers.py (``gemm_ptx_partial_fp4``, ``tmem_ld_red_max``,
``packed_float_to_e2m1`` / ``packed_float_to_ue4m3`` / ``packed_float_to_ue8m0``),
flash_attn/cute/softmax.py (``SoftmaxSm100``), flash_attn/cute/mma_sm100_desc.py,
flash_attn/cute/tile_scheduler.py, flash_attn/cute/block_info.py, flash_attn/cute/mask.py,
flash_attn/cute/pack_gqa.py, and the dispatch in flash_attn/cute/interface.py.

One traced kernel covers the six GB300 headline modes of the upstream README. ``qk_format``
selects NVFP4 (E2M1 data, E4M3 scale per 16) or MXFP8 (E4M3 data, E8M0 scale per 32) for Q and
K; ``pv_format`` selects a BF16, FP8 E4M3, NVFP4, or MXFP8 V (the last two also quantize the
softmax output P on the fly). Head dim 128 for every mode, head dim 64 additionally for the
NVFP4-QK modes with BF16 or FP8 V. GQA runs the unpacked path (``pack_gqa=False``): the upstream default packs Q heads, but
the fork's FP4 kernel is numerically wrong there, so both sides use one Q head per tile.
The SM103 defaults of the upstream tuning table are baked in: hardware ``tcgen05.ld.red``
row max, no exp2 emulation, log-domain NVFP4 P quantization.
"""

# K.kernel consumes concrete annotations; postponed annotations would stringify them.
import math
import os
from typing import Any

import tirx_kernels.kern as K

# ---------------------------------------------------------------------------------------------
# Static specialization (orig: FlashAttentionForwardSm100.__init__ / _setup_attributes)
# ---------------------------------------------------------------------------------------------

BLK_M = 128  # m_block_size (orig:L129)
BLK_N = 128  # n_block_size (orig:L130)
Q_STAGE = 2  # two Q tiles per CTA (orig:L166)
EPI_STAGE = 2  # orig:L370
TMEM_COLS = 512  # orig:L219-220
SMEM_BUDGET = 227 * 1024  # orig:L374
BUFFER_ALIGN = 1024  # orig:L275
NUM_WARPS = 16
SOFTMAX0_WARPS = (0, 1, 2, 3)  # orig:L210
SOFTMAX1_WARPS = (4, 5, 6, 7)  # orig:L211
CORRECTION_WARPS = (8, 9, 10, 11)  # orig:L213
MMA_WARP = 12  # orig:L215
EPILOGUE_WARP = 13  # orig:L216
LOAD_WARP = 14  # orig:L217
EMPTY_WARP = 15  # orig:L218

# (qk_format, pv_format) pairs that the upstream README benchmarks on GB300.
MODES = [
    ("nvfp4", "bf16"),
    ("nvfp4", "fp8"),
    ("nvfp4", "nvfp4"),
    ("nvfp4", "mxfp8"),
    ("mxfp8", "bf16"),
    ("mxfp8", "fp8"),
]

_QK_WIDTH = {"nvfp4": 4, "mxfp8": 8}
_QK_SF_VEC = {"nvfp4": 16, "mxfp8": 32}
_PV_WIDTH = {"bf16": 16, "fp8": 8, "nvfp4": 4, "mxfp8": 8}
_PV_SF_VEC = {"nvfp4": 16, "mxfp8": 32}  # only the quantized-PV formats carry P/V scales


def _align_up(x, a):
    return (x + a - 1) // a * a


def ceildiv(a, b):
    return (a + b - 1) // b


class Spec:
    """Everything ``__init__`` / ``_setup_attributes`` / ``__call__`` derive at trace time."""

    def __init__(self, qk_format, pv_format, head_dim, is_causal, num_qo_heads, num_kv_heads):
        if (qk_format, pv_format) not in MODES:
            raise ValueError(f"unsupported mode {qk_format}+{pv_format}; supported: {MODES}")
        if head_dim not in (64, 128):
            raise ValueError("head_dim must be 64 or 128")
        if head_dim == 64 and (qk_format, pv_format) not in (("nvfp4", "bf16"), ("nvfp4", "fp8")):
            # orig README: head_dim >= sf_vec_size * 4 for every block-scaled operand.
            raise ValueError("head_dim 64 is only supported for NVFP4 QK with BF16 or FP8 V")
        if num_qo_heads % num_kv_heads:
            raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
        self.qk_format = qk_format
        self.pv_format = pv_format
        self.head_dim = head_dim
        self.head_dim_v = head_dim
        self.is_causal = is_causal
        self.qhead_per_kvhead = num_qo_heads // num_kv_heads
        # The upstream interface defaults to pack_gqa=True for GQA (interface.py:432), but the
        # fork's FP4 kernel is numerically wrong on that path (scale-factor indexing; measured
        # cos_sim 0.94 vs fp64 where the unpacked path and the fork's BF16 kernel give 1.0000).
        # Both sides therefore run the unpacked GQA path (pack_gqa=False): one Q head per tile,
        # kv head = head // qhead_per_kvhead (orig:L2014-2016).
        self.pack_gqa = False
        # orig interface.py:955-960: persistent iff no causal/local/varlen/split.
        self.is_persistent = not is_causal

        self.quant_qk = True
        self.quant_pv = pv_format in _PV_SF_VEC
        self.q_width = self.k_width = _QK_WIDTH[qk_format]
        self.v_width = _PV_WIDTH[pv_format]
        self.o_width = 16
        self.sf_vec_size = _QK_SF_VEC[qk_format]
        self.sf_vec_size_pv = _PV_SF_VEC.get(pv_format, self.sf_vec_size)
        self.sf_e8m0 = qk_format == "mxfp8"  # E8M0 QK scales, else E4M3
        self.sf_pv_e8m0 = pv_format == "mxfp8"
        self.p_width = self.v_width  # P is produced in V's dtype (orig:L2771)

        # orig:L344-358 -- 256-bit MMA operand tile in K.
        self.mma_inst_tile_k = head_dim // (64 if self.q_width == 4 else 32)
        self.mma_inst_tile_k_pv = BLK_N // (64 if self.sf_vec_size_pv == 16 else 32)
        # K per tcgen05.mma instruction for each GEMM.
        self.qk_mma_k = 64 if self.q_width == 4 else 32
        self.pv_mma_k = {16: 16, 8: 32, 4: 64}[self.v_width]
        self.qk_k_tiles = head_dim // self.qk_mma_k
        self.pv_k_tiles = BLK_N // self.pv_mma_k

        # Register split (orig:L260-274). Warp 15 takes the warpgroup budget (48) instead of the
        # upstream 24 because setmaxnreg must be warpgroup-uniform under K.specialize.
        if head_dim < 96:
            self.num_regs_softmax, self.num_regs_correction, self.num_regs_other = 200, 64, 48
        else:
            self.num_regs_softmax, self.num_regs_correction = 192, 80
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        assert self.num_regs_other == 48

        # TMEM column map (orig:L243-258).
        self.tmem_s_offset = [0, BLK_N]
        self.tmem_o_offset = [2 * BLK_N + i * self.head_dim_v for i in range(Q_STAGE)]
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v
        assert self.tmem_total <= TMEM_COLS
        self.tmem_s_to_p_offset = BLK_N // 2
        self.tmem_p_offset = [s + self.tmem_s_to_p_offset for s in self.tmem_s_offset]
        self.tmem_vec_offset = self.tmem_s_offset

        self._setup_smem()
        self.p_split = self.mbar_p_split(self.pv_k_tiles)
        # TMEM P-store chunking (orig:L2790-2809): 32x32b repetition per tcgen05.st.
        p_cols = BLK_N * self.p_width // 32
        # bf16 P: repetition 16; every narrower P: 8. The upstream d<=64 fp8 default of 4 is
        # overridden by the pinned FA4_FP8_PV_TMEM_STORE_REP=8 (orig:L2790-2801).
        self.p_store_rep = 16 if self.v_width == 16 else 8
        self.p_store_chunks = p_cols // self.p_store_rep
        self.p_store_split = self.mbar_p_split(self.p_store_chunks)

    def mbar_p_split(self, k):
        """orig:L1105-1125 with the default FA4_*_P_SPLIT_NUM/DEN knobs."""
        if self.v_width == 8 and k > 1:
            return max(
                1, min(k - 1, k * 3 // 4)
            )  # FA4_FP8_PV_P_SPLIT_NUM/DEN = 3/4 at every head_dim
        if self.v_width > 8:
            return k // 4 * 3
        return max(1, min(k - 1, k * 1 // 2))

    def _setup_smem(self):
        """orig:_setup_attributes L369-424 (kv_stage from the 227 KB budget)."""
        d, dv = self.head_dim, self.head_dim_v
        smem_mbar = 512
        smem_tmem = 4
        smem_sScale = _align_up(Q_STAGE * BLK_M * 2 * 4, BUFFER_ALIGN)
        q_per_stage = BLK_M * d * self.q_width // 8
        o_per_stage = BLK_M * dv * self.o_width // 8
        smem_sO = _align_up(o_per_stage * EPI_STAGE, BUFFER_ALIGN)
        smem_sQ = _align_up(q_per_stage * Q_STAGE, BUFFER_ALIGN)
        sfq_per_stage = BLK_M * d // self.sf_vec_size
        sfp_per_stage = BLK_M * dv // self.sf_vec_size_pv
        smem_sSFQ = _align_up(sfq_per_stage * Q_STAGE, BUFFER_ALIGN)
        smem_sSFP = _align_up(sfp_per_stage * Q_STAGE, BUFFER_ALIGN) if self.quant_pv else 0
        smem_fixed = smem_mbar + smem_tmem + smem_sScale + smem_sO + smem_sQ + smem_sSFQ + smem_sSFP
        k_per_stage = BLK_M * d * self.k_width // 8
        v_per_stage = BLK_M * dv * self.v_width // 8
        self.k_aliases_v = self.k_width < self.v_width  # orig:L1157
        self.v_aliases_k = self.v_width == self.k_width  # orig:L1163
        if self.v_aliases_k or self.k_aliases_v:
            kv_per_stage = max(k_per_stage, v_per_stage)
        else:
            kv_per_stage = k_per_stage + v_per_stage
        sfk_per_stage = BLK_N * d // self.sf_vec_size
        sfv_per_stage = BLK_N * dv // self.sf_vec_size_pv
        kv_per_stage += sfk_per_stage
        fields = 3
        if self.quant_pv:
            kv_per_stage += sfv_per_stage
            fields += 1
        kv_per_stage += fields * 128
        kv_stage = (SMEM_BUDGET - smem_fixed) // kv_per_stage
        if not self.quant_pv and self.v_width == 8 and dv >= 128:
            kv_stage = min(kv_stage, 4)  # fp8_pv_kv_stage_cap (orig:L417-424)
        self.kv_stage = kv_stage
        # Byte sizes of the SharedStorage fields (orig:L1139-1192).
        self.q_stage_bytes = q_per_stage
        self.k_stage_bytes = k_per_stage
        self.v_stage_bytes = v_per_stage
        self.o_stage_bytes = o_per_stage
        self.sfq_stage_bytes = sfq_per_stage
        self.sfk_stage_bytes = sfk_per_stage
        self.sfp_stage_bytes = sfp_per_stage if self.quant_pv else 0
        self.sfv_stage_bytes = sfv_per_stage if self.quant_pv else 0
        self.tma_bytes_q = q_per_stage + sfq_per_stage
        self.tma_bytes_k = k_per_stage + sfk_per_stage
        self.tma_bytes_v = v_per_stage + (sfv_per_stage if self.quant_pv else 0)


# ---------------------------------------------------------------------------------------------
# Kernel (orig: FlashAttentionForwardSm100.kernel)
# ---------------------------------------------------------------------------------------------

# PTX spellings (one call is one instruction).
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_G2S_5D = (
    "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TMA_PREFETCH_4D = "cp.async.bulk.prefetch.tensor.4d.L2.global.tile"
# KV blocks prefetched into L2 ahead of the ring by the load warp (perf-gate addition, see
# make_kernel): the ring holds at most kv_stage/2 blocks, which at kv_stage 3 (bf16 V with an
# e4m3 K, 48 KB per block) does not cover a cold-L2 HBM fetch once the block time drops
# below ~1.2 us. Measured mxfp8_bf16 b1 s4096 h24 cold-L2: 125.7 us -> see ledger it-004.
KV_L2_PREFETCH_DISTANCE = 2  # applied only to rings of KV_L2_PREFETCH_MAX_STAGES or fewer slots
# Deeper rings hide the latency themselves and the prefetch cost them 1-2% on long streams.
KV_L2_PREFETCH_MAX_STAGES = 3
TCGEN05_CP = "tcgen05.cp.cta_group::1.32x128b.warpx4"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TMEM_LD_RED_X32 = "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
TMEM_LD_X16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_ST_X16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
MMA_MXF4NVF4_4X = "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
# `.block32` is the PTX 9.x spelling of `.scale_vec::1X`; the export writes the latter.
MMA_MXF8F6F4_1X = "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
MMA_F8F6F4 = "tcgen05.mma.cta_group::1.kind::f8f6f4"
NEG_INF = float("-inf")
L2_SIZE = 50 * 1024 * 1024  # tile_scheduler.py:428
# instruction descriptors, taken verbatim from the export (`mov.b32 idesc, ...`)
IDESC_QK_NVFP4 = 0x08201680
IDESC_BLOCK32_BASE = 0x08A00000  # | k << 29 | k << 4 (a_sf_id / b_sf_id = k)
IDESC_PV_BF16 = {128: 0x08210490, 64: 0x08110490}
IDESC_PV_FP8 = {128: 0x08210010, 64: 0x08110010}
IDESC_PV_NVFP4 = 0x08201680
# cuTensorMapEncodeTiled enums
_TMA_SWIZZLE = {0: 0, 32: 1, 64: 2, 128: 3}
_TMA_L2_PROMOTION_128B = 2
EXPECTED_KV_STAGE = {  # settled by the exports (fact 9 of the sketch)
    ("nvfp4", "bf16", 128): 4,
    ("nvfp4", "fp8", 128): 4,
    ("nvfp4", "nvfp4", 128): 13,
    ("nvfp4", "mxfp8", 128): 7,
    ("mxfp8", "bf16", 128): 3,
    ("mxfp8", "fp8", 128): 4,
    ("nvfp4", "bf16", 64): 10,
    ("nvfp4", "fp8", 64): 20,
}


def _l2_swizzle(seq_len_kv, head_dim, head_dim_v, element_size):
    """tile_scheduler.py:426-434 (SingleTileLPTScheduler)."""
    size_one_head = seq_len_kv * (head_dim + head_dim_v) * element_size
    if L2_SIZE < size_one_head:
        return 1
    return 1 << math.floor(math.log2(L2_SIZE // size_one_head))


def make_kernel(spec: Spec, batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, num_sms):
    """Trace the kernel for one specialization."""
    import inspect

    D, DV = spec.head_dim, spec.head_dim_v
    gqa = spec.qhead_per_kvhead
    heads = num_qo_heads  # unpacked GQA: one Q head per tile
    num_m_blocks = ceildiv(seq_len_q, Q_STAGE * BLK_M)
    num_tiles = batch_size * heads * num_m_blocks
    grid = min(num_sms, num_tiles) if spec.is_persistent else num_tiles
    is_causal = spec.is_causal
    quant_pv = spec.quant_pv
    pv_format, qk_format = spec.pv_format, spec.qk_format
    kv_stage = spec.kv_stage
    assert kv_stage == EXPECTED_KV_STAGE[(qk_format, pv_format, D)], (
        kv_stage,
        qk_format,
        pv_format,
        D,
    )
    QK_KT, PV_KT = spec.qk_k_tiles, spec.pv_k_tiles
    SF_TILE_K = D // (4 * spec.sf_vec_size)
    SF_TILE_K_PV = BLK_N // (4 * spec.sf_vec_size_pv) if quant_pv else 0
    P_COLS = BLK_N * spec.p_width // 32
    P_REP = spec.p_store_rep
    P_CHUNKS = spec.p_store_chunks
    P_SPLIT = spec.p_store_split
    PV_SPLIT = spec.p_split
    max_offset = {"nvfp4": math.log2(6.0), "mxfp8": math.log2(448.0)}.get(pv_format, 0.0)
    rescale_threshold = 0.0 if quant_pv else 8.0
    q_stage_bytes, k_stage_bytes, v_stage_bytes = (
        spec.q_stage_bytes,
        spec.k_stage_bytes,
        spec.v_stage_bytes,
    )
    kv_slot_bytes = (
        max(k_stage_bytes, v_stage_bytes) if (spec.k_aliases_v or spec.v_aliases_k) else None
    )
    o_stage_bytes = spec.o_stage_bytes
    sfq_chunk_bytes = 512 * SF_TILE_K
    kv_prefetch = KV_L2_PREFETCH_DISTANCE if kv_stage <= KV_L2_PREFETCH_MAX_STAGES else 0
    sfv_chunk_bytes = 512 * SF_TILE_K_PV
    # smem descriptor fields (16 B units); hi words match the export: 0x80004020 / 0xC0004010 / 0x40004040.
    qk_row_bytes = D * spec.q_width // 8
    QK_SWIZZLE = {32: 1, 64: 2, 128: 3}[qk_row_bytes]
    QK_SBO = 8 * qk_row_bytes // 16
    if pv_format in ("bf16", "fp8"):  # MN-major V: rows of min(DV, 64 bf16 | DV e4m3) elements
        v_row_bytes = min(DV, 64) * 2 if pv_format == "bf16" else DV
        v_k_step16 = (
            spec.pv_mma_k * v_row_bytes // 16
        )  # +0x80 (bf16), +0x100 / +0x80 (e4m3 d128 / d64)
        # LBO (export `or.b32 lo, 1024 << 16` for bf16 d128): the 16 KB stride between the two 64-column
        # swizzle-atom halves along N; a single-atom V (e4m3, or DV = 64) carries 0.
        V_LBO = 1024 if (pv_format == "bf16" and DV == 128) else 0
    else:  # K-major V: rows of 128 keys
        v_row_bytes = BLK_N * spec.v_width // 8
        v_k_step16 = 2  # 32 B per K tile
        V_LBO = 1  # K-major swizzled: unused, written as 1 like Q/K (export `or.b32 lo, 65536`)
    V_SWIZZLE = {32: 1, 64: 2, 128: 3}[v_row_bytes]
    V_SBO = 8 * v_row_bytes // 16
    v_half_bytes = 128 * 128 if pv_format == "bf16" else 0  # bf16 V: 64-column halves
    # TMEM columns (:243-258, :1594-1700), allocation base asserted 0.
    TMEM_S = spec.tmem_s_offset
    TMEM_O = spec.tmem_o_offset
    TMEM_P = spec.tmem_p_offset
    TMEM_SFQ = [TMEM_S[1], TMEM_S[0]]
    TMEM_SFK = [TMEM_S[1] + 4 * QK_KT, TMEM_S[0] + 4 * QK_KT]
    TMEM_SFP = [TMEM_S[0], TMEM_S[1]]
    TMEM_SFV = [TMEM_S[0] + 4 * SF_TILE_K_PV, TMEM_S[1] + 4 * SF_TILE_K_PV]
    l2_swizzle = _l2_swizzle(seq_len_kv, D, DV, max(spec.k_width // 8, 1))
    n_block_max_all = ceildiv(seq_len_kv, BLK_N)

    params = [
        ("tmap_q", K.TensorMap),
        ("tmap_k", K.TensorMap),
        ("tmap_v", K.TensorMap),
        ("tmap_o", K.TensorMap),
        ("tmap_sfq", K.TensorMap),
        ("tmap_sfk", K.TensorMap),
    ]
    if quant_pv:
        params.append(("tmap_sfv", K.TensorMap))
    params.append(("softmax_scale_log2", K.f32))
    names = tuple(n for n, _ in params)

    def body(tmap_q, tmap_k, tmap_v, tmap_o, tmap_sfq, tmap_sfk, tmap_sfv, softmax_scale_log2):
        warp = K.warp_id()
        tid = K.thread_id()
        tidx = tid & 127  # row within the warpgroup
        warp_in_wg = warp & 3
        lane_base = warp_in_wg * 32  # TMEM lane quarter of this warp

        # ---- SharedStorage (:1139-1192), declaration order -----------------------------------
        smem = K.smem_pool()
        q_load = K.Pipeline(smem, Q_STAGE, full="tma", empty="tcgen05")
        kv_load = K.Pipeline(smem, kv_stage, full="tma", empty="tcgen05")
        P_full_O_rescaled = K.MBarrier(smem, 2)
        P_full_O_rescaled.init(256)
        S_full = K.TCGen05Bar(smem, 2)
        S_full.init(1)
        O_full = K.TCGen05Bar(smem, 2)
        O_full.init(1)
        softmax_corr = K.Pipeline(smem, 2, full="mbar", empty="mbar", init_full=128, init_empty=128)
        corr_epi = K.Pipeline(smem, 2, full="mbar", empty="mbar", init_full=128, init_empty=32)
        tmem_dealloc = K.MBarrier(smem, 1)
        tmem_dealloc.init(384)
        P_full_2 = K.MBarrier(smem, 2)
        P_full_2.init(128)
        sfqk_load = K.MBarrier(smem, 2)
        sfqk_load.init(128)
        tmem_holding = smem.alloc((1,), K.u32)
        sScale = smem.alloc((Q_STAGE * BLK_M * 2,), K.f32, align=BUFFER_ALIGN)
        sO = smem.alloc((EPI_STAGE * o_stage_bytes,), K.u8, align=BUFFER_ALIGN)
        sQ = smem.alloc((Q_STAGE * q_stage_bytes,), K.u8, align=BUFFER_ALIGN)
        if kv_slot_bytes is not None:
            sKV = smem.alloc((kv_stage * kv_slot_bytes,), K.u8, align=BUFFER_ALIGN)
            sK = sV = sKV
        else:
            sK = smem.alloc((kv_stage * k_stage_bytes,), K.u8, align=BUFFER_ALIGN)
            sV = smem.alloc((kv_stage * v_stage_bytes,), K.u8, align=BUFFER_ALIGN)
        sSFQ = smem.alloc((Q_STAGE * sfq_chunk_bytes,), K.u8, align=BUFFER_ALIGN)
        sSFK = smem.alloc((kv_stage * sfq_chunk_bytes,), K.u8, align=BUFFER_ALIGN)
        if quant_pv:
            sSFP = smem.alloc((Q_STAGE * sfv_chunk_bytes,), K.u8, align=BUFFER_ALIGN)
            sSFV = smem.alloc((kv_stage * sfv_chunk_bytes,), K.u8, align=BUFFER_ALIGN)

        def k_slot_off(slot):
            return slot * (kv_slot_bytes if kv_slot_bytes is not None else k_stage_bytes)

        def v_slot_off(slot):
            return slot * (kv_slot_bytes if kv_slot_bytes is not None else v_stage_bytes)

        # ---- prologue (:1392-1514) -------------------------------------------------------------
        with K.If(warp == 0), K.Then():
            for tm in (tmap_q, tmap_k, tmap_v, tmap_o, tmap_sfq, tmap_sfk) + (
                (tmap_sfv,) if quant_pv else ()
            ):
                K.ptx.prefetch.tensormap(K.address_of(tm))
        K.ptx.fence.proxy.async_.shared__cta()
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        # ---- shared helpers --------------------------------------------------------------------
        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def tmem(lane, col):
            return K.cuda.get_tmem_addr(K.uint32(0), lane, col)

        def smem_u32(ptr):
            return K.cuda.cvta_generic_to_shared(ptr)

        def n_block_max_of(m_block):
            if not is_causal:
                return n_block_max_all
            n_idx = (m_block + 1) * (Q_STAGE * BLK_M) + (seq_len_kv - seq_len_q)
            return K.min(n_block_max_all, ceildiv(n_idx, BLK_N))

        def n_block_min_causal_of(m_block):
            n_idx = m_block * (Q_STAGE * BLK_M) + (seq_len_kv - seq_len_q)
            return K.max(0, n_idx // BLK_N)

        def make_scheduler(prefix):
            if is_causal:
                return K.FlashAttentionLPTScheduler(
                    prefix,
                    num_batches=batch_size,
                    num_heads=heads,
                    num_m_blocks=num_m_blocks,
                    l2_swizzle=l2_swizzle,
                )
            return K.FlashAttentionLinearScheduler(
                prefix,
                num_batches=batch_size,
                num_heads=heads,
                num_m_blocks=num_m_blocks,
                num_ctas=grid,
            )

        def fma_f32x2(vals, idx, mul_lo, mul_hi, add_lo, add_hi):
            """(vals[idx], vals[idx+1]) = fma.rn.f32x2((..), (mul), (add))."""
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            addend = K.local_scalar("uint64")
            K.ptx.mov.b64(packed, vals[idx], vals[idx + 1])
            K.ptx.mov.b64(rhs, mul_lo, mul_hi)
            K.ptx.mov.b64(addend, add_lo, add_hi)
            K.ptx.fma.rn.f32x2(packed, packed, rhs, addend)
            K.ptx.mov.b64(vals[idx], vals[idx + 1], packed)

        def mul_f32x2(vals, idx, m):
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            K.ptx.mov.b64(packed, vals[idx], vals[idx + 1])
            K.ptx.mov.b64(rhs, m, m)
            K.ptx.mul.rn.f32x2(packed, packed, rhs)
            K.ptx.mov.b64(vals[idx], vals[idx + 1], packed)

        def reduce_max(vals, base, n, init=None):
            """utils.py:410-432 (3-input max tree over vals[base:base+n])."""
            l = K.alloc_local([4], "float32")
            if init is None:
                K.ptx.max.f32(l[0], vals[base], vals[base + 1])
            else:
                K.ptx.max.f32(l[0], init, vals[base], vals[base + 1])
            for j in range(1, 4):
                K.ptx.max.f32(l[j], vals[base + 2 * j], vals[base + 2 * j + 1])
            for i in range(8, n, 8):
                for j in range(4):
                    K.ptx.max.f32(l[j], l[j], vals[base + i + 2 * j], vals[base + i + 2 * j + 1])
            out = K.local_scalar("float32")
            K.ptx.max.f32(l[0], l[0], l[1])
            K.ptx.max.f32(out, l[0], l[2], l[3])
            return out

        def reduce_sum(vals, base, n, init=None):
            """utils.py:455-473 (packed add.rn.f32x2 tree over vals[base:base+n])."""
            acc = K.alloc_local([8], "float32")
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            for j in range(4):
                K.assign(acc[2 * j], vals[base + 2 * j])
                K.assign(acc[2 * j + 1], vals[base + 2 * j + 1])
            if init is not None:
                K.ptx.mov.b64(packed, init, K.float32(0.0))
                K.ptx.mov.b64(rhs, vals[base], vals[base + 1])
                K.ptx.add.rn.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(acc[0], acc[1], packed)
            for i in range(8, n, 8):
                for j in range(4):
                    K.ptx.mov.b64(packed, acc[2 * j], acc[2 * j + 1])
                    K.ptx.mov.b64(rhs, vals[base + i + 2 * j], vals[base + i + 2 * j + 1])
                    K.ptx.add.rn.f32x2(packed, packed, rhs)
                    K.ptx.mov.b64(acc[2 * j], acc[2 * j + 1], packed)
            for lo, hi in ((0, 2), (4, 6), (0, 4)):
                K.ptx.mov.b64(packed, acc[lo], acc[lo + 1])
                K.ptx.mov.b64(rhs, acc[hi], acc[hi + 1])
                K.ptx.add.rn.f32x2(packed, packed, rhs)
                K.ptx.mov.b64(acc[lo], acc[lo + 1], packed)
            out = K.local_scalar("float32")
            K.ptx.add.f32(out, acc[0], acc[1])
            return out

        def pack_e4m3x4(word, f0, f1, f2, f3):
            lo = K.local_scalar("uint16")
            hi = K.local_scalar("uint16")
            K.ptx.cvt.rn.satfinite.e4m3x2.f32(lo, f1, f0)
            K.ptx.cvt.rn.satfinite.e4m3x2.f32(hi, f3, f2)
            K.ptx.mov.b32(word, lo, hi)

        def pack_ue8m0x4(word, f0, f1, f2, f3):
            lo = K.local_scalar("uint16")
            hi = K.local_scalar("uint16")
            K.ptx.cvt.rz.satfinite.ue8m0x2.f32(lo, f1, f0)
            K.ptx.cvt.rz.satfinite.ue8m0x2.f32(hi, f3, f2)
            K.ptx.mov.b32(word, lo, hi)

        def pack_e2m1x8(word, vals, base):
            bytes_ = K.alloc_local([4], "uint8")
            for j in range(4):
                K.ptx.cvt.rn.satfinite.e2m1x2.f32(
                    bytes_[j], vals[base + 2 * j + 1], vals[base + 2 * j]
                )
            K.assign(
                word,
                K.bitwise_or(
                    K.bitwise_or(
                        K.Cast("uint32", bytes_[0]),
                        K.shift_left(K.Cast("uint32", bytes_[1]), K.uint32(8)),
                    ),
                    K.bitwise_or(
                        K.shift_left(K.Cast("uint32", bytes_[2]), K.uint32(16)),
                        K.shift_left(K.Cast("uint32", bytes_[3]), K.uint32(24)),
                    ),
                ),
            )

        # ---- roles (:1737-1963) ----------------------------------------------------------------
        sp = K.specialize(chain_dispatch=True)
        r_softmax = sp.role("softmax", warps=list(range(8)), regs=spec.num_regs_softmax)
        r_correction = sp.role("correction", warps=[8, 9, 10, 11], regs=spec.num_regs_correction)
        wg3 = sp.warpgroup("wg3", warps=range(12, 16), regs=spec.num_regs_other)
        r_mma = sp.role("mma", warps=[MMA_WARP], group=wg3)
        r_epi = sp.role("epilogue", warps=[EPILOGUE_WARP], group=wg3)
        r_load = sp.role("load", warps=[LOAD_WARP], group=wg3)
        r_idle = sp.role("idle", warps=[EMPTY_WARP], group=wg3)

        with wg3:
            with r_idle:
                pass

            # ============================ load warp 14 (:1967-2235) ===========================
            with r_load:
                q_phase = K.local_scalar("int32", init=1)
                kv = K.PipelineState(kv_stage, phase=1)
                sched = make_scheduler("sched_load")
                sched.init(K.cta_id())
                with K.While(sched.valid()):
                    m_block = sched.m_block_idx
                    head = sched.head_idx
                    batch = sched.batch_idx
                    kv_head = head // gqa
                    n_max = K.local_scalar("int32", init=n_block_max_of(m_block))

                    def sf_coords(block):
                        # 5-D SF coordinates: the 128-row block index sits on dim 2 when a tile
                        # spans two 512 B chunks and on dim 1 when it spans one (sketch, fact 7).
                        if SF_TILE_K == 2:
                            return (K.int32(0), K.int32(0), K.Cast("int32", block))
                        return (K.int32(0), K.Cast("int32", block), K.int32(0))

                    def load_q(stage):
                        block = m_block * 2 + stage
                        q_load.empty.wait(stage, q_phase)
                        with K.If(elected()), K.Then():
                            q_load.full.arrive(stage, tx_count=spec.tma_bytes_q)
                        with K.If(elected()), K.Then():
                            K.ptx[TMA_G2S_4D](
                                sQ.ptr_to([stage * q_stage_bytes]),
                                K.address_of(tmap_q),
                                K.int32(0),
                                K.Cast("int32", block * BLK_M),
                                K.Cast("int32", head),
                                K.Cast("int32", batch),
                                smem_u32(q_load.full.ptr_to([stage])),
                                K.uint64(0),
                            )
                        with K.If(elected()), K.Then():
                            K.ptx[TMA_G2S_5D](
                                sSFQ.ptr_to([stage * sfq_chunk_bytes]),
                                K.address_of(tmap_sfq),
                                *sf_coords(block),
                                K.Cast("int32", head),
                                K.Cast("int32", batch),
                                smem_u32(q_load.full.ptr_to([stage])),
                                K.uint64(0),
                            )

                    def prefetch_kv(n):
                        """L2 prefetch of K/V block n (no smem): covers cold-L2 latency the ring cannot."""
                        with K.If(elected()), K.Then():
                            K.ptx[TMA_PREFETCH_4D](
                                K.address_of(tmap_k),
                                K.int32(0),
                                K.Cast("int32", n * BLK_N),
                                K.Cast("int32", kv_head),
                                K.Cast("int32", batch),
                            )
                            if pv_format == "bf16":
                                for half in range(DV // 64):
                                    K.ptx[TMA_PREFETCH_4D](
                                        K.address_of(tmap_v),
                                        K.int32(64 * half),
                                        K.Cast("int32", n * BLK_N),
                                        K.Cast("int32", kv_head),
                                        K.Cast("int32", batch),
                                    )
                            elif pv_format == "fp8":
                                K.ptx[TMA_PREFETCH_4D](
                                    K.address_of(tmap_v),
                                    K.int32(0),
                                    K.Cast("int32", n * BLK_N),
                                    K.Cast("int32", kv_head),
                                    K.Cast("int32", batch),
                                )
                            else:
                                key_units = BLK_N // 2 if pv_format == "nvfp4" else BLK_N
                                K.ptx[TMA_PREFETCH_4D](
                                    K.address_of(tmap_v),
                                    K.Cast("int32", n * key_units),
                                    K.int32(0),
                                    K.Cast("int32", kv_head),
                                    K.Cast("int32", batch),
                                )

                    def load_k(n):
                        slot = kv.stage
                        kv_load.empty.wait(slot, kv.phase)
                        with K.If(elected()), K.Then():
                            kv_load.full.arrive(slot, tx_count=spec.tma_bytes_k)
                        with K.If(elected()), K.Then():
                            K.ptx[TMA_G2S_4D](
                                sK.ptr_to([k_slot_off(slot)]),
                                K.address_of(tmap_k),
                                K.int32(0),
                                K.Cast("int32", n * BLK_N),
                                K.Cast("int32", kv_head),
                                K.Cast("int32", batch),
                                smem_u32(kv_load.full.ptr_to([slot])),
                                K.uint64(0),
                            )
                        with K.If(elected()), K.Then():
                            K.ptx[TMA_G2S_5D](
                                sSFK.ptr_to([slot * sfq_chunk_bytes]),
                                K.address_of(tmap_sfk),
                                *sf_coords(n),
                                K.Cast("int32", kv_head),
                                K.Cast("int32", batch),
                                smem_u32(kv_load.full.ptr_to([slot])),
                                K.uint64(0),
                            )
                        kv.advance()

                    def load_v(n):
                        slot = kv.stage
                        kv_load.empty.wait(slot, kv.phase)
                        with K.If(elected()), K.Then():
                            kv_load.full.arrive(slot, tx_count=spec.tma_bytes_v)
                        if pv_format == "bf16":
                            for half in range(DV // 64):
                                with K.If(elected()), K.Then():
                                    K.ptx[TMA_G2S_4D](
                                        sV.ptr_to([v_slot_off(slot) + half * v_half_bytes]),
                                        K.address_of(tmap_v),
                                        K.int32(64 * half),
                                        K.Cast("int32", n * BLK_N),
                                        K.Cast("int32", kv_head),
                                        K.Cast("int32", batch),
                                        smem_u32(kv_load.full.ptr_to([slot])),
                                        K.uint64(0),
                                    )
                        elif pv_format == "fp8":
                            with K.If(elected()), K.Then():
                                K.ptx[TMA_G2S_4D](
                                    sV.ptr_to([v_slot_off(slot)]),
                                    K.address_of(tmap_v),
                                    K.int32(0),
                                    K.Cast("int32", n * BLK_N),
                                    K.Cast("int32", kv_head),
                                    K.Cast("int32", batch),
                                    smem_u32(kv_load.full.ptr_to([slot])),
                                    K.uint64(0),
                                )
                        else:  # K-major V: the key block is the innermost coordinate (bytes for FP4)
                            key_units = BLK_N // 2 if pv_format == "nvfp4" else BLK_N
                            with K.If(elected()), K.Then():
                                K.ptx[TMA_G2S_4D](
                                    sV.ptr_to([v_slot_off(slot)]),
                                    K.address_of(tmap_v),
                                    K.Cast("int32", n * key_units),
                                    K.int32(0),
                                    K.Cast("int32", kv_head),
                                    K.Cast("int32", batch),
                                    smem_u32(kv_load.full.ptr_to([slot])),
                                    K.uint64(0),
                                )
                            if pv_format == "nvfp4":
                                sfv_coords = (K.int32(0), K.Cast("int32", n * 2), K.int32(0))
                            else:
                                sfv_coords = (K.int32(0), K.int32(0), K.Cast("int32", n))
                            with K.If(elected()), K.Then():
                                K.ptx[TMA_G2S_5D](
                                    sSFV.ptr_to([slot * sfv_chunk_bytes]),
                                    K.address_of(tmap_sfv),
                                    *sfv_coords,
                                    K.Cast("int32", kv_head),
                                    K.Cast("int32", batch),
                                    smem_u32(kv_load.full.ptr_to([slot])),
                                    K.uint64(0),
                                )
                        kv.advance()

                    if kv_prefetch:
                        for d in range(1, kv_prefetch + 1):
                            with K.If(n_max - d >= 0), K.Then():
                                prefetch_kv(n_max - d)
                    load_q(0)
                    load_k(n_max - 1)
                    load_q(1)
                    K.assign(q_phase, q_phase ^ 1)
                    load_v(n_max - 1)
                    with K.serial(n_max - 1, unroll=False) as i:
                        n = n_max - 2 - i
                        if kv_prefetch:
                            with K.If(n - kv_prefetch >= 0), K.Then():
                                prefetch_kv(n - kv_prefetch)
                        load_k(n)
                        load_v(n)
                    sched.next_tile()

            # ============================ MMA warp 12 (:1787-1843, :2238-2722) ================
            with r_mma:
                K.ptx[TMEM_ALLOC](K.address_of(tmem_holding[0]), K.uint32(TMEM_COLS))
                K.cuda.warp_sync()
                tmem_base = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(tmem_base, tmem_holding.ptr_to([0]))
                K.cuda.trap_when_assert_failed(tmem_base == K.uint32(0))

                # hoisted, lo-uniform descriptors for stage/slot 0 (bh:1224-1554)
                def make_desc(ptr, ldo, sdo, swizzle, uniform_lo=True):
                    desc = K.SmemDescriptor()
                    desc.init(ptr, ldo=ldo, sdo=sdo, swizzle=swizzle)
                    if uniform_lo:
                        desc.make_lo_uniform()
                    return desc

                desc_q = make_desc(sQ.ptr_to([0]), 1, QK_SBO, QK_SWIZZLE)
                desc_k = make_desc(sK.ptr_to([0]), 1, QK_SBO, QK_SWIZZLE)
                desc_v = make_desc(sV.ptr_to([0]), V_LBO, V_SBO, V_SWIZZLE)
                desc_sfq = make_desc(sSFQ.ptr_to([0]), 0, 8, 0, uniform_lo=False)
                desc_sfk = make_desc(sSFK.ptr_to([0]), 0, 8, 0, uniform_lo=False)
                if quant_pv:
                    desc_sfp = make_desc(sSFP.ptr_to([0]), 0, 8, 0, uniform_lo=False)
                    desc_sfv = make_desc(sSFV.ptr_to([0]), 0, 8, 0, uniform_lo=False)

                def desc_at(desc, off16):
                    return desc.add_16B_offset(off16)

                def cp_sf_chunks(desc, stride_bytes, slot, chunks, col_base):
                    for c in range(chunks):
                        with K.If(elected()), K.Then():
                            K.ptx[TCGEN05_CP](
                                K.uint32(col_base + 4 * c),
                                desc_at(desc, slot * (stride_bytes // 16) + 32 * c),
                            )

                def gemm_qk(stage, slot):
                    a_base = stage * (q_stage_bytes // 16)
                    b_base = slot * (k_slot_off(1) // 16)
                    for kt in range(QK_KT):
                        with K.If(elected()), K.Then():
                            if qk_format == "nvfp4":
                                K.ptx[MMA_MXF4NVF4_4X](
                                    K.uint32(TMEM_S[stage]),
                                    desc_at(desc_q, a_base + 2 * kt),
                                    desc_at(desc_k, b_base + 2 * kt),
                                    K.uint32(IDESC_QK_NVFP4),
                                    K.uint32(TMEM_SFQ[stage] + 4 * kt),
                                    K.uint32(TMEM_SFK[stage] + 4 * kt),
                                    kt != 0,
                                )
                            else:  # bh:1191-1220: per-K sf_id in the idesc and the SF address top bits
                                K.ptx[MMA_MXF8F6F4_1X](
                                    K.uint32(TMEM_S[stage]),
                                    desc_at(desc_q, a_base + 2 * kt),
                                    desc_at(desc_k, b_base + 2 * kt),
                                    K.uint32(IDESC_BLOCK32_BASE | (kt << 29) | (kt << 4)),
                                    K.uint32(TMEM_SFQ[stage] | (kt << 30)),
                                    K.uint32(TMEM_SFK[stage] | (kt << 30)),
                                    kt != 0,
                                )

                def gemm_pv(stage, slot, acc, p_phase):
                    b_base = slot * (v_slot_off(1) // 16)

                    def issue(kt):
                        enable = True if kt != 0 else K.Cast("bool", acc)
                        a_tmem = K.uint32(TMEM_P[stage] + 8 * kt)
                        b_desc = desc_at(desc_v, b_base + v_k_step16 * kt)
                        if pv_format == "bf16":
                            K.ptx[MMA_F16](
                                K.uint32(TMEM_O[stage]),
                                a_tmem,
                                b_desc,
                                K.uint32(IDESC_PV_BF16[DV]),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                enable,
                            )
                        elif pv_format == "fp8":
                            K.ptx[MMA_F8F6F4](
                                K.uint32(TMEM_O[stage]),
                                a_tmem,
                                b_desc,
                                K.uint32(IDESC_PV_FP8[DV]),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                enable,
                            )
                        elif pv_format == "nvfp4":
                            K.ptx[MMA_MXF4NVF4_4X](
                                K.uint32(TMEM_O[stage]),
                                a_tmem,
                                b_desc,
                                K.uint32(IDESC_PV_NVFP4),
                                K.uint32(TMEM_SFP[stage] + 4 * kt),
                                K.uint32(TMEM_SFV[stage] + 4 * kt),
                                enable,
                            )
                        else:  # mxfp8 PV (bh:1409-1554): static SF operands, sf_id = kt in the idesc
                            K.ptx[MMA_MXF8F6F4_1X](
                                K.uint32(TMEM_O[stage]),
                                a_tmem,
                                b_desc,
                                K.uint32(IDESC_BLOCK32_BASE | (kt << 29) | (kt << 4)),
                                K.uint32(TMEM_SFP[stage]),
                                K.uint32(TMEM_SFV[stage]),
                                enable,
                            )

                    for kt in range(PV_KT):
                        if kt == PV_SPLIT:
                            P_full_2.wait(stage, p_phase)
                        with K.If(elected()), K.Then():
                            issue(kt)

                def cp_sfqk(stage, slot):
                    cp_sf_chunks(desc_sfq, sfq_chunk_bytes, stage, SF_TILE_K, TMEM_SFQ[stage])
                    cp_sf_chunks(desc_sfk, sfq_chunk_bytes, slot, SF_TILE_K, TMEM_SFK[stage])

                def cp_sfpv(stage, slot):
                    if quant_pv:
                        cp_sf_chunks(
                            desc_sfp, sfv_chunk_bytes, stage, SF_TILE_K_PV, TMEM_SFP[stage]
                        )
                        cp_sf_chunks(desc_sfv, sfv_chunk_bytes, slot, SF_TILE_K_PV, TMEM_SFV[stage])

                q_phase = K.local_scalar("int32", init=0)
                p_phase = K.local_scalar("int32", init=0)
                sfqk_phase = K.local_scalar("int32", init=0)
                kv = K.PipelineState(kv_stage, phase=0)
                acc = K.local_scalar("int32", init=0)
                sched = make_scheduler("sched_mma")
                sched.init(K.cta_id())
                with K.While(sched.valid()):
                    m_block = sched.m_block_idx
                    n_blocks = K.local_scalar("int32", init=n_block_max_of(m_block))
                    K.assign(acc, 0)
                    # QK0, QK1 on the newest block (:2423-2489)
                    for stage in range(Q_STAGE):
                        q_load.full.wait(stage, q_phase)
                        if stage == 0:
                            kv_load.full.wait(kv.stage, kv.phase)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        sfqk_load.wait(stage, sfqk_phase)
                        cp_sfqk(stage, kv.stage)
                        gemm_qk(stage, kv.stage)
                        with K.If(elected()), K.Then():
                            S_full.arrive(stage)
                    K.assign(q_phase, q_phase ^ 1)
                    K.assign(sfqk_phase, sfqk_phase ^ 1)
                    with K.If(elected()), K.Then():
                        kv_load.empty.arrive(kv.stage)
                    kv.advance()
                    # steady loop (:2494-2637)
                    with K.serial(n_blocks - 1, unroll=False) as _i:
                        kv_load.full.wait(kv.stage, kv.phase)  # V_i
                        v_slot = K.local_scalar("int32", init=kv.stage)
                        for stage in range(Q_STAGE):
                            P_full_O_rescaled.wait(stage, p_phase)
                            cp_sfpv(stage, v_slot)
                            gemm_pv(stage, v_slot, acc, p_phase)
                            if stage == 1:
                                with K.If(elected()), K.Then():
                                    kv_load.empty.arrive(v_slot)
                            if stage == 0:
                                kv.advance()
                                kv_load.full.wait(kv.stage, kv.phase)  # K_i
                            K.ptx.tcgen05.fence__after_thread_sync()
                            sfqk_load.wait(stage, sfqk_phase)
                            cp_sfqk(stage, kv.stage)
                            gemm_qk(stage, kv.stage)
                            with K.If(elected()), K.Then():
                                S_full.arrive(stage)
                        with K.If(elected()), K.Then():
                            kv_load.empty.arrive(kv.stage)
                        kv.advance()
                        K.assign(p_phase, p_phase ^ 1)
                        K.assign(sfqk_phase, sfqk_phase ^ 1)
                        K.assign(acc, 1)
                    with K.If(elected()), K.Then():
                        for stage in range(Q_STAGE):
                            q_load.empty.arrive(stage)
                    # tail PV (:2645-2716)
                    kv_load.full.wait(kv.stage, kv.phase)
                    for stage in range(Q_STAGE):
                        P_full_O_rescaled.wait(stage, p_phase)
                        cp_sfpv(stage, kv.stage)
                        gemm_pv(stage, kv.stage, acc, p_phase)
                        with K.If(elected()), K.Then():
                            O_full.arrive(stage)
                    K.assign(p_phase, p_phase ^ 1)
                    with K.If(elected()), K.Then():
                        kv_load.empty.arrive(kv.stage)
                    kv.advance()
                    sched.next_tile()
                K.ptx[TMEM_RELINQUISH]()
                tmem_dealloc.wait(0, 0)
                dealloc = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(dealloc, tmem_holding.ptr_to([0]))
                K.ptx[TMEM_DEALLOC](dealloc, K.uint32(TMEM_COLS))

            # ============================ epilogue warp 13 (:4405-4510) =======================
            with r_epi:
                phase = K.local_scalar("int32", init=0)
                sched = make_scheduler("sched_epi")
                sched.init(K.cta_id())
                with K.While(sched.valid()):
                    m_block = sched.m_block_idx
                    head = sched.head_idx
                    batch = sched.batch_idx
                    for stage in range(EPI_STAGE):
                        corr_epi.full.wait(stage, phase)
                        for half in range(DV // 64):
                            K.ptx[TMA_S2G_4D](
                                K.address_of(tmap_o),
                                K.int32(64 * half),
                                K.Cast("int32", (m_block * 2 + stage) * BLK_M),
                                K.Cast("int32", head),
                                K.Cast("int32", batch),
                                sO.ptr_to([stage * o_stage_bytes + half * 16384]),
                                K.uint64(0),
                            )
                        K.ptx.cp.async_.bulk.commit_group()
                    K.ptx.cp.async_.bulk.wait_group.read(1)
                    corr_epi.empty.arrive(0)
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                    corr_epi.empty.arrive(1)
                    K.assign(phase, phase ^ 1)
                    sched.next_tile()

        # ============================ softmax warpgroups 0..7 (:2727-3063, :3559-3883) ========
        with r_softmax:
            stage = K.uniform(warp >> 2)  # runtime stage: one body for both warpgroups
            si_phase = K.local_scalar("int32", init=0)
            corr_phase = K.local_scalar("int32", init=1)
            row_max = K.local_scalar("float32")
            row_sum = K.local_scalar("float32")
            with K.If(stage == 1), K.Then():
                sfqk_load.arrive(0)
            s_col = stage * BLK_N  # S[stage] base column (128 * stage)
            p_col = s_col + spec.tmem_s_to_p_offset
            sched = make_scheduler("sched_softmax")
            sched.init(K.cta_id())

            def softmax_step(n, first, mask_seqlen, mask_causal, unmasked):
                m_block = sched.m_block_idx
                S_full.wait(stage, si_phase)
                s = K.alloc_local([BLK_N], "float32")
                tile_max = K.alloc_local([4], "float32")
                for j in range(4):
                    K.ptx[TMEM_LD_RED_X32](
                        *[s[32 * j + i] for i in range(32)],
                        tile_max[j],
                        tmem(lane_base, s_col + 32 * j),
                    )
                K.ptx.tcgen05.wait__ld.sync.aligned()
                sfqk_load.arrive(1 - stage)
                if mask_seqlen or mask_causal:
                    seqlen_limit = seq_len_kv - K.max(n, 0) * BLK_N
                    if mask_causal:
                        row = tidx + (m_block * 2 + stage) * BLK_M
                        limit = row + (seq_len_kv - n * BLK_N - seq_len_q) + 1
                        if mask_seqlen:
                            limit = K.min(limit, seqlen_limit)
                    else:
                        limit = seqlen_limit
                    lim = K.local_scalar("int32", init=limit)
                    for c in range(4):
                        bits = K.local_scalar("uint32")
                        K.ptx.shr.u32(
                            bits,
                            K.uint32(0xFFFFFFFF),
                            K.Cast("uint32", K.max((c + 1) * 32 - lim, 0)),
                        )
                        for i in range(32):
                            keep = K.bitwise_and(bits, K.uint32(1 << i)) != K.uint32(0)
                            K.ptx.mov.b32(
                                s[32 * c + i], K.Select(keep, s[32 * c + i], K.float32(NEG_INF))
                            )
                new_max = K.local_scalar("float32")
                if unmasked:
                    hw = K.local_scalar("float32")
                    K.ptx.max.f32(hw, tile_max[0], tile_max[1])
                    K.ptx.max.f32(hw, hw, tile_max[2])
                    K.ptx.max.f32(hw, hw, tile_max[3])
                    K.ptx.max.f32(new_max, hw, row_max)
                else:
                    K.assign(new_max, reduce_max(s, 0, BLK_N, init=None if first else row_max))
                safe = K.local_scalar(
                    "float32", init=K.Select(new_max != K.float32(NEG_INF), new_max, K.float32(0.0))
                )
                acc_scale = K.local_scalar("float32")
                if not first:
                    acc_scale_ = K.local_scalar("float32")
                    K.ptx.sub.f32(acc_scale_, row_max, safe)
                    K.ptx.mul.f32(acc_scale_, acc_scale_, softmax_scale_log2)
                    K.ptx.ex2.approx.ftz.f32(acc_scale, acc_scale_)
                    if rescale_threshold > 0.0:
                        keep_max = acc_scale_ >= K.float32(-rescale_threshold)
                        K.assign(new_max, K.Select(keep_max, row_max, new_max))
                        K.assign(safe, K.Select(keep_max, row_max, safe))
                        K.assign(acc_scale, K.Select(keep_max, K.float32(1.0), acc_scale))
                    K.ptx.st.shared.f32(sScale.ptr_to([tidx + stage * BLK_M]), acc_scale)
                K.assign(row_max, new_max)
                softmax_corr.full.arrive(stage)
                # scale_subtract_rowmax (sm:271-285)
                bias = K.local_scalar("float32")
                K.ptx.mul.f32(bias, safe, softmax_scale_log2)
                K.ptx.sub.f32(bias, K.float32(max_offset), bias)
                for i in range(0, BLK_N, 2):
                    fma_f32x2(s, i, softmax_scale_log2, softmax_scale_log2, bias, bias)
                p_words = K.alloc_local([P_COLS], "uint32")
                if pv_format == "bf16":
                    for frag in range(4):
                        for i in range(32):
                            K.ptx.ex2.approx.ftz.f32(s[32 * frag + i], s[32 * frag + i])
                        for i in range(16):
                            K.ptx.cvt.rn.bf16x2.f32(
                                p_words[16 * frag + i],
                                s[32 * frag + 2 * i + 1],
                                s[32 * frag + 2 * i],
                            )
                elif pv_format == "fp8":
                    for i in range(BLK_N):
                        K.ptx.ex2.approx.ftz.f32(s[i], s[i])
                    for i in range(32):
                        pack_e4m3x4(p_words[i], s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3])
                else:  # _fused_log2_group_quant (:3171-3295)
                    gsize = spec.sf_vec_size_pv
                    ngroups = BLK_N // gsize
                    sf = K.alloc_local([ngroups], "float32")
                    row_acc = K.local_scalar("float32")
                    for g in range(ngroups):
                        m = K.local_scalar("float32")
                        if unmasked and gsize == 32:
                            K.ptx.fma.rn.f32(m, tile_max[g], softmax_scale_log2, bias)
                        else:
                            K.assign(m, reduce_max(s, g * gsize, gsize))
                        b = K.local_scalar("float32")
                        K.ptx.max.f32(b, m, K.float32(-100.0))
                        K.ptx.add.f32(b, b, K.float32(-max_offset))
                        if spec.sf_pv_e8m0:
                            K.ptx.cvt.rpi.f32.f32(b, b)
                        nb = K.local_scalar("float32")
                        K.ptx.neg.f32(nb, b)
                        for i in range(0, gsize, 2):
                            fma_f32x2(s, g * gsize + i, K.float32(1.0), K.float32(1.0), nb, nb)
                        for i in range(gsize):
                            K.ptx.ex2.approx.ftz.f32(s[g * gsize + i], s[g * gsize + i])
                        K.ptx.ex2.approx.ftz.f32(sf[g], b)
                        gs = reduce_sum(s, g * gsize, gsize)
                        K.ptx.fma.rn.f32(row_acc, sf[g], gs, K.float32(0.0) if g == 0 else row_acc)
                        if pv_format == "nvfp4":
                            for w in range(gsize // 8):
                                pack_e2m1x8(p_words[g * (gsize // 8) + w], s, g * gsize + 8 * w)
                        else:
                            for w in range(gsize // 4):
                                base = g * gsize + 4 * w
                                pack_e4m3x4(
                                    p_words[g * (gsize // 4) + w],
                                    s[base],
                                    s[base + 1],
                                    s[base + 2],
                                    s[base + 3],
                                )
                    if first:
                        K.assign(row_sum, row_acc)
                    else:
                        K.ptx.fma.rn.f32(row_sum, row_sum, acc_scale, row_acc)
                    for w in range(max(1, ngroups // 4)):
                        sf_word = K.local_scalar("uint32")
                        if spec.sf_pv_e8m0:
                            pack_ue8m0x4(
                                sf_word, sf[4 * w], sf[4 * w + 1], sf[4 * w + 2], sf[4 * w + 3]
                            )
                        else:
                            pack_e4m3x4(
                                sf_word, sf[4 * w], sf[4 * w + 1], sf[4 * w + 2], sf[4 * w + 3]
                            )
                        K.ptx.st.shared.b32(
                            sSFP.ptr_to(
                                [
                                    stage * sfv_chunk_bytes
                                    + K.lane_id() * 16
                                    + warp_in_wg * 4
                                    + 512 * w
                                ]
                            ),
                            sf_word,
                        )
                    # The MMA warp reads sSFP through the async proxy (tcgen05.cp) after the P_full
                    # arrive below. The upstream kernel issues no proxy fence here and is measurably
                    # nondeterministic on multi-tile persistent shapes (its own back-to-back launches
                    # differ); one fence per step orders the generic-proxy stores before that read.
                    K.ptx.fence.proxy.async_.shared__cta()
                # P -> TMEM with the split handoff (:3861-3870)
                st = f"tcgen05.st.sync.aligned.32x32b.x{P_REP}.b32"
                for c in range(P_SPLIT):
                    K.ptx[st](
                        tmem(lane_base, p_col + c * P_REP),
                        *[p_words[c * P_REP + i] for i in range(P_REP)],
                    )
                K.ptx.tcgen05.wait__st.sync.aligned()
                P_full_O_rescaled.arrive(stage)
                for c in range(P_SPLIT, P_CHUNKS):
                    K.ptx[st](
                        tmem(lane_base, p_col + c * P_REP),
                        *[p_words[c * P_REP + i] for i in range(P_REP)],
                    )
                K.ptx.tcgen05.wait__st.sync.aligned()
                P_full_2.arrive(stage)
                softmax_corr.empty.wait(stage, corr_phase)
                if not quant_pv:
                    if first:
                        K.assign(row_sum, reduce_sum(s, 0, BLK_N))
                    else:
                        init = K.local_scalar("float32")
                        K.ptx.mul.f32(init, row_sum, acc_scale)
                        K.assign(row_sum, reduce_sum(s, 0, BLK_N, init=init))
                K.assign(si_phase, si_phase ^ 1)
                K.assign(corr_phase, corr_phase ^ 1)

            with K.While(sched.valid()):
                m_block = sched.m_block_idx
                n_max = K.local_scalar("int32", init=n_block_max_of(m_block))
                K.assign(row_max, K.float32(NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                softmax_corr.empty.wait(stage, corr_phase)
                K.assign(corr_phase, corr_phase ^ 1)
                softmax_step(
                    n_max - 1, first=True, mask_seqlen=True, mask_causal=is_causal, unmasked=False
                )
                if is_causal:
                    n_min_causal = K.local_scalar("int32", init=n_block_min_causal_of(m_block))
                    with K.serial(K.max(n_max - 1 - n_min_causal, 0), unroll=False) as i:
                        softmax_step(
                            n_max - 2 - i,
                            first=False,
                            mask_seqlen=False,
                            mask_causal=True,
                            unmasked=False,
                        )
                    with K.serial(K.min(n_max - 1, n_min_causal), unroll=False) as i:
                        softmax_step(
                            K.min(n_max - 1, n_min_causal) - 1 - i,
                            first=False,
                            mask_seqlen=False,
                            mask_causal=True,
                            unmasked=False,
                        )
                else:
                    with K.serial(n_max - 1, unroll=False) as i:
                        softmax_step(
                            n_max - 2 - i,
                            first=False,
                            mask_seqlen=False,
                            mask_causal=False,
                            unmasked=True,
                        )
                K.ptx.st.shared.f32(sScale.ptr_to([tidx + stage * BLK_M]), row_sum)
                softmax_corr.full.arrive(stage)
                sched.next_tile()
            tmem_dealloc.arrive(0)

        # ============================ correction warpgroup 8..11 (:3885-4360) =================
        with r_correction:
            P_full_O_rescaled.arrive(0)
            P_full_O_rescaled.arrive(1)
            sc_phase = K.local_scalar("int32", init=0)
            o_phase = K.local_scalar("int32", init=0)
            ce_phase = K.local_scalar("int32", init=1)
            sched = make_scheduler("sched_corr")
            sched.init(K.cta_id())
            with K.While(sched.valid()):
                m_block = sched.m_block_idx
                n_blocks = K.local_scalar("int32", init=n_block_max_of(m_block))
                softmax_corr.full.wait(0, sc_phase)
                softmax_corr.empty.arrive(0)
                softmax_corr.full.wait(1, sc_phase)
                K.assign(sc_phase, sc_phase ^ 1)
                with K.serial(n_blocks - 1, unroll=False) as _i:
                    for stage in range(Q_STAGE):
                        softmax_corr.full.wait(stage, sc_phase)
                        scale = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(scale, sScale.ptr_to([tidx + stage * BLK_M]))
                        ballot = K.local_scalar("uint32")
                        K.ptx.vote_sync.ballot.b32(
                            ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                        )
                        with K.If(ballot != K.uint32(0)), K.Then():
                            o = K.alloc_local([16], "float32")
                            for t in range(DV // 16):
                                addr = tmem(lane_base, TMEM_O[stage] + 16 * t)
                                K.ptx[TMEM_LD_X16](*[o[i] for i in range(16)], addr)
                                for j in range(0, 16, 2):
                                    mul_f32x2(o, j, scale)
                                K.ptx[TMEM_ST_X16](addr, *[o[i] for i in range(16)])
                            K.ptx.tcgen05.wait__st.sync.aligned()
                        P_full_O_rescaled.arrive(stage)
                        softmax_corr.empty.arrive(1 - stage)
                    K.assign(sc_phase, sc_phase ^ 1)
                softmax_corr.empty.arrive(1)
                for stage in range(Q_STAGE):
                    softmax_corr.full.wait(stage, sc_phase)
                    rs = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(rs, sScale.ptr_to([tidx + stage * BLK_M]))
                    softmax_corr.empty.arrive(stage)
                    inv = K.local_scalar("float32")
                    K.ptx.rcp.approx.ftz.f32(
                        inv, K.Select(rs != K.float32(0.0), rs, K.float32(1.0))
                    )
                    O_full.wait(stage, o_phase)
                    corr_epi.empty.wait(stage, ce_phase)
                    o = K.alloc_local([16], "float32")
                    h = K.alloc_local([8], "uint32")
                    row = tidx
                    for t in range(DV // 16):
                        K.ptx[TMEM_LD_X16](
                            *[o[i] for i in range(16)], tmem(lane_base, TMEM_O[stage] + 16 * t)
                        )
                        for j in range(0, 16, 2):
                            mul_f32x2(o, j, inv)
                        for j in range(8):
                            K.ptx.cvt.rn.bf16x2.f32(h[j], o[2 * j + 1], o[2 * j])
                        for half in range(2):
                            c = 16 * t + 8 * half
                            off = (
                                stage * o_stage_bytes
                                + (c // 64) * 16384
                                + (row // 8) * 1024
                                + (row % 8) * 128
                                + K.Cast(
                                    "int32",
                                    K.bitwise_xor(
                                        K.Cast("uint32", (c % 64) // 8), K.Cast("uint32", row % 8)
                                    ),
                                )
                                * 16
                                + (c % 8) * 2
                            )
                            K.ptx.st.shared.v4.b32(
                                sO.ptr_to([off]),
                                h[4 * half],
                                h[4 * half + 1],
                                h[4 * half + 2],
                                h[4 * half + 3],
                            )
                    K.ptx.fence.proxy.async_.shared__cta()
                    corr_epi.full.arrive(stage)
                    P_full_O_rescaled.arrive(stage)
                K.assign(o_phase, o_phase ^ 1)
                K.assign(sc_phase, sc_phase ^ 1)
                K.assign(ce_phase, ce_phase ^ 1)
                sched.next_tile()
            tmem_dealloc.arrive(0)

    def entry(*args):
        values = dict(zip(names, args, strict=True))
        body(
            values["tmap_q"],
            values["tmap_k"],
            values["tmap_v"],
            values["tmap_o"],
            values["tmap_sfq"],
            values["tmap_sfk"],
            values.get("tmap_sfv"),
            values["softmax_scale_log2"],
        )

    entry.__name__ = "flash_attention4_fp4"
    entry.__signature__ = inspect.Signature(
        [
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a)
            for n, a in params
        ]
    )
    return K.kernel(warps=NUM_WARPS, arch="sm_103a", min_blocks_per_sm=1, grid=grid)(entry)


# ---------------------------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------------------------

KERNEL_META = {
    "name": "flash_attention4_fp4",
    "category": "flashattention",
    "runtime_cuda_archs": ["sm_103a"],
    "reference_requirements": (
        {
            "package": "flash-attn-4",
            "git": {
                "url": "https://github.com/hao-ai-lab/flash-attention-fp4.git",
                "commit": "5aa37a9680f7b76a11799b5f4846100ed5a3e6d8",
            },
            "import": "flash_attn.cute",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
    ),
}


def _cfg(qk, pv, b, sq, sk, h, kv, d, causal=False):
    seq = f"s{sq}" if sq == sk else f"sq{sq}_sk{sk}"
    return {
        "qk_format": qk,
        "pv_format": pv,
        "batch_size": b,
        "seq_len_q": sq,
        "seq_len_kv": sk,
        "num_qo_heads": h,
        "num_kv_heads": kv,
        "head_dim": d,
        "is_causal": causal,
        "label": f"{qk}_{pv}_b{b}_{seq}_h{h}kv{kv}_d{d}{'_causal' if causal else ''}",
    }


CONFIGS = [
    # Core matrix: every mode x {noncausal, causal} x {MHA, GQA 4:1}.
    *[
        _cfg(qk, pv, 1, 1024, 1024, 32, kv, 128, causal)
        for qk, pv in MODES
        for causal in (False, True)
        for kv in (32, 8)
    ],
    # KV ring wrap for the deep NVFP4-PV ring.
    _cfg("nvfp4", "nvfp4", 1, 4096, 4096, 32, 32, 128),
    _cfg("nvfp4", "nvfp4", 1, 4096, 4096, 32, 32, 128, True),
    # multi-tile persistent schedule (512 tiles over 152 SMs) with a quantized P and E8M0 scales
    _cfg("nvfp4", "mxfp8", 1, 4096, 4096, 32, 32, 128),
    # Single 256-row / 128-column tile.
    _cfg("nvfp4", "bf16", 1, 256, 256, 32, 32, 128),
    _cfg("nvfp4", "nvfp4", 1, 256, 256, 32, 32, 128),
    # Odd 128-row block counts (stage-1 Q tile fully out of range).
    _cfg("nvfp4", "bf16", 1, 128, 128, 32, 32, 128),
    _cfg("nvfp4", "bf16", 1, 384, 384, 32, 32, 128, True),
    _cfg("nvfp4", "nvfp4", 1, 384, 384, 32, 32, 128),
    # seq_len_q != seq_len_kv.
    _cfg("nvfp4", "bf16", 1, 256, 1024, 32, 8, 128),
    _cfg("nvfp4", "bf16", 1, 512, 1024, 32, 8, 128, True),
    _cfg("nvfp4", "nvfp4", 1, 256, 2048, 32, 32, 128, True),
    # batch > 1.
    _cfg("nvfp4", "bf16", 2, 512, 512, 32, 8, 128, True),
    _cfg("mxfp8", "fp8", 2, 512, 512, 32, 32, 128),
    # head_dim 64.
    *[
        _cfg("nvfp4", pv, 1, 1024, 1024, 32, kv, 64, causal)
        for pv in ("bf16", "fp8")
        for causal in (False, True)
        for kv in (32, 8)
    ],
]

_HEADLINE = [
    (1, 256, 16),
    (1, 1024, 16),
    (4, 4096, 16),
    (1, 32768, 16),
    (4, 4096, 32),
    (1, 4096, 12),
    (1, 32768, 12),
    (1, 4096, 24),
    (1, 32768, 24),
]
BENCH_CONFIGS = [
    *[_cfg(qk, pv, b, s, s, h, h, 128) for qk, pv in MODES for b, s, h in _HEADLINE],
    _cfg("nvfp4", "bf16", 1, 32768, 32768, 24, 24, 64),
    _cfg("nvfp4", "fp8", 1, 32768, 32768, 24, 24, 64),
    *[
        _cfg(qk, pv, 1, s, s, 32, 8, 128, True)
        for qk, pv in (("nvfp4", "bf16"), ("nvfp4", "fp8"), ("mxfp8", "fp8"))
        for s in (1024, 2048, 4096, 8192, 16384, 32768)
    ],
]

# Upstream env knobs that change the compiled program; pinned to the SM103 defaults the port
# transcribes (orig:L282-335, L1105-1108, L2790-2794). Read at import by the fork, so they are
# set before the first ``flash_attn.cute`` import.
_UPSTREAM_KNOBS = {
    "FA4_LDRED_ROWMAX": "1",
    "FA4_FORCE_E2E": "0",
    "FA4_FP4_PV_LOG2_QUANT": "1",
    "FA4_FP8_PV_USE_EXPLICIT_PACK": "1",
    "FA4_FP8_PV_USE_FUSED_PACK": "0",
    "FA4_FP8_PV_PACK_STORE_PIPELINE": "0",
    "FA4_FP4_PV_QUANT_STORE_PIPELINE": "0",
    "FA4_FP8_PV_RANGE_UNROLL": "0",
    "FA4_FP8_PV_P_LOG2_OFFSET": "0.0",
    "FA4_FP8_PV_ZERO_FILL_REGS": "1",
    "FA4_PROFILE_PIPELINE": "0",
    "FA4_PROFILE_DETAIL": "0",
    "FA4_MXFP8_USE_INLINE_PTX": "0",
    "FA4_DEBUG_FORCE_GENERIC_MXFP8_QK": "0",
    "FA4_SFQK_TMEM_SLOT": "s",
    "FA4_FP8_PV_P_SPLIT_NUM": "3",
    "FA4_FP8_PV_P_SPLIT_DEN": "4",
    "FA4_FP4_PV_P_SPLIT_NUM": "1",
    "FA4_FP4_PV_P_SPLIT_DEN": "2",
    "FA4_FP8_PV_TMEM_STORE_REP": "8",
    "FA4_FP4_PV_TMEM_STORE_REP": "8",
}


def _pin_upstream_knobs():
    for key, value in _UPSTREAM_KNOBS.items():
        os.environ[key] = value


# ptxas --register-usage-level per (qk_format, pv_format, is_causal, single_wave). Measured on GB300
# (bench_suite, reference-paired): the NVFP4-PV kernel on a single-wave, non-causal grid (every CTA
# owns one tile) runs 15.5 us vs 18.5 us at level 10 for b1 s1024 h16 (levels 1-3 identical, 4-10
# identical); the same level costs +28% once tiles are long (s4096+), so it is keyed on the regime.
# Every other mode measured flat across levels 3-10 and keeps the checkout default 10.
_REG_LEVEL_TABLE: dict = {("nvfp4", "nvfp4", False, True): "3"}


def _select_reg_level(qk_format, pv_format, is_causal, single_wave):
    override = os.environ.get("FA4FP4_REG_LEVEL", "")
    if override:
        return override
    return _REG_LEVEL_TABLE.get((qk_format, pv_format, is_causal, single_wave), "10")


def _check_shape(seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads):
    if seq_len_q % BLK_M or seq_len_kv % BLK_N:
        raise ValueError("seq_len_q and seq_len_kv must be multiples of 128")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")


def get_kernel(
    qk_format,
    pv_format,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    **kwargs,
):
    from tirx_kernels.runner import hardware_num_sms

    _check_shape(seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads)
    spec = Spec(qk_format, pv_format, head_dim, is_causal, num_qo_heads, num_kv_heads)
    num_sms = hardware_num_sms()
    num_tiles = batch_size * num_qo_heads * ceildiv(seq_len_q, Q_STAGE * BLK_M)
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _select_reg_level(
        qk_format, pv_format, is_causal, num_tiles <= num_sms
    )
    return make_kernel(
        spec, batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, num_sms
    ).func


# ---------------------------------------------------------------------------------------------
# Data preparation (orig: benchmarks/bench_fp4.py create_{nvfp4,mxfp8}_attention_tensors)
# ---------------------------------------------------------------------------------------------


def _swizzled_sf(sf_data, batch, seqlen, nheads, headdim, sf_vec_size):
    """flashinfer 128x4-swizzled scale factors -> the upstream 7-D view (32,4,rest_m,4,rest_k,h,b).

    Returns ``(storage, view)``: ``storage`` is the contiguous ``(b, h, rest_m, rest_k, 32, 4, 4)``
    byte tensor the TIRx TensorMaps read, ``view`` is the permuted view the upstream interface takes.
    """
    rest_m = seqlen // BLK_M
    rest_k = headdim // sf_vec_size // 4
    sf = sf_data.reshape(batch * rest_m, nheads * (headdim // sf_vec_size) // 4, 32, 4, 4)
    sf = sf.reshape(batch, rest_m, nheads, rest_k, 32, 4, 4)
    storage = sf.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    return storage, storage.permute(4, 5, 2, 6, 3, 1, 0)


def _quantize_nvfp4(ref_bf16_2d):
    import torch
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    one = torch.ones(1, device=ref_bf16_2d.device, dtype=torch.float32)
    return nvfp4_quantize(ref_bf16_2d, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)


def _quantize_mxfp8(ref_bf16_2d):
    import torch
    from flashinfer.quantization import SfLayout, mxfp8_quantize

    data, sf = mxfp8_quantize(ref_bf16_2d, sf_swizzle_layout=SfLayout.layout_128x4)
    if data.dtype == torch.uint8:
        data = data.view(torch.float8_e4m3fn)
    return data, sf


def prepare_data(
    qk_format,
    pv_format,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seed=0,
    device="cuda",
):
    """Quantized inputs shared by the TIRx kernel and the upstream reference.

    Everything here happens outside the timed closures. Both implementations read the same
    bytes; only the outputs are separate.
    """
    import torch

    torch.manual_seed(seed)
    b, sq, sk, h, hk, d = batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim
    q_ref = torch.randn(b, sq, h, d, device=device, dtype=torch.float32)
    k_ref = torch.randn(b, sk, hk, d, device=device, dtype=torch.float32)
    v_ref = torch.randn(b, sk, hk, d, device=device, dtype=torch.float32)
    quantize = _quantize_nvfp4 if qk_format == "nvfp4" else _quantize_mxfp8
    sf_vec = _QK_SF_VEC[qk_format]

    def qk(ref, seqlen, nheads):
        data, sf = quantize(ref.to(torch.bfloat16).reshape(b * seqlen, nheads * d))
        if qk_format == "nvfp4":
            data = (
                data.reshape(b, seqlen, nheads, d // 2)
                .view(torch.uint8)
                .view(torch.float4_e2m1fn_x2)
            )
        else:
            data = data.reshape(b, seqlen, nheads, d)
        storage, view = _swizzled_sf(sf, b, seqlen, nheads, d, sf_vec)
        return data, storage, view, sf

    q, q_sf_storage, q_sf, q_sf_raw = qk(q_ref, sq, h)
    k, k_sf_storage, k_sf, k_sf_raw = qk(k_ref, sk, hk)

    v_sf_storage = v_sf = v_sf_raw = None
    if pv_format == "bf16":
        v = v_ref.to(torch.bfloat16)
    elif pv_format == "fp8":
        v = v_ref.to(torch.bfloat16).to(torch.float8_e4m3fn)
    else:
        # K-major V: quantize (b*hk*d, sk) so seqlen is the contiguous, scaled dimension.
        v_km = v_ref.to(torch.bfloat16).permute(0, 2, 3, 1).contiguous().reshape(b * hk * d, sk)
        if pv_format == "nvfp4":
            data, v_sf_raw = _quantize_nvfp4(v_km)
            v = data.reshape(b, hk, d, sk // 2).view(torch.uint8).view(torch.float4_e2m1fn_x2)
        else:
            data, v_sf_raw = _quantize_mxfp8(v_km)
            v = data.reshape(b, hk, d, sk).permute(
                0, 3, 1, 2
            )  # logical (b, s, hk, d), s contiguous
        v_sf_storage, v_sf = _swizzled_sf(v_sf_raw, b * hk, d, 1, sk, _PV_SF_VEC[pv_format])

    out_tirx = torch.empty(b, sq, h, d, device=device, dtype=torch.bfloat16)
    out_ref = torch.empty_like(out_tirx)
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_sf": q_sf,
        "k_sf": k_sf,
        "v_sf": v_sf,
        "q_sf_storage": q_sf_storage,
        "k_sf_storage": k_sf_storage,
        "v_sf_storage": v_sf_storage,
        "q_sf_raw": q_sf_raw,
        "k_sf_raw": k_sf_raw,
        "v_sf_raw": v_sf_raw,
        "q_ref": q_ref,
        "k_ref": k_ref,
        "v_ref": v_ref,
        "out_tirx": out_tirx,
        "out_ref": out_ref,
        "softmax_scale": 1.0 / math.sqrt(d),
    }


# ---------------------------------------------------------------------------------------------
# Upstream reference
# ---------------------------------------------------------------------------------------------


def _reference_kwargs(data, config):
    return dict(
        softmax_scale=data["softmax_scale"],
        causal=config["is_causal"],
        pack_gqa=False,  # see Spec: the upstream packed-GQA FP4 path is numerically wrong
        mSFQ=data["q_sf"],
        mSFK=data["k_sf"],
        mSFV=data["v_sf"],
        out=data["out_ref"],
    )


def run_reference(data, config):
    """Run the upstream FP4 forward on ``data`` through its production dispatch."""
    _pin_upstream_knobs()
    from flash_attn.cute.interface import _flash_attn_fwd

    out, _ = _flash_attn_fwd(data["q"], data["k"], data["v"], **_reference_kwargs(data, config))
    return out


def build_reference_launch(data, config):
    """A no-argument closure that launches exactly the upstream kernel object.

    The first production call compiles through the fork's own dispatch; the closure then
    reuses the compiled ``tvm_ffi`` callable with the very same call arguments the interface
    passes (interface.py:1002-1052), so the timed work is the kernel launch alone.
    """
    import torch

    _pin_upstream_knobs()
    from flash_attn.cute.interface import _flash_attn_fwd

    kwargs = _reference_kwargs(data, config)
    before = set(_flash_attn_fwd.compile_cache)
    _flash_attn_fwd(data["q"], data["k"], data["v"], **kwargs)
    torch.cuda.synchronize()
    new_keys = set(_flash_attn_fwd.compile_cache) - before
    if len(new_keys) != 1:
        raise RuntimeError(f"expected one new compile key, got {len(new_keys)}")
    (key,) = new_keys
    compiled = _flash_attn_fwd.compile_cache[key]
    captured = []

    def capture(*args):
        captured.append(args)
        return compiled(*args)

    _flash_attn_fwd.compile_cache[key] = capture
    try:
        _flash_attn_fwd(data["q"], data["k"], data["v"], **kwargs)
    finally:
        _flash_attn_fwd.compile_cache[key] = compiled
    (call_args,) = captured

    def launch():
        compiled(*call_args)

    launch._keep_alive = (call_args, data)
    return launch


# ---------------------------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------------------------


def _bf16_ulp_distance(a, b):
    import torch

    ia = a.view(torch.int16).to(torch.int32)
    ib = b.view(torch.int16).to(torch.int32)
    ia = torch.where(ia < 0, 0x8000 - ia, ia)
    ib = torch.where(ib < 0, 0x8000 - ib, ib)
    return (ia - ib).abs()


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _sf_swizzled_index(rows, cols, sf_vec, device):
    """Byte index into flashinfer's 128x4-swizzled scale buffer for each (row, k-group).

    Row tiles of 128 x (4 groups) are 512-byte chunks laid out [row//128][group//4]; inside a
    chunk the byte lives at (row%32)*16 + ((row%128)//32)*4 + group%4 -- the same
    ``((32,4),(sf_vec,4)) : ((16,4),(0,1))`` atom the kernels read.
    """
    import torch

    m = torch.arange(rows, device=device)[:, None]
    kg = torch.arange(cols // sf_vec, device=device)[None, :]
    num_k_tiles = (cols // sf_vec + 3) // 4
    return (
        ((m // 128) * num_k_tiles + kg // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (kg % 4)
    )


def _dequantize(qk_format, data_2d, sf_raw, sf_vec):
    """Inverse of the flashinfer quantizers on their raw 2-D outputs (fp32 result on device).

    Verified bit-identical to ``e2m1_and_ufp8sf_scale_to_float`` for NVFP4; the MXFP8 host
    dequantizer of flashinfer 0.6.18 segfaults, so both formats are decoded here in torch.
    """
    import torch

    rows = data_2d.shape[0]
    device = data_2d.device
    if qk_format == "nvfp4":
        packed = data_2d.view(torch.uint8).reshape(rows, -1)
        cols = packed.shape[1] * 2
        nib = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, cols).to(torch.int64)
        table = torch.tensor(_E2M1_VALUES, device=device)
        vals = table[nib & 7] * torch.where(nib >= 8, -1.0, 1.0)
        idx = _sf_swizzled_index(rows, cols, sf_vec, device)
        sf = sf_raw.view(torch.uint8).flatten()[idx].view(torch.float8_e4m3fn).float()
    else:
        vals = data_2d.view(torch.float8_e4m3fn).reshape(rows, -1).float()
        cols = vals.shape[1]
        idx = _sf_swizzled_index(rows, cols, sf_vec, device)
        e = sf_raw.view(torch.uint8).flatten()[idx].to(torch.int32)
        sf = torch.where(e == 0, torch.zeros((), device=device), torch.exp2((e - 127).float()))
    return vals * sf.repeat_interleave(sf_vec, dim=1)


def dequantized_inputs(data, config):
    """fp32 Q/K/V exactly as the kernels see them (block scales applied)."""

    b, sq, sk = config["batch_size"], config["seq_len_q"], config["seq_len_kv"]
    h, hk, d = config["num_qo_heads"], config["num_kv_heads"], config["head_dim"]
    qk_format, pv_format = config["qk_format"], config["pv_format"]
    sf_vec = _QK_SF_VEC[qk_format]
    q = _dequantize(qk_format, data["q"].reshape(b * sq, -1), data["q_sf_raw"], sf_vec).reshape(
        b, sq, h, d
    )
    k = _dequantize(qk_format, data["k"].reshape(b * sk, -1), data["k_sf_raw"], sf_vec).reshape(
        b, sk, hk, d
    )
    if pv_format in ("bf16", "fp8"):
        v = data["v"].float()
    else:
        fmt = "nvfp4" if pv_format == "nvfp4" else "mxfp8"
        v_km = (
            data["v"].reshape(b, hk, d, -1)
            if pv_format == "nvfp4"
            else data["v"].permute(0, 2, 3, 1)
        )
        v_km = v_km.reshape(b * hk * d, -1).contiguous()
        v = _dequantize(fmt, v_km, data["v_sf_raw"], _PV_SF_VEC[pv_format]).reshape(b, hk, d, sk)
        v = v.permute(0, 3, 1, 2)
    return q, k, v.contiguous()


def reference_attention_fp64(q, k, v, softmax_scale, is_causal):
    """Plain attention in fp64 on already-dequantized inputs; GQA by head repetition."""
    import torch

    b, sq, h, d = q.shape
    hk = k.shape[2]
    q64 = q.double().permute(0, 2, 1, 3)
    k64 = k.double().permute(0, 2, 1, 3).repeat_interleave(h // hk, dim=1)
    v64 = v.double().permute(0, 2, 1, 3).repeat_interleave(h // hk, dim=1)
    s = torch.matmul(q64, k64.transpose(-1, -2)) * softmax_scale
    if is_causal:
        sk = k.shape[1]
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        s = s.masked_fill(col > row + (sk - sq), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v64).permute(0, 2, 1, 3)


def _error_stats(out, ref64):
    import torch

    o = out.double()
    err = (o - ref64).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref64.flatten(), dim=0).item()
    return err, cos


# ---------------------------------------------------------------------------------------------
# Host TensorMaps and launch binding
# ---------------------------------------------------------------------------------------------


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte CUtensorMap."""

    def __init__(self):
        import ctypes

        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(tensor, dtype, dims, strides, box, swizzle_bytes):
    """cuTensorMapEncodeTiled: dims/box innermost first, strides in bytes for dims[1:]."""
    import ctypes

    import tvm

    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    desc = _AlignedTensorMap()
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * rank),
        0,
        _TMA_SWIZZLE[swizzle_bytes],
        _TMA_L2_PROMOTION_128B,
        0,
    )
    return desc


def build_tensor_maps(spec: Spec, data, config):
    """The kernel's TensorMap parameters, in declaration order (sketch: TensorMap table)."""
    b, sq, sk = config["batch_size"], config["seq_len_q"], config["seq_len_kv"]
    h, hk, d = config["num_qo_heads"], config["num_kv_heads"], config["head_dim"]
    qk_bytes = d * spec.q_width // 8
    qk_sw = qk_bytes if qk_bytes <= 128 else 128

    def qk_map(t, s, nh):
        return _encode_tensor_map(
            t,
            "uint8",
            (qk_bytes, s, nh, b),
            (nh * qk_bytes, qk_bytes, s * nh * qk_bytes),
            (qk_bytes, BLK_M, 1, 1),
            qk_sw,
        )

    def sf_map(storage, s, nh, sf_tile_k):
        # storage bytes [b][h][s/128][chunks][512]; u16 elements, 256 per chunk
        blocks = s // BLK_M
        if sf_tile_k == 2:
            dims = (256, 2, blocks, nh, b)
            strides = (512, 1024, 1024 * blocks, 1024 * blocks * nh)
            box = (256, 2, 1, 1, 1)
        else:
            dims = (256, blocks, 1, nh, b)
            strides = (512, 512 * blocks, 512 * blocks, 512 * blocks * nh)
            box = (256, 1, 1, 1, 1)
        return _encode_tensor_map(storage, "uint16", dims, strides, box, 0)

    sf_tile_k = d // (4 * spec.sf_vec_size)
    maps = [qk_map(data["q"], sq, h), qk_map(data["k"], sk, hk)]
    v = data["v"]
    if spec.pv_format == "bf16":
        maps.append(
            _encode_tensor_map(
                v,
                "bfloat16",
                (d, sk, hk, b),
                (hk * d * 2, d * 2, sk * hk * d * 2),
                (64, BLK_N, 1, 1),
                128,
            )
        )
    elif spec.pv_format == "fp8":
        maps.append(
            _encode_tensor_map(
                v, "uint8", (d, sk, hk, b), (hk * d, d, sk * hk * d), (d, BLK_N, 1, 1), min(d, 128)
            )
        )
    elif spec.pv_format == "nvfp4":  # storage (b, hk, d, sk/2) bytes
        maps.append(
            _encode_tensor_map(
                v,
                "uint8",
                (sk // 2, d, hk, b),
                (sk // 2, d * sk // 2, hk * d * sk // 2),
                (64, BLK_N, 1, 1),
                64,
            )
        )
    else:  # mxfp8 K-major, storage (b, hk, d, sk) bytes
        maps.append(
            _encode_tensor_map(
                v, "uint8", (sk, d, hk, b), (sk, d * sk, hk * d * sk), (BLK_N, BLK_N, 1, 1), 128
            )
        )
    o = data["out_tirx"]
    maps.append(
        _encode_tensor_map(
            o, "bfloat16", (d, sq, h, b), (h * d * 2, d * 2, sq * h * d * 2), (64, BLK_M, 1, 1), 128
        )
    )
    maps.append(sf_map(data["q_sf_storage"], sq, h, sf_tile_k))
    maps.append(sf_map(data["k_sf_storage"], sk, hk, sf_tile_k))
    if spec.quant_pv:
        # storage [b*hk][DV/128][sk/(4 sfv)] chunks
        sfv = spec.sf_vec_size_pv
        kchunks = sk // (4 * sfv)
        dblocks = d // 128
        if spec.pv_format == "nvfp4":
            dims = (256, kchunks, dblocks, hk, b)
            strides = (512, 512 * kchunks, 512 * kchunks * dblocks, 512 * kchunks * dblocks * hk)
            box = (256, 2, 1, 1, 1)
        else:
            dims = (256, 1, kchunks, hk, b)
            strides = (512, 512, 512 * kchunks * dblocks, 512 * kchunks * dblocks * hk)
            box = (256, 1, 1, 1, 1)
        maps.append(_encode_tensor_map(data["v_sf_storage"], "uint16", dims, strides, box, 0))
    return maps


def build_tirx_launch(executable, data, config):
    """Bind the compiled kernel to one data set; returns a no-argument launch closure."""
    spec = Spec(
        config["qk_format"],
        config["pv_format"],
        config["head_dim"],
        config["is_causal"],
        config["num_qo_heads"],
        config["num_kv_heads"],
    )
    maps = build_tensor_maps(spec, data, config)
    scale_log2 = float(data["softmax_scale"] * math.log2(math.e))
    argv = (*[m.ptr for m in maps], scale_log2)

    def launch():
        executable(*argv)

    launch._keep_alive = (data, maps, argv)
    return launch


def run_test(
    qk_format,
    pv_format,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    **kwargs,
):
    """Compile, run, and verify against the upstream kernel on identical quantized bytes."""
    from unittest import SkipTest

    import torch

    from tirx_kernels.runner import compile_kernel

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise SkipTest("flash_attention4_fp4 requires an SM103 (GB300) GPU")
    config = dict(
        qk_format=qk_format,
        pv_format=pv_format,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )
    data = prepare_data(**{k: v for k, v in config.items() if k != "is_causal"})
    executable = compile_kernel(get_kernel(**config))
    launch = build_tirx_launch(executable, data, config)
    launch()
    torch.cuda.synchronize()
    out = data["out_tirx"]
    if not torch.isfinite(out).all():
        raise AssertionError("output contains a non-finite value")
    # Determinism of the port itself.
    first = out.clone()
    launch()
    torch.cuda.synchronize()
    if not torch.equal(out, first):
        raise AssertionError("TIRx output differs between two identical launches")

    # Primary gate: the upstream kernel on the same bytes must agree bit for bit. It is launched
    # twice first: without a proxy fence between the softmax warps' sSFP store and the MMA warp's
    # tcgen05.cp read, the upstream kernel is nondeterministic on multi-tile persistent shapes
    # with a quantized P, and a racy reference cannot anchor a bitwise gate.
    ref = run_reference(data, config).clone()
    ref_again = run_reference(data, config)
    torch.cuda.synchronize()
    if not torch.isfinite(ref).all():
        raise AssertionError("reference output contains a non-finite value")
    if not torch.equal(ref, ref_again):
        raise AssertionError(
            f"upstream kernel is nondeterministic on {int((ref != ref_again).sum())} elements; "
            "the upstream sSFP store lacks a proxy fence before the tcgen05.cp read"
        )
    if not torch.equal(out, ref):
        ulps = _bf16_ulp_distance(out, ref)
        mism = ulps != 0
        rows = mism.any(dim=-1).nonzero()
        raise AssertionError(
            f"bitwise mismatch vs upstream: {int(mism.sum())} of {mism.numel()} elements, "
            f"max {int(ulps.max())} bf16 ulp; first rows (b, s, h): {rows[:8].tolist()}"
        )

    # Structural oracle: both kernels against fp64 attention on the dequantized inputs.
    q, k, v = dequantized_inputs(data, config)
    ref64 = reference_attention_fp64(q, k, v, data["softmax_scale"], is_causal)
    err_t, cos_t = _error_stats(out, ref64)
    err_r, cos_r = _error_stats(ref, ref64)
    if cos_r <= 0.98:
        raise AssertionError(
            f"upstream kernel vs fp64 cos_sim {cos_r:.5f} <= 0.98: input plumbing is wrong"
        )
    if err_t > err_r * (1 + 1e-3) + 1e-6 or cos_t < cos_r - 1e-6:
        raise AssertionError(
            f"TIRx err {err_t:.5g} / cos {cos_t:.6f} worse than upstream err {err_r:.5g} / cos {cos_r:.6f}"
        )


# ---------------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------------


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU (CUDA stays uninitialized)."""
    from tirx_kernels.runner import compile_kernel, hardware_num_sms, prepared_gpu_benchmark

    state = {
        "config": dict(kwargs),
        "num_sms": hardware_num_sms(),
        "executable": compile_kernel(get_kernel(**kwargs)),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup=None,
    repeat=None,
    timer=None,  # None inherits proton: the CuTeDSL reference cannot be CUDA-graph captured.
    **kwargs,
):
    import torch

    from tirx_kernels.runner import bench

    config = dict(prepared["config"])
    if torch.cuda.get_device_properties(0).multi_processor_count != prepared["num_sms"]:
        raise RuntimeError(
            "the persistent grid was compiled for a different SM count; set TIRX_PREPARE_NUM_SMS"
        )
    data = prepare_data(**{k: v for k, v in config.items() if k != "is_causal"})
    launch = build_tirx_launch(prepared["executable"], data, config)
    return bench(
        {"tirx": launch},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_fp4": lambda: build_reference_launch(data, config)},
        **kwargs,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, **kwargs):
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    return prepare_bench(**config).run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)
