# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ cc6e8794), Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell MSA BF16-query/FP8-KV paged Q1 decode for SM103a.

Upstream sources (FlashInfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839):

- csrc/blackwell_msa/sm103a/
  blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged.cu;
- csrc/blackwell_msa/sm103a/
  blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_binding.cu;
- flashinfer/msa_ops/_blackwell_sm100.py.

The source specialization serves causal eager Q1 decode with 128 requests,
64 BF16 query heads, four paged FP8 E4M3 KV heads, D=128, and TopK16.
"""

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_103a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "cc6e8794c49bf66172627bdb9742fcb17d18b839",
            },
            "import": "flashinfer",
        },
    ),
}

CUDA_ARCH = "sm_103a"
NUM_REQUESTS = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 128
MAX_PAGES = 32
TOPK = 16
NUM_WARPS = 16
SMEM_TOTAL = 216704


def _config(label: str, *, pattern: str, seed: int) -> dict[str, Any]:
    return {"label": label, "pattern": pattern, "seed": seed}


CONFIGS = [
    _config("decode_fp8_b128_q1_min_valid_token", pattern="min_valid", seed=11),
    _config("decode_fp8_b128_q1_kv4096_h64_seed53", pattern="upstream", seed=53),
    _config("decode_fp8_b128_q1_ragged_boundaries_h64", pattern="ragged", seed=59),
    _config("decode_fp8_b128_q1_kv4096_h64_scattered_pages", pattern="scattered", seed=61),
    _config("decode_fp8_b128_q1_invalid_page_entries", pattern="invalid_pages", seed=67),
    _config("decode_fp8_b128_q1_numeric_ties_extremes", pattern="numeric", seed=71),
]

BENCH_CONFIGS = [
    _config("decode_fp8_b128_q1_kv4096_h64_seed53", pattern="upstream", seed=53),
    _config("decode_fp8_b128_q1_ragged_boundaries_h64", pattern="ragged", seed=59),
    _config("decode_fp8_b128_q1_kv4096_h64_scattered_pages", pattern="scattered", seed=61),
]


# cuTensorMap enums and the exact rank-three source descriptors.
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    num_requests = params["num_requests"]

    def encode(tensor, dtype, dims, strides_bytes, box, swizzle):
        descriptor = K.stack_alloca("tensormap", 1)
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            dtype,
            3,
            tensor.data,
            *dims,
            *strides_bytes,
            *box,
            1,
            1,
            1,
            _TMA_INTERLEAVE_NONE,
            swizzle,
            _TMA_L2_PROMOTION_NONE,
            _TMA_OOB_FILL_NONE,
        )
        return descriptor

    q_map = encode(
        params["q"],
        "bfloat16",
        (64, num_requests * num_q_heads, 2),
        (HEAD_DIM * 2, 64 * 2),
        (64, 16, 2),
        _TMA_SWIZZLE_128B,
    )
    # Paged inputs are contiguous (num_pages, num_kv_heads, 128, 128).
    num_page_heads = num_requests * params["msa_max_pages"] * num_kv_heads
    kv_dims = (HEAD_DIM, PAGE_SIZE, num_page_heads)
    kv_strides = (HEAD_DIM, PAGE_SIZE * HEAD_DIM)
    kv_box = (HEAD_DIM, 64, 1)
    k_map = encode(params["k"], "uint8", kv_dims, kv_strides, kv_box, _TMA_SWIZZLE_NONE)
    v_map = encode(params["v"], "uint8", kv_dims, kv_strides, kv_box, _TMA_SWIZZLE_NONE)
    return q_map, k_map, v_map


# Byte offsets in the source's one rank-one dynamic shared-memory arena.
_MBAR_Q_FULL = 0
_MBAR_Q_EMPTY = 8
_MBAR_KV_FULL = 16
_MBAR_KV_SRC_FULL = 80
_MBAR_KV_EMPTY = 112
_MBAR_S_FULL = 144
_MBAR_P_FULL = 160
_MBAR_CORR_SIG = 176
_MBAR_CORR_DONE = 192
_MBAR_O_FULL = 208
_MBAR_DECODE_DONE = 224
_SMEM_TMEM_MAILBOX = 232
_SMEM_CORR = (1024, 1088)
_SMEM_EXCH = (1152, 1408)
_SMEM_QT = 1664
_SMEM_KV = 6144
_SMEM_KV_STAGE_BYTES = 32768
_SMEM_P = (137216, 141312)
_SMEM_PAGE_INDICES = 145408
_SMEM_ROW_MAX = 150144
_SMEM_ROW_SUM = 150656
_SMEM_KV_FP8 = 151168
_SMEM_KV_FP8_STAGE_BYTES = 16384

_TMEM_S = (0, 16)
_TMEM_O = (32, 48)
_TMEM_STATS = (64, 80)
_TMEM_COLS_ALLOC = 128
_DESC_HI = 0x40004040
_QK_IDESC = 0x08040490
_PV_IDESC = 0x08048490
_QK_A_OFFSETS = (0, 2, 4, 6, 1024, 1026, 1028, 1030)
_QK_B_OFFSETS = (0, 2, 4, 6, 128, 130, 132, 134)
_PV_A_OFFSETS = (0, 128, 256, 384, 512, 640, 768, 896)
_PV_B_OFFSETS = (0, 2, 4, 6, 128, 130, 132, 134)
_REG_SOFTMAX = 192
_REG_CORRECTION = 80
_REG_OTHER = 48
_FULL_MASK = 0xFFFFFFFF
_NEG_INF = float("-inf")
_LN2_F32 = 0.6931471805599453

_TMA_G2S_3D = "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"


def _u32(value):
    return K.uint32(value)


def _i32(value):
    return K.int32(value)


def _f32(value):
    return K.float32(value)


def _mbar_wait(addr, phase):
    K.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, tx_bytes):
    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(tx_bytes))


def _load_shared_f32(addr):
    value = K.local_scalar("float32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _flip(phase):
    K.assign(phase, phase ^ _i32(1))


def _pack2(lo, hi):
    packed = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, lo, hi)
    return packed


def _max_f32(a, b):
    out = K.local_scalar("float32")
    K.ptx.max.f32(out, a, b)
    return out


def _shfl_xor_f32(value, lane_xor):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), _u32(lane_xor), _u32(31), _u32(_FULL_MASK)
    )
    return K.reinterpret("float32", out)


def _shfl_idx_f32(value, source_lane):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.idx.b32(
        out, K.reinterpret("uint32", value), _u32(source_lane), _u32(31), _u32(_FULL_MASK)
    )
    return K.reinterpret("float32", out)


def _tmem_load_x16(dst, addr):
    K.ptx[_TMEM_LD_X16](*(dst[i] for i in range(16)), addr)


def _tmem_store_x16(addr, src):
    K.ptx[_TMEM_ST_X16](addr, *(src[i] for i in range(16)))


def _build_kernel():
    @K.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: [p["num_requests"], p["num_kv_heads"]],
        host_prelude=_host_prelude,
    )
    def blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103(
        q: K.gptr[K.bf16],
        k: K.gptr[K.u8],
        v: K.gptr[K.u8],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        page_table: K.gptr[K.i32],
        cu_k: K.gptr[K.i32],
        q2k_indices: K.gptr[K.i32],
        q_offsets: K.gptr[K.i32],
        kv_lens: K.gptr[K.i32],
        num_requests: K.i32,
        num_q_heads: K.i32,
        num_kv_heads: K.i32,
        softmax_scale_log2: K.f32,
        msa_max_pages: K.i32,
        *,
        host,
    ):
        q_map, k_map, v_map = host
        del q, k, v, cu_k, q_offsets
        # >>> kernel_blackwell_batch_attention_msa_decode_q1_fp8_paged_xform2_v1 body starts here
        bid_x, bid_y = K.cta_id()
        tid = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()

        arena = K.alloc_buffer((SMEM_TOTAL,), K.u8, scope="shared.dyn", align=1024)
        smem = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        init_layout = (
            (_MBAR_Q_FULL, 1),
            (_MBAR_Q_EMPTY, 1),
            *tuple((_MBAR_KV_FULL + 8 * i, 1) for i in range(8)),
            *tuple((_MBAR_KV_SRC_FULL + 8 * i, 1) for i in range(4)),
            *tuple((_MBAR_KV_EMPTY + 8 * i, 1) for i in range(4)),
            (_MBAR_S_FULL, 1),
            (_MBAR_S_FULL + 8, 1),
            (_MBAR_P_FULL, 256),
            (_MBAR_P_FULL + 8, 256),
            (_MBAR_CORR_SIG, 128),
            (_MBAR_CORR_SIG + 8, 128),
            (_MBAR_CORR_DONE, 128),
            (_MBAR_CORR_DONE + 8, 128),
            (_MBAR_O_FULL, 1),
            (_MBAR_O_FULL + 8, 1),
            (_MBAR_DECODE_DONE, 128),
        )
        with K.If(warp == 0), K.Then():
            init_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(init_leader != _u32(0)), K.Then():
                for offset, count in init_layout:
                    K.ptx.mbarrier.init.shared__cta.b64(bar(offset), _u32(count))
                K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.warp.sync(_u32(_FULL_MASK))

        with K.If(warp == 0), K.Then():
            K.ptx[_TMEM_ALLOC](bar(_SMEM_TMEM_MAILBOX), _u32(_TMEM_COLS_ALLOC))
            K.ptx[_TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        K.ptx["tcgen05.fence::after_thread_sync"]()
        taddr = K.local_scalar("uint32")
        K.ptx.ld.volatile.shared.b32(taddr, bar(_SMEM_TMEM_MAILBOX))

        total_work = num_requests * num_kv_heads
        first_work = bid_x * num_kv_heads + bid_y
        work_stride = total_work

        roles = K.specialize(chain_dispatch=False)
        other_regs = roles.register_scope("other", warps=range(12, 16), regs=_REG_OTHER)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=_REG_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=_REG_CORRECTION)
        r_mma = roles.role("mma", warps=[12], register_scope=other_regs)
        r_producer = roles.role("producer", warps=[13], register_scope=other_regs)
        r_aux = roles.role("aux", warps=range(14, 16), register_scope=other_regs)

        with K.If(K.And(warp >= 12, warp <= 15)), K.Then():
            other_regs.emit()

        with r_softmax:
            phase_s = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            phase_o = K.local_scalar("int32", init=_i32(0))
            work_s = K.local_scalar("int32", init=first_work)
            with K.While(work_s < total_work):
                instance = K.uniform(K.if_then_else(warp >= 4, _i32(1), _i32(0)))
                warp_in_wg = warp % 4
                wg_tid = warp_in_wg * 32 + lane
                row_base = warp * 16
                tmem_row = warp_in_wg * 32
                my_s = (
                    taddr
                    + K.cast(instance * 16, "uint32")
                    + K.shift_left(K.cast(tmem_row, "uint32"), _u32(16))
                )
                my_stats = (
                    taddr
                    + K.cast(64 + instance * 16, "uint32")
                    + K.shift_left(K.cast(tmem_row, "uint32"), _u32(16))
                )
                exch_base = K.local_scalar("uint32", init=bar(_SMEM_EXCH[0]))
                corr_base = K.local_scalar("uint32", init=bar(_SMEM_CORR[0]))
                p_base = K.local_scalar("uint32", init=bar(_SMEM_P[0]))
                with K.If(instance != _i32(0)), K.Then():
                    K.assign(exch_base, bar(_SMEM_EXCH[1]))
                    K.assign(corr_base, bar(_SMEM_CORR[1]))
                    K.assign(p_base, bar(_SMEM_P[1]))

                # Each head's row state is owned by its corresponding lane.
                with K.If(lane < 16), K.Then():
                    K.ptx.st.shared.b32(
                        bar(_SMEM_ROW_MAX) + K.cast(row_base + lane, "uint32") * _u32(4),
                        _f32(_NEG_INF),
                    )
                    K.ptx.st.shared.b32(
                        bar(_SMEM_ROW_SUM) + K.cast(row_base + lane, "uint32") * _u32(4), _f32(0.0)
                    )

                pair = K.local_scalar("int32", init=_i32(0))
                with K.While(pair < _i32(8)):
                    with K.If(instance == _i32(0)):
                        with K.Then():
                            _mbar_wait(bar(_MBAR_S_FULL), phase_s[0])
                            _flip(phase_s[0])
                        with K.Else():
                            _mbar_wait(bar(_MBAR_S_FULL + 8), phase_s[1])
                            _flip(phase_s[1])

                    score = K.alloc_local((16,), "float32")
                    _tmem_load_x16(score, my_s)
                    valid_cols = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(
                        valid_cols,
                        bar(_SMEM_PAGE_INDICES) + K.cast(pair * 2 + instance, "uint32") * _u32(4),
                    )
                    token = warp_in_wg * 32 + lane
                    with K.If(token >= valid_cols), K.Then():
                        for h in range(16):
                            K.assign(score[h], _f32(_NEG_INF))

                    for h in range(16):
                        for delta in (16, 8, 4, 2, 1):
                            K.assign(score[h], _max_f32(score[h], _shfl_xor_f32(score[h], delta)))

                    with K.If(lane < 16), K.Then():
                        K.ptx.st.shared.b32(
                            exch_base + K.cast(warp_in_wg * 16 + lane, "uint32") * _u32(4),
                            score[lane],
                        )
                    with K.If(instance == _i32(0)):
                        with K.Then():
                            K.ptx.barrier.sync(8, 128)
                        with K.Else():
                            K.ptx.barrier.sync(9, 128)

                    tile_max_lane = K.local_scalar("float32")
                    with K.If(lane < 16), K.Then():
                        max01 = _max_f32(
                            _load_shared_f32(exch_base + K.cast(lane, "uint32") * _u32(4)),
                            _load_shared_f32(exch_base + K.cast(16 + lane, "uint32") * _u32(4)),
                        )
                        max23 = _max_f32(
                            _load_shared_f32(exch_base + K.cast(32 + lane, "uint32") * _u32(4)),
                            _load_shared_f32(exch_base + K.cast(48 + lane, "uint32") * _u32(4)),
                        )
                        K.assign(tile_max_lane, _max_f32(max01, max23))
                    new_max_lane = K.local_scalar("float32", init=_f32(_NEG_INF))
                    acc_scale_lane = K.local_scalar("float32", init=_f32(1.0))
                    with K.If(lane < 16), K.Then():
                        state_addr = bar(_SMEM_ROW_MAX) + K.cast(row_base + lane, "uint32") * _u32(
                            4
                        )
                        old_max = _load_shared_f32(state_addr)
                        K.assign(new_max_lane, _max_f32(old_max, tile_max_lane))
                        K.ptx.st.shared.b32(state_addr, new_max_lane)
                        delta = K.local_scalar("float32")
                        K.ptx.sub.rn.f32(delta, old_max, new_max_lane)
                        K.ptx.mul.rn.f32(delta, softmax_scale_log2, delta)
                        exp_delta = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_delta, delta)
                        K.assign(
                            acc_scale_lane,
                            K.if_then_else(old_max > _f32(_NEG_INF), exp_delta, _f32(1.0)),
                        )

                    tile_max = K.alloc_local((16,), "float32")
                    acc_scale = K.alloc_local((16,), "float32")
                    for h in range(16):
                        K.assign(tile_max[h], _shfl_idx_f32(new_max_lane, h))
                        K.assign(acc_scale[h], _shfl_idx_f32(acc_scale_lane, h))

                    _tmem_store_x16(my_stats, acc_scale)
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    K.ptx.fence.proxy.async_.shared__cta()
                    with K.If(instance == _i32(0)):
                        with K.Then():
                            _mbar_arrive(bar(_MBAR_CORR_SIG))
                        with K.Else():
                            _mbar_arrive(bar(_MBAR_CORR_SIG + 8))

                    _tmem_load_x16(score, my_s)
                    with K.If(token >= valid_cols), K.Then():
                        for h in range(16):
                            K.assign(score[h], _f32(_NEG_INF))
                    exp_values = K.alloc_local((16,), "float32")
                    for h in range(16):
                        new_max = tile_max[h]
                        safe_max = K.if_then_else(new_max == _f32(_NEG_INF), _f32(0.0), new_max)
                        max_scaled = K.local_scalar("float32")
                        K.ptx["mul.f32"](max_scaled, safe_max, softmax_scale_log2)
                        score_scaled = K.local_scalar("float32")
                        K.ptx["mul.f32"](score_scaled, score[h], softmax_scale_log2)
                        K.ptx["sub.f32"](exp_values[h], score_scaled, max_scaled)
                        K.ptx.ex2.approx.ftz.f32(exp_values[h], exp_values[h])

                        p_elem = (wg_tid // 64 * 16 + h) * 128 + (wg_tid % 64) * 2
                        p_byte = p_elem ^ (((p_elem >> 7) & 7) << 4)
                        p_bits = K.local_scalar("uint16")
                        K.ptx.cvt.rn.bf16.f32(p_bits, exp_values[h])
                        K.ptx.st.shared.b16(p_base + K.cast(p_byte, "uint32"), p_bits)

                    for h in range(16):
                        for delta in (16, 8, 4, 2, 1):
                            shuffled = _shfl_xor_f32(exp_values[h], delta)
                            K.ptx.add.rn.f32(exp_values[h], exp_values[h], shuffled)
                        with K.If(lane == h), K.Then():
                            sum_addr = bar(_SMEM_ROW_SUM) + K.cast(row_base + h, "uint32") * _u32(4)
                            old_sum = _load_shared_f32(sum_addr)
                            new_sum = K.local_scalar("float32")
                            K.ptx.fma.rn.f32(new_sum, old_sum, acc_scale[h], exp_values[h])
                            K.ptx.st.shared.b32(sum_addr, new_sum)
                    K.ptx.fence.proxy.async_()
                    with K.If(instance == _i32(0)):
                        with K.Then():
                            _mbar_arrive(bar(_MBAR_P_FULL))
                        with K.Else():
                            _mbar_arrive(bar(_MBAR_P_FULL + 8))
                    K.assign(pair, pair + _i32(1))

                with K.If(instance == _i32(0)):
                    with K.Then():
                        K.ptx.barrier.sync(8, 128)
                    with K.Else():
                        K.ptx.barrier.sync(9, 128)
                with K.If(lane < 16), K.Then():
                    K.ptx.st.shared.b32(
                        exch_base + K.cast(warp_in_wg * 16 + lane, "uint32") * _u32(4),
                        _load_shared_f32(
                            bar(_SMEM_ROW_SUM) + K.cast(row_base + lane, "uint32") * _u32(4)
                        ),
                    )
                with K.If(instance == _i32(0)):
                    with K.Then():
                        K.ptx.barrier.sync(8, 128)
                    with K.Else():
                        K.ptx.barrier.sync(9, 128)

                total_sum_lane = K.local_scalar("float32")
                with K.If(lane < 16), K.Then():
                    sum0 = _load_shared_f32(exch_base + K.cast(lane, "uint32") * _u32(4))
                    sum1 = _load_shared_f32(exch_base + K.cast(16 + lane, "uint32") * _u32(4))
                    sum2 = _load_shared_f32(exch_base + K.cast(32 + lane, "uint32") * _u32(4))
                    sum3 = _load_shared_f32(exch_base + K.cast(48 + lane, "uint32") * _u32(4))
                    K.ptx.add.rn.f32(total_sum_lane, sum0, sum1)
                    K.ptx.add.rn.f32(total_sum_lane, total_sum_lane, sum2)
                    K.ptx.add.rn.f32(total_sum_lane, total_sum_lane, sum3)
                with K.If(instance == _i32(0)):
                    with K.Then():
                        K.ptx.barrier.sync(8, 128)
                    with K.Else():
                        K.ptx.barrier.sync(9, 128)
                with K.If(K.And(warp_in_wg == 0, lane < 16)), K.Then():
                    K.ptx.st.shared.b32(
                        corr_base + K.cast(lane, "uint32") * _u32(4), total_sum_lane
                    )
                    K.ptx.st.shared.b32(
                        exch_base + K.cast(lane, "uint32") * _u32(4),
                        _load_shared_f32(
                            bar(_SMEM_ROW_MAX) + K.cast(row_base + lane, "uint32") * _u32(4)
                        ),
                    )
                with K.If(instance == _i32(0)):
                    with K.Then():
                        K.ptx.barrier.sync(8, 128)
                    with K.Else():
                        K.ptx.barrier.sync(9, 128)
                # Final PV completion acknowledges consumption of the last correction signal.
                _mbar_wait(bar(_MBAR_O_FULL), phase_o)
                _flip(phase_o)
                with K.If(instance == _i32(0)):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_CORR_SIG))
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_CORR_SIG + 8))
                K.assign(work_s, work_s + work_stride)

        with r_correction:
            phase_corr = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            phase_o = K.local_scalar("int32", init=_i32(0))
            work_c = K.local_scalar("int32", init=first_work)
            with K.While(work_c < total_work):
                row_base_c = (warp % 4) * 32
                corr_row = K.shift_left(K.cast(row_base_c, "uint32"), _u32(16))
                d_idx = row_base_c + lane
                request = work_c // num_kv_heads
                kv_head = work_c % num_kv_heads
                group_size = num_q_heads // num_kv_heads

                pair_c = K.local_scalar("int32", init=_i32(0))
                with K.While(pair_c < _i32(8)):
                    for instance in range(2):
                        _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr[instance])
                        _flip(phase_corr[instance])
                        K.ptx["tcgen05.fence::after_thread_sync"]()
                        scale = K.alloc_local((16,), "float32")
                        partial = K.alloc_local((16,), "float32")
                        _tmem_load_x16(scale, taddr + _u32(_TMEM_STATS[instance]) + corr_row)
                        _tmem_load_x16(partial, taddr + _u32(_TMEM_O[instance]) + corr_row)
                        for h in range(16):
                            K.ptx.mul.rn.f32(partial[h], partial[h], scale[h])
                        _tmem_store_x16(taddr + _u32(_TMEM_O[instance]) + corr_row, partial)
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _mbar_arrive(bar(_MBAR_P_FULL + 8 * instance))
                    K.assign(pair_c, pair_c + _i32(1))

                _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr[0])
                _flip(phase_corr[0])
                _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr[1])
                _flip(phase_corr[1])

                scale0 = K.alloc_local((16,), "float32")
                scale1 = K.alloc_local((16,), "float32")
                final_sum = K.alloc_local((16,), "float32")
                final_max = K.alloc_local((16,), "float32")
                for h in range(16):
                    max0 = _shfl_idx_f32(_load_shared_f32(bar(_SMEM_EXCH[0] + 4 * h)), h)
                    max1 = _shfl_idx_f32(_load_shared_f32(bar(_SMEM_EXCH[1] + 4 * h)), h)
                    sum0 = _shfl_idx_f32(_load_shared_f32(bar(_SMEM_CORR[0] + 4 * h)), h)
                    sum1 = _shfl_idx_f32(_load_shared_f32(bar(_SMEM_CORR[1] + 4 * h)), h)
                    fm = _max_f32(max0, max1)
                    K.assign(final_max[h], fm)

                    d0 = K.local_scalar("float32", init=_f32(0.0))
                    d1 = K.local_scalar("float32", init=_f32(0.0))
                    with K.If(max0 != _f32(_NEG_INF)), K.Then():
                        K.ptx.sub.rn.f32(d0, max0, fm)
                        K.ptx.mul.rn.f32(d0, softmax_scale_log2, d0)
                    with K.If(max1 != _f32(_NEG_INF)), K.Then():
                        K.ptx.sub.rn.f32(d1, max1, fm)
                        K.ptx.mul.rn.f32(d1, softmax_scale_log2, d1)
                    K.ptx.ex2.approx.ftz.f32(scale0[h], d0)
                    K.ptx.ex2.approx.ftz.f32(scale1[h], d1)
                    sum1_scaled = K.local_scalar("float32")
                    K.ptx.mul.rn.f32(sum1_scaled, sum1, scale1[h])
                    K.ptx.fma.rn.f32(final_sum[h], sum0, scale0[h], sum1_scaled)

                _mbar_wait(bar(_MBAR_O_FULL), phase_o)
                _flip(phase_o)
                K.ptx["tcgen05.fence::after_thread_sync"]()
                inv_sum = K.alloc_local((16,), "float32")
                for h in range(16):
                    reciprocal = K.local_scalar("float32")
                    K.ptx.rcp.approx.ftz.f32(reciprocal, final_sum[h])
                    K.assign(
                        inv_sum[h], K.if_then_else(final_sum[h] > _f32(0.0), reciprocal, _f32(0.0))
                    )

                out0 = K.alloc_local((16,), "float32")
                out1 = K.alloc_local((16,), "float32")
                _tmem_load_x16(out0, taddr + _u32(_TMEM_O[0]) + corr_row)
                _tmem_load_x16(out1, taddr + _u32(_TMEM_O[1]) + corr_row)
                for h in range(16):
                    with K.If(group_size > _i32(h)), K.Then():
                        out1_scaled = K.local_scalar("float32")
                        merged = K.local_scalar("float32")
                        normalized = K.local_scalar("float32")
                        K.ptx.mul.rn.f32(out1_scaled, out1[h], scale1[h])
                        K.ptx.fma.rn.f32(merged, out0[h], scale0[h], out1_scaled)
                        K.ptx.mul.rn.f32(normalized, merged, inv_sum[h])
                        q_row = request * num_q_heads + kv_head * group_size + h
                        with K.If(d_idx == 0), K.Then():
                            natural_lse = K.local_scalar("float32", init=_f32(_NEG_INF))
                            with K.If(final_sum[h] > _f32(0.0)), K.Then():
                                log2_sum = K.local_scalar("float32")
                                max_scaled = K.local_scalar("float32")
                                max_ln = K.local_scalar("float32")
                                K.ptx.lg2.approx.ftz.f32(log2_sum, final_sum[h])
                                K.ptx.mul.rn.f32(max_scaled, final_max[h], softmax_scale_log2)
                                K.ptx.mul.rn.f32(max_ln, max_scaled, _f32(_LN2_F32))
                                K.ptx.fma.rn.f32(natural_lse, log2_sum, _f32(_LN2_F32), max_ln)
                            K.ptx.st.global_.b32(lse.ptr_to([q_row]), natural_lse)
                        out_bits = K.local_scalar("uint16")
                        K.ptx.cvt.rn.bf16.f32(out_bits, normalized)
                        K.ptx.st.global_.b16(out.ptr_to([q_row * HEAD_DIM + d_idx]), out_bits)
                _mbar_arrive(bar(_MBAR_DECODE_DONE))
                K.assign(work_c, work_c + work_stride)
        with r_mma:
            phase_q = K.local_scalar("int32", init=_i32(0))
            phase_p = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            phase_decode = K.local_scalar("int32", init=_i32(0))
            work_m = K.local_scalar("int32", init=first_work)

            def commit_one(addr):
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                K.ptx[_TCGEN05_COMMIT](addr, pred=leader)

            def qk_chain(stage, instance):
                a_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        ((bar(_SMEM_KV) >> 4) & _u32(0x3FFF)) + K.cast(stage, "uint32") * _u32(2048)
                    ),
                )
                b_lo = K.local_scalar("uint32", init=K.uniform((bar(_SMEM_QT) >> 4) & _u32(0x3FFF)))
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                for site in range(8):
                    K.ptx[_MMA_F16](
                        taddr + _u32(_TMEM_S[instance]),
                        _pack2(a_lo + _u32(_QK_A_OFFSETS[site]), _u32(_DESC_HI)),
                        _pack2(b_lo + _u32(_QK_B_OFFSETS[site]), _u32(_DESC_HI)),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(_u32(0 if site == 0 else 1)),
                        pred=leader,
                    )

            def pv_chain(stage, instance, first_pv):
                a_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        (((bar(_SMEM_KV) >> 4) & _u32(0x3FFF)) | _u32(0x04000000))
                        + K.cast(stage, "uint32") * _u32(2048)
                    ),
                )
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        ((bar(_SMEM_P[instance]) >> 4) & _u32(0x3FFF)) | _u32(0x00800000)
                    ),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                enable_first = K.local_scalar(
                    "uint32", init=K.if_then_else(first_pv != _i32(0), _u32(0), _u32(1))
                )
                for site in range(8):
                    K.ptx[_MMA_F16](
                        taddr + _u32(_TMEM_O[instance]),
                        _pack2(a_lo + _u32(_PV_A_OFFSETS[site]), _u32(_DESC_HI)),
                        _pack2(b_lo + _u32(_PV_B_OFFSETS[site]), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(enable_first if site == 0 else _u32(1)),
                        pred=leader,
                    )

            with K.While(work_m < total_work):
                first_pv = [K.local_scalar("int32", init=_i32(1)) for _ in range(2)]
                inst0_stage = K.local_scalar("int32", init=_i32(0))
                _mbar_wait(bar(_MBAR_Q_FULL), phase_q)
                _flip(phase_q)

                _mbar_wait(bar(_MBAR_KV_FULL), _i32(0))
                qk_chain(_i32(0), 0)
                commit_one(bar(_MBAR_S_FULL))
                commit_one(bar(_MBAR_KV_EMPTY))

                pair_m = K.local_scalar("int32", init=_i32(0))
                with K.While(pair_m < _i32(7)):
                    s0 = K.local_scalar("int32", init=inst0_stage)
                    s1 = K.local_scalar("int32", init=(inst0_stage + _i32(1)) % _i32(4))
                    s0_next = K.local_scalar("int32", init=(inst0_stage + _i32(2)) % _i32(4))

                    _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s1, "uint32") * _u32(8), _i32(0))
                    qk_chain(s1, 1)
                    commit_one(bar(_MBAR_S_FULL + 8))
                    commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s1, "uint32") * _u32(8))

                    _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s0, "uint32") * _u32(8), _i32(1))
                    _mbar_wait(bar(_MBAR_P_FULL), phase_p[0])
                    _flip(phase_p[0])
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    pv_chain(s0, 0, first_pv[0])
                    K.assign(first_pv[0], _i32(0))
                    commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s0, "uint32") * _u32(8))

                    _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s0_next, "uint32") * _u32(8), _i32(0))
                    qk_chain(s0_next, 0)
                    commit_one(bar(_MBAR_S_FULL))
                    commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s0_next, "uint32") * _u32(8))

                    _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s1, "uint32") * _u32(8), _i32(1))
                    _mbar_wait(bar(_MBAR_P_FULL + 8), phase_p[1])
                    _flip(phase_p[1])
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    pv_chain(s1, 1, first_pv[1])
                    K.assign(first_pv[1], _i32(0))
                    commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s1, "uint32") * _u32(8))
                    K.assign(inst0_stage, s0_next)
                    K.assign(pair_m, pair_m + _i32(1))

                s0_last = K.local_scalar("int32", init=inst0_stage)
                s1_last = K.local_scalar("int32", init=(inst0_stage + _i32(1)) % _i32(4))
                _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s1_last, "uint32") * _u32(8), _i32(0))
                qk_chain(s1_last, 1)
                final_qk_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                K.ptx[_TCGEN05_COMMIT](bar(_MBAR_S_FULL + 8), pred=final_qk_leader)
                K.ptx[_TCGEN05_COMMIT](bar(_MBAR_Q_EMPTY), pred=final_qk_leader)
                commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s1_last, "uint32") * _u32(8))

                _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s0_last, "uint32") * _u32(8), _i32(1))
                _mbar_wait(bar(_MBAR_P_FULL), phase_p[0])
                _flip(phase_p[0])
                K.ptx["tcgen05.fence::after_thread_sync"]()
                pv_chain(s0_last, 0, first_pv[0])
                commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s0_last, "uint32") * _u32(8))

                _mbar_wait(bar(_MBAR_KV_FULL) + K.cast(s1_last, "uint32") * _u32(8), _i32(1))
                _mbar_wait(bar(_MBAR_P_FULL + 8), phase_p[1])
                _flip(phase_p[1])
                K.ptx["tcgen05.fence::after_thread_sync"]()
                pv_chain(s1_last, 1, first_pv[1])
                commit_one(bar(_MBAR_KV_EMPTY) + K.cast(s1_last, "uint32") * _u32(8))
                commit_one(bar(_MBAR_O_FULL))
                _mbar_wait(bar(_MBAR_DECODE_DONE), phase_decode)
                _flip(phase_decode)
                K.assign(work_m, work_m + work_stride)
        with r_producer:
            phase_q_empty = K.local_scalar("int32", init=_i32(1))
            work_p = K.local_scalar("int32", init=first_work)

            def page_info(request, kv_head, selected_slot):
                selected_block = K.local_scalar("int32")
                K.ptx.ld.global_.s32(
                    selected_block,
                    q2k_indices.ptr_to([(kv_head * num_requests + request) * TOPK + selected_slot]),
                )
                kv_len = K.local_scalar("int32")
                K.ptx.ld.global_.s32(kv_len, kv_lens.ptr_to([request]))
                valid = K.local_scalar("int32", init=_i32(0))
                physical_page = K.local_scalar("int32", init=_i32(0))
                with K.If(selected_block >= _i32(0)), K.Then():
                    block_start = selected_block * PAGE_SIZE
                    K.assign(valid, kv_len - block_start)
                    with K.If(valid > PAGE_SIZE), K.Then():
                        K.assign(valid, PAGE_SIZE)
                    with K.If(valid < _i32(0)), K.Then():
                        K.assign(valid, _i32(0))
                    causal_cols = kv_len - _i32(1) - block_start + _i32(1)
                    with K.If(valid > causal_cols), K.Then():
                        K.assign(valid, causal_cols)
                    with K.If(valid < _i32(0)), K.Then():
                        K.assign(valid, _i32(0))
                    K.ptx.ld.global_.s32(
                        physical_page, page_table.ptr_to([request * msa_max_pages + selected_block])
                    )
                    with K.If(physical_page < _i32(0)), K.Then():
                        K.assign(valid, _i32(0))
                        K.assign(physical_page, _i32(0))
                page_head = K.local_scalar("int32", init=physical_page * num_kv_heads + kv_head)
                return valid, page_head

            def issue_fp8_page(tmap, stage, page_head):
                completion = bar(_MBAR_KV_SRC_FULL) + K.cast(stage, "uint32") * _u32(8)
                _mbar_expect_tx(completion, _SMEM_KV_FP8_STAGE_BYTES)
                dst = bar(_SMEM_KV_FP8) + K.cast(stage, "uint32") * _u32(_SMEM_KV_FP8_STAGE_BYTES)
                K.ptx[_TMA_G2S_3D](dst, K.address_of(tmap), _i32(0), _i32(0), page_head, completion)
                K.ptx[_TMA_G2S_3D](
                    dst + _u32(8192), K.address_of(tmap), _i32(0), _i32(64), page_head, completion
                )

            with K.While(work_p < total_work):
                request_p = work_p // num_kv_heads
                kv_head_p = work_p % num_kv_heads
                group_size_p = num_q_heads // num_kv_heads
                _mbar_wait(bar(_MBAR_Q_EMPTY), phase_q_empty)
                _flip(phase_q_empty)

                q_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                with K.If(q_leader != _u32(0)), K.Then():
                    q_row = request_p * num_q_heads + kv_head_p * group_size_p
                    _mbar_expect_tx(bar(_MBAR_Q_FULL), 4096)
                    K.ptx[_TMA_G2S_3D](
                        bar(_SMEM_QT),
                        K.address_of(q_map),
                        _i32(0),
                        q_row,
                        _i32(0),
                        bar(_MBAR_Q_FULL),
                    )

                kv_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                with K.If(kv_leader != _u32(0)), K.Then():
                    for ni in range(4):
                        valid, page_head = page_info(request_p, kv_head_p, _i32(15 - ni))
                        K.ptx.st.shared.b32(bar(_SMEM_PAGE_INDICES + 4 * ni), valid)
                        K.ptx.fence.proxy.async_.shared__cta()
                        _mbar_wait(bar(_MBAR_KV_EMPTY + 8 * ni), _i32(1))
                        issue_fp8_page(k_map, _i32(ni), page_head)

                    ni_p = K.local_scalar("int32", init=_i32(0))
                    with K.While(ni_p < TOPK):
                        stage = K.local_scalar("int32", init=ni_p % _i32(4))
                        _, page_head_v = page_info(request_p, kv_head_p, _i32(15) - ni_p)
                        _mbar_wait(bar(_MBAR_KV_EMPTY) + K.cast(stage, "uint32") * _u32(8), _i32(0))
                        issue_fp8_page(v_map, stage, page_head_v)
                        next_ni = K.local_scalar("int32", init=ni_p + _i32(4))
                        with K.If(next_ni < TOPK), K.Then():
                            next_valid, next_page_head = page_info(
                                request_p, kv_head_p, _i32(15) - next_ni
                            )
                            K.ptx.st.shared.b32(
                                bar(_SMEM_PAGE_INDICES) + K.cast(next_ni, "uint32") * _u32(4),
                                next_valid,
                            )
                            K.ptx.fence.proxy.async_.shared__cta()
                            _mbar_wait(
                                bar(_MBAR_KV_EMPTY) + K.cast(stage, "uint32") * _u32(8), _i32(1)
                            )
                            issue_fp8_page(k_map, stage, next_page_head)
                        K.assign(ni_p, ni_p + _i32(1))
                K.assign(work_p, work_p + work_stride)
        with r_aux:
            aux_tid = tid - _i32(14 * 32)

            def convert_fp8_stage(stage):
                src_base = bar(_SMEM_KV_FP8) + K.cast(stage, "uint32") * _u32(
                    _SMEM_KV_FP8_STAGE_BYTES
                )
                dst_base = bar(_SMEM_KV) + K.cast(stage, "uint32") * _u32(_SMEM_KV_STAGE_BYTES)
                off_base = K.local_scalar("int32", init=aux_tid)
                with K.While(off_base < _i32(2048)):
                    for unroll in range(4):
                        off = off_base + _i32(64 * unroll)
                        with K.If(off < _i32(2048)), K.Then():
                            src64 = K.local_scalar("uint64")
                            K.ptx["ld.shared.b64"](
                                src64, src_base + K.cast(off, "uint32") * _u32(8)
                            )
                            packed = K.alloc_local((4,), "uint32")
                            for cv in range(4):
                                e4m3x2 = K.local_scalar(
                                    "uint16",
                                    init=K.cast(
                                        (src64 >> _u32(16 * cv)) & K.uint64(0xFFFF), "uint16"
                                    ),
                                )
                                K.ptx["cvt.rn.bf16x2.e4m3x2"](packed[cv], e4m3x2)
                            elt = off * _i32(8)
                            row = ((elt % _i32(128)) // _i32(64)) * _i32(128) + elt // _i32(128)
                            byte_off = row * _i32(128) + ((elt % _i32(64)) * _i32(16)) // _i32(8)
                            swizzled = byte_off ^ ((row % _i32(8)) * _i32(16))
                            K.ptx["st.shared.v4.b32"](
                                dst_base + K.cast(swizzled, "uint32"),
                                packed[0],
                                packed[1],
                                packed[2],
                                packed[3],
                            )
                    K.assign(off_base, off_base + _i32(256))
                K.ptx.fence.proxy.async_.shared__cta()

            def publish_converted(stage):
                K.ptx.barrier.sync(10, 64)
                with K.If(warp == 14), K.Then():
                    leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If(leader != _u32(0)), K.Then():
                        _mbar_arrive(bar(_MBAR_KV_FULL) + K.cast(stage, "uint32") * _u32(8))

            for stage in range(4):
                _mbar_wait(bar(_MBAR_KV_SRC_FULL + 8 * stage), _i32(0))
                K.ptx.fence.proxy.async_.shared__cta()
                convert_fp8_stage(_i32(stage))
                publish_converted(_i32(stage))

            ni_a = K.local_scalar("int32", init=_i32(0))
            with K.While(ni_a < TOPK):
                stage_a = K.local_scalar("int32", init=ni_a % _i32(4))
                _mbar_wait(bar(_MBAR_KV_SRC_FULL) + K.cast(stage_a, "uint32") * _u32(8), _i32(1))
                K.ptx.fence.proxy.async_.shared__cta()
                convert_fp8_stage(stage_a)
                publish_converted(stage_a)
                next_ni_a = K.local_scalar("int32", init=ni_a + _i32(4))
                with K.If(next_ni_a < TOPK), K.Then():
                    _mbar_wait(
                        bar(_MBAR_KV_SRC_FULL) + K.cast(stage_a, "uint32") * _u32(8), _i32(0)
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    convert_fp8_stage(stage_a)
                    publish_converted(stage_a)
                K.assign(ni_a, ni_a + _i32(1))

        K.cuda.cta_sync()
        with K.If(warp == 0), K.Then():
            K.ptx[_TMEM_DEALLOC](taddr, _u32(_TMEM_COLS_ALLOC))

    return blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103


def get_kernel(**config: Any):
    del config
    return _kernel().func


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


@lru_cache(maxsize=1)
def _compiled_kernel():
    from tirx_kernels.runner import compile_kernel

    # CUDA 13.3 emits PTX ISA 9.3 for both this NVCC build and the pinned source.
    return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")


_GUARD_ELEMS = 64
_OUT_GUARD = 42.5
_LSE_GUARD = -54321.25
_SOURCE_ROOT = Path("/root-vol/aarch64-ws/kernel-libs/gb300/flashinfer")


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _guarded_tensor(shape, dtype, *, fill, guard, device):
    import torch

    elements = math.prod(shape)
    storage = torch.empty(elements + 2 * _GUARD_ELEMS, dtype=dtype, device=device)
    storage[:_GUARD_ELEMS].fill_(guard)
    storage[-_GUARD_ELEMS:].fill_(guard)
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + elements].view(shape)
    view.fill_(fill)
    return view, storage


def _fill_random_fp8(tensor, generator, amplitude=1.0 / 3.0):
    import torch

    # Materialize one BF16 staging tensor at a time to keep fixture peak memory bounded.
    values = torch.randn(
        tensor.shape, device=tensor.device, dtype=torch.bfloat16, generator=generator
    )
    values.mul_(amplitude)
    tensor.copy_(values.to(torch.float8_e4m3fn))


def _make_metadata(pattern: str, seed: int, device):
    import torch

    generator = torch.Generator().manual_seed(seed)
    logical = torch.arange(MAX_PAGES, dtype=torch.int32).view(1, MAX_PAGES)
    page_table = (
        logical + torch.arange(NUM_REQUESTS, dtype=torch.int32).view(-1, 1) * MAX_PAGES
    ).contiguous()
    kv_lens = torch.full((NUM_REQUESTS,), MAX_PAGES * PAGE_SIZE, dtype=torch.int32)
    q2k = torch.empty((NUM_KV_HEADS, NUM_REQUESTS, TOPK), dtype=torch.int32)
    for head in range(NUM_KV_HEADS):
        for request in range(NUM_REQUESTS):
            q2k[head, request] = torch.randperm(MAX_PAGES, generator=generator)[:TOPK]

    if pattern == "min_valid":
        kv_lens.fill_(1)
        q2k.fill_(-1)
        q2k[:, :, 0] = 0
    elif pattern == "ragged":
        lengths = torch.tensor(
            [1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2047, 2048, 2049, 4095, 4096],
            dtype=torch.int32,
        )
        kv_lens.copy_(lengths.repeat(NUM_REQUESTS // lengths.numel()))
        for request in range(NUM_REQUESTS):
            last = max(0, (int(kv_lens[request]) - 1) // PAGE_SIZE)
            candidates = [(last - offset) % MAX_PAGES for offset in range(TOPK)]
            q2k[:, request] = torch.tensor(candidates, dtype=torch.int32)
    elif pattern == "scattered":
        page_table.copy_(
            torch.randperm(NUM_REQUESTS * MAX_PAGES, generator=generator, dtype=torch.int64)
            .to(torch.int32)
            .view(NUM_REQUESTS, MAX_PAGES)
        )
    elif pattern == "invalid_pages":
        q2k[:, ::4, :].fill_(-1)
        q2k[:, 1::4, 1::3] = q2k[:, 1::4, :1]
        for request in range(2, NUM_REQUESTS, 4):
            selected = q2k[:, request, :4].reshape(-1).unique()
            page_table[request, selected.to(torch.int64)] = -1
        kv_lens[3::4] = 129
    elif pattern == "numeric":
        base = torch.arange(TOPK, dtype=torch.int32)
        q2k.copy_(base.view(1, 1, TOPK).expand_as(q2k))
        kv_lens[::3] = 2048
        kv_lens[1::3] = 2047
    elif pattern != "upstream":
        raise ValueError(f"unknown fixture pattern {pattern!r}")

    return page_table.to(device), kv_lens.to(device), q2k.to(device)


def prepare_data(**config: Any) -> dict[str, Any]:
    """Create the frozen serving shape with guards around every input/output buffer."""
    import torch

    cfg = _without_label(config)
    pattern = str(cfg["pattern"])
    seed = int(cfg["seed"])
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    num_pages = NUM_REQUESTS * MAX_PAGES

    q, q_storage = _guarded_tensor(
        (NUM_REQUESTS, NUM_Q_HEADS, HEAD_DIM), torch.bfloat16, fill=0.0, guard=3.25, device=device
    )
    q.copy_(
        torch.randn(q.shape, device=device, dtype=torch.bfloat16, generator=generator).mul_(
            1.0 / 3.0
        )
    )
    k, k_storage = _guarded_tensor(
        (num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM),
        torch.float8_e4m3fn,
        fill=0.0,
        guard=1.0,
        device=device,
    )
    v, v_storage = _guarded_tensor(
        (num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM),
        torch.float8_e4m3fn,
        fill=0.0,
        guard=-1.0,
        device=device,
    )
    _fill_random_fp8(k, generator)
    _fill_random_fp8(v, generator)
    if pattern == "numeric":
        q_flat = q.view(-1)
        q_flat[0::4] = 0.0
        q_flat[1::4] = 0.5
        q_flat[2::4] = -0.5
        q_flat[3::4] = 1.0
        k_flat, v_flat = k.view(-1), v.view(-1)
        k_flat[0::4] = 0.0
        k_flat[1::4] = 0.5
        k_flat[2::4] = -0.5
        k_flat[3::4] = 1.0
        v_flat[0::4] = 1.0
        v_flat[1::4] = -1.0
        v_flat[2::4] = 0.25
        v_flat[3::4] = -0.25

    page_table_cpu, kv_lens_cpu, q2k_cpu = _make_metadata(pattern, seed, "cpu")

    def guarded_copy(source, guard):
        view, storage = _guarded_tensor(
            tuple(source.shape), source.dtype, fill=0, guard=guard, device=device
        )
        view.copy_(source.to(device))
        return view, storage

    page_table, page_table_storage = guarded_copy(page_table_cpu, -777777)
    kv_lens, kv_lens_storage = guarded_copy(kv_lens_cpu, -666666)
    q2k_indices, q2k_storage = guarded_copy(q2k_cpu, -555555)
    cu_k, cu_k_storage = guarded_copy(kv_lens_cpu, -444444)
    q_offsets, q_offsets_storage = guarded_copy(kv_lens_cpu, -333333)

    def outputs():
        out, out_storage = _guarded_tensor(
            (NUM_REQUESTS, NUM_Q_HEADS, HEAD_DIM),
            torch.bfloat16,
            fill=float("nan"),
            guard=_OUT_GUARD,
            device=device,
        )
        lse, lse_storage = _guarded_tensor(
            (NUM_REQUESTS, NUM_Q_HEADS),
            torch.float32,
            fill=float("nan"),
            guard=_LSE_GUARD,
            device=device,
        )
        return {"out": out, "lse": lse, "out_storage": out_storage, "lse_storage": lse_storage}

    input_guards = {
        "q": (q_storage, 3.25),
        "k": (k_storage, 1.0),
        "v": (v_storage, -1.0),
        "page_table": (page_table_storage, -777777),
        "kv_lens": (kv_lens_storage, -666666),
        "q2k_indices": (q2k_storage, -555555),
        "cu_k": (cu_k_storage, -444444),
        "q_offsets": (q_offsets_storage, -333333),
    }
    return {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "page_table": page_table,
        "cu_k": cu_k,
        "q2k_indices": q2k_indices,
        "q_offsets": q_offsets,
        "kv_lens": kv_lens,
        "num_requests": NUM_REQUESTS,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "softmax_scale_log2": (HEAD_DIM**-0.5) / math.log(2.0),
        "msa_max_pages": MAX_PAGES,
        "input_guards": input_guards,
        "tirx": outputs(),
        "source": outputs(),
    }


def _tirx_args(data: dict[str, Any], slot: str = "tirx"):
    buffers = data[slot]
    return (
        data["q"].view(-1),
        data["k"].view(__import__("torch").uint8).view(-1),
        data["v"].view(__import__("torch").uint8).view(-1),
        buffers["out"].view(-1),
        buffers["lse"].view(-1),
        data["page_table"].view(-1),
        data["cu_k"].view(-1),
        data["q2k_indices"].view(-1),
        data["q_offsets"].view(-1),
        data["kv_lens"].view(-1),
        data["num_requests"],
        data["num_q_heads"],
        data["num_kv_heads"],
        data["softmax_scale_log2"],
        data["msa_max_pages"],
    )


def _tirx_launch(executable, data: dict[str, Any]):
    arguments = _tirx_args(data)

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


def _verify_source_checkout() -> None:
    import flashinfer.msa_ops._blackwell_sm100 as source_api

    api_path = Path(source_api.__file__).resolve()
    if _SOURCE_ROOT not in api_path.parents:
        raise RuntimeError(
            f"reference imported from {api_path}, not pinned checkout {_SOURCE_ROOT}"
        )


@lru_cache(maxsize=1)
def _source_module():
    from flashinfer.msa_ops import _blackwell_sm100

    _verify_source_checkout()
    return _blackwell_sm100._get_module("decode_q1_bf16_query_fp8_kv_xform2_paged", "sm103a")


def _source_launch(data: dict[str, Any]):
    import torch
    import tvm_ffi

    buffers = data["source"]
    arguments = (
        data["q"],
        data["k"].view(torch.uint8),
        data["v"].view(torch.uint8),
        buffers["out"],
        buffers["lse"],
        data["page_table"],
        data["cu_k"],
        data["q2k_indices"],
        data["q_offsets"],
        data["kv_lens"],
        data["num_requests"],
        data["num_q_heads"],
        data["num_kv_heads"],
        data["softmax_scale_log2"],
        data["msa_max_pages"],
        data["num_requests"],
        data["num_kv_heads"],
        1,
        int(torch.cuda.current_stream().cuda_stream),
    )

    def launch():
        with tvm_ffi.use_torch_stream():
            _source_module().run(*arguments)

    launch._keep_alive = arguments
    return launch


def _assert_bitwise(name: str, ours, source) -> None:
    import torch

    view_dtype = torch.int16 if ours.dtype.itemsize == 2 else torch.int32
    ours_bits = ours.contiguous().view(view_dtype)
    source_bits = source.contiguous().view(view_dtype)
    mismatch = ours_bits != source_bits
    count = int(mismatch.sum())
    if count:
        index = int(mismatch.reshape(-1).nonzero()[0])
        abs_error = float((ours.float() - source.float()).abs().nan_to_num().max())
        raise AssertionError(
            f"{name}: {count}/{mismatch.numel()} bit mismatches; first flat index {index}, "
            f"tirx={ours.reshape(-1)[index].item()!r}, "
            f"source={source.reshape(-1)[index].item()!r}, max_abs={abs_error:.9g}"
        )


def _assert_guards(data: dict[str, Any]) -> None:
    import torch

    for name, (storage, guard) in data["input_guards"].items():
        if not bool(torch.all(storage[:_GUARD_ELEMS] == guard)):
            raise AssertionError(f"{name} prefix guard overwritten")
        if not bool(torch.all(storage[-_GUARD_ELEMS:] == guard)):
            raise AssertionError(f"{name} suffix guard overwritten")
    for slot in ("tirx", "source"):
        for name, guard in (("out", _OUT_GUARD), ("lse", _LSE_GUARD)):
            storage = data[slot][f"{name}_storage"]
            if not bool(torch.all(storage[:_GUARD_ELEMS] == guard)):
                raise AssertionError(f"{slot} {name} prefix guard overwritten")
            if not bool(torch.all(storage[-_GUARD_ELEMS:] == guard)):
                raise AssertionError(f"{slot} {name} suffix guard overwritten")


def _validate_outputs(data: dict[str, Any]) -> dict[str, float]:
    import torch

    _assert_guards(data)
    for slot in ("tirx", "source"):
        out = data[slot]["out"].float()
        lse = data[slot]["lse"]
        if not bool(torch.isfinite(out).all()):
            raise AssertionError(f"{slot} output contains NaN/Inf or poison")
        if bool(torch.isnan(lse).any()) or bool(torch.isposinf(lse).any()):
            raise AssertionError(f"{slot} LSE contains NaN/+Inf or poison")
    _assert_bitwise("O", data["tirx"]["out"], data["source"]["out"])
    _assert_bitwise("LSE", data["tirx"]["lse"], data["source"]["lse"])
    out_abs = float((data["tirx"]["out"].float() - data["source"]["out"].float()).abs().max())
    finite = torch.isfinite(data["source"]["lse"])
    lse_abs = (
        float((data["tirx"]["lse"][finite] - data["source"]["lse"][finite]).abs().max())
        if bool(finite.any())
        else 0.0
    )
    return {"out_max_abs_err": out_abs, "lse_max_abs_err": lse_abs}


def _skip_unless_supported() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required")
    if torch.cuda.get_device_capability() != (10, 3):
        raise SkipTest("this kernel requires compute capability 10.3")


def _snapshot_inputs(data: dict[str, Any]):
    return {
        name: data[name].clone()
        for name in ("q", "k", "v", "page_table", "cu_k", "q2k_indices", "q_offsets", "kv_lens")
    }


def _assert_inputs_unchanged(data: dict[str, Any], snapshots: dict[str, Any]) -> None:
    import torch

    for name, before in snapshots.items():
        if not torch.equal(data[name], before):
            raise AssertionError(f"input {name} was modified")


def run_test(**config: Any):
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    snapshots = _snapshot_inputs(data)
    tirx_launch = _tirx_launch(_compiled_kernel(), data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    stats = _validate_outputs(data)
    _assert_inputs_unchanged(data, snapshots)

    first_out = data["tirx"]["out"].clone()
    first_lse = data["tirx"]["lse"].clone()
    tirx_launch()
    torch.cuda.synchronize()
    _assert_bitwise("O repeat", data["tirx"]["out"], first_out)
    _assert_bitwise("LSE repeat", data["tirx"]["lse"], first_lse)
    _assert_guards(data)
    _assert_inputs_unchanged(data, snapshots)
    return stats


def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    return prepared_gpu_benchmark(
        run_gpu, {"config": _without_label(config), "executable": _compiled_kernel()}
    )


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executable"], data),
            "source_launch": None,
            "with_source": with_source,
            "validated": False,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
        _assert_guards(data)
        if with_source:
            with defer_gpu_interrupts():
                source_launch = _source_launch(data)
                gpu_state["source_launch"] = source_launch
                source_launch()
                torch.cuda.synchronize()
            _validate_outputs(data)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"flashinfer": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config: Any):
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
    "run_gpu",
    "run_test",
]
