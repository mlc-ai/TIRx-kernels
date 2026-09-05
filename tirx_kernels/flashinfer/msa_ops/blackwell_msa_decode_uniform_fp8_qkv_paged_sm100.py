# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ 9f5051736e9fd5cab41c06118a7d4b5c1de23a6d), Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell MSA paged causal decode with uniform FP8 Q/K/V on SM100a.

Upstream sources (FlashInfer @ 9f5051736e9fd5cab41c06118a7d4b5c1de23a6d):

- ``csrc/blackwell_msa/sm100a/blackwell_msa_decode_uniform_fp8_qkv_paged.cu``
- ``csrc/blackwell_msa/sm100a/blackwell_msa_decode_uniform_fp8_qkv_paged_binding.cu``
- ``flashinfer/msa_ops/_blackwell_sm100.py``
"""

import math
from functools import lru_cache
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "blackwell_msa_decode_uniform_fp8_qkv_paged_sm100",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "9f5051736e9fd5cab41c06118a7d4b5c1de23a6d",
            },
            "import": "flashinfer",
        },
    ),
}

HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
THREADS = 384
NUM_WARPS = 12
NUM_SMS_GB200 = 148
EVEN_WAVE_GRID = 128
SMEM_TOTAL = 156672
TMEM_COLS = 512
CUDA_ARCH = "sm_100a"

# TIRx's nvcc path defaults ptxas `--register-usage-level` to 10.  A targeted
# 0..6 sweep found level 2 best for this source-exact warp-specialized kernel:
# it keeps 151 registers with no stack or local traffic while improving the
# q5/q16 schedule by roughly 1.4%/1.7% over level 10 on GB200.
_PTXAS_REG_LEVEL = "2"


def _config(
    label: str,
    *,
    batch_size: int,
    seqlen_q: int,
    kv_lens: int | tuple[int, ...],
    num_q_heads: int,
    num_kv_heads: int,
    pattern: str,
    seed: int,
    softmax_scale: float | None = None,
    k_global_scale: float = 1.0,
    v_global_scale: float = 1.0,
) -> dict[str, Any]:
    return {
        "label": label,
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "kv_lens": kv_lens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "pattern": pattern,
        "seed": seed,
        "softmax_scale": softmax_scale,
        "k_global_scale": k_global_scale,
        "v_global_scale": v_global_scale,
    }


CONFIGS = [
    _config(
        "b128_q1_kv4096_h64_hkv4",
        batch_size=128,
        seqlen_q=1,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=101,
    ),
    _config(
        "b128_q4_kv4096_h64_hkv4",
        batch_size=128,
        seqlen_q=4,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=104,
    ),
    _config(
        "b128_q16_kv4096_h64_hkv4",
        batch_size=128,
        seqlen_q=16,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=116,
    ),
    _config(
        "b16_q4_kv4096_h64_hkv4_even128",
        batch_size=16,
        seqlen_q=4,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=204,
    ),
    _config(
        "b16_q3_kv4096_h64_hkv4_threshold148",
        batch_size=16,
        seqlen_q=3,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=203,
    ),
    _config(
        "b2_q32_kv129_257_h4_hkv4_gqa1_partial",
        batch_size=2,
        seqlen_q=32,
        kv_lens=(129, 257),
        num_q_heads=4,
        num_kv_heads=4,
        pattern="partial_permuted",
        seed=232,
    ),
    _config(
        "b8_q1_kv384_1920_h8_hkv2_gqa4_sparse",
        batch_size=8,
        seqlen_q=1,
        kv_lens=(384, 512, 640, 768, 1024, 1280, 1664, 1920),
        num_q_heads=8,
        num_kv_heads=2,
        pattern="sparse_variable",
        seed=301,
    ),
    _config(
        "b4_q7_kv4096_h64_hkv4_gqa16_scale_stress",
        batch_size=4,
        seqlen_q=7,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="scale_stress",
        seed=407,
        softmax_scale=0.0625,
        k_global_scale=0.75,
        v_global_scale=1.25,
    ),
]

BENCH_CONFIGS = [
    _config(
        f"b128_q{seqlen_q}_kv4096_h64_hkv4",
        batch_size=128,
        seqlen_q=seqlen_q,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=1000 + seqlen_q,
    )
    for seqlen_q in range(1, 17)
]
BENCH_CONFIGS.append(
    _config(
        "b16_q4_kv4096_h64_hkv4_even128",
        batch_size=16,
        seqlen_q=4,
        kv_lens=4096,
        num_q_heads=64,
        num_kv_heads=4,
        pattern="canonical",
        seed=2004,
    )
)


# cuTensorMapEncodeTiled enum values used by the upstream binding.
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    """Encode the source's three by-value rank-3 UINT8 tensor maps."""
    total_q = params["total_q"]
    seqlen_q = params["seqlen_q"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    max_pages = params["max_pages"]
    batch_size = total_q // seqlen_q
    physical_page_heads = batch_size * max_pages * num_kv_heads

    def encode(tensor, dims, strides, box):
        descriptor = K.stack_alloca("tensormap", 1)
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            "uint8",
            3,
            tensor.data,
            *dims,
            *strides,
            *box,
            1,
            1,
            1,
            _TMA_INTERLEAVE_NONE,
            _TMA_SWIZZLE_128B,
            _TMA_L2_PROMOTION_NONE,
            _TMA_OOB_FILL_NONE,
        )
        return descriptor

    q_map = encode(
        params["q"], (HEAD_DIM, total_q * num_q_heads, 1), (HEAD_DIM, HEAD_DIM), (HEAD_DIM, 16, 1)
    )
    kv_dims = (HEAD_DIM, PAGE_SIZE, physical_page_heads)
    kv_strides = (HEAD_DIM, HEAD_DIM * PAGE_SIZE)
    kv_box = (HEAD_DIM, PAGE_SIZE, 1)
    k_map = encode(params["k"], kv_dims, kv_strides, kv_box)
    v_map = encode(params["v"], kv_dims, kv_strides, kv_box)
    return (q_map, k_map, v_map)


def _grid(params):
    work = params["total_q"] * params["num_kv_heads"]
    default = K.min(work, K.int32(NUM_SMS_GB200))
    even = K.And(
        params["seqlen_q"] >= K.int32(4),
        K.And(
            work % K.int32(EVEN_WAVE_GRID) == K.int32(0),
            (work + K.int32(EVEN_WAVE_GRID - 1)) // K.int32(EVEN_WAVE_GRID)
            == (work + default - K.int32(1)) // default,
        ),
    )
    return [K.if_then_else(even, K.int32(EVEN_WAVE_GRID), default)]


# Exact dynamic-SMEM byte map from the frozen source.
_MBAR_Q_FULL = 0
_MBAR_Q_EMPTY = 8
_MBAR_KV_FULL = 16
_MBAR_KV_EMPTY = 80
_MBAR_S_FULL = 112
_MBAR_P_FULL = 128
_MBAR_CORR_SIG = 144
_MBAR_P_STORE_TURN = 160
_MBAR_O_FULL = 176
_MBAR_DECODE_DONE = 184
_SMEM_TMEM_MAILBOX = 192
_SMEM_Q = 1024
_SMEM_KV = 17408
_SMEM_KV_STAGE_BYTES = 16384
_SMEM_META = 148480
_SMEM_META_PAGE_HEAD = _SMEM_META + 64
_SMEM_ACC_SCALE = 152576
_SMEM_ROW_SUM = 153600
_SMEM_ROW_MAX = 154624

_MBARRIER_INIT = (
    (_MBAR_Q_FULL, 1),
    (_MBAR_Q_EMPTY, 1),
    *((_MBAR_KV_FULL + 8 * i, 1) for i in range(8)),
    *((_MBAR_KV_EMPTY + 8 * i, 1) for i in range(4)),
    (_MBAR_S_FULL, 1),
    (_MBAR_S_FULL + 8, 1),
    (_MBAR_P_FULL, 64),
    (_MBAR_P_FULL + 8, 64),
    (_MBAR_CORR_SIG, 32),
    (_MBAR_CORR_SIG + 8, 32),
    (_MBAR_P_STORE_TURN, 32),
    (_MBAR_P_STORE_TURN + 8, 32),
    (_MBAR_O_FULL, 1),
    (_MBAR_DECODE_DONE, 32),
)

_TMEM_SCORE = (0, 128)
_TMEM_OUTPUT = (256, 384)
_TMEM_P_OFFSET = 64
_DESC_HI = 0x40004040
_QK_IDESC = 136314896
_PV_IDESC = 136380432
_V_LBO_BIT = 0x04000000
_FULL_MASK = 0xFFFFFFFF
_NEG_INF = float("-inf")
_LN2 = math.log(2.0)

_TMA_G2S_3D = "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F8 = "tcgen05.mma.cta_group::1.kind::f8f6f4"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_X16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"


def _u32(value):
    return K.uint32(value)


def _i32(value):
    return K.int32(value)


def _udiv_work_i32(value, divisor):
    """Divide a non-negative uint32 work index by a positive runtime divisor."""
    return K.cast(value // K.cast(divisor, "uint32"), "int32")


def _umod_work_i32(value, divisor):
    """Remainder for a non-negative uint32 work index and positive divisor."""
    return K.cast(value % K.cast(divisor, "uint32"), "int32")


def _f32(value):
    return K.float32(value)


def _bar(smem, offset):
    return smem + _u32(offset)


def _mbar_wait(addr, phase):
    K.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, tx_bytes):
    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(tx_bytes))


def _flip(phase):
    K.assign(phase, phase ^ _i32(1))


def _advance_ring(stage, phase):
    K.assign(stage, stage + _u32(1))
    with K.If(stage == _u32(4)), K.Then():
        K.assign(stage, _u32(0))
        _flip(phase)


def _pack2(lo, hi):
    packed = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, lo, hi)
    return packed


def _packed_fma(values, base, multiplier, addend):
    packed = K.local_scalar("uint64")
    K.ptx.fma.rn.ftz.f32x2(
        packed,
        _pack2(values[base], values[base + 1]),
        _pack2(multiplier, multiplier),
        _pack2(addend, addend),
    )
    K.ptx.mov.b64(values[base], values[base + 1], packed)


def _packed_mul(values, base, multiplier):
    packed = K.local_scalar("uint64")
    K.ptx.mul.rn.ftz.f32x2(
        packed, _pack2(values[base], values[base + 1]), _pack2(multiplier, multiplier)
    )
    K.ptx.mov.b64(values[base], values[base + 1], packed)


def _max_f32(a, b):
    out = K.local_scalar("float32")
    K.ptx.max.f32(out, a, b)
    return out


def _row_max(values):
    acc0 = K.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = K.local_scalar("float32", init=_f32(_NEG_INF))
    for chunk in range(4):
        for pair in range(16):
            pair_max = _max_f32(values[chunk * 32 + pair * 2], values[chunk * 32 + pair * 2 + 1])
            if pair % 2 == 0:
                K.assign(acc0, _max_f32(acc0, pair_max))
            else:
                K.assign(acc1, _max_f32(acc1, pair_max))
    return _max_f32(acc0, acc1)


def _block_sum(values):
    packed = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, _f32(0.0), _f32(0.0))
    for pair in range(64):
        K.ptx.add.f32x2(packed, packed, _pack2(values[2 * pair], values[2 * pair + 1]))
    lo = K.local_scalar("float32")
    hi = K.local_scalar("float32")
    K.ptx.mov.b64(lo, hi, packed)
    result = K.local_scalar("float32")
    K.ptx.add.ftz.f32(result, lo, hi)
    return result


def _tmem_load_x32(values, base, addr):
    K.ptx[_TMEM_LD_X32](*(values[base + i] for i in range(32)), addr)


def _tmem_load_x16(values, addr):
    K.ptx[_TMEM_LD_X16](*(values[i] for i in range(16)), addr)


def _tmem_store_x16(addr, values, base=0):
    K.ptx[_TMEM_ST_X16](addr, *(values[base + i] for i in range(16)))


def _pack_fp8x4(dst, f0, f1, f2, f3):
    lo = K.local_scalar("uint16")
    hi = K.local_scalar("uint16")
    K.ptx.cvt.rn.satfinite.e4m3x2.f32(lo, f1, f0)
    K.ptx.cvt.rn.satfinite.e4m3x2.f32(hi, f3, f2)
    K.ptx.mov.b32(dst, lo, hi)


def _commit(addr):
    leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
    K.ptx[_TCGEN05_COMMIT](addr, pred=leader)


def _commit2(addr0, addr1):
    leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
    K.ptx[_TCGEN05_COMMIT](addr0, pred=leader)
    K.ptx[_TCGEN05_COMMIT](addr1, pred=leader)


def _build_kernel():
    @K.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        grid=_grid,
        host_prelude=_host_prelude,
        # The pinned CUDA source legally uses setmaxnreg with one-operand
        # __launch_bounds__(384).  K's generic low-level check models
        # setmaxnreg direction only for a pinned occupancy, so opt out here
        # rather than inventing the .minnctapersm attribute rejected by the
        # source/sketch review.  The body still uses only K PTX statements.
        check_ir=False,
    )
    def blackwell_msa_decode_uniform_fp8_qkv_paged_sm100(
        q: K.gptr[K.u8],
        k: K.gptr[K.u8],
        v: K.gptr[K.u8],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        page_table: K.gptr[K.i32],
        kv_indptr: K.gptr[K.i32],
        q2k_indices: K.gptr[K.i32],
        q_offsets: K.gptr[K.i32],
        kv_lens: K.gptr[K.i32],
        total_q: K.i32,
        seqlen_q: K.i32,
        num_q_heads: K.i32,
        num_kv_heads: K.i32,
        softmax_scale_log2: K.f32,
        output_scale: K.f32,
        max_pages: K.i32,
        *,
        host,
    ):
        q_map, k_map, v_map = host
        del q, k, v, kv_indptr, q_offsets
        # >>> kernel_blackwell_batch_attention_msa_decode_uniform_fp8_natural_sm100_v1
        warp = K.warp_id()
        lane = K.lane_id()
        arena = K.alloc_buffer((SMEM_TOTAL,), K.u8, scope="shared.dyn", align=1024)
        smem = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        with K.If(warp == _i32(0)), K.Then():
            init_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(init_leader != _u32(0)), K.Then():
                for offset, count in _MBARRIER_INIT:
                    K.ptx.mbarrier.init.shared__cta.b64(_bar(smem, offset), _u32(count))
                K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.warp.sync(_u32(_FULL_MASK))

        with K.If(warp == _i32(0)), K.Then():
            K.ptx[_TMEM_ALLOC](_bar(smem, _SMEM_TMEM_MAILBOX), _u32(TMEM_COLS))
            K.ptx[_TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        K.ptx["tcgen05.fence::after_thread_sync"]()
        taddr = K.local_scalar("uint32")
        K.ptx.ld.volatile.shared.b32(taddr, _bar(smem, _SMEM_TMEM_MAILBOX))

        with K.If(K.And(warp >= _i32(8), warp <= _i32(11))), K.Then():
            K.ptx.setmaxnreg.dec.sync.aligned.u32(_u32(96))
        K.cuda.cta_sync()
        with K.If(warp <= _i32(7)), K.Then():
            K.ptx.setmaxnreg.inc.sync.aligned.u32(_u32(232))
        K.cuda.cta_sync()

        total_work = K.cast(total_q * num_kv_heads, "uint32")
        grid_x = K.cast(
            _grid({"total_q": total_q, "seqlen_q": seqlen_q, "num_kv_heads": num_kv_heads})[0],
            "uint32",
        )

        # ---- score / softmax roles: warps 0 and 4 -------------------------
        with K.If(K.Or(warp == _i32(0), warp == _i32(4))), K.Then():
            stage = K.local_scalar("int32", init=warp // _i32(4))
            s_phase = K.local_scalar("int32", init=_i32(0))
            p_store_phase = K.local_scalar(
                "int32", init=K.if_then_else(stage == _i32(0), _i32(1), _i32(0))
            )
            work = K.local_scalar("uint32", init=K.cast(K.cta_id(), "uint32"))
            with K.While(work < total_work):
                state_row = stage * _i32(128) + lane
                row_max = K.local_scalar("float32", init=_f32(_NEG_INF))
                row_sum = K.local_scalar("float32", init=_f32(0.0))
                scores = K.alloc_local((128,), "float32")
                probabilities = K.alloc_local((32,), "uint32")
                block_sum = K.local_scalar("float32")
                block_max = K.local_scalar("float32")
                new_max = K.local_scalar("float32")
                safe_max = K.local_scalar("float32")
                max_scaled = K.local_scalar("float32")
                delta = K.local_scalar("float32")
                acc_scale = K.local_scalar("float32")

                with K.serial(8, unroll=False) as pair:
                    s_bar = _bar(smem, _MBAR_S_FULL) + K.cast(stage, "uint32") * _u32(8)
                    _mbar_wait(s_bar, s_phase)
                    _flip(s_phase)
                    K.ptx["tcgen05.fence::after_thread_sync"]()

                    tile = pair * _i32(2) + stage
                    valid = K.local_scalar("int32")
                    K.ptx.ld.shared.b32(
                        valid, _bar(smem, _SMEM_META) + K.cast(tile, "uint32") * _u32(4)
                    )
                    score_addr = taddr + K.cast(stage, "uint32") * _u32(128)
                    for chunk in range(4):
                        _tmem_load_x32(scores, chunk * 32, score_addr + _u32(chunk * 32))
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()

                    with K.If(valid < _i32(128)), K.Then():
                        for quarter in range(4):
                            limit = K.local_scalar("int32", init=valid - _i32(quarter * 32))
                            with K.If(limit < _i32(0)), K.Then():
                                K.assign(limit, _i32(0))
                            with K.If(limit > _i32(32)), K.Then():
                                K.assign(limit, _i32(32))
                            bits = K.local_scalar("uint32", init=_u32(0))
                            with K.If(limit >= _i32(32)):
                                with K.Then():
                                    K.assign(bits, _u32(_FULL_MASK))
                                with K.Else():
                                    with K.If(limit > _i32(0)), K.Then():
                                        K.assign(
                                            bits,
                                            K.shift_left(_u32(1), K.cast(limit, "uint32"))
                                            - _u32(1),
                                        )
                            for element in range(32):
                                with K.If((bits & _u32(1 << element)) == _u32(0)), K.Then():
                                    K.assign(scores[quarter * 32 + element], _f32(_NEG_INF))

                    K.assign(block_max, _row_max(scores))
                    K.ptx.max.f32(new_max, row_max, block_max)
                    K.assign(
                        safe_max, K.if_then_else(new_max == _f32(_NEG_INF), _f32(0.0), new_max)
                    )
                    K.ptx.mul.ftz.f32(max_scaled, safe_max, softmax_scale_log2)
                    K.ptx.fma.rn.ftz.f32(delta, row_max, softmax_scale_log2, -max_scaled)
                    exp_delta = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(exp_delta, delta)
                    K.assign(
                        acc_scale, K.if_then_else(row_max > _f32(_NEG_INF), exp_delta, _f32(1.0))
                    )
                    K.assign(row_max, new_max)
                    K.ptx.st.shared.b32(
                        _bar(smem, _SMEM_ACC_SCALE) + K.cast(state_row, "uint32") * _u32(4),
                        acc_scale,
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _mbar_arrive(_bar(smem, _MBAR_CORR_SIG) + K.cast(stage, "uint32") * _u32(8))

                    for pair_value in range(64):
                        _packed_fma(scores, pair_value * 2, softmax_scale_log2, -max_scaled)
                    for element in range(128):
                        K.ptx.ex2.approx.ftz.f32(scores[element], scores[element])
                    K.assign(block_sum, _block_sum(scores))

                    with K.If(pair == _i32(0)), K.Then():
                        _mbar_wait(
                            _bar(smem, _MBAR_P_STORE_TURN) + K.cast(stage, "uint32") * _u32(8),
                            p_store_phase,
                        )
                    for group4 in range(32):
                        _pack_fp8x4(
                            probabilities[group4],
                            scores[group4 * 4],
                            scores[group4 * 4 + 1],
                            scores[group4 * 4 + 2],
                            scores[group4 * 4 + 3],
                        )
                    p_addr = score_addr + _u32(_TMEM_P_OFFSET)
                    _tmem_store_x16(p_addr, probabilities, 0)
                    _tmem_store_x16(p_addr + _u32(16), probabilities, 16)
                    K.ptx.fma.rn.ftz.f32(row_sum, row_sum, acc_scale, block_sum)
                    K.ptx["tcgen05.wait::st.sync.aligned"]()
                    K.ptx["tcgen05.fence::before_thread_sync"]()
                    with K.If(pair == _i32(0)), K.Then():
                        other_stage = _i32(1) - stage
                        _mbar_arrive(
                            _bar(smem, _MBAR_P_STORE_TURN) + K.cast(other_stage, "uint32") * _u32(8)
                        )
                        _flip(p_store_phase)
                    _mbar_arrive(_bar(smem, _MBAR_P_FULL) + K.cast(stage, "uint32") * _u32(8))

                K.ptx.st.shared.b32(
                    _bar(smem, _SMEM_ROW_SUM) + K.cast(state_row, "uint32") * _u32(4), row_sum
                )
                K.ptx.st.shared.b32(
                    _bar(smem, _SMEM_ROW_MAX) + K.cast(state_row, "uint32") * _u32(4), row_max
                )
                K.ptx.fence.proxy.async_.shared__cta()
                _mbar_arrive(_bar(smem, _MBAR_CORR_SIG) + K.cast(stage, "uint32") * _u32(8))
                K.assign(work, work + grid_x)

        # ---- elected Q / K / V producer: warp 2 ---------------------------
        with K.If(warp == _i32(2)), K.Then():
            q_empty_phase = K.local_scalar("int32", init=_i32(1))
            work = K.local_scalar("uint32", init=K.cast(K.cta_id(), "uint32"))
            with K.While(work < total_work):
                query = K.local_scalar("int32", init=_udiv_work_i32(work, num_kv_heads))
                kv_head = K.local_scalar("int32", init=_umod_work_i32(work, num_kv_heads))
                group = K.local_scalar("int32", init=num_q_heads // num_kv_heads)
                _mbar_wait(_bar(smem, _MBAR_Q_EMPTY), q_empty_phase)
                _flip(q_empty_phase)

                q_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                with K.If(q_leader != _u32(0)), K.Then():
                    q_row = query * num_q_heads + kv_head * group
                    _mbar_expect_tx(_bar(smem, _MBAR_Q_FULL), 2048)
                    K.ptx[_TMA_G2S_3D](
                        _bar(smem, _SMEM_Q),
                        K.address_of(q_map),
                        _i32(0),
                        q_row,
                        _i32(0),
                        _bar(smem, _MBAR_Q_FULL),
                    )

                producer_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                with K.If(producer_leader != _u32(0)), K.Then():
                    ring_stage = K.local_scalar("uint32", init=_u32(0))
                    ring_phase = K.local_scalar("int32", init=_i32(1))
                    batch = query // seqlen_q
                    query_in_batch = query - batch * seqlen_q

                    def resolve_tile(tile):
                        selected_position = _i32(15) - tile
                        selected_index = (kv_head * total_q + query) * _i32(
                            TOPK
                        ) + selected_position
                        logical_page = K.local_scalar("int32")
                        kv_len = K.local_scalar("int32")
                        K.ptx.ld.global_.nc.b32(logical_page, q2k_indices.ptr_to([selected_index]))
                        K.ptx.ld.global_.nc.b32(kv_len, kv_lens.ptr_to([batch]))
                        valid = K.local_scalar("int32", init=_i32(0))
                        physical_page = K.local_scalar("int32", init=_i32(0))
                        with K.If(logical_page >= _i32(0)), K.Then():
                            block_start = logical_page * _i32(PAGE_SIZE)
                            K.assign(
                                valid, K.max(_i32(0), K.min(_i32(PAGE_SIZE), kv_len - block_start))
                            )
                            query_position = kv_len - seqlen_q + query_in_batch
                            K.assign(
                                valid,
                                K.max(
                                    _i32(0), K.min(valid, query_position - block_start + _i32(1))
                                ),
                            )
                            K.ptx.ld.global_.nc.b32(
                                physical_page, page_table.ptr_to([batch * max_pages + logical_page])
                            )
                            with K.If(physical_page < _i32(0)), K.Then():
                                K.assign(valid, _i32(0))
                                K.assign(physical_page, _i32(0))
                        page_head = physical_page * num_kv_heads + kv_head
                        K.ptx.st.shared.b32(
                            _bar(smem, _SMEM_META) + K.cast(tile, "uint32") * _u32(4), valid
                        )
                        K.ptx.st.shared.b32(
                            _bar(smem, _SMEM_META_PAGE_HEAD) + K.cast(tile, "uint32") * _u32(4),
                            page_head,
                        )
                        K.ptx.fence.proxy.async_.shared__cta()
                        return page_head

                    def push_tile(tensor_map, page_head):
                        _mbar_wait(_bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8), ring_phase)
                        full_bar = _bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8)
                        _mbar_expect_tx(full_bar, 16384)
                        K.ptx[_TMA_G2S_3D](
                            _bar(smem, _SMEM_KV) + ring_stage * _u32(_SMEM_KV_STAGE_BYTES),
                            K.address_of(tensor_map),
                            _i32(0),
                            _i32(0),
                            page_head,
                            full_bar,
                        )
                        _advance_ring(ring_stage, ring_phase)

                    for tile in range(2):
                        page_head = resolve_tile(_i32(tile))
                        push_tile(k_map, page_head)

                    with K.serial(2, 16, unroll=False) as next_tile:
                        v_tile = next_tile - _i32(2)
                        v_page_head = K.local_scalar("int32")
                        K.ptx.ld.shared.b32(
                            v_page_head,
                            _bar(smem, _SMEM_META_PAGE_HEAD) + K.cast(v_tile, "uint32") * _u32(4),
                        )
                        push_tile(v_map, v_page_head)
                        page_head = resolve_tile(next_tile)
                        push_tile(k_map, page_head)

                    for v_tile in range(14, 16):
                        page_head = K.local_scalar("int32")
                        K.ptx.ld.shared.b32(
                            page_head, _bar(smem, _SMEM_META_PAGE_HEAD + 4 * v_tile)
                        )
                        push_tile(v_map, page_head)
                K.assign(work, work + grid_x)

        # ---- sole tcgen05 issuer: warp 3 ----------------------------------
        with K.If(warp == _i32(3)), K.Then():
            q_full_phase = K.local_scalar("int32", init=_i32(0))
            p0_phase = K.local_scalar("int32", init=_i32(0))
            p1_phase = K.local_scalar("int32", init=_i32(0))
            decode_phase = K.local_scalar("int32", init=_i32(0))
            work = K.local_scalar("uint32", init=K.cast(K.cta_id(), "uint32"))

            def qk(stage, ring_stage):
                q_lo = K.local_scalar(
                    "uint32", init=K.uniform((_bar(smem, _SMEM_Q) >> _u32(4)) & _u32(0x3FFF))
                )
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        ((_bar(smem, _SMEM_KV) >> _u32(4)) & _u32(0x3FFF)) + ring_stage * _u32(1024)
                    ),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                for step in range(4):
                    K.ptx[_MMA_F8](
                        taddr + K.cast(stage, "uint32") * _u32(128),
                        _pack2(q_lo + _u32(2 * step), _u32(_DESC_HI)),
                        _pack2(b_lo + _u32(2 * step), _u32(_DESC_HI)),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(_u32(0 if step == 0 else 1)),
                        pred=leader,
                    )

            def pv(stage, ring_stage, first):
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        (((_bar(smem, _SMEM_KV) >> _u32(4)) & _u32(0x3FFF)) | _u32(_V_LBO_BIT))
                        + ring_stage * _u32(1024)
                    ),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                enable_first = K.local_scalar(
                    "uint32", init=K.if_then_else(first != _i32(0), _u32(0), _u32(1))
                )
                for step in range(4):
                    K.ptx[_MMA_F8](
                        taddr + _u32(256) + K.cast(stage, "uint32") * _u32(128),
                        taddr
                        + K.cast(stage, "uint32") * _u32(128)
                        + _u32(_TMEM_P_OFFSET + 8 * step),
                        _pack2(b_lo + _u32(256 * step), _u32(_DESC_HI)),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(enable_first if step == 0 else _u32(1)),
                        pred=leader,
                    )

            with K.While(work < total_work):
                ring_stage = K.local_scalar("uint32", init=_u32(0))
                ring_phase = K.local_scalar("int32", init=_i32(0))
                first_pv0 = K.local_scalar("int32", init=_i32(1))
                first_pv1 = K.local_scalar("int32", init=_i32(1))
                _mbar_wait(_bar(smem, _MBAR_Q_FULL), q_full_phase)
                _flip(q_full_phase)

                for stage in range(2):
                    _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                    qk(_i32(stage), ring_stage)
                    _commit2(
                        _bar(smem, _MBAR_S_FULL + 8 * stage),
                        _bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8),
                    )
                    _advance_ring(ring_stage, ring_phase)

                with K.serial(7, unroll=False) as pair:
                    _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                    _mbar_wait(_bar(smem, _MBAR_P_FULL), p0_phase)
                    _flip(p0_phase)
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    pv(_i32(0), ring_stage, first_pv0)
                    _commit(_bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8))
                    _advance_ring(ring_stage, ring_phase)

                    _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                    qk(_i32(0), ring_stage)
                    _commit2(
                        _bar(smem, _MBAR_S_FULL), _bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8)
                    )
                    _advance_ring(ring_stage, ring_phase)

                    _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                    _mbar_wait(_bar(smem, _MBAR_P_FULL + 8), p1_phase)
                    _flip(p1_phase)
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    pv(_i32(1), ring_stage, first_pv1)
                    _commit(_bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8))
                    _advance_ring(ring_stage, ring_phase)

                    _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                    qk(_i32(1), ring_stage)
                    with K.If(pair == _i32(6)):
                        with K.Then():
                            _commit2(_bar(smem, _MBAR_S_FULL + 8), _bar(smem, _MBAR_Q_EMPTY))
                        with K.Else():
                            _commit(_bar(smem, _MBAR_S_FULL + 8))
                    _commit(_bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8))
                    _advance_ring(ring_stage, ring_phase)
                    K.assign(first_pv0, _i32(0))
                    K.assign(first_pv1, _i32(0))

                _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                _mbar_wait(_bar(smem, _MBAR_P_FULL), p0_phase)
                _flip(p0_phase)
                K.ptx["tcgen05.fence::after_thread_sync"]()
                pv(_i32(0), ring_stage, first_pv0)
                _commit(_bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8))
                _advance_ring(ring_stage, ring_phase)

                _mbar_wait(_bar(smem, _MBAR_KV_FULL) + ring_stage * _u32(8), ring_phase)
                _mbar_wait(_bar(smem, _MBAR_P_FULL + 8), p1_phase)
                _flip(p1_phase)
                K.ptx["tcgen05.fence::after_thread_sync"]()
                pv(_i32(1), ring_stage, first_pv1)
                _commit(_bar(smem, _MBAR_KV_EMPTY) + ring_stage * _u32(8))
                _advance_ring(ring_stage, ring_phase)
                _commit(_bar(smem, _MBAR_O_FULL))
                _mbar_wait(_bar(smem, _MBAR_DECODE_DONE), decode_phase)
                _flip(decode_phase)
                K.assign(work, work + grid_x)

        # ---- online correction and epilogue: warp 8 -----------------------
        with K.If(warp == _i32(8)), K.Then():
            corr0_phase = K.local_scalar("int32", init=_i32(0))
            corr1_phase = K.local_scalar("int32", init=_i32(0))
            o_phase = K.local_scalar("int32", init=_i32(0))
            work = K.local_scalar("uint32", init=K.cast(K.cta_id(), "uint32"))
            with K.While(work < total_work):
                query = K.local_scalar("int32", init=_udiv_work_i32(work, num_kv_heads))
                kv_head = K.local_scalar("int32", init=_umod_work_i32(work, num_kv_heads))
                group = K.local_scalar("int32", init=num_q_heads // num_kv_heads)
                tmem_row_base = (warp - _i32(8)) * _i32(32)
                row = tmem_row_base + lane
                row_bits = K.shift_left(K.cast(tmem_row_base, "uint32"), _u32(16))
                values = K.alloc_local((16,), "float32")

                with K.serial(8, unroll=False) as pair:
                    for stage in range(2):
                        phase = corr0_phase if stage == 0 else corr1_phase
                        _mbar_wait(_bar(smem, _MBAR_CORR_SIG + 8 * stage), phase)
                        _flip(phase)
                        K.ptx["tcgen05.fence::after_thread_sync"]()
                        scale = K.local_scalar("float32")
                        K.ptx.ld.shared.b32(
                            scale,
                            _bar(smem, _SMEM_ACC_SCALE + 128 * 4 * stage)
                            + K.cast(row, "uint32") * _u32(4),
                        )
                        with K.If(pair > _i32(0)), K.Then():
                            for col in range(0, 128, 16):
                                addr = taddr + _u32(_TMEM_OUTPUT[stage] + col) + row_bits
                                _tmem_load_x16(values, addr)
                                K.ptx["tcgen05.wait::ld.sync.aligned"]()
                                for packed_pair in range(8):
                                    _packed_mul(values, packed_pair * 2, scale)
                                _tmem_store_x16(addr, values)
                            K.ptx["tcgen05.wait::st.sync.aligned"]()
                        _mbar_arrive(_bar(smem, _MBAR_P_FULL + 8 * stage))

                _mbar_wait(_bar(smem, _MBAR_O_FULL), o_phase)
                _flip(o_phase)
                K.ptx["tcgen05.fence::after_thread_sync"]()
                _mbar_wait(_bar(smem, _MBAR_CORR_SIG), corr0_phase)
                _flip(corr0_phase)
                _mbar_wait(_bar(smem, _MBAR_CORR_SIG + 8), corr1_phase)
                _flip(corr1_phase)
                K.ptx["tcgen05.fence::after_thread_sync"]()

                sum0 = K.local_scalar("float32")
                sum1 = K.local_scalar("float32")
                max0 = K.local_scalar("float32")
                max1 = K.local_scalar("float32")
                K.ptx.ld.shared.b32(
                    sum0, _bar(smem, _SMEM_ROW_SUM) + K.cast(row, "uint32") * _u32(4)
                )
                K.ptx.ld.shared.b32(
                    sum1, _bar(smem, _SMEM_ROW_SUM + 128 * 4) + K.cast(row, "uint32") * _u32(4)
                )
                K.ptx.ld.shared.b32(
                    max0, _bar(smem, _SMEM_ROW_MAX) + K.cast(row, "uint32") * _u32(4)
                )
                K.ptx.ld.shared.b32(
                    max1, _bar(smem, _SMEM_ROW_MAX + 128 * 4) + K.cast(row, "uint32") * _u32(4)
                )
                final_max = _max_f32(max0, max1)
                d0 = K.local_scalar("float32", init=_f32(0.0))
                d1 = K.local_scalar("float32", init=_f32(0.0))
                with K.If(max0 != _f32(_NEG_INF)), K.Then():
                    K.ptx.mul.ftz.f32(d0, max0 - final_max, softmax_scale_log2)
                with K.If(max1 != _f32(_NEG_INF)), K.Then():
                    K.ptx.mul.ftz.f32(d1, max1 - final_max, softmax_scale_log2)
                merge0 = K.local_scalar("float32")
                merge1 = K.local_scalar("float32")
                K.ptx.ex2.approx.ftz.f32(merge0, d0)
                K.ptx.ex2.approx.ftz.f32(merge1, d1)
                sum1_scaled = K.local_scalar("float32")
                final_sum = K.local_scalar("float32")
                K.ptx.mul.ftz.f32(sum1_scaled, sum1, merge1)
                K.ptx.fma.rn.ftz.f32(final_sum, sum0, merge0, sum1_scaled)
                inv_sum = K.local_scalar("float32", init=_f32(0.0))
                with K.If(final_sum > _f32(0.0)), K.Then():
                    K.ptx.rcp.approx.ftz.f32(inv_sum, final_sum)
                norm_scale = K.local_scalar("float32")
                K.ptx.mul.ftz.f32(norm_scale, inv_sum, output_scale)

                values0 = K.alloc_local((16,), "float32")
                values1 = K.alloc_local((16,), "float32")
                merged = K.alloc_local((16,), "float32")
                packed_bf16 = K.alloc_local((8,), "uint32")
                for col in range(0, 128, 16):
                    _tmem_load_x16(values0, taddr + _u32(_TMEM_OUTPUT[0] + col) + row_bits)
                    _tmem_load_x16(values1, taddr + _u32(_TMEM_OUTPUT[1] + col) + row_bits)
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    for element in range(16):
                        temporary = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(temporary, values1[element], merge1)
                        K.ptx.fma.rn.ftz.f32(merged[element], values0[element], merge0, temporary)
                    with K.If(row < group), K.Then():
                        for packed_pair in range(8):
                            _packed_mul(merged, packed_pair * 2, norm_scale)
                            K.ptx.cvt.rn.bf16x2.f32(
                                packed_bf16[packed_pair],
                                merged[packed_pair * 2 + 1],
                                merged[packed_pair * 2],
                            )
                        output_row = query * num_q_heads + kv_head * group + row
                        output_base = output_row * _i32(HEAD_DIM) + _i32(col)
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([output_base]),
                            packed_bf16[0],
                            packed_bf16[1],
                            packed_bf16[2],
                            packed_bf16[3],
                        )
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([output_base + _i32(8)]),
                            packed_bf16[4],
                            packed_bf16[5],
                            packed_bf16[6],
                            packed_bf16[7],
                        )

                with K.If(row < group), K.Then():
                    log_sum = K.local_scalar("float32")
                    K.ptx.lg2.approx.ftz.f32(log_sum, final_sum)
                    lse_left = K.local_scalar("float32")
                    lse_right = K.local_scalar("float32")
                    lse_value = K.local_scalar("float32", init=_f32(_NEG_INF))
                    K.ptx.mul.ftz.f32(lse_left, final_max, softmax_scale_log2)
                    K.ptx.mul.ftz.f32(lse_right, log_sum, _f32(_LN2))
                    with K.If(final_sum > _f32(0.0)), K.Then():
                        K.ptx.fma.rn.ftz.f32(lse_value, lse_left, _f32(_LN2), lse_right)
                    output_row = query * num_q_heads + kv_head * group + row
                    K.ptx.st.global_.b32(lse.ptr_to([output_row]), lse_value)
                _mbar_arrive(_bar(smem, _MBAR_DECODE_DONE))
                K.assign(work, work + grid_x)

        K.cuda.cta_sync()
        with K.If(warp == _i32(0)), K.Then():
            K.ptx[_TMEM_DEALLOC](taddr, _u32(TMEM_COLS))

    return blackwell_msa_decode_uniform_fp8_qkv_paged_sm100


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_config(config: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    batch_size = int(config["batch_size"])
    seqlen_q = int(config["seqlen_q"])
    num_q_heads = int(config["num_q_heads"])
    num_kv_heads = int(config["num_kv_heads"])
    if batch_size <= 0 or not 1 <= seqlen_q <= 32:
        raise ValueError("requires positive batch_size and 1 <= seqlen_q <= 32")
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not 1 <= num_q_heads // num_kv_heads <= 16:
        raise ValueError("GQA group size must be in [1, 16]")
    raw_lens = config["kv_lens"]
    kv_lens = (
        (int(raw_lens),) * batch_size
        if isinstance(raw_lens, int)
        else tuple(int(value) for value in raw_lens)
    )
    if len(kv_lens) != batch_size or min(kv_lens) <= 0:
        raise ValueError("kv_lens must provide one positive length per batch")
    max_pages = max((length + PAGE_SIZE - 1) // PAGE_SIZE for length in kv_lens)
    return kv_lens, max_pages


def get_kernel(**config: Any):
    if config:
        _validate_config(_without_label(config))
    return _kernel().func


@lru_cache(maxsize=1)
def _compiled_kernel():
    import os

    from tirx_kernels.runner import compile_kernel

    # CUDA 13.2 emits PTX ISA 9.2 for sm_100a, matching the pinned source build.
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REG_LEVEL
    try:
        return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def _make_selected_blocks(*, config: dict[str, Any], kv_lens: tuple[int, ...], max_pages: int):
    import torch

    batch_size = int(config["batch_size"])
    seqlen_q = int(config["seqlen_q"])
    num_kv_heads = int(config["num_kv_heads"])
    total_q = batch_size * seqlen_q
    pattern = str(config["pattern"])
    selected = torch.full((num_kv_heads, total_q, TOPK), -1, dtype=torch.int32)
    for batch, kv_len in enumerate(kv_lens):
        logical_pages = list(range((kv_len + PAGE_SIZE - 1) // PAGE_SIZE))
        if pattern == "sparse_variable":
            logical_pages = logical_pages[::2]
            final_page = (kv_len - 1) // PAGE_SIZE
            if final_page not in logical_pages:
                logical_pages.append(final_page)
        logical_pages = logical_pages[-TOPK:]
        slots = [-1] * (TOPK - len(logical_pages)) + logical_pages
        for q_in_batch in range(seqlen_q):
            query = batch * seqlen_q + q_in_batch
            for kv_head in range(num_kv_heads):
                row = list(slots)
                if pattern in {"partial_permuted", "sparse_variable"}:
                    shift = (query + 3 * kv_head) % TOPK
                    row = row[shift:] + row[:shift]
                selected[kv_head, query] = torch.tensor(row, dtype=torch.int32)
    if pattern == "partial_permuted":
        # Explicit negative selected slots complement the negative physical-page
        # entry below while leaving valid pages in every row.
        selected[:, :, 0] = -1
    return selected.contiguous()


def _guarded_outputs(shape, device):
    import torch

    out_elements = math.prod(shape)
    out_guard = 16
    lse_elements = shape[0] * shape[1]
    lse_guard = 8
    out_storage = torch.full(
        (out_elements + 2 * out_guard,), 123.0, dtype=torch.bfloat16, device=device
    )
    lse_storage = torch.full(
        (lse_elements + 2 * lse_guard,), 12345.25, dtype=torch.float32, device=device
    )
    out = out_storage[out_guard : out_guard + out_elements].view(shape)
    lse = lse_storage[lse_guard : lse_guard + lse_elements].view(shape[0], shape[1])
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    return {
        "out": out,
        "lse": lse,
        "out_storage": out_storage,
        "lse_storage": lse_storage,
        "out_guard": out_guard,
        "lse_guard": lse_guard,
    }


def prepare_data(**config: Any) -> dict[str, Any]:
    """Create deterministic FP8 paged inputs and independent guarded outputs."""
    import torch

    config = _without_label(config)
    kv_lens_tuple, max_pages = _validate_config(config)
    batch_size = int(config["batch_size"])
    seqlen_q = int(config["seqlen_q"])
    num_q_heads = int(config["num_q_heads"])
    num_kv_heads = int(config["num_kv_heads"])
    total_q = batch_size * seqlen_q
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))

    def fp8_random(shape, amplitude=1.0):
        values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        return (values * float(amplitude)).clamp_(-6.0, 6.0).to(torch.float8_e4m3fn)

    q_amp = 2.0 if config["pattern"] == "scale_stress" else 1.0
    q = fp8_random((total_q, num_q_heads, HEAD_DIM), q_amp)
    physical_pages = batch_size * max_pages
    k = fp8_random((physical_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM))
    v = fp8_random((physical_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM))

    page_table = torch.empty((batch_size, max_pages), dtype=torch.int32)
    cpu_generator = torch.Generator().manual_seed(int(config["seed"]) + 991)
    for batch in range(batch_size):
        physical = torch.arange(batch * max_pages, (batch + 1) * max_pages, dtype=torch.int32)
        if config["pattern"] == "partial_permuted":
            physical = physical[torch.randperm(max_pages, generator=cpu_generator)]
        page_table[batch] = physical
    if config["pattern"] == "partial_permuted":
        page_table[:, 0] = -1

    q2k_indices = _make_selected_blocks(
        config=config, kv_lens=kv_lens_tuple, max_pages=max_pages
    ).to(device)
    page_table = page_table.contiguous().to(device)
    kv_lens = torch.tensor(kv_lens_tuple, dtype=torch.int32, device=device)
    softmax_scale = config["softmax_scale"]
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    softmax_scale *= float(config["k_global_scale"])
    softmax_scale_log2 = float(softmax_scale) / math.log(2.0)
    output_scale = float(config["v_global_scale"])

    shape = (total_q, num_q_heads, HEAD_DIM)
    return {
        "config": config,
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "page_table": page_table,
        "q2k_indices": q2k_indices,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "seqlen_q": seqlen_q,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "max_pages": max_pages,
        "softmax_scale": float(softmax_scale),
        "softmax_scale_log2": softmax_scale_log2,
        "output_scale": output_scale,
        "tirx": _guarded_outputs(shape, device),
        "source": _guarded_outputs(shape, device),
    }


def _tirx_launch(executable, data: dict[str, Any]):
    import torch

    arguments = (
        data["q"].view(torch.uint8).view(-1),
        data["k"].view(torch.uint8).view(-1),
        data["v"].view(torch.uint8).view(-1),
        data["tirx"]["out"].view(-1),
        data["tirx"]["lse"].view(-1),
        data["page_table"].view(-1),
        data["kv_lens"].view(-1),
        data["q2k_indices"].view(-1),
        data["kv_lens"].view(-1),
        data["kv_lens"].view(-1),
        data["total_q"],
        data["seqlen_q"],
        data["num_q_heads"],
        data["num_kv_heads"],
        data["softmax_scale_log2"],
        data["output_scale"],
        data["max_pages"],
    )

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


@lru_cache(maxsize=1)
def _source_backend():
    from flashinfer.msa_ops import _blackwell_sm100

    return _blackwell_sm100


def _source_launch(data: dict[str, Any]):
    import torch
    import tvm_ffi

    backend = _source_backend()
    target = backend._select_target(data["q"].device)

    def launch():
        with tvm_ffi.use_torch_stream():
            backend._run_fp8_direct_module(
                schedule="paged_uniform_fp8",
                target=target,
                q=data["q"],
                k=data["k"],
                v=data["v"],
                out=data["source"]["out"],
                lse=data["source"]["lse"],
                q2k_indices=data["q2k_indices"],
                cu_k=data["kv_lens"],
                q_offsets=data["kv_lens"],
                kv_lens=data["kv_lens"],
                page_table=data["page_table"],
                paged=True,
                max_pages=data["max_pages"],
                seqlen_q=data["seqlen_q"],
                softmax_scale_log2=data["softmax_scale_log2"],
                output_scale=data["output_scale"],
                stream_ptr=int(torch.cuda.current_stream(data["q"].device).cuda_stream),
                workspace=None,
                capturing=False,
            )

    launch._keep_alive = data
    return launch


def _assert_bitwise(name: str, actual, expected) -> None:
    import torch

    actual_bits = actual.contiguous().view(torch.uint8)
    expected_bits = expected.contiguous().view(torch.uint8)
    if not torch.equal(actual_bits, expected_bits):
        mismatches = int((actual_bits != expected_bits).sum())
        max_abs = float((actual.float() - expected.float()).abs().max())
        raise AssertionError(
            f"{name} differs bitwise: {mismatches} byte mismatches, max abs {max_abs:.9g}"
        )


def _check_outputs(data: dict[str, Any]) -> dict[str, float]:
    import torch

    for implementation in ("tirx", "source"):
        buffers = data[implementation]
        if not bool(torch.isfinite(buffers["out"].float()).all()):
            raise AssertionError(f"{implementation} output contains non-finite values")
        lse_values = buffers["lse"]
        if bool(torch.isnan(lse_values).any()) or bool(torch.isposinf(lse_values).any()):
            raise AssertionError(f"{implementation} LSE contains NaN or +inf")
        out_guard = buffers["out_guard"]
        lse_guard = buffers["lse_guard"]
        out_storage = buffers["out_storage"]
        lse_storage = buffers["lse_storage"]
        out_edges = torch.cat((out_storage[:out_guard], out_storage[-out_guard:]))
        lse_edges = torch.cat((lse_storage[:lse_guard], lse_storage[-lse_guard:]))
        if not bool((out_edges == torch.tensor(123.0, dtype=torch.bfloat16, device="cuda")).all()):
            raise AssertionError(f"{implementation} output guard was overwritten")
        if not bool((lse_edges == 12345.25).all()):
            raise AssertionError(f"{implementation} LSE guard was overwritten")
    _assert_bitwise("output", data["tirx"]["out"], data["source"]["out"])
    _assert_bitwise("LSE", data["tirx"]["lse"], data["source"]["lse"])
    # Bitwise equality above is strictly stronger than any finite atol/rtol.
    # Report exact zero directly so matching -inf LSE sentinels do not form NaN
    # through the diagnostic subtraction.
    return {"out_max_abs_err": 0.0, "lse_max_abs_err": 0.0}


def _skip_unless_supported() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required")
    if torch.cuda.get_device_capability() != (10, 0):
        raise SkipTest("uniform FP8 paged MSA decode requires compute capability 10.0")


def _run_public_api(data: dict[str, Any]):
    backend = _source_backend()
    return backend.blackwell_msa_sparse_decode_attention(
        data["q"],
        data["k"],
        data["v"],
        data["q2k_indices"],
        page_table=data["page_table"],
        seqused_k=data["kv_lens"],
        seqlen_q=data["seqlen_q"],
        causal=True,
        softmax_scale=data["softmax_scale"] / float(data["config"]["k_global_scale"]),
        return_softmax_lse=True,
        k_global_scale=float(data["config"]["k_global_scale"]),
        v_global_scale=float(data["config"]["v_global_scale"]),
        force_fused=True,
    )


def run_test(**config: Any) -> dict[str, float]:
    """Run one config, requiring source/TIRx bitwise equality and determinism."""
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    tirx_launch = _tirx_launch(_compiled_kernel(), data)
    source_launch = _source_launch(data)

    metadata_before = (
        data["page_table"].clone(),
        data["q2k_indices"].clone(),
        data["kv_lens"].clone(),
    )
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    stats = _check_outputs(data)
    for name, actual, expected in zip(
        ("page_table", "q2k_indices", "kv_lens"),
        (data["page_table"], data["q2k_indices"], data["kv_lens"]),
        metadata_before,
    ):
        _assert_bitwise(f"{name} input", actual, expected)

    first_out = data["tirx"]["out"].clone()
    first_lse = data["tirx"]["lse"].clone()
    tirx_launch()
    torch.cuda.synchronize()
    _assert_bitwise("output repeat", data["tirx"]["out"], first_out)
    _assert_bitwise("LSE repeat", data["tirx"]["lse"], first_lse)

    if data["config"]["pattern"] == "sparse_variable":
        api_out, api_lse = _run_public_api(data)
        torch.cuda.synchronize()
        _assert_bitwise("public API output", data["source"]["out"], api_out)
        _assert_bitwise("public API LSE", data["source"]["lse"], api_lse)
    return stats


def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_config(kernel_config)
    state = {"config": kernel_config, "executable": _compiled_kernel()}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**config)
        tirx_launch = _tirx_launch(prepared["executable"], data)
        source_launch = _source_launch(data) if with_source else None
        gpu_state = {
            "data": data,
            "tirx_launch": tirx_launch,
            "source_launch": source_launch,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    if not gpu_state["validated"]:
        gpu_state["tirx_launch"]()
        if gpu_state["source_launch"] is not None:
            gpu_state["source_launch"]()
        torch.cuda.synchronize()
        if gpu_state["source_launch"] is not None:
            _check_outputs(gpu_state["data"])
        gpu_state["validated"] = True

    references = (
        {"flashinfer": lambda: gpu_state["source_launch"]}
        if gpu_state["source_launch"] is not None
        else None
    )
    return bench(
        {"tirx": gpu_state["tirx_launch"]},
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
