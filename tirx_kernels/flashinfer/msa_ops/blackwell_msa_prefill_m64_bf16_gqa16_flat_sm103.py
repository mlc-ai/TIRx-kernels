# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ cc6e8794), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell MSA M64 BF16 GQA16 flat prefill for SM103a.

Upstream sources (FlashInfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839):

- csrc/blackwell_msa/sm103a/blackwell_msa_prefill_m64_bf16_gqa16_flat.cu
- csrc/blackwell_msa/sm103a/blackwell_msa_prefill_m64_bf16_gqa16_flat_binding.cu
- flashinfer/msa_ops/_blackwell_sm100.py

The source specialization consumes flat BF16 Q/K/V with D=128, GQA ratio 16,
TopK=16, and KV lengths no larger than 8192. It launches sixteen warps, uses
one 134784-byte dynamic shared-memory arena, and runs two four-query attention
instances through a three-stage aliased K/V ring and 512 TMEM columns.
"""

import math
import os
from functools import lru_cache
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103",
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

HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 16
GQA_RATIO = 16
THREADS = 512
NUM_WARPS = THREADS // 32
SMEM_TOTAL = 134784
TMEM_COLS = 512
CUDA_ARCH = "sm_103a"
_PTXAS_REGISTER_USAGE_LEVEL = "5"


def _config(
    label: str,
    *,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    num_q_heads: int,
    num_kv_heads: int,
    causal: bool = True,
    q_offsets: tuple[int, ...] | None = None,
    selection: str = "random_valid",
    return_softmax_lse: bool = False,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    return {
        "label": label,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "causal": causal,
        "q_offsets": q_offsets,
        "selection": selection,
        "return_softmax_lse": return_softmax_lse,
        "return_temperature_lse": return_temperature_lse,
        "lse_temperature_scale": lse_temperature_scale,
        "seed": seed,
    }


CONFIGS = [
    _config(
        "tail_q17_kv1024_h16_hkv1_bothlse",
        q_lens=(17,),
        kv_lens=(1024,),
        num_q_heads=16,
        num_kv_heads=1,
        return_softmax_lse=True,
        return_temperature_lse=True,
        lse_temperature_scale=0.7,
        seed=18,
    ),
    _config(
        "ragged_q33_17_kv1024_768_h16_hkv1_lse",
        q_lens=(33, 17),
        kv_lens=(1024, 768),
        num_q_heads=16,
        num_kv_heads=1,
        return_softmax_lse=True,
        seed=20,
    ),
    _config(
        "tile_edges_q1_7_8_9_15_16_17_kv2048_h16_hkv1",
        q_lens=(1, 7, 8, 9, 15, 16, 17),
        kv_lens=(2048, 2048, 2048, 2048, 2048, 2048, 2048),
        num_q_heads=16,
        num_kv_heads=1,
        return_softmax_lse=True,
        seed=31,
    ),
    _config(
        "noncausal_q100_37_kv2048_700_h32_hkv2",
        q_lens=(100, 37),
        kv_lens=(2048, 700),
        num_q_heads=32,
        num_kv_heads=2,
        causal=False,
        return_softmax_lse=True,
        seed=17,
    ),
    _config(
        "offset_q9_33_kv2048_4096_h16_hkv1",
        q_lens=(9, 33),
        kv_lens=(2048, 4096),
        num_q_heads=16,
        num_kv_heads=1,
        q_offsets=(0, 127),
        return_softmax_lse=True,
        seed=37,
    ),
    _config(
        "masked_q1_kv256_h16_hkv1_bothlse",
        q_lens=(1,),
        kv_lens=(256,),
        num_q_heads=16,
        num_kv_heads=1,
        q_offsets=(0,),
        selection="future_only",
        return_softmax_lse=True,
        return_temperature_lse=True,
        lse_temperature_scale=0.7,
        seed=29,
    ),
    _config(
        "union_capacity_q8_kv8192_h16_hkv1",
        q_lens=(8,),
        kv_lens=(8192,),
        num_q_heads=16,
        num_kv_heads=1,
        selection="disjoint_union",
        return_softmax_lse=True,
        seed=41,
    ),
    _config(
        "maxkv_q129_kv8192_h64_hkv4_lse",
        q_lens=(129,),
        kv_lens=(8192,),
        num_q_heads=64,
        num_kv_heads=4,
        return_softmax_lse=True,
        seed=42,
    ),
    _config(
        "production_b1_q4096_kv4096_h64_hkv4",
        q_lens=(4096,),
        kv_lens=(4096,),
        num_q_heads=64,
        num_kv_heads=4,
        seed=43,
    ),
]


BENCH_CONFIGS = [
    _config(
        "b1_q1024_kv2048_h16_hkv1_causal",
        q_lens=(1024,),
        kv_lens=(2048,),
        num_q_heads=16,
        num_kv_heads=1,
        seed=100,
    ),
    _config(
        "b1_q4096_kv4096_h64_hkv4_causal",
        q_lens=(4096,),
        kv_lens=(4096,),
        num_q_heads=64,
        num_kv_heads=4,
        seed=43,
    ),
    _config(
        "b1_q8192_kv8192_h64_hkv4_causal",
        q_lens=(8192,),
        kv_lens=(8192,),
        num_q_heads=64,
        num_kv_heads=4,
        seed=102,
    ),
    _config(
        "b3_q513_1025_2049_kv1024_4096_8192_h32_hkv2",
        q_lens=(513, 1025, 2049),
        kv_lens=(1024, 4096, 8192),
        num_q_heads=32,
        num_kv_heads=2,
        seed=103,
    ),
    _config(
        "b2_q2048_1024_kv8192_4096_h32_hkv2_noncausal",
        q_lens=(2048, 1024),
        kv_lens=(8192, 4096),
        num_q_heads=32,
        num_kv_heads=2,
        causal=False,
        seed=104,
    ),
    _config(
        "b1_q4096_kv8192_h16_hkv1_bothlse",
        q_lens=(4096,),
        kv_lens=(8192,),
        num_q_heads=16,
        num_kv_heads=1,
        return_softmax_lse=True,
        return_temperature_lse=True,
        lse_temperature_scale=0.7,
        seed=105,
    ),
]


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_config(**config: Any) -> None:
    q_lens = tuple(config["q_lens"])
    kv_lens = tuple(config["kv_lens"])
    if not q_lens or len(q_lens) != len(kv_lens):
        raise ValueError("q_lens and kv_lens must be non-empty and have equal length")
    if any(q <= 0 for q in q_lens) or any(k <= 0 or k > 8192 for k in kv_lens):
        raise ValueError("query lengths must be positive and KV lengths must be in [1, 8192]")
    if config["num_q_heads"] != GQA_RATIO * config["num_kv_heads"]:
        raise ValueError("this specialization requires num_q_heads / num_kv_heads == 16")
    q_offsets = config.get("q_offsets")
    if q_offsets is not None and len(q_offsets) != len(q_lens):
        raise ValueError("q_offsets must have one entry per batch")


# ---------------------------------------------------------------------------
# TIRx kernel
# ---------------------------------------------------------------------------
# cuTensorMapEncodeTiled enum values (CUtensorMapInterleave / Swizzle /
# L2promotion / FloatOOBfill) as used by the upstream host launcher.
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    """Encode the three upstream tensor maps from the runtime shape scalars.

    Mirrors ``EncodeTma_q`` / ``EncodeTma_k`` / ``EncodeTma_v`` in the upstream
    binding: rank-4 bf16 maps, 128-byte swizzle, no L2 promotion, no OOB fill.
    Q is ``{64, Hq, total_q, 2}`` with a ``{64, 16, 4, 2}`` box; K and V are
    ``{64, N, 2, Hkv}`` with a ``{64, 64, 1, 1}`` box (8 KB per transaction).
    """
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    total_q = params["total_q"]
    total_k = params["k"].shape[0]
    elem_bytes = 2

    def encode(tensor, dims, strides_bytes, box):
        descriptor = K.stack_alloca("tensormap", 1)
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            "bfloat16",
            4,
            tensor.data,
            *dims,
            *strides_bytes,
            *box,
            1,
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
        params["q"],
        (64, num_q_heads, total_q, HEAD_DIM // 64),
        (HEAD_DIM * elem_bytes, num_q_heads * HEAD_DIM * elem_bytes, 64 * elem_bytes),
        (64, GQA_RATIO, 4, HEAD_DIM // 64),
    )
    kv_dims = (64, total_k, HEAD_DIM // 64, num_kv_heads)
    kv_strides = (num_kv_heads * HEAD_DIM * elem_bytes, 64 * elem_bytes, HEAD_DIM * elem_bytes)
    kv_box = (64, 64, 1, 1)
    k_map = encode(params["k"], kv_dims, kv_strides, kv_box)
    v_map = encode(params["v"], kv_dims, kv_strides, kv_box)
    return (q_map, k_map, v_map)


# Byte offsets inside the 134784-byte dynamic SMEM arena (source macro table).
_MBAR_Q_FULL = 0
_MBAR_UNION_READY = 8
_MBAR_K_FULL = 16  # 3 stages
_MBAR_K_EMPTY = 40  # 3 stages
_MBAR_S_FULL = 64  # 2 instances
_MBAR_P_FULL = 80  # 2 instances
_MBAR_CORR_SIG = 96  # 2 instances
_MBAR_CORR_DONE = 112  # 2 instances
_MBAR_O_FULL = 128  # 2 instances
_MBAR_TMEM_DEALLOC = 144
_SMEM_TMEM_MAILBOX = 152
_SMEM_Q0 = 1024
_SMEM_Q1 = 17408
_SMEM_K = 33792  # 3 x 32768
_SMEM_V = 33792  # aliases the three-stage K ring
_SMEM_STAGE_BYTES = 32768
_SMEM_SCALE = 132096  # 512 f32: acc_scale / row_sum / row_max / temperature_sum
_SMEM_MASK_LOW = 134144  # 8 u32
_SMEM_MASK_HIGH = 134176  # 8 u32
_SMEM_UNION_COUNT = 134208  # 2 i32
_SMEM_UNION_BLOCKS = 134216  # 2 x 64 i32
MAX_SELECTED_BLOCKS = 64
# (offset, arrival count) in exact source order.
_MBARRIER_INIT = (
    (_MBAR_Q_FULL, 1),
    (_MBAR_UNION_READY, 32),
    (_MBAR_K_FULL, 1),
    (_MBAR_K_FULL + 8, 1),
    (_MBAR_K_FULL + 16, 1),
    (_MBAR_K_EMPTY, 1),
    (_MBAR_K_EMPTY + 8, 1),
    (_MBAR_K_EMPTY + 16, 1),
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
    (_MBAR_TMEM_DEALLOC, 128),
)
# TMEM column bases per softmax instance: S (128 columns; P overwrites the upper 64
# in place) and the O accumulator (128 columns).
_TMEM_SCORES = (0, 256)
_TMEM_OUTPUT = (128, 384)
_TMEM_P_OFFSET = 64
# UMMA descriptor words and instruction descriptors (source immediates).
_DESC_HI = 0x40004040  # SBO 1024 B, base-offset bit, 128 B swizzle
_QK_IDESC = 0x04200490  # bf16 x bf16 -> f32, M=64, N=128, K-major A and B
_PV_IDESC = 0x04210490  # same with MN-major B (V is token-major)
_QK_A_STEPS = (2, 2, 2, 506, 2, 2, 2)  # Q low-word walk over K=128 (bytes/16)
_QK_B_STEPS = (2, 2, 2, 1018, 2, 2, 2)  # K low-word walk (second dim half at +16384 B)
_V_LBO_BIT = 0x4000000  # LBO = 16384 B: dim-half stride of the V tile
_PV_B_STEP = 128  # 2048 B = 16 tokens x 128 B per K16 step
_PV_A_STEP = 8  # 16 bf16 P columns = 8 TMEM 32-bit columns per K16 step
_REG_SOFTMAX = 192
_REG_CORRECTION = 80
_REG_OTHER = 48
_WAIT_TICKS = 0x989680

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
_TMEM_ST_X64 = "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
_LN2_F32 = 0.6931471805599453
_NEG_INF = float("-inf")
_FULL_MASK = 0xFFFFFFFF


def _u32(value):
    return K.uint32(value)


def _i32(value):
    return K.int32(value)


def _f32(value):
    return K.float32(value)


def _mbar_wait(addr, phase):
    """``mbarrier.try_wait.parity.acquire.cta`` retry loop with the source's suspend hint."""
    K.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, tx_bytes):
    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(tx_bytes))


def _ld_shared_i32(addr):
    value = K.local_scalar("int32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _ld_global_i32(buffer, index):
    value = K.local_scalar("int32")
    K.ptx.ld.global_.nc.b32(value, buffer.ptr_to([index]))
    return value


def _ld_shared_f32(addr):
    value = K.local_scalar("float32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _ld_shared_counts(addr):
    """Both instance counts in one ``ld.shared.v2.b32`` (the source's vectorized pair)."""
    count0 = K.local_scalar("int32")
    count1 = K.local_scalar("int32")
    K.ptx.ld.shared.v2.b32(count0, count1, addr)
    return count0, count1


def _flip(phase):
    K.assign(phase, phase ^ _i32(1))


def _advance_ring(stage, phase, stages):
    """Advance a K/V ring cursor and flip parity exactly at its stage count."""
    K.assign(stage, stage + _u32(1))
    with K.If(stage == _u32(stages)), K.Then():
        K.assign(stage, _u32(0))
        _flip(phase)


def _pack2(lo, hi):
    packed = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, lo, hi)
    return packed


def _packed_fma_inplace(values, base, scale_pair, bias_pair):
    """``fma.rn.ftz.f32x2`` on values[base:base+2] with packed scale and bias."""
    result = K.local_scalar("uint64")
    K.ptx.fma.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair, bias_pair)
    K.ptx.mov.b64(values[base], values[base + 1], result)


def _packed_mul_inplace(values, base, scale_pair):
    """``mul.rn.ftz.f32x2`` on values[base:base+2] with a packed scale."""
    result = K.local_scalar("uint64")
    K.ptx.mul.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair)
    K.ptx.mov.b64(values[base], values[base + 1], result)


def _exp2_inplace(values, index):
    K.ptx.ex2.approx.ftz.f32(values[index], values[index])


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


def _tmem_load_x32(dst, base, taddr):
    K.ptx[_TMEM_LD_X32](*(dst[base + i] for i in range(32)), taddr, 64)


def _block_sum(values):
    """``softmax_block_sum`` over 64 values: 32 packed ``add.f32x2`` into one accumulator."""
    acc = K.local_scalar("uint64")
    K.ptx.mov.b64(acc, _f32(0.0), _f32(0.0))
    for j in range(32):
        K.ptx.add.f32x2(acc, acc, _pack2(values[2 * j], values[2 * j + 1]))
    acc_x = K.local_scalar("float32")
    acc_y = K.local_scalar("float32")
    K.ptx.mov.b64(acc_x, acc_y, acc)
    half = K.local_scalar("float32")
    K.ptx.add.f32(half, acc_x, acc_y)
    total = K.local_scalar("float32")
    K.ptx.add.f32(total, half, _shfl_xor_f32(half, 16))
    return total


def _row_max(values):
    """``row_max_x32_accum`` x2 + ``row_max_reduce``: alternating accumulators, 65 ``max.f32``."""
    acc0 = K.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = K.local_scalar("float32", init=_f32(_NEG_INF))
    for half in range(2):
        for j in range(16):
            pair = _max_f32(values[half * 32 + 2 * j], values[half * 32 + 2 * j + 1])
            if j % 2 == 0:
                K.assign(acc0, _max_f32(acc0, pair))
            else:
                K.assign(acc1, _max_f32(acc1, pair))
    return _max_f32(acc0, acc1)


def _build_kernel():
    @K.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: [(p["total_q"] + 7) // 8 + p["batch_size"] - 1, p["num_kv_heads"]],
        host_prelude=_host_prelude,
    )
    def blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103(
        q: K.gptr[K.bf16, 3],
        k: K.gptr[K.bf16, 3],
        v: K.gptr[K.bf16, 3],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        temperature_lse: K.gptr[K.f32],
        q2k_indices: K.gptr[K.i32],
        cu_seqlens_q: K.gptr[K.i32],
        cu_seqlens_k: K.gptr[K.i32],
        q_offsets: K.gptr[K.i32],
        kv_lens: K.gptr[K.i32],
        total_q: K.i32,
        num_q_heads: K.i32,
        num_kv_heads: K.i32,
        topk: K.i32,
        batch_size: K.i32,
        uniform_q_len: K.i32,
        causal: K.i32,
        derive_q_offset: K.i32,
        softmax_scale_log2: K.f32,
        lse_temperature_scale: K.f32,
        return_softmax_lse: K.i32,
        return_temperature_lse: K.i32,
        *,
        host,
    ):
        q_map, k_map, v_map = host
        del q, k, v, kv_lens, num_kv_heads
        # >>> kernel_minimax_sparse_prefill_exact_union_m64_sm100 body starts here
        linear_tile, kv_head = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()

        arena = K.alloc_buffer((SMEM_TOTAL,), K.u8, scope="shared.dyn", align=1024)
        smem = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        # Mbarrier init (10 groups, 19 barriers) by one elected lane of warp 0.
        with K.If(warp == 0), K.Then():
            init_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(init_leader != _u32(0)), K.Then():
                for offset, count in _MBARRIER_INIT:
                    K.ptx.mbarrier.init.shared__cta.b64(bar(offset), _u32(count))
                K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.warp.sync(_u32(_FULL_MASK))

        # TMEM alloc (512 columns) by warp 0; the address lands in the shared mailbox.
        with K.If(warp == 0), K.Then():
            K.ptx[_TMEM_ALLOC](bar(_SMEM_TMEM_MAILBOX), _u32(TMEM_COLS))
            K.ptx[_TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        K.ptx["tcgen05.fence::after_thread_sync"]()
        taddr = K.local_scalar("uint32")
        K.ptx.ld.volatile.shared.b32(taddr, bar(_SMEM_TMEM_MAILBOX))

        def decode_tile():
            batch = K.local_scalar("int32", init=_i32(0))
            q_tile = K.local_scalar("int32", init=_i32(0))
            tile_prefix = K.local_scalar("int32", init=_i32(0))
            tile_active = K.local_scalar("int32", init=_i32(0))
            with K.If(uniform_q_len > _i32(0)):
                with K.Then():
                    uniform_tiles = (uniform_q_len + _i32(7)) // _i32(8)
                    K.assign(batch, linear_tile // uniform_tiles)
                    K.assign(q_tile, linear_tile - batch * uniform_tiles)
                    with K.If(batch < batch_size), K.Then():
                        K.assign(tile_active, _i32(1))
                with K.Else():
                    candidate = K.local_scalar("int32", init=_i32(0))
                    with K.While(candidate < batch_size):
                        candidate_begin = _ld_global_i32(cu_seqlens_q, candidate)
                        candidate_end = _ld_global_i32(cu_seqlens_q, candidate + _i32(1))
                        candidate_tiles = (candidate_end - candidate_begin + _i32(7)) // _i32(8)
                        with (
                            K.If(
                                K.And(
                                    linear_tile >= tile_prefix,
                                    linear_tile < tile_prefix + candidate_tiles,
                                )
                            ),
                            K.Then(),
                        ):
                            K.assign(batch, candidate)
                            K.assign(q_tile, linear_tile - tile_prefix)
                            K.assign(tile_active, _i32(1))
                        K.assign(tile_prefix, tile_prefix + candidate_tiles)
                        K.assign(candidate, candidate + _i32(1))
            q_begin = _ld_global_i32(cu_seqlens_q, batch)
            q_end = _ld_global_i32(cu_seqlens_q, batch + _i32(1))
            q_len = q_end - q_begin
            q_local_base = q_tile * _i32(8)
            q_valid = K.local_scalar("int32", init=q_len - q_local_base)
            with K.If(q_valid > _i32(8)), K.Then():
                K.assign(q_valid, _i32(8))
            with K.If(q_valid < _i32(0)), K.Then():
                K.assign(q_valid, _i32(0))
            with K.If(tile_active == _i32(0)), K.Then():
                K.assign(q_valid, _i32(0))
            query_base = q_begin + q_local_base
            k_start = _ld_global_i32(cu_seqlens_k, batch)
            k_end = _ld_global_i32(cu_seqlens_k, batch + _i32(1))
            kv_len = k_end - k_start
            query_offset = _ld_global_i32(q_offsets, batch)
            with K.If(derive_q_offset != _i32(0)), K.Then():
                K.assign(query_offset, kv_len - q_len)
            num_n_blocks = K.local_scalar("int32", init=(kv_len + _i32(127)) // _i32(128))
            with K.If(causal != _i32(0)), K.Then():
                visible_tokens = query_offset + q_local_base + q_valid
                visible_blocks = (visible_tokens + _i32(127)) // _i32(128)
                with K.If(num_n_blocks > visible_blocks), K.Then():
                    K.assign(num_n_blocks, visible_blocks)
            with K.If(num_n_blocks < _i32(0)), K.Then():
                K.assign(num_n_blocks, _i32(0))
            with K.If(num_n_blocks > _i32(MAX_SELECTED_BLOCKS)), K.Then():
                K.assign(num_n_blocks, _i32(MAX_SELECTED_BLOCKS))
            return (
                batch,
                q_tile,
                tile_active,
                q_begin,
                q_len,
                q_local_base,
                q_valid,
                query_base,
                k_start,
                kv_len,
                query_offset,
                num_n_blocks,
            )

        # Sibling role guards, as in the source; folding them into an if/else-if chain
        # measured 1.2-2.3% slower on the two small shapes with no register pressure to relieve.
        roles = K.specialize(chain_dispatch=False)
        other_regs = roles.register_scope("other", warps=range(12, 16), regs=_REG_OTHER)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=_REG_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=_REG_CORRECTION)
        r_mma = roles.role("mma", warps=[12], register_scope=other_regs)
        r_idle = roles.role("idle", warps=range(13, 15), register_scope=other_regs)
        r_load = roles.role("load", warps=[15], register_scope=other_regs)

        # Register redistribution: warps 12..15 decrease first so the softmax
        # increase can be granted.
        with K.If(K.And(warp >= 12, warp <= 15)), K.Then():
            other_regs.emit()

        # ---- Role: softmax (warps 0..7, two four-warp instances) ----
        with r_softmax:
            (
                batch_s,
                q_tile_s,
                tile_active_s,
                q_begin_s,
                q_len_s,
                q_local_base_s,
                q_valid,
                query_base_s,
                k_start_s,
                kv_len,
                query_offset_s,
                num_n_blocks_s,
            ) = decode_tile()
            del batch_s, q_tile_s, tile_active_s, q_len_s, k_start_s, num_n_blocks_s
            union_ready_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase)
            _flip(union_ready_phase)
            instance = K.uniform(warp // 4)
            instance_row_offset = K.uniform(instance * 64)
            instance_token_offset = K.uniform(instance * 4)
            instance_tmem_offset = K.uniform(instance * 256)
            union_count = _ld_shared_i32(
                bar(_SMEM_UNION_COUNT) + K.cast(instance, "uint32") * _u32(4)
            )
            warp_in_instance = warp % 4
            tmem_row_origin = warp_in_instance * 32
            my_row = warp_in_instance * 16 + lane % 16
            col_half = lane // 16
            query_in_instance = my_row // 16
            query_in_tile = instance_token_offset + query_in_instance
            instance_valid = K.local_scalar("int32", init=q_valid - instance_token_offset)
            with K.If(instance_valid > _i32(4)), K.Then():
                K.assign(instance_valid, _i32(4))
            with K.If(instance_valid < _i32(0)), K.Then():
                K.assign(instance_valid, _i32(0))
            row_valid = K.local_scalar(
                "int32", init=K.cast(query_in_instance < instance_valid, "int32")
            )
            row_mask_low = _ld_shared_i32(
                bar(_SMEM_MASK_LOW) + K.cast(query_in_tile, "uint32") * _u32(4)
            )
            row_mask_high = _ld_shared_i32(
                bar(_SMEM_MASK_HIGH) + K.cast(query_in_tile, "uint32") * _u32(4)
            )
            row_max = K.local_scalar("float32", init=_f32(_NEG_INF))
            row_sum = K.local_scalar("float32", init=_f32(0.0))
            temperature_sum = K.local_scalar("float32", init=_f32(0.0))
            phase_s_full_0 = K.local_scalar("int32", init=_i32(0))
            phase_s_full_1 = K.local_scalar("int32", init=_i32(0))
            phase_corr_done_0 = K.local_scalar("int32", init=_i32(0))
            phase_corr_done_1 = K.local_scalar("int32", init=_i32(0))
            score_addr = K.local_scalar(
                "uint32",
                init=taddr
                + K.cast(instance_tmem_offset, "uint32")
                + K.shift_left(K.cast(tmem_row_origin, "uint32"), _u32(16)),
            )
            p_addr = K.local_scalar("uint32", init=score_addr + _u32(_TMEM_P_OFFSET))
            scale_row_addr = bar(_SMEM_SCALE) + K.cast(
                instance_row_offset + my_row, "uint32"
            ) * _u32(4)
            scores = K.alloc_local((64,), "float32")
            packed_p = K.alloc_local((32,), "uint32")
            block_sum = K.local_scalar("float32", init=_f32(0.0))
            block_temperature_sum = K.local_scalar("float32", init=_f32(0.0))
            acc_scale = K.local_scalar("float32", init=_f32(1.0))
            temperature_acc_scale = K.local_scalar("float32", init=_f32(1.0))
            selected_max = K.local_scalar("float32", init=_f32(_NEG_INF))
            new_max_scaled = K.local_scalar("float32", init=_f32(0.0))
            safe_max = K.local_scalar("float32", init=_f32(0.0))

            def probabilities_from(frag, bias):
                """fma, exp2, block sum, bf16 pack and the two P stores of one branch."""
                scale_pair = _pack2(softmax_scale_log2, softmax_scale_log2)
                bias_pair = _pack2(bias, bias)
                for j in range(32):
                    _packed_fma_inplace(frag, 2 * j, scale_pair, bias_pair)
                for j in range(64):
                    _exp2_inplace(frag, j)
                K.assign(block_sum, _block_sum(frag))
                for j in range(32):
                    K.ptx.cvt.rn.bf16x2.f32(packed_p[j], frag[2 * j + 1], frag[2 * j])
                K.ptx[_TMEM_ST_X16](p_addr, 32, *(packed_p[i] for i in range(16)))
                K.ptx[_TMEM_ST_X16](p_addr + _u32(16), 32, *(packed_p[16 + i] for i in range(16)))

            def mask_invalid_suffix(frag, half_valid):
                with K.If(K.And(valid_cols > _i32(0), half_valid < _i32(64))), K.Then():

                    def prefix_mask(limit):
                        mask = K.local_scalar("uint32", init=_u32(0))
                        with K.If(limit <= _i32(0)):
                            with K.Then():
                                K.assign(mask, _u32(0))
                            with K.Else():
                                with K.If(limit >= _i32(32)):
                                    with K.Then():
                                        K.assign(mask, _u32(_FULL_MASK))
                                    with K.Else():
                                        shifted = K.local_scalar("uint32")
                                        K.ptx.shl.b32(shifted, _u32(1), K.cast(limit, "uint32"))
                                        K.ptx.add.u32(mask, shifted, _u32(_FULL_MASK))
                        return mask

                    mask_lo = prefix_mask(half_valid)
                    mask_hi = prefix_mask(half_valid - _i32(32))
                    for j in range(64):
                        masked_bit = K.local_scalar("uint32")
                        K.ptx.and_.b32(
                            masked_bit, mask_lo if j < 32 else mask_hi, _u32(1 << (j % 32))
                        )
                        keep = K.local_scalar("uint32")
                        K.ptx.setp.ne.b32(keep, masked_bit, _u32(0))
                        selected_bits = K.local_scalar("uint32")
                        K.ptx.selp.b32(
                            selected_bits,
                            K.reinterpret("uint32", frag[j]),
                            _u32(0xFF800000),
                            K.ptx.pred(keep),
                        )
                        K.assign(frag[j], K.reinterpret("float32", selected_bits))

            union_index = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index < union_count):
                n_block = _ld_shared_i32(
                    bar(_SMEM_UNION_BLOCKS)
                    + K.cast(instance * MAX_SELECTED_BLOCKS + union_index, "uint32") * _u32(4)
                )
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_wait(bar(_MBAR_S_FULL), phase_s_full_0)
                        _flip(phase_s_full_0)
                    with K.Else():
                        _mbar_wait(bar(_MBAR_S_FULL + 8), phase_s_full_1)
                        _flip(phase_s_full_1)
                selected = K.local_scalar("int32", init=_i32(0))
                with K.If(row_valid != _i32(0)), K.Then():
                    with K.If(n_block < _i32(32)):
                        with K.Then():
                            K.assign(
                                selected,
                                K.cast(
                                    (
                                        K.cast(row_mask_low, "uint32")
                                        & K.shift_left(_u32(1), K.cast(n_block, "uint32"))
                                    )
                                    != _u32(0),
                                    "int32",
                                ),
                            )
                        with K.Else():
                            K.assign(
                                selected,
                                K.cast(
                                    (
                                        K.cast(row_mask_high, "uint32")
                                        & K.shift_left(
                                            _u32(1), K.cast(n_block - _i32(32), "uint32")
                                        )
                                    )
                                    != _u32(0),
                                    "int32",
                                ),
                            )
                valid_cols = K.local_scalar("int32", init=_i32(0))
                with K.If(selected != _i32(0)), K.Then():
                    K.assign(valid_cols, kv_len - n_block * 128)
                    with K.If(valid_cols > _i32(128)), K.Then():
                        K.assign(valid_cols, _i32(128))
                    with K.If(causal != _i32(0)), K.Then():
                        query_position = query_offset_s + q_local_base_s + query_in_tile
                        causal_cols = query_position - n_block * _i32(128) + _i32(1)
                        with K.If(valid_cols > causal_cols), K.Then():
                            K.assign(valid_cols, causal_cols)
                    with K.If(valid_cols < _i32(0)), K.Then():
                        K.assign(valid_cols, _i32(0))
                _tmem_load_x32(scores, 0, score_addr)
                _tmem_load_x32(scores, 32, score_addr + _u32(32))
                half_valid = K.local_scalar("int32", init=valid_cols - col_half * 64)
                with K.If(half_valid < _i32(0)), K.Then():
                    K.assign(half_valid, _i32(0))
                with K.If(half_valid > _i32(64)), K.Then():
                    K.assign(half_valid, _i32(64))
                mask_invalid_suffix(scores, half_valid)
                tile_max = K.local_scalar("float32", init=_row_max(scores))
                with K.If(half_valid <= _i32(0)), K.Then():
                    K.assign(tile_max, _f32(_NEG_INF))
                K.assign(tile_max, _max_f32(tile_max, _shfl_xor_f32(tile_max, 16)))
                new_max = _max_f32(tile_max, row_max)
                K.assign(safe_max, K.if_then_else(new_max == _f32(_NEG_INF), _f32(0.0), new_max))
                K.ptx.mul.f32(new_max_scaled, safe_max, softmax_scale_log2)
                neg_new_max_scaled = K.local_scalar("float32")
                K.ptx.neg.f32(neg_new_max_scaled, new_max_scaled)
                acc_scale_log2 = K.local_scalar("float32")
                K.ptx.fma.rn.f32(acc_scale_log2, row_max, softmax_scale_log2, neg_new_max_scaled)
                with K.If(acc_scale_log2 >= _f32(-8.0)):
                    with K.Then():
                        # The running max moves by less than 2^8: keep the old max, skip the rescale.
                        K.assign(selected_max, row_max)
                        K.assign(
                            safe_max, K.if_then_else(row_max == _f32(_NEG_INF), _f32(0.0), row_max)
                        )
                        K.assign(acc_scale, _f32(1.0))
                        K.assign(temperature_acc_scale, _f32(1.0))
                        K.ptx.mul.f32(new_max_scaled, safe_max, softmax_scale_log2)
                    with K.Else():
                        K.assign(selected_max, new_max)
                        exp_scale = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_scale, acc_scale_log2)
                        K.assign(
                            acc_scale,
                            K.if_then_else(row_max > _f32(_NEG_INF), exp_scale, _f32(1.0)),
                        )
                        tau_log2 = K.local_scalar("float32")
                        K.ptx.mul.f32(tau_log2, acc_scale_log2, lse_temperature_scale)
                        exp_tau = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_tau, tau_log2)
                        K.assign(
                            temperature_acc_scale,
                            K.if_then_else(row_max > _f32(_NEG_INF), exp_tau, _f32(1.0)),
                        )
                K.assign(row_max, selected_max)
                with K.If(col_half == 0), K.Then():
                    K.ptx.st.shared.b32(scale_row_addr, acc_scale)
                K.ptx.fence.proxy.async_.shared__cta()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_CORR_SIG))
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_CORR_SIG + 8))
                # The bias negates the (possibly re-selected) new_max_scaled after the branch.
                neg_bias = K.local_scalar("float32")
                K.ptx.neg.f32(neg_bias, new_max_scaled)
                score_bias = K.local_scalar("float32")
                K.assign(score_bias, K.if_then_else(valid_cols > _i32(0), neg_bias, _f32(_NEG_INF)))
                K.assign(block_temperature_sum, _f32(0.0))
                K.assign(block_sum, _f32(0.0))
                with K.If(return_temperature_lse != _i32(0)):
                    with K.Then():
                        tau_scale = K.local_scalar("float32")
                        K.ptx.mul.f32(tau_scale, softmax_scale_log2, lse_temperature_scale)
                        tau_bias = K.local_scalar("float32")
                        K.ptx.mul.f32(tau_bias, score_bias, lse_temperature_scale)
                        tau_scale_pair = _pack2(tau_scale, tau_scale)
                        tau_bias_pair = _pack2(tau_bias, tau_bias)
                        for j in range(32):
                            _packed_fma_inplace(scores, 2 * j, tau_scale_pair, tau_bias_pair)
                        for j in range(64):
                            _exp2_inplace(scores, j)
                        K.assign(block_temperature_sum, _block_sum(scores))
                        # S is re-read because the temperature sum consumed the first copy.
                        _tmem_load_x32(scores, 0, score_addr)
                        _tmem_load_x32(scores, 32, score_addr + _u32(32))
                        mask_invalid_suffix(scores, half_valid)
                        probabilities_from(scores, score_bias)
                    with K.Else():
                        probabilities_from(scores, score_bias)
                K.ptx.tcgen05.wait__st.sync.aligned()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_P_FULL))
                        _mbar_wait(bar(_MBAR_CORR_DONE), phase_corr_done_0)
                        _flip(phase_corr_done_0)
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_P_FULL + 8))
                        _mbar_wait(bar(_MBAR_CORR_DONE + 8), phase_corr_done_1)
                        _flip(phase_corr_done_1)
                K.ptx.fma.rn.f32(row_sum, row_sum, acc_scale, block_sum)
                temperature_candidate = K.local_scalar("float32")
                K.ptx.fma.rn.f32(
                    temperature_candidate,
                    temperature_sum,
                    temperature_acc_scale,
                    block_temperature_sum,
                )
                K.assign(
                    temperature_sum,
                    K.if_then_else(
                        return_temperature_lse != _i32(0), temperature_candidate, temperature_sum
                    ),
                )
                K.assign(union_index, union_index + _i32(1))
            with K.If(col_half == 0), K.Then():
                K.ptx.st.shared.b32(scale_row_addr + _u32(128 * 4), row_sum)
                K.ptx.st.shared.b32(scale_row_addr + _u32(256 * 4), row_max)
                K.ptx.st.shared.b32(scale_row_addr + _u32(384 * 4), temperature_sum)
            K.ptx.fence.proxy.async_.shared__cta()
            with K.If(instance == 0):
                with K.Then():
                    _mbar_arrive(bar(_MBAR_CORR_SIG))
                with K.Else():
                    _mbar_arrive(bar(_MBAR_CORR_SIG + 8))

        # ---- Role: correction and epilogue (warps 8..11) ----
        with r_correction:
            (
                batch_c,
                q_tile_c,
                tile_active_c,
                q_begin_c,
                q_len_c,
                q_local_base_c,
                q_valid_c,
                query_base_c,
                k_start_c,
                kv_len_c,
                query_offset_c,
                num_n_blocks_c,
            ) = decode_tile()
            del (
                batch_c,
                q_tile_c,
                tile_active_c,
                q_begin_c,
                q_len_c,
                k_start_c,
                kv_len_c,
                query_offset_c,
                num_n_blocks_c,
            )
            union_ready_phase_c = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_c)
            _flip(union_ready_phase_c)
            union_count_0_c, union_count_1_c = _ld_shared_counts(bar(_SMEM_UNION_COUNT))
            max_union_count_c = K.local_scalar("int32", init=union_count_0_c)
            with K.If(max_union_count_c < union_count_1_c), K.Then():
                K.assign(max_union_count_c, union_count_1_c)
            warp_in_role = warp - 8
            tmem_row_origin_c = warp_in_role * 32
            my_row_c = warp_in_role * 16 + lane % 16
            col_half_c = lane // 16
            row_addr = K.local_scalar(
                "uint32", init=K.shift_left(K.cast(tmem_row_origin_c, "uint32"), _u32(16))
            )
            scale_base_c = bar(_SMEM_SCALE) + K.cast(my_row_c, "uint32") * _u32(4)
            _mbar_arrive(bar(_MBAR_P_FULL))
            _mbar_arrive(bar(_MBAR_P_FULL + 8))
            phase_corr_sig_0 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr_sig_0)
            _flip(phase_corr_sig_0)
            _mbar_arrive(bar(_MBAR_CORR_DONE))
            phase_corr_sig_1 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr_sig_1)
            _flip(phase_corr_sig_1)
            _mbar_arrive(bar(_MBAR_CORR_DONE + 8))
            o_frag = K.alloc_local((64,), "float32")

            def rescale_instance(instance, phase_corr_sig, union_index_1):
                count = union_count_0_c if instance == 0 else union_count_1_c
                with K.If(count > union_index_1), K.Then():
                    _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr_sig)
                    _flip(phase_corr_sig)
                    acc_scale_c = _ld_shared_f32(scale_base_c + _u32(64 * 4 * instance))
                    any_rescale = K.local_scalar("uint32")
                    K.ptx.vote_sync.any.pred(
                        any_rescale,
                        K.ptx.pred(K.cast(acc_scale_c < _f32(1.0), "bool")),
                        _u32(_FULL_MASK),
                    )
                    with K.If(any_rescale != _u32(0)), K.Then():
                        o_addr = taddr + _u32(_TMEM_OUTPUT[instance]) + row_addr
                        _tmem_load_x32(o_frag, 0, o_addr)
                        _tmem_load_x32(o_frag, 32, o_addr + _u32(32))
                        scale_pair = _pack2(acc_scale_c, acc_scale_c)
                        for j in range(32):
                            _packed_mul_inplace(o_frag, 2 * j, scale_pair)
                        K.ptx[_TMEM_ST_X64](o_addr, 64, *(o_frag[i] for i in range(64)))
                        K.ptx.tcgen05.wait__st.sync.aligned()
                    _mbar_arrive(bar(_MBAR_P_FULL + 8 * instance))
                    _mbar_arrive(bar(_MBAR_CORR_DONE + 8 * instance))

            union_index_1 = K.local_scalar("int32", init=_i32(1))
            with K.While(union_index_1 < max_union_count_c):
                rescale_instance(0, phase_corr_sig_0, union_index_1)
                rescale_instance(1, phase_corr_sig_1, union_index_1)
                K.assign(union_index_1, union_index_1 + _i32(1))
            phase_o_full_0 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_O_FULL), phase_o_full_0)
            _flip(phase_o_full_0)
            phase_o_full_1 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_O_FULL + 8), phase_o_full_1)
            _flip(phase_o_full_1)
            _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr_sig_0)
            _flip(phase_corr_sig_0)
            _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr_sig_1)
            _flip(phase_corr_sig_1)
            K.ptx["tcgen05.fence::after_thread_sync"]()
            for instance in range(2):
                stats_addr = scale_base_c + _u32(64 * 4 * instance)
                final_sum = _ld_shared_f32(stats_addr + _u32(128 * 4))
                final_max = _ld_shared_f32(stats_addr + _u32(256 * 4))
                final_temperature_sum = _ld_shared_f32(stats_addr + _u32(384 * 4))
                rcp_sum = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(rcp_sum, final_sum)
                sum_positive = K.local_scalar("int32", init=K.cast(final_sum > _f32(0.0), "int32"))
                inv_sum = K.local_scalar(
                    "float32", init=K.if_then_else(sum_positive != _i32(0), rcp_sum, _f32(0.0))
                )
                instance_valid_c = K.local_scalar("int32", init=q_valid_c - _i32(4 * instance))
                with K.If(instance_valid_c > _i32(4)), K.Then():
                    K.assign(instance_valid_c, _i32(4))
                with K.If(instance_valid_c < _i32(0)), K.Then():
                    K.assign(instance_valid_c, _i32(0))
                query_in_instance_c = my_row_c // 16
                q_head_c = kv_head * GQA_RATIO + my_row_c % 16
                query = query_base_c + 4 * instance + query_in_instance_c
                output_row = (query * num_q_heads + q_head_c) * 128
                o_addr = taddr + _u32(_TMEM_OUTPUT[instance]) + row_addr
                _tmem_load_x32(o_frag, 0, o_addr)
                _tmem_load_x32(o_frag, 32, o_addr + _u32(32))
                col_base = col_half_c * 64
                with K.If(query_in_instance_c < instance_valid_c), K.Then():
                    inv_pair = _pack2(inv_sum, inv_sum)
                    for chunk in range(8):
                        base = 8 * chunk
                        for j in range(4):
                            _packed_mul_inplace(o_frag, base + 2 * j, inv_pair)
                        words = K.alloc_local((4,), "uint32")
                        for j in range(4):
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[j], o_frag[base + 2 * j + 1], o_frag[base + 2 * j]
                            )
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([output_row + col_base + base]),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                    with K.If(col_half_c == 0), K.Then():
                        stat_idx = query * num_q_heads + q_head_c
                        with K.If(return_softmax_lse != _i32(0)), K.Then():
                            log2_sum = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(log2_sum, final_sum)
                            max_scaled = K.local_scalar("float32")
                            K.ptx.mul.f32(max_scaled, final_max, softmax_scale_log2)
                            log_sum = K.local_scalar("float32")
                            K.ptx.mul.f32(log_sum, log2_sum, _f32(_LN2_F32))
                            lse_value = K.local_scalar("float32")
                            K.ptx.fma.rn.f32(lse_value, max_scaled, _f32(_LN2_F32), log_sum)
                            K.ptx.st.global_.b32(
                                lse.ptr_to([stat_idx]),
                                K.if_then_else(sum_positive != _i32(0), lse_value, _f32(_NEG_INF)),
                            )
                        with K.If(return_temperature_lse != _i32(0)), K.Then():
                            log2_tsum = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(log2_tsum, final_temperature_sum)
                            t_value = K.local_scalar("float32", init=_f32(_NEG_INF))
                            with K.If(final_temperature_sum > _f32(0.0)), K.Then():
                                max_scaled_t = K.local_scalar("float32")
                                K.ptx.mul.f32(max_scaled_t, final_max, softmax_scale_log2)
                                max_ln = K.local_scalar("float32")
                                K.ptx.mul.f32(max_ln, max_scaled_t, _f32(_LN2_F32))
                                log_tsum = K.local_scalar("float32")
                                K.ptx.mul.f32(log_tsum, log2_tsum, _f32(_LN2_F32))
                                K.ptx.fma.rn.f32(t_value, lse_temperature_scale, max_ln, log_tsum)
                            K.ptx.st.global_.b32(temperature_lse.ptr_to([stat_idx]), t_value)
            K.ptx.tcgen05.wait__ld.sync.aligned()
            K.ptx["tcgen05.fence::before_thread_sync"]()
            _mbar_arrive(bar(_MBAR_TMEM_DEALLOC))

        # ---- Role: MMA issuer (warp 12) ----
        with r_mma:
            union_ready_phase_m = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_m)
            _flip(union_ready_phase_m)
            union_count_0_m, union_count_1_m = _ld_shared_counts(bar(_SMEM_UNION_COUNT))
            max_union_count_m = K.local_scalar("int32", init=union_count_0_m)
            with K.If(max_union_count_m < union_count_1_m), K.Then():
                K.assign(max_union_count_m, union_count_1_m)
            kv_stage = K.local_scalar("uint32", init=_u32(0))
            kv_phase = K.local_scalar("int32", init=_i32(0))
            q_full_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_Q_FULL), q_full_phase)
            _flip(q_full_phase)
            first_pv = [K.local_scalar("int32", init=_i32(1)) for _ in range(2)]
            phase_p_full = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            q_addr = (bar(_SMEM_Q0), bar(_SMEM_Q1))
            desc_hi = _u32(_DESC_HI)

            def qk_chain(instance, k_stage_now):
                a_lo = K.local_scalar(
                    "uint32", init=K.uniform((q_addr[instance] >> 4) & _u32(0x3FFF))
                )
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(((bar(_SMEM_K) >> 4) & _u32(0x3FFF)) + k_stage_now * _u32(2048)),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                d_tmem = taddr + _u32(_TMEM_SCORES[instance])
                for k16 in range(8):
                    K.ptx[_MMA_F16](
                        d_tmem,
                        _pack2(a_lo, desc_hi),
                        _pack2(b_lo, desc_hi),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(_u32(0 if k16 == 0 else 1)),
                        pred=leader,
                    )
                    if k16 < 7:
                        K.assign(a_lo, a_lo + _u32(_QK_A_STEPS[k16]))
                        K.assign(b_lo, b_lo + _u32(_QK_B_STEPS[k16]))
                commit_s_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                K.ptx[_TCGEN05_COMMIT](bar(_MBAR_S_FULL + 8 * instance), pred=commit_s_leader)
                commit_k_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                K.ptx[_TCGEN05_COMMIT](
                    bar(_MBAR_K_EMPTY) + k_stage_now * _u32(8), pred=commit_k_leader
                )

            def pv_chain(instance, v_stage_now):
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        (((bar(_SMEM_V) >> 4) & _u32(0x3FFF)) | _u32(_V_LBO_BIT))
                        + v_stage_now * _u32(2048)
                    ),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                d_tmem = taddr + _u32(_TMEM_OUTPUT[instance])
                a_tmem = K.local_scalar(
                    "uint32", init=taddr + _u32(_TMEM_SCORES[instance] + _TMEM_P_OFFSET)
                )
                enable_first = K.local_scalar(
                    "uint32", init=K.if_then_else(first_pv[instance] != _i32(0), _u32(0), _u32(1))
                )
                for k16 in range(8):
                    K.ptx[_MMA_F16](
                        d_tmem,
                        a_tmem,
                        _pack2(b_lo, desc_hi),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(enable_first if k16 == 0 else _u32(1)),
                        pred=leader,
                    )
                    if k16 < 7:
                        K.assign(a_tmem, a_tmem + _u32(_PV_A_STEP))
                        K.assign(b_lo, b_lo + _u32(_PV_B_STEP))
                K.assign(first_pv[instance], _i32(0))

            union_index_m = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_m < max_union_count_m):
                for instance in range(2):
                    count = union_count_0_m if instance == 0 else union_count_1_m
                    with K.If(count > union_index_m), K.Then():
                        k_stage_now = K.local_scalar("uint32", init=kv_stage)
                        k_phase_now = K.local_scalar("int32", init=kv_phase)
                        _advance_ring(kv_stage, kv_phase, 3)
                        _mbar_wait(bar(_MBAR_K_FULL) + k_stage_now * _u32(8), k_phase_now)
                        qk_chain(instance, k_stage_now)
                for instance in range(2):
                    count = union_count_0_m if instance == 0 else union_count_1_m
                    with K.If(count > union_index_m), K.Then():
                        v_stage_now = K.local_scalar("uint32", init=kv_stage)
                        v_phase_now = K.local_scalar("int32", init=kv_phase)
                        _advance_ring(kv_stage, kv_phase, 3)
                        _mbar_wait(bar(_MBAR_K_FULL) + v_stage_now * _u32(8), v_phase_now)
                        _mbar_wait(bar(_MBAR_P_FULL + 8 * instance), phase_p_full[instance])
                        _flip(phase_p_full[instance])
                        pv_chain(instance, v_stage_now)
                        with K.If(union_index_m + _i32(1) == count), K.Then():
                            commit_o_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                            K.ptx[_TCGEN05_COMMIT](
                                bar(_MBAR_O_FULL + 8 * instance), pred=commit_o_leader
                            )
                        commit_v_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                        K.ptx[_TCGEN05_COMMIT](
                            bar(_MBAR_K_EMPTY) + v_stage_now * _u32(8), pred=commit_v_leader
                        )
                K.assign(union_index_m, union_index_m + _i32(1))
            dealloc_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_TMEM_DEALLOC), dealloc_phase)
            _flip(dealloc_phase)
            taddr_dealloc = K.local_scalar("uint32")
            K.ptx.ld.volatile.shared.b32(taddr_dealloc, bar(_SMEM_TMEM_MAILBOX))
            K.ptx[_TMEM_DEALLOC](taddr_dealloc, _u32(TMEM_COLS))

        # ---- Role: idle (warps 13-14) ----
        with r_idle:
            pass

        # ---- Role: Q/mask/union/K/V loader (warp 15) ----
        with r_load:
            (
                batch_l,
                q_tile_l,
                tile_active_l,
                q_begin_l,
                q_len_l,
                q_local_base_l,
                q_valid_l,
                query_base_l,
                k_start_l,
                kv_len_l,
                query_offset_l,
                num_n_blocks_l,
            ) = decode_tile()
            del batch_l, q_tile_l, tile_active_l, q_begin_l, q_len_l, kv_len_l, query_offset_l
            q_head_base_l = kv_head * GQA_RATIO
            load_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(load_leader != _u32(0)), K.Then():
                _mbar_expect_tx(bar(_MBAR_Q_FULL), 32768)
                K.ptx[_TMA_G2S_4D](
                    bar(_SMEM_Q0),
                    K.address_of(q_map),
                    _i32(0),
                    q_head_base_l,
                    query_base_l,
                    _i32(0),
                    bar(_MBAR_Q_FULL),
                )
                K.ptx[_TMA_G2S_4D](
                    bar(_SMEM_Q1),
                    K.address_of(q_map),
                    _i32(0),
                    q_head_base_l,
                    query_base_l + _i32(4),
                    _i32(0),
                    bar(_MBAR_Q_FULL),
                )

            token = lane
            token_mask_low = K.local_scalar("uint32", init=_u32(0))
            token_mask_high = K.local_scalar("uint32", init=_u32(0))
            with K.If(token < q_valid_l), K.Then():
                selection_base = (kv_head * total_q + query_base_l + token) * topk
                slot = K.local_scalar("int32", init=_i32(0))
                with K.While(slot < topk):
                    selected_block = _ld_global_i32(q2k_indices, selection_base + slot)
                    with (
                        K.If(K.And(selected_block >= _i32(0), selected_block < num_n_blocks_l)),
                        K.Then(),
                    ):
                        with K.If(selected_block < _i32(32)):
                            with K.Then():
                                K.assign(
                                    token_mask_low,
                                    token_mask_low
                                    | K.shift_left(_u32(1), K.cast(selected_block, "uint32")),
                                )
                            with K.Else():
                                K.assign(
                                    token_mask_high,
                                    token_mask_high
                                    | K.shift_left(
                                        _u32(1), K.cast(selected_block - _i32(32), "uint32")
                                    ),
                                )
                    K.assign(slot, slot + _i32(1))
            with K.If(token < _i32(8)), K.Then():
                K.ptx.st.shared.b32(
                    bar(_SMEM_MASK_LOW) + K.cast(token, "uint32") * _u32(4), token_mask_low
                )
                K.ptx.st.shared.b32(
                    bar(_SMEM_MASK_HIGH) + K.cast(token, "uint32") * _u32(4), token_mask_high
                )
            K.ptx.barrier.sync(8, 32)

            with K.If(lane < _i32(2)), K.Then():
                instance_l = lane
                union_low = K.local_scalar("uint32", init=_u32(0))
                union_high = K.local_scalar("uint32", init=_u32(0))
                for query_in_instance in range(4):
                    query_in_tile = instance_l * _i32(4) + _i32(query_in_instance)
                    mask_lo = _ld_shared_i32(
                        bar(_SMEM_MASK_LOW) + K.cast(query_in_tile, "uint32") * _u32(4)
                    )
                    mask_hi = _ld_shared_i32(
                        bar(_SMEM_MASK_HIGH) + K.cast(query_in_tile, "uint32") * _u32(4)
                    )
                    K.assign(union_low, union_low | K.cast(mask_lo, "uint32"))
                    K.assign(union_high, union_high | K.cast(mask_hi, "uint32"))
                union_count_l = K.local_scalar("int32", init=_i32(0))
                high_count = K.local_scalar("uint32")
                K.ptx.popc.b32(high_count, union_high)
                high_index = K.local_scalar("int32", init=_i32(0))
                with K.While(high_index < _i32(32)):
                    with K.If(K.cast(high_count, "int32") > high_index), K.Then():
                        leading = K.local_scalar("uint32")
                        K.ptx.clz.b32(leading, union_high)
                        bit = _i32(31) - K.cast(leading, "int32")
                        union_addr = bar(_SMEM_UNION_BLOCKS) + K.cast(
                            instance_l * _i32(MAX_SELECTED_BLOCKS) + union_count_l, "uint32"
                        ) * _u32(4)
                        K.ptx.st.shared.b32(union_addr, bit + _i32(32))
                        K.assign(
                            union_high, union_high ^ K.shift_left(_u32(1), K.cast(bit, "uint32"))
                        )
                        K.assign(union_count_l, union_count_l + _i32(1))
                    K.assign(high_index, high_index + _i32(1))
                low_count = K.local_scalar("uint32")
                K.ptx.popc.b32(low_count, union_low)
                low_index = K.local_scalar("int32", init=_i32(0))
                with K.While(low_index < _i32(32)):
                    with K.If(K.cast(low_count, "int32") > low_index), K.Then():
                        leading = K.local_scalar("uint32")
                        K.ptx.clz.b32(leading, union_low)
                        bit = _i32(31) - K.cast(leading, "int32")
                        union_addr = bar(_SMEM_UNION_BLOCKS) + K.cast(
                            instance_l * _i32(MAX_SELECTED_BLOCKS) + union_count_l, "uint32"
                        ) * _u32(4)
                        K.ptx.st.shared.b32(union_addr, bit)
                        K.assign(
                            union_low, union_low ^ K.shift_left(_u32(1), K.cast(bit, "uint32"))
                        )
                        K.assign(union_count_l, union_count_l + _i32(1))
                    K.assign(low_index, low_index + _i32(1))
                with K.If(union_count_l == _i32(0)), K.Then():
                    K.ptx.st.shared.b32(
                        bar(_SMEM_UNION_BLOCKS)
                        + K.cast(instance_l * _i32(MAX_SELECTED_BLOCKS), "uint32") * _u32(4),
                        _i32(0),
                    )
                    K.assign(union_count_l, _i32(1))
                K.ptx.st.shared.b32(
                    bar(_SMEM_UNION_COUNT) + K.cast(instance_l, "uint32") * _u32(4), union_count_l
                )
            K.ptx.barrier.sync(8, 32)
            K.ptx.fence.proxy.async_.shared__cta()
            _mbar_arrive(bar(_MBAR_UNION_READY))

            union_count_0_l, union_count_1_l = _ld_shared_counts(bar(_SMEM_UNION_COUNT))
            max_union_count_l = K.local_scalar("int32", init=union_count_0_l)
            with K.If(max_union_count_l < union_count_1_l), K.Then():
                K.assign(max_union_count_l, union_count_1_l)
            load_stage = K.local_scalar("uint32", init=_u32(0))
            phase_kv_empty = K.local_scalar("int32", init=_i32(1))

            def load_kv_tile(tensor_map, n_block, is_v):
                token_base = k_start_l + n_block * _i32(BLOCK_SIZE)
                stage_now = K.local_scalar("uint32", init=load_stage)
                phase_now = K.local_scalar("int32", init=phase_kv_empty)
                _mbar_wait(bar(_MBAR_K_EMPTY) + stage_now * _u32(8), phase_now)
                push_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                with K.If(push_leader != _u32(0)), K.Then():
                    full_bar = bar(_MBAR_K_FULL) + stage_now * _u32(8)
                    _mbar_expect_tx(full_bar, 32768)
                    stage_addr = bar(_SMEM_V if is_v else _SMEM_K) + stage_now * _u32(
                        _SMEM_STAGE_BYTES
                    )
                    for dim_half in range(2):
                        for token_half in range(2):
                            token_coord = token_base + _i32(64 * token_half)
                            K.ptx[_TMA_G2S_4D](
                                stage_addr + _u32(8192 * token_half + 16384 * dim_half),
                                K.address_of(tensor_map),
                                _i32(0),
                                token_coord,
                                _i32(dim_half),
                                kv_head,
                                full_bar,
                            )
                _advance_ring(load_stage, phase_kv_empty, 3)

            union_index_l = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_l < max_union_count_l):
                for instance in range(2):
                    count = union_count_0_l if instance == 0 else union_count_1_l
                    with K.If(count > union_index_l), K.Then():
                        n_block = _ld_shared_i32(
                            bar(_SMEM_UNION_BLOCKS + MAX_SELECTED_BLOCKS * 4 * instance)
                            + K.cast(union_index_l, "uint32") * _u32(4)
                        )
                        load_kv_tile(k_map, n_block, False)
                for instance in range(2):
                    count = union_count_0_l if instance == 0 else union_count_1_l
                    with K.If(count > union_index_l), K.Then():
                        n_block = _ld_shared_i32(
                            bar(_SMEM_UNION_BLOCKS + MAX_SELECTED_BLOCKS * 4 * instance)
                            + K.cast(union_index_l, "uint32") * _u32(4)
                        )
                        load_kv_tile(v_map, n_block, True)
                K.assign(union_index_l, union_index_l + _i32(1))

    return blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


def get_kernel(**config: Any):
    """Return the shape-generic M64 TIRx PrimFunc."""
    if config:
        _validate_config(**_without_label(config))
    return _kernel().func


@lru_cache(maxsize=1)
def _compiled_kernel():
    from tirx_kernels.runner import compile_kernel

    # nvcc mode keeps the TIRx build on the CUDA toolkit on PATH (13.3, PTX ISA
    # 9.3) -- the same toolchain the upstream ``nvcc -cubin`` build uses -- so
    # both binaries share one PTX ISA for SASS and profiler comparisons.
    # Match the native ptxas register-usage level. The narrow 48-register
    # producer roles spill descriptor and uniform values at TVM level 10,
    # default even though the source build has no local-memory traffic.
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REGISTER_USAGE_LEVEL
    try:
        return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def _tirx_args(data: dict[str, Any], slot: str = "tirx") -> tuple[Any, ...]:
    buffers = data[slot]
    return (
        data["q"],
        data["k"],
        data["v"],
        buffers["out"].view(-1),
        buffers["lse"].view(-1),
        buffers["tlse"].view(-1),
        data["q2k_indices"].view(-1),
        data["cu_q"].view(-1),
        data["cu_k"].view(-1),
        data["q_offsets"].view(-1),
        data["kv_lens_tensor"].view(-1),
        data["total_q"],
        data["num_q_heads"],
        data["num_kv_heads"],
        TOPK,
        data["batch_size"],
        0,
        int(data["causal"]),
        int(data["derive_q_offset"]),
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"] or data["return_temperature_lse"]),
        int(data["return_temperature_lse"]),
    )


def _tirx_launch(executable, data: dict[str, Any]):
    arguments = _tirx_args(data)

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


@lru_cache(maxsize=1)
def _source_module():
    """Load the exact FlashInfer M64 BF16 GQA16 flat source specialization."""
    from flashinfer.msa_ops import _blackwell_sm100

    return _blackwell_sm100._get_module("prefill_m64_bf16_gqa16_flat", "sm103a")


def _source_launch(data: dict[str, Any]):
    import torch

    module = _source_module()
    buffers = data["source"]
    grid_x = (data["total_q"] + 7) // 8 + data["batch_size"] - 1
    arguments = (
        data["q"],
        data["k"],
        data["v"],
        buffers["out"],
        buffers["lse"],
        buffers["tlse"],
        data["q2k_indices"],
        data["cu_q"],
        data["cu_k"],
        data["q_offsets"],
        data["kv_lens_tensor"],
        data["total_q"],
        data["num_q_heads"],
        data["num_kv_heads"],
        TOPK,
        data["batch_size"],
        0,
        int(data["causal"]),
        int(data["derive_q_offset"]),
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"] or data["return_temperature_lse"]),
        int(data["return_temperature_lse"]),
        grid_x,
        data["num_kv_heads"],
        1,
        int(torch.cuda.current_stream(data["q"].device).cuda_stream),
    )

    def launch():
        module.run(*arguments)

    launch._keep_alive = (module, arguments)
    return launch


_GUARD_ELEMS = 64
_OUT_GUARD = 42.5
_STATS_GUARD = -54321.25
_STATS_SENTINEL = 12345.25
_LN2 = math.log(2.0)


def _guarded_tensor(shape, dtype, *, fill: float, guard: float, device):
    import torch

    elements = math.prod(shape)
    storage = torch.full((elements + 2 * _GUARD_ELEMS,), guard, dtype=dtype, device=device)
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + elements].view(shape)
    view.fill_(fill)
    return view, storage


def _make_q2k_indices(
    *, q_lens, kv_lens, num_kv_heads: int, total_q: int, selection: str, seed: int
):
    import torch

    generator = torch.Generator().manual_seed(seed)
    result = torch.full((num_kv_heads, total_q, TOPK), -1, dtype=torch.int32)
    q_begin = 0
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        del batch
        blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        if selection == "future_only":
            if blocks > 1:
                result[:, q_begin : q_begin + q_len, 0] = blocks - 1
        elif selection == "disjoint_union":
            if blocks < 64 or q_len < 8:
                raise ValueError("disjoint_union requires at least 64 blocks and eight queries")
            for query in range(q_len):
                base = (query % 4) * TOPK
                values = (torch.arange(TOPK, dtype=torch.int32) + base) % blocks
                result[:, q_begin + query, :] = values
        elif selection == "random_valid":
            take = min(TOPK, blocks)
            scores = torch.rand((num_kv_heads, q_len, blocks), generator=generator)
            choices = scores.argsort(dim=-1)[..., :take].to(torch.int32)
            result[:, q_begin : q_begin + q_len, :take] = choices
        else:
            raise ValueError(f"unknown selection pattern {selection!r}")
        q_begin += q_len
    return result.contiguous()


def prepare_data(**config: Any) -> dict[str, Any]:
    """Create deterministic flat-ragged MSA inputs and independent guarded outputs."""
    import torch

    config = _without_label(config)
    _validate_config(**config)
    q_lens = tuple(int(x) for x in config["q_lens"])
    kv_lens = tuple(int(x) for x in config["kv_lens"])
    num_q_heads = int(config["num_q_heads"])
    num_kv_heads = int(config["num_kv_heads"])
    total_q = sum(q_lens)
    total_k = sum(kv_lens)
    batch_size = len(q_lens)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    q = torch.randn(
        (total_q, num_q_heads, HEAD_DIM), device=device, dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        (total_k, num_kv_heads, HEAD_DIM), device=device, dtype=torch.bfloat16, generator=generator
    )
    v = torch.randn(
        (total_k, num_kv_heads, HEAD_DIM), device=device, dtype=torch.bfloat16, generator=generator
    )
    q2k_indices = _make_q2k_indices(
        q_lens=q_lens,
        kv_lens=kv_lens,
        num_kv_heads=num_kv_heads,
        total_q=total_q,
        selection=config["selection"],
        seed=int(config["seed"]) + 1729,
    ).to(device)
    cu_q = torch.tensor(
        (0, *tuple(torch.tensor(q_lens).cumsum(0).tolist())), dtype=torch.int32, device=device
    )
    cu_k = torch.tensor(
        (0, *tuple(torch.tensor(kv_lens).cumsum(0).tolist())), dtype=torch.int32, device=device
    )
    derive_q_offset = config.get("q_offsets") is None
    explicit_offsets = (
        tuple(0 for _ in q_lens) if derive_q_offset else tuple(int(x) for x in config["q_offsets"])
    )
    q_offsets = torch.tensor(explicit_offsets, dtype=torch.int32, device=device)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return_lse = bool(config["return_softmax_lse"])
    return_tlse = bool(config["return_temperature_lse"])

    def outputs():
        out, out_storage = _guarded_tensor(
            (total_q, num_q_heads, HEAD_DIM),
            torch.bfloat16,
            fill=float("nan"),
            guard=_OUT_GUARD,
            device=device,
        )
        lse, lse_storage = _guarded_tensor(
            (total_q, num_q_heads),
            torch.float32,
            fill=float("nan") if (return_lse or return_tlse) else _STATS_SENTINEL,
            guard=_STATS_GUARD,
            device=device,
        )
        tlse, tlse_storage = _guarded_tensor(
            (total_q, num_q_heads),
            torch.float32,
            fill=float("nan") if return_tlse else _STATS_SENTINEL,
            guard=_STATS_GUARD,
            device=device,
        )
        return {
            "out": out,
            "lse": lse,
            "tlse": tlse,
            "guards": {"out": out_storage, "lse": lse_storage, "tlse": tlse_storage},
        }

    return {
        "config": config,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "total_k": total_k,
        "batch_size": batch_size,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "causal": bool(config["causal"]),
        "derive_q_offset": derive_q_offset,
        "return_lse": return_lse,
        "return_temperature_lse": return_tlse,
        "softmax_scale_log2": (HEAD_DIM**-0.5) / _LN2,
        "lse_temperature_scale": float(config["lse_temperature_scale"]),
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "q_offsets": q_offsets,
        "kv_lens_tensor": kv_lens_tensor,
        "tirx": outputs(),
        "source": outputs(),
    }


def _assert_guards(name: str, buffers: dict[str, Any]) -> None:
    import torch

    for key, storage in buffers["guards"].items():
        guard = _OUT_GUARD if key == "out" else _STATS_GUARD
        expected = torch.full((_GUARD_ELEMS,), guard, dtype=storage.dtype, device=storage.device)
        if not torch.equal(storage[:_GUARD_ELEMS], expected):
            raise AssertionError(f"{name}.{key} prefix guard was modified")
        if not torch.equal(storage[-_GUARD_ELEMS:], expected):
            raise AssertionError(f"{name}.{key} suffix guard was modified")


def _assert_bitwise(name: str, ours, source) -> None:
    import torch

    ours_bits = ours.view(torch.uint16) if ours.dtype == torch.bfloat16 else ours.view(torch.int32)
    source_bits = (
        source.view(torch.uint16) if source.dtype == torch.bfloat16 else source.view(torch.int32)
    )
    equal = ours_bits == source_bits
    if bool(equal.all()):
        return
    first = int((~equal).view(-1).nonzero()[0].item())
    ours_value = float(ours.view(-1)[first].item())
    source_value = float(source.view(-1)[first].item())
    finite = torch.isfinite(ours.float()) & torch.isfinite(source.float())
    max_abs = (
        float((ours.float()[finite] - source.float()[finite]).abs().max().item())
        if bool(finite.any())
        else float("nan")
    )
    raise AssertionError(
        f"{name} is not bitwise equal to FlashInfer at flat index {first}: "
        f"tirx={ours_value}, source={source_value}, max_finite_abs={max_abs}"
    )


def _validate_outputs(data: dict[str, Any]) -> dict[str, float]:
    _assert_guards("tirx", data["tirx"])
    _assert_guards("source", data["source"])
    _assert_bitwise("out", data["tirx"]["out"], data["source"]["out"])
    if data["return_lse"] or data["return_temperature_lse"]:
        _assert_bitwise("lse", data["tirx"]["lse"], data["source"]["lse"])
    else:
        if not bool((data["tirx"]["lse"] == _STATS_SENTINEL).all()):
            raise AssertionError("disabled TIRx LSE storage was modified")
        if not bool((data["source"]["lse"] == _STATS_SENTINEL).all()):
            raise AssertionError("disabled source LSE storage was modified")
    if data["return_temperature_lse"]:
        _assert_bitwise("temperature_lse", data["tirx"]["tlse"], data["source"]["tlse"])
    else:
        if not bool((data["tirx"]["tlse"] == _STATS_SENTINEL).all()):
            raise AssertionError("disabled TIRx temperature-LSE storage was modified")
        if not bool((data["source"]["tlse"] == _STATS_SENTINEL).all()):
            raise AssertionError("disabled source temperature-LSE storage was modified")
    return {"out_max_abs": 0.0, "lse_max_abs": 0.0, "temperature_lse_max_abs": 0.0}


def _skip_unless_supported() -> None:
    import unittest

    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (10, 3):
        raise unittest.SkipTest("this kernel requires compute capability 10.3")


def run_test(**config: Any) -> dict[str, float]:
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    executable = _compiled_kernel()
    tirx_launch = _tirx_launch(executable, data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    return _validate_outputs(data)


def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_config(**kernel_config)
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
        _skip_unless_supported()
        data = prepare_data(**config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executable"], data),
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
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
