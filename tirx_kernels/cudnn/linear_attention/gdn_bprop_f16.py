# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell FP16/BF16 Gated Delta Net backward TIRx port.

Upstream source:
``python/cudnn/linear_attention/frost/kernel/gdn_bprop_f16.py``
(``prologue_kernel``, ``kernel``, and the two-launch ``run_bwd`` entry).

The implementation preserves the upstream two-launch TensorMap prologue and
persistent backward-main contract using only ``tirx_kernels.kern`` device APIs.
"""

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_gdn_bprop_f16",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}
CONFIGS = [
    {"label": "basic_bf16_full", "seq_lens": (64,), "heads": 1},
    {"label": "basic_fp16_full", "seq_lens": (64,), "heads": 1, "dtype": "float16"},
    {"label": "tail_zero", "seq_lens": (1, 63, 64, 65, 0), "heads": 2},
    {"label": "gqa", "seq_lens": (128,), "heads": 64, "q_heads": 64, "k_heads": 16, "v_heads": 16},
    {"label": "gva", "seq_lens": (128,), "heads": 64, "q_heads": 16, "k_heads": 16, "v_heads": 64},
    {"label": "safe_gate", "seq_lens": (128,), "heads": 2, "safe_gate": True},
    {"label": "beta_sigmoid", "seq_lens": (128,), "heads": 2, "beta_sigmoid": True},
    {
        "label": "state_io",
        "seq_lens": (32, 0, 96),
        "heads": 2,
        "use_initial_state": True,
        "use_dstate_in": True,
        "use_dstate0": True,
    },
    {"label": "initial_state_only", "seq_lens": (0, 65), "heads": 2, "use_initial_state": True},
    {"label": "final_state_only", "seq_lens": (65,), "heads": 2, "use_dstate_in": True},
    {"label": "raw_gate", "seq_lens": (17, 65), "heads": 2, "log_gate": False},
    {"label": "dynamic_scheduler", "seq_lens": (32, 96), "heads": 2, "dynamic_scheduler": True},
    {"label": "order_scratch", "seq_lens": (32, 96), "heads": 2, "run_order": True},
    {
        "label": "order_generate",
        "seq_lens": (32, 96),
        "heads": 2,
        "run_order": True,
        "order_generate": True,
    },
    {"label": "split_scratch", "seq_lens": (257,), "heads": 2, "run_order": True, "split": True},
    {"label": "nondefault_scale", "seq_lens": (128,), "heads": 2, "scale": 0.073},
    {"label": "multi_tile", "seq_lens": (192,) * 8, "heads": 64},
    {"label": "strong_decay_ragged", "seq_lens": (17, 129, 255), "heads": 4, "strong_decay": True},
    {"label": "int64_cu", "seq_lens": (63, 65), "heads": 2, "cu_dtype": "int64"},
]

BENCH_CONFIGS = [
    {"label": "perf_basic_b1_s2048_h16", "seq_lens": (2048,), "heads": 16},
    {"label": "perf_basic_b1_s8192_h16", "seq_lens": (8192,), "heads": 16},
    {"label": "perf_fp16_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "dtype": "float16"},
    {"label": "perf_ragged_b3_h16", "seq_lens": (2047, 0, 4093), "heads": 16},
    {
        "label": "perf_gqa_b1_s8192_hq64_hk16_hv16",
        "seq_lens": (8192,),
        "heads": 64,
        "q_heads": 64,
        "k_heads": 16,
        "v_heads": 16,
    },
    {
        "label": "perf_gva_b1_s8192_hq16_hk16_hv64",
        "seq_lens": (8192,),
        "heads": 64,
        "q_heads": 16,
        "k_heads": 16,
        "v_heads": 64,
    },
    {"label": "perf_safe_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "safe_gate": True},
    {"label": "perf_beta_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "beta_sigmoid": True},
    {
        "label": "perf_state_b1_s8192_h16",
        "seq_lens": (8192,),
        "heads": 16,
        "use_initial_state": True,
        "use_dstate_in": True,
        "use_dstate0": True,
    },
    {
        "label": "perf_dynamic_b4_s2048_h64",
        "seq_lens": (2048,) * 4,
        "heads": 64,
        "dynamic_scheduler": True,
    },
    {
        "label": "perf_order_scratch_b4_s2048_h64",
        "seq_lens": (2048,) * 4,
        "heads": 64,
        "run_order": True,
    },
    {
        "label": "perf_order_generate_b4_s2048_h64",
        "seq_lens": (2048,) * 4,
        "heads": 64,
        "run_order": True,
        "order_generate": True,
    },
    {"label": "perf_b4_s2048_h64", "seq_lens": (2048,) * 4, "heads": 64},
    {"label": "perf_b4_s4096_h64", "seq_lens": (4096,) * 4, "heads": 64},
    {"label": "perf_b4_s8192_h64", "seq_lens": (8192,) * 4, "heads": 64},
    {"label": "perf_b4_s16384_h64", "seq_lens": (16384,) * 4, "heads": 64},
    {"label": "perf_b4_s32768_h64", "seq_lens": (32768,) * 4, "heads": 64},
    {"label": "perf_b1_s8192_h64", "seq_lens": (8192,), "heads": 64},
    {"label": "perf_b2_s8192_h64", "seq_lens": (8192,) * 2, "heads": 64},
    {"label": "perf_b8_s8192_h64", "seq_lens": (8192,) * 8, "heads": 64},
    {"label": "perf_b16_s8192_h64", "seq_lens": (8192,) * 16, "heads": 64},
]


_BT = 64
_DK = 128
_DV = 128
_TENSOR_MAP_BYTES = 128
_TENSOR_MAP_WORDS = 16
_TENSOR_MAP_ARRAYS = 8
_TRY_WAIT_TICKS = 0
_RCP_LN2 = 1.4426950408889634
_LN2 = 0.6931471805599453
_TMA_G2S_3D = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
_TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"

# Reviewer-reconciled declaration-order mbarrier header.
_BAR_Q_READY = 0
_BAR_Q_MMA_DONE = 16
_BAR_Q_CG1_DONE = 32
_BAR_K_READY = 48
_BAR_K_MMA_DONE = 64
_BAR_K_CG0_DONE = 80
_BAR_V_READY = 96
_BAR_V_MMA_DONE = 112
_BAR_DO_READY = 128
_BAR_DO_MMA_DONE = 144
_BAR_STATE_READY = 160
_BAR_STATE_MMA_DONE = 176
_BAR_GATE_READY = 192
_BAR_GATE_DONE = 208
_BAR_BETA_READY = 224
_BAR_BETA_DONE = 240
_BAR_DSTATE_ACC_READY = 256
_BAR_DSTATE_SCALE_DONE = 272
_BAR_DU_SCALE_READY = 288
_BAR_DU_SCALE_DONE = 304
_BAR_DU_TOTAL_READY = 320
_BAR_DK_SCALE_READY = 336
_BAR_DK_SCALE_DONE = 352
_BAR_DK_ATTN_READY = 368
_BAR_DK_ATTN_DONE = 384
_BAR_DK_TOTAL_READY = 400
_BAR_DK_TOTAL_DONE = 416
_BAR_DQ_SCALE_READY = 432
_BAR_DQ_SCALE_DONE = 448
_BAR_DQ_TOTAL_READY = 464
_BAR_DQ_TOTAL_DONE = 480
_BAR_KK_READY = 496
_BAR_KK_DONE = 512
_BAR_A_ACC_READY = 528
_BAR_K_STATE_READY = 544
_BAR_U_ACC_READY = 560
_BAR_DY_ACC_READY = 576
_BAR_DA_ACC_READY = 592
_BAR_DM_ACC_READY = 608
_BAR_DM_ACC_DONE = 624
_BAR_DK_STATE_READY = 640
_BAR_DSTATE_INP_READY = 656
_BAR_DSTATE_INP_DONE = 672
_BAR_DO_PRIME_READY = 688
_BAR_DU_INP_READY = 704
_BAR_DYP_INP_READY = 720
_BAR_Y_READY = 736
_BAR_TINV_READY = 752
_BAR_A_READY = 768
_BAR_A_DONE = 784
_BAR_U_READY = 800
_BAR_DSTATE_SMEM_READY = 816
_BAR_STATE_DOT_DONE = 832
_BAR_DA_READY = 848
_BAR_DBETA_CG1_READY = 864
_BAR_DGATE_CG1_READY = 880
_BAR_DQ_STG_READY = 896
_BAR_DQ_STG_DONE = 912
_BAR_DK_STG_READY = 928
_BAR_DK_STG_DONE = 944
_BAR_DV_STG_READY = 960
_BAR_DV_STG_DONE = 976
_BAR_SDV_DONE = 992
_BAR_TMEM_DONE = 1008
_BAR_SCHED_READY = 1024
_BAR_SCHED_DONE = 1040

_SCHED_BASE = 1056
_TMEM_MAILBOX = 1072
_CUMSUMLOG_BASE = 1152
_CUMPROD_BASE = 1664
_BETA_BASE = 2176
_Q_BASE = 3072
_K_BASE = 19456
_DO_BASE = 52224
_STATE_BASE = 68608
_TINV_BASE = 101376
_KK_BASE = 109568
_A_DA_BASE = 117760
_DM_BASE = 125952
_V_U_BASE = 134144
_DSTATE_BASE = 150528
_DQ_BASE = 183296
_DK_BASE = 199680
_DV_DY_BASE = 216064
_ARENA_BYTES = 232448

_TM_DSTATE = 0
_TM_DVDK = 128
_TM_DSTATE_INP = 192
_TM_ACC0 = 256
_TM_ACC1 = 320
_TM_INP0 = 384
_TM_INP1 = 416
_TM_Y = 448
_TM_GK = 480

_IDESC_M64_N64_BF16 = 68158608
_IDESC_M128_N64_BF16 = 135267472
_IDESC_M128_N64_A_BF16 = 135300240
_IDESC_M128_N64_B_BF16 = 135333008
_IDESC_M128_N64_AB_BF16 = 135365776
_IDESC_M128_N128_B_BF16 = 136381584


def _elected():
    lane = K.local_scalar("uint32")
    pred = K.local_scalar("uint32")
    K.ptx.elect_sync(lane, pred, K.uint32(0xFFFFFFFF))
    return pred == K.uint32(1)


def _load_tensormap(payload, src_map):
    """Load one host-encoded 128-byte TensorMap image into registers.

    The base images use ordinary global pointers because K's PTX instruction
    surface intentionally has no parameter-space load; issuing the loads
    before the order pass hides their cold-memory latency, and one load
    serves every per-sequence slot of the array.
    """
    source = K.reinterpret("uint64", src_map.ptr_to([0]))
    for group in range(4):
        offset = K.uint64(group * 32)
        K.ptx.ld.global_.v4.b64(
            payload[group * 4],
            payload[group * 4 + 1],
            payload[group * 4 + 2],
            payload[group * 4 + 3],
            K.reinterpret("handle", source + offset),
        )


def _store_tensormap(dst, payload):
    target = K.reinterpret("uint64", dst)
    for word in range(16):
        K.ptx.st.global_.b64(K.reinterpret("handle", target + K.uint64(word * 8)), payload[word])


def _replace_tensormap_address(desc, address):
    K.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
        desc, K.reinterpret("uint64", address)
    )


def _replace_tensormap_dim(desc, ordinal, value):
    K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(
        desc, K.uint32(ordinal), K.cast(value, "uint32")
    )


def _barrier_ptr(arena, byte_offset, stage=0):
    return arena.ptr_to([byte_offset + K.cast(stage, "int32") * 8])


def _wait_barrier(arena, byte_offset, stage, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        barrier_address = K.cuda.cvta_generic_to_shared(arena.ptr_to([0])) + K.cast(
            byte_offset + stage * 8, "uint32"
        )
        K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, barrier_address, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _arrive_barrier(arena, byte_offset, stage=0):
    K.ptx.mbarrier.arrive.shared.b64(_barrier_ptr(arena, byte_offset, stage))


def _expect_tx(arena, byte_offset, stage, nbytes, *, pred=None):
    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
        _barrier_ptr(arena, byte_offset, stage), K.uint32(nbytes), pred=pred
    )


def _tcgen_commit(arena, byte_offset, stage=0, *, leader):
    K.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        _barrier_ptr(arena, byte_offset, stage), pred=leader
    )


def _load_work_item(work_items, tile):
    values = K.alloc_local((8,), "int32")
    K.ptx.ld.global_.v4.b32(
        values[0], values[1], values[2], values[3], work_items.ptr_to([tile * 8])
    )
    K.ptx.ld.global_.v4.b32(
        values[4], values[5], values[6], values[7], work_items.ptr_to([tile * 8 + 4])
    )
    return values


def _udiv_nonnegative(value, divisor):
    """Divide a scheduler field known nonnegative without signed fixup math."""
    return K.cast(K.cast(value, "uint32") // K.uint32(divisor), "int32")


def _load_cu(cu_seqlens, index, cu_dtype):
    if cu_dtype == "int64":
        wide = K.local_scalar("int64")
        K.ptx.ld.global_.s64(wide, cu_seqlens.ptr_to([index]))
        return K.cast(wide, "int32")
    value = K.local_scalar("int32")
    K.ptx.ld.global_.s32(value, cu_seqlens.ptr_to([index]))
    return value


def _load_state_value(pointer, index, state_dtype):
    value = K.local_scalar("float32")
    if state_dtype == "float32":
        K.ptx.ld.global_.f32(value, pointer.ptr_to([index]))
    else:
        bits = K.local_scalar("uint16")
        K.ptx.ld.global_.b16(bits, pointer.ptr_to([index]))
        K.ptx.cvt.f32.bf16(value, bits)
    return value


def _store_state_value(pointer, index, value, state_dtype):
    if state_dtype == "float32":
        K.ptx.st.global_.f32(pointer.ptr_to([index]), value)
    else:
        K.ptx.st.global_.b16(
            pointer.ptr_to([index]), K.reinterpret("uint16", K.cast(value, "bfloat16"))
        )


def _descriptor_slot(workspace, n_desc, array, batch):
    return workspace.ptr_to([(K.cast(array, "int64") * n_desc + batch) * _TENSOR_MAP_WORDS])


def _pack_io_pair(lo, hi, io_dtype):
    packed = K.local_scalar("uint32")
    if io_dtype == "float16":
        K.ptx.cvt.rn.f16x2.f32(packed, hi, lo)
    else:
        K.ptx.cvt.rn.bf16x2.f32(packed, hi, lo)
    return packed


def _unpack_io_pair(packed, lo, hi, io_dtype):
    halves = K.alloc_local((2,), "uint16")
    K.ptx.mov.b32(halves[0], halves[1], packed)
    if io_dtype == "float16":
        K.ptx.cvt.f32.f16(lo, halves[0])
        K.ptx.cvt.f32.f16(hi, halves[1])
    else:
        low_bits = K.local_scalar("uint32")
        high_bits = K.local_scalar("uint32")
        K.ptx.mov.b32(low_bits, K.uint16(0), halves[0])
        K.ptx.mov.b32(high_bits, K.uint16(0), halves[1])
        K.assign(lo, K.reinterpret("float32", low_bits))
        K.assign(hi, K.reinterpret("float32", high_bits))


def _sub_io_pair(dst, lhs, rhs, io_dtype):
    if io_dtype == "float16":
        K.ptx.sub.f16x2(dst, lhs, rhs)
    else:
        K.ptx["sub.bf16x2"](dst, lhs, rhs)


def _fadd2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.add.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _fsub2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.sub.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _fmul2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.mul.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _exp2(value):
    result = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(result, value)
    return result


def _lg2(value):
    result = K.local_scalar("float32")
    K.ptx.lg2.approx.ftz.f32(result, value)
    return result


def _tanh(value):
    result = K.local_scalar("float32")
    K.ptx.tanh.approx.f32(result, value)
    return result


def _softplus(value):
    # log(1 + exp(x)) with the source's linear tail (x > 20 returns x).
    grown = _exp2(value * K.float32(_RCP_LN2))
    smooth = _lg2(K.float32(1.0) + grown) * K.float32(_LN2)
    return K.if_then_else(value < K.float32(20.0), smooth, value)


def _opaque_zero():
    zero = K.local_scalar("float32")
    K.ptx.mov.b32(zero, K.uint32(0))
    return zero


def _shfl_idx_f32(value, source_lane, clamp):
    shuffled = K.local_scalar("uint32")
    K.ptx.shfl_sync.idx.b32(
        shuffled,
        K.reinterpret("uint32", value),
        K.cast(source_lane, "uint32"),
        K.uint32(clamp),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shuffled)


def _shfl_up_f32(value, delta):
    shuffled = K.local_scalar("uint32")
    K.ptx.shfl_sync.up.b32(
        shuffled, K.reinterpret("uint32", value), K.uint32(delta), K.uint32(0), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret("float32", shuffled)


def _shfl_down_f32(value, delta):
    shuffled = K.local_scalar("uint32")
    K.ptx.shfl_sync.down.b32(
        shuffled,
        K.reinterpret("uint32", value),
        K.uint32(delta),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shuffled)


def _shfl_bfly_f32(value, lane_mask):
    shuffled = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        shuffled,
        K.reinterpret("uint32", value),
        K.uint32(lane_mask),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shuffled)


def _ffma2(a0, a1, b0, b1, c0, c1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.fma.rn.f32x2(
        packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1), K.cuda.make_float2(c0, c1)
    )
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _rcp(value):
    result = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(result, value)
    return result


def _io_byte(base, stage, row, channel):
    segment = channel // 64
    within = channel % 64
    return base + stage * 16_384 + 2 * (segment * 4096 + row * 64 + _swizzle_xor_128b(row, within))


def _state_byte(base, key, value):
    segment = value // 64
    within = value % 64
    return base + 2 * (segment * 8192 + key * 64 + _swizzle_xor_128b(key, within))


def _tcgen_load_16x256x8(tmem_address):
    values = K.alloc_local((32,), "float32")
    K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[index] for index in range(32)], K.cast(tmem_address, "uint32")
    )
    return values


def _tcgen_store_16x256x8(tmem_address, values):
    K.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
        K.cast(tmem_address, "uint32"), *[values[index] for index in range(32)]
    )


def _tcgen_load_32x32x32(tmem_address):
    values = K.alloc_local((32,), "float32")
    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
        *[values[index] for index in range(32)], K.cast(tmem_address, "uint32")
    )
    return values


def _tcgen_store_32x32x32(tmem_address, values):
    K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
        K.cast(tmem_address, "uint32"), *[values[index] for index in range(32)]
    )


def _tcgen_store_32x32x16(tmem_address, values):
    K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
        K.cast(tmem_address, "uint32"), *[values[index] for index in range(16)]
    )


def _tcgen_store_16x128x8(tmem_address, values):
    K.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
        K.cast(tmem_address, "uint32"), *[values[index] for index in range(16)]
    )


def _tcgen_load_16x128x8(tmem_address):
    values = K.alloc_local((16,), "uint32")
    K.ptx["tcgen05.ld.sync.aligned.16x128b.x8.b32"](
        *[values[index] for index in range(16)], K.cast(tmem_address, "uint32")
    )
    return values


def _tcgen_wait_load():
    K.ptx.tcgen05.wait__ld.sync.aligned()


def _tcgen_wait_store():
    K.ptx.tcgen05.wait__st.sync.aligned()


def _raw_descriptor(arena, byte_offset, leading_bytes, stride_bytes, layout):
    shared = K.cuda.cvta_generic_to_shared(arena.ptr_to([byte_offset]))
    address = K.bitwise_and(K.shift_right(shared, K.uint32(4)), K.uint32(0x3FFF))
    fixed = (
        (((leading_bytes >> 4) & 0x3FFF) << 16) | (((stride_bytes >> 4) & 0x3FFF) << 32) | (1 << 46)
    )
    desc = K.bitwise_or(K.uint64(fixed), K.cast(address, "uint64"))
    return K.bitwise_or(desc, K.shift_left(K.uint64(layout & 7), K.uint64(61)))


def _swizzle_xor_128b(row, column):
    # physical column element for the 128-byte XOR swizzle of a bf16/f16 row
    return K.bitwise_xor(column, (row & 7) * 8)


def _swizzle_lin_128b(linear):
    # 128-byte XOR swizzle over a linear 64-element-row bf16/f16 index
    return K.bitwise_xor(linear, K.bitwise_and(K.shift_right(linear, K.int32(3)), K.int32(0x38)))


def _io_tile_byte(base, stage_bytes, stage, row, column):
    # byte offset of (row, column) in one 64-column swizzled IO tile stage
    return base + stage * stage_bytes + (row * 64 + _swizzle_xor_128b(row, column)) * 2


def _mma_m16n8k16(acc, acc_base, a, b0, b1, io_dtype):
    opcode = (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        if io_dtype == "float16"
        else "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    )
    K.ptx[opcode](
        acc[acc_base],
        acc[acc_base + 1],
        acc[acc_base + 2],
        acc[acc_base + 3],
        a[0],
        a[1],
        a[2],
        a[3],
        b0,
        b1,
        acc[acc_base],
        acc[acc_base + 1],
        acc[acc_base + 2],
        acc[acc_base + 3],
    )


def _mma_m16n8k8(acc, acc_base, a0, a1, b0, io_dtype):
    opcode = (
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        if io_dtype == "float16"
        else "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"
    )
    K.ptx[opcode](
        acc[acc_base],
        acc[acc_base + 1],
        acc[acc_base + 2],
        acc[acc_base + 3],
        a0,
        a1,
        b0,
        acc[acc_base],
        acc[acc_base + 1],
        acc[acc_base + 2],
        acc[acc_base + 3],
    )


def _ldmatrix_x4(dst, pointer, *, trans):
    if trans:
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            dst[0], dst[1], dst[2], dst[3], pointer
        )
    else:
        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(dst[0], dst[1], dst[2], dst[3], pointer)


def _ldmatrix_x1(pointer, *, trans):
    frag = K.local_scalar("uint32")
    if trans:
        K.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(frag, pointer)
    else:
        K.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(frag, pointer)
    return frag


def _stmatrix_x4(pointer, words):
    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(pointer, words[0], words[1], words[2], words[3])


def _tinv_row_ptr(arena, tinv_byte_base, d, tidx_in_group):
    row_linear = (d * 8 + tidx_in_group) * 64 + d * 8
    return arena.ptr_to([tinv_byte_base + _swizzle_lin_128b(row_linear) * 2])


def _invert_diagonal_8x8(arena, raw_byte_base, tinv_byte_base, d, tidx_in_group, io_dtype):
    """Gauss-Jordan inversion of one diagonal 8x8 block in place (IO SMEM)."""
    row_ptr_in = _tinv_row_ptr(arena, raw_byte_base, d, tidx_in_group)
    row_ptr = _tinv_row_ptr(arena, tinv_byte_base, d, tidx_in_group)
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], row_ptr_in)
    row = K.alloc_local((8,), "float32")
    for pair in range(4):
        _unpack_io_pair(words[pair], row[pair * 2], row[pair * 2 + 1], io_dtype)
    for i in range(8):
        K.assign(row[i], K.if_then_else(tidx_in_group == i, K.float32(1.0), row[i]))
    for src_row in range(7):
        row_scale = K.local_scalar("float32", init=-row[src_row])
        for i in range(src_row):
            shuffled = _shfl_idx_f32(row[i], src_row, 0b1100000011111)
            K.assign(
                row[i],
                K.if_then_else(tidx_in_group > src_row, row[i] + row_scale * shuffled, row[i]),
            )
        K.assign(row[src_row], K.if_then_else(tidx_in_group > src_row, row_scale, row[src_row]))
    for pair in range(4):
        K.assign(words[pair], _pack_io_pair(row[pair * 2], row[pair * 2 + 1], io_dtype))
    K.ptx.st.shared.v4.b32(row_ptr, words[0], words[1], words[2], words[3])


def _blockwise_8_to_16(arena, tinv_byte_base, raw_byte_base, d0, lane, io_dtype):
    """Off-diagonal correction 8x8 -> 16x16: C <- -(D^-1 C) A^-1."""
    lane_off = (lane % 8) * 64
    d_frag = _ldmatrix_x1(
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 8) * 64 + d0 + 8 + lane_off) * 2]),
        trans=False,
    )
    c_frag = _ldmatrix_x1(
        arena.ptr_to([raw_byte_base + _swizzle_lin_128b((d0 + 8) * 64 + d0 + lane_off) * 2]),
        trans=True,
    )
    c_regs = K.alloc_local((4,), "float32")
    for accum in range(4):
        K.assign(c_regs[accum], K.float32(0.0))
    _mma_m16n8k8(c_regs, 0, d_frag, d_frag, c_frag, io_dtype)
    a_pack = K.alloc_local((2,), "uint32")
    for pair in range(2):
        K.assign(a_pack[pair], _pack_io_pair(-c_regs[2 * pair], -c_regs[2 * pair + 1], io_dtype))
    ai_frag = _ldmatrix_x1(
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b(d0 * 64 + d0 + lane_off) * 2]), trans=True
    )
    o_regs = K.alloc_local((4,), "float32")
    for accum in range(4):
        K.assign(o_regs[accum], K.float32(0.0))
    _mma_m16n8k8(o_regs, 0, a_pack[0], a_pack[1], ai_frag, io_dtype)
    o_pack = _pack_io_pair(o_regs[0], o_regs[1], io_dtype)
    K.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 8) * 64 + d0 + lane_off) * 2]),
        o_pack,
    )


def _blockwise_16_to_32(arena, tinv_byte_base, raw_byte_base, d0, lane, io_dtype):
    """Off-diagonal correction 16x16 -> 32x32."""
    lane_off = (lane % 16) * 64 + (lane // 16) * 8
    d_frag = K.alloc_local((4,), "uint32")
    _ldmatrix_x4(
        d_frag,
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 16) * 64 + d0 + 16 + lane_off) * 2]),
        trans=False,
    )
    c_frag = K.alloc_local((4,), "uint32")
    _ldmatrix_x4(
        c_frag,
        arena.ptr_to([raw_byte_base + _swizzle_lin_128b((d0 + 16) * 64 + d0 + lane_off) * 2]),
        trans=True,
    )
    c_regs = K.alloc_local((8,), "float32")
    for accum in range(8):
        K.assign(c_regs[accum], K.float32(0.0))
    for n_frag in range(2):
        _mma_m16n8k16(
            c_regs, n_frag * 4, d_frag, c_frag[n_frag * 2], c_frag[n_frag * 2 + 1], io_dtype
        )
    a_pack = K.alloc_local((4,), "uint32")
    for pair in range(4):
        K.assign(a_pack[pair], _pack_io_pair(-c_regs[2 * pair], -c_regs[2 * pair + 1], io_dtype))
    ai_frag = K.alloc_local((4,), "uint32")
    _ldmatrix_x4(
        ai_frag,
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b(d0 * 64 + d0 + lane_off) * 2]),
        trans=True,
    )
    o_regs = K.alloc_local((8,), "float32")
    for accum in range(8):
        K.assign(o_regs[accum], K.float32(0.0))
    for n_frag in range(2):
        _mma_m16n8k16(
            o_regs, n_frag * 4, a_pack, ai_frag[n_frag * 2], ai_frag[n_frag * 2 + 1], io_dtype
        )
    o_pack = K.alloc_local((4,), "uint32")
    for pair in range(4):
        K.assign(o_pack[pair], _pack_io_pair(o_regs[2 * pair], o_regs[2 * pair + 1], io_dtype))
    _stmatrix_x4(
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 16) * 64 + d0 + lane_off) * 2]),
        o_pack,
    )


def _blockwise_32_to_64(arena, tinv_byte_base, raw_byte_base, band, lane, io_dtype, store_result):
    """Off-diagonal correction 32x32 -> 64x64 (two warps, one 16-row band each)."""
    lane_off = (lane % 16) * 64 + (lane // 16) * 8
    a_frags = K.alloc_local((8,), "uint32")
    for vs in range(2):
        chunk = K.alloc_local((4,), "uint32")
        _ldmatrix_x4(
            chunk,
            arena.ptr_to(
                [
                    tinv_byte_base
                    + _swizzle_lin_128b((32 + band * 16) * 64 + 32 + vs * 16 + lane_off) * 2
                ]
            ),
            trans=False,
        )
        for i in range(4):
            K.assign(a_frags[vs * 4 + i], chunk[i])
    b_frags = K.alloc_local((16,), "uint32")
    for vs in range(4):
        chunk = K.alloc_local((4,), "uint32")
        _ldmatrix_x4(
            chunk,
            arena.ptr_to(
                [
                    raw_byte_base
                    + _swizzle_lin_128b((32 + (vs // 2) * 16) * 64 + (vs % 2) * 16 + lane_off) * 2
                ]
            ),
            trans=True,
        )
        for i in range(4):
            K.assign(b_frags[vs * 4 + i], chunk[i])
    c_regs = K.alloc_local((16,), "float32")
    for accum in range(16):
        K.assign(c_regs[accum], K.float32(0.0))
    for k_step in range(2):
        a_view = [a_frags[k_step * 4 + i] for i in range(4)]
        for n_frag in range(4):
            _mma_m16n8k16(
                c_regs,
                n_frag * 4,
                a_view,
                b_frags[k_step * 8 + n_frag * 2],
                b_frags[k_step * 8 + n_frag * 2 + 1],
                io_dtype,
            )
    a_pack = K.alloc_local((8,), "uint32")
    for pair in range(8):
        K.assign(a_pack[pair], _pack_io_pair(-c_regs[2 * pair], -c_regs[2 * pair + 1], io_dtype))
    ai_frags = K.alloc_local((16,), "uint32")
    for vs in range(4):
        chunk = K.alloc_local((4,), "uint32")
        _ldmatrix_x4(
            chunk,
            arena.ptr_to(
                [
                    tinv_byte_base
                    + _swizzle_lin_128b(((vs // 2) * 16) * 64 + (vs % 2) * 16 + lane_off) * 2
                ]
            ),
            trans=True,
        )
        for i in range(4):
            K.assign(ai_frags[vs * 4 + i], chunk[i])
    o_regs = K.alloc_local((16,), "float32")
    for accum in range(16):
        K.assign(o_regs[accum], K.float32(0.0))
    for k_step in range(2):
        a_view = [a_pack[k_step * 4 + i] for i in range(4)]
        for n_frag in range(4):
            _mma_m16n8k16(
                o_regs,
                n_frag * 4,
                a_view,
                ai_frags[k_step * 8 + n_frag * 2],
                ai_frags[k_step * 8 + n_frag * 2 + 1],
                io_dtype,
            )
    o_pack = K.alloc_local((8,), "uint32")
    for pair in range(8):
        K.assign(o_pack[pair], _pack_io_pair(o_regs[2 * pair], o_regs[2 * pair + 1], io_dtype))
    K.ptx.bar.sync(K.uint32(3), K.uint32(64))
    with K.If(store_result), K.Then():
        _stmatrix_x4(
            arena.ptr_to(
                [tinv_byte_base + _swizzle_lin_128b((32 + band * 16) * 64 + lane_off) * 2]
            ),
            [o_pack[0], o_pack[1], o_pack[2], o_pack[3]],
        )
        _stmatrix_x4(
            arena.ptr_to(
                [tinv_byte_base + _swizzle_lin_128b((32 + band * 16) * 64 + 16 + lane_off) * 2]
            ),
            [o_pack[4], o_pack[5], o_pack[6], o_pack[7]],
        )


def _acc_operand(accumulate):
    if isinstance(accumulate, bool | int):
        return K.uint32(int(accumulate))
    return K.cast(accumulate, "uint32")


def _tcgen_mma_ss(
    dst,
    a_desc,
    b_desc,
    idesc,
    *,
    leader,
    m,
    n,
    k_extent,
    a_transpose=False,
    b_transpose=False,
    accumulate,
):
    def swizzle_bytes(inner_bytes):
        if inner_bytes % 128 == 0:
            return 128
        if inner_bytes % 64 == 0:
            return 64
        return 32

    swizzle_a = swizzle_bytes((m if a_transpose else k_extent) * 2)
    swizzle_b = swizzle_bytes((n if b_transpose else k_extent) * 2)
    intra_a = 16 * swizzle_a if a_transpose else 32
    intra_b = 16 * swizzle_b if b_transpose else 32
    subtile_a = k_extent * swizzle_a if a_transpose else swizzle_a * m
    subtile_b = k_extent * swizzle_b if b_transpose else swizzle_b * n
    steps_a = k_extent // 16 if a_transpose else (swizzle_a // 2) // 16
    steps_b = k_extent // 16 if b_transpose else (swizzle_b // 2) // 16
    for step in range(k_extent // 16):
        inc_a = (intra_a * (step % steps_a) + subtile_a * (step // steps_a)) >> 4
        inc_b = (intra_b * (step % steps_b) + subtile_b * (step // steps_b)) >> 4
        K.ptx[_MMA_F16](
            K.cast(dst, "uint32"),
            a_desc + K.uint64(inc_a),
            b_desc + K.uint64(inc_b),
            K.uint32(idesc),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.ptx.pred(_acc_operand(accumulate) if step == 0 else K.uint32(1)),
            pred=leader,
        )


def _tcgen_mma_ts(
    dst, tmem_a, b_desc, idesc, *, leader, n, k_extent, b_transpose=False, accumulate
):
    inner_bytes = (n if b_transpose else k_extent) * 2
    swizzle_b = 128 if inner_bytes % 128 == 0 else 64 if inner_bytes % 64 == 0 else 32
    intra_b = 16 * swizzle_b if b_transpose else 32
    subtile_b = k_extent * swizzle_b if b_transpose else swizzle_b * n
    steps_b = k_extent // 16 if b_transpose else (swizzle_b // 2) // 16
    for step in range(k_extent // 16):
        inc_b = (intra_b * (step % steps_b) + subtile_b * (step // steps_b)) >> 4
        K.ptx[_MMA_F16](
            K.cast(dst, "uint32"),
            K.cast(tmem_a + step * 8, "uint32"),
            b_desc + K.uint64(inc_b),
            K.uint32(idesc),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.ptx.pred(_acc_operand(accumulate) if step == 0 else K.uint32(1)),
            pred=leader,
        )


def _make_prologue(*, run_order, order_generate, dynamic_scheduler, n_heads_out, cu_dtype):
    cu_t = K.i64 if cu_dtype == "int64" else K.i32

    @K.kernel(warps=32, arch="sm_100a", grid=1)
    def prologue(
        base_q: K.gptr[K.i64],
        base_k: K.gptr[K.i64],
        base_v: K.gptr[K.i64],
        base_do: K.gptr[K.i64],
        base_checkpoint: K.gptr[K.i64],
        base_dq: K.gptr[K.i64],
        base_dk: K.gptr[K.i64],
        base_dv: K.gptr[K.i64],
        descriptor_workspace: K.gptr[K.i64],
        cu_seqlens: K.gptr[cu_t],
        q: K.gptr[K.u8],
        k: K.gptr[K.u8],
        v: K.gptr[K.u8],
        do: K.gptr[K.u8],
        checkpoint: K.gptr[K.u8],
        dq: K.gptr[K.u8],
        dk: K.gptr[K.u8],
        dv: K.gptr[K.u8],
        work_item_staging: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        work_items: K.gptr[K.i32],
        scheduler_all: K.gptr[K.i32],
        n_batch: K.i32,
        q_row_stride_bytes: K.i32,
        k_row_stride_bytes: K.i32,
        v_row_stride_bytes: K.i32,
        do_row_stride_bytes: K.i32,
        checkpoint_row_stride_bytes: K.i32,
        dq_row_stride_bytes: K.i32,
        dk_row_stride_bytes: K.i32,
        dv_row_stride_bytes: K.i32,
        checkpoint_every_n: K.i32,
    ):
        thread = K.thread_id()
        warp = K.warp_id()
        map_payload = K.alloc_local((16,), "uint64")
        preload_maps = (base_q, base_k, base_v, base_do, base_checkpoint, base_dq, base_dk, base_dv)
        for array_index, base_map in enumerate(preload_maps):
            with K.If(warp == array_index), K.Then():
                with K.If(_elected()), K.Then():
                    _load_tensormap(map_payload, base_map)

        if run_order:
            order_arena = K.alloc_buffer((32_776,), K.u8, scope="shared.dyn", align=16)
            with K.If(thread == 0), K.Then():
                for scheduler_word in range(4):
                    K.ptx.st.global_.s32(scheduler_all.ptr_to([scheduler_word]), K.int32(0))

            item_count = K.local_scalar("int32")
            if order_generate:
                K.assign(item_count, n_batch * n_heads_out)
                with K.If(thread == 0), K.Then():
                    K.ptx.st.global_.s32(work_count.ptr_to([0]), item_count)
            else:
                K.ptx.ld.global_.s32(item_count, work_count.ptr_to([0]))

            def write_work_item(destination, source):
                if order_generate:
                    batch = source // n_heads_out
                    head = source - batch * n_heads_out
                    begin = _load_cu(cu_seqlens, batch, cu_dtype)
                    end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                    chunks = (end - begin + _BT - 1) // _BT
                    K.ptx.st.global_.v4.b32(
                        work_items.ptr_to([destination * 8]), batch, head, K.int32(0), chunks
                    )
                    K.ptx.st.global_.v4.b32(
                        work_items.ptr_to([destination * 8 + 4]), K.int32(0), chunks, begin, end
                    )
                else:
                    for field in range(8):
                        value = K.local_scalar("int32")
                        K.ptx.ld.global_.s32(value, work_item_staging.ptr_to([source * 8 + field]))
                        K.ptx.st.global_.s32(work_items.ptr_to([destination * 8 + field]), value)

            with K.If(item_count > 4096), K.Then():
                item = K.local_scalar("int32", init=thread)
                with K.While(item < item_count):
                    write_work_item(item, item)
                    K.assign(item, item + 1024)
            with K.If(item_count <= 4096), K.Then():
                with K.If(thread == 0), K.Then():
                    K.ptx.st.shared.v2.u32(
                        order_arena.ptr_to([32_768]), K.uint32(2_147_483_647), K.uint32(0x80000000)
                    )
                padded_count = K.local_scalar("int32", init=K.int32(1))
                with K.While(padded_count < item_count):
                    K.assign(padded_count, padded_count * 2)
                K.cuda.cta_sync()
                local_min = K.local_scalar("int32", init=K.int32(2_147_483_647))
                local_max = K.local_scalar("int32", init=K.int32(-2_147_483_648))
                for element in range(4):
                    item = thread + element * 1024
                    with K.If(item < padded_count), K.Then():
                        key = K.local_scalar("int32", init=K.int32(-2_147_483_648))
                        with K.If(item < item_count), K.Then():
                            if order_generate:
                                batch = item // n_heads_out
                                begin = _load_cu(cu_seqlens, batch, cu_dtype)
                                end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                                K.assign(key, (end - begin + _BT - 1) // _BT)
                            else:
                                cstart = K.local_scalar("int32")
                                cend = K.local_scalar("int32")
                                K.ptx.ld.global_.s32(
                                    cstart, work_item_staging.ptr_to([item * 8 + 4])
                                )
                                K.ptx.ld.global_.s32(cend, work_item_staging.ptr_to([item * 8 + 5]))
                                K.assign(key, cend - cstart)
                        K.ptx.st.shared.s32(order_arena.ptr_to([item * 4]), key)
                        K.ptx.st.shared.s32(order_arena.ptr_to([16_384 + item * 4]), item)
                        with K.If(item < item_count), K.Then():
                            K.assign(local_min, K.if_then_else(local_min < key, local_min, key))
                            K.assign(local_max, K.if_then_else(local_max > key, local_max, key))
                old = K.local_scalar("int32")
                K.ptx["atom.shared::cta.min.s32"](old, order_arena.ptr_to([32_768]), local_min)
                K.ptx["atom.shared::cta.max.s32"](old, order_arena.ptr_to([32_772]), local_max)
                K.cuda.cta_sync()
                spread_min = K.local_scalar("int32")
                spread_max = K.local_scalar("int32")
                K.ptx.ld.shared.s32(spread_min, order_arena.ptr_to([32_768]))
                K.ptx.ld.shared.s32(spread_max, order_arena.ptr_to([32_772]))
                with K.If(spread_min == spread_max), K.Then():
                    destination = K.local_scalar("int32", init=thread)
                    with K.While(destination < item_count):
                        write_work_item(destination, destination)
                        K.assign(destination, destination + 1024)
                with K.If(spread_min != spread_max), K.Then():
                    network = K.local_scalar("int32", init=K.int32(2))
                    with K.While(network <= padded_count):
                        distance = K.local_scalar("int32", init=network // 2)
                        with K.While(distance > 0):
                            for element in range(4):
                                item = thread + element * 1024
                                partner = K.bitwise_xor(item, distance)
                                with K.If(K.And(item < padded_count, partner > item)), K.Then():
                                    key_i = K.local_scalar("int32")
                                    key_j = K.local_scalar("int32")
                                    K.ptx.ld.shared.s32(key_i, order_arena.ptr_to([item * 4]))
                                    K.ptx.ld.shared.s32(key_j, order_arena.ptr_to([partner * 4]))
                                    ascending = ((item // network) % 2) == 0
                                    swap = K.Or(
                                        K.And(ascending, key_i < key_j),
                                        K.And(K.Not(ascending), key_i > key_j),
                                    )
                                    with K.If(swap), K.Then():
                                        index_i = K.local_scalar("int32")
                                        index_j = K.local_scalar("int32")
                                        K.ptx.ld.shared.s32(
                                            index_i, order_arena.ptr_to([16_384 + item * 4])
                                        )
                                        K.ptx.ld.shared.s32(
                                            index_j, order_arena.ptr_to([16_384 + partner * 4])
                                        )
                                        K.ptx.st.shared.s32(order_arena.ptr_to([item * 4]), key_j)
                                        K.ptx.st.shared.s32(
                                            order_arena.ptr_to([partner * 4]), key_i
                                        )
                                        K.ptx.st.shared.s32(
                                            order_arena.ptr_to([16_384 + item * 4]), index_j
                                        )
                                        K.ptx.st.shared.s32(
                                            order_arena.ptr_to([16_384 + partner * 4]), index_i
                                        )
                            K.cuda.cta_sync()
                            K.assign(distance, distance // 2)
                        K.assign(network, network * 2)
                    for element in range(4):
                        destination = thread + element * 1024
                        with K.If(destination < item_count), K.Then():
                            source = K.local_scalar("int32")
                            K.ptx.ld.shared.s32(
                                source, order_arena.ptr_to([16_384 + destination * 4])
                            )
                            write_work_item(destination, source)

        bases = (q, k, v, do, checkpoint, dq, dk, dv)
        strides = (
            q_row_stride_bytes,
            k_row_stride_bytes,
            v_row_stride_bytes,
            do_row_stride_bytes,
            checkpoint_row_stride_bytes,
            dq_row_stride_bytes,
            dk_row_stride_bytes,
            dv_row_stride_bytes,
        )
        for array_index in range(8):
            with K.If(warp == array_index), K.Then():
                with K.If(_elected()), K.Then():
                    checkpoint_prefix = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(n_batch) as batch:
                        begin = _load_cu(cu_seqlens, batch, cu_dtype)
                        end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                        length = end - begin
                        slot = _descriptor_slot(descriptor_workspace, n_batch, array_index, batch)
                        _store_tensormap(slot, map_payload)
                        if array_index == 4:
                            count = K.local_scalar("int32", init=K.int32(0))
                            with K.If(length > 0), K.Then():
                                K.assign(count, (length - 1) // checkpoint_every_n + 1)
                            _replace_tensormap_address(
                                slot,
                                checkpoint.ptr_to(
                                    [
                                        K.cast(checkpoint_prefix, "int64")
                                        * checkpoint_row_stride_bytes
                                    ]
                                ),
                            )
                            _replace_tensormap_dim(slot, 2, count)
                            K.assign(checkpoint_prefix, checkpoint_prefix + count)
                        else:
                            _replace_tensormap_address(
                                slot,
                                bases[array_index].ptr_to(
                                    [K.cast(begin, "int64") * strides[array_index]]
                                ),
                            )
                            _replace_tensormap_dim(slot, 2, length)
                    K.ptx.fence.proxy.tensormap__generic.release.gpu()

    return prologue


def _make_main(
    *,
    num_sms,
    io_dtype,
    cu_dtype,
    use_initial_state,
    use_dstate_in,
    use_dstate0,
    log_gate,
    safe_gate,
    beta_sigmoid,
    dynamic_scheduler,
    q_ratio,
    k_ratio,
    v_ratio,
    n_heads_out,
):
    io_t = K.f16 if io_dtype == "float16" else K.bf16
    cu_t = K.i64 if cu_dtype == "int64" else K.i32
    beta_t = io_t if beta_sigmoid else K.f32
    first_state_chunk = 0 if use_initial_state else 1
    idesc_delta = 1152 if io_dtype == "float16" else 0
    idesc_m64_n64 = _IDESC_M64_N64_BF16 - idesc_delta
    idesc_m128_n64 = _IDESC_M128_N64_BF16 - idesc_delta
    idesc_m128_n64_a = _IDESC_M128_N64_A_BF16 - idesc_delta
    idesc_m128_n64_b = _IDESC_M128_N64_B_BF16 - idesc_delta
    idesc_m128_n64_ab = _IDESC_M128_N64_AB_BF16 - idesc_delta
    idesc_m128_n128_b = _IDESC_M128_N128_B_BF16 - idesc_delta

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=num_sms)
    def main(
        descriptor_workspace: K.gptr[K.i64],
        n_desc: K.i32,
        gate: K.gptr[K.f32],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        beta: K.gptr[beta_t],
        dgate: K.gptr[K.f32],
        dbeta: K.gptr[beta_t],
        cu_seqlens: K.gptr[cu_t],
        dstate0: K.gptr[K.f32],
        dstate_in: K.gptr[K.f32],
        work_items: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        scale: K.f32,
    ):
        arena = K.alloc_buffer((_ARENA_BYTES,), K.u8, scope="shared.dyn", align=1024)
        K.smem_pool(base=arena)
        thread = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()
        with K.attr({"tirx.required_block_size": 1}):
            _cluster_x, _cluster_y, _cluster_z = K.cta_id_in_cluster([1, 1, 1])

        protocol = (
            (_BAR_Q_READY, 1, 1),
            (_BAR_Q_MMA_DONE, 1, 1),
            (_BAR_Q_CG1_DONE, 1, 128),
            (_BAR_K_READY, 2, 1),
            (_BAR_K_MMA_DONE, 2, 1),
            (_BAR_K_CG0_DONE, 2, 128),
            (_BAR_V_READY, 1, 1),
            (_BAR_V_MMA_DONE, 1, 1),
            (_BAR_DO_READY, 1, 1),
            (_BAR_DO_MMA_DONE, 1, 1),
            (_BAR_STATE_READY, 1, 1),
            (_BAR_STATE_MMA_DONE, 1, 1),
            (_BAR_GATE_READY, 2, 32),
            (_BAR_GATE_DONE, 2, 256),
            (_BAR_BETA_READY, 2, 32),
            (_BAR_BETA_DONE, 2, 256),
            (_BAR_DSTATE_ACC_READY, 1, 1),
            (_BAR_DSTATE_SCALE_DONE, 1, 128),
            (_BAR_DU_SCALE_READY, 1, 1),
            (_BAR_DU_SCALE_DONE, 1, 128),
            (_BAR_DU_TOTAL_READY, 1, 1),
            (_BAR_DK_SCALE_READY, 1, 1),
            (_BAR_DK_SCALE_DONE, 1, 128),
            (_BAR_DK_ATTN_READY, 1, 1),
            (_BAR_DK_ATTN_DONE, 1, 128),
            (_BAR_DK_TOTAL_READY, 1, 1),
            (_BAR_DK_TOTAL_DONE, 1, 128),
            (_BAR_DQ_SCALE_READY, 1, 1),
            (_BAR_DQ_SCALE_DONE, 1, 128),
            (_BAR_DQ_TOTAL_READY, 1, 1),
            (_BAR_DQ_TOTAL_DONE, 1, 128),
            (_BAR_KK_READY, 1, 1),
            (_BAR_KK_DONE, 1, 128),
            (_BAR_A_ACC_READY, 1, 1),
            (_BAR_K_STATE_READY, 1, 1),
            (_BAR_U_ACC_READY, 1, 1),
            (_BAR_DY_ACC_READY, 1, 1),
            (_BAR_DA_ACC_READY, 1, 1),
            (_BAR_DM_ACC_READY, 1, 1),
            (_BAR_DM_ACC_DONE, 1, 128),
            (_BAR_DK_STATE_READY, 1, 1),
            (_BAR_DSTATE_INP_READY, 1, 128),
            (_BAR_DSTATE_INP_DONE, 1, 1),
            (_BAR_DO_PRIME_READY, 1, 128),
            (_BAR_DU_INP_READY, 1, 128),
            (_BAR_DYP_INP_READY, 1, 128),
            (_BAR_Y_READY, 1, 128),
            (_BAR_TINV_READY, 1, 128),
            (_BAR_A_READY, 1, 128),
            (_BAR_A_DONE, 1, 1),
            (_BAR_U_READY, 1, 128),
            (_BAR_DSTATE_SMEM_READY, 1, 128),
            (_BAR_STATE_DOT_DONE, 1, 128),
            (_BAR_DA_READY, 1, 128),
            (_BAR_DBETA_CG1_READY, 1, 128),
            (_BAR_DGATE_CG1_READY, 1, 128),
            (_BAR_DQ_STG_READY, 1, 128),
            (_BAR_DQ_STG_DONE, 1, 32),
            (_BAR_DK_STG_READY, 1, 128),
            (_BAR_DK_STG_DONE, 1, 32),
            (_BAR_DV_STG_READY, 1, 128),
            (_BAR_DV_STG_DONE, 1, 32),
            (_BAR_SDV_DONE, 1, 1),
            (_BAR_TMEM_DONE, 1, 256),
            (_BAR_SCHED_READY, 2, 1),
            (_BAR_SCHED_DONE, 2, 11),
        )
        with K.If(warp == 9), K.Then():
            with K.If(_elected()), K.Then():
                for offset, stages, count in protocol:
                    for stage in range(stages):
                        K.ptx.mbarrier.init.shared.b64(
                            _barrier_ptr(arena, offset, stage), K.uint32(count)
                        )
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        total_tiles = K.local_scalar("int32")
        K.ptx.ld.global_.s32(total_tiles, work_count.ptr_to([0]))
        roles = K.specialize()
        cg0 = roles.role("cg0", warps=range(0, 4), regs=192)
        cg1 = roles.role("cg1", warps=range(4, 8), regs=248)
        mma = roles.role("mma", warps=[8], regs=64)
        tma = roles.role("tma", warps=[9], regs=64)
        gate_beta = roles.role("gate_beta", warps=[10], regs=64)
        epilogue = roles.role("epilogue", warps=[11], regs=64)

        def sched_consume(state, tile):
            if dynamic_scheduler:
                _wait_barrier(arena, _BAR_SCHED_READY, state.stage, state.phase)
                K.ptx.ld.shared.s32(
                    tile, arena.ptr_to([_SCHED_BASE + K.cast(state.stage, "int32") * 4])
                )
                with K.If(_elected()), K.Then():
                    _arrive_barrier(arena, _BAR_SCHED_DONE, state.stage)
                state.advance()
            else:
                K.assign(tile, tile + num_sms)

        def sched_produce(state, tile):
            if dynamic_scheduler:
                _wait_barrier(arena, _BAR_SCHED_DONE, state.stage, state.phase)
                with K.If(_elected()), K.Then():
                    ticket = K.local_scalar("uint32")
                    K.ptx.atom.global_.add.u32(ticket, scheduler.ptr_to([0]), K.uint32(1))
                    K.ptx.st.shared.u32(
                        arena.ptr_to([_SCHED_BASE + K.cast(state.stage, "int32") * 4]),
                        K.uint32(num_sms) + ticket,
                    )
                K.cuda.warp_sync()
                K.ptx.ld.shared.s32(
                    tile, arena.ptr_to([_SCHED_BASE + K.cast(state.stage, "int32") * 4])
                )
                with K.If(_elected()), K.Then():
                    _arrive_barrier(arena, _BAR_SCHED_READY, state.stage)
                state.advance()
            else:
                K.assign(tile, tile + num_sms)

        # Compute-group bodies are inserted below after their exact arithmetic
        # fragments are translated. Keeping their role scopes here makes the
        # service pipeline independently lowerable while that work proceeds.
        with cg0:
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_TMEM_MAILBOX]))
            warp_in_group = K.warp_id_in_role()
            cg0_thread = warp_in_group * 32 + lane
            tmem_row = K.shift_left(warp_in_group * 32, K.int32(16))
            store_row = warp_in_group * 16 + lane % 16
            store_col = (lane // 16) * 8
            frag_row = cg0_thread % 8 + ((cg0_thread // 16) % 2) * 8
            frag_col = ((cg0_thread // 8) % 2) * 8 + ((cg0_thread // 32) % 2) * 32
            frag_segment = cg0_thread // 64

            gate_cursor = K.PipelineState(2, phase=0)
            beta_cursor = K.PipelineState(2, phase=0)
            a_cursor = K.PipelineState(1, phase=1)
            tinv_cursor = K.PipelineState(1, phase=1)
            da_cursor = K.PipelineState(1, phase=0)
            dm_cursor = K.PipelineState(1, phase=0)
            dbeta_cursor = K.PipelineState(1, phase=0)
            k_cursor = K.PipelineState(2, phase=0)
            dgate_cursor = K.PipelineState(1, phase=0)
            kk_cursor = K.PipelineState(1, phase=0)
            a_ready_cursor = K.PipelineState(1, phase=0)
            dk_scale_cursor = K.PipelineState(1, phase=0)
            dq_scale_cursor = K.PipelineState(1, phase=0)
            dstate_smem_cursor = K.PipelineState(1, phase=0)
            dk_attn_cursor = K.PipelineState(1, phase=0)
            sched_state = K.PipelineState(2, phase=0)

            def reduce_scatter_16(values):
                stage8 = K.alloc_local((8,), "float32")
                for output, source in enumerate((0, 1, 4, 5, 8, 9, 12, 13)):
                    high = ((lane // 4) % 2) == 1
                    sent = K.if_then_else(high, values[source], values[source + 2])
                    kept = K.if_then_else(high, values[source + 2], values[source])
                    K.assign(stage8[output], kept + _shfl_bfly_f32(sent, 4))
                stage4 = K.alloc_local((4,), "float32")
                for output, source in enumerate((0, 1, 4, 5)):
                    high = ((lane // 8) % 2) == 1
                    sent = K.if_then_else(high, stage8[source], stage8[source + 2])
                    kept = K.if_then_else(high, stage8[source + 2], stage8[source])
                    K.assign(stage4[output], kept + _shfl_bfly_f32(sent, 8))
                stage2 = K.alloc_local((2,), "float32")
                for output in range(2):
                    high = ((lane // 16) % 2) == 1
                    sent = K.if_then_else(high, stage4[output], stage4[output + 2])
                    kept = K.if_then_else(high, stage4[output + 2], stage4[output])
                    K.assign(stage2[output], kept + _shfl_bfly_f32(sent, 16))
                return stage2[0], stage2[1]

            for word in range(16):
                K.ptx.st.shared.b32(
                    arena.ptr_to([_TINV_BASE + (cg0_thread + word * 128) * 4]), K.uint32(0)
                )

            tile = K.local_scalar("int32", init=K.cta_id())
            dstate_in0 = 1 if use_dstate_in else 0
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                wstart = item[2]
                cend = item[5]
                num_item_chunks = cend - wstart
                with K.serial(num_item_chunks, unroll=False) as reverse_index:
                    chunk = cend - 1 - reverse_index
                    have_dstate = K.bool(True) if use_dstate_in else reverse_index > 0

                    gate_stage = K.local_scalar("int32", init=gate_cursor.stage)
                    _wait_barrier(arena, _BAR_GATE_READY, gate_stage, gate_cursor.phase)
                    gate_cursor.advance()

                    row_cs = K.alloc_local((2,), "float32")
                    for row_part in range(2):
                        row = warp_in_group * 16 + lane // 4 + row_part * 8
                        K.ptx.ld.shared.f32(
                            row_cs[row_part],
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + row * 4]),
                        )
                    col_cs = K.alloc_local((16,), "float32")
                    for group in range(8):
                        for pair in range(2):
                            column = (lane % 4) * 2 + group * 8 + pair
                            K.ptx.ld.shared.f32(
                                col_cs[group * 2 + pair],
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + column * 4]),
                            )

                    decay = K.alloc_local((32,), "float32")
                    strict_decay = K.alloc_local((32,), "float32")
                    for value in range(32):
                        row = warp_in_group * 16 + lane // 4 + ((value // 2) % 2) * 8
                        column = (lane % 4) * 2 + (value // 4) * 8 + value % 2
                        candidate = _exp2(
                            row_cs[(value // 2) % 2] - col_cs[(value // 4) * 2 + value % 2]
                        )
                        K.assign(
                            decay[value], K.if_then_else(row >= column, candidate, _opaque_zero())
                        )
                        K.assign(
                            strict_decay[value],
                            K.if_then_else(row == column, _opaque_zero(), decay[value]),
                        )

                    last_cs = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(
                        last_cs, arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + 63 * 4])
                    )
                    decay_scale = K.alloc_local((32,), "float32")
                    for value in range(32):
                        column_part = (value // 4) * 2 + value % 2
                        K.assign(decay_scale[value], _exp2(last_cs - col_cs[column_part]))
                    cumprod_total = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(
                        cumprod_total, arena.ptr_to([_CUMPROD_BASE + gate_stage * 256 + 63 * 4])
                    )

                    beta_stage = K.local_scalar("int32", init=beta_cursor.stage)
                    _wait_barrier(arena, _BAR_BETA_READY, beta_stage, beta_cursor.phase)
                    beta_cursor.advance()
                    beta_rows = K.alloc_local((32,), "float32")
                    for value in range(32):
                        row = warp_in_group * 16 + lane // 4 + ((value // 2) % 2) * 8
                        K.ptx.ld.shared.f32(
                            beta_rows[value],
                            arena.ptr_to([_BETA_BASE + beta_stage * 256 + row * 4]),
                        )

                    tinv_stage = K.local_scalar("int32", init=tinv_cursor.stage)
                    tinv_cursor.advance()
                    _wait_barrier(arena, _BAR_KK_READY, 0, kk_cursor.phase)
                    kk_cursor.advance()
                    kk_values = _tcgen_load_16x256x8(tmem_base + _TM_ACC0 + tmem_row)
                    kk_pack = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        product0, product1 = _fmul2(
                            kk_values[pair * 2],
                            kk_values[pair * 2 + 1],
                            strict_decay[pair * 2],
                            strict_decay[pair * 2 + 1],
                        )
                        value0, value1 = _fmul2(
                            product0, product1, beta_rows[pair * 2], beta_rows[pair * 2 + 1]
                        )
                        K.assign(kk_pack[pair], _pack_io_pair(value0, value1, io_dtype))
                    # A prior reverse iteration used this scratch for the
                    # dgate reduction.  Delay only the next overwrite until
                    # every CG0 warp has completed that reduction.
                    with K.If(reverse_index > 0), K.Then():
                        K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    for fragment in range(4):
                        _stmatrix_x4(
                            arena.ptr_to(
                                [
                                    _KK_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            [
                                kk_pack[fragment * 4],
                                kk_pack[fragment * 4 + 1],
                                kk_pack[fragment * 4 + 2],
                                kk_pack[fragment * 4 + 3],
                            ],
                        )
                    with K.If(reverse_index < cend - first_state_chunk), K.Then():
                        _arrive_barrier(arena, _BAR_KK_DONE)

                    a_stage = K.local_scalar("int32", init=a_cursor.stage)
                    a_phase = K.local_scalar("int32", init=a_cursor.phase)
                    a_cursor.advance()
                    _wait_barrier(arena, _BAR_A_ACC_READY, 0, a_ready_cursor.phase)
                    a_ready_cursor.advance()
                    a_values = _tcgen_load_16x256x8(tmem_base + _TM_ACC1 + tmem_row)
                    a_pack = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        product0, product1 = _fmul2(
                            a_values[pair * 2],
                            a_values[pair * 2 + 1],
                            decay[pair * 2],
                            decay[pair * 2 + 1],
                        )
                        value0, value1 = _fmul2(product0, product1, scale, scale)
                        K.assign(a_pack[pair], _pack_io_pair(value0, value1, io_dtype))
                    _wait_barrier(arena, _BAR_A_DONE, a_stage, a_phase)
                    for fragment in range(4):
                        _stmatrix_x4(
                            arena.ptr_to(
                                [
                                    _A_DA_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            [
                                a_pack[fragment * 4],
                                a_pack[fragment * 4 + 1],
                                a_pack[fragment * 4 + 2],
                                a_pack[fragment * 4 + 3],
                            ],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_A_READY, a_stage)

                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    with K.If(warp_in_group < 2), K.Then():
                        _invert_diagonal_8x8(
                            arena, _KK_BASE, _TINV_BASE, cg0_thread // 8, cg0_thread % 8, io_dtype
                        )
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    _blockwise_8_to_16(
                        arena, _TINV_BASE, _KK_BASE, warp_in_group * 16, lane, io_dtype
                    )
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    with K.If(warp_in_group < 2), K.Then():
                        _blockwise_16_to_32(
                            arena, _TINV_BASE, _KK_BASE, warp_in_group * 32, lane, io_dtype
                        )
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    with K.If(warp_in_group < 2), K.Then():
                        _blockwise_32_to_64(
                            arena, _TINV_BASE, _KK_BASE, warp_in_group, lane, io_dtype, True
                        )
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))

                    beta_columns = K.alloc_local((32,), "float32")
                    for value in range(32):
                        column = (lane % 4) * 2 + (value // 4) * 8 + value % 2
                        K.ptx.ld.shared.f32(
                            beta_columns[value],
                            arena.ptr_to([_BETA_BASE + beta_stage * 256 + column * 4]),
                        )
                    tinv_words = K.alloc_local((16,), "uint32")
                    for fragment in range(4):
                        words = K.alloc_local((4,), "uint32")
                        _ldmatrix_x4(
                            words,
                            arena.ptr_to(
                                [
                                    _TINV_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            trans=False,
                        )
                        for word in range(4):
                            K.assign(tinv_words[fragment * 4 + word], words[word])
                    tinv_pack = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        value0 = K.local_scalar("float32")
                        value1 = K.local_scalar("float32")
                        _unpack_io_pair(tinv_words[pair], value0, value1, io_dtype)
                        scaled0, scaled1 = _fmul2(
                            value0, value1, beta_columns[pair * 2], beta_columns[pair * 2 + 1]
                        )
                        K.assign(tinv_pack[pair], _pack_io_pair(scaled0, scaled1, io_dtype))
                    for fragment in range(4):
                        _stmatrix_x4(
                            arena.ptr_to(
                                [
                                    _TINV_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            [
                                tinv_pack[fragment * 4],
                                tinv_pack[fragment * 4 + 1],
                                tinv_pack[fragment * 4 + 2],
                                tinv_pack[fragment * 4 + 3],
                            ],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_TINV_READY, tinv_stage)

                    with K.If(chunk >= first_state_chunk), K.Then():
                        cumprod_values = K.alloc_local((32,), "float32")
                        for value in range(32):
                            column = (lane % 4) * 2 + (value // 4) * 8 + value % 2
                            K.ptx.ld.shared.f32(
                                cumprod_values[value],
                                arena.ptr_to([_CUMPROD_BASE + gate_stage * 256 + column * 4]),
                            )
                        _wait_barrier(arena, _BAR_DQ_SCALE_READY, 0, dq_scale_cursor.phase)
                        dq_scale_cursor.advance()
                        for sub in range(2):
                            address = (
                                tmem_base
                                + _TM_DSTATE_INP
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                            values = _tcgen_load_16x256x8(address)
                            scaled = K.alloc_local((32,), "float32")
                            for pair in range(16):
                                product0, product1 = _fmul2(
                                    values[pair * 2],
                                    values[pair * 2 + 1],
                                    cumprod_values[pair * 2],
                                    cumprod_values[pair * 2 + 1],
                                )
                                value0, value1 = _fmul2(product0, product1, scale, scale)
                                K.assign(scaled[pair * 2], value0)
                                K.assign(scaled[pair * 2 + 1], value1)
                            _tcgen_store_16x256x8(address, scaled)
                        _tcgen_wait_store()
                        _arrive_barrier(arena, _BAR_DQ_SCALE_DONE)

                    with K.If(have_dstate), K.Then():
                        _wait_barrier(arena, _BAR_DSTATE_SMEM_READY, 0, dstate_smem_cursor.phase)
                        dstate_smem_cursor.advance()
                    state_dot_lo = K.alloc_local((4,), "float32")
                    state_dot_hi = K.alloc_local((4,), "float32")
                    for word in range(4):
                        K.assign(state_dot_lo[word], _opaque_zero())
                        K.assign(state_dot_hi[word], _opaque_zero())
                    for octet in range(16):
                        dstate_words = K.alloc_local((4,), "uint32")
                        state_words = K.alloc_local((4,), "uint32")
                        K.ptx.ld.shared.v4.b32(
                            dstate_words[0],
                            dstate_words[1],
                            dstate_words[2],
                            dstate_words[3],
                            arena.ptr_to([_state_byte(_DSTATE_BASE, cg0_thread, octet * 8)]),
                        )
                        K.ptx.ld.shared.v4.b32(
                            state_words[0],
                            state_words[1],
                            state_words[2],
                            state_words[3],
                            arena.ptr_to([_state_byte(_STATE_BASE, cg0_thread, octet * 8)]),
                        )
                        for word in range(4):
                            dstate_value0 = K.local_scalar("float32")
                            dstate_value1 = K.local_scalar("float32")
                            state0 = K.local_scalar("float32")
                            state1 = K.local_scalar("float32")
                            _unpack_io_pair(
                                dstate_words[word], dstate_value0, dstate_value1, io_dtype
                            )
                            _unpack_io_pair(state_words[word], state0, state1, io_dtype)
                            next0, next1 = _ffma2(
                                dstate_value0,
                                dstate_value1,
                                state0,
                                state1,
                                state_dot_lo[word],
                                state_dot_hi[word],
                            )
                            K.assign(state_dot_lo[word], next0)
                            K.assign(state_dot_hi[word], next1)
                    state_dot = K.local_scalar(
                        "float32",
                        init=(
                            (state_dot_lo[0] + state_dot_lo[1])
                            + (state_dot_lo[2] + state_dot_lo[3])
                            + (state_dot_hi[0] + state_dot_hi[1])
                            + (state_dot_hi[2] + state_dot_hi[3])
                        ),
                    )
                    for distance in (1, 2, 4, 8, 16):
                        K.assign(state_dot, state_dot + _shfl_bfly_f32(state_dot, distance))
                    _arrive_barrier(arena, _BAR_STATE_DOT_DONE)

                    k_stage = K.local_scalar("int32", init=k_cursor.stage)
                    k_cursor.advance()
                    k_dot = K.local_scalar("float32", init=K.float32(0.0))
                    with K.If(have_dstate), K.Then():
                        _wait_barrier(arena, _BAR_DK_SCALE_READY, 0, dk_scale_cursor.phase)
                        dk_scale_cursor.advance()
                        k_dot_lo = K.alloc_local((4,), "float32")
                        k_dot_hi = K.alloc_local((4,), "float32")
                        for word in range(4):
                            K.assign(k_dot_lo[word], _opaque_zero())
                            K.assign(k_dot_hi[word], _opaque_zero())
                        for sub in range(2):
                            address = (
                                tmem_base
                                + _TM_DVDK
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                            values = _tcgen_load_16x256x8(address)
                            scaled = K.alloc_local((32,), "float32")
                            for pair in range(16):
                                value0, value1 = _fmul2(
                                    values[pair * 2],
                                    values[pair * 2 + 1],
                                    decay_scale[pair * 2],
                                    decay_scale[pair * 2 + 1],
                                )
                                K.assign(scaled[pair * 2], value0)
                                K.assign(scaled[pair * 2 + 1], value1)
                            _tcgen_store_16x256x8(address, scaled)
                            for matrix_row in range(4):
                                k_words = K.alloc_local((4,), "uint32")
                                row = frag_row + matrix_row * 16
                                channel = frag_segment * 64 + frag_col + sub * 16
                                _ldmatrix_x4(
                                    k_words,
                                    arena.ptr_to([_io_byte(_K_BASE, k_stage, row, channel)]),
                                    trans=True,
                                )
                                for word in range(4):
                                    k0 = K.local_scalar("float32")
                                    k1 = K.local_scalar("float32")
                                    _unpack_io_pair(k_words[word], k0, k1, io_dtype)
                                    index = matrix_row * 8 + word * 2
                                    next0, next1 = _ffma2(
                                        scaled[index],
                                        scaled[index + 1],
                                        k0,
                                        k1,
                                        k_dot_lo[word],
                                        k_dot_hi[word],
                                    )
                                    K.assign(k_dot_lo[word], next0)
                                    K.assign(k_dot_hi[word], next1)
                        _tcgen_wait_store()
                        _arrive_barrier(arena, _BAR_DK_SCALE_DONE)
                        K.assign(
                            k_dot,
                            (k_dot_lo[0] + k_dot_lo[1])
                            + (k_dot_lo[2] + k_dot_lo[3])
                            + (k_dot_hi[0] + k_dot_hi[1])
                            + (k_dot_hi[2] + k_dot_hi[3]),
                        )
                        for distance in (1, 2, 4, 8, 16):
                            K.assign(k_dot, k_dot + _shfl_bfly_f32(k_dot, distance))

                    _wait_barrier(arena, _BAR_DA_ACC_READY, 0, da_cursor.phase)
                    da_cursor.advance()
                    da_values = _tcgen_load_16x256x8(tmem_base + _TM_ACC1 + tmem_row)
                    da_pack = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        product0, product1 = _fmul2(
                            da_values[pair * 2],
                            da_values[pair * 2 + 1],
                            decay[pair * 2],
                            decay[pair * 2 + 1],
                        )
                        value0, value1 = _fmul2(product0, product1, scale, scale)
                        K.assign(da_pack[pair], _pack_io_pair(value0, value1, io_dtype))
                    for fragment in range(4):
                        _stmatrix_x4(
                            arena.ptr_to(
                                [
                                    _A_DA_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            [
                                da_pack[fragment * 4],
                                da_pack[fragment * 4 + 1],
                                da_pack[fragment * 4 + 2],
                                da_pack[fragment * 4 + 3],
                            ],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DA_READY)

                    _wait_barrier(arena, _BAR_DM_ACC_READY, 0, dm_cursor.phase)
                    dm_cursor.advance()
                    dm_values = _tcgen_load_16x256x8(tmem_base + _TM_ACC0 + tmem_row)
                    dm_pack = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        product0, product1 = _fmul2(
                            dm_values[pair * 2],
                            dm_values[pair * 2 + 1],
                            strict_decay[pair * 2],
                            strict_decay[pair * 2 + 1],
                        )
                        K.assign(dm_pack[pair], _pack_io_pair(-product0, -product1, io_dtype))
                    for fragment in range(4):
                        _stmatrix_x4(
                            arena.ptr_to(
                                [
                                    _DM_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            [
                                dm_pack[fragment * 4],
                                dm_pack[fragment * 4 + 1],
                                dm_pack[fragment * 4 + 2],
                                dm_pack[fragment * 4 + 3],
                            ],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DM_ACC_DONE)

                    _wait_barrier(arena, _BAR_DK_ATTN_READY, 0, dk_attn_cursor.phase)
                    dk_attn_cursor.advance()
                    dks = []
                    for sub in range(2):
                        dks.append(
                            _tcgen_load_16x256x8(
                                tmem_base
                                + _TM_DVDK
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                        )
                    _tcgen_wait_load()
                    _arrive_barrier(arena, _BAR_DK_ATTN_DONE)

                    kk_words = K.alloc_local((16,), "uint32")
                    for fragment in range(4):
                        words = K.alloc_local((4,), "uint32")
                        _ldmatrix_x4(
                            words,
                            arena.ptr_to(
                                [
                                    _KK_BASE
                                    + (
                                        store_row * 64
                                        + _swizzle_xor_128b(store_row, store_col + fragment * 16)
                                    )
                                    * 2
                                ]
                            ),
                            trans=False,
                        )
                        for word in range(4):
                            K.assign(kk_words[fragment * 4 + word], words[word])
                    beta_inverse = K.alloc_local((2,), "float32")
                    K.assign(beta_inverse[0], _rcp(beta_rows[0] + K.float32(1.0e-10)))
                    K.assign(beta_inverse[1], _rcp(beta_rows[2] + K.float32(1.0e-10)))
                    row_acc = K.alloc_local((8,), "float32")
                    col_part = K.alloc_local((16,), "float32")
                    for value in range(8):
                        K.assign(row_acc[value], K.float32(0.0))
                    for value in range(16):
                        K.assign(col_part[value], _opaque_zero())
                    for pair in range(16):
                        kk0 = K.local_scalar("float32")
                        kk1 = K.local_scalar("float32")
                        _unpack_io_pair(kk_words[pair], kk0, kk1, io_dtype)
                        product0, product1 = _fmul2(
                            dm_values[pair * 2], dm_values[pair * 2 + 1], kk0, kk1
                        )
                        e0, e1 = _fmul2(
                            product0, product1, beta_inverse[pair % 2], beta_inverse[pair % 2]
                        )
                        bucket = (pair % 2) * 4 + (pair // 2) % 4
                        K.assign(row_acc[bucket], row_acc[bucket] + e0 + e1)
                        column = (pair // 2) * 2
                        next0, next1 = _fadd2(col_part[column], col_part[column + 1], e0, e1)
                        K.assign(col_part[column], next0)
                        K.assign(col_part[column + 1], next1)
                    row_part = K.alloc_local((2,), "float32")
                    K.assign(row_part[0], (row_acc[0] + row_acc[1]) + (row_acc[2] + row_acc[3]))
                    K.assign(row_part[1], (row_acc[4] + row_acc[5]) + (row_acc[6] + row_acc[7]))
                    for distance in (1, 2):
                        for part in range(2):
                            K.assign(
                                row_part[part],
                                row_part[part] + _shfl_bfly_f32(row_part[part], distance),
                            )

                    _wait_barrier(arena, _BAR_DBETA_CG1_READY, 0, dbeta_cursor.phase)
                    dbeta_cursor.advance()
                    with K.If(lane % 4 == 0), K.Then():
                        for part in range(2):
                            row = warp_in_group * 16 + lane // 4 + part * 8
                            beta_partial = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                beta_partial,
                                arena.ptr_to([_BETA_BASE + beta_stage * 256 + row * 4]),
                            )
                            result = beta_partial - row_part[part] * beta_inverse[part]
                            if beta_sigmoid:
                                beta_value = beta_rows[part * 2]
                                result = result * (beta_value - beta_value * beta_value)
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_BETA_BASE + beta_stage * 256 + row * 4]), result
                            )

                    part_k = K.alloc_local((16,), "float32")
                    for value in range(16):
                        K.assign(part_k[value], K.float32(0.0))
                    for sub in range(2):
                        for matrix_row in range(4):
                            k_words = K.alloc_local((4,), "uint32")
                            row = frag_row + matrix_row * 16
                            channel = frag_segment * 64 + frag_col + sub * 16
                            _ldmatrix_x4(
                                k_words,
                                arena.ptr_to([_io_byte(_K_BASE, k_stage, row, channel)]),
                                trans=True,
                            )
                            for word in range(4):
                                k0 = K.local_scalar("float32")
                                k1 = K.local_scalar("float32")
                                _unpack_io_pair(k_words[word], k0, k1, io_dtype)
                                fragment_value = matrix_row * 8 + word * 2
                                part_value = (fragment_value // 4) * 2 + fragment_value % 2
                                if sub == 0 and word % 2 == 0:
                                    next0, next1 = _fmul2(
                                        dks[sub][fragment_value],
                                        dks[sub][fragment_value + 1],
                                        k0,
                                        k1,
                                    )
                                else:
                                    next0, next1 = _ffma2(
                                        dks[sub][fragment_value],
                                        dks[sub][fragment_value + 1],
                                        k0,
                                        k1,
                                        part_k[part_value],
                                        part_k[part_value + 1],
                                    )
                                K.assign(part_k[part_value], next0)
                                K.assign(part_k[part_value + 1], next1)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_K_CG0_DONE, k_stage)

                    dgate_parts = K.alloc_local((16,), "float32")
                    for value in range(16):
                        K.assign(dgate_parts[value], col_part[value] - part_k[value])
                    dgate0, dgate1 = reduce_scatter_16(dgate_parts)
                    dgate_last = K.local_scalar("float32", init=k_dot)
                    with (
                        K.If(
                            K.And(
                                have_dstate,
                                K.And(
                                    reverse_index + dstate_in0 >= 1,
                                    reverse_index < cend - first_state_chunk,
                                ),
                            )
                        ),
                        K.Then(),
                    ):
                        K.assign(dgate_last, dgate_last + cumprod_total * state_dot)
                    # All four CG0 warps read the shared KK tile through
                    # ldmatrix above.  Do not let a faster warp reuse that
                    # storage for the dgate reduction until every peer has
                    # finished its read.
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    token0 = (lane // 4) * 8 + (lane % 4) * 2
                    K.ptx.st.shared.f32(
                        arena.ptr_to([_KK_BASE + (warp_in_group * 64 + token0) * 4]), dgate0
                    )
                    K.ptx.st.shared.f32(
                        arena.ptr_to([_KK_BASE + (warp_in_group * 64 + token0 + 1) * 4]),
                        K.if_then_else(lane == 31, dgate1 + dgate_last, dgate1),
                    )

                    _wait_barrier(arena, _BAR_DGATE_CG1_READY, 0, dgate_cursor.phase)
                    dgate_cursor.advance()
                    with K.If(lane % 4 == 0), K.Then():
                        for part in range(2):
                            row = warp_in_group * 16 + lane // 4 + part * 8
                            gate_partial = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                gate_partial,
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + row * 4]),
                            )
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + row * 4]),
                                gate_partial - row_part[part],
                            )
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    with K.If(cg0_thread < 64), K.Then():
                        dgate_sum = K.local_scalar("float32", init=K.float32(0.0))
                        for group in range(4):
                            partial = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                partial, arena.ptr_to([_KK_BASE + (group * 64 + cg0_thread) * 4])
                            )
                            K.assign(dgate_sum, dgate_sum + partial)
                        gate_partial = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            gate_partial,
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + cg0_thread * 4]),
                        )
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + cg0_thread * 4]),
                            gate_partial + dgate_sum,
                        )

                    _arrive_barrier(arena, _BAR_GATE_DONE, gate_stage)
                    _arrive_barrier(arena, _BAR_BETA_DONE, beta_stage)
                sched_consume(sched_state, tile)

            _wait_barrier(arena, _BAR_A_DONE, a_cursor.stage, a_cursor.phase)
            a_cursor.advance()
            _arrive_barrier(arena, _BAR_TMEM_DONE)

        with cg1:
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_TMEM_MAILBOX]))
            warp_in_group = K.warp_id_in_role()
            cg1_thread = warp_in_group * 32 + lane
            tmem_row = K.shift_left(warp_in_group * 32, K.int32(16))
            frag_row = cg1_thread % 8 + ((cg1_thread // 16) % 2) * 8
            frag_col = ((cg1_thread // 8) % 2) * 8 + ((cg1_thread // 32) % 2) * 32
            frag_segment = cg1_thread // 64

            v_cursor = K.PipelineState(1, phase=0)
            do_cursor = K.PipelineState(1, phase=0)
            gate_cursor = K.PipelineState(2, phase=0)
            k_state_cursor = K.PipelineState(1, phase=0)
            u_cursor = K.PipelineState(1, phase=0)
            dy_cursor = K.PipelineState(1, phase=0)
            du_scale_cursor = K.PipelineState(1, phase=0)
            du_total_cursor = K.PipelineState(1, phase=0)
            dk_total_cursor = K.PipelineState(1, phase=0)
            dstate_acc_cursor = K.PipelineState(1, phase=0)
            state_dot_cursor = K.PipelineState(1, phase=0)
            dq_cursor = K.PipelineState(1, phase=1)
            beta_cursor = K.PipelineState(2, phase=0)
            sdv_cursor = K.PipelineState(1, phase=1)
            dk_state_cursor = K.PipelineState(1, phase=0)
            dq_total_cursor = K.PipelineState(1, phase=0)
            dstate_inp_cursor = K.PipelineState(1, phase=1)
            dk_cursor = K.PipelineState(1, phase=1)
            dv_cursor = K.PipelineState(1, phase=1)
            sched_state = K.PipelineState(2, phase=0)

            def reduce_scatter_16(values):
                stage8 = K.alloc_local((8,), "float32")
                for output, source in enumerate((0, 1, 4, 5, 8, 9, 12, 13)):
                    high = ((lane // 4) % 2) == 1
                    sent = K.if_then_else(high, values[source], values[source + 2])
                    kept = K.if_then_else(high, values[source + 2], values[source])
                    K.assign(stage8[output], kept + _shfl_bfly_f32(sent, 4))
                stage4 = K.alloc_local((4,), "float32")
                for output, source in enumerate((0, 1, 4, 5)):
                    high = ((lane // 8) % 2) == 1
                    sent = K.if_then_else(high, stage8[source], stage8[source + 2])
                    kept = K.if_then_else(high, stage8[source + 2], stage8[source])
                    K.assign(stage4[output], kept + _shfl_bfly_f32(sent, 8))
                stage2 = K.alloc_local((2,), "float32")
                for output in range(2):
                    high = ((lane // 16) % 2) == 1
                    sent = K.if_then_else(high, stage4[output], stage4[output + 2])
                    kept = K.if_then_else(high, stage4[output + 2], stage4[output])
                    K.assign(stage2[output], kept + _shfl_bfly_f32(sent, 16))
                return stage2[0], stage2[1]

            def load_col_fragment(base, stage, sub, matrix_row):
                words = K.alloc_local((4,), "uint32")
                row = frag_row + matrix_row * 16
                channel = frag_segment * 64 + frag_col + sub * 16
                _ldmatrix_x4(words, arena.ptr_to([_io_byte(base, stage, row, channel)]), trans=True)
                return words

            def store_col_fragment(base, stage, sub, matrix_row, words):
                row = frag_row + matrix_row * 16
                channel = frag_segment * 64 + frag_col + sub * 16
                K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    arena.ptr_to([_io_byte(base, stage, row, channel)]),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )

            def stage_dstate_smem():
                for sub in range(4):
                    values = _tcgen_load_32x32x32(tmem_base + _TM_DSTATE + sub * 32 + tmem_row)
                    for group in range(4):
                        packed = K.alloc_local((4,), "uint32")
                        for pair in range(4):
                            K.assign(
                                packed[pair],
                                _pack_io_pair(
                                    values[group * 8 + pair * 2],
                                    values[group * 8 + pair * 2 + 1],
                                    io_dtype,
                                ),
                            )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to(
                                [_state_byte(_DSTATE_BASE, cg1_thread, sub * 32 + group * 8)]
                            ),
                            packed[0],
                            packed[1],
                            packed[2],
                            packed[3],
                        )

            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                cend = item[5]
                bos = item[6]
                eos = item[7]
                num_chunks = (eos - bos + _BT - 1) // _BT
                num_item_chunks = cend - wstart
                state_base = (K.cast(batch, "int64") * n_heads_out + head) * (_DV * _DK)

                if use_dstate_in:
                    with K.If(num_item_chunks > 0), K.Then():
                        inp_stage = K.local_scalar("int32", init=dstate_inp_cursor.stage)
                        _wait_barrier(
                            arena, _BAR_DSTATE_INP_DONE, inp_stage, dstate_inp_cursor.phase
                        )
                        dstate_inp_cursor.advance()
                        for sub in range(4):
                            seed_values = K.alloc_local((32,), "float32")
                            seed_pack = K.alloc_local((16,), "uint32")
                            for value in range(32):
                                loaded = K.local_scalar("float32")
                                K.ptx.ld.global_.f32(
                                    loaded,
                                    dstate_in.ptr_to(
                                        [
                                            state_base
                                            + K.cast(cg1_thread, "int64") * _DK
                                            + sub * 32
                                            + value
                                        ]
                                    ),
                                )
                                K.assign(
                                    seed_values[value],
                                    K.if_then_else(cend == num_chunks, loaded, K.float32(0.0)),
                                )
                            _tcgen_store_32x32x32(
                                tmem_base + _TM_DSTATE + sub * 32 + tmem_row, seed_values
                            )
                            for pair in range(16):
                                K.assign(
                                    seed_pack[pair],
                                    _pack_io_pair(
                                        seed_values[pair * 2], seed_values[pair * 2 + 1], io_dtype
                                    ),
                                )
                            _tcgen_store_32x32x16(
                                tmem_base + _TM_DSTATE_INP + sub * 16 + tmem_row, seed_pack
                            )
                        _tcgen_wait_store()
                        _arrive_barrier(arena, _BAR_DSTATE_INP_READY, inp_stage)
                        stage_dstate_smem()
                        K.ptx.fence.proxy.async_.shared__cta()
                        _arrive_barrier(arena, _BAR_DSTATE_SMEM_READY)

                with K.serial(num_item_chunks, unroll=False) as reverse_index:
                    chunk = cend - 1 - reverse_index
                    have_dstate = K.bool(True) if use_dstate_in else reverse_index > 0
                    gate_stage = K.local_scalar("int32", init=gate_cursor.stage)
                    _wait_barrier(arena, _BAR_GATE_READY, gate_stage, gate_cursor.phase)
                    gate_cursor.advance()
                    beta_stage = K.local_scalar("int32", init=beta_cursor.stage)
                    _wait_barrier(arena, _BAR_BETA_READY, beta_stage, beta_cursor.phase)
                    beta_cursor.advance()

                    with K.If(have_dstate), K.Then():
                        cumprod_top = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            cumprod_top, arena.ptr_to([_CUMPROD_BASE + gate_stage * 256 + 63 * 4])
                        )
                        for sub in range(4):
                            address = tmem_base + _TM_DSTATE + sub * 32 + tmem_row
                            values = _tcgen_load_32x32x32(address)
                            scaled = K.alloc_local((32,), "float32")
                            for pair in range(16):
                                value0, value1 = _fmul2(
                                    values[pair * 2], values[pair * 2 + 1], cumprod_top, cumprod_top
                                )
                                K.assign(scaled[pair * 2], value0)
                                K.assign(scaled[pair * 2 + 1], value1)
                            _tcgen_store_32x32x32(address, scaled)
                        _tcgen_wait_store()
                        _arrive_barrier(arena, _BAR_DSTATE_SCALE_DONE)

                    cumprod_values = K.alloc_local((32,), "float32")
                    for value in range(32):
                        column = (lane % 4) * 2 + (value // 4) * 8 + value % 2
                        K.ptx.ld.shared.f32(
                            cumprod_values[value],
                            arena.ptr_to([_CUMPROD_BASE + gate_stage * 256 + column * 4]),
                        )
                    negative_cumprod = K.alloc_local((32,), "float32")
                    for value in range(32):
                        K.assign(negative_cumprod[value], -cumprod_values[value])

                    do_stage = K.local_scalar("int32", init=do_cursor.stage)
                    _wait_barrier(arena, _BAR_DO_READY, do_stage, do_cursor.phase)
                    do_cursor.advance()
                    do_values = [K.alloc_local((32,), "float32"), K.alloc_local((32,), "float32")]
                    for sub in range(2):
                        for matrix_row in range(4):
                            words = load_col_fragment(_DO_BASE, do_stage, sub, matrix_row)
                            for word in range(4):
                                do0 = K.local_scalar("float32")
                                do1 = K.local_scalar("float32")
                                _unpack_io_pair(words[word], do0, do1, io_dtype)
                                index = matrix_row * 8 + word * 2
                                product0, product1 = _fmul2(
                                    do0, do1, cumprod_values[index], cumprod_values[index + 1]
                                )
                                value0, value1 = _fmul2(product0, product1, scale, scale)
                                K.assign(do_values[sub][index], value0)
                                K.assign(do_values[sub][index + 1], value1)
                    for sub in range(2):
                        packed = K.alloc_local((16,), "uint32")
                        for pair in range(16):
                            K.assign(
                                packed[pair],
                                _pack_io_pair(
                                    do_values[sub][pair * 2], do_values[sub][pair * 2 + 1], io_dtype
                                ),
                            )
                        _tcgen_store_16x128x8(
                            tmem_base
                            + _TM_INP0
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                            packed,
                        )
                    _tcgen_wait_store()
                    _arrive_barrier(arena, _BAR_DO_PRIME_READY)

                    with K.If(have_dstate), K.Then():
                        last_cs = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            last_cs, arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + 63 * 4])
                        )
                        decay_scale = K.alloc_local((32,), "float32")
                        for value in range(32):
                            column = (lane % 4) * 2 + (value // 4) * 8 + value % 2
                            column_cs = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                column_cs,
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + column * 4]),
                            )
                            K.assign(decay_scale[value], _exp2(last_cs - column_cs))
                        _wait_barrier(arena, _BAR_DU_SCALE_READY, 0, du_scale_cursor.phase)
                        du_scale_cursor.advance()
                        for sub in range(2):
                            address = (
                                tmem_base
                                + _TM_DVDK
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                            values = _tcgen_load_16x256x8(address)
                            scaled = K.alloc_local((32,), "float32")
                            for pair in range(16):
                                value0, value1 = _fmul2(
                                    values[pair * 2],
                                    values[pair * 2 + 1],
                                    decay_scale[pair * 2],
                                    decay_scale[pair * 2 + 1],
                                )
                                K.assign(scaled[pair * 2], value0)
                                K.assign(scaled[pair * 2 + 1], value1)
                            _tcgen_store_16x256x8(address, scaled)
                        _tcgen_wait_store()
                        _arrive_barrier(arena, _BAR_DU_SCALE_DONE)

                    v_stage = K.local_scalar("int32", init=v_cursor.stage)
                    _wait_barrier(arena, _BAR_V_READY, v_stage, v_cursor.phase)
                    v_cursor.advance()
                    with K.If(chunk >= first_state_chunk), K.Then():
                        v_words = [K.alloc_local((16,), "uint32"), K.alloc_local((16,), "uint32")]
                        for sub in range(2):
                            for matrix_row in range(4):
                                words = load_col_fragment(_V_U_BASE, v_stage, sub, matrix_row)
                                for word in range(4):
                                    K.assign(v_words[sub][matrix_row * 4 + word], words[word])
                        _wait_barrier(arena, _BAR_K_STATE_READY, 0, k_state_cursor.phase)
                        k_state_cursor.advance()
                        for sub in range(2):
                            values = _tcgen_load_16x256x8(
                                tmem_base
                                + _TM_ACC0
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                            g_pack = K.alloc_local((16,), "uint32")
                            y_pack = K.alloc_local((16,), "uint32")
                            for pair in range(16):
                                g0, g1 = _fmul2(
                                    values[pair * 2],
                                    values[pair * 2 + 1],
                                    cumprod_values[pair * 2],
                                    cumprod_values[pair * 2 + 1],
                                )
                                K.assign(g_pack[pair], _pack_io_pair(g0, g1, io_dtype))
                                y_word = K.local_scalar("uint32")
                                _sub_io_pair(y_word, v_words[sub][pair], g_pack[pair], io_dtype)
                                K.assign(y_pack[pair], y_word)
                            _tcgen_store_16x128x8(
                                tmem_base
                                + _TM_Y
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                                y_pack,
                            )
                            _tcgen_store_16x128x8(
                                tmem_base
                                + _TM_GK
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                                g_pack,
                            )
                    with K.If(chunk < first_state_chunk), K.Then():
                        for sub in range(2):
                            v_pack = K.alloc_local((16,), "uint32")
                            for matrix_row in range(4):
                                words = load_col_fragment(_V_U_BASE, v_stage, sub, matrix_row)
                                for word in range(4):
                                    K.assign(v_pack[matrix_row * 4 + word], words[word])
                            _tcgen_store_16x128x8(
                                tmem_base
                                + _TM_Y
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                                v_pack,
                            )
                    _tcgen_wait_store()
                    _arrive_barrier(arena, _BAR_Y_READY)

                    _wait_barrier(arena, _BAR_DU_TOTAL_READY, 0, du_total_cursor.phase)
                    du_total_cursor.advance()
                    for sub in range(2):
                        values = _tcgen_load_16x256x8(
                            tmem_base
                            + _TM_DVDK
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                        )
                        packed = K.alloc_local((16,), "uint32")
                        for pair in range(16):
                            K.assign(
                                packed[pair],
                                _pack_io_pair(values[pair * 2], values[pair * 2 + 1], io_dtype),
                            )
                        _tcgen_store_16x128x8(
                            tmem_base
                            + _TM_INP1
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                            packed,
                        )
                    _tcgen_wait_store()
                    _arrive_barrier(arena, _BAR_DU_INP_READY)

                    _wait_barrier(arena, _BAR_U_ACC_READY, 0, u_cursor.phase)
                    u_cursor.advance()
                    u_values = []
                    for sub in range(2):
                        u_values.append(
                            _tcgen_load_16x256x8(
                                tmem_base
                                + _TM_ACC1
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                        )
                    for sub in range(2):
                        for matrix_row in range(4):
                            packed = K.alloc_local((4,), "uint32")
                            for pair in range(4):
                                K.assign(
                                    packed[pair],
                                    _pack_io_pair(
                                        u_values[sub][matrix_row * 8 + pair * 2],
                                        u_values[sub][matrix_row * 8 + pair * 2 + 1],
                                        io_dtype,
                                    ),
                                )
                            store_col_fragment(_V_U_BASE, v_stage, sub, matrix_row, packed)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_U_READY)

                    _wait_barrier(arena, _BAR_DY_ACC_READY, 0, dy_cursor.phase)
                    dy_cursor.advance()
                    dy_values = []
                    for sub in range(2):
                        dy_values.append(
                            _tcgen_load_16x256x8(
                                tmem_base
                                + _TM_ACC0
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                        )
                    for sub in range(2):
                        dyp_pack = K.alloc_local((16,), "uint32")
                        for pair in range(16):
                            value0, value1 = _fmul2(
                                dy_values[sub][pair * 2],
                                dy_values[sub][pair * 2 + 1],
                                negative_cumprod[pair * 2],
                                negative_cumprod[pair * 2 + 1],
                            )
                            K.assign(dyp_pack[pair], _pack_io_pair(value0, value1, io_dtype))
                        _tcgen_store_16x128x8(
                            tmem_base
                            + _TM_INP1
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                            dyp_pack,
                        )
                    _tcgen_wait_store()
                    _arrive_barrier(arena, _BAR_DYP_INP_READY)

                    dv_stage = K.local_scalar("int32", init=dv_cursor.stage)
                    _wait_barrier(arena, _BAR_DV_STG_DONE, dv_stage, dv_cursor.phase)
                    dv_cursor.advance()
                    _wait_barrier(arena, _BAR_SDV_DONE, 0, sdv_cursor.phase)
                    sdv_cursor.advance()
                    for sub in range(2):
                        for matrix_row in range(4):
                            packed = K.alloc_local((4,), "uint32")
                            for pair in range(4):
                                K.assign(
                                    packed[pair],
                                    _pack_io_pair(
                                        dy_values[sub][matrix_row * 8 + pair * 2],
                                        dy_values[sub][matrix_row * 8 + pair * 2 + 1],
                                        io_dtype,
                                    ),
                                )
                            store_col_fragment(_DV_DY_BASE, dv_stage, sub, matrix_row, packed)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DV_STG_READY, dv_stage)

                    dq_stage = K.local_scalar("int32", init=dq_cursor.stage)
                    _wait_barrier(arena, _BAR_DQ_STG_DONE, dq_stage, dq_cursor.phase)
                    dq_cursor.advance()

                    part_y = K.alloc_local((16,), "float32")
                    for value in range(16):
                        K.assign(part_y[value], K.float32(0.0))
                    for sub in range(2):
                        y_words = _tcgen_load_16x128x8(
                            tmem_base
                            + _TM_Y
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                        )
                        for pair in range(16):
                            y0 = K.local_scalar("float32")
                            y1 = K.local_scalar("float32")
                            _unpack_io_pair(y_words[pair], y0, y1, io_dtype)
                            fragment_value = pair * 2
                            part_value = (fragment_value // 4) * 2 + fragment_value % 2
                            if sub == 0 and pair % 2 == 0:
                                next0, next1 = _fmul2(
                                    dy_values[sub][fragment_value],
                                    dy_values[sub][fragment_value + 1],
                                    y0,
                                    y1,
                                )
                            else:
                                next0, next1 = _ffma2(
                                    dy_values[sub][fragment_value],
                                    dy_values[sub][fragment_value + 1],
                                    y0,
                                    y1,
                                    part_y[part_value],
                                    part_y[part_value + 1],
                                )
                            K.assign(part_y[part_value], next0)
                            K.assign(part_y[part_value + 1], next1)
                    part_y0, part_y1 = reduce_scatter_16(part_y)
                    token0 = (lane // 4) * 8 + (lane % 4) * 2
                    K.ptx.st.shared.f32(
                        arena.ptr_to([_DQ_BASE + (warp_in_group * 64 + token0) * 4]), part_y0
                    )
                    K.ptx.st.shared.f32(
                        arena.ptr_to([_DQ_BASE + (warp_in_group * 64 + token0 + 1) * 4]), part_y1
                    )

                    with K.If(chunk >= first_state_chunk), K.Then():
                        part_g = K.alloc_local((16,), "float32")
                        for value in range(16):
                            K.assign(part_g[value], K.float32(0.0))
                        for sub in range(2):
                            g_words = _tcgen_load_16x128x8(
                                tmem_base
                                + _TM_GK
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                            for pair in range(16):
                                g0 = K.local_scalar("float32")
                                g1 = K.local_scalar("float32")
                                _unpack_io_pair(g_words[pair], g0, g1, io_dtype)
                                fragment_value = pair * 2
                                part_value = (fragment_value // 4) * 2 + fragment_value % 2
                                if sub == 0 and pair % 2 == 0:
                                    next0, next1 = _fmul2(
                                        dy_values[sub][fragment_value],
                                        dy_values[sub][fragment_value + 1],
                                        g0,
                                        g1,
                                    )
                                else:
                                    next0, next1 = _ffma2(
                                        dy_values[sub][fragment_value],
                                        dy_values[sub][fragment_value + 1],
                                        g0,
                                        g1,
                                        part_g[part_value],
                                        part_g[part_value + 1],
                                    )
                                K.assign(part_g[part_value], next0)
                                K.assign(part_g[part_value + 1], next1)
                        part_g0, part_g1 = reduce_scatter_16(part_g)
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_DQ_BASE + (256 + warp_in_group * 64 + token0) * 4]),
                            part_g0,
                        )
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_DQ_BASE + (256 + warp_in_group * 64 + token0 + 1) * 4]),
                            part_g1,
                        )
                    K.ptx.bar.sync(K.uint32(5), K.uint32(128))
                    with K.If(cg1_thread < 64), K.Then():
                        ysum = K.local_scalar("float32", init=K.float32(0.0))
                        for group in range(4):
                            partial = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                partial, arena.ptr_to([_DQ_BASE + (group * 64 + cg1_thread) * 4])
                            )
                            K.assign(ysum, ysum + partial)
                        beta_value = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            beta_value,
                            arena.ptr_to([_BETA_BASE + beta_stage * 256 + cg1_thread * 4]),
                        )
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_BETA_BASE + beta_stage * 256 + cg1_thread * 4]),
                            ysum * _rcp(beta_value + K.float32(1.0e-10)),
                        )
                        with K.If(chunk >= first_state_chunk), K.Then():
                            gsum = K.local_scalar("float32", init=K.float32(0.0))
                            for group in range(4):
                                partial = K.local_scalar("float32")
                                K.ptx.ld.shared.f32(
                                    partial,
                                    arena.ptr_to([_DQ_BASE + (256 + group * 64 + cg1_thread) * 4]),
                                )
                                K.assign(gsum, gsum + partial)
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + cg1_thread * 4]),
                                -gsum,
                            )
                        with K.If(chunk < first_state_chunk), K.Then():
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + cg1_thread * 4]),
                                K.float32(0.0),
                            )
                    _arrive_barrier(arena, _BAR_BETA_DONE, beta_stage)
                    _arrive_barrier(arena, _BAR_DBETA_CG1_READY)
                    K.ptx.bar.sync(K.uint32(5), K.uint32(128))

                    q_words = [K.alloc_local((16,), "uint32"), K.alloc_local((16,), "uint32")]
                    for sub in range(2):
                        for matrix_row in range(4):
                            words = load_col_fragment(_Q_BASE, 0, sub, matrix_row)
                            for word in range(4):
                                K.assign(q_words[sub][matrix_row * 4 + word], words[word])
                        _tcgen_store_16x128x8(
                            tmem_base
                            + _TM_Y
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16)),
                            q_words[sub],
                        )
                    _tcgen_wait_store()
                    _arrive_barrier(arena, _BAR_Q_CG1_DONE)

                    _wait_barrier(arena, _BAR_DQ_TOTAL_READY, 0, dq_total_cursor.phase)
                    dq_total_cursor.advance()
                    dq_values = []
                    for sub in range(2):
                        values = _tcgen_load_16x256x8(
                            tmem_base
                            + _TM_DSTATE_INP
                            + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                        )
                        dq_values.append(values)
                        for matrix_row in range(4):
                            packed = K.alloc_local((4,), "uint32")
                            for pair in range(4):
                                K.assign(
                                    packed[pair],
                                    _pack_io_pair(
                                        values[matrix_row * 8 + pair * 2],
                                        values[matrix_row * 8 + pair * 2 + 1],
                                        io_dtype,
                                    ),
                                )
                            store_col_fragment(_DQ_BASE, dq_stage, sub, matrix_row, packed)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DQ_TOTAL_DONE)
                    _arrive_barrier(arena, _BAR_DQ_STG_READY, dq_stage)

                    part_q = K.alloc_local((16,), "float32")
                    for value in range(16):
                        K.assign(part_q[value], K.float32(0.0))
                    for sub in range(2):
                        for matrix_row in range(4):
                            for word in range(4):
                                q0 = K.local_scalar("float32")
                                q1 = K.local_scalar("float32")
                                _unpack_io_pair(
                                    q_words[sub][matrix_row * 4 + word], q0, q1, io_dtype
                                )
                                fragment_value = matrix_row * 8 + word * 2
                                part_value = (fragment_value // 4) * 2 + fragment_value % 2
                                if sub == 0 and word % 2 == 0:
                                    next0, next1 = _fmul2(
                                        dq_values[sub][fragment_value],
                                        dq_values[sub][fragment_value + 1],
                                        q0,
                                        q1,
                                    )
                                else:
                                    next0, next1 = _ffma2(
                                        dq_values[sub][fragment_value],
                                        dq_values[sub][fragment_value + 1],
                                        q0,
                                        q1,
                                        part_q[part_value],
                                        part_q[part_value + 1],
                                    )
                                K.assign(part_q[part_value], next0)
                                K.assign(part_q[part_value + 1], next1)
                    part_q0, part_q1 = reduce_scatter_16(part_q)
                    token0 = (lane // 4) * 8 + (lane % 4) * 2
                    K.ptx.st.shared.f32(
                        arena.ptr_to([_DSTATE_BASE + (warp_in_group * 64 + token0) * 4]), part_q0
                    )
                    K.ptx.st.shared.f32(
                        arena.ptr_to([_DSTATE_BASE + (warp_in_group * 64 + token0 + 1) * 4]),
                        part_q1,
                    )
                    K.ptx.bar.sync(K.uint32(5), K.uint32(128))
                    with K.If(cg1_thread < 64), K.Then():
                        qsum = K.local_scalar("float32", init=K.float32(0.0))
                        for group in range(4):
                            partial = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                partial,
                                arena.ptr_to([_DSTATE_BASE + (group * 64 + cg1_thread) * 4]),
                            )
                            K.assign(qsum, qsum + partial)
                        gate_partial = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            gate_partial,
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + cg1_thread * 4]),
                        )
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + cg1_thread * 4]),
                            gate_partial + qsum,
                        )
                    _arrive_barrier(arena, _BAR_GATE_DONE, gate_stage)
                    _arrive_barrier(arena, _BAR_DGATE_CG1_READY)

                    with K.If(chunk >= wstart + 1), K.Then():
                        dstate_stage = K.local_scalar("int32", init=dstate_acc_cursor.stage)
                        _wait_barrier(
                            arena, _BAR_DSTATE_ACC_READY, dstate_stage, dstate_acc_cursor.phase
                        )
                        dstate_acc_cursor.advance()
                        inp_stage = K.local_scalar("int32", init=dstate_inp_cursor.stage)
                        _wait_barrier(
                            arena, _BAR_DSTATE_INP_DONE, inp_stage, dstate_inp_cursor.phase
                        )
                        dstate_inp_cursor.advance()
                        for sub in range(4):
                            values = _tcgen_load_32x32x32(
                                tmem_base + _TM_DSTATE + sub * 32 + tmem_row
                            )
                            packed = K.alloc_local((16,), "uint32")
                            for pair in range(16):
                                K.assign(
                                    packed[pair],
                                    _pack_io_pair(values[pair * 2], values[pair * 2 + 1], io_dtype),
                                )
                            _tcgen_store_32x32x16(
                                tmem_base + _TM_DSTATE_INP + sub * 16 + tmem_row, packed
                            )
                        _tcgen_wait_store()
                        _arrive_barrier(arena, _BAR_DSTATE_INP_READY, inp_stage)

                    dk_stage = K.local_scalar("int32", init=dk_cursor.stage)
                    _wait_barrier(arena, _BAR_DK_STG_DONE, dk_stage, dk_cursor.phase)
                    dk_cursor.advance()
                    state_part = [K.alloc_local((32,), "float32"), K.alloc_local((32,), "float32")]
                    for sub in range(2):
                        for value in range(32):
                            K.assign(state_part[sub][value], K.float32(0.0))
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(arena, _BAR_DK_STATE_READY, 0, dk_state_cursor.phase)
                        dk_state_cursor.advance()
                        state_raw = []
                        for sub in range(2):
                            state_raw.append(
                                _tcgen_load_16x256x8(
                                    tmem_base
                                    + _TM_INP0
                                    + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                                )
                            )
                        _tcgen_wait_load()
                        for sub in range(2):
                            for pair in range(16):
                                value0, value1 = _fmul2(
                                    state_raw[sub][pair * 2],
                                    state_raw[sub][pair * 2 + 1],
                                    negative_cumprod[pair * 2],
                                    negative_cumprod[pair * 2 + 1],
                                )
                                K.assign(state_part[sub][pair * 2], value0)
                                K.assign(state_part[sub][pair * 2 + 1], value1)

                    _wait_barrier(arena, _BAR_DK_TOTAL_READY, 0, dk_total_cursor.phase)
                    dk_total_cursor.advance()
                    total_dk = []
                    for sub in range(2):
                        total_dk.append(
                            _tcgen_load_16x256x8(
                                tmem_base
                                + _TM_DVDK
                                + K.shift_left(warp_in_group * 32 + sub * 16, K.int32(16))
                            )
                        )
                    _tcgen_wait_load()
                    _arrive_barrier(arena, _BAR_DK_TOTAL_DONE)
                    for sub in range(2):
                        for matrix_row in range(4):
                            packed = K.alloc_local((4,), "uint32")
                            for pair in range(4):
                                index = matrix_row * 8 + pair * 2
                                K.assign(
                                    packed[pair],
                                    _pack_io_pair(
                                        total_dk[sub][index] + state_part[sub][index],
                                        total_dk[sub][index + 1] + state_part[sub][index + 1],
                                        io_dtype,
                                    ),
                                )
                            store_col_fragment(_DK_BASE, dk_stage, sub, matrix_row, packed)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DK_STG_READY, dk_stage)

                    with K.If(chunk >= wstart + 1), K.Then():
                        _wait_barrier(arena, _BAR_STATE_DOT_DONE, 0, state_dot_cursor.phase)
                    state_dot_cursor.advance()
                    with K.If(chunk >= wstart + 1), K.Then():
                        stage_dstate_smem()
                        K.ptx.fence.proxy.async_.shared__cta()
                        _arrive_barrier(arena, _BAR_DSTATE_SMEM_READY)

                with K.If(num_item_chunks > 0), K.Then():
                    dstate_stage = K.local_scalar("int32", init=dstate_acc_cursor.stage)
                    _wait_barrier(
                        arena, _BAR_DSTATE_ACC_READY, dstate_stage, dstate_acc_cursor.phase
                    )
                    dstate_acc_cursor.advance()
                    if use_dstate0:
                        with K.If(wstart == 0), K.Then():
                            for sub in range(4):
                                values = _tcgen_load_32x32x32(
                                    tmem_base + _TM_DSTATE + sub * 32 + tmem_row
                                )
                                for value in range(32):
                                    K.ptx.st.global_.f32(
                                        dstate0.ptr_to(
                                            [
                                                state_base
                                                + K.cast(cg1_thread, "int64") * _DK
                                                + sub * 32
                                                + value
                                            ]
                                        ),
                                        values[value],
                                    )
                    if not use_dstate_in:
                        _arrive_barrier(arena, _BAR_DSTATE_SCALE_DONE, dstate_stage)
                if use_dstate0:
                    with K.If(K.And(num_item_chunks <= 0, wstart == 0)), K.Then():
                        for sub in range(4):
                            for value in range(32):
                                output_index = (
                                    state_base
                                    + K.cast(cg1_thread, "int64") * _DK
                                    + sub * 32
                                    + value
                                )
                                if use_dstate_in:
                                    passthrough = K.local_scalar("float32")
                                    K.ptx.ld.global_.f32(
                                        passthrough, dstate_in.ptr_to([output_index])
                                    )
                                    K.ptx.st.global_.f32(
                                        dstate0.ptr_to([output_index]), passthrough
                                    )
                                else:
                                    K.ptx.st.global_.f32(
                                        dstate0.ptr_to([output_index]), K.float32(0.0)
                                    )
                sched_consume(sched_state, tile)

            _arrive_barrier(arena, _BAR_TMEM_DONE)
            _wait_barrier(
                arena, _BAR_DSTATE_INP_DONE, dstate_inp_cursor.stage, dstate_inp_cursor.phase
            )
            dstate_inp_cursor.advance()
            _wait_barrier(arena, _BAR_DK_STG_DONE, dk_cursor.stage, dk_cursor.phase)
            dk_cursor.advance()
            _wait_barrier(arena, _BAR_DV_STG_DONE, dv_cursor.stage, dv_cursor.phase)
            dv_cursor.advance()

        with mma:
            elected_lane = K.local_scalar("uint32")
            leader = K.local_scalar("uint32")
            K.ptx.elect_sync(elected_lane, leader, K.uint32(0xFFFFFFFF))
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                arena.ptr_to([_TMEM_MAILBOX]), K.uint32(512)
            )
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_TMEM_MAILBOX]))

            kk_done = K.PipelineState(1, phase=0)
            dk_total_done = K.PipelineState(1, phase=1)
            du_scale_done = K.PipelineState(1, phase=0)
            dk_scale_done = K.PipelineState(1, phase=0)
            dk_attn_done = K.PipelineState(1, phase=0)
            dstate_acc = K.PipelineState(1, phase=0 if use_dstate_in else 1)
            k_ready = K.PipelineState(2, phase=0)
            q_ready = K.PipelineState(1, phase=0)
            state_ready = K.PipelineState(1, phase=0)
            v_release = K.PipelineState(1, phase=0)
            tinv_ready = K.PipelineState(1, phase=0)
            do_ready = K.PipelineState(1, phase=0)
            y_ready = K.PipelineState(1, phase=0)
            u_ready = K.PipelineState(1, phase=0)
            da_ready = K.PipelineState(1, phase=0)
            dv_ready = K.PipelineState(1, phase=0)
            dm_done = K.PipelineState(1, phase=0)
            dstate_smem = K.PipelineState(1, phase=0)
            dq_scale_done = K.PipelineState(1, phase=0)
            dq_total_done = K.PipelineState(1, phase=1)
            a_ready = K.PipelineState(1, phase=0)
            do_prime_ready = K.PipelineState(1, phase=0)
            du_inp_ready = K.PipelineState(1, phase=0)
            dyp_inp_ready = K.PipelineState(1, phase=0)
            dstate_inp_ready = K.PipelineState(1, phase=0)
            sched_state = K.PipelineState(2, phase=0)

            def mma_ss(
                destination,
                desc_a,
                desc_b,
                instruction,
                accumulate,
                *,
                m,
                n,
                k_extent,
                a_transpose=False,
                b_transpose=False,
            ):
                _tcgen_mma_ss(
                    tmem_base + destination,
                    desc_a,
                    desc_b,
                    instruction,
                    leader=leader,
                    m=m,
                    n=n,
                    k_extent=k_extent,
                    a_transpose=a_transpose,
                    b_transpose=b_transpose,
                    accumulate=accumulate,
                )

            def mma_ts(
                destination,
                source,
                desc_b,
                instruction,
                accumulate,
                *,
                n,
                k_extent,
                b_transpose=False,
            ):
                _tcgen_mma_ts(
                    tmem_base + destination,
                    tmem_base + source,
                    desc_b,
                    instruction,
                    leader=leader,
                    n=n,
                    k_extent=k_extent,
                    b_transpose=b_transpose,
                    accumulate=accumulate,
                )

            desc_q = _raw_descriptor(arena, _Q_BASE, 16, 1024, 2)
            desc_q_trans = _raw_descriptor(arena, _Q_BASE, 8192, 1024, 2)
            desc_do = _raw_descriptor(arena, _DO_BASE, 8192, 1024, 2)
            desc_do_kmaj = _raw_descriptor(arena, _DO_BASE, 16, 1024, 2)
            desc_state = _raw_descriptor(arena, _STATE_BASE, 16_384, 1024, 2)
            desc_state_kmaj = _raw_descriptor(arena, _STATE_BASE, 16, 1024, 2)
            desc_tinv = _raw_descriptor(arena, _TINV_BASE, 16, 1024, 2)
            desc_tinv_trans = _raw_descriptor(arena, _TINV_BASE, 8192, 1024, 2)
            desc_a_trans = _raw_descriptor(arena, _A_DA_BASE, 8192, 1024, 2)
            desc_da = _raw_descriptor(arena, _A_DA_BASE, 16, 1024, 2)
            desc_da_trans = _raw_descriptor(arena, _A_DA_BASE, 8192, 1024, 2)
            desc_v_kmaj = _raw_descriptor(arena, _V_U_BASE, 16, 1024, 2)
            desc_dv_kmaj = _raw_descriptor(arena, _DV_DY_BASE, 16, 1024, 2)
            desc_dstate = _raw_descriptor(arena, _DSTATE_BASE, 16_384, 1024, 2)
            desc_dm = _raw_descriptor(arena, _DM_BASE, 16, 1024, 2)
            desc_dm_trans = _raw_descriptor(arena, _DM_BASE, 8192, 1024, 2)

            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                wstart = item[2]
                cend = item[5]
                with K.serial(cend - wstart, unroll=False) as reverse_index:
                    chunk = cend - 1 - reverse_index
                    have_dstate = K.bool(True) if use_dstate_in else reverse_index > 0

                    k_stage = K.local_scalar("int32", init=k_ready.stage)
                    _wait_barrier(arena, _BAR_K_READY, k_stage, k_ready.phase)
                    k_ready.advance()
                    desc_k = _raw_descriptor(arena, _K_BASE + k_stage * 16_384, 16, 1024, 2)
                    desc_k_trans = _raw_descriptor(arena, _K_BASE + k_stage * 16_384, 8192, 1024, 2)
                    mma_ss(_TM_ACC0, desc_k, desc_k, idesc_m64_n64, False, m=64, n=64, k_extent=128)
                    _tcgen_commit(arena, _BAR_KK_READY, leader=leader)

                    q_stage = K.local_scalar("int32", init=q_ready.stage)
                    _wait_barrier(arena, _BAR_Q_READY, q_stage, q_ready.phase)
                    q_ready.advance()
                    mma_ss(_TM_ACC1, desc_q, desc_k, idesc_m64_n64, False, m=64, n=64, k_extent=128)
                    _tcgen_commit(arena, _BAR_A_ACC_READY, leader=leader)

                    state_stage = K.local_scalar("int32", init=state_ready.stage)
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(arena, _BAR_STATE_READY, state_stage, state_ready.phase)
                        state_ready.advance()
                        _wait_barrier(arena, _BAR_KK_DONE, 0, kk_done.phase)
                        kk_done.advance()
                        mma_ss(
                            _TM_ACC0,
                            desc_state_kmaj,
                            desc_k,
                            idesc_m128_n64,
                            False,
                            m=128,
                            n=64,
                            k_extent=128,
                        )
                        _tcgen_commit(arena, _BAR_K_STATE_READY, leader=leader)

                    dstate_inp_stage = K.local_scalar("int32", init=dstate_inp_ready.stage)
                    with K.If(have_dstate), K.Then():
                        _wait_barrier(
                            arena, _BAR_DSTATE_INP_READY, dstate_inp_stage, dstate_inp_ready.phase
                        )
                        dstate_inp_ready.advance()
                    _wait_barrier(arena, _BAR_DK_TOTAL_DONE, 0, dk_total_done.phase)
                    dk_total_done.advance()
                    with K.If(have_dstate), K.Then():
                        mma_ts(
                            _TM_DVDK,
                            _TM_DSTATE_INP,
                            desc_k,
                            idesc_m128_n64,
                            False,
                            n=64,
                            k_extent=128,
                        )
                        _tcgen_commit(arena, _BAR_DU_SCALE_READY, leader=leader)
                        _tcgen_commit(arena, _BAR_DSTATE_INP_DONE, dstate_inp_stage, leader=leader)

                    do_stage = K.local_scalar("int32", init=do_ready.stage)
                    _wait_barrier(arena, _BAR_DO_READY, do_stage, do_ready.phase)
                    do_ready.advance()
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(arena, _BAR_DQ_TOTAL_DONE, 0, dq_total_done.phase)
                        dq_total_done.advance()
                        mma_ss(
                            _TM_DSTATE_INP,
                            desc_state,
                            desc_do_kmaj,
                            idesc_m128_n64_a,
                            False,
                            m=128,
                            n=64,
                            k_extent=128,
                            a_transpose=True,
                        )
                        _tcgen_commit(arena, _BAR_DQ_SCALE_READY, leader=leader)

                    a_stage = K.local_scalar("int32", init=a_ready.stage)
                    _wait_barrier(arena, _BAR_A_READY, a_stage, a_ready.phase)
                    a_ready.advance()
                    with K.If(have_dstate), K.Then():
                        _wait_barrier(arena, _BAR_DU_SCALE_DONE, 0, du_scale_done.phase)
                        du_scale_done.advance()
                    mma_ss(
                        _TM_DVDK,
                        desc_do,
                        desc_a_trans,
                        idesc_m128_n64_ab,
                        have_dstate,
                        m=128,
                        n=64,
                        k_extent=64,
                        a_transpose=True,
                        b_transpose=True,
                    )
                    _tcgen_commit(arena, _BAR_DU_TOTAL_READY, leader=leader)

                    _wait_barrier(arena, _BAR_DO_PRIME_READY, 0, do_prime_ready.phase)
                    do_prime_ready.advance()
                    dstate_stage = K.local_scalar("int32", init=dstate_acc.stage)
                    _wait_barrier(arena, _BAR_DSTATE_SCALE_DONE, dstate_stage, dstate_acc.phase)
                    dstate_acc.advance()
                    mma_ts(
                        _TM_DSTATE,
                        _TM_INP0,
                        desc_q_trans,
                        idesc_m128_n128_b,
                        have_dstate,
                        n=128,
                        k_extent=64,
                        b_transpose=True,
                    )

                    _wait_barrier(arena, _BAR_TINV_READY, 0, tinv_ready.phase)
                    tinv_ready.advance()
                    _wait_barrier(arena, _BAR_Y_READY, 0, y_ready.phase)
                    y_ready.advance()
                    v_stage = K.local_scalar("int32", init=v_release.stage)
                    v_release.advance()
                    mma_ts(_TM_ACC1, _TM_Y, desc_tinv, idesc_m128_n64, False, n=64, k_extent=64)
                    _tcgen_commit(arena, _BAR_U_ACC_READY, leader=leader)

                    _wait_barrier(arena, _BAR_DU_INP_READY, 0, du_inp_ready.phase)
                    du_inp_ready.advance()
                    mma_ts(
                        _TM_ACC0,
                        _TM_INP1,
                        desc_tinv_trans,
                        idesc_m128_n64_b,
                        False,
                        n=64,
                        k_extent=64,
                        b_transpose=True,
                    )
                    _tcgen_commit(arena, _BAR_DY_ACC_READY, leader=leader)

                    _wait_barrier(arena, _BAR_U_READY, 0, u_ready.phase)
                    u_ready.advance()
                    with K.If(have_dstate), K.Then():
                        _wait_barrier(arena, _BAR_DSTATE_SMEM_READY, 0, dstate_smem.phase)
                        dstate_smem.advance()
                        mma_ss(
                            _TM_DVDK,
                            desc_dstate,
                            desc_v_kmaj,
                            idesc_m128_n64_a,
                            False,
                            m=128,
                            n=64,
                            k_extent=128,
                            a_transpose=True,
                        )
                        _tcgen_commit(arena, _BAR_DK_SCALE_READY, leader=leader)

                    mma_ss(
                        _TM_ACC1,
                        desc_do_kmaj,
                        desc_v_kmaj,
                        idesc_m64_n64,
                        False,
                        m=64,
                        n=64,
                        k_extent=128,
                    )
                    _tcgen_commit(arena, _BAR_DA_ACC_READY, leader=leader)
                    _tcgen_commit(arena, _BAR_DO_MMA_DONE, do_stage, leader=leader)

                    _wait_barrier(arena, _BAR_DV_STG_READY, 0, dv_ready.phase)
                    dv_ready.advance()
                    mma_ss(
                        _TM_ACC0,
                        desc_dv_kmaj,
                        desc_v_kmaj,
                        idesc_m64_n64,
                        False,
                        m=64,
                        n=64,
                        k_extent=128,
                    )
                    _tcgen_commit(arena, _BAR_DM_ACC_READY, leader=leader)
                    _tcgen_commit(arena, _BAR_V_MMA_DONE, v_stage, leader=leader)

                    _wait_barrier(arena, _BAR_DYP_INP_READY, 0, dyp_inp_ready.phase)
                    dyp_inp_ready.advance()
                    mma_ts(
                        _TM_DSTATE,
                        _TM_INP1,
                        desc_k_trans,
                        idesc_m128_n128_b,
                        True,
                        n=128,
                        k_extent=64,
                        b_transpose=True,
                    )
                    _tcgen_commit(arena, _BAR_DSTATE_ACC_READY, dstate_stage, leader=leader)

                    with K.If(have_dstate), K.Then():
                        _wait_barrier(arena, _BAR_DK_SCALE_DONE, 0, dk_scale_done.phase)
                        dk_scale_done.advance()
                    _wait_barrier(arena, _BAR_DA_READY, 0, da_ready.phase)
                    da_ready.advance()
                    mma_ss(
                        _TM_DVDK,
                        desc_q_trans,
                        desc_da_trans,
                        idesc_m128_n64_ab,
                        have_dstate,
                        m=128,
                        n=64,
                        k_extent=64,
                        a_transpose=True,
                        b_transpose=True,
                    )
                    _tcgen_commit(arena, _BAR_DK_ATTN_READY, leader=leader)
                    _tcgen_commit(arena, _BAR_Q_MMA_DONE, q_stage, leader=leader)

                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(arena, _BAR_DQ_SCALE_DONE, 0, dq_scale_done.phase)
                        dq_scale_done.advance()
                    with K.If(chunk < first_state_chunk), K.Then():
                        _wait_barrier(arena, _BAR_DQ_TOTAL_DONE, 0, dq_total_done.phase)
                        dq_total_done.advance()
                    with K.If(chunk >= first_state_chunk), K.Then():
                        mma_ss(
                            _TM_DSTATE_INP,
                            desc_k_trans,
                            desc_da,
                            idesc_m128_n64_a,
                            True,
                            m=128,
                            n=64,
                            k_extent=64,
                            a_transpose=True,
                        )
                    with K.If(chunk < first_state_chunk), K.Then():
                        mma_ss(
                            _TM_DSTATE_INP,
                            desc_k_trans,
                            desc_da,
                            idesc_m128_n64_a,
                            False,
                            m=128,
                            n=64,
                            k_extent=64,
                            a_transpose=True,
                        )
                    _tcgen_commit(arena, _BAR_DQ_TOTAL_READY, leader=leader)
                    _tcgen_commit(arena, _BAR_A_DONE, a_stage, leader=leader)

                    with K.If(chunk >= first_state_chunk), K.Then():
                        mma_ss(
                            _TM_INP0,
                            desc_state,
                            desc_dv_kmaj,
                            idesc_m128_n64_a,
                            False,
                            m=128,
                            n=64,
                            k_extent=128,
                            a_transpose=True,
                        )
                        _tcgen_commit(arena, _BAR_DK_STATE_READY, leader=leader)
                        _tcgen_commit(arena, _BAR_STATE_MMA_DONE, state_stage, leader=leader)
                    _tcgen_commit(arena, _BAR_SDV_DONE, leader=leader)

                    _wait_barrier(arena, _BAR_DM_ACC_DONE, 0, dm_done.phase)
                    dm_done.advance()
                    _wait_barrier(arena, _BAR_DK_ATTN_DONE, 0, dk_attn_done.phase)
                    dk_attn_done.advance()
                    mma_ss(
                        _TM_DVDK,
                        desc_k_trans,
                        desc_dm_trans,
                        idesc_m128_n64_ab,
                        True,
                        m=128,
                        n=64,
                        k_extent=64,
                        a_transpose=True,
                        b_transpose=True,
                    )
                    mma_ss(
                        _TM_DVDK,
                        desc_k_trans,
                        desc_dm,
                        idesc_m128_n64_a,
                        True,
                        m=128,
                        n=64,
                        k_extent=64,
                        a_transpose=True,
                    )
                    _tcgen_commit(arena, _BAR_DK_TOTAL_READY, leader=leader)
                    _tcgen_commit(arena, _BAR_K_MMA_DONE, k_stage, leader=leader)
                sched_consume(sched_state, tile)

            _wait_barrier(arena, _BAR_DK_TOTAL_DONE, 0, dk_total_done.phase)
            _wait_barrier(arena, _BAR_TMEM_DONE, 0, 0)
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                K.cast(tmem_base, "uint32"), K.uint32(512)
            )

        # Warp 9: K -> Q -> V -> dO -> checkpoint TensorMap loads.
        with tma:
            q_state = K.PipelineState(1, phase=1)
            k_state = K.PipelineState(2, phase=1)
            v_state = K.PipelineState(1, phase=1)
            do_state = K.PipelineState(1, phase=1)
            checkpoint_state = K.PipelineState(1, phase=1)
            sched_state = K.PipelineState(2, phase=1)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                cend = item[5]
                head_q = head if q_ratio == 1 else _udiv_nonnegative(head, q_ratio)
                head_k = head if k_ratio == 1 else _udiv_nonnegative(head, k_ratio)
                head_v = head if v_ratio == 1 else _udiv_nonnegative(head, v_ratio)
                desc_q = _descriptor_slot(descriptor_workspace, n_desc, 0, batch)
                desc_k = _descriptor_slot(descriptor_workspace, n_desc, 1, batch)
                desc_v = _descriptor_slot(descriptor_workspace, n_desc, 2, batch)
                desc_do = _descriptor_slot(descriptor_workspace, n_desc, 3, batch)
                desc_checkpoint = _descriptor_slot(descriptor_workspace, n_desc, 4, batch)
                with K.If(_elected()), K.Then():
                    for descriptor in (desc_q, desc_k, desc_v, desc_do, desc_checkpoint):
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor)
                with K.serial(cend - wstart, unroll=False) as reverse_index:
                    chunk = cend - 1 - reverse_index
                    token = chunk * _BT
                    leader = K.cuda.elect_sync()

                    k_stage = K.local_scalar("int32", init=k_state.stage)
                    _wait_barrier(arena, _BAR_K_MMA_DONE, k_stage, k_state.phase)
                    _wait_barrier(arena, _BAR_K_CG0_DONE, k_stage, k_state.phase)
                    k_state.advance()
                    _expect_tx(arena, _BAR_K_READY, k_stage, 16_384, pred=leader)
                    for channel, byte_offset in ((0, 0), (64, 8192)):
                        K.ptx[_TMA_G2S_3D](
                            arena.ptr_to([_K_BASE + k_stage * 16_384 + byte_offset]),
                            desc_k,
                            K.int32(channel),
                            K.cast(head_k, "int32"),
                            K.cast(token, "int32"),
                            _barrier_ptr(arena, _BAR_K_READY, k_stage),
                            pred=leader,
                        )

                    q_stage = K.local_scalar("int32", init=q_state.stage)
                    _wait_barrier(arena, _BAR_Q_MMA_DONE, q_stage, q_state.phase)
                    _wait_barrier(arena, _BAR_Q_CG1_DONE, q_stage, q_state.phase)
                    q_state.advance()
                    _expect_tx(arena, _BAR_Q_READY, q_stage, 16_384, pred=leader)
                    for channel, byte_offset in ((0, 0), (64, 8192)):
                        K.ptx[_TMA_G2S_3D](
                            arena.ptr_to([_Q_BASE + byte_offset]),
                            desc_q,
                            K.int32(channel),
                            K.cast(head_q, "int32"),
                            K.cast(token, "int32"),
                            _barrier_ptr(arena, _BAR_Q_READY, q_stage),
                            pred=leader,
                        )

                    v_stage = K.local_scalar("int32", init=v_state.stage)
                    _wait_barrier(arena, _BAR_V_MMA_DONE, v_stage, v_state.phase)
                    v_state.advance()
                    _expect_tx(arena, _BAR_V_READY, v_stage, 16_384, pred=leader)
                    for channel, byte_offset in ((0, 0), (64, 8192)):
                        K.ptx[_TMA_G2S_3D](
                            arena.ptr_to([_V_U_BASE + byte_offset]),
                            desc_v,
                            K.int32(channel),
                            K.cast(head_v, "int32"),
                            K.cast(token, "int32"),
                            _barrier_ptr(arena, _BAR_V_READY, v_stage),
                            pred=leader,
                        )

                    do_stage = K.local_scalar("int32", init=do_state.stage)
                    _wait_barrier(arena, _BAR_DO_MMA_DONE, do_stage, do_state.phase)
                    do_state.advance()
                    _expect_tx(arena, _BAR_DO_READY, do_stage, 16_384, pred=leader)
                    for channel, byte_offset in ((0, 0), (64, 8192)):
                        K.ptx[_TMA_G2S_3D](
                            arena.ptr_to([_DO_BASE + byte_offset]),
                            desc_do,
                            K.int32(channel),
                            K.cast(head, "int32"),
                            K.cast(token, "int32"),
                            _barrier_ptr(arena, _BAR_DO_READY, do_stage),
                            pred=leader,
                        )

                    with K.If(chunk >= first_state_chunk), K.Then():
                        checkpoint_stage = K.local_scalar("int32", init=checkpoint_state.stage)
                        _wait_barrier(
                            arena, _BAR_STATE_MMA_DONE, checkpoint_stage, checkpoint_state.phase
                        )
                        checkpoint_state.advance()
                        _expect_tx(arena, _BAR_STATE_READY, checkpoint_stage, 32_768, pred=leader)
                        for key_coord, byte_offset in ((0, 0), (64, 16_384)):
                            K.ptx[_TMA_G2S_4D](
                                arena.ptr_to([_STATE_BASE + byte_offset]),
                                desc_checkpoint,
                                K.int32(key_coord),
                                K.int32(0),
                                K.cast(chunk, "int32"),
                                K.cast(head, "int32"),
                                _barrier_ptr(arena, _BAR_STATE_READY, checkpoint_stage),
                                pred=leader,
                            )
                sched_produce(sched_state, tile)

            for _ in range(1):
                _wait_barrier(arena, _BAR_Q_MMA_DONE, q_state.stage, q_state.phase)
                _wait_barrier(arena, _BAR_Q_CG1_DONE, q_state.stage, q_state.phase)
                q_state.advance()
            for _ in range(2):
                _wait_barrier(arena, _BAR_K_MMA_DONE, k_state.stage, k_state.phase)
                _wait_barrier(arena, _BAR_K_CG0_DONE, k_state.stage, k_state.phase)
                k_state.advance()
            for _ in range(1):
                _wait_barrier(arena, _BAR_V_MMA_DONE, v_state.stage, v_state.phase)
                v_state.advance()
            for _ in range(1):
                _wait_barrier(arena, _BAR_DO_MMA_DONE, do_state.stage, do_state.phase)
                do_state.advance()

        # Warp 10: first/next scalar prefetch, suffix fold, and in-place stores.
        with gate_beta:
            gate_load = K.PipelineState(2, phase=1)
            beta_load = K.PipelineState(2, phase=1)
            gate_store = K.PipelineState(2, phase=0)
            beta_store = K.PipelineState(2, phase=0)
            sched_state = K.PipelineState(2, phase=0)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cend = item[5]
                batch_begin = item[6]
                batch_end = item[7]
                count = cend - wstart
                write_end = K.if_then_else(
                    batch_begin + wend * _BT < batch_end, batch_begin + wend * _BT, batch_end
                )
                a_l2 = K.local_scalar("float32", init=K.float32(0.0))
                bias = K.local_scalar("float32", init=K.float32(0.0))
                if safe_gate:
                    with K.If(count > 0), K.Then():
                        amplitude = K.local_scalar("float32")
                        K.ptx.ld.global_.f32(amplitude, a_log.ptr_to([head]))
                        K.assign(
                            a_l2, -_exp2(amplitude * K.float32(_RCP_LN2)) * K.float32(_RCP_LN2)
                        )
                        K.ptx.ld.global_.f32(bias, dt_bias.ptr_to([head]))

                def prefetch(chunk):
                    chunk_offset = batch_begin + chunk * _BT
                    gate_stage = K.local_scalar("int32", init=gate_load.stage)
                    gate_load.advance()
                    gate_values = K.alloc_local((2,), "float32")
                    for column in range(2):
                        position = lane + column * 32
                        token = chunk_offset + position
                        neutral = 0.0 if log_gate else 1.0
                        K.assign(gate_values[column], K.float32(neutral))
                        with K.If(token < batch_end), K.Then():
                            K.ptx.ld.global_.f32(
                                gate_values[column],
                                gate.ptr_to([K.cast(token, "int64") * n_heads_out + head]),
                            )
                    if safe_gate:
                        for column in range(2):
                            position = lane + column * 32
                            token = chunk_offset + position
                            transformed = a_l2 * _softplus(gate_values[column] + bias)
                            K.assign(
                                gate_values[column],
                                K.if_then_else(token < batch_end, transformed, K.float32(0.0)),
                            )
                    elif log_gate:
                        for column in range(2):
                            K.assign(gate_values[column], gate_values[column] * K.float32(_RCP_LN2))
                    else:
                        for column in range(2):
                            K.assign(
                                gate_values[column], _lg2(gate_values[column] + K.float32(1e-10))
                            )
                    for distance in (1, 2, 4, 8, 16):
                        for column in range(2):
                            prior = _shfl_up_f32(gate_values[column], distance)
                            K.assign(
                                gate_values[column],
                                K.if_then_else(
                                    lane >= distance,
                                    gate_values[column] + prior,
                                    gate_values[column],
                                ),
                            )
                    K.assign(gate_values[1], gate_values[1] + _shfl_idx_f32(gate_values[0], 31, 31))
                    for column in range(2):
                        position = lane + column * 32
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + position * 4]),
                            gate_values[column],
                        )
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_CUMPROD_BASE + gate_stage * 256 + position * 4]),
                            _exp2(gate_values[column]),
                        )
                    _arrive_barrier(arena, _BAR_GATE_READY, gate_stage)

                    beta_stage = K.local_scalar("int32", init=beta_load.stage)
                    beta_load.advance()
                    if beta_sigmoid:
                        for column in range(2):
                            position = lane + column * 32
                            token = chunk_offset + position
                            beta_value = K.local_scalar("float32", init=K.float32(0.0))
                            with K.If(token < batch_end), K.Then():
                                bits = K.local_scalar("uint16")
                                K.ptx.ld.global_.b16(
                                    bits, beta.ptr_to([K.cast(token, "int64") * n_heads_out + head])
                                )
                                raw = K.local_scalar("float32")
                                if io_dtype == "float16":
                                    K.assign(raw, K.cast(K.reinterpret("float16", bits), "float32"))
                                else:
                                    K.ptx.cvt.f32.bf16(raw, bits)
                                rounded = _pack_io_pair(
                                    _tanh(raw * K.float32(0.5)) * K.float32(0.5) + K.float32(0.5),
                                    K.float32(0.0),
                                    io_dtype,
                                )
                                ignored = K.local_scalar("float32")
                                _unpack_io_pair(rounded, beta_value, ignored, io_dtype)
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_BETA_BASE + beta_stage * 256 + position * 4]),
                                beta_value,
                            )
                        _arrive_barrier(arena, _BAR_BETA_READY, beta_stage)
                    else:
                        for column in range(2):
                            position = lane + column * 32
                            token = chunk_offset + position
                            copy_bytes = K.if_then_else(token < batch_end, K.uint32(4), K.uint32(0))
                            K.ptx["cp.async.ca.shared.global"](
                                arena.ptr_to([_BETA_BASE + beta_stage * 256 + position * 4]),
                                beta.ptr_to([K.cast(token, "int64") * n_heads_out + head]),
                                K.uint32(4),
                                copy_bytes,
                            )
                        K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                            _barrier_ptr(arena, _BAR_BETA_READY, beta_stage)
                        )

                with K.If(count > 0), K.Then():
                    prefetch(cend - 1)
                with K.serial(count, unroll=False) as reverse_index:
                    with K.If(reverse_index + 1 < count), K.Then():
                        prefetch(cend - 2 - reverse_index)
                    chunk = cend - 1 - reverse_index
                    chunk_offset = batch_begin + chunk * _BT

                    gate_stage = K.local_scalar("int32", init=gate_store.stage)
                    _wait_barrier(arena, _BAR_GATE_DONE, gate_stage, gate_store.phase)
                    gate_store.advance()
                    gradient = K.alloc_local((2,), "float32")
                    for column in range(2):
                        position = lane + column * 32
                        K.ptx.ld.shared.f32(
                            gradient[column],
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_stage * 256 + position * 4]),
                        )
                    for distance in (1, 2, 4, 8, 16):
                        for column in range(2):
                            later = _shfl_down_f32(gradient[column], distance)
                            K.assign(
                                gradient[column],
                                K.if_then_else(
                                    lane < 32 - distance, gradient[column] + later, gradient[column]
                                ),
                            )
                    K.assign(gradient[0], gradient[0] + _shfl_idx_f32(gradient[1], 0, 0))
                    for column in range(2):
                        position = lane + column * 32
                        token = chunk_offset + position
                        with K.If(token < write_end), K.Then():
                            K.ptx.st.global_.f32(
                                dgate.ptr_to([K.cast(token, "int64") * n_heads_out + head]),
                                gradient[column],
                            )

                    beta_stage = K.local_scalar("int32", init=beta_store.stage)
                    _wait_barrier(arena, _BAR_BETA_DONE, beta_stage, beta_store.phase)
                    beta_store.advance()
                    for column in range(2):
                        position = lane + column * 32
                        token = chunk_offset + position
                        with K.If(token < write_end), K.Then():
                            value = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                value, arena.ptr_to([_BETA_BASE + beta_stage * 256 + position * 4])
                            )
                            if beta_sigmoid:
                                K.ptx.st.global_.b16(
                                    dbeta.ptr_to([K.cast(token, "int64") * n_heads_out + head]),
                                    K.reinterpret("uint16", K.cast(value, io_dtype)),
                                )
                            else:
                                K.ptx.st.global_.f32(
                                    dbeta.ptr_to([K.cast(token, "int64") * n_heads_out + head]),
                                    value,
                                )
                sched_consume(sched_state, tile)

        # Warp 11: direct current dV -> dQ -> dK TMA store ladder.
        with epilogue:
            dq_state = K.PipelineState(1, phase=0)
            dk_state = K.PipelineState(1, phase=0)
            dv_state = K.PipelineState(1, phase=0)
            sched_state = K.PipelineState(2, phase=0)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cend = item[5]
                desc_dq = _descriptor_slot(descriptor_workspace, n_desc, 5, batch)
                desc_dk = _descriptor_slot(descriptor_workspace, n_desc, 6, batch)
                desc_dv = _descriptor_slot(descriptor_workspace, n_desc, 7, batch)
                with K.If(_elected()), K.Then():
                    for descriptor in (desc_dv, desc_dq, desc_dk):
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor)
                with K.serial(cend - wstart, unroll=False) as reverse_index:
                    chunk = cend - 1 - reverse_index
                    token = chunk * _BT
                    writes = chunk < wend

                    dv_stage = K.local_scalar("int32", init=dv_state.stage)
                    _wait_barrier(arena, _BAR_DV_STG_READY, dv_stage, dv_state.phase)
                    dv_state.advance()
                    with K.If(writes), K.Then():
                        for channel, byte_offset in ((0, 0), (64, 8192)):
                            K.ptx[_TMA_S2G_3D](
                                desc_dv,
                                K.int32(channel),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                arena.ptr_to([_DV_DY_BASE + byte_offset]),
                            )
                        K.ptx.cp.async_.bulk.commit_group()

                    dq_stage = K.local_scalar("int32", init=dq_state.stage)
                    _wait_barrier(arena, _BAR_DQ_STG_READY, dq_stage, dq_state.phase)
                    dq_state.advance()
                    with K.If(writes), K.Then():
                        for channel, byte_offset in ((0, 0), (64, 8192)):
                            K.ptx[_TMA_S2G_3D](
                                desc_dq,
                                K.int32(channel),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                arena.ptr_to([_DQ_BASE + byte_offset]),
                            )
                        K.ptx.cp.async_.bulk.commit_group()

                    dk_stage = K.local_scalar("int32", init=dk_state.stage)
                    _wait_barrier(arena, _BAR_DK_STG_READY, dk_stage, dk_state.phase)
                    dk_state.advance()
                    with K.If(writes), K.Then():
                        for channel, byte_offset in ((0, 0), (64, 8192)):
                            K.ptx[_TMA_S2G_3D](
                                desc_dk,
                                K.int32(channel),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                arena.ptr_to([_DK_BASE + byte_offset]),
                            )
                        K.ptx.cp.async_.bulk.commit_group()

                    K.ptx.cp.async_.bulk.wait_group.read(2)
                    _arrive_barrier(arena, _BAR_DV_STG_DONE, dv_stage)
                    K.ptx.cp.async_.bulk.wait_group.read(1)
                    _arrive_barrier(arena, _BAR_DQ_STG_DONE, dq_stage)
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                    _arrive_barrier(arena, _BAR_DK_STG_DONE, dk_stage)
                sched_consume(sched_state, tile)

    return main


def _normalized_config(config):
    import math

    config = {key: value for key, value in config.items() if key != "label"}
    if "dtype" in config:
        dtype = config.pop("dtype")
        if "io_dtype" in config and config["io_dtype"] != dtype:
            raise ValueError("dtype and io_dtype disagree")
        config["io_dtype"] = dtype
    config.setdefault("seq_lens", (64,))
    config["seq_lens"] = tuple(int(value) for value in config["seq_lens"])
    config.setdefault("heads", 1)
    config.setdefault("q_heads", config["heads"])
    config.setdefault("k_heads", config["heads"])
    config.setdefault("v_heads", config["heads"])
    config.setdefault("io_dtype", "bfloat16")
    config.setdefault("cu_dtype", "int32")
    config.setdefault("num_sms", 152)
    config.setdefault("scale", 1.0 / (_DK**0.5))
    config.setdefault("use_initial_state", False)
    config.setdefault("use_dstate_in", False)
    config.setdefault("use_dstate0", False)
    config.setdefault("log_gate", True)
    config.setdefault("safe_gate", False)
    config.setdefault("beta_sigmoid", False)
    config.setdefault("dynamic_scheduler", False)
    config.setdefault("split", False)
    config.setdefault("run_order", False)
    config.setdefault("order_generate", False)
    config.setdefault("strong_decay", False)

    if config["io_dtype"] not in ("bfloat16", "float16"):
        raise ValueError("io_dtype must be 'bfloat16' or 'float16'")
    if config["cu_dtype"] not in ("int32", "int64"):
        raise ValueError("cu_dtype must be 'int32' or 'int64'")
    if any(length < 0 for length in config["seq_lens"]):
        raise ValueError("sequence lengths must be nonnegative")
    if sum(config["seq_lens"]) == 0:
        raise ValueError("at least one sequence must be nonempty")
    heads = int(config["heads"])
    if heads <= 0:
        raise ValueError("heads must be positive")
    for name in ("q_heads", "k_heads", "v_heads"):
        value = int(config[name])
        if value <= 0 or heads % value:
            raise ValueError(f"{name}={value} must be a positive divisor of heads={heads}")
    if heads != max(int(config["q_heads"]), int(config["v_heads"])):
        raise ValueError("heads must equal max(q_heads, v_heads)")
    if int(config["k_heads"]) not in (int(config["q_heads"]), int(config["v_heads"])):
        raise ValueError("k_heads must equal q_heads or v_heads")
    if config["safe_gate"] and not config["log_gate"]:
        # safe_gate directly constructs ln(alpha); the source specializes its
        # log_gate flag independently, but this pairing is the public mode.
        config["log_gate"] = True
    if config["use_dstate0"] and not config["use_initial_state"]:
        raise ValueError("use_dstate0 requires use_initial_state")
    if config["order_generate"] and not config["run_order"]:
        raise ValueError("order_generate requires run_order")
    if config["split"] and (not config["run_order"] or config["order_generate"]):
        raise ValueError("split work rows require caller-scratch ordering")
    if config["dynamic_scheduler"] and not config["run_order"]:
        # The order prologue owns reset of the dynamic ticket counter.  Keep
        # replay correct even when the shorthand config names only the
        # scheduler specialization.
        config["run_order"] = True
        config["order_generate"] = True
    scale = float(config["scale"])
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("scale must be finite and nonzero")
    if int(config["num_sms"]) <= 0:
        raise ValueError("num_sms must be positive")
    return config


def _work_rows(seq_lens, heads, *, split):
    rows = []
    token_begin = 0
    for batch, sequence_length in enumerate(seq_lens):
        chunks = (sequence_length + _BT - 1) // _BT
        token_end = token_begin + sequence_length
        for head in range(heads):
            if split and chunks >= 4:
                midpoint = chunks // 2
                rows.append((batch, head, 0, midpoint, 0, midpoint, token_begin, token_end))
                rows.append((batch, head, midpoint, chunks, 0, chunks, token_begin, token_end))
            else:
                rows.append((batch, head, 0, chunks, 0, chunks, token_begin, token_end))
        token_begin = token_end
    return rows


def get_kernel(**config):
    """Return the source-ordered TensorMap prologue and persistent main."""
    config = _normalized_config(config)
    prologue = _make_prologue(
        run_order=bool(config["run_order"]),
        order_generate=bool(config["order_generate"]),
        dynamic_scheduler=bool(config["dynamic_scheduler"]),
        n_heads_out=int(config["heads"]),
        cu_dtype=config["cu_dtype"],
    )
    main = _make_main(
        num_sms=int(config["num_sms"]),
        io_dtype=config["io_dtype"],
        cu_dtype=config["cu_dtype"],
        use_initial_state=bool(config["use_initial_state"]),
        use_dstate_in=bool(config["use_dstate_in"]),
        use_dstate0=bool(config["use_dstate0"]),
        log_gate=bool(config["log_gate"]),
        safe_gate=bool(config["safe_gate"]),
        beta_sigmoid=bool(config["beta_sigmoid"]),
        dynamic_scheduler=bool(config["dynamic_scheduler"]),
        q_ratio=int(config["heads"]) // int(config["q_heads"]),
        k_ratio=int(config["heads"]) // int(config["k_heads"]),
        v_ratio=int(config["heads"]) // int(config["v_heads"]),
        n_heads_out=int(config["heads"]),
    )
    return [prologue.func, main.func]


def _aligned_i64(torch, size, alignment=128):
    if size % 8:
        raise ValueError(f"workspace size must be divisible by 8, got {size}")
    owner = torch.empty(size + alignment - 1, dtype=torch.uint8, device="cuda")
    offset = (-owner.data_ptr()) % alignment
    view = owner[offset : offset + size].view(torch.int64)
    if view.data_ptr() % alignment:
        raise AssertionError("failed to construct an aligned TensorMap workspace")
    return owner, view


def _checkpoint_count(seq_lens):
    return sum(0 if length == 0 else (length - 1) // _BT + 1 for length in seq_lens)


def _prepare_work_tables(torch, config):
    base = torch.tensor(
        _work_rows(config["seq_lens"], config["heads"], split=config["split"]),
        dtype=torch.int32,
        device="cuda",
    )

    def one_side():
        work_items = torch.empty_like(base) if config["run_order"] else base.clone()
        staging = None
        if config["run_order"] and not config["order_generate"]:
            staging = base.clone()
        sched_all = torch.zeros(4, dtype=torch.int32, device="cuda")
        return {
            "work_items": work_items,
            "work_count": torch.tensor([base.shape[0]], dtype=torch.int32, device="cuda"),
            "staging": staging,
            "sched_all": sched_all,
            "scheduler": sched_all[:2] if config["dynamic_scheduler"] else None,
        }

    return {"tirx": one_side(), "source": one_side()}


def _guarded_full(torch, shape, value, *, dtype):
    elements = 1
    for extent in shape:
        elements *= int(extent)
    guard_elements = 256 // torch.empty((), dtype=dtype).element_size()
    sentinel = 37.0
    owner = torch.full((elements + 2 * guard_elements,), sentinel, dtype=dtype, device="cuda")
    tensor = owner[guard_elements : guard_elements + elements].view(shape)
    tensor.fill_(value)
    return tensor, (owner, guard_elements, elements, sentinel)


def _new_outputs(torch, config, total_tokens):
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    beta_t = io_t if config["beta_sigmoid"] else torch.float32
    specifications = {
        "dq": ((total_tokens, config["heads"], _DK), io_t),
        "dk": ((total_tokens, config["heads"], _DK), io_t),
        "dv": ((total_tokens, config["heads"], _DV), io_t),
        "dgate": ((total_tokens, config["heads"]), torch.float32),
        "dbeta": ((total_tokens, config["heads"]), beta_t),
    }
    if config["use_dstate0"]:
        specifications["d_initial_state"] = (
            (len(config["seq_lens"]), config["heads"], _DV, _DK),
            torch.float32,
        )
    outputs = {}
    guards = {}
    for name, (shape, dtype) in specifications.items():
        outputs[name], guards[name] = _guarded_full(torch, shape, float("nan"), dtype=dtype)
    return outputs, guards


def _load_recompute_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module("cudnn.linear_attention.frost.kernel.gdn_recompute_f16")


def _prepare_state_checkpoints(data):
    """Run the pinned state-only forward before either timed backward path."""
    import torch

    config = data["config"]
    source = _load_recompute_source()
    rows = torch.empty(
        (len(config["seq_lens"]) * config["heads"], 8), dtype=torch.int32, device="cuda"
    )
    count = torch.tensor([rows.shape[0]], dtype=torch.int32, device="cuda")
    sched_all = torch.zeros(4, dtype=torch.int32, device="cuda")
    stream = int(torch.cuda.current_stream().cuda_stream)
    source.chunk_gdn_recompute_sm100(
        data["k"],
        data["v"],
        data["gate"],
        data["beta"],
        data["cu_seqlens"],
        data["initial_state"],
        None,
        checkpoint_every_n_tokens=_BT,
        output_state_checkpoints=data["checkpoints"],
        work_items=rows,
        work_count=count,
        sched_ctr=None,
        sched_all=sched_all,
        work_item_scratch=None,
        order_in_prologue=True,
        log_gate=bool(config["log_gate"]),
        safe_gate=bool(config["safe_gate"]),
        a_log=data["a_log"],
        dt_bias=data["dt_bias"],
        use_beta_sigmoid=bool(config["beta_sigmoid"]),
        workspace=data["checkpoint_workspace"],
        stream=stream,
    )
    data["_checkpoint_keep_alive"] = (rows, count, sched_all)


def _prepare_data(config):
    import torch

    config = _normalized_config(config)
    torch.manual_seed(20260831)
    total_tokens = sum(config["seq_lens"])
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    q = (0.2 * torch.randn(total_tokens, config["q_heads"], _DK, device="cuda")).to(io_t)
    k = (0.2 * torch.randn(total_tokens, config["k_heads"], _DK, device="cuda")).to(io_t)
    v = (0.2 * torch.randn(total_tokens, config["v_heads"], _DV, device="cuda")).to(io_t)
    gate = torch.empty(total_tokens, config["heads"], dtype=torch.float32, device="cuda")
    if config["safe_gate"]:
        gate.normal_(std=0.2)
    elif config["log_gate"]:
        if config["strong_decay"]:
            gate.uniform_(-5.0, -1.5)
        else:
            gate.uniform_(-0.9, -0.01)
    elif config["strong_decay"]:
        gate.uniform_(0.01, 0.25)
    else:
        gate.uniform_(0.4, 0.99)
    if config["beta_sigmoid"]:
        beta = (0.3 * torch.randn(total_tokens, config["heads"], device="cuda")).to(io_t)
    else:
        beta = torch.sigmoid(0.3 * torch.randn(total_tokens, config["heads"], device="cuda"))
    do = (0.2 * torch.randn(total_tokens, config["heads"], _DV, device="cuda")).to(io_t)
    cu_t = torch.int64 if config["cu_dtype"] == "int64" else torch.int32
    endpoints = [0]
    for length in config["seq_lens"]:
        endpoints.append(endpoints[-1] + length)
    cu_seqlens = torch.tensor(endpoints, dtype=cu_t, device="cuda")
    a_log = (
        0.1 * torch.randn(config["heads"], dtype=torch.float32, device="cuda")
        if config["safe_gate"]
        else None
    )
    dt_bias = (
        -4.0 + 0.1 * torch.randn(config["heads"], dtype=torch.float32, device="cuda")
        if config["safe_gate"]
        else None
    )
    initial_state = (
        0.01
        * torch.randn(
            len(config["seq_lens"]), config["heads"], _DV, _DK, dtype=torch.float32, device="cuda"
        )
        if config["use_initial_state"]
        else None
    )
    d_final_state = (
        0.02
        * torch.randn(
            len(config["seq_lens"]), config["heads"], _DV, _DK, dtype=torch.float32, device="cuda"
        )
        if config["use_dstate_in"]
        else None
    )
    checkpoint_rows = max(_checkpoint_count(config["seq_lens"]), 1)
    checkpoints = torch.full(
        (checkpoint_rows, config["heads"], _DV, _DK), float("nan"), dtype=io_t, device="cuda"
    )
    tirx_outputs, tirx_guards = _new_outputs(torch, config, total_tokens)
    source_outputs, source_guards = _new_outputs(torch, config, total_tokens)
    workspace_bytes = _TENSOR_MAP_BYTES * (_TENSOR_MAP_ARRAYS * len(config["seq_lens"]) + 1)
    tirx_owner, tirx_workspace = _aligned_i64(torch, workspace_bytes)
    source_owner, source_workspace = _aligned_i64(torch, workspace_bytes)
    checkpoint_owner, checkpoint_workspace = _aligned_i64(torch, workspace_bytes)
    data = {
        "config": config,
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "do": do,
        "cu_seqlens": cu_seqlens,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "d_final_state": d_final_state,
        "checkpoints": checkpoints,
        "tirx": tirx_outputs,
        "source": source_outputs,
        "guards": {"tirx": tirx_guards, "source": source_guards},
        "work": _prepare_work_tables(torch, config),
        "tirx_workspace": tirx_workspace,
        "source_workspace": source_workspace,
        "checkpoint_workspace": checkpoint_workspace,
        "_workspace_owners": (tirx_owner, source_owner, checkpoint_owner),
    }
    _prepare_state_checkpoints(data)
    return data


def prepare_data(**config):
    """Allocate inputs/outputs and construct per-chunk entering states."""
    return _prepare_data(config)


def _encode_tiled_map(tensor, dimensions, strides, box):
    import torch
    from cuda.bindings import driver as cuda

    if tensor.dtype == torch.float16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    elif tensor.dtype == torch.bfloat16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    else:
        raise TypeError(f"unsupported TensorMap dtype {tensor.dtype}")
    u64 = cuda.cuuint64_t
    u32 = cuda.cuuint32_t
    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        data_type,
        len(dimensions),
        tensor.data_ptr(),
        [u64(int(value)) for value in dimensions],
        [u64(int(value)) for value in strides],
        [u32(int(value)) for value in box],
        [u32(1) for _ in dimensions],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with {result}")
    return (
        torch.tensor([int(word) for word in tensor_map.opaque], dtype=torch.uint64)
        .view(torch.int64)
        .to(device=tensor.device)
    )


def _base_tensor_maps(data, side):
    config = data["config"]
    total_tokens = data["q"].shape[0]

    def headed(tensor, channels, heads):
        return _encode_tiled_map(
            tensor,
            (channels, heads, total_tokens),
            (tensor.stride(1) * tensor.element_size(), tensor.stride(0) * tensor.element_size()),
            (64, 1, _BT),
        )

    output = data[side]
    checkpoint = data["checkpoints"]
    return [
        headed(data["q"], _DK, config["q_heads"]),
        headed(data["k"], _DK, config["k_heads"]),
        headed(data["v"], _DV, config["v_heads"]),
        headed(data["do"], _DV, config["heads"]),
        _encode_tiled_map(
            checkpoint,
            (_DV, _DK, checkpoint.shape[0], config["heads"]),
            (
                checkpoint.stride(2) * checkpoint.element_size(),
                checkpoint.stride(0) * checkpoint.element_size(),
                checkpoint.stride(1) * checkpoint.element_size(),
            ),
            (64, _DK, 1, 1),
        ),
        headed(output["dq"], _DK, config["heads"]),
        headed(output["dk"], _DK, config["heads"]),
        headed(output["dv"], _DV, config["heads"]),
    ]


def _tirx_launch(executables, data):
    import torch

    config = data["config"]
    prologue, main = executables
    maps = _base_tensor_maps(data, "tirx")
    work = data["work"]["tirx"]
    output = data["tirx"]
    dummy_i32 = torch.zeros(8, dtype=torch.int32, device="cuda")
    dummy_f32 = torch.zeros(max(config["heads"], 1), dtype=torch.float32, device="cuda")

    def launch(*, prologue_only=False):
        prologue(
            *maps,
            data["tirx_workspace"],
            data["cu_seqlens"],
            data["q"].view(torch.uint8).reshape(-1),
            data["k"].view(torch.uint8).reshape(-1),
            data["v"].view(torch.uint8).reshape(-1),
            data["do"].view(torch.uint8).reshape(-1),
            data["checkpoints"].view(torch.uint8).reshape(-1),
            output["dq"].view(torch.uint8).reshape(-1),
            output["dk"].view(torch.uint8).reshape(-1),
            output["dv"].view(torch.uint8).reshape(-1),
            work["staging"].reshape(-1) if work["staging"] is not None else dummy_i32,
            work["work_count"],
            work["work_items"].reshape(-1),
            work["sched_all"],
            len(config["seq_lens"]),
            data["q"].stride(0) * data["q"].element_size(),
            data["k"].stride(0) * data["k"].element_size(),
            data["v"].stride(0) * data["v"].element_size(),
            data["do"].stride(0) * data["do"].element_size(),
            data["checkpoints"].stride(0) * data["checkpoints"].element_size(),
            output["dq"].stride(0) * output["dq"].element_size(),
            output["dk"].stride(0) * output["dk"].element_size(),
            output["dv"].stride(0) * output["dv"].element_size(),
            _BT,
        )
        if prologue_only:
            return
        main(
            data["tirx_workspace"],
            len(config["seq_lens"]),
            data["gate"].reshape(-1),
            data["a_log"] if data["a_log"] is not None else dummy_f32,
            data["dt_bias"] if data["dt_bias"] is not None else dummy_f32,
            data["beta"].reshape(-1),
            output["dgate"].reshape(-1),
            output["dbeta"].reshape(-1),
            data["cu_seqlens"],
            output.get("d_initial_state", dummy_f32).reshape(-1),
            (data["d_final_state"].reshape(-1) if data["d_final_state"] is not None else dummy_f32),
            work["work_items"].reshape(-1),
            work["work_count"],
            work["scheduler"] if work["scheduler"] is not None else dummy_i32,
            float(config["scale"]),
        )

    launch._keep_alive = (maps, dummy_i32, dummy_f32)
    return launch


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module("cudnn.linear_attention.frost.kernel.gdn_bprop_f16")


def _source_launch(data):
    import torch

    source = _load_reference_source()
    config = data["config"]
    output = data["source"]
    work = data["work"]["source"]
    stream = int(torch.cuda.current_stream().cuda_stream)

    def launch():
        source.chunk_gdn_bwd_sm100(
            data["q"],
            data["k"],
            data["v"],
            data["gate"],
            data["beta"],
            data["do"],
            data["checkpoints"],
            output["dq"],
            output["dk"],
            output["dv"],
            output["dgate"],
            output["dbeta"],
            data["cu_seqlens"],
            float(config["scale"]),
            use_initial_state=bool(config["use_initial_state"]),
            d_initial_state=output.get("d_initial_state"),
            d_final_state=data["d_final_state"],
            work_items=work["work_items"],
            work_count=work["work_count"],
            sched_ctr=work["scheduler"],
            sched_all=work["sched_all"],
            work_item_scratch=work["staging"],
            order_in_prologue=bool(config["run_order"]),
            log_gate=bool(config["log_gate"]),
            safe_gate=bool(config["safe_gate"]),
            a_log=data["a_log"],
            dt_bias=data["dt_bias"],
            use_beta_sigmoid=bool(config["beta_sigmoid"]),
            workspace=data["source_workspace"],
            stream=stream,
        )

    return launch


def _check_redzones(data, side):
    import torch

    for name, (owner, guard, elements, sentinel) in data["guards"][side].items():
        prefix = owner[:guard]
        suffix = owner[guard + elements :]
        if not bool(torch.all(prefix == sentinel)) or not bool(torch.all(suffix == sentinel)):
            raise AssertionError(f"{side}.{name} wrote outside its output allocation")


def _assert_storage_equal(actual, expected, *, name):
    import torch

    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.device != expected.device
        or actual.layout != expected.layout
        or actual.stride() != expected.stride()
    ):
        raise AssertionError(f"{name} storage contract changed")
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError(f"{name} is not bitwise identical")


def _assert_logical_inputs_unchanged(actual, pristine):
    names = (
        "q",
        "k",
        "v",
        "gate",
        "beta",
        "do",
        "cu_seqlens",
        "a_log",
        "dt_bias",
        "initial_state",
        "d_final_state",
        "checkpoints",
    )
    for name in names:
        actual_tensor = actual[name]
        pristine_tensor = pristine[name]
        if actual_tensor is None or pristine_tensor is None:
            if actual_tensor is not pristine_tensor:
                raise AssertionError(f"logical input {name} optionality changed")
            continue
        _assert_storage_equal(actual_tensor, pristine_tensor, name=f"input.{name}")


def _validate_outputs(data, *, sources):
    import torch

    config = data["config"]
    expected_dtypes = {
        "dq": torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16,
        "dk": torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16,
        "dv": torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16,
        "dgate": torch.float32,
        "dbeta": (
            torch.float16
            if config["io_dtype"] == "float16" and config["beta_sigmoid"]
            else torch.bfloat16
            if config["io_dtype"] == "bfloat16" and config["beta_sigmoid"]
            else torch.float32
        ),
    }
    if config["use_dstate0"]:
        expected_dtypes["d_initial_state"] = torch.float32
    for side in sources:
        _check_redzones(data, side)
        if set(data[side]) != set(expected_dtypes):
            raise AssertionError(f"{side} output names do not match the backward ABI")
        for name, tensor in data[side].items():
            if tensor.dtype != expected_dtypes[name]:
                raise AssertionError(
                    f"{side}.{name} dtype {tensor.dtype} != {expected_dtypes[name]}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"{side}.{name} contains NaN or Inf")
    if "tirx" in sources and "source" in sources:
        for name, actual in data["tirx"].items():
            expected = data["source"][name]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError(f"tirx/source contract mismatch for {name}")
            if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
                raise AssertionError(f"tirx/source NaN classification mismatch for {name}")
            if not torch.equal(torch.isinf(actual), torch.isinf(expected)):
                raise AssertionError(f"tirx/source Inf classification mismatch for {name}")
            if actual.dtype == torch.bfloat16:
                rtol, atol = 2.0**-7, 2.0**-10
            elif actual.dtype == torch.float16:
                rtol, atol = 2.0**-10, 2.0**-14
            else:
                rtol, atol = 2.0**-16, 2.0**-20
            torch.testing.assert_close(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                check_device=True,
                check_dtype=True,
                check_layout=True,
                check_stride=True,
            )


def run_test(**config):
    """Compare TIRx with the pinned source and exact replay."""
    import torch

    from tirx_kernels.runner import compile_kernel

    config = _normalized_config(config)
    data = _prepare_data(config)
    executables = [compile_kernel(func) for func in get_kernel(**config)]
    tirx_launch = _tirx_launch(executables, data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    _validate_outputs(data, sources=("tirx", "source"))

    replay_data = _prepare_data(config)
    _assert_logical_inputs_unchanged(data, replay_data)
    replay_launch = _tirx_launch(executables, replay_data)
    replay_launch()
    torch.cuda.synchronize()
    _validate_outputs(replay_data, sources=("tirx",))
    for name in data["tirx"]:
        _assert_storage_equal(
            data["tirx"][name], replay_data["tirx"][name], name=f"tirx replay.{name}"
        )
    return {"tokens": sum(config["seq_lens"]), "heads": config["heads"]}


def prepare_bench(**config):
    """Compile the two TIRx launches without importing torch or touching CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config = _normalized_config(config)
    state = {
        "config": config,
        "executables": [compile_kernel(func) for func in get_kernel(**config)],
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Prepare checkpoints, validate once, then time only prologue plus main."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = _normalized_config({**prepared["config"], **kwargs})
    data = _prepare_data(config)
    tirx_launch = _tirx_launch(prepared["executables"], data)
    tirx_launch()
    torch.cuda.synchronize()
    references = None
    if external_references_enabled():

        def source_builder():
            source_launch = _source_launch(data)
            source_launch()
            torch.cuda.synchronize()
            _validate_outputs(data, sources=("tirx", "source"))
            return source_launch

        references = {"cudnn_frontend": source_builder}
    else:
        _validate_outputs(data, sources=("tirx",))
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
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
