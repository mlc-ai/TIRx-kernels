# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell BF16 Gated Delta Net v2 backward kernel.

Upstream source:
``python/cudnn/linear_attention/frost/kernel/gdn2_bprop_f16.py``
(``prologue_kernel``, ``kernel``, and the two-launch ``run_bwd`` entry).

GDN-2 carries a per-key-channel erase gate ``beta`` and a per-value-channel
write gate ``w``, so ``K_decay`` folds beta into the operand, ``Y`` uses
``w * v``, and the kernel emits ``dw`` alongside a per-channel ``dbeta``.
"""

import tirx_kernels.kern as K

KERNEL_META = {"name": "cudnn_sm100_gdn2_bprop_f16", "category": "cudnn", "compute_capability": 10}

CONFIGS = [
    {"label": "basic", "seq_lens": (64,), "heads": 1},
    {"label": "tail", "seq_lens": (17, 0, 33), "heads": 2},
    {"label": "grouped", "seq_lens": (64,), "heads": 4, "q_heads": 4, "k_heads": 1, "v_heads": 1},
    {"label": "l2norm", "seq_lens": (128,), "heads": 2, "l2norm": True},
    {"label": "safe_gate", "seq_lens": (64,), "heads": 2, "safe_gate": True},
    {"label": "beta_sigmoid", "seq_lens": (128,), "heads": 2, "beta_sigmoid": True},
    {
        "label": "state",
        "seq_lens": (32, 48),
        "heads": 2,
        "use_initial_state": True,
        "use_dstate_in": True,
        "use_dstate0": True,
    },
    {"label": "dynamic", "seq_lens": (32, 48), "heads": 2, "dynamic_scheduler": True},
    {"label": "order_scratch", "seq_lens": (32, 48), "heads": 2, "run_order": True},
    {
        "label": "order_generate",
        "seq_lens": (32, 48),
        "heads": 2,
        "run_order": True,
        "order_generate": True,
    },
]

# The performance matrix spans every accepted specialization at production scale
# plus the upstream benchmark's batch and sequence-length sweep, which runs at 64
# heads with the fused q/k L2 norm enabled. Checkpoint construction stays outside
# timing on both sides.
BENCH_CONFIGS = [
    {"label": "perf_basic_b1_s2048_h16", "seq_lens": (2048,), "heads": 16},
    {"label": "perf_tail_b2_ragged_h16", "seq_lens": (2047, 4093), "heads": 16},
    {
        "label": "perf_grouped_b1_s8192_hq64_hk16_hv16",
        "seq_lens": (8192,),
        "heads": 64,
        "q_heads": 64,
        "k_heads": 16,
        "v_heads": 16,
    },
    {"label": "perf_l2_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "l2norm": True},
    {"label": "perf_l2_b4_s8192_h64", "seq_lens": (8192,) * 4, "heads": 64, "l2norm": True},
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
    # Upstream benchmark sweep: sequence length at batch 4, then batch at 8192.
    {"label": "perf_l2_b4_s2048_h64", "seq_lens": (2048,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b4_s4096_h64", "seq_lens": (4096,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b4_s16384_h64", "seq_lens": (16384,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b4_s32768_h64", "seq_lens": (32768,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b1_s8192_h64", "seq_lens": (8192,), "heads": 64, "l2norm": True},
    {"label": "perf_l2_b2_s8192_h64", "seq_lens": (8192,) * 2, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b8_s8192_h64", "seq_lens": (8192,) * 8, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b16_s8192_h64", "seq_lens": (8192,) * 16, "heads": 64, "l2norm": True},
]


_BT = 16
_DK = 128
_DV = 128
_TENSOR_MAP_BYTES = 128
_TENSOR_MAP_WORDS = 16
_TENSOR_MAP_ARRAYS = 14
_MAIN_SMEM_BYTES = 225_280
_TRY_WAIT_TICKS = 10_000_000
_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)

# (ready byte, done byte, stages, ready arrivals, done arrivals, init warp).
# This is the frozen 115-word protocol table; None denotes a one-way edge.
# Frozen mbarrier protocol: 70 objects over 123 physical words (984 bytes).
# Each row is (byte offset, unused pair slot, stages, arrival count, unused,
# initializing warp).  Arrival counts encode the reader set: 128 per compute
# group, 32 for a single warp, 1 for a tcgen05 commit, 256 where two groups
# read the same tile, and 15 on the scheduler's done edge.
_PROTOCOL = (
    (0, None, 2, 1, None, 14),  # q_ready
    (16, None, 2, 128, None, 14),  # q_done
    (32, None, 2, 1, None, 14),  # k_ready
    (48, None, 2, 128, None, 14),  # k_done
    (64, None, 2, 1, None, 14),  # gate_ready
    (80, None, 2, 256, None, 14),  # gate_done
    (96, None, 2, 1, None, 14),  # beta_ready
    (112, None, 2, 256, None, 14),  # beta_done
    (128, None, 2, 1, None, 14),  # do_ready
    (144, None, 2, 32, None, 14),  # do_done
    (160, None, 2, 1, None, 14),  # do_mma_done
    (176, None, 2, 1, None, 14),  # v_ready
    (192, None, 2, 128, None, 14),  # v_done
    (208, None, 2, 1, None, 14),  # w_ready
    (224, None, 2, 128, None, 14),  # w_done
    (240, None, 1, 1, None, 14),  # state_ready
    (248, None, 1, 1, None, 14),  # state_done
    (256, None, 1, 128, None, 14),  # state_cg0_done
    (264, None, 2, 128, None, 14),  # state_inp_ready
    (280, None, 2, 1, None, 14),  # state_inp_done
    (296, None, 2, 128, None, 14),  # state_inp_cg2_done
    (312, None, 2, 128, None, 12),  # k_decay_inv_ready
    (328, None, 2, 128, None, 12),  # q_decay_k_restore_ready
    (344, None, 2, 1, None, 12),  # decay_done
    (360, None, 2, 32, None, 12),  # t_inv_ready
    (376, None, 2, 32, None, 12),  # a_ready
    (392, None, 2, 32, None, 12),  # da_ready
    (408, None, 2, 32, None, 12),  # dm_ready
    (424, None, 2, 1, None, 12),  # a_done
    (440, None, 2, 1, None, 12),  # t_inv_done
    (456, None, 2, 1, None, 12),  # da_done
    (472, None, 2, 1, None, 12),  # dm_done
    (488, None, 1, 1, None, 13),  # state_k_acc_ready
    (496, None, 1, 128, None, 13),  # y_inp_ready
    (504, None, 1, 1, None, 13),  # u_acc_ready
    (512, None, 1, 128, None, 13),  # u_smem_ready
    (520, None, 1, 1, None, 13),  # du_acc_ready
    (528, None, 1, 128, None, 13),  # du_inp_ready
    (536, None, 1, 1, None, 13),  # dy_acc_ready
    (544, None, 1, 128, None, 13),  # neg_dy_inp_ready
    (552, None, 1, 128, None, 13),  # dy_smem_ready
    (560, None, 1, 1, None, 13),  # dy_smem_done
    (568, None, 1, 1, None, 13),  # dstate_acc_ready
    (576, None, 1, 128, None, 13),  # dstate_inp_ready
    (584, None, 1, 128, None, 13),  # dstate_smem_ready
    (592, None, 1, 1, None, 13),  # dstate_smem_done
    (600, None, 1, 128, None, 13),  # dstate_smem_cg2_done
    (608, None, 1, 1, None, 13),  # dq_acc_ready
    (616, None, 1, 1, None, 13),  # dk_decay_part_acc_ready
    (624, None, 1, 1, None, 13),  # dk_inv_part_acc_ready
    (632, None, 1, 1, None, 13),  # dk_restore_part_acc_ready
    (640, None, 1, 128, None, 13),  # dqk_acc_done
    (648, None, 4, 128, None, 12),  # qk_raw_ready
    (680, None, 4, 128, None, 12),  # qk_raw_done
    (712, None, 1, 128, None, 15),  # dq_tmastg_ready
    (720, None, 1, 32, None, 15),  # dq_tmastg_done
    (728, None, 1, 128, None, 15),  # dk_tmastg_ready
    (736, None, 1, 32, None, 15),  # dk_tmastg_done
    (744, None, 2, 128, None, 15),  # dv_tmastg_ready
    (760, None, 2, 32, None, 15),  # dv_tmastg_done
    (776, None, 1, 128, None, 15),  # dgate_tmastg_ready
    (784, None, 1, 32, None, 15),  # dgate_tmastg_done
    (792, None, 1, 128, None, 15),  # db_tmastg_ready
    (800, None, 1, 32, None, 15),  # db_tmastg_done
    (808, None, 2, 128, None, 15),  # dwo_tmastg_ready
    (824, None, 2, 32, None, 15),  # dwo_tmastg_done
    (840, None, 1, 128, None, 13),  # dstate0_acc_stored
    (848, None, 1, 256, None, 13),  # tmem_done
    (856, None, 8, 1, None, 15),  # sched_ready
    (920, None, 8, 15, None, 15),  # sched_done
)

_BAR_Q_READY = 0
_BAR_Q_DONE = 16
_BAR_K_READY = 32
_BAR_K_DONE = 48
_BAR_GATE_READY = 64
_BAR_GATE_DONE = 80
_BAR_BETA_READY = 96
_BAR_BETA_DONE = 112
_BAR_DO_READY = 128
_BAR_DO_DONE = 144
_BAR_DO_MMA_DONE = 160
_BAR_V_READY = 176
_BAR_V_DONE = 192
_BAR_W_READY = 208
_BAR_W_DONE = 224
_BAR_STATE_READY = 240
_BAR_STATE_DONE = 248
_BAR_STATE_CG0_DONE = 256
_BAR_STATE_INP_READY = 264
_BAR_STATE_INP_DONE = 280
_BAR_STATE_INP_CG2_DONE = 296
_BAR_K_DECAY_INV_READY = 312
_BAR_Q_DECAY_K_RESTORE_READY = 328
_BAR_DECAY_DONE = 344
_BAR_T_INV_READY = 360
_BAR_A_READY = 376
_BAR_DA_READY = 392
_BAR_DM_READY = 408
_BAR_A_DONE = 424
_BAR_T_INV_DONE = 440
_BAR_DA_DONE = 456
_BAR_DM_DONE = 472
_BAR_STATE_K_ACC_READY = 488
_BAR_Y_INP_READY = 496
_BAR_U_ACC_READY = 504
_BAR_U_SMEM_READY = 512
_BAR_DU_ACC_READY = 520
_BAR_DU_INP_READY = 528
_BAR_DY_ACC_READY = 536
_BAR_NEG_DY_INP_READY = 544
_BAR_DY_SMEM_READY = 552
_BAR_DY_SMEM_DONE = 560
_BAR_DSTATE_ACC_READY = 568
_BAR_DSTATE_INP_READY = 576
_BAR_DSTATE_SMEM_READY = 584
_BAR_DSTATE_SMEM_DONE = 592
_BAR_DSTATE_SMEM_CG2_DONE = 600
_BAR_DQ_ACC_READY = 608
_BAR_DK_DECAY_PART_ACC_READY = 616
_BAR_DK_INV_PART_ACC_READY = 624
_BAR_DK_RESTORE_PART_ACC_READY = 632
_BAR_DQK_ACC_DONE = 640
_BAR_QK_RAW_READY = 648
_BAR_QK_RAW_DONE = 680
_BAR_DQ_TMASTG_READY = 712
_BAR_DQ_TMASTG_DONE = 720
_BAR_DK_TMASTG_READY = 728
_BAR_DK_TMASTG_DONE = 736
_BAR_DV_TMASTG_READY = 744
_BAR_DV_TMASTG_DONE = 760
_BAR_DGATE_TMASTG_READY = 776
_BAR_DGATE_TMASTG_DONE = 784
_BAR_DB_TMASTG_READY = 792
_BAR_DB_TMASTG_DONE = 800
_BAR_DWO_TMASTG_READY = 808
_BAR_DWO_TMASTG_DONE = 824
_BAR_DSTATE0_ACC_STORED = 840
_BAR_TMEM_DONE = 848
_BAR_SCHED_READY = 856
_BAR_SCHED_DONE = 920

# The single rank-1 arena follows the frozen source allocation order.
_SMEM_BARRIERS = 0
_SMEM_TMEM_MAILBOX = 984
_SMEM_SCHED = 992
_SMEM_RED1 = 1024
_SMEM_NORM = 1280
_SMEM_STATE = 2048
_SMEM_DSTATE = 34_816
_SMEM_K_DECAY = 67_584
_SMEM_K_INV = 75_776
_SMEM_K_RESTORE = 83_968
_SMEM_Q_DECAY = 92_160
_SMEM_DIAG = 100_352
_SMEM_INTERMEDIATE = 108_544
_SMEM_DO = 113_664
_SMEM_BETA = 121_856
# Sub-bank fill: the pad places the generic-client group at the 128 KB midpoint
# exactly, keeping the tcgen05-descriptor operands in the low banks.
_SMEM_BANK_FILL = 130_048
_SMEM_Q = 131_072
_SMEM_K = 139_264
_SMEM_V = 147_456
_SMEM_GATE = 155_648
_SMEM_W = 172_032
_SMEM_U = 180_224
_SMEM_DY = 184_320
_SMEM_DQ = 188_416
_SMEM_DK = 192_512
_SMEM_DV = 196_608
_SMEM_DGATE = 204_800
_SMEM_DWO = 212_992
_SMEM_DB = 221_184

_TMEM_DSTATE = 0
_TMEM_DSTATE_INPUT = 128
_TMEM_STATE_INPUT = 192
_TMEM_STATE_K_DY = 320
_TMEM_U = 336
_TMEM_DU = 352
_TMEM_DQ = 368
_TMEM_DK_DECAY = 384
_TMEM_DK_INV = 400
_TMEM_DK_RESTORE = 416
_TMEM_Y_NEG_DY = 432
_TMEM_DU_INPUT = 440
_TMEM_Q_RAW = 448
_TMEM_K_RAW = 480


def _elected():
    lane = K.local_scalar("uint32")
    pred = K.local_scalar("uint32")
    K.ptx.elect_sync(lane, pred, K.uint32(0xFFFFFFFF))
    return pred == K.uint32(1)


def _copy_tensormap(src_map, dst):
    """Copy one 128-byte TensorMap image using the source's vector traffic."""
    payload = K.alloc_local((4,), "uint64")
    src = K.reinterpret("uint64", src_map.ptr_to([0]))
    target = K.reinterpret("uint64", dst)
    for group in range(4):
        offset = K.uint64(group * 32)
        K.ptx.ld.global_.v4.b64(
            payload[0], payload[1], payload[2], payload[3], K.reinterpret("handle", src + offset)
        )
        K.ptx.st.global_.v4.b64(
            K.reinterpret("handle", target + offset), payload[0], payload[1], payload[2], payload[3]
        )


def _replace_tensormap_address(desc, address):
    K.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
        desc, K.reinterpret("uint64", address)
    )


def _replace_tensormap_dim(desc, ordinal, value):
    K.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(
        desc, K.uint32(ordinal), K.cast(value, "uint32")
    )


def _barrier_ptr(arena, byte_offset, stage=0):
    return arena.ptr_to([byte_offset + stage * 8])


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
    K.ptx.mbarrier.arrive.shared.b64(_barrier_ptr(arena, byte_offset, stage), K.uint32(1))


def _expect_tx(arena, byte_offset, stage, nbytes):
    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
        _barrier_ptr(arena, byte_offset, stage), K.uint32(nbytes)
    )


def _tcgen_commit(arena, byte_offset, stage=0):
    K.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        _barrier_ptr(arena, byte_offset, stage), pred=K.cuda.elect_sync()
    )


def _load_work_item(work_items, tile):
    row = K.alloc_local((8,), "int32")
    base = K.cast(tile, "int64") * K.int64(8)
    K.ptx.ld.global_.v4.b32(row[0], row[1], row[2], row[3], work_items.ptr_to([base]))
    K.ptx.ld.global_.v4.b32(row[4], row[5], row[6], row[7], work_items.ptr_to([base + K.int64(4)]))
    return row


def _descriptor_slot(workspace, n_desc, array, batch):
    return workspace.ptr_to([(K.cast(array, "int64") * n_desc + batch) * _TENSOR_MAP_WORDS])


def _mma_m16n16k16(acc, a, b):
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a[0],
        a[1],
        a[2],
        a[3],
        b[0],
        b[1],
        acc[0],
        acc[1],
        acc[2],
        acc[3],
    )
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        acc[4],
        acc[5],
        acc[6],
        acc[7],
        a[0],
        a[1],
        a[2],
        a[3],
        b[2],
        b[3],
        acc[4],
        acc[5],
        acc[6],
        acc[7],
    )


def _pack_bf16_pair(lo, hi):
    packed = K.local_scalar("uint32")
    K.ptx.cvt.rn.bf16x2.f32(packed, hi, lo)
    return packed


def _movmatrix_b16(value):
    transposed = K.local_scalar("uint32")
    K.ptx["movmatrix.sync.aligned.m8n8.trans.b16"](transposed, value)
    return transposed


def _unpack_bf16_pair(packed, lo, hi):
    K.ptx.cvt.f32.bf16(lo, K.cast(packed, "uint16"))
    K.ptx.cvt.f32.bf16(hi, K.cast(K.shift_right(packed, K.uint32(16)), "uint16"))


def _shuffle_xor_f32(value, delta):
    shuffled = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        shuffled,
        K.reinterpret("uint32", value),
        K.uint32(delta),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shuffled)


def _fadd2(a_lo, a_hi, b_lo, b_hi):
    packed = K.local_scalar("uint64")
    out_lo = K.local_scalar("float32")
    out_hi = K.local_scalar("float32")
    K.ptx.add.rn.f32x2(packed, K.cuda.make_float2(a_lo, a_hi), K.cuda.make_float2(b_lo, b_hi))
    K.ptx.mov.b64(out_lo, out_hi, packed)
    return out_lo, out_hi


def _fmul2(a_lo, a_hi, b_lo, b_hi):
    packed = K.local_scalar("uint64")
    out_lo = K.local_scalar("float32")
    out_hi = K.local_scalar("float32")
    K.ptx.mul.rn.f32x2(packed, K.cuda.make_float2(a_lo, a_hi), K.cuda.make_float2(b_lo, b_hi))
    K.ptx.mov.b64(out_lo, out_hi, packed)
    return out_lo, out_hi


def _fsub2(a_lo, a_hi, b_lo, b_hi):
    packed = K.local_scalar("uint64")
    out_lo = K.local_scalar("float32")
    out_hi = K.local_scalar("float32")
    K.ptx.sub.rn.f32x2(packed, K.cuda.make_float2(a_lo, a_hi), K.cuda.make_float2(b_lo, b_hi))
    K.ptx.mov.b64(out_lo, out_hi, packed)
    return out_lo, out_hi


def _ffma2(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi):
    packed = K.local_scalar("uint64")
    out_lo = K.local_scalar("float32")
    out_hi = K.local_scalar("float32")
    K.ptx.fma.rn.f32x2(
        packed,
        K.cuda.make_float2(a_lo, a_hi),
        K.cuda.make_float2(b_lo, b_hi),
        K.cuda.make_float2(c_lo, c_hi),
    )
    K.ptx.mov.b64(out_lo, out_hi, packed)
    return out_lo, out_hi


def _tcgen_mma_ss(
    dst,
    a_desc,
    b_desc,
    idesc,
    *,
    m,
    n,
    k_extent,
    a_transpose=False,
    b_transpose=False,
    accumulate=False,
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
    leader = K.cuda.elect_sync()
    for step in range(k_extent // 16):
        inc_a = (intra_a * (step % steps_a) + subtile_a * (step // steps_a)) >> 4
        inc_b = (intra_b * (step % steps_b) + subtile_b * (step // steps_b)) >> 4
        K.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            K.cast(dst, "uint32"),
            a_desc + K.uint64(inc_a),
            b_desc + K.uint64(inc_b),
            K.uint32(idesc),
            0,
            0,
            0,
            0,
            K.ptx.pred(accumulate if step == 0 else 1),
            pred=leader,
        )


def _tcgen_mma_ts(dst, tmem_a, b_desc, idesc, *, n, k_extent, b_transpose=False, accumulate=False):
    inner_bytes = (n if b_transpose else k_extent) * 2
    swizzle_b = 128 if inner_bytes % 128 == 0 else 64 if inner_bytes % 64 == 0 else 32
    intra_b = 16 * swizzle_b if b_transpose else 32
    subtile_b = k_extent * swizzle_b if b_transpose else swizzle_b * n
    steps_b = k_extent // 16 if b_transpose else (swizzle_b // 2) // 16
    leader = K.cuda.elect_sync()
    for step in range(k_extent // 16):
        inc_b = (intra_b * (step % steps_b) + subtile_b * (step // steps_b)) >> 4
        K.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            K.cast(dst, "uint32"),
            K.cast(tmem_a + step * 8, "uint32"),
            b_desc + K.uint64(inc_b),
            K.uint32(idesc),
            0,
            0,
            0,
            0,
            K.ptx.pred(accumulate if step == 0 else 1),
            pred=leader,
        )


def _exp2_approx(value):
    result = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(result, value)
    return result


def _rcp_approx(value):
    result = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(result, value)
    return result


def _rsqrt_approx(value):
    result = K.local_scalar("float32")
    K.ptx.rsqrt.approx.ftz.f32(result, value)
    return result


def _opaque_f32_zero():
    value = K.local_scalar("float32")
    K.ptx.mov.b32(value, K.uint32(0))
    return value


def _tanh_approx(value):
    result = K.local_scalar("float32")
    K.ptx.tanh.approx.f32(result, value)
    return result


def _raw_smem_descriptor(arena, base_byte, leading_bytes, stride_bytes, layout_code):
    shared_address = K.cuda.cvta_generic_to_shared(arena.ptr_to([base_byte]))
    address = K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF))
    fixed = (
        (((leading_bytes >> 4) & 0x3FFF) << 16) | (((stride_bytes >> 4) & 0x3FFF) << 32) | (1 << 46)
    )
    value = K.bitwise_or(K.uint64(fixed), K.cast(address, "uint64"))
    return K.bitwise_or(value, K.shift_left(K.uint64(layout_code & 7), K.uint64(61)))


def _swizzle_xor_128b(row, column, elem_bytes):
    elements = 128 // elem_bytes
    return K.bitwise_xor(column, (row % 8) * (16 // elem_bytes)) % elements


def _raw_bf16_byte(base, stage, token, channel):
    segment = channel // 64
    within = channel - segment * 64
    element = segment * 1024 + token * 64 + _swizzle_xor_128b(token, within, 2)
    return base + stage * 4096 + element * 2


def _raw_f32_byte(base, stage, token, channel):
    segment = channel // 32
    within = channel - segment * 32
    element = segment * 512 + token * 32 + _swizzle_xor_128b(token, within, 4)
    return base + stage * 8192 + element * 4


def _tmem_cell(base, row, row_delta, column):
    return base + column + K.shift_left(row + row_delta, K.int32(16))


def _make_prologue(*, run_order, order_generate, dynamic_scheduler, n_heads_out):
    @K.kernel(warps=32, arch="sm_100a", grid=1)
    def prologue(
        base_q: K.gptr[K.i64],
        base_k: K.gptr[K.i64],
        base_v: K.gptr[K.i64],
        base_gate: K.gptr[K.i64],
        base_do: K.gptr[K.i64],
        base_dq: K.gptr[K.i64],
        base_dk: K.gptr[K.i64],
        base_dv: K.gptr[K.i64],
        base_dgate: K.gptr[K.i64],
        base_checkpoint: K.gptr[K.i64],
        descriptor_workspace: K.gptr[K.i64],
        cu_seqlens: K.gptr[K.i32],
        q: K.gptr[K.u8],
        k: K.gptr[K.u8],
        v: K.gptr[K.u8],
        gate: K.gptr[K.u8],
        do: K.gptr[K.u8],
        dq: K.gptr[K.u8],
        dk: K.gptr[K.u8],
        dv: K.gptr[K.u8],
        dgate: K.gptr[K.u8],
        checkpoints: K.gptr[K.u8],
        work_item_staging: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        work_items: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        n_batch: K.i32,
        q_row_stride_bytes: K.i32,
        k_row_stride_bytes: K.i32,
        v_row_stride_bytes: K.i32,
        gate_row_stride_bytes: K.i32,
        do_row_stride_bytes: K.i32,
        dq_row_stride_bytes: K.i32,
        dk_row_stride_bytes: K.i32,
        dv_row_stride_bytes: K.i32,
        dgate_row_stride_bytes: K.i32,
        checkpoint_row_stride_bytes: K.i32,
        checkpoint_every_n: K.i32,
    ):
        thread = K.thread_id()
        warp = K.warp_id()
        if run_order:
            order_arena = K.alloc_buffer((32_776,), K.u8, scope="shared.dyn", align=16)

            # The ordering launch owns both consumers' four scheduler words.
            # Their extent is fixed by the frozen two-ring source ABI.
            with K.If(thread == 0), K.Then():
                sched_index = K.local_scalar("int32", init=K.int32(0))
                with K.While(sched_index < 4):
                    K.ptx.st.global_.s32(scheduler.ptr_to([sched_index]), K.int32(0))
                    K.assign(sched_index, sched_index + 1)

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
                    sequence_begin = K.local_scalar("int32")
                    sequence_end = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(sequence_begin, cu_seqlens.ptr_to([batch]))
                    K.ptx.ld.global_.s32(sequence_end, cu_seqlens.ptr_to([batch + 1]))
                    num_chunks = (sequence_end - sequence_begin + _BT - 1) // _BT
                    values = (
                        batch,
                        head,
                        K.int32(0),
                        num_chunks,
                        K.int32(0),
                        num_chunks,
                        sequence_begin,
                        sequence_end,
                    )
                    K.ptx.st.global_.v4.b32(
                        work_items.ptr_to([destination * 8]),
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                    )
                    K.ptx.st.global_.v4.b32(
                        work_items.ptr_to([destination * 8 + 4]),
                        values[4],
                        values[5],
                        values[6],
                        values[7],
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

                # Four entries per thread fill the fixed 4096-item capacity.
                # Padding carries INT_MIN so the descending bitonic network
                # leaves it behind every real work item.
                local_min = K.local_scalar("int32", init=K.int32(2_147_483_647))
                local_max = K.local_scalar("int32", init=K.int32(-2_147_483_648))
                for element in range(4):
                    item = thread + element * 1024
                    with K.If(item < padded_count), K.Then():
                        key = K.local_scalar("int32", init=K.int32(-2_147_483_648))
                        with K.If(item < item_count), K.Then():
                            if order_generate:
                                batch = item // n_heads_out
                                begin = K.local_scalar("int32")
                                end = K.local_scalar("int32")
                                K.ptx.ld.global_.s32(begin, cu_seqlens.ptr_to([batch]))
                                K.ptx.ld.global_.s32(end, cu_seqlens.ptr_to([batch + 1]))
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
                atomic_old = K.local_scalar("int32")
                K.ptx["atom.shared::cta.min.s32"](
                    atomic_old, order_arena.ptr_to([32_768]), local_min
                )
                K.ptx["atom.shared::cta.max.s32"](
                    atomic_old, order_arena.ptr_to([32_772]), local_max
                )
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
                    network_size = K.local_scalar("int32", init=K.int32(2))
                    with K.While(network_size <= padded_count):
                        distance = K.local_scalar("int32", init=network_size // K.int32(2))
                        with K.While(distance > 0):
                            for element in range(4):
                                item = thread + element * 1024
                                partner = K.bitwise_xor(item, distance)
                                with K.If(K.And(item < padded_count, partner > item)), K.Then():
                                    key_i = K.local_scalar("int32")
                                    key_j = K.local_scalar("int32")
                                    K.ptx.ld.shared.s32(key_i, order_arena.ptr_to([item * 4]))
                                    K.ptx.ld.shared.s32(key_j, order_arena.ptr_to([partner * 4]))
                                    ascending_half = ((item // network_size) % K.int32(2)) == 0
                                    swap = K.Or(
                                        K.And(ascending_half, key_i < key_j),
                                        K.And(K.Not(ascending_half), key_i > key_j),
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
                        K.assign(network_size, network_size * 2)

                    for element in range(4):
                        destination = thread + element * 1024
                        with K.If(destination < item_count), K.Then():
                            source = K.local_scalar("int32")
                            K.ptx.ld.shared.s32(
                                source, order_arena.ptr_to([16_384 + destination * 4])
                            )
                            write_work_item(destination, source)
        maps = (base_q, base_k, base_v, base_gate, base_do, base_dq, base_dk, base_dv, base_dgate)
        bases = (q, k, v, gate, do, dq, dk, dv, dgate)
        strides = (
            q_row_stride_bytes,
            k_row_stride_bytes,
            v_row_stride_bytes,
            gate_row_stride_bytes,
            do_row_stride_bytes,
            dq_row_stride_bytes,
            dk_row_stride_bytes,
            dv_row_stride_bytes,
            dgate_row_stride_bytes,
        )
        for array_index in range(9):
            with K.If(warp == array_index), K.Then():
                with K.If(_elected()), K.Then():
                    with K.serial(n_batch) as batch:
                        sequence_begin = K.local_scalar("int32")
                        sequence_end = K.local_scalar("int32")
                        K.ptx.ld.global_.s32(sequence_begin, cu_seqlens.ptr_to([batch]))
                        K.ptx.ld.global_.s32(sequence_end, cu_seqlens.ptr_to([batch + 1]))
                        sequence_length = sequence_end - sequence_begin
                        slot = descriptor_workspace.ptr_to(
                            [(array_index * n_batch + batch) * _TENSOR_MAP_WORDS]
                        )
                        _copy_tensormap(maps[array_index], slot)
                        _replace_tensormap_address(
                            slot,
                            bases[array_index].ptr_to(
                                [K.cast(sequence_begin, "int64") * strides[array_index]]
                            ),
                        )
                        _replace_tensormap_dim(slot, 2, sequence_length)
                    K.ptx.fence.proxy.tensormap__generic.release.gpu()
        with K.If(warp == 9), K.Then():
            with K.If(_elected()), K.Then():
                checkpoint_prefix = K.local_scalar("int32", init=K.int32(0))
                with K.serial(n_batch) as batch:
                    sequence_begin = K.local_scalar("int32")
                    sequence_end = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(sequence_begin, cu_seqlens.ptr_to([batch]))
                    K.ptx.ld.global_.s32(sequence_end, cu_seqlens.ptr_to([batch + 1]))
                    sequence_length = sequence_end - sequence_begin
                    checkpoint_count = K.local_scalar("int32", init=K.int32(0))
                    with K.If(sequence_length > 0), K.Then():
                        K.assign(checkpoint_count, (sequence_length - 1) // checkpoint_every_n + 1)
                    slot = descriptor_workspace.ptr_to([(9 * n_batch + batch) * _TENSOR_MAP_WORDS])
                    _copy_tensormap(base_checkpoint, slot)
                    _replace_tensormap_address(
                        slot,
                        checkpoints.ptr_to(
                            [K.cast(checkpoint_prefix, "int64") * checkpoint_row_stride_bytes]
                        ),
                    )
                    _replace_tensormap_dim(slot, 2, checkpoint_count)
                    K.assign(checkpoint_prefix, checkpoint_prefix + checkpoint_count)
                K.ptx.fence.proxy.tensormap__generic.release.gpu()

    return prologue


# KERNEL_SKETCH_START
def _make_main(
    *,
    num_sms,
    beta_sigmoid,
    use_dstate_in,
    use_dstate0,
    use_initial_state,
    safe_gate,
    full_tiles,
    dynamic_scheduler,
    l2norm,
    n_heads_out,
    q_ratio,
    k_ratio,
    v_ratio,
):
    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=num_sms)
    def main(
        descriptor_workspace: K.gptr[K.i64],
        n_desc: K.i32,
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        cu_seqlens: K.gptr[K.i32],
        dgate: K.gptr[K.f32],
        d_initial_state: K.gptr[K.f32],
        d_final_state: K.gptr[K.f32],
        work_items: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        scale: K.f32,
    ):
        arena = K.alloc_buffer((_MAIN_SMEM_BYTES,), K.u8, scope="shared.dyn", align=1024)
        # The pool owns only the fixed pipeline/barrier header over the arena;
        # all data storage below is addressed by integer byte offsets.
        K.smem_pool(base=arena)
        thread = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()

        # Every physical protocol word has one elected initialization owner.
        for ready_off, done_off, stages, ready_count, done_count, owner in _PROTOCOL:
            with K.If(warp == owner), K.Then():
                with K.If(_elected()), K.Then():
                    for stage in range(stages):
                        K.ptx.mbarrier.init.shared.b64(
                            _barrier_ptr(arena, ready_off, stage), K.uint32(ready_count)
                        )
                        if done_off is not None:
                            K.ptx.mbarrier.init.shared.b64(
                                _barrier_ptr(arena, done_off, stage), K.uint32(done_count)
                            )
        # The two diagonal stages are born zero and later receive only their
        # diagonal entries from CG0.
        with K.unroll(8) as diagonal_pass:
            element = thread + diagonal_pass * 512
            K.ptx.st.shared.b16(arena.ptr_to([_SMEM_DIAG + element * 2]), K.uint16(0))
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        total_tiles = K.local_scalar("int32")
        K.ptx.ld.global_.s32(total_tiles, work_count.ptr_to([0]))
        roles = K.specialize()
        cg0 = roles.role("cg0", warps=range(0, 4), regs=128)
        cg1 = roles.role("cg1", warps=range(4, 8), regs=184)
        cg2 = roles.role("cg2", warps=range(8, 12), regs=136)
        super_mma = roles.role("super_mma", warps=[12], regs=64)
        tcgen = roles.role("tcgen", warps=[13], regs=64)
        tma = roles.role("tma", warps=[14], regs=64)
        epilogue = roles.role("epilogue", warps=[15], regs=64)
        with tma:
            tile = K.local_scalar("int32", init=K.cta_id())
            raw = K.PipelineState(2, phase=1)
            state_cursor = K.PipelineState(1, phase=1)
            sched_producer = K.PipelineState(8, phase=1)
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                cend = item[5]

                next_tile = K.local_scalar("int32", init=tile + num_sms)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_DONE, sched_producer.stage, sched_producer.phase
                    )
                    with K.If(_elected()), K.Then():
                        ticket = K.local_scalar("uint32")
                        K.ptx.atom.global_.add.u32(ticket, scheduler.ptr_to([0]), K.uint32(1))
                        K.ptx.st.shared.u32(
                            arena.ptr_to([_SMEM_SCHED + K.cast(sched_producer.stage, "int32") * 4]),
                            K.uint32(num_sms) + ticket,
                        )
                    K.cuda.warp_sync()
                    K.ptx.ld.shared.s32(
                        next_tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_producer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_READY, sched_producer.stage)
                    sched_producer.advance()

                head_q = head if q_ratio == 1 else head // q_ratio
                head_k = head if k_ratio == 1 else head // k_ratio
                head_v = head if v_ratio == 1 else head // v_ratio
                desc_q = _descriptor_slot(descriptor_workspace, n_desc, 0, batch)
                desc_k = _descriptor_slot(descriptor_workspace, n_desc, 1, batch)
                desc_v = _descriptor_slot(descriptor_workspace, n_desc, 2, batch)
                desc_gate = _descriptor_slot(descriptor_workspace, n_desc, 3, batch)
                desc_do = _descriptor_slot(descriptor_workspace, n_desc, 4, batch)
                desc_beta = _descriptor_slot(descriptor_workspace, n_desc, 5, batch)
                desc_w = _descriptor_slot(descriptor_workspace, n_desc, 6, batch)
                desc_state = _descriptor_slot(descriptor_workspace, n_desc, 13, batch)
                with K.If(_elected()), K.Then():
                    for desc in (
                        desc_q,
                        desc_k,
                        desc_v,
                        desc_gate,
                        desc_do,
                        desc_beta,
                        desc_w,
                        desc_state,
                    ):
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc)

                num_chunks = cend - wstart
                with K.serial(num_chunks, unroll=False) as reverse_index:
                    chunk = cend - 1 - reverse_index
                    token = chunk * _BT
                    raw_stage = raw.stage
                    raw_phase = raw.phase
                    with K.If(_elected()), K.Then():
                        _wait_barrier(arena, _BAR_Q_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_Q_READY, raw_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_Q + K.cast(raw_stage, "int32") * 4096 + d_coord * 32]
                                ),
                                desc_q,
                                K.int32(d_coord),
                                K.cast(head_q, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_Q_READY, raw_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, _BAR_K_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_K_READY, raw_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_K + K.cast(raw_stage, "int32") * 4096 + d_coord * 32]
                                ),
                                desc_k,
                                K.int32(d_coord),
                                K.cast(head_k, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_K_READY, raw_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, _BAR_GATE_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_GATE_READY, raw_stage, 8192)
                        for d_coord in (0, 32, 64, 96):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_GATE + K.cast(raw_stage, "int32") * 8192 + d_coord * 64]
                                ),
                                desc_gate,
                                K.int32(d_coord),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_GATE_READY, raw_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, _BAR_BETA_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_BETA_READY, raw_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_BETA + K.cast(raw_stage, "int32") * 4096 + d_coord * 32]
                                ),
                                desc_beta,
                                K.int32(d_coord),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_BETA_READY, raw_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, _BAR_DO_DONE, raw_stage, raw_phase)
                        _wait_barrier(arena, _BAR_DO_MMA_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_DO_READY, raw_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_DO + K.cast(raw_stage, "int32") * 4096 + d_coord * 32]
                                ),
                                desc_do,
                                K.int32(d_coord),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_DO_READY, raw_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, _BAR_V_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_V_READY, raw_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_V + K.cast(raw_stage, "int32") * 4096 + d_coord * 32]
                                ),
                                desc_v,
                                K.int32(d_coord),
                                K.cast(head_v, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_V_READY, raw_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, _BAR_W_DONE, raw_stage, raw_phase)
                        _expect_tx(arena, _BAR_W_READY, raw_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to(
                                    [_SMEM_W + K.cast(raw_stage, "int32") * 4096 + d_coord * 32]
                                ),
                                desc_w,
                                K.int32(d_coord),
                                K.cast(head, "int32"),
                                K.cast(token, "int32"),
                                _barrier_ptr(arena, _BAR_W_READY, raw_stage),
                                K.uint64(0),
                            )

                    first_state_chunk = 0 if use_initial_state else 1
                    with K.If(chunk >= first_state_chunk), K.Then():
                        with K.If(_elected()), K.Then():
                            _wait_barrier(
                                arena, _BAR_STATE_CG0_DONE, state_cursor.stage, state_cursor.phase
                            )
                            _wait_barrier(
                                arena, _BAR_STATE_DONE, state_cursor.stage, state_cursor.phase
                            )
                            _expect_tx(arena, _BAR_STATE_READY, state_cursor.stage, 32768)
                            for value_coord in (0, 64):
                                K.ptx[_TMA_G2S_4D](
                                    arena.ptr_to([_SMEM_STATE + value_coord * 256]),
                                    desc_state,
                                    K.int32(value_coord),
                                    K.int32(0),
                                    K.cast(chunk, "int32"),
                                    K.cast(head, "int32"),
                                    _barrier_ptr(arena, _BAR_STATE_READY, state_cursor.stage),
                                    K.uint64(0),
                                )
                        state_cursor.advance()
                    raw.advance()
                K.assign(tile, next_tile)
        with super_mma:
            tile = K.local_scalar("int32", init=K.cta_id())
            chunk_serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            dy_smem_consumer = K.PipelineState(1, phase=0)
            rhs_row = lane % 8 + K.if_then_else(lane // 16 != 0, 8, 0)
            rhs_col = K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_row = lane % 8 + K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_row = (lane & 7) + K.if_then_else((lane // 8) & 1 != 0, 8, 0)
            store_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_linear = store_row * 16 + store_col
            store_swizzled = K.bitwise_xor(
                store_linear,
                K.shift_left(
                    K.bitwise_and(K.shift_right(store_linear, K.uint32(6)), K.uint32(1)),
                    K.uint32(3),
                ),
            )
            row_lo = lane // 4
            row_hi = row_lo + 8
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                num_chunks = item[5] - item[2]
                with K.serial(num_chunks, unroll=False) as reverse_index:
                    serial = chunk_serial_base + reverse_index
                    decay_stage = serial % 2
                    inter_stage = serial % 2
                    _wait_barrier(arena, _BAR_T_INV_DONE, inter_stage, (serial // 2 + 1) & 1)
                    _wait_barrier(arena, _BAR_K_DECAY_INV_READY, decay_stage, (serial // 2) & 1)

                    kk = K.alloc_local((8,), "float32")
                    for accum_index in range(8):
                        K.assign(kk[accum_index], K.float32(0.0))
                    with K.unroll(8) as key_block:
                        a = K.alloc_local((4,), "uint32")
                        b = K.alloc_local((4,), "uint32")
                        a_channel = key_block * 16 + lhs_col
                        b_channel = key_block * 16 + rhs_col
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a[0],
                            a[1],
                            a[2],
                            a[3],
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_K_DECAY, decay_stage, lhs_row, a_channel)]
                            ),
                        )
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            b[0],
                            b[1],
                            b[2],
                            b[3],
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_K_INV, decay_stage, rhs_row, b_channel)]
                            ),
                        )
                        _mma_m16n16k16(kk, a, b)

                    strict = K.alloc_local((8,), "float32")
                    l_word = K.alloc_local((4,), "uint32")
                    rounded = K.alloc_local((8,), "float32")
                    tinv = K.alloc_local((8,), "float32")
                    for accum_index in range(8):
                        row_coord = row_hi if accum_index % 4 >= 2 else row_lo
                        col_coord = (accum_index // 4) * 8 + 2 * (lane % 4)
                        if accum_index % 2:
                            col_coord = col_coord + 1
                        # No Beta row scale here: GDN-2 folds the erase gate into
                        # the K_decay operand, so L is plain strict_tril(KK).
                        K.assign(
                            strict[accum_index],
                            K.if_then_else(row_coord > col_coord, kk[accum_index], 0.0),
                        )
                    for pair in range(4):
                        K.assign(
                            l_word[pair], _pack_bf16_pair(strict[pair * 2], strict[pair * 2 + 1])
                        )
                        _unpack_bf16_pair(l_word[pair], rounded[pair * 2], rounded[pair * 2 + 1])
                    for accum_index in range(8):
                        row_coord = row_hi if accum_index % 4 >= 2 else row_lo
                        col_coord = (accum_index // 4) * 8 + 2 * (lane % 4)
                        if accum_index % 2:
                            col_coord = col_coord + 1
                        K.assign(
                            tinv[accum_index],
                            K.if_then_else(row_coord == col_coord, 1.0, 0.0) - rounded[accum_index],
                        )

                    l_power = K.alloc_local((4,), "uint32")
                    l_power_t = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        K.assign(l_power[pair], l_word[pair])
                        K.assign(l_power_t[pair], _movmatrix_b16(l_power[pair]))
                    for _ in range(3):
                        square = K.alloc_local((8,), "float32")
                        for accum_index in range(8):
                            K.assign(square[accum_index], K.float32(0.0))
                        _mma_m16n16k16(square, l_power, l_power_t)
                        for pair in range(4):
                            K.assign(
                                l_power[pair],
                                _pack_bf16_pair(square[pair * 2], square[pair * 2 + 1]),
                            )
                            K.assign(l_power_t[pair], _movmatrix_b16(l_power[pair]))
                        tinv_word = K.alloc_local((4,), "uint32")
                        for pair in range(4):
                            K.assign(
                                tinv_word[pair], _pack_bf16_pair(tinv[pair * 2], tinv[pair * 2 + 1])
                            )
                        update = K.alloc_local((8,), "float32")
                        for accum_index in range(8):
                            K.assign(update[accum_index], K.float32(0.0))
                        _mma_m16n16k16(update, tinv_word, l_power_t)
                        for pair in range(4):
                            old_lo = K.local_scalar("float32")
                            old_hi = K.local_scalar("float32")
                            _unpack_bf16_pair(tinv_word[pair], old_lo, old_hi)
                            next_lo, next_hi = _fadd2(
                                old_lo, old_hi, update[pair * 2], update[pair * 2 + 1]
                            )
                            K.assign(tinv[pair * 2], next_lo)
                            K.assign(tinv[pair * 2 + 1], next_hi)

                    tinv_out = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        K.assign(
                            tinv_out[pair], _pack_bf16_pair(tinv[pair * 2], tinv[pair * 2 + 1])
                        )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to(
                            [_SMEM_INTERMEDIATE + inter_stage * 2560 + 512 + store_swizzled * 2]
                        ),
                        tinv_out[0],
                        tinv_out[1],
                        tinv_out[2],
                        tinv_out[3],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_T_INV_READY, inter_stage)

                    _wait_barrier(arena, _BAR_DM_DONE, inter_stage, (serial // 2 + 1) & 1)
                    _wait_barrier(
                        arena, _BAR_DY_SMEM_READY, dy_smem_consumer.stage, dy_smem_consumer.phase
                    )
                    dy_smem_consumer.advance()
                    dm = K.alloc_local((8,), "float32")
                    for accum_index in range(8):
                        K.assign(dm[accum_index], K.float32(0.0))
                    with K.unroll(8) as value_block:
                        a = K.alloc_local((4,), "uint32")
                        b = K.alloc_local((4,), "uint32")
                        a_channel = value_block * 16 + lhs_col
                        b_channel = value_block * 16 + rhs_col
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a[0],
                            a[1],
                            a[2],
                            a[3],
                            arena.ptr_to([_raw_bf16_byte(_SMEM_DY, 0, lhs_row, a_channel)]),
                        )
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            b[0],
                            b[1],
                            b[2],
                            b[3],
                            arena.ptr_to([_raw_bf16_byte(_SMEM_U, 0, rhs_row, b_channel)]),
                        )
                        _mma_m16n16k16(dm, a, b)

                    dm_word = K.alloc_local((4,), "uint32")
                    ndm_word = K.alloc_local((4,), "uint32")
                    for accum_index in range(8):
                        row_coord = row_hi if accum_index % 4 >= 2 else row_lo
                        col_coord = (accum_index // 4) * 8 + 2 * (lane % 4)
                        if accum_index % 2:
                            col_coord = col_coord + 1
                        # Unscaled: the erase gate lives in the K_decay operand, so
                        # the strict dM tile carries no per-row factor.
                        K.assign(
                            strict[accum_index],
                            K.if_then_else(row_coord > col_coord, dm[accum_index], 0.0),
                        )
                    for pair in range(4):
                        K.assign(
                            dm_word[pair], _pack_bf16_pair(strict[pair * 2], strict[pair * 2 + 1])
                        )
                        K.assign(
                            ndm_word[pair],
                            _pack_bf16_pair(-strict[pair * 2], -strict[pair * 2 + 1]),
                        )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to(
                            [_SMEM_INTERMEDIATE + inter_stage * 2560 + 1536 + store_swizzled * 2]
                        ),
                        dm_word[0],
                        dm_word[1],
                        dm_word[2],
                        dm_word[3],
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to(
                            [_SMEM_INTERMEDIATE + inter_stage * 2560 + 2048 + store_swizzled * 2]
                        ),
                        ndm_word[0],
                        ndm_word[1],
                        ndm_word[2],
                        ndm_word[3],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DM_READY, inter_stage)

                K.assign(chunk_serial_base, chunk_serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_READY, sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(
                        tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_consumer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_DONE, sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
        with tcgen:
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                arena.ptr_to([_SMEM_TMEM_MAILBOX]), K.uint32(512)
            )
            K.ptx.bar.sync(K.uint32(3), K.uint32(416))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_SMEM_TMEM_MAILBOX]))
            tile = K.local_scalar("int32", init=K.cta_id())
            serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            state_consumer = K.PipelineState(1, phase=0)
            dqk_done_consumer = K.PipelineState(1, phase=1)
            dstate_input_consumer = K.PipelineState(1, phase=0)
            y_consumer = K.PipelineState(1, phase=0)
            du_consumer = K.PipelineState(1, phase=0)
            neg_dy_consumer = K.PipelineState(1, phase=0)
            u_smem_consumer = K.PipelineState(1, phase=0)
            dstate_smem_consumer = K.PipelineState(1, phase=0)
            dstate0_consumer = K.PipelineState(1, phase=0)
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                wstart = item[2]
                cend = item[5]
                num_chunks = cend - wstart
                with K.serial(num_chunks, unroll=False) as reverse_index:
                    serial = serial_base + reverse_index
                    chunk = cend - 1 - reverse_index
                    raw_stage = serial % 2
                    decay_stage = serial % 2
                    inter_stage = serial % 2
                    decay_phase = (serial // 2) & 1
                    inter_phase = (serial // 2) & 1
                    has_dstate = reverse_index > 0
                    if use_dstate_in:
                        has_dstate = K.bool(True)

                    state_direct = _raw_smem_descriptor(arena, _SMEM_STATE, 16, 1024, 2)
                    state_alt = _raw_smem_descriptor(arena, _SMEM_STATE, 16384, 1024, 2)
                    dstate_alt = _raw_smem_descriptor(arena, _SMEM_DSTATE, 16384, 1024, 2)
                    k_decay_lead = _raw_smem_descriptor(
                        arena, _SMEM_K_DECAY + decay_stage * 4096, 16, 1024, 2
                    )
                    k_decay_trans = _raw_smem_descriptor(
                        arena, _SMEM_K_DECAY + decay_stage * 4096, 2048, 1024, 2
                    )
                    k_inverse_lead = _raw_smem_descriptor(
                        arena, _SMEM_K_INV + decay_stage * 4096, 16, 1024, 2
                    )
                    k_inverse_amajor = _raw_smem_descriptor(
                        arena, _SMEM_K_INV + decay_stage * 4096, 2048, 1024, 2
                    )
                    k_restore_lead = _raw_smem_descriptor(
                        arena, _SMEM_K_RESTORE + decay_stage * 4096, 16, 1024, 2
                    )
                    q_decay_trans = _raw_smem_descriptor(
                        arena, _SMEM_Q_DECAY + decay_stage * 4096, 2048, 1024, 2
                    )
                    do_lead = _raw_smem_descriptor(arena, _SMEM_DO + raw_stage * 4096, 16, 1024, 2)
                    do_amajor = _raw_smem_descriptor(
                        arena, _SMEM_DO + raw_stage * 4096, 2048, 1024, 2
                    )
                    dv_lead = _raw_smem_descriptor(
                        arena, _SMEM_DV + (serial % 2) * 4096, 16, 1024, 2
                    )
                    u_lead = _raw_smem_descriptor(arena, _SMEM_U, 16, 1024, 2)
                    inter_a = _raw_smem_descriptor(
                        arena, _SMEM_INTERMEDIATE + inter_stage * 2560, 16, 256, 6
                    )
                    inter_tinv = _raw_smem_descriptor(
                        arena, _SMEM_INTERMEDIATE + inter_stage * 2560 + 512, 16, 256, 6
                    )
                    inter_da = _raw_smem_descriptor(
                        arena, _SMEM_INTERMEDIATE + inter_stage * 2560 + 1024, 16, 256, 6
                    )
                    inter_dm = _raw_smem_descriptor(
                        arena, _SMEM_INTERMEDIATE + inter_stage * 2560 + 1536, 16, 256, 6
                    )
                    inter_ndm = _raw_smem_descriptor(
                        arena, _SMEM_INTERMEDIATE + inter_stage * 2560 + 2048, 16, 256, 6
                    )

                    _wait_barrier(arena, _BAR_K_DECAY_INV_READY, decay_stage, decay_phase)
                    first_state_chunk = 0 if use_initial_state else 1
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(
                            arena, _BAR_STATE_READY, state_consumer.stage, state_consumer.phase
                        )
                        _tcgen_mma_ss(
                            tmem_base + _TMEM_STATE_K_DY,
                            state_direct,
                            k_decay_lead,
                            134481040,
                            m=128,
                            n=16,
                            k_extent=128,
                        )
                        _tcgen_commit(arena, _BAR_STATE_K_ACC_READY)
                        _tcgen_commit(arena, _BAR_STATE_DONE, state_consumer.stage)
                        state_consumer.advance()

                    _wait_barrier(
                        arena, _BAR_DQK_ACC_DONE, dqk_done_consumer.stage, dqk_done_consumer.phase
                    )
                    dqk_done_consumer.advance()
                    _wait_barrier(arena, _BAR_STATE_INP_READY, serial % 2, (serial // 2) & 1)
                    _wait_barrier(arena, _BAR_DO_READY, raw_stage, (serial // 2) & 1)
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _tcgen_mma_ts(
                            tmem_base + _TMEM_DQ,
                            tmem_base + _TMEM_STATE_INPUT + (serial % 2) * 64,
                            do_lead,
                            134481040,
                            n=16,
                            k_extent=128,
                        )

                    _wait_barrier(arena, _BAR_Q_DECAY_K_RESTORE_READY, decay_stage, decay_phase)
                    with K.If(has_dstate), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_INP_READY,
                            dstate_input_consumer.stage,
                            dstate_input_consumer.phase,
                        )
                        _tcgen_mma_ts(
                            tmem_base + _TMEM_DU,
                            tmem_base + _TMEM_DSTATE_INPUT,
                            k_restore_lead,
                            134481040,
                            n=16,
                            k_extent=128,
                        )
                        for key_block in range(8):
                            diagonal = _raw_smem_descriptor(
                                arena, _SMEM_DIAG + decay_stage * 4096 + key_block * 512, 16, 256, 6
                            )
                            _tcgen_mma_ts(
                                tmem_base + _TMEM_DSTATE + key_block * 16,
                                tmem_base + _TMEM_DSTATE_INPUT + key_block * 8,
                                diagonal,
                                134481040,
                                n=16,
                                k_extent=16,
                            )
                        dstate_input_consumer.advance()

                    _wait_barrier(arena, _BAR_A_READY, inter_stage, inter_phase)
                    _tcgen_mma_ss(
                        tmem_base + _TMEM_DU,
                        do_amajor,
                        inter_a,
                        134579344,
                        m=128,
                        n=16,
                        k_extent=16,
                        a_transpose=True,
                        b_transpose=True,
                        accumulate=has_dstate,
                    )
                    _tcgen_commit(arena, _BAR_DU_ACC_READY)
                    _tcgen_commit(arena, _BAR_A_DONE, inter_stage)
                    _tcgen_mma_ss(
                        tmem_base + _TMEM_DSTATE,
                        do_amajor,
                        q_decay_trans,
                        136414352,
                        m=128,
                        n=128,
                        k_extent=16,
                        a_transpose=True,
                        b_transpose=True,
                        accumulate=has_dstate,
                    )

                    _wait_barrier(arena, _BAR_T_INV_READY, inter_stage, inter_phase)
                    _wait_barrier(arena, _BAR_Y_INP_READY, y_consumer.stage, y_consumer.phase)
                    y_consumer.advance()
                    _tcgen_mma_ts(
                        tmem_base + _TMEM_U,
                        tmem_base + _TMEM_Y_NEG_DY,
                        inter_tinv,
                        134481040,
                        n=16,
                        k_extent=16,
                    )
                    _tcgen_commit(arena, _BAR_U_ACC_READY)

                    _wait_barrier(arena, _BAR_DU_INP_READY, du_consumer.stage, du_consumer.phase)
                    du_consumer.advance()
                    _tcgen_mma_ts(
                        tmem_base + _TMEM_STATE_K_DY,
                        tmem_base + _TMEM_DU_INPUT,
                        inter_tinv,
                        134546576,
                        n=16,
                        k_extent=16,
                        b_transpose=True,
                    )
                    _tcgen_commit(arena, _BAR_DY_ACC_READY)
                    _tcgen_commit(arena, _BAR_T_INV_DONE, inter_stage)

                    _wait_barrier(
                        arena, _BAR_U_SMEM_READY, u_smem_consumer.stage, u_smem_consumer.phase
                    )
                    u_smem_consumer.advance()
                    with K.If(has_dstate), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_SMEM_READY,
                            dstate_smem_consumer.stage,
                            dstate_smem_consumer.phase,
                        )
                        dstate_smem_consumer.advance()
                        _tcgen_mma_ss(
                            tmem_base + _TMEM_DK_RESTORE,
                            dstate_alt,
                            u_lead,
                            134513808,
                            m=128,
                            n=16,
                            k_extent=128,
                            a_transpose=True,
                        )
                        _tcgen_commit(arena, _BAR_DK_RESTORE_PART_ACC_READY)
                        _tcgen_commit(arena, _BAR_DSTATE_SMEM_DONE)

                    _wait_barrier(
                        arena, _BAR_NEG_DY_INP_READY, neg_dy_consumer.stage, neg_dy_consumer.phase
                    )
                    neg_dy_consumer.advance()
                    _tcgen_mma_ts(
                        tmem_base + _TMEM_DSTATE,
                        tmem_base + _TMEM_Y_NEG_DY,
                        k_decay_trans,
                        136381584,
                        n=128,
                        k_extent=16,
                        b_transpose=True,
                        accumulate=True,
                    )
                    _tcgen_commit(arena, _BAR_DSTATE_ACC_READY)

                    _wait_barrier(arena, _BAR_DA_READY, inter_stage, inter_phase)
                    _tcgen_mma_ss(
                        tmem_base + _TMEM_DK_INV,
                        q_decay_trans,
                        inter_da,
                        134579344,
                        m=128,
                        n=16,
                        k_extent=16,
                        a_transpose=True,
                        b_transpose=True,
                    )
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _tcgen_mma_ss(
                            tmem_base + _TMEM_DQ,
                            k_inverse_amajor,
                            inter_da,
                            134513808,
                            m=128,
                            n=16,
                            k_extent=16,
                            a_transpose=True,
                            accumulate=True,
                        )
                    with K.If(chunk < first_state_chunk), K.Then():
                        _tcgen_mma_ss(
                            tmem_base + _TMEM_DQ,
                            k_inverse_amajor,
                            inter_da,
                            134513808,
                            m=128,
                            n=16,
                            k_extent=16,
                            a_transpose=True,
                            accumulate=False,
                        )
                    _tcgen_commit(arena, _BAR_DQ_ACC_READY)
                    _tcgen_commit(arena, _BAR_DA_DONE, inter_stage)

                    _wait_barrier(arena, _BAR_DV_TMASTG_READY, serial % 2, (serial // 2) & 1)
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _tcgen_mma_ts(
                            tmem_base + _TMEM_DK_DECAY,
                            tmem_base + _TMEM_STATE_INPUT + (serial % 2) * 64,
                            dv_lead,
                            134481040,
                            n=16,
                            k_extent=128,
                        )
                    _tcgen_commit(arena, _BAR_STATE_INP_DONE, serial % 2)

                    _wait_barrier(arena, _BAR_DM_READY, inter_stage, inter_phase)
                    _tcgen_mma_ss(
                        tmem_base + _TMEM_DK_INV,
                        k_decay_trans,
                        inter_ndm,
                        134579344,
                        m=128,
                        n=16,
                        k_extent=16,
                        a_transpose=True,
                        b_transpose=True,
                        accumulate=True,
                    )
                    _tcgen_commit(arena, _BAR_DK_INV_PART_ACC_READY)
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _tcgen_mma_ss(
                            tmem_base + _TMEM_DK_DECAY,
                            k_inverse_amajor,
                            inter_dm,
                            134513808,
                            m=128,
                            n=16,
                            k_extent=16,
                            a_transpose=True,
                            accumulate=True,
                        )
                    with K.If(chunk < first_state_chunk), K.Then():
                        _tcgen_mma_ss(
                            tmem_base + _TMEM_DK_DECAY,
                            k_inverse_amajor,
                            inter_dm,
                            134513808,
                            m=128,
                            n=16,
                            k_extent=16,
                            a_transpose=True,
                            accumulate=False,
                        )
                    _tcgen_commit(arena, _BAR_DK_DECAY_PART_ACC_READY)
                    _tcgen_commit(arena, _BAR_DM_DONE, inter_stage)
                    _tcgen_commit(arena, _BAR_DECAY_DONE, decay_stage)

                _wait_barrier(
                    arena, _BAR_DSTATE0_ACC_STORED, dstate0_consumer.stage, dstate0_consumer.phase
                )
                dstate0_consumer.advance()
                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_READY, sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(
                        tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_consumer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_DONE, sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
            _wait_barrier(arena, _BAR_TMEM_DONE, 0, 0)
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                K.cast(tmem_base, "uint32"), K.uint32(512)
            )
        with epilogue:
            rhs_row = lane % 8 + K.if_then_else(lane // 16 != 0, 8, 0)
            rhs_col = K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_row = lane % 8 + K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_row = (lane & 7) + K.if_then_else((lane // 8) & 1 != 0, 8, 0)
            store_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_linear = store_row * 16 + store_col
            store_swizzled = K.bitwise_xor(
                store_linear,
                K.shift_left(
                    K.bitwise_and(K.shift_right(store_linear, K.uint32(6)), K.uint32(1)),
                    K.uint32(3),
                ),
            )
            row_lo = lane // 4
            row_hi = row_lo + 8
            tile = K.local_scalar("int32", init=K.cta_id())
            serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            u_consumer = K.PipelineState(1, phase=0)
            dq_store = K.PipelineState(1, phase=0)
            dk_store = K.PipelineState(1, phase=0)
            dgate_store = K.PipelineState(1, phase=0)
            db_store = K.PipelineState(1, phase=0)
            dv_store = K.PipelineState(2, phase=0)
            dwo_store = K.PipelineState(2, phase=0)

            def store_pending(
                pend_token,
                pend_writes,
                head,
                desc_dq,
                desc_dk,
                desc_dv,
                desc_dgate,
                desc_db,
                desc_dwo,
            ):
                _wait_barrier(arena, _BAR_DQ_TMASTG_READY, dq_store.stage, dq_store.phase)
                with K.If(pend_writes), K.Then():
                    for d_coord in (0, 64):
                        K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                            desc_dq,
                            K.int32(d_coord),
                            K.cast(head, "int32"),
                            K.cast(pend_token, "int32"),
                            arena.ptr_to([_SMEM_DQ + d_coord * 32]),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                _wait_barrier(arena, _BAR_DK_TMASTG_READY, dk_store.stage, dk_store.phase)
                with K.If(pend_writes), K.Then():
                    for d_coord in (0, 64):
                        K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                            desc_dk,
                            K.int32(d_coord),
                            K.cast(head, "int32"),
                            K.cast(pend_token, "int32"),
                            arena.ptr_to([_SMEM_DK + d_coord * 32]),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                _wait_barrier(arena, _BAR_DGATE_TMASTG_READY, dgate_store.stage, dgate_store.phase)
                with K.If(pend_writes), K.Then():
                    for d_coord in (0, 32, 64, 96):
                        K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                            desc_dgate,
                            K.int32(d_coord),
                            K.cast(head, "int32"),
                            K.cast(pend_token, "int32"),
                            arena.ptr_to([_SMEM_DGATE + d_coord * 64]),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                _wait_barrier(arena, _BAR_DB_TMASTG_READY, db_store.stage, db_store.phase)
                with K.If(pend_writes), K.Then():
                    for d_coord in (0, 64):
                        K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                            desc_db,
                            K.int32(d_coord),
                            K.cast(head, "int32"),
                            K.cast(pend_token, "int32"),
                            arena.ptr_to([_SMEM_DB + d_coord * 32]),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                _wait_barrier(arena, _BAR_DV_TMASTG_READY, dv_store.stage, dv_store.phase)
                with K.If(pend_writes), K.Then():
                    for d_coord in (0, 64):
                        K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                            desc_dv,
                            K.int32(d_coord),
                            K.cast(head, "int32"),
                            K.cast(pend_token, "int32"),
                            arena.ptr_to(
                                [_SMEM_DV + K.cast(dv_store.stage, "int32") * 4096 + d_coord * 32]
                            ),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                _wait_barrier(arena, _BAR_DWO_TMASTG_READY, dwo_store.stage, dwo_store.phase)
                with K.If(pend_writes), K.Then():
                    for d_coord in (0, 64):
                        K.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                            desc_dwo,
                            K.int32(d_coord),
                            K.cast(head, "int32"),
                            K.cast(pend_token, "int32"),
                            arena.ptr_to(
                                [_SMEM_DWO + K.cast(dwo_store.stage, "int32") * 4096 + d_coord * 32]
                            ),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                # Six staged releases, one staging buffer freed per step.
                K.ptx.cp.async_.bulk.wait_group.read(5)
                _arrive_barrier(arena, _BAR_DQ_TMASTG_DONE, dq_store.stage)
                K.ptx.cp.async_.bulk.wait_group.read(4)
                _arrive_barrier(arena, _BAR_DK_TMASTG_DONE, dk_store.stage)
                K.ptx.cp.async_.bulk.wait_group.read(3)
                _arrive_barrier(arena, _BAR_DGATE_TMASTG_DONE, dgate_store.stage)
                K.ptx.cp.async_.bulk.wait_group.read(2)
                _arrive_barrier(arena, _BAR_DB_TMASTG_DONE, db_store.stage)
                K.ptx.cp.async_.bulk.wait_group.read(1)
                _arrive_barrier(arena, _BAR_DV_TMASTG_DONE, dv_store.stage)
                K.ptx.cp.async_.bulk.wait_group.read(0)
                _arrive_barrier(arena, _BAR_DWO_TMASTG_DONE, dwo_store.stage)
                dq_store.advance()
                dk_store.advance()
                dgate_store.advance()
                db_store.advance()
                dv_store.advance()
                dwo_store.advance()

            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cend = item[5]
                num_chunks = cend - wstart
                desc_dq = _descriptor_slot(descriptor_workspace, n_desc, 7, batch)
                desc_dk = _descriptor_slot(descriptor_workspace, n_desc, 8, batch)
                desc_dv = _descriptor_slot(descriptor_workspace, n_desc, 9, batch)
                desc_dgate = _descriptor_slot(descriptor_workspace, n_desc, 10, batch)
                desc_dwo = _descriptor_slot(descriptor_workspace, n_desc, 11, batch)
                desc_db = _descriptor_slot(descriptor_workspace, n_desc, 12, batch)
                with K.If(_elected()), K.Then():
                    for desc in (desc_dq, desc_dk, desc_dv, desc_dgate):
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc)
                pending_token = K.local_scalar("int32", init=K.int32(0))
                pending_writes = K.local_scalar("bool", init=K.bool(False))
                with K.serial(num_chunks, unroll=False) as reverse_index:
                    serial = serial_base + reverse_index
                    chunk = cend - 1 - reverse_index
                    token = chunk * 16
                    raw_stage = serial % 2
                    decay_stage = serial % 2
                    inter_stage = serial % 2

                    _wait_barrier(arena, _BAR_A_DONE, inter_stage, (serial // 2 + 1) & 1)
                    _wait_barrier(
                        arena, _BAR_Q_DECAY_K_RESTORE_READY, decay_stage, (serial // 2) & 1
                    )
                    a_acc = K.alloc_local((8,), "float32")
                    for accum_index in range(8):
                        K.assign(a_acc[accum_index], K.float32(0.0))
                    with K.unroll(8) as key_block:
                        a_frag = K.alloc_local((4,), "uint32")
                        b_frag = K.alloc_local((4,), "uint32")
                        a_channel = key_block * 16 + lhs_col
                        b_channel = key_block * 16 + rhs_col
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a_frag[0],
                            a_frag[1],
                            a_frag[2],
                            a_frag[3],
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_Q_DECAY, decay_stage, lhs_row, a_channel)]
                            ),
                        )
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            b_frag[0],
                            b_frag[1],
                            b_frag[2],
                            b_frag[3],
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_K_INV, decay_stage, rhs_row, b_channel)]
                            ),
                        )
                        _mma_m16n16k16(a_acc, a_frag, b_frag)
                    a_words = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        row_coord = row_hi if pair * 2 % 4 >= 2 else row_lo
                        for parity in range(2):
                            accum_index = pair * 2 + parity
                            row_coord = row_hi if accum_index % 4 >= 2 else row_lo
                            col_coord = (accum_index // 4) * 8 + 2 * (lane % 4) + (accum_index & 1)
                            K.assign(
                                a_acc[accum_index],
                                K.if_then_else(row_coord >= col_coord, a_acc[accum_index], 0.0),
                            )
                        K.assign(
                            a_words[pair], _pack_bf16_pair(a_acc[pair * 2], a_acc[pair * 2 + 1])
                        )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to(
                            [_SMEM_INTERMEDIATE + inter_stage * 2560 + store_swizzled * 2]
                        ),
                        *[a_words[i] for i in range(4)],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_A_READY, inter_stage)

                    _wait_barrier(arena, _BAR_U_SMEM_READY, u_consumer.stage, u_consumer.phase)
                    u_consumer.advance()
                    da_acc = K.alloc_local((8,), "float32")
                    for accum_index in range(8):
                        K.assign(da_acc[accum_index], K.float32(0.0))
                    with K.unroll(8) as value_block:
                        a_frag = K.alloc_local((4,), "uint32")
                        b_frag = K.alloc_local((4,), "uint32")
                        a_channel = value_block * 16 + lhs_col
                        b_channel = value_block * 16 + rhs_col
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a_frag[0],
                            a_frag[1],
                            a_frag[2],
                            a_frag[3],
                            arena.ptr_to([_raw_bf16_byte(_SMEM_DO, raw_stage, lhs_row, a_channel)]),
                        )
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            b_frag[0],
                            b_frag[1],
                            b_frag[2],
                            b_frag[3],
                            arena.ptr_to([_raw_bf16_byte(_SMEM_U, 0, rhs_row, b_channel)]),
                        )
                        _mma_m16n16k16(da_acc, a_frag, b_frag)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DO_DONE, raw_stage)
                    _wait_barrier(arena, _BAR_DA_DONE, inter_stage, (serial // 2 + 1) & 1)
                    da_words = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        for parity in range(2):
                            accum_index = pair * 2 + parity
                            row_coord = row_hi if accum_index % 4 >= 2 else row_lo
                            col_coord = (accum_index // 4) * 8 + 2 * (lane % 4) + (accum_index & 1)
                            K.assign(
                                da_acc[accum_index],
                                K.if_then_else(row_coord >= col_coord, da_acc[accum_index], 0.0),
                            )
                        K.assign(
                            da_words[pair], _pack_bf16_pair(da_acc[pair * 2], da_acc[pair * 2 + 1])
                        )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to(
                            [_SMEM_INTERMEDIATE + inter_stage * 2560 + 1024 + store_swizzled * 2]
                        ),
                        *[da_words[i] for i in range(4)],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DA_READY, inter_stage)

                    with K.If(reverse_index > 0), K.Then():
                        store_pending(
                            pending_token,
                            pending_writes,
                            head,
                            desc_dq,
                            desc_dk,
                            desc_dv,
                            desc_dgate,
                            desc_db,
                            desc_dwo,
                        )
                    K.assign(pending_token, token)
                    K.assign(pending_writes, chunk < wend)

                with K.If(num_chunks > 0), K.Then():
                    store_pending(
                        pending_token,
                        pending_writes,
                        head,
                        desc_dq,
                        desc_dk,
                        desc_dv,
                        desc_dgate,
                        desc_db,
                        desc_dwo,
                    )
                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_READY, sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(
                        tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_consumer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_DONE, sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
        with cg0:
            K.ptx.bar.sync(K.uint32(3), K.uint32(416))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_SMEM_TMEM_MAILBOX]))
            warp_in_group = K.warp_id_in_role()
            channel = warp_in_group * 32 + lane
            value_channel = (warp % 4) * 32 + lane
            tmem_row = (tmem_base >> 16) + (warp % 4) * 32
            tmem_col = tmem_base & 0xFFFF
            tile = K.local_scalar("int32", init=K.cta_id())
            serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            state_consumer = K.PipelineState(1, phase=0)
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                cend = item[5]
                bos = item[6]
                eos = item[7]
                sequence_length = eos - bos
                num_chunks = cend - wstart
                safe_a = K.local_scalar("float32", init=K.float32(1.0))
                safe_bias = K.local_scalar("float32", init=K.float32(0.0))
                if safe_gate:
                    with K.If(num_chunks > 0), K.Then():
                        a_value = K.local_scalar("float32")
                        K.ptx.ld.global_.f32(a_value, a_log.ptr_to([head]))
                        K.assign(safe_a, _exp2_approx(a_value * K.float32(1.4426950408889634)))
                        K.ptx.ld.global_.f32(
                            safe_bias, dt_bias.ptr_to([K.cast(head, "int64") * 128 + channel])
                        )
                with K.serial(num_chunks, unroll=False) as reverse_index:
                    serial = serial_base + reverse_index
                    chunk = cend - 1 - reverse_index
                    token_base = chunk * 16
                    raw_stage = serial % 2
                    decay_stage = serial % 2

                    _wait_barrier(arena, _BAR_GATE_READY, raw_stage, (serial // 2) & 1)
                    _wait_barrier(arena, _BAR_Q_READY, raw_stage, (serial // 2) & 1)
                    _wait_barrier(arena, _BAR_K_READY, raw_stage, (serial // 2) & 1)

                    gate_prefix = K.alloc_local((16,), "float32")
                    with K.unroll(16) as row:
                        K.ptx.ld.shared.f32(
                            gate_prefix[row],
                            arena.ptr_to([_raw_f32_byte(_SMEM_GATE, raw_stage, row, channel)]),
                        )
                    if safe_gate:
                        with K.unroll(8) as row_pair:
                            row0 = row_pair * 2
                            row1 = row0 + 1
                            gate0 = (
                                _tanh_approx(safe_a * (gate_prefix[row0] + safe_bias) * 0.5) * 0.5
                                + 0.5
                            )
                            gate1 = (
                                _tanh_approx(safe_a * (gate_prefix[row1] + safe_bias) * 0.5) * 0.5
                                + 0.5
                            )
                            if full_tiles:
                                K.assign(gate_prefix[row0], K.float32(-7.213475204444817) * gate0)
                                K.assign(gate_prefix[row1], K.float32(-7.213475204444817) * gate1)
                            else:
                                K.assign(
                                    gate_prefix[row0],
                                    K.if_then_else(
                                        token_base + row0 < sequence_length,
                                        K.float32(-7.213475204444817) * gate0,
                                        K.float32(0.0),
                                    ),
                                )
                                K.assign(
                                    gate_prefix[row1],
                                    K.if_then_else(
                                        token_base + row1 < sequence_length,
                                        K.float32(-7.213475204444817) * gate1,
                                        K.float32(0.0),
                                    ),
                                )
                    else:
                        with K.unroll(16) as row:
                            with K.If(token_base + row < sequence_length), K.Then():
                                K.assign(
                                    gate_prefix[row],
                                    gate_prefix[row] * K.float32(1.4426950408889634),
                                )
                            with K.If(token_base + row >= sequence_length), K.Then():
                                K.assign(gate_prefix[row], K.float32(0.0))
                    prefix = K.local_scalar("float32", init=K.float32(0.0))
                    with K.unroll(8) as row_pair:
                        row0 = row_pair * 2
                        row1 = row0 + 1
                        prefix0, pair_sum = _fadd2(
                            prefix, gate_prefix[row0], gate_prefix[row0], gate_prefix[row1]
                        )
                        prefix1 = prefix + pair_sum
                        K.assign(gate_prefix[row0], prefix0)
                        K.assign(gate_prefix[row1], prefix1)
                        K.assign(prefix, prefix1)
                    with K.unroll(16) as row:
                        K.assign(gate_prefix[row], _exp2_approx(gate_prefix[row]))

                    _wait_barrier(arena, _BAR_DECAY_DONE, decay_stage, (serial // 2 + 1) & 1)
                    with K.unroll(16) as row:
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_raw_f32_byte(_SMEM_GATE, raw_stage, row, channel)]),
                            gate_prefix[row],
                        )
                    diagonal_block = channel // 16
                    diagonal_coord = channel % 16
                    diagonal_linear = diagonal_block * 256 + diagonal_coord * 16 + diagonal_coord
                    diagonal_swizzled = K.bitwise_xor(
                        diagonal_linear,
                        K.shift_left(
                            K.bitwise_and(K.shift_right(diagonal_linear, K.uint32(6)), K.uint32(1)),
                            K.uint32(3),
                        ),
                    )
                    K.ptx.st.shared.b16(
                        arena.ptr_to([_SMEM_DIAG + decay_stage * 4096 + diagonal_swizzled * 2]),
                        K.reinterpret("uint16", K.cast(gate_prefix[15], "bfloat16")),
                    )

                    qk_stage = serial % 4
                    _wait_barrier(arena, _BAR_QK_RAW_DONE, qk_stage, (serial // 4 + 1) & 1)
                    q_words = K.alloc_local((8,), "uint32")
                    k_words = K.alloc_local((8,), "uint32")
                    raw_segment = channel // 64
                    raw_channel = channel % 64
                    with K.unroll(8) as pair:
                        q_lo_bits = K.local_scalar("uint16")
                        q_hi_bits = K.local_scalar("uint16")
                        k_lo_bits = K.local_scalar("uint16")
                        k_hi_bits = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(
                            q_lo_bits,
                            arena.ptr_to([_raw_bf16_byte(_SMEM_Q, raw_stage, pair * 2, channel)]),
                        )
                        K.ptx.ld.shared.b16(
                            q_hi_bits,
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_Q, raw_stage, pair * 2 + 1, channel)]
                            ),
                        )
                        K.ptx.ld.shared.b16(
                            k_lo_bits,
                            arena.ptr_to([_raw_bf16_byte(_SMEM_K, raw_stage, pair * 2, channel)]),
                        )
                        K.ptx.ld.shared.b16(
                            k_hi_bits,
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_K, raw_stage, pair * 2 + 1, channel)]
                            ),
                        )
                        K.assign(
                            q_words[pair],
                            K.bitwise_or(
                                K.cast(q_lo_bits, "uint32"),
                                K.shift_left(K.cast(q_hi_bits, "uint32"), K.uint32(16)),
                            ),
                        )
                        K.assign(
                            k_words[pair],
                            K.bitwise_or(
                                K.cast(k_lo_bits, "uint32"),
                                K.shift_left(K.cast(k_hi_bits, "uint32"), K.uint32(16)),
                            ),
                        )
                    q_tmem = (
                        tmem_col + _TMEM_Q_RAW + qk_stage * 8 + K.shift_left(tmem_row, K.int32(16))
                    )
                    k_tmem = (
                        tmem_col + _TMEM_K_RAW + qk_stage * 8 + K.shift_left(tmem_row, K.int32(16))
                    )
                    K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                        K.cast(q_tmem, "uint32"), *[q_words[i] for i in range(8)]
                    )
                    K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                        K.cast(k_tmem, "uint32"), *[k_words[i] for i in range(8)]
                    )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    _arrive_barrier(arena, _BAR_QK_RAW_READY, qk_stage)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))

                    row_group_start = warp_in_group * 4
                    row_in_group = lane // 8
                    lane_in_group = lane % 8
                    decay_row = row_group_start + row_in_group
                    raw_q = K.alloc_local((16,), "float32")
                    raw_k = K.alloc_local((16,), "float32")
                    qk0_lo = K.local_scalar("float32", init=_opaque_f32_zero())
                    qk0_hi = K.local_scalar("float32", init=_opaque_f32_zero())
                    qk1_lo = K.local_scalar("float32", init=_opaque_f32_zero())
                    qk1_hi = K.local_scalar("float32", init=_opaque_f32_zero())
                    with K.unroll(2) as half:
                        dim_base = half * 64 + lane_in_group * 8
                        q_fragment = K.alloc_local((4,), "uint32")
                        k_fragment = K.alloc_local((4,), "uint32")
                        K.ptx.ld.shared.v4.b32(
                            q_fragment[0],
                            q_fragment[1],
                            q_fragment[2],
                            q_fragment[3],
                            arena.ptr_to([_raw_bf16_byte(_SMEM_Q, raw_stage, decay_row, dim_base)]),
                        )
                        K.ptx.ld.shared.v4.b32(
                            k_fragment[0],
                            k_fragment[1],
                            k_fragment[2],
                            k_fragment[3],
                            arena.ptr_to([_raw_bf16_byte(_SMEM_K, raw_stage, decay_row, dim_base)]),
                        )
                        with K.unroll(8) as dim_offset:
                            shift = K.cast((dim_offset % 2) * 16, "uint32")
                            q_word = K.shift_right(q_fragment[dim_offset // 2], shift)
                            k_word = K.shift_right(k_fragment[dim_offset // 2], shift)
                            q_bits = K.cast(q_word, "uint16")
                            k_bits = K.cast(k_word, "uint16")
                            K.ptx.cvt.f32.bf16(raw_q[half * 8 + dim_offset], q_bits)
                            K.ptx.cvt.f32.bf16(raw_k[half * 8 + dim_offset], k_bits)
                            if l2norm:
                                if dim_offset % 2 == 0:
                                    next_q, next_k = _ffma2(
                                        raw_q[half * 8 + dim_offset],
                                        raw_k[half * 8 + dim_offset],
                                        raw_q[half * 8 + dim_offset],
                                        raw_k[half * 8 + dim_offset],
                                        qk0_lo,
                                        qk0_hi,
                                    )
                                    K.assign(qk0_lo, next_q)
                                    K.assign(qk0_hi, next_k)
                                else:
                                    next_q, next_k = _ffma2(
                                        raw_q[half * 8 + dim_offset],
                                        raw_k[half * 8 + dim_offset],
                                        raw_q[half * 8 + dim_offset],
                                        raw_k[half * 8 + dim_offset],
                                        qk1_lo,
                                        qk1_hi,
                                    )
                                    K.assign(qk1_lo, next_q)
                                    K.assign(qk1_hi, next_k)
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_Q_DONE, raw_stage)
                    _arrive_barrier(arena, _BAR_K_DONE, raw_stage)
                    # beta_done carries CG0 + CG2 = 256: both groups read the tile.
                    _arrive_barrier(arena, _BAR_BETA_DONE, raw_stage)

                    q_inv_norm = K.local_scalar("float32", init=K.float32(1.0))
                    k_inv_norm = K.local_scalar("float32", init=K.float32(1.0))
                    if l2norm:
                        q_sum = K.local_scalar("float32", init=qk0_lo + qk1_lo)
                        k_sum = K.local_scalar("float32", init=qk0_hi + qk1_hi)
                        for delta in (4, 2, 1):
                            K.assign(q_sum, q_sum + _shuffle_xor_f32(q_sum, delta))
                            K.assign(k_sum, k_sum + _shuffle_xor_f32(k_sum, delta))
                        K.assign(q_inv_norm, _rsqrt_approx(K.max(q_sum, 1.0e-24)))
                        K.assign(k_inv_norm, _rsqrt_approx(K.max(k_sum, 1.0e-24)))
                        with K.If(lane_in_group == 0), K.Then():
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_SMEM_NORM + (qk_stage * 32 + decay_row) * 4]),
                                q_inv_norm,
                            )
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_SMEM_NORM + (qk_stage * 32 + 16 + decay_row) * 4]),
                                k_inv_norm,
                            )
                    q_scale = q_inv_norm * scale

                    exp_g = K.alloc_local((16,), "float32")
                    exp_last = K.alloc_local((16,), "float32")
                    with K.unroll(2) as half:
                        dim_base = half * 64 + lane_in_group * 8
                        with K.unroll(2) as group:
                            fragment_base = half * 8 + group * 4
                            K.ptx.ld.shared.v4.f32(
                                exp_g[fragment_base],
                                exp_g[fragment_base + 1],
                                exp_g[fragment_base + 2],
                                exp_g[fragment_base + 3],
                                arena.ptr_to(
                                    [
                                        _raw_f32_byte(
                                            _SMEM_GATE, raw_stage, decay_row, dim_base + group * 4
                                        )
                                    ]
                                ),
                            )
                            K.ptx.ld.shared.v4.f32(
                                exp_last[fragment_base],
                                exp_last[fragment_base + 1],
                                exp_last[fragment_base + 2],
                                exp_last[fragment_base + 3],
                                arena.ptr_to(
                                    [_raw_f32_byte(_SMEM_GATE, raw_stage, 15, dim_base + group * 4)]
                                ),
                            )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_GATE_DONE, raw_stage)

                    k_inverse_words = K.alloc_local((8,), "uint32")
                    with K.unroll(2) as half:
                        dim_base = half * 64 + lane_in_group * 8
                        decay_words = K.alloc_local((4,), "uint32")
                        # The per-key erase gate arrives as a [BT, DK] tile and is
                        # folded into K_decay only; K_inv stays ungated.  This is
                        # the whole of the GDN-2 gating delta on the key side.
                        raw_beta = K.alloc_local((8,), "float32")
                        with K.unroll(2) as beta_group:
                            beta_bits = K.alloc_local((4,), "uint32")
                            K.ptx.ld.shared.v4.b32(
                                beta_bits[0],
                                beta_bits[1],
                                beta_bits[2],
                                beta_bits[3],
                                arena.ptr_to(
                                    [
                                        _raw_bf16_byte(
                                            _SMEM_BETA,
                                            raw_stage,
                                            decay_row,
                                            dim_base + beta_group * 4,
                                        )
                                    ]
                                ),
                            )
                            for word in range(4):
                                _unpack_bf16_pair(
                                    beta_bits[word],
                                    raw_beta[beta_group * 4 + word * 2],
                                    raw_beta[beta_group * 4 + word * 2 + 1],
                                )
                        with K.unroll(4) as pair:
                            reg = half * 8 + pair * 2
                            k0, k1 = _fmul2(raw_k[reg], raw_k[reg + 1], k_inv_norm, k_inv_norm)
                            k_pair = _pack_bf16_pair(k0, k1)
                            beta0, beta1 = _fmul2(
                                k0, k1, raw_beta[pair * 2], raw_beta[pair * 2 + 1]
                            )
                            k_beta_pair = _pack_bf16_pair(beta0, beta1)
                            exp_pair = _pack_bf16_pair(exp_g[reg], exp_g[reg + 1])
                            inverse_pair = _pack_bf16_pair(
                                _rcp_approx(exp_g[reg]), _rcp_approx(exp_g[reg + 1])
                            )
                            K.ptx.mul.bf16x2(decay_words[pair], k_beta_pair, exp_pair)
                            K.ptx.mul.bf16x2(k_inverse_words[half * 4 + pair], k_pair, inverse_pair)
                        operand_byte = _raw_bf16_byte(
                            _SMEM_K_DECAY, decay_stage, decay_row, dim_base
                        )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to([operand_byte]),
                            decay_words[0],
                            decay_words[1],
                            decay_words[2],
                            decay_words[3],
                        )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_K_INV, decay_stage, decay_row, dim_base)]
                            ),
                            k_inverse_words[half * 4],
                            k_inverse_words[half * 4 + 1],
                            k_inverse_words[half * 4 + 2],
                            k_inverse_words[half * 4 + 3],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_K_DECAY_INV_READY, decay_stage)

                    with K.unroll(2) as half:
                        dim_base = half * 64 + lane_in_group * 8
                        q_decay_words = K.alloc_local((4,), "uint32")
                        restore_words = K.alloc_local((4,), "uint32")
                        with K.unroll(4) as pair:
                            reg = half * 8 + pair * 2
                            q0, q1 = _fmul2(raw_q[reg], raw_q[reg + 1], q_scale, q_scale)
                            q_pair = _pack_bf16_pair(q0, q1)
                            exp_pair = _pack_bf16_pair(exp_g[reg], exp_g[reg + 1])
                            K.ptx.mul.bf16x2(q_decay_words[pair], q_pair, exp_pair)
                            last_pair = _pack_bf16_pair(exp_last[reg], exp_last[reg + 1])
                            K.ptx.mul.bf16x2(
                                restore_words[pair], k_inverse_words[half * 4 + pair], last_pair
                            )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_Q_DECAY, decay_stage, decay_row, dim_base)]
                            ),
                            q_decay_words[0],
                            q_decay_words[1],
                            q_decay_words[2],
                            q_decay_words[3],
                        )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to(
                                [_raw_bf16_byte(_SMEM_K_RESTORE, decay_stage, decay_row, dim_base)]
                            ),
                            restore_words[0],
                            restore_words[1],
                            restore_words[2],
                            restore_words[3],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_Q_DECAY_K_RESTORE_READY, decay_stage)

                    _wait_barrier(arena, _BAR_STATE_INP_DONE, serial % 2, (serial // 2 + 1) & 1)
                    _wait_barrier(arena, _BAR_STATE_INP_CG2_DONE, serial % 2, (serial // 2 + 1) & 1)
                    first_state_chunk = 0 if use_initial_state else 1
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(
                            arena, _BAR_STATE_READY, state_consumer.stage, state_consumer.phase
                        )
                        with K.unroll(2) as value_segment:
                            with K.unroll(8) as value_group:
                                state_words = K.alloc_local((4,), "uint32")
                                with K.unroll(4) as pair:
                                    lo_bits = K.local_scalar("uint16")
                                    hi_bits = K.local_scalar("uint16")
                                    key_lo = value_segment * 64 + value_group * 8 + pair * 2
                                    key_hi = key_lo + 1
                                    K.ptx.ld.shared.b16(
                                        lo_bits,
                                        arena.ptr_to(
                                            [
                                                _SMEM_STATE
                                                + (
                                                    (value_channel // 64) * 8192
                                                    + key_lo * 64
                                                    + _swizzle_xor_128b(
                                                        key_lo, value_channel % 64, 2
                                                    )
                                                )
                                                * 2
                                            ]
                                        ),
                                    )
                                    K.ptx.ld.shared.b16(
                                        hi_bits,
                                        arena.ptr_to(
                                            [
                                                _SMEM_STATE
                                                + (
                                                    (value_channel // 64) * 8192
                                                    + key_hi * 64
                                                    + _swizzle_xor_128b(
                                                        key_hi, value_channel % 64, 2
                                                    )
                                                )
                                                * 2
                                            ]
                                        ),
                                    )
                                    K.assign(
                                        state_words[pair],
                                        K.bitwise_or(
                                            K.cast(lo_bits, "uint32"),
                                            K.shift_left(K.cast(hi_bits, "uint32"), K.uint32(16)),
                                        ),
                                    )
                                state_tmem = (
                                    tmem_col
                                    + _TMEM_STATE_INPUT
                                    + (serial % 2) * 64
                                    + value_segment * 32
                                    + value_group * 4
                                    + K.shift_left(tmem_row, K.int32(16))
                                )
                                K.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
                                    K.cast(state_tmem, "uint32"),
                                    state_words[0],
                                    state_words[1],
                                    state_words[2],
                                    state_words[3],
                                )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _arrive_barrier(arena, _BAR_STATE_CG0_DONE, state_consumer.stage)
                        state_consumer.advance()
                    _arrive_barrier(arena, _BAR_STATE_INP_READY, serial % 2)

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_READY, sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(
                        tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_consumer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_DONE, sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
        with cg2:
            K.ptx.bar.sync(K.uint32(3), K.uint32(416))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_SMEM_TMEM_MAILBOX]))
            tmem_col = tmem_base & 0xFFFF
            tmem_alloc_row = tmem_base >> 16
            warp_in_group = K.warp_id_in_role()
            subpartition = warp % 4
            channel = subpartition * 32 + lane
            tmem_row = tmem_alloc_row + subpartition * 32
            tile = K.local_scalar("int32", init=K.cta_id())
            serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            dq_consumer = K.PipelineState(1, phase=0)
            dk_decay_consumer = K.PipelineState(1, phase=0)
            dk_inverse_consumer = K.PipelineState(1, phase=0)
            dk_restore_consumer = K.PipelineState(1, phase=0)
            dstate_smem_consumer = K.PipelineState(1, phase=0)
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                wstart = item[2]
                wend = item[3]
                cend = item[5]
                num_chunks = cend - wstart
                with K.serial(num_chunks, unroll=False) as reverse_index:
                    serial = serial_base + reverse_index
                    chunk = cend - 1 - reverse_index
                    raw_stage = serial % 2
                    decay_stage = serial % 2
                    has_dstate = reverse_index > 0
                    if use_dstate_in:
                        has_dstate = K.bool(True)

                    _wait_barrier(arena, _BAR_K_DECAY_INV_READY, decay_stage, (serial // 2) & 1)
                    gate_decay = K.alloc_local((16,), "float32")
                    with K.unroll(16) as token:
                        K.ptx.ld.shared.f32(
                            gate_decay[token],
                            arena.ptr_to([_raw_f32_byte(_SMEM_GATE, raw_stage, token, channel)]),
                        )
                    gate_last = gate_decay[15]
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_GATE_DONE, raw_stage)

                    qk_stage = serial % 4
                    qraw_col = tmem_col + _TMEM_Q_RAW + qk_stage * 8
                    kraw_col = tmem_col + _TMEM_K_RAW + qk_stage * 8
                    _wait_barrier(arena, _BAR_QK_RAW_READY, qk_stage, (serial // 4) & 1)

                    dgate_last = K.local_scalar("float32", init=K.float32(0.0))
                    _wait_barrier(arena, _BAR_STATE_INP_READY, serial % 2, (serial // 2) & 1)
                    with K.If(has_dstate), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_SMEM_READY,
                            dstate_smem_consumer.stage,
                            dstate_smem_consumer.phase,
                        )
                        dstate_smem_consumer.advance()
                        with K.unroll(2) as value_plane:
                            with K.unroll(2) as row_half:
                                state_words = K.alloc_local((16,), "uint32")
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                                    *[state_words[i] for i in range(16)],
                                    K.cast(
                                        tmem_col
                                        + _TMEM_STATE_INPUT
                                        + (serial % 2) * 64
                                        + value_plane * 32
                                        + row_half * 16
                                        + K.shift_left(tmem_row, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                                K.ptx.tcgen05.wait__ld.sync.aligned()
                                hacc = K.alloc_local((8,), "float32")
                                for accum_slot in range(8):
                                    K.assign(hacc[accum_slot], _opaque_f32_zero())
                                with K.unroll(16) as pair:
                                    state_lo = K.cast(state_words[pair], "uint16")
                                    state_hi = K.cast(
                                        K.shift_right(state_words[pair], K.uint32(16)), "uint16"
                                    )
                                    value0 = value_plane * 64 + row_half * 32 + pair * 2
                                    h0_bits = K.local_scalar("uint16")
                                    h1_bits = K.local_scalar("uint16")
                                    K.ptx.ld.shared.b16(
                                        h0_bits,
                                        arena.ptr_to(
                                            [
                                                _SMEM_DSTATE
                                                + (
                                                    (channel // 64) * 8192
                                                    + value0 * 64
                                                    + _swizzle_xor_128b(value0, channel % 64, 2)
                                                )
                                                * 2
                                            ]
                                        ),
                                    )
                                    K.ptx.ld.shared.b16(
                                        h1_bits,
                                        arena.ptr_to(
                                            [
                                                _SMEM_DSTATE
                                                + (
                                                    (channel // 64) * 8192
                                                    + (value0 + 1) * 64
                                                    + _swizzle_xor_128b(value0 + 1, channel % 64, 2)
                                                )
                                                * 2
                                            ]
                                        ),
                                    )
                                    accum_lo = (2 * pair) % 8
                                    accum_hi = (2 * pair + 1) % 8
                                    next_lo = K.local_scalar("float32")
                                    next_hi = K.local_scalar("float32")
                                    K.ptx.fma.rn.f32.bf16(
                                        next_lo, h0_bits, state_lo, hacc[accum_lo]
                                    )
                                    K.ptx.fma.rn.f32.bf16(
                                        next_hi, h1_bits, state_hi, hacc[accum_hi]
                                    )
                                    K.assign(hacc[accum_lo], next_lo)
                                    K.assign(hacc[accum_hi], next_hi)
                                pa0, pb0 = _fadd2(hacc[0], hacc[2], hacc[4], hacc[6])
                                pa1, pb1 = _fadd2(hacc[1], hacc[3], hacc[5], hacc[7])
                                part_a, part_b = _fadd2(pa0, pb0, pa1, pb1)
                                K.assign(dgate_last, dgate_last + part_a + part_b)
                        _arrive_barrier(arena, _BAR_DSTATE_SMEM_CG2_DONE)
                    _arrive_barrier(arena, _BAR_STATE_INP_CG2_DONE, serial % 2)

                    dq_values = K.alloc_local((16,), "float32")
                    dk_values = K.alloc_local((16,), "float32")
                    dgate_values = K.alloc_local((16,), "float32")
                    last_dot = K.alloc_local((4,), "float32")
                    for slot in range(4):
                        K.assign(last_dot[slot], K.float32(0.0))
                    for token in range(16):
                        K.assign(dk_values[token], K.float32(0.0))

                    with K.If(has_dstate), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_DK_RESTORE_PART_ACC_READY,
                            dk_restore_consumer.stage,
                            dk_restore_consumer.phase,
                        )
                        dk_restore_consumer.advance()
                        restore = K.alloc_local((16,), "float32")
                        kraw = K.alloc_local((8,), "uint32")
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *[restore[i] for i in range(16)],
                            K.cast(
                                tmem_col + _TMEM_DK_RESTORE + K.shift_left(tmem_row, K.int32(16)),
                                "uint32",
                            ),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                            *[kraw[i] for i in range(8)],
                            K.cast(kraw_col + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        with K.unroll(16) as token:
                            K.assign(
                                dk_values[token],
                                gate_last * _rcp_approx(gate_decay[token]) * restore[token],
                            )
                            k_lo = K.local_scalar("float32")
                            k_hi = K.local_scalar("float32")
                            _unpack_bf16_pair(kraw[token // 2], k_lo, k_hi)
                            k_value = K.local_scalar("float32")
                            K.assign(k_value, K.if_then_else(token % 2 == 0, k_lo, k_hi))
                            if l2norm:
                                k_norm = K.local_scalar("float32")
                                K.ptx.ld.shared.f32(
                                    k_norm,
                                    arena.ptr_to([_SMEM_NORM + (qk_stage * 32 + 16 + token) * 4]),
                                )
                                K.assign(k_value, k_value * k_norm)
                            K.assign(
                                last_dot[token % 4],
                                last_dot[token % 4] + k_value * dk_values[token],
                            )

                    _wait_barrier(arena, _BAR_DQ_ACC_READY, dq_consumer.stage, dq_consumer.phase)
                    dq_consumer.advance()
                    dq_acc = K.alloc_local((16,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                        *[dq_acc[i] for i in range(16)],
                        K.cast(tmem_col + _TMEM_DQ + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                    )
                    with K.unroll(8) as pair:
                        token = pair * 2
                        scaled_lo, scaled_hi = _fmul2(
                            gate_decay[token], gate_decay[token + 1], scale, scale
                        )
                        dq_lo, dq_hi = _fmul2(
                            scaled_lo, scaled_hi, dq_acc[token], dq_acc[token + 1]
                        )
                        K.assign(dq_values[token], dq_lo)
                        K.assign(dq_values[token + 1], dq_hi)

                    _wait_barrier(
                        arena,
                        _BAR_DK_INV_PART_ACC_READY,
                        dk_inverse_consumer.stage,
                        dk_inverse_consumer.phase,
                    )
                    dk_inverse_consumer.advance()
                    inverse = K.alloc_local((16,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                        *[inverse[i] for i in range(16)],
                        K.cast(
                            tmem_col + _TMEM_DK_INV + K.shift_left(tmem_row, K.int32(16)), "uint32"
                        ),
                    )
                    with K.unroll(16) as token:
                        K.assign(
                            dk_values[token],
                            dk_values[token] + inverse[token] * _rcp_approx(gate_decay[token]),
                        )

                    _wait_barrier(
                        arena,
                        _BAR_DK_DECAY_PART_ACC_READY,
                        dk_decay_consumer.stage,
                        dk_decay_consumer.phase,
                    )
                    dk_decay_consumer.advance()
                    decay = K.alloc_local((16,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                        *[decay[i] for i in range(16)],
                        K.cast(
                            tmem_col + _TMEM_DK_DECAY + K.shift_left(tmem_row, K.int32(16)),
                            "uint32",
                        ),
                    )
                    # GDN-2 seeds BOTH dBeta and dGate from the dK_decay drain:
                    # dBeta is per key channel (k_n * dk_decay), and dGate takes the
                    # Beta-weighted share of the same product.
                    kd_words = K.alloc_local((8,), "uint32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                        *[kd_words[i] for i in range(8)],
                        K.cast(kraw_col + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                    )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    db_values = K.alloc_local((16,), "float32")
                    beta_self = K.alloc_local((16,), "float32")
                    with K.unroll(16) as token:
                        beta_bits = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(
                            beta_bits,
                            arena.ptr_to([_raw_bf16_byte(_SMEM_BETA, raw_stage, token, channel)]),
                        )
                        beta_value = K.local_scalar("float32")
                        K.ptx.cvt.f32.bf16(beta_value, beta_bits)
                        if beta_sigmoid:
                            K.assign(beta_value, _tanh_approx(beta_value * 0.5) * 0.5 + 0.5)
                            # The source rounds sigmoid(beta) through the io dtype
                            # before using it, and the gradient depends on that.
                            K.assign(beta_value, K.cast(K.cast(beta_value, "bfloat16"), "float32"))
                        K.assign(beta_self[token], beta_value)

                        kd_lo = K.local_scalar("float32")
                        kd_hi = K.local_scalar("float32")
                        _unpack_bf16_pair(kd_words[token // 2], kd_lo, kd_hi)
                        k_native = K.local_scalar("float32")
                        K.assign(k_native, K.if_then_else(token % 2 == 0, kd_lo, kd_hi))
                        if l2norm:
                            k_scale = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                k_scale,
                                arena.ptr_to([_SMEM_NORM + (qk_stage * 32 + 16 + token) * 4]),
                            )
                            K.assign(k_native, k_native * k_scale)

                        dk_decay = K.local_scalar("float32")
                        K.assign(dk_decay, -gate_decay[token] * decay[token])
                        K.assign(db_values[token], k_native * dk_decay)
                        K.assign(dgate_values[token], beta_value * dk_decay)
                        K.assign(dk_values[token], dk_values[token] + dgate_values[token])
                    _arrive_barrier(arena, _BAR_DQK_ACC_DONE)

                    qraw = K.alloc_local((8,), "uint32")
                    kraw = K.alloc_local((8,), "uint32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                        *[qraw[i] for i in range(8)],
                        K.cast(qraw_col + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                    )
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                        *[kraw[i] for i in range(8)],
                        K.cast(kraw_col + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                    )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    with K.unroll(16) as token:
                        q_lo = K.local_scalar("float32")
                        q_hi = K.local_scalar("float32")
                        k_lo = K.local_scalar("float32")
                        k_hi = K.local_scalar("float32")
                        _unpack_bf16_pair(qraw[token // 2], q_lo, q_hi)
                        _unpack_bf16_pair(kraw[token // 2], k_lo, k_hi)
                        q_value = K.local_scalar("float32")
                        k_value = K.local_scalar("float32")
                        K.assign(q_value, K.if_then_else(token % 2 == 0, q_lo, q_hi))
                        K.assign(k_value, K.if_then_else(token % 2 == 0, k_lo, k_hi))
                        if l2norm:
                            q_norm = K.local_scalar("float32")
                            k_norm = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(
                                q_norm, arena.ptr_to([_SMEM_NORM + (qk_stage * 32 + token) * 4])
                            )
                            K.ptx.ld.shared.f32(
                                k_norm,
                                arena.ptr_to([_SMEM_NORM + (qk_stage * 32 + 16 + token) * 4]),
                            )
                            K.assign(q_value, q_value * q_norm)
                            K.assign(k_value, k_value * k_norm)
                        K.assign(
                            dgate_values[token],
                            q_value * dq_values[token]
                            + beta_self[token] * db_values[token]
                            - k_value * (dk_values[token] - dgate_values[token]),
                        )
                        if beta_sigmoid:
                            # Applied AFTER dGate consumed db_values: the source
                            # feeds dGate the pre-chain-rule value, and swapping
                            # these silently corrupts dGate.
                            K.assign(
                                db_values[token],
                                db_values[token]
                                * (beta_self[token] - beta_self[token] * beta_self[token]),
                            )
                    K.assign(
                        dgate_values[15],
                        dgate_values[15] + last_dot[0] + last_dot[1] + last_dot[2] + last_dot[3],
                    )

                    if l2norm:
                        for grad, raw_column, norm_offset in (
                            (dq_values, qraw_col, 0),
                            (dk_values, kraw_col, 16),
                        ):
                            dots = K.alloc_local((16,), "float32")
                            for half in range(2):
                                half_words = K.alloc_local((4,), "uint32")
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
                                    *[half_words[i] for i in range(4)],
                                    K.cast(
                                        raw_column + half * 4 + K.shift_left(tmem_row, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                                for pair in range(4):
                                    token = half * 8 + pair * 2
                                    raw_lo = K.local_scalar("float32")
                                    raw_hi = K.local_scalar("float32")
                                    _unpack_bf16_pair(half_words[pair], raw_lo, raw_hi)
                                    norm_lo = K.local_scalar("float32")
                                    norm_hi = K.local_scalar("float32")
                                    K.ptx.ld.shared.f32(
                                        norm_lo,
                                        arena.ptr_to(
                                            [_SMEM_NORM + (qk_stage * 32 + norm_offset + token) * 4]
                                        ),
                                    )
                                    K.ptx.ld.shared.f32(
                                        norm_hi,
                                        arena.ptr_to(
                                            [
                                                _SMEM_NORM
                                                + (qk_stage * 32 + norm_offset + token + 1) * 4
                                            ]
                                        ),
                                    )
                                    product_lo, product_hi = _fmul2(
                                        grad[token], grad[token + 1], raw_lo, raw_hi
                                    )
                                    product_lo, product_hi = _fmul2(
                                        product_lo, product_hi, norm_lo, norm_hi
                                    )
                                    K.assign(dots[token], product_lo)
                                    K.assign(dots[token + 1], product_hi)
                            for delta in (1, 2, 4, 8, 16):
                                for pair in range(8):
                                    token = pair * 2
                                    other_lo = _shuffle_xor_f32(dots[token], delta)
                                    other_hi = _shuffle_xor_f32(dots[token + 1], delta)
                                    sum_lo, sum_hi = _fadd2(
                                        dots[token], dots[token + 1], other_lo, other_hi
                                    )
                                    K.assign(dots[token], sum_lo)
                                    K.assign(dots[token + 1], sum_hi)
                            with K.If(lane == 0), K.Then():
                                for token in range(16):
                                    K.ptx.st.shared.f32(
                                        arena.ptr_to(
                                            [_SMEM_RED1 + (subpartition * 16 + token) * 4]
                                        ),
                                        dots[token],
                                    )
                            K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                            for half in range(2):
                                half_words = K.alloc_local((4,), "uint32")
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
                                    *[half_words[i] for i in range(4)],
                                    K.cast(
                                        raw_column + half * 4 + K.shift_left(tmem_row, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                                for pair in range(4):
                                    token = half * 8 + pair * 2
                                    partials = K.alloc_local((8,), "float32")
                                    for reduction_warp in range(4):
                                        K.ptx.ld.shared.f32(
                                            partials[reduction_warp * 2],
                                            arena.ptr_to(
                                                [_SMEM_RED1 + (reduction_warp * 16 + token) * 4]
                                            ),
                                        )
                                        K.ptx.ld.shared.f32(
                                            partials[reduction_warp * 2 + 1],
                                            arena.ptr_to(
                                                [_SMEM_RED1 + (reduction_warp * 16 + token + 1) * 4]
                                            ),
                                        )
                                    dot_lo, dot_hi = _fadd2(
                                        partials[0], partials[1], partials[2], partials[3]
                                    )
                                    dot_lo, dot_hi = _fadd2(
                                        dot_lo, dot_hi, partials[4], partials[5]
                                    )
                                    dot_lo, dot_hi = _fadd2(
                                        dot_lo, dot_hi, partials[6], partials[7]
                                    )
                                    raw_lo = K.local_scalar("float32")
                                    raw_hi = K.local_scalar("float32")
                                    _unpack_bf16_pair(half_words[pair], raw_lo, raw_hi)
                                    norm_lo = K.local_scalar("float32")
                                    norm_hi = K.local_scalar("float32")
                                    K.ptx.ld.shared.f32(
                                        norm_lo,
                                        arena.ptr_to(
                                            [_SMEM_NORM + (qk_stage * 32 + norm_offset + token) * 4]
                                        ),
                                    )
                                    K.ptx.ld.shared.f32(
                                        norm_hi,
                                        arena.ptr_to(
                                            [
                                                _SMEM_NORM
                                                + (qk_stage * 32 + norm_offset + token + 1) * 4
                                            ]
                                        ),
                                    )
                                    unit_lo, unit_hi = _fmul2(raw_lo, raw_hi, norm_lo, norm_hi)
                                    correction_lo, correction_hi = _fmul2(
                                        unit_lo, unit_hi, dot_lo, dot_hi
                                    )
                                    residual_lo, residual_hi = _fsub2(
                                        grad[token], grad[token + 1], correction_lo, correction_hi
                                    )
                                    result_lo, result_hi = _fmul2(
                                        residual_lo, residual_hi, norm_lo, norm_hi
                                    )
                                    K.assign(grad[token], result_lo)
                                    K.assign(grad[token + 1], result_hi)
                            K.ptx.bar.sync(K.uint32(2), K.uint32(128))

                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    _arrive_barrier(arena, _BAR_QK_RAW_DONE, qk_stage)
                    _wait_barrier(arena, _BAR_DQ_TMASTG_DONE, 0, (serial + 1) & 1)
                    _wait_barrier(arena, _BAR_DK_TMASTG_DONE, 0, (serial + 1) & 1)
                    with K.unroll(16) as token:
                        K.ptx.st.shared.b16(
                            arena.ptr_to([_raw_bf16_byte(_SMEM_DQ, 0, token, channel)]),
                            K.reinterpret("uint16", K.cast(dq_values[token], "bfloat16")),
                        )
                        K.ptx.st.shared.b16(
                            arena.ptr_to([_raw_bf16_byte(_SMEM_DK, 0, token, channel)]),
                            K.reinterpret("uint16", K.cast(dk_values[token], "bfloat16")),
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DQ_TMASTG_READY)
                    _arrive_barrier(arena, _BAR_DK_TMASTG_READY)

                    first_state_chunk = 0 if use_initial_state else 1
                    with K.If(K.And(has_dstate, chunk >= first_state_chunk)), K.Then():
                        K.assign(dgate_values[15], dgate_values[15] + gate_last * dgate_last)
                    suffix = K.local_scalar("float32", init=K.float32(0.0))
                    for reverse_token in range(16):
                        token = 15 - reverse_token
                        K.assign(suffix, suffix + dgate_values[token])
                        K.assign(dgate_values[token], suffix)
                    _wait_barrier(arena, _BAR_DGATE_TMASTG_DONE, 0, (serial + 1) & 1)
                    _wait_barrier(arena, _BAR_DB_TMASTG_DONE, 0, (serial + 1) & 1)
                    with K.unroll(16) as token:
                        K.ptx.st.shared.f32(
                            arena.ptr_to([_raw_f32_byte(_SMEM_DGATE, 0, token, channel)]),
                            dgate_values[token],
                        )
                        K.ptx.st.shared.b16(
                            arena.ptr_to([_raw_bf16_byte(_SMEM_DB, 0, token, channel)]),
                            K.reinterpret("uint16", K.cast(db_values[token], "bfloat16")),
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DGATE_TMASTG_READY)
                    _arrive_barrier(arena, _BAR_DB_TMASTG_READY)

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_READY, sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(
                        tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_consumer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_DONE, sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
            _arrive_barrier(arena, _BAR_TMEM_DONE)
        with cg1:
            K.ptx.bar.sync(K.uint32(3), K.uint32(416))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_SMEM_TMEM_MAILBOX]))
            tmem_col = tmem_base & 0xFFFF
            tmem_alloc_row = tmem_base >> 16
            warp_in_group = K.warp_id_in_role()
            subpartition = warp % 4
            value_dim = subpartition * 32 + lane
            value_base = subpartition * 32
            token_row = (lane // 16) * 8 + (lane & 7)
            value_col_offset = ((lane // 8) & 1) * 8
            group_thread = warp_in_group * 32 + lane
            tile = K.local_scalar("int32", init=K.cta_id())
            serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            state_k_consumer = K.PipelineState(1, phase=0)
            du_consumer = K.PipelineState(1, phase=0)
            u_consumer = K.PipelineState(1, phase=0)
            dy_consumer = K.PipelineState(1, phase=0)
            dstate_consumer = K.PipelineState(1, phase=0)
            dstate_smem_done = K.PipelineState(1, phase=1)
            dbeta_m_consumer = K.PipelineState(1, phase=0)
            dv_done_consumer = K.PipelineState(2, phase=1)
            dwo_done_consumer = K.PipelineState(2, phase=1)
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cend = item[5]
                bos = item[6]
                eos = item[7]
                sequence_length = eos - bos
                sequence_chunks = (sequence_length + 15) // 16
                num_chunks = cend - wstart
                dstate_index_base = (
                    (K.cast(batch, "int64") * n_heads_out + head) * 128 + value_dim
                ) * 128

                if use_dstate_in:
                    with K.If(num_chunks > 0), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_SMEM_DONE,
                            dstate_smem_done.stage,
                            dstate_smem_done.phase,
                        )
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_SMEM_CG2_DONE,
                            dstate_smem_done.stage,
                            dstate_smem_done.phase,
                        )
                        dstate_smem_done.advance()
                        seed_enabled = cend == sequence_chunks
                        seed_row = tmem_alloc_row + subpartition * 32
                        with K.unroll(8) as key_subtile:
                            seed = K.alloc_local((16,), "float32")
                            seed_pack = K.alloc_local((8,), "uint32")
                            with K.unroll(16) as key_offset:
                                K.ptx.ld.global_.f32(
                                    seed[key_offset],
                                    d_final_state.ptr_to(
                                        [dstate_index_base + key_subtile * 16 + key_offset]
                                    ),
                                )
                                with K.If(K.Not(seed_enabled)), K.Then():
                                    K.assign(seed[key_offset], K.float32(0.0))
                            K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                                K.cast(
                                    tmem_col
                                    + _TMEM_DSTATE
                                    + key_subtile * 16
                                    + K.shift_left(seed_row, K.int32(16)),
                                    "uint32",
                                ),
                                *[seed[i] for i in range(16)],
                            )
                            for pair in range(8):
                                K.assign(
                                    seed_pack[pair],
                                    _pack_bf16_pair(seed[pair * 2], seed[pair * 2 + 1]),
                                )
                            K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                                K.cast(
                                    tmem_col
                                    + _TMEM_DSTATE_INPUT
                                    + key_subtile * 8
                                    + K.shift_left(seed_row, K.int32(16)),
                                    "uint32",
                                ),
                                *[seed_pack[i] for i in range(8)],
                            )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _arrive_barrier(arena, _BAR_DSTATE_INP_READY)
                        with K.unroll(8) as key_subtile:
                            seed_words = K.alloc_local((8,), "uint32")
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                                *[seed_words[i] for i in range(8)],
                                K.cast(
                                    tmem_col
                                    + _TMEM_DSTATE_INPUT
                                    + key_subtile * 8
                                    + K.shift_left(seed_row, K.int32(16)),
                                    "uint32",
                                ),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            with K.unroll(2) as half:
                                key_base = key_subtile * 16 + half * 8
                                K.ptx.st.shared.v4.b32(
                                    arena.ptr_to(
                                        [
                                            _SMEM_DSTATE
                                            + (
                                                (key_base // 64) * 8192
                                                + value_dim * 64
                                                + _swizzle_xor_128b(value_dim, key_base % 64, 2)
                                            )
                                            * 2
                                        ]
                                    ),
                                    seed_words[half * 4],
                                    seed_words[half * 4 + 1],
                                    seed_words[half * 4 + 2],
                                    seed_words[half * 4 + 3],
                                )
                        K.ptx.fence.proxy.async_.shared__cta()
                        _arrive_barrier(arena, _BAR_DSTATE_SMEM_READY)

                with K.serial(num_chunks, unroll=False) as reverse_index:
                    serial = serial_base + reverse_index
                    chunk = cend - 1 - reverse_index
                    raw_stage = serial % 2
                    has_dstate = reverse_index > 0
                    if use_dstate_in:
                        has_dstate = K.bool(True)
                    row_lo = tmem_alloc_row
                    row_hi = tmem_alloc_row + 16
                    projection_row0 = tmem_alloc_row + value_base
                    projection_row1 = projection_row0 + 16

                    _wait_barrier(arena, _BAR_V_READY, raw_stage, (serial // 2) & 1)
                    raw_v0 = K.alloc_local((4,), "uint32")
                    raw_v1 = K.alloc_local((4,), "uint32")
                    K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        raw_v0[0],
                        raw_v0[1],
                        raw_v0[2],
                        raw_v0[3],
                        arena.ptr_to(
                            [
                                _raw_bf16_byte(
                                    _SMEM_V, raw_stage, token_row, value_base + value_col_offset
                                )
                            ]
                        ),
                    )
                    K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        raw_v1[0],
                        raw_v1[1],
                        raw_v1[2],
                        raw_v1[3],
                        arena.ptr_to(
                            [
                                _raw_bf16_byte(
                                    _SMEM_V,
                                    raw_stage,
                                    token_row,
                                    value_base + 16 + value_col_offset,
                                )
                            ]
                        ),
                    )
                    K.ptx.fence.proxy.async_.shared__cta()

                    # The per-value write gate is read in the same transposed
                    # fragment form as V, so Y = W * V - state_k.
                    _wait_barrier(arena, _BAR_W_READY, raw_stage, (serial // 2) & 1)
                    raw_w0 = K.alloc_local((4,), "uint32")
                    raw_w1 = K.alloc_local((4,), "uint32")
                    K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        raw_w0[0],
                        raw_w0[1],
                        raw_w0[2],
                        raw_w0[3],
                        arena.ptr_to(
                            [
                                _raw_bf16_byte(
                                    _SMEM_W, raw_stage, token_row, value_base + value_col_offset
                                )
                            ]
                        ),
                    )
                    K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        raw_w1[0],
                        raw_w1[1],
                        raw_w1[2],
                        raw_w1[3],
                        arena.ptr_to(
                            [
                                _raw_bf16_byte(
                                    _SMEM_W,
                                    raw_stage,
                                    token_row,
                                    value_base + value_col_offset + 64,
                                )
                            ]
                        ),
                    )

                    difference0 = K.alloc_local((4,), "uint32")
                    difference1 = K.alloc_local((4,), "uint32")
                    y_pack0 = K.alloc_local((4,), "uint32")
                    y_pack1 = K.alloc_local((4,), "uint32")
                    for reg in range(4):
                        raw_matrix = (1 - reg // 2) * 2 + (reg & 1)
                        K.assign(difference0[reg ^ 2], raw_v0[raw_matrix])
                        K.assign(difference1[reg ^ 2], raw_v1[raw_matrix])
                        K.ptx.mul.bf16x2(y_pack0[reg ^ 2], raw_w0[raw_matrix], raw_v0[raw_matrix])
                        K.ptx.mul.bf16x2(y_pack1[reg ^ 2], raw_w1[raw_matrix], raw_v1[raw_matrix])
                    first_state_chunk = 0 if use_initial_state else 1
                    with K.If(chunk >= first_state_chunk), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_STATE_K_ACC_READY,
                            state_k_consumer.stage,
                            state_k_consumer.phase,
                        )
                        state_k_consumer.advance()
                        state0 = K.alloc_local((8,), "float32")
                        state1 = K.alloc_local((8,), "float32")
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[state0[i] for i in range(8)],
                            K.cast(
                                tmem_col
                                + _TMEM_STATE_K_DY
                                + K.shift_left(projection_row0, K.int32(16)),
                                "uint32",
                            ),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[state1[i] for i in range(8)],
                            K.cast(
                                tmem_col
                                + _TMEM_STATE_K_DY
                                + K.shift_left(projection_row1, K.int32(16)),
                                "uint32",
                            ),
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        for reg in range(4):
                            raw_matrix = (1 - reg // 2) * 2 + (reg & 1)
                            frag_pair = (reg ^ 2) * 2
                            state_pair0 = _pack_bf16_pair(state0[frag_pair], state0[frag_pair + 1])
                            state_pair1 = _pack_bf16_pair(state1[frag_pair], state1[frag_pair + 1])
                            # Y = W * V - state_k: the write gate scales V, and the
                            # entering-state term is subtracted unscaled.
                            K.ptx.mul.bf16x2(
                                difference0[reg ^ 2], raw_w0[raw_matrix], raw_v0[raw_matrix]
                            )
                            K.ptx.mul.bf16x2(
                                difference1[reg ^ 2], raw_w1[raw_matrix], raw_v1[raw_matrix]
                            )
                            K.ptx.sub.bf16x2(y_pack0[reg ^ 2], difference0[reg ^ 2], state_pair0)
                            K.ptx.sub.bf16x2(y_pack1[reg ^ 2], difference1[reg ^ 2], state_pair1)
                    K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                        K.cast(
                            tmem_col + _TMEM_Y_NEG_DY + K.shift_left(row_lo, K.int32(16)), "uint32"
                        ),
                        *[y_pack0[i] for i in range(4)],
                    )
                    K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                        K.cast(
                            tmem_col + _TMEM_Y_NEG_DY + K.shift_left(row_hi, K.int32(16)), "uint32"
                        ),
                        *[y_pack1[i] for i in range(4)],
                    )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    _arrive_barrier(arena, _BAR_Y_INP_READY)

                    _wait_barrier(arena, _BAR_DU_ACC_READY, du_consumer.stage, du_consumer.phase)
                    du_consumer.advance()
                    du0 = K.alloc_local((8,), "float32")
                    du1 = K.alloc_local((8,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                        *[du0[i] for i in range(8)],
                        K.cast(
                            tmem_col + _TMEM_DU + K.shift_left(projection_row0, K.int32(16)),
                            "uint32",
                        ),
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                        *[du1[i] for i in range(8)],
                        K.cast(
                            tmem_col + _TMEM_DU + K.shift_left(projection_row1, K.int32(16)),
                            "uint32",
                        ),
                    )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    du_pack0 = K.alloc_local((4,), "uint32")
                    du_pack1 = K.alloc_local((4,), "uint32")
                    for reg in range(4):
                        K.assign(du_pack0[reg], _pack_bf16_pair(du0[2 * reg], du0[2 * reg + 1]))
                        K.assign(du_pack1[reg], _pack_bf16_pair(du1[2 * reg], du1[2 * reg + 1]))
                    K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                        K.cast(
                            tmem_col + _TMEM_DU_INPUT + K.shift_left(row_lo, K.int32(16)), "uint32"
                        ),
                        *[du_pack0[i] for i in range(4)],
                    )
                    K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                        K.cast(
                            tmem_col + _TMEM_DU_INPUT + K.shift_left(row_hi, K.int32(16)), "uint32"
                        ),
                        *[du_pack1[i] for i in range(4)],
                    )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    _arrive_barrier(arena, _BAR_DU_INP_READY)

                    _wait_barrier(arena, _BAR_U_ACC_READY, u_consumer.stage, u_consumer.phase)
                    u_consumer.advance()
                    u0 = K.alloc_local((8,), "float32")
                    u1 = K.alloc_local((8,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                        *[u0[i] for i in range(8)],
                        K.cast(
                            tmem_col + _TMEM_U + K.shift_left(projection_row0, K.int32(16)),
                            "uint32",
                        ),
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                        *[u1[i] for i in range(8)],
                        K.cast(
                            tmem_col + _TMEM_U + K.shift_left(projection_row1, K.int32(16)),
                            "uint32",
                        ),
                    )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    u_pack0 = K.alloc_local((4,), "uint32")
                    u_pack1 = K.alloc_local((4,), "uint32")
                    for reg in range(4):
                        K.assign(u_pack0[reg], _pack_bf16_pair(u0[2 * reg], u0[2 * reg + 1]))
                        K.assign(u_pack1[reg], _pack_bf16_pair(u1[2 * reg], u1[2 * reg + 1]))
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        arena.ptr_to(
                            [_raw_bf16_byte(_SMEM_U, 0, token_row, value_base + value_col_offset)]
                        ),
                        *[u_pack0[i] for i in range(4)],
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        arena.ptr_to(
                            [
                                _raw_bf16_byte(
                                    _SMEM_U, 0, token_row, value_base + 16 + value_col_offset
                                )
                            ]
                        ),
                        *[u_pack1[i] for i in range(4)],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_U_SMEM_READY)

                    _wait_barrier(arena, _BAR_DY_ACC_READY, dy_consumer.stage, dy_consumer.phase)
                    dy_consumer.advance()
                    dy0 = K.alloc_local((8,), "float32")
                    dy1 = K.alloc_local((8,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                        *[dy0[i] for i in range(8)],
                        K.cast(
                            tmem_col
                            + _TMEM_STATE_K_DY
                            + K.shift_left(projection_row0, K.int32(16)),
                            "uint32",
                        ),
                    )
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                        *[dy1[i] for i in range(8)],
                        K.cast(
                            tmem_col
                            + _TMEM_STATE_K_DY
                            + K.shift_left(projection_row1, K.int32(16)),
                            "uint32",
                        ),
                    )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    dy_pack0 = K.alloc_local((4,), "uint32")
                    dy_pack1 = K.alloc_local((4,), "uint32")
                    for reg in range(4):
                        K.assign(dy_pack0[reg], _pack_bf16_pair(dy0[2 * reg], dy0[2 * reg + 1]))
                        K.assign(dy_pack1[reg], _pack_bf16_pair(dy1[2 * reg], dy1[2 * reg + 1]))
                    dy_addr0 = _raw_bf16_byte(_SMEM_DY, 0, token_row, value_base + value_col_offset)
                    dy_addr1 = _raw_bf16_byte(
                        _SMEM_DY, 0, token_row, value_base + 16 + value_col_offset
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        arena.ptr_to([dy_addr0]), *[dy_pack0[i] for i in range(4)]
                    )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        arena.ptr_to([dy_addr1]), *[dy_pack1[i] for i in range(4)]
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DY_SMEM_READY)

                    # GDN-2 stages plain -dY, not -(Beta * dY): the erase gate is
                    # already folded into the K_decay operand, and dV is formed
                    # separately from the write gate in the scalar pass below.
                    neg0 = K.alloc_local((4,), "uint32")
                    neg1 = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        K.assign(neg0[pair], _pack_bf16_pair(-dy0[pair * 2], -dy0[pair * 2 + 1]))
                        K.assign(neg1[pair], _pack_bf16_pair(-dy1[pair * 2], -dy1[pair * 2 + 1]))
                    K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                        K.cast(
                            tmem_col + _TMEM_Y_NEG_DY + K.shift_left(row_lo, K.int32(16)), "uint32"
                        ),
                        *[neg0[i] for i in range(4)],
                    )
                    K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                        K.cast(
                            tmem_col + _TMEM_Y_NEG_DY + K.shift_left(row_hi, K.int32(16)), "uint32"
                        ),
                        *[neg1[i] for i in range(4)],
                    )
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    _arrive_barrier(arena, _BAR_NEG_DY_INP_READY)

                    # dV and dW_out come from one scalar pass over this lane's
                    # value channel: GDN-2 needs two different products of dY, so
                    # a single store from a gate-scaled dY does not suffice.  V and W
                    # stay live until after this pass.
                    dv_stage = serial % 2
                    dwo_stage = serial % 2
                    _wait_barrier(
                        arena, _BAR_DV_TMASTG_DONE, dv_done_consumer.stage, dv_done_consumer.phase
                    )
                    dv_done_consumer.advance()
                    _wait_barrier(
                        arena,
                        _BAR_DWO_TMASTG_DONE,
                        dwo_done_consumer.stage,
                        dwo_done_consumer.phase,
                    )
                    dwo_done_consumer.advance()
                    K.ptx.bar.sync(K.uint32(4), K.uint32(128))

                    value_segment = group_thread // 64
                    value_column = group_thread - value_segment * 64
                    for token in range(16):
                        element = (
                            value_segment * 1024
                            + token * 64
                            + _swizzle_xor_128b(token, value_column, 2)
                        )
                        dy_bits = K.local_scalar("uint16")
                        w_bits = K.local_scalar("uint16")
                        v_bits = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(dy_bits, arena.ptr_to([_SMEM_DY + element * 2]))
                        K.ptx.ld.shared.b16(
                            w_bits, arena.ptr_to([_SMEM_W + raw_stage * 4096 + element * 2])
                        )
                        K.ptx.ld.shared.b16(
                            v_bits, arena.ptr_to([_SMEM_V + raw_stage * 4096 + element * 2])
                        )
                        dy_value = K.cast(K.reinterpret("bfloat16", dy_bits), "float32")
                        w_value = K.cast(K.reinterpret("bfloat16", w_bits), "float32")
                        v_value = K.cast(K.reinterpret("bfloat16", v_bits), "float32")
                        K.ptx.st.shared.b16(
                            arena.ptr_to([_SMEM_DV + dv_stage * 4096 + element * 2]),
                            K.reinterpret("uint16", K.cast(w_value * dy_value, "bfloat16")),
                        )
                        K.ptx.st.shared.b16(
                            arena.ptr_to([_SMEM_DWO + dwo_stage * 4096 + element * 2]),
                            K.reinterpret("uint16", K.cast(v_value * dy_value, "bfloat16")),
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, _BAR_DV_TMASTG_READY, dv_stage)
                    _arrive_barrier(arena, _BAR_DWO_TMASTG_READY, dwo_stage)
                    _arrive_barrier(arena, _BAR_V_DONE, raw_stage)
                    _arrive_barrier(arena, _BAR_W_DONE, raw_stage)

                    K.ptx.bar.sync(K.uint32(4), K.uint32(128))

                    _wait_barrier(
                        arena, _BAR_DSTATE_ACC_READY, dstate_consumer.stage, dstate_consumer.phase
                    )
                    dstate_consumer.advance()
                    with K.If(reverse_index + 1 < num_chunks), K.Then():
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_SMEM_DONE,
                            dstate_smem_done.stage,
                            dstate_smem_done.phase,
                        )
                        _wait_barrier(
                            arena,
                            _BAR_DSTATE_SMEM_CG2_DONE,
                            dstate_smem_done.stage,
                            dstate_smem_done.phase,
                        )
                        dstate_smem_done.advance()
                        dstate_row = tmem_alloc_row + subpartition * 32
                        with K.unroll(4) as key_subtile:
                            dstate_values = K.alloc_local((32,), "float32")
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[dstate_values[i] for i in range(32)],
                                K.cast(
                                    tmem_col
                                    + _TMEM_DSTATE
                                    + key_subtile * 32
                                    + K.shift_left(dstate_row, K.int32(16)),
                                    "uint32",
                                ),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            dstate_words = K.alloc_local((16,), "uint32")
                            for pair in range(16):
                                K.assign(
                                    dstate_words[pair],
                                    _pack_bf16_pair(
                                        dstate_values[pair * 2], dstate_values[pair * 2 + 1]
                                    ),
                                )
                            K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                                K.cast(
                                    tmem_col
                                    + _TMEM_DSTATE_INPUT
                                    + key_subtile * 16
                                    + K.shift_left(dstate_row, K.int32(16)),
                                    "uint32",
                                ),
                                *[dstate_words[i] for i in range(16)],
                            )
                            with K.unroll(4) as half:
                                key_base = key_subtile * 32 + half * 8
                                K.ptx.st.shared.v4.b32(
                                    arena.ptr_to(
                                        [
                                            _SMEM_DSTATE
                                            + (
                                                (key_base // 64) * 8192
                                                + value_dim * 64
                                                + _swizzle_xor_128b(value_dim, key_base % 64, 2)
                                            )
                                            * 2
                                        ]
                                    ),
                                    dstate_words[half * 4],
                                    dstate_words[half * 4 + 1],
                                    dstate_words[half * 4 + 2],
                                    dstate_words[half * 4 + 3],
                                )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _arrive_barrier(arena, _BAR_DSTATE_INP_READY)
                        K.ptx.fence.proxy.async_.shared__cta()
                        _arrive_barrier(arena, _BAR_DSTATE_SMEM_READY)

                if use_dstate0:
                    with K.If(K.And(num_chunks > 0, wstart == 0)), K.Then():
                        dstate_row = tmem_alloc_row + subpartition * 32
                        with K.unroll(4) as key_subtile:
                            dstate_values = K.alloc_local((32,), "float32")
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[dstate_values[i] for i in range(32)],
                                K.cast(
                                    tmem_col
                                    + _TMEM_DSTATE
                                    + key_subtile * 32
                                    + K.shift_left(dstate_row, K.int32(16)),
                                    "uint32",
                                ),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            for key_offset in range(32):
                                K.ptx.st.global_.f32(
                                    d_initial_state.ptr_to(
                                        [dstate_index_base + key_subtile * 32 + key_offset]
                                    ),
                                    dstate_values[key_offset],
                                )
                    with K.If(num_chunks == 0), K.Then():
                        with K.unroll(128) as key:
                            passthrough = K.local_scalar("float32", init=K.float32(0.0))
                            if use_dstate_in:
                                K.ptx.ld.global_.f32(
                                    passthrough, d_final_state.ptr_to([dstate_index_base + key])
                                )
                            K.ptx.st.global_.f32(
                                d_initial_state.ptr_to([dstate_index_base + key]), passthrough
                            )
                _arrive_barrier(arena, _BAR_DSTATE0_ACC_STORED)
                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, _BAR_SCHED_READY, sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(
                        tile,
                        arena.ptr_to([_SMEM_SCHED + K.cast(sched_consumer.stage, "int32") * 4]),
                    )
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_DONE, sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
            _arrive_barrier(arena, _BAR_TMEM_DONE)

    return main


def _normalized_config(config):
    config = {key: value for key, value in config.items() if key != "label"}
    config.setdefault("seq_lens", (64,))
    config.setdefault("heads", 1)
    config.setdefault("q_heads", config["heads"])
    config.setdefault("k_heads", config["heads"])
    config.setdefault("v_heads", config["heads"])
    config.setdefault("num_sms", 148)
    config.setdefault("scale", 1.0 / (_DK**0.5))
    for key in (
        "l2norm",
        "safe_gate",
        "beta_sigmoid",
        "use_initial_state",
        "use_dstate_in",
        "use_dstate0",
        "dynamic_scheduler",
        "run_order",
        "order_generate",
    ):
        config.setdefault(key, False)
    heads = int(config["heads"])
    for name in ("q_heads", "k_heads", "v_heads"):
        if heads % int(config[name]) != 0:
            raise ValueError(f"{name}={config[name]} must divide heads={heads}")
    return config


def get_kernel(**config):
    config = _normalized_config(config)
    num_sms = int(config.get("num_sms", 148))
    prologue = _make_prologue(
        run_order=bool(config.get("run_order", False)),
        order_generate=bool(config.get("order_generate", False)),
        dynamic_scheduler=bool(config.get("dynamic_scheduler", False)),
        n_heads_out=int(config.get("n_heads_out", config.get("heads", 1))),
    )
    main = _make_main(
        num_sms=num_sms,
        beta_sigmoid=bool(config.get("beta_sigmoid", False)),
        use_dstate_in=bool(config.get("use_dstate_in", False)),
        use_dstate0=bool(config.get("use_dstate0", False)),
        use_initial_state=bool(config.get("use_initial_state", False)),
        safe_gate=bool(config.get("safe_gate", False)),
        full_tiles=all(int(length) % _BT == 0 for length in config["seq_lens"]),
        dynamic_scheduler=bool(config.get("dynamic_scheduler", False)),
        l2norm=bool(config.get("l2norm", False)),
        n_heads_out=int(config.get("n_heads_out", config.get("heads", 1))),
        q_ratio=int(config.get("q_ratio", config["heads"] // config["q_heads"])),
        k_ratio=int(config.get("k_ratio", config["heads"] // config["k_heads"])),
        v_ratio=int(config.get("v_ratio", config["heads"] // config["v_heads"])),
    )
    return [prologue.func, main.func]


def _aligned_i64(torch, size, alignment=128):
    if size % 8 != 0:
        raise ValueError(f"workspace size must be divisible by 8, got {size}")
    owner = torch.empty(size + alignment - 1, dtype=torch.uint8, device="cuda")
    offset = (-owner.data_ptr()) % alignment
    view = owner[offset : offset + size].view(torch.int64)
    if view.data_ptr() % alignment != 0:
        raise AssertionError("failed to construct an aligned TensorMap workspace")
    return owner, view


def _make_work_items(torch, seq_lens, heads):
    rows = []
    token_begin = 0
    for batch, sequence_length in enumerate(seq_lens):
        chunks = (sequence_length + _BT - 1) // _BT
        token_end = token_begin + sequence_length
        for head in range(heads):
            rows.append((batch, head, 0, chunks, 0, chunks, token_begin, token_end))
        token_begin = token_end
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def _effective_inputs(torch, q, k, gate, beta, *, l2norm, safe_gate, beta_sigmoid, a_log, dt_bias):
    q_effective = q
    k_effective = k
    if l2norm:
        q_effective = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_effective = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
    gate_effective = gate
    if safe_gate:
        gate_effective = -5.0 * torch.sigmoid(
            a_log.exp()[None, :, None] * (gate + dt_bias[None, :, :])
        )
    beta_effective = beta.sigmoid() if beta_sigmoid else beta
    return q_effective, k_effective, gate_effective, beta_effective


def _forward_and_checkpoints(
    torch, q, k, v, gate, beta, w, cu_seqlens, *, scale, initial_state, checkpoint_dtype
):
    """FP64 chunk-entering state series for the GDN-2 recurrence.

    Per-key decay first, the erase term reads the ALREADY-DECAYED state and is
    gated per key channel, the write gate scales v, and the rank-1 update uses
    the raw key.
    """
    from itertools import pairwise

    bounds = [int(value) for value in cu_seqlens.cpu().tolist()]
    batch_count = len(bounds) - 1
    heads = gate.shape[1]
    checkpoint_rows = max((q.shape[0] // _BT) + batch_count, 1)
    checkpoints = torch.zeros(
        checkpoint_rows, heads, _DV, _DK, dtype=checkpoint_dtype, device=q.device
    )
    outputs = []
    final_states = []
    for batch, (begin, end) in enumerate(pairwise(bounds)):
        if initial_state is None:
            state = torch.zeros(heads, _DK, _DV, dtype=torch.float64, device=q.device)
        else:
            state = initial_state[batch].transpose(-1, -2)
        checkpoint_base = begin // _BT + batch
        for local_token, token in enumerate(range(begin, end)):
            if local_token % _BT == 0:
                checkpoints[checkpoint_base + local_token // _BT] = state.transpose(-1, -2).to(
                    checkpoint_dtype
                )
            state = gate[token].exp()[..., None] * state
            erase = torch.einsum("hd,hdv->hv", beta[token] * k[token], state)
            v_new = w[token] * v[token] - erase
            state = state + torch.einsum("hd,hv->hdv", k[token], v_new)
            outputs.append(torch.einsum("hd,hdv->hv", q[token] * scale, state))
        final_states.append(state.transpose(-1, -2))
    output = torch.stack(outputs)
    return output, torch.stack(final_states), checkpoints


def _io_dtype(torch, config):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.get("dtype", "bfloat16")]


def _new_outputs(torch, config, total_tokens):
    heads = config["heads"]
    io_dtype = _io_dtype(torch, config)
    result = {
        "dq": torch.zeros(total_tokens, heads, _DK, dtype=io_dtype, device="cuda"),
        "dk": torch.zeros(total_tokens, heads, _DK, dtype=io_dtype, device="cuda"),
        "dv": torch.zeros(total_tokens, heads, _DV, dtype=io_dtype, device="cuda"),
        "dgate": torch.zeros(total_tokens, heads, _DK, dtype=torch.float32, device="cuda"),
        "dbeta": torch.zeros(total_tokens, heads, _DK, dtype=io_dtype, device="cuda"),
        "dw": torch.zeros(total_tokens, heads, _DV, dtype=io_dtype, device="cuda"),
    }
    if config["use_dstate0"]:
        result["d_initial_state"] = torch.zeros(
            len(config["seq_lens"]), heads, _DV, _DK, dtype=torch.float32, device="cuda"
        )
    return result


def _prepare_work_tables(torch, config):
    base = _make_work_items(torch, config["seq_lens"], config["heads"])

    def one_side():
        if config["run_order"]:
            work_items = torch.empty_like(base)
            staging = None if config["order_generate"] else base.clone()
            sched_all = torch.zeros(4, dtype=torch.int32, device="cuda")
        else:
            work_items = base.clone()
            staging = None
            sched_all = None
        return {
            "work_items": work_items,
            "work_count": torch.tensor([base.shape[0]], dtype=torch.int32, device="cuda"),
            "staging": staging,
            "sched_all": sched_all,
            "sched_ctr": (
                torch.zeros(2, dtype=torch.int32, device="cuda")
                if config["dynamic_scheduler"]
                else None
            ),
        }

    return {"tirx": one_side(), "source": one_side()}


def _prepare_data(config, *, with_oracle):
    import torch

    config = _normalized_config(config)
    torch.manual_seed(20260823)
    total_tokens = sum(config["seq_lens"])
    heads = config["heads"]
    q_heads = config["q_heads"]
    k_heads = config["k_heads"]
    v_heads = config["v_heads"]
    q = (0.2 * torch.randn(total_tokens, q_heads, _DK, device="cuda")).to(io_dtype)
    k = (0.2 * torch.randn(total_tokens, k_heads, _DK, device="cuda")).to(io_dtype)
    v = (0.2 * torch.randn(total_tokens, v_heads, _DV, device="cuda")).to(io_dtype)
    io_dtype = _io_dtype(torch, config)
    gate = torch.empty(total_tokens, heads, _DK, dtype=torch.float32, device="cuda").uniform_(
        -1.5, -0.5
    )
    if config["safe_gate"]:
        gate = 0.2 * torch.randn_like(gate)
    # GDN-2 carries a per-key-channel erase gate and a per-value write gate,
    # both in the I/O dtype: beta is [T, HO, DK] and w is [T, HO, DV].
    beta = (0.3 * torch.randn(total_tokens, heads, _DK, device="cuda")).to(io_dtype)
    if not config["beta_sigmoid"]:
        beta = beta.float().sigmoid().to(io_dtype)
    w = torch.rand(total_tokens, heads, _DV, device="cuda").to(io_dtype)
    do = (0.2 * torch.randn(total_tokens, heads, _DV, device="cuda")).to(io_dtype)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(config["seq_lens"]).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    a_log = (
        0.1 * torch.randn(heads, dtype=torch.float32, device="cuda")
        if config["safe_gate"]
        else None
    )
    dt_bias = (
        -4.0 + 0.1 * torch.randn(heads, _DK, dtype=torch.float32, device="cuda")
        if config["safe_gate"]
        else None
    )
    initial_state = (
        0.01
        * torch.randn(len(config["seq_lens"]), heads, _DV, _DK, dtype=torch.float64, device="cuda")
        if config["use_initial_state"]
        else None
    )
    if initial_state is not None and with_oracle:
        initial_state.requires_grad_(True)
    d_final_state = (
        0.02
        * torch.randn(len(config["seq_lens"]), heads, _DV, _DK, dtype=torch.float32, device="cuda")
        if config["use_dstate_in"]
        else None
    )

    oracle = None
    if with_oracle:
        q_ref = q.double().repeat_interleave(heads // q_heads, dim=1).detach().requires_grad_(True)
        k_ref = k.double().repeat_interleave(heads // k_heads, dim=1).detach().requires_grad_(True)
        v_ref = v.double().repeat_interleave(heads // v_heads, dim=1).detach().requires_grad_(True)
        gate_ref = gate.double().detach().requires_grad_(True)
        beta_ref = beta.double().detach().requires_grad_(True)
        w_ref = w.double().detach().requires_grad_(True)
        q_effective, k_effective, gate_effective, beta_effective = _effective_inputs(
            torch,
            q_ref,
            k_ref,
            gate_ref,
            beta_ref,
            l2norm=config["l2norm"],
            safe_gate=config["safe_gate"],
            beta_sigmoid=config["beta_sigmoid"],
            a_log=a_log.double() if a_log is not None else None,
            dt_bias=dt_bias.double() if dt_bias is not None else None,
        )
        output_ref, final_ref, checkpoints = _forward_and_checkpoints(
            torch,
            q_effective,
            k_effective,
            v_ref,
            gate_effective,
            beta_effective,
            w_ref,
            cu_seqlens,
            scale=config["scale"],
            initial_state=initial_state,
            checkpoint_dtype=io_dtype,
        )
        loss = (output_ref * do.double()).sum()
        if d_final_state is not None:
            loss = loss + (final_ref * d_final_state.double()).sum()
        grad_inputs = [q_ref, k_ref, v_ref, gate_ref, beta_ref, w_ref]
        if config["safe_gate"]:
            grad_inputs.append(gate_effective)
        if initial_state is not None:
            grad_inputs.append(initial_state)
        grads = torch.autograd.grad(loss, grad_inputs)
        oracle = {
            "dq": grads[0],
            "dk": grads[1],
            "dv": grads[2],
            "dgate": grads[6] if config["safe_gate"] else grads[3],
            "dbeta": grads[4],
            "dw": grads[5],
        }
        if initial_state is not None:
            oracle["d_initial_state"] = grads[-1]
        checkpoints = checkpoints.detach()
    else:
        checkpoint_rows = max(total_tokens // _BT + len(config["seq_lens"]), 1)
        checkpoints = (
            0.02 * torch.randn(checkpoint_rows, heads, _DV, _DK, dtype=torch.float32, device="cuda")
        ).to(torch.bfloat16)

    workspace_bytes = _TENSOR_MAP_BYTES * _TENSOR_MAP_ARRAYS * len(config["seq_lens"]) + 128
    tirx_owner, tirx_workspace = _aligned_i64(torch, workspace_bytes)
    source_owner, source_workspace = _aligned_i64(torch, workspace_bytes)
    return {
        "config": config,
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "w": w,
        "do": do,
        "cu_seqlens": cu_seqlens,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "checkpoints": checkpoints,
        "d_final_state": d_final_state,
        "tirx": _new_outputs(torch, config, total_tokens),
        "source": _new_outputs(torch, config, total_tokens),
        "work": _prepare_work_tables(torch, config),
        "oracle": oracle,
        "tirx_workspace": tirx_workspace,
        "source_workspace": source_workspace,
        "_workspace_owners": (tirx_owner, source_owner),
    }


def prepare_data(**config):
    """Allocate source/TIRx outputs and an independent FP64 recurrence oracle."""
    return _prepare_data(config, with_oracle=True)


def _encode_tiled_map(tensor, dimensions, strides, box):
    import torch
    from cuda.bindings import driver as cuda

    if tensor.dtype.is_floating_point and tensor.element_size() == 2:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    elif tensor.dtype.is_floating_point and tensor.element_size() == 4:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
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


def _base_tensor_maps(data):
    config = data["config"]
    total_tokens = data["q"].shape[0]

    def headed(tensor, channels, heads, box_channels):
        return _encode_tiled_map(
            tensor,
            (channels, heads, total_tokens),
            (tensor.stride(1) * tensor.element_size(), tensor.stride(0) * tensor.element_size()),
            (box_channels, 1, _BT),
        )

    output = data["tirx"]
    # Source index order: q0 k1 v2 gate3 do4 beta5 w6 dq7 dk8 dv9 dgate10
    # dwo11 dbeta12 checkpoint13.  Only the FP32 gate pair uses a 32-channel box.
    maps = [
        headed(data["q"], _DK, config["q_heads"], 64),
        headed(data["k"], _DK, config["k_heads"], 64),
        headed(data["v"], _DV, config["v_heads"], 64),
        headed(data["gate"], _DK, config["heads"], 32),
        headed(data["do"], _DV, config["heads"], 64),
        headed(data["beta"], _DK, config["heads"], 64),
        headed(data["w"], _DV, config["heads"], 64),
        headed(output["dq"], _DK, config["heads"], 64),
        headed(output["dk"], _DK, config["heads"], 64),
        headed(output["dv"], _DV, config["heads"], 64),
        headed(output["dgate"], _DK, config["heads"], 32),
        headed(output["dw"], _DV, config["heads"], 64),
        headed(output["dbeta"], _DK, config["heads"], 64),
    ]
    checkpoints = data["checkpoints"]
    maps.append(
        _encode_tiled_map(
            checkpoints,
            (_DV, _DK, checkpoints.shape[0], config["heads"]),
            (
                checkpoints.stride(2) * checkpoints.element_size(),
                checkpoints.stride(0) * checkpoints.element_size(),
                checkpoints.stride(1) * checkpoints.element_size(),
            ),
            (64, _DK, 1, 1),
        )
    )
    return maps


def _tirx_launch(executables, data):
    import torch

    config = data["config"]
    prologue, main = executables
    maps = _base_tensor_maps(data)
    work = data["work"]["tirx"]
    dummy_i32 = torch.zeros(8, dtype=torch.int32, device="cuda")
    dummy_a_log = torch.zeros(config["heads"], dtype=torch.float32, device="cuda")
    dummy_dt_bias = torch.zeros(config["heads"], _DK, dtype=torch.float32, device="cuda")
    dummy_dstate_in = torch.zeros(1, dtype=torch.float32, device="cuda")
    dummy_dstate_out = torch.zeros(1, dtype=torch.float32, device="cuda")
    output = data["tirx"]

    def launch(*, prologue_only=False):
        prologue(
            *maps,
            data["tirx_workspace"],
            data["cu_seqlens"],
            data["q"].view(torch.uint8).reshape(-1),
            data["k"].view(torch.uint8).reshape(-1),
            data["v"].view(torch.uint8).reshape(-1),
            data["gate"].view(torch.uint8).reshape(-1),
            data["do"].view(torch.uint8).reshape(-1),
            output["dq"].view(torch.uint8).reshape(-1),
            output["dk"].view(torch.uint8).reshape(-1),
            output["dv"].view(torch.uint8).reshape(-1),
            output["dgate"].view(torch.uint8).reshape(-1),
            data["checkpoints"].view(torch.uint8).reshape(-1),
            work["staging"].reshape(-1) if work["staging"] is not None else dummy_i32,
            work["work_count"],
            work["work_items"].reshape(-1),
            work["sched_all"] if work["sched_all"] is not None else dummy_i32,
            len(config["seq_lens"]),
            data["q"].stride(0) * data["q"].element_size(),
            data["k"].stride(0) * data["k"].element_size(),
            data["v"].stride(0) * data["v"].element_size(),
            data["gate"].stride(0) * data["gate"].element_size(),
            data["do"].stride(0) * data["do"].element_size(),
            output["dq"].stride(0) * output["dq"].element_size(),
            output["dk"].stride(0) * output["dk"].element_size(),
            output["dv"].stride(0) * output["dv"].element_size(),
            output["dgate"].stride(0) * output["dgate"].element_size(),
            data["checkpoints"].stride(0) * data["checkpoints"].element_size(),
            _BT,
        )
        if prologue_only:
            return
        main(
            data["tirx_workspace"],
            len(config["seq_lens"]),
            data["a_log"] if data["a_log"] is not None else dummy_a_log,
            (
                data["dt_bias"].reshape(-1)
                if data["dt_bias"] is not None
                else dummy_dt_bias.reshape(-1)
            ),
            data["cu_seqlens"],
            output["dgate"].reshape(-1),
            output.get("d_initial_state", dummy_dstate_out).reshape(-1),
            (
                data["d_final_state"].reshape(-1)
                if data["d_final_state"] is not None
                else dummy_dstate_in
            ),
            work["work_items"].reshape(-1),
            work["work_count"],
            work["sched_ctr"] if work["sched_ctr"] is not None else dummy_i32,
            config["scale"],
        )

    launch._keep_alive = (
        maps,
        dummy_i32,
        dummy_a_log,
        dummy_dt_bias,
        dummy_dstate_in,
        dummy_dstate_out,
    )
    return launch


_REFERENCE_SOURCE = None


def _load_reference_source():
    global _REFERENCE_SOURCE
    if _REFERENCE_SOURCE is not None:
        return _REFERENCE_SOURCE
    import importlib
    import os
    import sys

    root = os.environ.get("CUDNN_FRONTEND_PATH")
    if root is None:
        raise RuntimeError("CUDNN_FRONTEND_PATH must point to a cuDNN Frontend source checkout")
    python_root = os.path.join(root, "python")
    if python_root not in sys.path:
        sys.path.append(python_root)
    import cudnn

    cudnn_root = os.path.join(python_root, "cudnn")
    if cudnn_root not in cudnn.__path__:
        cudnn.__path__.append(cudnn_root)
    _REFERENCE_SOURCE = importlib.import_module(
        "cudnn.linear_attention.frost.kernel.gdn2_bprop_f16"
    )
    return _REFERENCE_SOURCE


def _source_launch(data):
    import torch

    source = _load_reference_source()
    config = data["config"]
    output = data["source"]
    work = data["work"]["source"]
    stream = int(torch.cuda.current_stream().cuda_stream)

    def launch():
        source.chunk_gdn2_bwd_sm100(
            data["q"],
            data["k"],
            data["v"],
            data["gate"],
            data["beta"],
            data["w"],
            data["do"],
            data["checkpoints"],
            output["dq"],
            output["dk"],
            output["dv"],
            output["dgate"],
            output["dbeta"],
            output["dw"],
            data["cu_seqlens"],
            config["scale"],
            use_initial_state=config["use_initial_state"],
            d_initial_state=output.get("d_initial_state"),
            d_final_state=data["d_final_state"],
            use_qk_l2norm_in_kernel=config["l2norm"],
            safe_gate=config["safe_gate"],
            gate_lower_bound=-5.0,
            a_log=data["a_log"],
            dt_bias=data["dt_bias"],
            use_beta_sigmoid=config["beta_sigmoid"],
            work_items=work["work_items"],
            work_count=work["work_count"],
            sched_ctr=work["sched_ctr"],
            sched_all=work["sched_all"],
            work_item_scratch=work["staging"],
            order_in_prologue=config["run_order"],
            tensormap_workspace=data["source_workspace"],
            stream=stream,
        )

    return launch


def _rms_ratio(torch, actual, expected):
    actual = actual.detach().double()
    expected = expected.detach().double()
    denominator = expected.square().mean().sqrt().clamp_min(1e-12)
    return float(((actual - expected).square().mean().sqrt() / denominator).item())


def _validate_outputs(data, *, sources, with_oracle):
    import math

    import torch

    config = data["config"]
    limits = {
        "dq": 0.10,
        "dk": 0.10,
        "dv": 0.10,
        "dgate": 0.30,
        "dbeta": 0.10,
        "d_initial_state": 0.10,
    }
    failures = {}
    if with_oracle:
        if data["oracle"] is None:
            raise AssertionError("oracle validation requested without oracle data")
        for source_name in sources:
            for output_name, expected in data["oracle"].items():
                ratio = _rms_ratio(torch, data[source_name][output_name], expected)
                if not math.isfinite(ratio) or ratio >= limits[output_name]:
                    failures[f"{source_name}.{output_name}"] = ratio
    if "tirx" in sources and "source" in sources:
        for output_name in data["tirx"]:
            ratio = _rms_ratio(torch, data["tirx"][output_name], data["source"][output_name])
            if not math.isfinite(ratio) or ratio >= limits[output_name]:
                failures[f"tirx_vs_source.{output_name}"] = ratio
    if failures:
        raise AssertionError(f"GDN-2 backward validation failed for {config}: {failures}")


def run_test(**config):
    """Compare both kernels with the standalone FP64 recurrence oracle."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _normalized_config(config)
    data = _prepare_data(kernel_config, with_oracle=True)
    executables = [compile_kernel(func) for func in get_kernel(**kernel_config)]
    tirx_launch = _tirx_launch(executables, data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    _validate_outputs(data, sources=("tirx", "source"), with_oracle=True)
    return {"tokens": sum(kernel_config["seq_lens"]), "heads": kernel_config["heads"]}


def prepare_bench(**config):
    """Compile the two TIRx launches without importing torch or touching CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _normalized_config(config)
    state = {
        "config": kernel_config,
        "executables": [compile_kernel(func) for func in get_kernel(**kernel_config)],
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then let bench_suite time the exact two-launch paths."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = _normalized_config({**prepared["config"], **kwargs})
    data = _prepare_data(config, with_oracle=False)
    tirx_launch = _tirx_launch(prepared["executables"], data)
    tirx_launch()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source_launch = _source_launch(data)
        source_launch()
        torch.cuda.synchronize()
        _validate_outputs(data, sources=("tirx", "source"), with_oracle=False)
        references = {"cudnn_frontend": lambda: source_launch}
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
