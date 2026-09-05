# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell FP16/BF16 Gated DeltaNet (v1) state-recompute kernels.

Upstream source:
``python/cudnn/linear_attention/frost/kernel/gdn_recompute_f16.py``
(``prologue_kernel``, ``kernel``, and the two-launch ``run_recompute`` entry,
driven by the ``chunk_gdn_recompute_sm100`` THD/varlen host entry).

The kernel re-runs the chunked delta-rule forward recurrence to regenerate the
per-chunk recurrent-state checkpoint series a backward pass consumes; the query
and output paths of the prefill kernel are absent.
"""

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_gdn_recompute_f16",
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

# Deterministic pairwise cover of the frozen legal capability set.  Ragged
# cases include zero- and one-token sequences while retaining a nonzero
# TensorMap outer dimension.  ``log_gate`` defaults to True (the backward
# plan's production path); raw-alpha (log_gate=False) and safe-gate rows cover
# the other two gate interpretations.  Checkpoint cadences are the validated
# positive multiples of the 64-token chunk; a zero cadence requires the final
# state, which is the source's "checkpoints or final state" precondition.
CONFIGS = [
    {
        "label": "pairwise_00",
        "seq_lens": (63, 0, 129, 1),
        "heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "cu_dtype": "int64",
        "use_initial_state": True,
        "checkpoint_every_n_tokens": 64,
        "safe_gate": True,
        "beta_sigmoid": True,
        "dynamic_scheduler": True,
        "order_generate": False,
        "split": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_01",
        "seq_lens": (17, 65),
        "heads": 1,
        "io_dtype": "float16",
        "state_dtype": "bfloat16",
        "store_final_state": True,
        "checkpoint_every_n_tokens": 128,
        "safe_gate": True,
        "beta_sigmoid": True,
        "dynamic_scheduler": True,
        "order_generate": False,
        "split": True,
    },
    {
        "label": "pairwise_02",
        "seq_lens": (17, 65),
        "heads": 1,
        "io_dtype": "float16",
        "cu_dtype": "int64",
        "use_initial_state": True,
        "checkpoint_every_n_tokens": 192,
        "log_gate": False,
    },
    {
        "label": "pairwise_03",
        "seq_lens": (63, 0, 129, 1),
        "heads": 4,
        "k_heads": 2,
        "v_heads": 4,
        "io_dtype": "float16",
        "state_dtype": "bfloat16",
        "store_final_state": True,
        "checkpoint_every_n_tokens": 64,
        "log_gate": False,
        "order_generate": False,
        "num_sms": 2,
    },
    {
        "label": "pairwise_04",
        "seq_lens": (192,),
        "heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "io_dtype": "float16",
        "cu_dtype": "int64",
        "use_initial_state": True,
        "beta_sigmoid": True,
        "dynamic_scheduler": True,
        "order_generate": False,
        "split": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_05",
        "seq_lens": (63, 0, 129, 1),
        "heads": 4,
        "k_heads": 2,
        "v_heads": 4,
        "state_dtype": "bfloat16",
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 192,
        "safe_gate": True,
        "beta_sigmoid": True,
        "dynamic_scheduler": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_06",
        "seq_lens": (192,),
        "heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "state_dtype": "bfloat16",
        "cu_dtype": "int64",
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 128,
        "safe_gate": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_07",
        "seq_lens": (17, 65),
        "heads": 4,
        "k_heads": 2,
        "v_heads": 4,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 192,
        "order_generate": False,
        "split": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_08",
        "seq_lens": (63, 0, 129, 1),
        "heads": 1,
        "state_dtype": "bfloat16",
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 0,
        "safe_gate": True,
        "dynamic_scheduler": True,
    },
    {
        "label": "pairwise_09",
        "seq_lens": (63, 0, 129, 1),
        "heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "cu_dtype": "int64",
        "checkpoint_every_n_tokens": 128,
        "beta_sigmoid": True,
        "log_gate": False,
        "num_sms": 2,
    },
    {
        "label": "pairwise_10",
        "seq_lens": (17, 65),
        "heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "state_dtype": "bfloat16",
        "store_final_state": True,
        "checkpoint_every_n_tokens": 64,
        "num_sms": 2,
    },
    {
        "label": "pairwise_11",
        "seq_lens": (192,),
        "heads": 1,
        "checkpoint_every_n_tokens": 64,
        "log_gate": False,
        "run_order": False,
        "order_generate": False,
    },
    {
        "label": "pairwise_12",
        "seq_lens": (17, 65),
        "heads": 4,
        "k_heads": 2,
        "v_heads": 4,
        "state_dtype": "bfloat16",
        "store_final_state": True,
        "checkpoint_every_n_tokens": 0,
        "num_sms": 2,
    },
    {
        "label": "pairwise_13",
        "seq_lens": (64, 320),
        "heads": 6,
        "k_heads": 6,
        "v_heads": 2,
        "io_dtype": "float16",
        "use_initial_state": True,
        "checkpoint_every_n_tokens": 64,
        "dynamic_scheduler": True,
    },
]

# Production sweep mirroring the upstream gdn benchmark's backward pass: h64,
# natural-log gates, the per-chunk cadence the backward consumes, and the
# device-side dynamic scheduler the engine always enables for this kernel.
# ``nostate`` is the regen path exactly as the backward plan runs it
# (checkpoints only); ``state`` seeds the recurrence and also stores the final
# state, which is a separate compiled specialization (wider register split).
BENCH_CONFIGS = [
    {
        "label": f"perf_b{batch}_s{seqlen}_h64_{mode}",
        "seq_lens": (seqlen,) * batch,
        "heads": 64,
        "checkpoint_every_n_tokens": 64,
        "dynamic_scheduler": True,
        **({"use_initial_state": True, "store_final_state": True} if mode == "state" else {}),
    }
    for mode in ("nostate", "state")
    for batch, seqlen in (
        (4, 2048),
        (4, 4096),
        (4, 8192),
        (4, 16384),
        (4, 32768),
        (1, 8192),
        (2, 8192),
        (8, 8192),
        (16, 8192),
    )
]


_BT = 64
_DK = 128
_DV = 128
_TENSOR_MAP_BYTES = 128
_TENSOR_MAP_WORDS = 16
_TENSOR_MAP_ARRAYS = 3  # per-batch runtime TMA descriptors: K, V, checkpoints
_TRY_WAIT_TICKS = 10_000_000
_SCHED_ALL_CELLS = 4  # the engine's shared sched block: both consumers' ticket rings
_RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
_LN2 = 0.6931471805599453
_TMA_G2S_3D = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"

# Declaration-ordered physical mbarrier header (fixed capacity for the
# four-stage K ring; checkpoint builds initialize only three).  Each barrier
# is 8 bytes; ready and done rings of one pipeline are adjacent.
_BAR_KQ_READY = 0  # 4 stages
_BAR_KQ_DONE = 32  # 4 stages
_BAR_V_READY = 64  # 2 stages
_BAR_V_DONE = 80  # 2 stages
_BAR_GATE_READY = 96  # 3 stages
_BAR_GATE_DONE = 120  # 3 stages
_BAR_BETA_READY = 144  # 3 stages
_BAR_BETA_DONE = 168  # 3 stages
_BAR_STATE_ACC_READY = 192
_BAR_STATE_SCALE_DONE = 200
_BAR_CG0_ACC_READY = 208  # 2 stages
_BAR_CG0_ACC_DONE = 224  # 2 stages
_BAR_K_STATE_READY = 240
_BAR_U_ACC_READY = 248
_BAR_T_INV_READY = 256  # 3 stages
_BAR_T_INV_DONE = 280  # 3 stages
_BAR_STATE_INP_READY = 304
_BAR_Y_INP_READY = 312
_BAR_DECAY_U_READY = 320
_BAR_CKPT_READY = 328
_BAR_CKPT_DONE = 336
_BAR_TMEM_DONE = 344
_BAR_SCHED_READY = 352  # 2 stages
_BAR_SCHED_DONE = 368  # 2 stages
_TMEM_MAILBOX = 384
_SCHED_BASE = 400  # 2 x i32 ticket ring
_CUMSUMLOG_BASE = 512  # f32 [64 x 3 stages]
_CUMPROD_BASE = 1280
_BETA_RING_BASE = 2048
_CKPT_BASE = 3072  # checkpoint builds only: IO [128 x 128] x 1 stage
_ARENA_BYTES = 191488

# TMEM columns (512 allocated, 448 used); packed IO regions hold two elements
# per cell.
_TM_STATE = 0
_TM_STATE_INP = 128
_TM_CG0 = 192  # 2 stages x 64 columns
_TM_CG1 = 320  # K*state accumulator, then the U accumulator in the same cells
_TM_Y = 384
_TM_DECAY_U = 416

# tcgen05 instruction descriptors (M=128; N=64 for the KK/K-state/U GEMMs,
# N=128 for the state update), extracted from the source export; FP16 clears
# the BF16 dtype fields (delta 0x480).
_IDESC_N64_BF16 = 135267472
_IDESC_N128_BF16 = 136381584
_KQ_BOX = 512  # 8192 B in 16-byte descriptor units: pair-member box offset
_KQ_SEG = 1024  # 16384 B in descriptor units: second-K-half subtile offset


def _elected():
    lane = K.local_scalar("uint32")
    pred = K.local_scalar("uint32")
    K.ptx.elect_sync(lane, pred, K.uint32(0xFFFFFFFF))
    return pred == K.uint32(1)


def _load_tensormap(payload, src_map):
    """Load one host-encoded 128-byte TensorMap image into registers.

    The base images use ordinary global pointers because K's PTX instruction
    surface intentionally has no parameter-space load.  One load serves every
    per-sequence slot of the array; it is issued at the point of use, inside
    the owning warp's guard.
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
        K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready,
            _barrier_ptr(arena, byte_offset, stage),
            K.cast(phase, "uint32"),
            K.uint32(_TRY_WAIT_TICKS),
        )


def _arrive_barrier(arena, byte_offset, stage=0):
    K.ptx.mbarrier.arrive.shared.b64(_barrier_ptr(arena, byte_offset, stage))


def _expect_tx(arena, byte_offset, stage, nbytes):
    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
        _barrier_ptr(arena, byte_offset, stage), K.uint32(nbytes)
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


def _load_state_row(pointer, index, count, state_dtype):
    """Load ``count`` state values starting at ``index`` in 16-byte accesses.

    A state row is 128 contiguous elements, so its start is at least 256-byte
    aligned and every 16-byte group is in bounds.  The source moves this row
    with 128-bit accesses; one scalar load per element would add a per-work-item
    fixed cost to the seed path.
    """
    values = K.alloc_local((count,), "float32")
    if state_dtype == "float32":
        for group in range(count // 4):
            K.ptx["ld.global.v4.f32"](
                values[group * 4],
                values[group * 4 + 1],
                values[group * 4 + 2],
                values[group * 4 + 3],
                pointer.ptr_to([index + group * 4]),
            )
    else:
        for group in range(count // 8):
            words = K.alloc_local((4,), "uint32")
            K.ptx["ld.global.v4.b32"](
                words[0], words[1], words[2], words[3], pointer.ptr_to([index + group * 8])
            )
            for pair in range(4):
                _unpack_io_pair(
                    words[pair],
                    values[group * 8 + pair * 2],
                    values[group * 8 + pair * 2 + 1],
                    state_dtype,
                )
    return values


def _store_state_row(pointer, index, values, count, state_dtype):
    """Store ``count`` state values starting at ``index`` in 16-byte accesses."""
    if state_dtype == "float32":
        for group in range(count // 4):
            K.ptx["st.global.v4.f32"](
                pointer.ptr_to([index + group * 4]),
                values[group * 4],
                values[group * 4 + 1],
                values[group * 4 + 2],
                values[group * 4 + 3],
            )
    else:
        for group in range(count // 8):
            words = K.alloc_local((4,), "uint32")
            for pair in range(4):
                K.assign(
                    words[pair],
                    _pack_io_pair(
                        values[group * 8 + pair * 2], values[group * 8 + pair * 2 + 1], state_dtype
                    ),
                )
            K.ptx["st.global.v4.b32"](
                pointer.ptr_to([index + group * 8]), words[0], words[1], words[2], words[3]
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
    low = K.cast(K.bitwise_and(packed, K.uint32(0xFFFF)), "uint16")
    high = K.cast(K.shift_right(packed, K.uint32(16)), "uint16")
    if io_dtype == "float16":
        K.assign(lo, K.cast(K.reinterpret("float16", low), "float32"))
        K.assign(hi, K.cast(K.reinterpret("float16", high), "float32"))
    else:
        K.ptx.cvt.f32.bf16(lo, low)
        K.ptx.cvt.f32.bf16(hi, high)


def _sub_io_pair(dst, lhs, rhs, io_dtype):
    if io_dtype == "float16":
        K.ptx.sub.f16x2(dst, lhs, rhs)
    else:
        K.ptx["sub.bf16x2"](dst, lhs, rhs)


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


def _invert_diagonal_8x8(arena, tinv_byte_base, d, tidx_in_group, io_dtype):
    """Gauss-Jordan inversion of one diagonal 8x8 block in place (IO SMEM)."""
    row_ptr = _tinv_row_ptr(arena, tinv_byte_base, d, tidx_in_group)
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], row_ptr)
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


def _blockwise_8_to_16(arena, tinv_byte_base, d0, lane, io_dtype):
    """Off-diagonal correction 8x8 -> 16x16: C <- -(D^-1 C) A^-1."""
    lane_off = (lane % 8) * 64
    d_frag = _ldmatrix_x1(
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 8) * 64 + d0 + 8 + lane_off) * 2]),
        trans=False,
    )
    c_frag = _ldmatrix_x1(
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 8) * 64 + d0 + lane_off) * 2]),
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


def _blockwise_16_to_32(arena, tinv_byte_base, d0, lane, io_dtype):
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
        arena.ptr_to([tinv_byte_base + _swizzle_lin_128b((d0 + 16) * 64 + d0 + lane_off) * 2]),
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


def _blockwise_32_to_64(arena, tinv_byte_base, band, lane, io_dtype, store_result):
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
                    tinv_byte_base
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
    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
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


def _tcgen_mma_ss(dst, a_desc, b_desc, idesc, *, leader, accumulate):
    """Fused KK/QK half: four K-phases of the SMEM-A x SMEM-B tcgen05 chain."""
    for step in range(4):
        K.ptx[_MMA_F16](
            K.cast(dst, "uint32"),
            a_desc + K.uint64(2 * step),
            b_desc + K.uint64(2 * step),
            K.uint32(idesc),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.ptx.pred(_acc_operand(accumulate) if step == 0 else K.uint32(1)),
            pred=leader,
        )


def _tcgen_mma_ts(dst, tmem_a, b_desc, idesc, *, leader, accumulate, b_step_units=2):
    """Four K-phases of the TMEM-A x SMEM-B tcgen05 chain (16-wide steps)."""
    for step in range(4):
        K.ptx[_MMA_F16](
            K.cast(dst, "uint32"),
            K.cast(tmem_a + step * 8, "uint32"),
            b_desc + K.uint64(b_step_units * step),
            K.uint32(idesc),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.uint32(0),
            K.ptx.pred(_acc_operand(accumulate) if step == 0 else K.uint32(1)),
            pred=leader,
        )


def _make_prologue(*, run_order, order_generate, n_heads_out, checkpoints, cu_dtype):
    cu_t = K.i64 if cu_dtype == "int64" else K.i32

    @K.kernel(warps=32, arch="sm_100a", grid=1)
    def prologue(
        base_k: K.gptr[K.i64],
        base_v: K.gptr[K.i64],
        base_checkpoint: K.gptr[K.i64],
        descriptor_workspace: K.gptr[K.i64],
        cu_seqlens: K.gptr[cu_t],
        k: K.gptr[K.u8],
        v: K.gptr[K.u8],
        checkpoint: K.gptr[K.u8],
        work_item_staging: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        work_items: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        n_batch: K.i32,
        k_row_stride_bytes: K.i32,
        v_row_stride_bytes: K.i32,
        checkpoint_row_stride_bytes: K.i32,
        checkpoint_every_n: K.i32,
    ):
        # --- kernel body starts here ---
        thread = K.thread_id()
        warp = K.warp_id()
        if run_order:
            order_arena = K.alloc_buffer((32_776,), K.u8, scope="shared.dyn", align=16)
            # The order pass owns both consumers' ticket rings: thread 0 walks
            # every cell serially, as the source's order_body does.
            with K.If(thread == 0), K.Then():
                for cell in range(_SCHED_ALL_CELLS):
                    K.ptx.st.global_.s32(scheduler.ptr_to([cell]), K.int32(0))

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

        # One descriptor array per warp: K, V, then the checkpoint series.
        # The elected lane reads the 128-byte base image once for the array and
        # replays it into every per-sequence slot, patching the global address
        # and the sequence extent.
        bases = (k, v)
        strides = (k_row_stride_bytes, v_row_stride_bytes)
        base_maps = (base_k, base_v)
        for array_index in range(2):
            with K.If(warp == array_index), K.Then():
                with K.If(_elected()), K.Then():
                    map_payload = K.alloc_local((16,), "uint64")
                    _load_tensormap(map_payload, base_maps[array_index])
                    with K.serial(n_batch) as batch:
                        begin = _load_cu(cu_seqlens, batch, cu_dtype)
                        end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                        slot = _descriptor_slot(descriptor_workspace, n_batch, array_index, batch)
                        _store_tensormap(slot, map_payload)
                        _replace_tensormap_address(
                            slot,
                            bases[array_index].ptr_to(
                                [K.cast(begin, "int64") * strides[array_index]]
                            ),
                        )
                        _replace_tensormap_dim(slot, 2, end - begin)
                    K.ptx.fence.proxy.tensormap__generic.release.gpu()

        if checkpoints:
            # The checkpoint array's outer extent is the per-sequence entry
            # count, running-prefix-summed, matching emit_checkpoint_seq_descs.
            with K.If(warp == 2), K.Then():
                with K.If(_elected()), K.Then():
                    map_payload = K.alloc_local((16,), "uint64")
                    _load_tensormap(map_payload, base_checkpoint)
                    checkpoint_prefix = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(n_batch) as batch:
                        begin = _load_cu(cu_seqlens, batch, cu_dtype)
                        end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                        length = end - begin
                        count = K.local_scalar("int32", init=K.int32(0))
                        with K.If(length > 0), K.Then():
                            K.assign(count, (length - 1) // checkpoint_every_n + 1)
                        slot = _descriptor_slot(descriptor_workspace, n_batch, 2, batch)
                        _store_tensormap(slot, map_payload)
                        _replace_tensormap_address(
                            slot,
                            checkpoint.ptr_to(
                                [K.cast(checkpoint_prefix, "int64") * checkpoint_row_stride_bytes]
                            ),
                        )
                        _replace_tensormap_dim(slot, 2, count)
                        K.assign(checkpoint_prefix, checkpoint_prefix + count)
                    K.ptx.fence.proxy.tensormap__generic.release.gpu()

    return prologue


def _make_main(
    *,
    num_sms,
    io_dtype,
    state_dtype,
    cu_dtype,
    use_initial_state,
    store_final_state,
    checkpoints,
    log_gate,
    safe_gate,
    beta_sigmoid,
    dynamic_scheduler,
    k_ratio,
    v_ratio,
    n_heads_out,
):
    io_t = K.f16 if io_dtype == "float16" else K.bf16
    state_t = K.bf16 if state_dtype == "bfloat16" else K.f32
    cu_t = K.i64 if cu_dtype == "int64" else K.i32
    beta_t = io_t if beta_sigmoid else K.f32
    # The checkpoint build trades one K stage for the staging buffer, so both
    # builds allocate the same arena.
    kq_stages = 3 if checkpoints else 4
    kq_base = _CKPT_BASE + (32768 if checkpoints else 0)
    tinv_base = kq_base + kq_stages * 32768
    v_base = tinv_base + 3 * 8192
    idesc_delta = 1152 if io_dtype == "float16" else 0
    idesc_n64 = _IDESC_N64_BF16 - idesc_delta
    idesc_n128 = _IDESC_N128_BF16 - idesc_delta
    cg1_regs = 256 if use_initial_state else 232
    other_regs = 24 if use_initial_state else 48

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=num_sms)
    def main(
        descriptor_workspace: K.gptr[K.i64],
        n_desc: K.i32,
        k: K.gptr[io_t],
        v: K.gptr[io_t],
        gate: K.gptr[K.f32],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        beta: K.gptr[beta_t],
        cu_seqlens: K.gptr[cu_t],
        initial_state: K.gptr[state_t],
        final_state: K.gptr[state_t],
        work_items: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        checkpoint_every_n: K.i32,
    ):
        # --- kernel body starts here ---
        arena = K.alloc_buffer((_ARENA_BYTES,), K.u8, scope="shared.dyn", align=1024)
        K.smem_pool(base=arena)
        thread = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()

        # Declaration-ordered physical mbarrier protocol; each row is
        # (byte offset, stages, arrive count, initializing warp).
        protocol = [
            (_BAR_KQ_READY, kq_stages, 1, 9),
            (_BAR_KQ_DONE, kq_stages, 1, 9),
            (_BAR_V_READY, 2, 1, 9),
            (_BAR_V_DONE, 2, 128, 9),
            (_BAR_SCHED_READY, 2, 1, 9),
            (_BAR_SCHED_DONE, 2, 11, 9),
            (_BAR_GATE_READY, 3, 32, 8),
            (_BAR_GATE_DONE, 3, 256, 8),
            (_BAR_BETA_READY, 3, 32, 8),
            (_BAR_BETA_DONE, 3, 128, 8),
            (_BAR_STATE_ACC_READY, 1, 1, 10),
            (_BAR_CG0_ACC_READY, 2, 1, 10),
            (_BAR_CG0_ACC_DONE, 2, 64, 10),
            (_BAR_K_STATE_READY, 1, 1, 10),
            (_BAR_U_ACC_READY, 1, 1, 10),
            (_BAR_T_INV_DONE, 3, 1, 10),
            (_BAR_T_INV_READY, 3, 128, 0),
            (_BAR_STATE_SCALE_DONE, 1, 128, 4),
            (_BAR_STATE_INP_READY, 1, 128, 4),
            (_BAR_Y_INP_READY, 1, 128, 4),
            (_BAR_DECAY_U_READY, 1, 128, 4),
            (_BAR_TMEM_DONE, 1, 128, 4),
            (_BAR_CKPT_READY, 1, 4, 11),
            (_BAR_CKPT_DONE, 1, 32, 11),
        ]
        for owner in (9, 8, 10, 0, 4, 11):
            with K.If(warp == owner), K.Then():
                with K.If(_elected()), K.Then():
                    for offset, stages, count, row_owner in protocol:
                        if row_owner != owner:
                            continue
                        for stage in range(stages):
                            K.ptx.mbarrier.init.shared.b64(
                                _barrier_ptr(arena, offset, stage), K.uint32(count)
                            )
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        total_tiles = K.local_scalar("int32")
        K.ptx.ld.global_.s32(total_tiles, work_count.ptr_to([0]))

        roles = K.specialize()
        cg0 = roles.role("cg0", warps=range(0, 4), regs=224)
        cg1 = roles.role("cg1", warps=range(4, 8), regs=cg1_regs)
        gate_beta = roles.role("gate_beta", warps=[8], regs=other_regs)
        mma = roles.role("mma", warps=[10], regs=other_regs)
        tma = roles.role("tma", warps=[9], regs=other_regs)
        epilogue = roles.role("epilogue", warps=[11], regs=other_regs)

        def sched_consume(sched_ps, tile):
            if dynamic_scheduler:
                _wait_barrier(arena, _BAR_SCHED_READY, sched_ps.stage, sched_ps.phase)
                K.ptx.ld.shared.s32(tile, arena.ptr_to([_SCHED_BASE + sched_ps.stage * 4]))
                with K.If(_elected()), K.Then():
                    _arrive_barrier(arena, _BAR_SCHED_DONE, sched_ps.stage)
                sched_ps.advance()
            else:
                K.assign(tile, tile + num_sms)

        with cg0:
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_TMEM_MAILBOX]))
            tmem_col = tmem_base & 0xFFFF
            tmem_row = tmem_base >> 16
            cg0_tidx = thread
            inv_warp = warp % 2
            pair_half = warp // 2
            half_row_base = inv_warp * 32
            tidx_in_group = cg0_tidx % 8
            gj_d = (inv_warp * 32 + lane) // 8
            store_row_frag = lane % 16
            store_col = (lane // 16) * 8
            beta_scale_row = warp * 16 + lane % 16
            row_u0_lo = half_row_base + lane // 4
            mask_zero = _opaque_zero()
            gate_ps = K.PipelineState(3, phase=0)
            beta_ps = K.PipelineState(3, phase=0)
            cg0_ps = K.PipelineState(2, phase=0)
            tinv_ps = K.PipelineState(3, phase=1)
            sched_ps = K.PipelineState(2, phase=0)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                n_local = item[3] - item[4]
                n_pairs = (n_local + 1) // 2
                with K.serial(n_pairs, unroll=False) as pair_i:
                    have_m1 = pair_i * 2 + 1 < n_local
                    do_kk = K.Or(have_m1, pair_half == 0)

                    # ---- gate acquires (ready waits per acquired stage) ----
                    gate0_idx = K.local_scalar("int32", init=gate_ps.stage)
                    gate0_phase = K.local_scalar("int32", init=gate_ps.phase)
                    gate_ps.advance()
                    _wait_barrier(arena, _BAR_GATE_READY, gate0_idx, gate0_phase)
                    gate1_idx = K.local_scalar("int32", init=gate0_idx)
                    with K.If(have_m1), K.Then():
                        K.assign(gate1_idx, gate_ps.stage)
                        _wait_barrier(arena, _BAR_GATE_READY, gate_ps.stage, gate_ps.phase)
                        gate_ps.advance()
                    kk_gate_idx = K.if_then_else(pair_half == 1, gate1_idx, gate0_idx)

                    # ---- T-pairwise decay fragments for both members -------
                    kk_rows = K.alloc_local((4,), "float32")
                    row_offsets = [row_u0_lo, row_u0_lo + 8, row_u0_lo + 16, row_u0_lo + 24]
                    for index in range(4):
                        K.ptx.ld.shared.f32(
                            kk_rows[index],
                            arena.ptr_to(
                                [_CUMSUMLOG_BASE + kk_gate_idx * 256 + row_offsets[index] * 4]
                            ),
                        )
                    kk_cols = K.alloc_local((16,), "float32")
                    for group in range(8):
                        col_base = (lane % 4) * 2 + group * 8
                        K.ptx.ld.shared.v2.f32(
                            kk_cols[group * 2],
                            kk_cols[group * 2 + 1],
                            arena.ptr_to([_CUMSUMLOG_BASE + kk_gate_idx * 256 + col_base * 4]),
                        )
                    decay_kk = K.alloc_local((64,), "float32")
                    for member in range(2):
                        for cell in range(32):
                            hi_row = ((cell // 2) % 2) == 1
                            row_index = member * 2 + (1 if hi_row else 0)
                            col_index = (cell // 4) * 2 + (cell % 2)
                            crow = row_offsets[row_index]
                            ccol = (lane % 4) * 2 + ((cell // 4) * 8 + cell % 2)
                            kk_val = _exp2(kk_rows[row_index] - kk_cols[col_index])
                            K.assign(
                                decay_kk[member * 32 + cell],
                                K.if_then_else(crow >= ccol, kk_val, mask_zero),
                            )
                    _arrive_barrier(arena, _BAR_GATE_DONE, gate0_idx)
                    with K.If(have_m1), K.Then():
                        _arrive_barrier(arena, _BAR_GATE_DONE, gate1_idx)

                    # ---- beta acquires (ready waits per acquired stage) ----
                    beta0_idx = K.local_scalar("int32", init=beta_ps.stage)
                    beta0_phase = K.local_scalar("int32", init=beta_ps.phase)
                    beta_ps.advance()
                    _wait_barrier(arena, _BAR_BETA_READY, beta0_idx, beta0_phase)
                    beta1_idx = K.local_scalar("int32", init=beta0_idx)
                    with K.If(have_m1), K.Then():
                        K.assign(beta1_idx, beta_ps.stage)
                        _wait_barrier(arena, _BAR_BETA_READY, beta_ps.stage, beta_ps.phase)
                        beta_ps.advance()
                    kk_beta_idx = K.if_then_else(pair_half == 1, beta1_idx, beta0_idx)
                    kk_beta = K.alloc_local((4,), "float32")
                    for index in range(4):
                        K.ptx.ld.shared.f32(
                            kk_beta[index],
                            arena.ptr_to(
                                [_BETA_RING_BASE + kk_beta_idx * 256 + row_offsets[index] * 4]
                            ),
                        )

                    # ---- accumulator / T-inverse slot pairing --------------
                    acc0_idx = K.local_scalar("int32", init=cg0_ps.stage)
                    acc0_phase = K.local_scalar("int32", init=cg0_ps.phase)
                    cg0_ps.advance()
                    acc1_idx = K.local_scalar("int32", init=acc0_idx)
                    acc1_phase = K.local_scalar("int32", init=acc0_phase)
                    with K.If(have_m1), K.Then():
                        K.assign(acc1_idx, cg0_ps.stage)
                        K.assign(acc1_phase, cg0_ps.phase)
                        cg0_ps.advance()
                    kk_acc_idx = K.if_then_else(pair_half == 1, acc1_idx, acc0_idx)
                    kk_acc_phase = K.if_then_else(pair_half == 1, acc1_phase, acc0_phase)
                    tinv0_idx = K.local_scalar("int32", init=tinv_ps.stage)
                    tinv0_phase = K.local_scalar("int32", init=tinv_ps.phase)
                    tinv_ps.advance()
                    tinv1_idx = K.local_scalar("int32", init=tinv0_idx)
                    tinv1_phase = K.local_scalar("int32", init=tinv0_phase)
                    with K.If(have_m1), K.Then():
                        K.assign(tinv1_idx, tinv_ps.stage)
                        K.assign(tinv1_phase, tinv_ps.phase)
                        tinv_ps.advance()
                    kk_tinv_idx = K.if_then_else(pair_half == 1, tinv1_idx, tinv0_idx)
                    kk_tinv_phase = K.if_then_else(pair_half == 1, tinv1_phase, tinv0_phase)

                    # ---- KK epilogue into the T-inverse buffer -------------
                    with K.If(do_kk), K.Then():
                        _wait_barrier(arena, _BAR_CG0_ACC_READY, kk_acc_idx, kk_acc_phase)
                        kk_vec0 = K.alloc_local((32,), "float32")
                        kk_vec1 = K.alloc_local((32,), "float32")
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
                            *[kk_vec0[i] for i in range(32)],
                            K.cast(
                                tmem_col
                                + _TM_CG0
                                + kk_acc_idx * 64
                                + K.shift_left(tmem_row + warp * 32, K.int32(16)),
                                "uint32",
                            ),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
                            *[kk_vec1[i] for i in range(32)],
                            K.cast(
                                tmem_col
                                + _TM_CG0
                                + kk_acc_idx * 64
                                + K.shift_left(tmem_row + warp * 32 + 16, K.int32(16)),
                                "uint32",
                            ),
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        _arrive_barrier(arena, _BAR_CG0_ACC_DONE, kk_acc_idx)
                        _wait_barrier(arena, _BAR_T_INV_DONE, kk_tinv_idx, kk_tinv_phase)
                        for member in range(2):
                            source_vec = kk_vec1 if member == 1 else kk_vec0
                            packs = K.alloc_local((16,), "uint32")
                            for cell in range(16):
                                beta_row = kk_beta[member * 2 + (cell % 2)]
                                p0, p1 = _fmul2(
                                    source_vec[2 * cell],
                                    source_vec[2 * cell + 1],
                                    decay_kk[member * 32 + 2 * cell],
                                    decay_kk[member * 32 + 2 * cell + 1],
                                )
                                v0, v1 = _fmul2(p0, p1, beta_row, beta_row)
                                K.assign(packs[cell], _pack_io_pair(v0, v1, io_dtype))
                            st_row = half_row_base + member * 16 + store_row_frag
                            for frag in range(4):
                                _stmatrix_x4(
                                    arena.ptr_to(
                                        [
                                            tinv_base
                                            + kk_tinv_idx * 8192
                                            + (
                                                st_row * 64
                                                + _swizzle_xor_128b(st_row, store_col + frag * 16)
                                            )
                                            * 2
                                        ]
                                    ),
                                    [packs[frag * 4 + j] for j in range(4)],
                                )

                    # ---- four-level hierarchical pair inverse --------------
                    inv_tinv_idx = K.local_scalar("int32", init=tinv0_idx)
                    with K.If(have_m1), K.Then():
                        with K.If(warp >= 2), K.Then():
                            K.assign(inv_tinv_idx, tinv1_idx)
                    do_inv = K.Or(have_m1, warp < 2)
                    inv_byte_base = tinv_base + inv_tinv_idx * 8192
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    with K.If(do_inv), K.Then():
                        _invert_diagonal_8x8(arena, inv_byte_base, gj_d, tidx_in_group, io_dtype)
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    _blockwise_8_to_16(
                        arena, tinv_base + tinv0_idx * 8192, warp * 16, lane, io_dtype
                    )
                    with K.If(have_m1), K.Then():
                        _blockwise_8_to_16(
                            arena, tinv_base + tinv1_idx * 8192, warp * 16, lane, io_dtype
                        )
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    with K.If(do_inv), K.Then():
                        _blockwise_16_to_32(arena, inv_byte_base, inv_warp * 32, lane, io_dtype)
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                    _blockwise_32_to_64(arena, inv_byte_base, inv_warp, lane, io_dtype, do_inv)
                    K.ptx.bar.sync(K.uint32(2), K.uint32(128))

                    # ---- post-inverse beta column scaling + publish --------
                    for stage_member in range(2):
                        publish = K.bool(True) if stage_member == 0 else have_m1
                        stage_idx = tinv0_idx if stage_member == 0 else tinv1_idx
                        stage_beta = beta0_idx if stage_member == 0 else beta1_idx
                        with K.If(publish), K.Then():
                            beta_col = K.alloc_local((16,), "float32")
                            for group in range(8):
                                col_base = (lane % 4) * 2 + group * 8
                                K.ptx.ld.shared.v2.f32(
                                    beta_col[group * 2],
                                    beta_col[group * 2 + 1],
                                    arena.ptr_to(
                                        [_BETA_RING_BASE + stage_beta * 256 + col_base * 4]
                                    ),
                                )
                            tinv_words = K.alloc_local((16,), "uint32")
                            for frag in range(4):
                                chunk_words = K.alloc_local((4,), "uint32")
                                _ldmatrix_x4(
                                    chunk_words,
                                    arena.ptr_to(
                                        [
                                            tinv_base
                                            + stage_idx * 8192
                                            + (
                                                beta_scale_row * 64
                                                + _swizzle_xor_128b(
                                                    beta_scale_row, store_col + frag * 16
                                                )
                                            )
                                            * 2
                                        ]
                                    ),
                                    trans=False,
                                )
                                for j in range(4):
                                    K.assign(tinv_words[frag * 4 + j], chunk_words[j])
                            scaled = K.alloc_local((16,), "uint32")
                            for j in range(16):
                                lo = K.local_scalar("float32")
                                hi = K.local_scalar("float32")
                                _unpack_io_pair(tinv_words[j], lo, hi, io_dtype)
                                s0, s1 = _fmul2(
                                    lo, hi, beta_col[(j // 2) * 2], beta_col[(j // 2) * 2 + 1]
                                )
                                K.assign(scaled[j], _pack_io_pair(s0, s1, io_dtype))
                            for frag in range(4):
                                _stmatrix_x4(
                                    arena.ptr_to(
                                        [
                                            tinv_base
                                            + stage_idx * 8192
                                            + (
                                                beta_scale_row * 64
                                                + _swizzle_xor_128b(
                                                    beta_scale_row, store_col + frag * 16
                                                )
                                            )
                                            * 2
                                        ]
                                    ),
                                    [scaled[frag * 4 + j] for j in range(4)],
                                )
                            K.ptx.fence.proxy.async_.shared__cta()
                            _arrive_barrier(arena, _BAR_T_INV_READY, stage_idx)
                            _arrive_barrier(arena, _BAR_BETA_DONE, stage_beta)

                sched_consume(sched_ps, tile)
            for _ in range(3):
                _wait_barrier(arena, _BAR_T_INV_DONE, tinv_ps.stage, tinv_ps.phase)
                tinv_ps.advance()

        # ================================================================
        # Warps 4..7 (CG1): state seed/restage/checkpoint/rescale, Y, U, final
        # ================================================================
        with cg1:
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_TMEM_MAILBOX]))
            tmem_col = tmem_base & 0xFFFF
            tmem_row = tmem_base >> 16
            cg1_tidx = thread - 128
            value_dim = cg1_tidx
            row_id = tmem_row + (cg1_tidx // 32) * 32
            v_tok = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
            v_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
            v_sub_off = (cg1_tidx // 64) * 8192
            v_ps = K.PipelineState(2, phase=0)
            gate_ps = K.PipelineState(3, phase=0)
            state_ps = K.PipelineState(1, phase=0)
            seed_ps = K.PipelineState(1, phase=1)
            kst_ps = K.PipelineState(1, phase=0)
            uacc_ps = K.PipelineState(1, phase=0)
            sched_ps = K.PipelineState(2, phase=0)
            ckpt_cnt = K.local_scalar("int32", init=K.int32(0))
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cstart = item[4]
                bos = item[6]
                eos = item[7]
                num_chunks_b = (eos - bos + _BT - 1) // _BT
                n_local = wend - cstart
                state_index = (
                    (K.cast(batch, "int64") * n_heads_out + head) * 128 + value_dim
                ) * 128
                ckpt_chunks = K.local_scalar("int32", init=K.int32(1))
                ckpt_mod = K.local_scalar("int32", init=K.int32(0))
                if checkpoints:
                    K.assign(ckpt_chunks, checkpoint_every_n // _BT)
                    K.assign(ckpt_mod, cstart % ckpt_chunks)
                with K.If(n_local > 0), K.Then():
                    if use_initial_state:
                        # ---- initial-state seed: GMEM (or zero) -> TMEM ----
                        _wait_barrier(arena, _BAR_STATE_SCALE_DONE, 0, seed_ps.phase)
                        seed_ps.advance()
                        with K.If(cstart == 0), K.Then():
                            for sub in range(4):
                                seeded = _load_state_row(
                                    initial_state, state_index + sub * 32, 32, state_dtype
                                )
                                K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                                    K.cast(
                                        tmem_col
                                        + _TM_STATE
                                        + sub * 32
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                    *[seeded[i] for i in range(32)],
                                )
                        with K.If(cstart != 0), K.Then():
                            for sub in range(4):
                                K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                                    K.cast(
                                        tmem_col
                                        + _TM_STATE
                                        + sub * 32
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                    *[K.float32(0.0) for _ in range(32)],
                                )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        K.ptx.bar.sync(K.uint32(4), K.uint32(128))

                    with K.serial(n_local, unroll=False) as local_idx:
                        chunk = cstart + local_idx
                        do_ckpt_now = K.local_scalar("int32", init=K.int32(0))
                        if checkpoints:
                            K.assign(do_ckpt_now, K.cast(ckpt_mod == 0, "int32"))
                            K.assign(ckpt_mod, ckpt_mod + 1)
                            with K.If(ckpt_mod == ckpt_chunks), K.Then():
                                K.assign(ckpt_mod, K.int32(0))
                        if checkpoints and not use_initial_state:
                            # The entering state of an unseeded sequence is zero
                            # and no TMEM read can supply it.
                            with K.If(K.And(chunk == 0, wstart == 0)), K.Then():
                                _wait_barrier(arena, _BAR_CKPT_DONE, 0, 1 - (ckpt_cnt & 1))
                                for zero_step in range(64):
                                    K.ptx.st.shared.u32(
                                        arena.ptr_to(
                                            [_CKPT_BASE + (cg1_tidx + zero_step * 128) * 4]
                                        ),
                                        K.uint32(0),
                                    )
                                K.ptx.fence.proxy.async_.shared__cta()
                                with K.If(_elected()), K.Then():
                                    _arrive_barrier(arena, _BAR_CKPT_READY, 0)
                                K.assign(ckpt_cnt, ckpt_cnt + 1)
                        if use_initial_state:
                            have_state = K.bool(True)
                            seed_ps.advance()
                        else:
                            have_state = local_idx > 0

                        gate_idx = K.local_scalar("int32", init=gate_ps.stage)
                        gate_phase = K.local_scalar("int32", init=gate_ps.phase)
                        gate_ps.advance()
                        _wait_barrier(arena, _BAR_GATE_READY, gate_idx, gate_phase)
                        cumprod_total = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            cumprod_total, arena.ptr_to([_CUMPROD_BASE + gate_idx * 256 + 63 * 4])
                        )

                        # ---- state restage + checkpoint + rescale ----------
                        state_vals = K.alloc_local((128,), "float32")
                        with K.If(have_state), K.Then():
                            _wait_barrier(arena, _BAR_STATE_ACC_READY, 0, state_ps.phase)
                            state_ps.advance()
                            for sub in range(4):
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                    *[state_vals[sub * 32 + i] for i in range(32)],
                                    K.cast(
                                        tmem_col
                                        + _TM_STATE
                                        + sub * 32
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                            for sub in range(4):
                                packed = K.alloc_local((16,), "uint32")
                                for pair in range(16):
                                    K.assign(
                                        packed[pair],
                                        _pack_io_pair(
                                            state_vals[sub * 32 + 2 * pair],
                                            state_vals[sub * 32 + 2 * pair + 1],
                                            io_dtype,
                                        ),
                                    )
                                K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                                    K.cast(
                                        tmem_col
                                        + _TM_STATE_INP
                                        + sub * 16
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                    *[packed[i] for i in range(16)],
                                )
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            _arrive_barrier(arena, _BAR_STATE_INP_READY, 0)
                            if checkpoints:
                                # The snapshot is taken before the rescale, so
                                # row j of the series is the state entering
                                # token j*N.  The source re-reads the cells the
                                # restage already holds; transcribed as-is.
                                with (
                                    K.If(
                                        K.And(
                                            do_ckpt_now == 1, K.And(chunk >= wstart, chunk < wend)
                                        )
                                    ),
                                    K.Then(),
                                ):
                                    _wait_barrier(arena, _BAR_CKPT_DONE, 0, 1 - (ckpt_cnt & 1))
                                    # ``state_vals`` still holds these cells:
                                    # the restage read them in this same
                                    # acquire window and nothing has stored to
                                    # them since, so the source's second read is
                                    # pure redundancy on the per-chunk critical
                                    # path.  The fragment stays live past this
                                    # point for the rescale, so forwarding it
                                    # costs no extra live range.
                                    for sub in range(4):
                                        for group in range(4):
                                            dk = sub * 32 + group * 8
                                            element = (
                                                (dk // 64) * 8192
                                                + cg1_tidx * 64
                                                + _swizzle_xor_128b(cg1_tidx, dk % 64)
                                            )
                                            words = K.alloc_local((4,), "uint32")
                                            for pair in range(4):
                                                K.assign(
                                                    words[pair],
                                                    _pack_io_pair(
                                                        state_vals[sub * 32 + group * 8 + 2 * pair],
                                                        state_vals[
                                                            sub * 32 + group * 8 + 2 * pair + 1
                                                        ],
                                                        io_dtype,
                                                    ),
                                                )
                                            K.ptx.st.shared.v4.b32(
                                                arena.ptr_to([_CKPT_BASE + element * 2]),
                                                words[0],
                                                words[1],
                                                words[2],
                                                words[3],
                                            )
                                    K.ptx.fence.proxy.async_.shared__cta()
                                    with K.If(_elected()), K.Then():
                                        _arrive_barrier(arena, _BAR_CKPT_READY, 0)
                                    K.assign(ckpt_cnt, ckpt_cnt + 1)
                            for sub in range(4):
                                rescaled = K.alloc_local((32,), "float32")
                                for pair in range(16):
                                    s0, s1 = _fmul2(
                                        state_vals[sub * 32 + 2 * pair],
                                        state_vals[sub * 32 + 2 * pair + 1],
                                        cumprod_total,
                                        cumprod_total,
                                    )
                                    K.assign(rescaled[2 * pair], s0)
                                    K.assign(rescaled[2 * pair + 1], s1)
                                K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                                    K.cast(
                                        tmem_col
                                        + _TM_STATE
                                        + sub * 32
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                    *[rescaled[i] for i in range(32)],
                                )
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            _arrive_barrier(arena, _BAR_STATE_SCALE_DONE, 0)

                        # ---- per-row gate registers ------------------------
                        cumprod_vals = K.alloc_local((16,), "float32")
                        decay_scale = K.alloc_local((16,), "float32")
                        cumsum_cols = K.alloc_local((16,), "float32")
                        for group in range(8):
                            col_base = (lane % 4) * 2 + group * 8
                            pair_lo = K.local_scalar("float32")
                            pair_hi = K.local_scalar("float32")
                            K.ptx.ld.shared.v2.f32(
                                pair_lo,
                                pair_hi,
                                arena.ptr_to([_CUMPROD_BASE + gate_idx * 256 + col_base * 4]),
                            )
                            K.assign(cumprod_vals[group * 2], pair_lo)
                            K.assign(cumprod_vals[group * 2 + 1], pair_hi)
                            K.ptx.ld.shared.v2.f32(
                                cumsum_cols[group * 2],
                                cumsum_cols[group * 2 + 1],
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_idx * 256 + col_base * 4]),
                            )
                        last_cumsumlog = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            last_cumsumlog,
                            arena.ptr_to([_CUMSUMLOG_BASE + gate_idx * 256 + 63 * 4]),
                        )
                        for pair in range(8):
                            d0, d1 = _fsub2(
                                last_cumsumlog,
                                last_cumsumlog,
                                cumsum_cols[pair * 2],
                                cumsum_cols[pair * 2 + 1],
                            )
                            K.assign(decay_scale[pair * 2], _exp2(d0))
                            K.assign(decay_scale[pair * 2 + 1], _exp2(d1))
                        _arrive_barrier(arena, _BAR_GATE_DONE, gate_idx)

                        # ---- Y = V - cumprod * (K*S), packed 16-bit --------
                        v_idx = K.local_scalar("int32", init=v_ps.stage)
                        v_phase = K.local_scalar("int32", init=v_ps.phase)
                        v_ps.advance()
                        _wait_barrier(arena, _BAR_V_READY, v_idx, v_phase)
                        v_frags = K.alloc_local((32,), "uint32")
                        for piece in range(8):
                            m0 = piece % 4
                            sub = piece // 4
                            frag = K.alloc_local((4,), "uint32")
                            _ldmatrix_x4(
                                frag,
                                arena.ptr_to(
                                    [
                                        v_base
                                        + v_idx * 16384
                                        + v_sub_off
                                        + (
                                            (v_tok + m0 * 16) * 64
                                            + _swizzle_xor_128b(v_tok + m0 * 16, v_col + sub * 16)
                                        )
                                        * 2
                                    ]
                                ),
                                trans=True,
                            )
                            for i in range(4):
                                K.assign(v_frags[(4 * m0 + i) * 2 + sub], frag[i])
                        with K.If(have_state), K.Then():
                            _wait_barrier(arena, _BAR_K_STATE_READY, 0, kst_ps.phase)
                            kst_ps.advance()
                            for sub in range(2):
                                k_state_vec = K.alloc_local((32,), "float32")
                                K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
                                    *[k_state_vec[i] for i in range(32)],
                                    K.cast(
                                        tmem_col
                                        + _TM_CG1
                                        + K.shift_left(row_id + sub * 16, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                                for j in range(16):
                                    s0, s1 = _fmul2(
                                        k_state_vec[2 * j],
                                        k_state_vec[2 * j + 1],
                                        cumprod_vals[(2 * j // 4) * 2 + (2 * j) % 2],
                                        cumprod_vals[((2 * j + 1) // 4) * 2 + (2 * j + 1) % 2],
                                    )
                                    word = _pack_io_pair(s0, s1, io_dtype)
                                    _sub_io_pair(
                                        v_frags[j * 2 + sub], v_frags[j * 2 + sub], word, io_dtype
                                    )
                        for sub in range(2):
                            K.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
                                K.cast(
                                    tmem_col + _TM_Y + K.shift_left(row_id + sub * 16, K.int32(16)),
                                    "uint32",
                                ),
                                *[v_frags[j * 2 + sub] for j in range(16)],
                            )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _arrive_barrier(arena, _BAR_Y_INP_READY, 0)

                        # ---- U epilogue: decayed-U publish -----------------
                        # Nothing consumes the undecayed U, so unlike the
                        # prefill sibling there is no second republish.
                        _wait_barrier(arena, _BAR_U_ACC_READY, 0, uacc_ps.phase)
                        uacc_ps.advance()
                        _arrive_barrier(arena, _BAR_V_DONE, v_idx)
                        u_vals = K.alloc_local((64,), "float32")
                        for sub in range(2):
                            K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
                                *[u_vals[sub * 32 + i] for i in range(32)],
                                K.cast(
                                    tmem_col
                                    + _TM_CG1
                                    + K.shift_left(row_id + sub * 16, K.int32(16)),
                                    "uint32",
                                ),
                            )
                        for sub in range(2):
                            decay_pack = K.alloc_local((16,), "uint32")
                            for j in range(16):
                                d0, d1 = _fmul2(
                                    u_vals[sub * 32 + 2 * j],
                                    u_vals[sub * 32 + 2 * j + 1],
                                    decay_scale[(2 * j // 4) * 2 + (2 * j) % 2],
                                    decay_scale[((2 * j + 1) // 4) * 2 + (2 * j + 1) % 2],
                                )
                                K.assign(decay_pack[j], _pack_io_pair(d0, d1, io_dtype))
                            K.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
                                K.cast(
                                    tmem_col
                                    + _TM_DECAY_U
                                    + K.shift_left(row_id + sub * 16, K.int32(16)),
                                    "uint32",
                                ),
                                *[decay_pack[i] for i in range(16)],
                            )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _arrive_barrier(arena, _BAR_DECAY_U_READY, 0)

                    # ---- final state: TMEM -> GMEM after the last chunk ----
                    # The wait/arrive pair runs whether or not the final state
                    # is stored; it also closes the last chunk's state edge.
                    _wait_barrier(arena, _BAR_STATE_ACC_READY, 0, state_ps.phase)
                    state_ps.advance()
                    if store_final_state:
                        with K.If(wend == num_chunks_b), K.Then():
                            for sub in range(4):
                                final_vals = K.alloc_local((32,), "float32")
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                    *[final_vals[i] for i in range(32)],
                                    K.cast(
                                        tmem_col
                                        + _TM_STATE
                                        + sub * 32
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                                _store_state_row(
                                    final_state, state_index + sub * 32, final_vals, 32, state_dtype
                                )
                    _arrive_barrier(arena, _BAR_STATE_SCALE_DONE, 0)
                if store_final_state:
                    with K.If(n_local <= 0), K.Then():
                        # Empty sequence: never touches TMEM; pass the initial
                        # state through when present, else store state zero.
                        with K.If(wend == num_chunks_b), K.Then():
                            # A runtime loop, as the source writes it: this path
                            # runs only for empty sequences, and unrolling it
                            # would leave 256 unreachable global accesses inside
                            # CG1's persistent loop body.
                            with K.serial(128, unroll=False) as key:
                                if use_initial_state:
                                    value = _load_state_value(
                                        initial_state, state_index + key, state_dtype
                                    )
                                else:
                                    value = K.float32(0.0)
                                _store_state_value(
                                    final_state, state_index + key, value, state_dtype
                                )
                sched_consume(sched_ps, tile)
            _arrive_barrier(arena, _BAR_TMEM_DONE, 0)
            if checkpoints:
                _wait_barrier(arena, _BAR_CKPT_DONE, 0, 1 - (ckpt_cnt & 1))

        with gate_beta:
            gate_ps = K.PipelineState(3, phase=1)
            beta_ps = K.PipelineState(3, phase=1)
            sched_ps = K.PipelineState(2, phase=0)
            lidx = lane
            a_l2 = K.local_scalar("float32", init=K.float32(0.0))
            bias = K.local_scalar("float32", init=K.float32(0.0))
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                head = item[1]
                cstart = item[4]
                wend = item[3]
                batch_start = item[6]
                batch_end = item[7]
                n_local = wend - cstart
                if safe_gate:
                    with K.If(n_local > 0), K.Then():
                        a_value = K.local_scalar("float32")
                        K.ptx.ld.global_.f32(a_value, a_log.ptr_to([head]))
                        K.assign(a_l2, -_exp2(a_value * K.float32(_RCP_LN2)) * K.float32(_RCP_LN2))
                        K.ptx.ld.global_.f32(bias, dt_bias.ptr_to([head]))
                with K.If(n_local > 0), K.Then():
                    with K.serial(n_local, unroll=False) as local_idx:
                        chunk = cstart + local_idx
                        chunk_offset = batch_start + chunk * _BT
                        gate_idx = K.local_scalar("int32", init=gate_ps.stage)
                        gate_phase = K.local_scalar("int32", init=gate_ps.phase)
                        gate_ps.advance()
                        oob_neutral = 0.0 if log_gate else 1.0
                        gate_vals = K.alloc_local((2,), "float32")
                        for col in range(2):
                            tok = chunk_offset + lidx + col * 32
                            K.assign(gate_vals[col], K.float32(oob_neutral))
                            with K.If(tok < batch_end), K.Then():
                                clamped = K.if_then_else(tok < batch_end - 1, tok, batch_end - 1)
                                K.ptx.ld.global_.f32(
                                    gate_vals[col],
                                    gate.ptr_to([K.cast(clamped, "int64") * n_heads_out + head]),
                                )
                        if safe_gate:
                            for col in range(2):
                                tok = chunk_offset + lidx + col * 32
                                contribution = a_l2 * _softplus(gate_vals[col] + bias)
                                K.assign(
                                    gate_vals[col],
                                    K.if_then_else(tok < batch_end, contribution, K.float32(0.0)),
                                )
                        elif log_gate:
                            for col in range(2):
                                K.assign(gate_vals[col], gate_vals[col] * K.float32(_RCP_LN2))
                        else:
                            for col in range(2):
                                K.assign(gate_vals[col], _lg2(gate_vals[col] + K.float32(1e-10)))
                        for offset in (1, 2, 4, 8, 16):
                            for col in range(2):
                                neighbor = _shfl_up_f32(gate_vals[col], offset)
                                K.assign(
                                    gate_vals[col],
                                    K.if_then_else(
                                        lidx >= offset, gate_vals[col] + neighbor, gate_vals[col]
                                    ),
                                )
                        carry = _shfl_idx_f32(gate_vals[0], 31, 31)
                        K.assign(gate_vals[1], gate_vals[1] + carry)
                        _wait_barrier(arena, _BAR_GATE_DONE, gate_idx, gate_phase)
                        for col in range(2):
                            position = lidx + col * 32
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_CUMSUMLOG_BASE + gate_idx * 256 + position * 4]),
                                gate_vals[col],
                            )
                            K.ptx.st.shared.f32(
                                arena.ptr_to([_CUMPROD_BASE + gate_idx * 256 + position * 4]),
                                _exp2(gate_vals[col]),
                            )
                        _arrive_barrier(arena, _BAR_GATE_READY, gate_idx)

                        # ---- beta: cp.async per element or in-register sigmoid
                        beta_idx = K.local_scalar("int32", init=beta_ps.stage)
                        _wait_barrier(arena, _BAR_BETA_DONE, beta_ps.stage, beta_ps.phase)
                        beta_ps.advance()
                        if beta_sigmoid:
                            for col in range(2):
                                position = lidx + col * 32
                                tok = chunk_offset + position
                                beta_value = K.local_scalar("float32", init=K.float32(0.0))
                                with K.If(tok < batch_end), K.Then():
                                    raw_bits = K.local_scalar("uint16")
                                    K.ptx.ld.global_.b16(
                                        raw_bits,
                                        beta.ptr_to([K.cast(tok, "int64") * n_heads_out + head]),
                                    )
                                    raw = K.local_scalar("float32")
                                    if io_dtype == "float16":
                                        K.assign(
                                            raw,
                                            K.cast(K.reinterpret("float16", raw_bits), "float32"),
                                        )
                                    else:
                                        K.ptx.cvt.f32.bf16(raw, raw_bits)
                                    half_tanh = _tanh(raw * K.float32(0.5)) * K.float32(0.5)
                                    rounded_word = _pack_io_pair(
                                        half_tanh + K.float32(0.5),
                                        half_tanh + K.float32(0.5),
                                        io_dtype,
                                    )
                                    rounded_lo = K.local_scalar("float32")
                                    rounded_hi = K.local_scalar("float32")
                                    _unpack_io_pair(rounded_word, rounded_lo, rounded_hi, io_dtype)
                                    K.assign(beta_value, rounded_lo)
                                K.ptx.st.shared.f32(
                                    arena.ptr_to([_BETA_RING_BASE + beta_idx * 256 + position * 4]),
                                    beta_value,
                                )
                            _arrive_barrier(arena, _BAR_BETA_READY, beta_idx)
                        else:
                            for col in range(2):
                                position = lidx + col * 32
                                tok = chunk_offset + position
                                copy_size = K.if_then_else(
                                    tok < batch_end, K.uint32(4), K.uint32(0)
                                )
                                K.ptx["cp.async.ca.shared.global"](
                                    arena.ptr_to([_BETA_RING_BASE + beta_idx * 256 + position * 4]),
                                    beta.ptr_to([K.cast(tok, "int64") * n_heads_out + head]),
                                    K.uint32(4),
                                    copy_size,
                                )
                            K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                                _barrier_ptr(arena, _BAR_BETA_READY, beta_idx)
                            )
                sched_consume(sched_ps, tile)
            for _ in range(3):
                _wait_barrier(arena, _BAR_GATE_DONE, gate_ps.stage, gate_ps.phase)
                gate_ps.advance()
            for _ in range(3):
                _wait_barrier(arena, _BAR_BETA_DONE, beta_ps.stage, beta_ps.phase)
                beta_ps.advance()

        # ================================================================
        # Warp 10: sole tcgen05 issuer and TMEM lifecycle
        # ================================================================
        with mma:
            tcgen_lane = K.local_scalar("uint32")
            tcgen_leader = K.local_scalar("uint32")
            K.ptx.elect_sync(tcgen_lane, tcgen_leader, K.uint32(0xFFFFFFFF))
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                arena.ptr_to([_TMEM_MAILBOX]), K.uint32(512)
            )
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([_TMEM_MAILBOX]))
            kvacc_ps = K.PipelineState(1, phase=1)
            kq_ps = K.PipelineState(kq_stages, phase=0)
            cg0_ps = K.PipelineState(2, phase=1)
            kqf_ps = K.PipelineState(kq_stages, phase=0)
            tinv_ps = K.PipelineState(3, phase=0)
            sinp_ps = K.PipelineState(1, phase=0)
            y_ps = K.PipelineState(1, phase=0)
            du_ps = K.PipelineState(1, phase=0)
            sched_ps = K.PipelineState(2, phase=0)
            leader = tcgen_leader

            def fused_kk(member_one):
                # M=128 spans both pair boxes of the stage while N=64 spans only
                # the member's box, so half of each accumulator is never read.
                # The shape is inherited from the prefill kernel's genuinely
                # fused KK/QK pair and is kept: dropping it would change the
                # issue count and the accumulator ring geometry.
                fused_idx = K.local_scalar("int32", init=cg0_ps.stage)
                _wait_barrier(arena, _BAR_CG0_ACC_DONE, cg0_ps.stage, cg0_ps.phase)
                cg0_ps.advance()
                kqf_idx = K.local_scalar("int32", init=kqf_ps.stage)
                _wait_barrier(arena, _BAR_KQ_READY, kqf_ps.stage, kqf_ps.phase)
                kqf_ps.advance()
                pair_desc = _raw_descriptor(arena, kq_base + kqf_idx * 32768, 16, 1024, 2)
                b_desc = pair_desc + K.uint64(_KQ_BOX) if member_one else pair_desc
                destination = tmem_base + _TM_CG0 + fused_idx * 64
                _tcgen_mma_ss(
                    destination, pair_desc, b_desc, idesc_n64, leader=leader, accumulate=0
                )
                _tcgen_mma_ss(
                    destination,
                    pair_desc + K.uint64(_KQ_SEG),
                    b_desc + K.uint64(_KQ_SEG),
                    idesc_n64,
                    leader=leader,
                    accumulate=1,
                )
                _tcgen_commit(arena, _BAR_CG0_ACC_READY, fused_idx, leader=leader)

            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                n_local = item[3] - item[4]
                with K.If(n_local > 0), K.Then():
                    fused_kk(False)
                with K.If(n_local > 1), K.Then():
                    fused_kk(True)
                with K.serial(n_local, unroll=False) as local_idx:
                    member = local_idx % 2
                    if use_initial_state:
                        with K.If(local_idx == 0), K.Then():
                            _tcgen_commit(arena, _BAR_STATE_ACC_READY, 0, leader=leader)
                            kvacc_ps.advance()
                        have_state = K.bool(True)
                    else:
                        have_state = local_idx > 0
                    kq_idx = K.local_scalar("int32", init=kq_ps.stage)
                    kq_ps.advance()
                    base_desc = _raw_descriptor(arena, kq_base + kq_idx * 32768, 16, 1024, 2)
                    desc_k = base_desc + K.cast(member * _KQ_BOX, "uint64")
                    desc_kt = _raw_descriptor(
                        arena, kq_base + kq_idx * 32768, 16384, 1024, 2
                    ) + K.cast(member * _KQ_BOX, "uint64")

                    with K.If(member == 1), K.Then():
                        with K.If(local_idx + 2 < n_local), K.Then():
                            fused_kk(True)

                    # ---- GEMM 3: (K*S)^T = packed S^T @ K^T ----------------
                    with K.If(have_state), K.Then():
                        _wait_barrier(arena, _BAR_STATE_INP_READY, 0, sinp_ps.phase)
                        sinp_ps.advance()
                        _tcgen_mma_ts(
                            tmem_base + _TM_CG1,
                            tmem_base + _TM_STATE_INP,
                            desc_k,
                            idesc_n64,
                            leader=leader,
                            accumulate=0,
                        )
                        _tcgen_mma_ts(
                            tmem_base + _TM_CG1,
                            tmem_base + _TM_STATE_INP + 32,
                            desc_k + K.uint64(_KQ_SEG),
                            idesc_n64,
                            leader=leader,
                            accumulate=1,
                        )
                        _tcgen_commit(arena, _BAR_K_STATE_READY, 0, leader=leader)

                    # ---- GEMM 5: U^T = packed Y^T @ T_inv ------------------
                    tinv_idx = K.local_scalar("int32", init=tinv_ps.stage)
                    _wait_barrier(arena, _BAR_T_INV_READY, tinv_ps.stage, tinv_ps.phase)
                    tinv_ps.advance()
                    _wait_barrier(arena, _BAR_Y_INP_READY, 0, y_ps.phase)
                    y_ps.advance()
                    _tcgen_mma_ts(
                        tmem_base + _TM_CG1,
                        tmem_base + _TM_Y,
                        _raw_descriptor(arena, tinv_base + tinv_idx * 8192, 16, 1024, 2),
                        idesc_n64,
                        leader=leader,
                        accumulate=0,
                    )
                    _tcgen_commit(arena, _BAR_U_ACC_READY, 0, leader=leader)
                    _tcgen_commit(arena, _BAR_T_INV_DONE, tinv_idx, leader=leader)

                    with K.If(member == 0), K.Then():
                        with K.If(local_idx + 2 < n_local), K.Then():
                            fused_kk(False)

                    # ---- GEMM 7: S^T += packed decayed-U^T @ K -------------
                    _wait_barrier(arena, _BAR_DECAY_U_READY, 0, du_ps.phase)
                    du_ps.advance()
                    kvacc_ps.advance()
                    _tcgen_mma_ts(
                        tmem_base + _TM_STATE,
                        tmem_base + _TM_DECAY_U,
                        desc_kt,
                        idesc_n128,
                        leader=leader,
                        accumulate=have_state,
                        b_step_units=128,
                    )
                    _tcgen_commit(arena, _BAR_STATE_ACC_READY, 0, leader=leader)
                    _tcgen_commit(arena, _BAR_KQ_DONE, kq_idx, leader=leader)
                sched_consume(sched_ps, tile)
            _wait_barrier(arena, _BAR_TMEM_DONE, 0, 0)
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                K.cast(tmem_base, "uint32"), K.uint32(512)
            )

        # ================================================================
        # Warp 9: TMA-LDG and scheduler producer
        # ================================================================
        with tma:
            kq_ps = K.PipelineState(kq_stages, phase=1)
            v_ps = K.PipelineState(2, phase=1)
            sched_ps = K.PipelineState(2, phase=1)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wend = item[3]
                cstart = item[4]
                head_k = head if k_ratio == 1 else head // k_ratio
                head_v = head if v_ratio == 1 else head // v_ratio
                desc_k_slot = _descriptor_slot(descriptor_workspace, n_desc, 0, batch)
                desc_v_slot = _descriptor_slot(descriptor_workspace, n_desc, 1, batch)
                with K.If(_elected()), K.Then():
                    for slot in (desc_k_slot, desc_v_slot):
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(slot)

                def issue_k(stage, box, tok):
                    # One chunk's K tile: two 8-KB 64-channel subtiles 16384 B
                    # apart inside the box the pair member owns.
                    for d_coord, byte_off in ((0, 0), (64, 16384)):
                        K.ptx[_TMA_G2S_3D](
                            arena.ptr_to([kq_base + stage * 32768 + box * 8192 + byte_off]),
                            desc_k_slot,
                            K.int32(d_coord),
                            head_k,
                            tok,
                            _barrier_ptr(arena, _BAR_KQ_READY, stage),
                        )

                def issue_v(stage, tok):
                    for d_coord, byte_off in ((0, 0), (64, 8192)):
                        K.ptx[_TMA_G2S_3D](
                            arena.ptr_to([v_base + stage * 16384 + byte_off]),
                            desc_v_slot,
                            K.int32(d_coord),
                            head_v,
                            tok,
                            _barrier_ptr(arena, _BAR_V_READY, stage),
                        )

                with K.If(wend > cstart), K.Then():
                    # The item's first chunk always lands in box 0 of its stage.
                    kq_idx = K.local_scalar("int32", init=kq_ps.stage)
                    _wait_barrier(arena, _BAR_KQ_DONE, kq_ps.stage, kq_ps.phase)
                    kq_ps.advance()
                    with K.If(_elected()), K.Then():
                        _expect_tx(arena, _BAR_KQ_READY, kq_idx, 16384)
                        issue_k(kq_idx, 0, cstart * _BT)
                    with K.serial(wend - cstart - 1, unroll=False) as ahead:
                        chunk = cstart + 1 + ahead
                        member = (chunk - cstart) % 2
                        loop_kq_idx = K.local_scalar("int32", init=kq_ps.stage)
                        _wait_barrier(arena, _BAR_KQ_DONE, kq_ps.stage, kq_ps.phase)
                        kq_ps.advance()
                        with K.If(_elected()), K.Then():
                            _expect_tx(arena, _BAR_KQ_READY, loop_kq_idx, 16384)
                            # Member parity selects the box through two static
                            # sites, not one site with a runtime byte offset.
                            with K.If(member == 0), K.Then():
                                issue_k(loop_kq_idx, 0, chunk * _BT)
                            with K.If(member == 1), K.Then():
                                issue_k(loop_kq_idx, 1, chunk * _BT)
                        v_idx = K.local_scalar("int32", init=v_ps.stage)
                        _wait_barrier(arena, _BAR_V_DONE, v_ps.stage, v_ps.phase)
                        v_ps.advance()
                        with K.If(_elected()), K.Then():
                            _expect_tx(arena, _BAR_V_READY, v_idx, 16384)
                            issue_v(v_idx, (chunk - 1) * _BT)
                    tail_v_idx = K.local_scalar("int32", init=v_ps.stage)
                    _wait_barrier(arena, _BAR_V_DONE, v_ps.stage, v_ps.phase)
                    v_ps.advance()
                    with K.If(_elected()), K.Then():
                        _expect_tx(arena, _BAR_V_READY, tail_v_idx, 16384)
                        issue_v(tail_v_idx, (wend - 1) * _BT)
                if dynamic_scheduler:
                    _wait_barrier(arena, _BAR_SCHED_DONE, sched_ps.stage, sched_ps.phase)
                    with K.If(_elected()), K.Then():
                        ticket = K.local_scalar("uint32")
                        K.ptx.atom.global_.add.u32(ticket, scheduler.ptr_to([0]), K.uint32(1))
                        K.ptx.st.shared.u32(
                            arena.ptr_to([_SCHED_BASE + sched_ps.stage * 4]),
                            K.uint32(num_sms) + ticket,
                        )
                    K.cuda.warp_sync()
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([_SCHED_BASE + sched_ps.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, _BAR_SCHED_READY, sched_ps.stage)
                    sched_ps.advance()
                else:
                    K.assign(tile, tile + num_sms)
            for _ in range(kq_stages):
                _wait_barrier(arena, _BAR_KQ_DONE, kq_ps.stage, kq_ps.phase)
                kq_ps.advance()
            for _ in range(2):
                _wait_barrier(arena, _BAR_V_DONE, v_ps.stage, v_ps.phase)
                v_ps.advance()

        # ================================================================
        # Warp 11: epilogue checkpoint TensorMap stores
        # ================================================================
        with epilogue:
            sched_ps = K.PipelineState(2, phase=0)
            ckpt_cnt = K.local_scalar("int32", init=K.int32(0))
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cstart = item[4]
                n_local = wend - cstart
                if checkpoints:
                    desc_c_slot = _descriptor_slot(descriptor_workspace, n_desc, 2, batch)
                    ckpt_chunks = K.local_scalar("int32", init=checkpoint_every_n // _BT)
                    # Pre-chunk indexing: the snapshot CG1 stages is the state
                    # entering the chunk, so the window is [wstart, wend) and
                    # the running modulus starts at cstart, with no pre-loop
                    # store.
                    ckpt_coord = K.local_scalar(
                        "int32", init=(wstart + ckpt_chunks - 1) // ckpt_chunks
                    )
                    ckpt_mod = K.local_scalar("int32", init=cstart % ckpt_chunks)
                    with K.If(_elected()), K.Then():
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc_c_slot)
                    with K.If(n_local > 0), K.Then():
                        with K.serial(n_local, unroll=False) as local_idx:
                            chunk = cstart + local_idx
                            did_ckpt = K.local_scalar("int32", init=K.int32(0))
                            with (
                                K.If(K.And(K.And(chunk >= wstart, chunk < wend), ckpt_mod == 0)),
                                K.Then(),
                            ):
                                _wait_barrier(arena, _BAR_CKPT_READY, 0, ckpt_cnt & 1)
                                for value_coord, byte_off in ((0, 0), (64, 16384)):
                                    K.ptx[_TMA_S2G_4D](
                                        desc_c_slot,
                                        K.int32(value_coord),
                                        K.int32(0),
                                        ckpt_coord,
                                        head,
                                        arena.ptr_to([_CKPT_BASE + byte_off]),
                                    )
                                K.ptx.cp.async_.bulk.commit_group()
                                K.assign(ckpt_coord, ckpt_coord + 1)
                                K.assign(did_ckpt, K.int32(1))
                            K.assign(ckpt_mod, ckpt_mod + 1)
                            with K.If(ckpt_mod == ckpt_chunks), K.Then():
                                K.assign(ckpt_mod, K.int32(0))
                            with K.If(did_ckpt == 1), K.Then():
                                K.ptx.cp.async_.bulk.wait_group.read(0)
                                _arrive_barrier(arena, _BAR_CKPT_DONE, 0)
                                K.assign(ckpt_cnt, ckpt_cnt + 1)
                sched_consume(sched_ps, tile)

    return main


def _normalized_config(config):
    config = {key: value for key, value in config.items() if key != "label"}
    config.setdefault("seq_lens", (64,))
    config["seq_lens"] = tuple(int(value) for value in config["seq_lens"])
    config.setdefault("heads", 1)
    config.setdefault("k_heads", config["heads"])
    config.setdefault("v_heads", config["heads"])
    config.setdefault("io_dtype", "bfloat16")
    config.setdefault("state_dtype", "float32")
    config.setdefault("cu_dtype", "int32")
    config.setdefault("num_sms", 148)
    # The backward plan's regen call: per-chunk checkpoints, no final state.
    config.setdefault("checkpoint_every_n_tokens", 64)
    config.setdefault("use_initial_state", False)
    config.setdefault("store_final_state", False)
    config.setdefault("log_gate", True)
    config.setdefault("safe_gate", False)
    config.setdefault("beta_sigmoid", False)
    config.setdefault("dynamic_scheduler", False)
    config.setdefault("split", False)
    # This kernel's prologue runs the order pass only when it is the backward
    # pair's first work-table consumer.  Generated uncut rows are the default;
    # scratch LPT ordering owns split rows.
    config.setdefault("run_order", True)
    config.setdefault("order_generate", not config.get("split", False))

    if config["io_dtype"] not in ("bfloat16", "float16"):
        raise ValueError("io_dtype must be 'bfloat16' or 'float16'")
    if config["state_dtype"] not in ("float32", "bfloat16"):
        raise ValueError("state_dtype must be 'float32' or 'bfloat16'")
    if config["cu_dtype"] not in ("int32", "int64"):
        raise ValueError("cu_dtype must be 'int32' or 'int64'")
    if any(length < 0 for length in config["seq_lens"]):
        raise ValueError("sequence lengths must be nonnegative")
    if sum(config["seq_lens"]) == 0:
        raise ValueError("at least one sequence must be nonempty for CUDA TensorMap encoding")
    heads = int(config["heads"])
    for name in ("k_heads", "v_heads"):
        value = int(config[name])
        if value <= 0 or heads % value != 0:
            raise ValueError(f"{name}={value} must be a positive divisor of heads={heads}")
    if heads != max(int(config["k_heads"]), int(config["v_heads"])):
        raise ValueError("heads must equal max(k_heads, v_heads), matching the source HO rule")
    cadence = int(config["checkpoint_every_n_tokens"])
    if cadence < 0 or cadence % _BT != 0:
        raise ValueError("checkpoint cadence must be zero or a positive multiple of 64")
    if cadence not in (0, 64, 128, 192):
        raise ValueError("validated checkpoint cadences are 0, 64, 128, and 192")
    if not (cadence or config["store_final_state"]):
        raise ValueError("the checkpoint series or the final state is required")
    if not (config["use_initial_state"] or config["store_final_state"]) and (
        config["state_dtype"] != "float32"
    ):
        raise ValueError("state_dtype is float32 when neither state tensor is present")
    if config["split"] and config["order_generate"]:
        raise ValueError("split work rows require scratch ordering")
    if not config["run_order"] and config["dynamic_scheduler"]:
        raise ValueError("the dynamic scheduler needs the order pass to zero its ticket ring")
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
    """Return the source-ordered prologue and persistent main kernels."""
    config = _normalized_config(config)
    rows = _work_rows(config["seq_lens"], config["heads"], split=config["split"])
    num_ctas = min(int(config["num_sms"]), max(len(rows), 1))
    prologue = _make_prologue(
        run_order=bool(config["run_order"]),
        order_generate=bool(config["order_generate"]),
        n_heads_out=int(config["heads"]),
        checkpoints=int(config["checkpoint_every_n_tokens"]) > 0,
        cu_dtype=config["cu_dtype"],
    )
    main = _make_main(
        num_sms=num_ctas,
        io_dtype=config["io_dtype"],
        state_dtype=config["state_dtype"],
        cu_dtype=config["cu_dtype"],
        use_initial_state=bool(config["use_initial_state"]),
        store_final_state=bool(config["store_final_state"]),
        checkpoints=int(config["checkpoint_every_n_tokens"]) > 0,
        log_gate=bool(config["log_gate"]),
        safe_gate=bool(config["safe_gate"]),
        beta_sigmoid=bool(config["beta_sigmoid"]),
        dynamic_scheduler=bool(config["dynamic_scheduler"]),
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


def _checkpoint_count(seq_lens, cadence):
    # Matches the packed per-batch layout of the source's
    # emit_checkpoint_seq_descs: count_b = (len - 1) // cadence + 1 per
    # nonempty sequence, running-prefix-summed.
    if not cadence:
        return 0
    return sum(0 if length == 0 else (length - 1) // cadence + 1 for length in seq_lens)


def _prepare_work_tables(torch, config):
    base = torch.tensor(
        _work_rows(config["seq_lens"], config["heads"], split=config["split"]),
        dtype=torch.int32,
        device="cuda",
    )

    def one_side():
        # Without the order pass the caller owns an already-ordered table.
        work_items = torch.empty_like(base) if config["run_order"] else base.clone()
        staging = None if config["order_generate"] else base.clone()
        # The order pass zeroes every cell of this region; the ticket counter
        # is its first cell, matching the engine's shared sched block.
        sched_all = torch.zeros(4, dtype=torch.int32, device="cuda")
        return {
            "work_items": work_items,
            "work_count": torch.tensor([base.shape[0]], dtype=torch.int32, device="cuda"),
            "staging": staging,
            "sched_all": sched_all,
            "scheduler": sched_all[:1] if config["dynamic_scheduler"] else None,
        }

    return {"tirx": one_side(), "source": one_side()}


def _new_outputs(torch, config):
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    state_t = torch.bfloat16 if config["state_dtype"] == "bfloat16" else torch.float32
    result = {}
    if config["store_final_state"]:
        result["final_state"] = torch.full(
            (len(config["seq_lens"]), config["heads"], _DV, _DK),
            float("nan"),
            dtype=state_t,
            device="cuda",
        )
    checkpoint_count = _checkpoint_count(config["seq_lens"], config["checkpoint_every_n_tokens"])
    if checkpoint_count:
        result["checkpoints"] = torch.full(
            (checkpoint_count, config["heads"], _DV, _DK), float("nan"), dtype=io_t, device="cuda"
        )
    return result


def _prepare_data(config):
    import torch

    config = _normalized_config(config)
    torch.manual_seed(20260827)
    total_tokens = sum(config["seq_lens"])
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    state_t = torch.bfloat16 if config["state_dtype"] == "bfloat16" else torch.float32
    k = (0.2 * torch.randn(total_tokens, config["k_heads"], _DK, device="cuda")).to(io_t)
    v = (0.2 * torch.randn(total_tokens, config["v_heads"], _DV, device="cuda")).to(io_t)
    # The scalar forget gate: raw logits under safe_gate, natural-log decay
    # under log_gate, raw linear alpha otherwise.
    gate = torch.empty(total_tokens, config["heads"], dtype=torch.float32, device="cuda")
    if config["safe_gate"]:
        gate.normal_(std=0.2)
    elif config["log_gate"]:
        gate.uniform_(-0.9, -0.01)
    else:
        gate.uniform_(0.4, 0.99)
    # The scalar update gate: fp32 post-sigmoid, or io-dtype logits when the
    # kernel applies the sigmoid.
    if config["beta_sigmoid"]:
        beta = (0.3 * torch.randn(total_tokens, config["heads"], device="cuda")).to(io_t)
    else:
        beta = torch.sigmoid(0.3 * torch.randn(total_tokens, config["heads"], device="cuda"))
    cu_t = torch.int64 if config["cu_dtype"] == "int64" else torch.int32
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(config["seq_lens"]).cumsum(0).tolist()], dtype=cu_t, device="cuda"
    )
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
        (0.01 * torch.randn(len(config["seq_lens"]), config["heads"], _DV, _DK, device="cuda")).to(
            state_t
        )
        if config["use_initial_state"]
        else None
    )
    outputs = {"tirx": _new_outputs(torch, config), "source": _new_outputs(torch, config)}
    workspace_bytes = _TENSOR_MAP_BYTES * (_TENSOR_MAP_ARRAYS * len(config["seq_lens"]) + 1)
    tirx_owner, tirx_workspace = _aligned_i64(torch, workspace_bytes)
    source_owner, source_workspace = _aligned_i64(torch, workspace_bytes)

    return {
        "config": config,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "tirx": outputs["tirx"],
        "source": outputs["source"],
        "work": _prepare_work_tables(torch, config),
        "tirx_workspace": tirx_workspace,
        "source_workspace": source_workspace,
        "_workspace_owners": (tirx_owner, source_owner),
    }


def prepare_data(**config):
    """Allocate the shared input set plus source/TIRx output buffers."""
    return _prepare_data(config)


def _encode_tiled_map(tensor, dimensions, strides, box):
    import torch
    from cuda.bindings import driver as cuda

    if tensor.dtype == torch.float16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    elif tensor.dtype == torch.bfloat16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    elif tensor.dtype == torch.float32:
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


def _base_tensor_maps(data, side):
    config = data["config"]
    total_tokens = data["k"].shape[0]

    def headed(tensor, channels, heads, box_channels):
        return _encode_tiled_map(
            tensor,
            (channels, heads, total_tokens),
            (tensor.stride(1) * tensor.element_size(), tensor.stride(0) * tensor.element_size()),
            (box_channels, 1, _BT),
        )

    output = data[side]
    maps = [
        headed(data["k"], _DK, config["k_heads"], 64),
        headed(data["v"], _DV, config["v_heads"], 64),
    ]
    checkpoints = output.get("checkpoints")
    if checkpoints is None:
        # The source aliases the checkpoint descriptor onto V's when the
        # series is absent.
        maps.append(maps[-1])
    else:
        maps.append(
            _encode_tiled_map(
                checkpoints,
                (_DK, _DV, checkpoints.shape[0], config["heads"]),
                (
                    checkpoints.stride(2) * checkpoints.element_size(),
                    checkpoints.stride(0) * checkpoints.element_size(),
                    checkpoints.stride(1) * checkpoints.element_size(),
                ),
                (64, _DV, 1, 1),
            )
        )
    return maps


def _tirx_launch(executables, data):
    import torch

    config = data["config"]
    prologue, main = executables
    maps = _base_tensor_maps(data, "tirx")
    work = data["work"]["tirx"]
    output = data["tirx"]
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    state_t = torch.bfloat16 if config["state_dtype"] == "bfloat16" else torch.float32
    beta_t = io_t if config["beta_sigmoid"] else torch.float32
    dummy_i32 = torch.zeros(8, dtype=torch.int32, device="cuda")
    dummy_io = torch.zeros(1, dtype=io_t, device="cuda")
    dummy_state = torch.zeros(1, dtype=state_t, device="cuda")
    dummy_f32 = torch.zeros(config["heads"], dtype=torch.float32, device="cuda")
    checkpoint = output.get("checkpoints", dummy_io)
    beta = data["beta"].to(beta_t)

    def launch(*, prologue_only=False):
        prologue(
            *maps,
            data["tirx_workspace"],
            data["cu_seqlens"],
            data["k"].view(torch.uint8).reshape(-1),
            data["v"].view(torch.uint8).reshape(-1),
            checkpoint.view(torch.uint8).reshape(-1),
            work["staging"].reshape(-1) if work["staging"] is not None else dummy_i32,
            work["work_count"],
            work["work_items"].reshape(-1),
            work["sched_all"],
            len(config["seq_lens"]),
            data["k"].stride(0) * data["k"].element_size(),
            data["v"].stride(0) * data["v"].element_size(),
            checkpoint.stride(0) * checkpoint.element_size() if checkpoint.ndim > 1 else 0,
            int(config["checkpoint_every_n_tokens"]),
        )
        if prologue_only:
            return
        main(
            data["tirx_workspace"],
            len(config["seq_lens"]),
            data["k"].reshape(-1),
            data["v"].reshape(-1),
            data["gate"].reshape(-1),
            data["a_log"] if data["a_log"] is not None else dummy_f32,
            data["dt_bias"] if data["dt_bias"] is not None else dummy_f32,
            beta.reshape(-1),
            data["cu_seqlens"],
            (
                data["initial_state"].reshape(-1)
                if data["initial_state"] is not None
                else dummy_state
            ),
            output.get("final_state", dummy_state).reshape(-1),
            work["work_items"].reshape(-1),
            work["work_count"],
            work["scheduler"] if work["scheduler"] is not None else dummy_i32,
            int(config["checkpoint_every_n_tokens"]),
        )

    launch._keep_alive = (maps, beta, dummy_i32, dummy_io, dummy_state, dummy_f32)
    return launch


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module("cudnn.linear_attention.frost.kernel.gdn_recompute_f16")


def _source_launch(data):
    import torch

    source = _load_reference_source()
    config = data["config"]
    output = data["source"]
    work = data["work"]["source"]
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    beta = data["beta"].to(io_t) if config["beta_sigmoid"] else data["beta"]
    stream = int(torch.cuda.current_stream().cuda_stream)

    def launch():
        source.chunk_gdn_recompute_sm100(
            data["k"],
            data["v"],
            data["gate"],
            beta,
            data["cu_seqlens"],
            data["initial_state"],
            output.get("final_state"),
            checkpoint_every_n_tokens=int(config["checkpoint_every_n_tokens"]),
            output_state_checkpoints=output.get("checkpoints"),
            work_items=work["work_items"],
            work_count=work["work_count"],
            sched_ctr=work["scheduler"],
            sched_all=work["sched_all"] if config["run_order"] else None,
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

    launch._keep_alive = (beta,)
    return launch


def _rms_ratio(torch, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(
            f"RMS operands must have the same shape: {actual.shape} != {expected.shape}"
        )
    actual = actual.detach().reshape(-1)
    expected = expected.detach().reshape(-1)
    elements = actual.numel()
    if elements == 0:
        return 0.0
    difference_square_sum = torch.zeros((), dtype=torch.float64, device=actual.device)
    expected_square_sum = torch.zeros((), dtype=torch.float64, device=actual.device)
    chunk_elements = 1 << 24
    for begin in range(0, elements, chunk_elements):
        end = min(begin + chunk_elements, elements)
        actual_chunk = actual[begin:end].double()
        expected_chunk = expected[begin:end].double()
        difference = actual_chunk - expected_chunk
        difference_square_sum.add_(difference.square().sum())
        expected_square_sum.add_(expected_chunk.square().sum())
    denominator = (expected_square_sum / elements).sqrt().clamp_min(1e-12)
    return float(((difference_square_sum / elements).sqrt() / denominator).item())


def _validate_outputs(data, *, sources):
    import math

    import torch

    limit = 0.01 if data["config"]["io_dtype"] == "float16" else 0.02
    failures = {}
    if "tirx" in sources and "source" in sources:
        for name in data["tirx"]:
            actual = data["tirx"][name]
            expected = data["source"][name]
            if name == "checkpoints":
                # The source clips checkpoint stores in hardware; compare only
                # the entries the source actually wrote and require the TIRx
                # side to leave the same tail untouched.
                written = ~torch.isnan(expected.float())
                if bool(torch.isnan(actual.float())[written].any()):
                    failures["tirx_vs_source.checkpoints.missing"] = float("nan")
                    continue
                actual = torch.where(written, actual, torch.zeros_like(actual))
                expected = torch.where(written, expected, torch.zeros_like(expected))
            ratio = _rms_ratio(torch, actual, expected)
            if not math.isfinite(ratio) or ratio >= limit:
                failures[f"tirx_vs_source.{name}"] = ratio
    if failures:
        raise AssertionError(
            f"GDN recompute validation failed for {data['config']}: {failures}; limit={limit}"
        )


def run_test(**config):
    """Compare TIRx with the upstream kernel on identical inputs."""
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
    return {"tokens": sum(config["seq_lens"]), "heads": config["heads"]}


def prepare_bench(**config):
    """Compile both TIRx launches without importing torch or touching CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config = _normalized_config(config)
    state = {
        "config": config,
        "executables": [compile_kernel(func) for func in get_kernel(**config)],
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then expose the exact two-launch paths to bench_suite."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = _normalized_config({**prepared["config"], **kwargs})
    data = _prepare_data(config)
    tirx_launch = _tirx_launch(prepared["executables"], data)
    tirx_launch()
    torch.cuda.synchronize()
    references = None
    if external_references_enabled():
        source_launch = _source_launch(data)
        source_launch()
        torch.cuda.synchronize()
        _validate_outputs(data, sources=("tirx", "source"))
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
