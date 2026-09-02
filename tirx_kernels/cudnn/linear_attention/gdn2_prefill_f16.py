# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell FP16/BF16 Gated DeltaNet v2 prefill kernels.

Upstream source:
``python/cudnn/linear_attention/frost/kernel/gdn2_prefill_f16.py``
(``prologue_kernel``, ``kernel``, and the two-launch ``run_prefill`` entry).
"""

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_gdn2_prefill_f16",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
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

# Deterministic pairwise cover of the frozen legal capability set.  Candidates
# are scored by uncovered pairs, ties use the categorical label order, and a
# reverse pass removes redundant cases.  Ragged cases include zero- and
# one-token sequences while retaining a nonzero TensorMap outer dimension.
CONFIGS = [
    {
        "label": "pairwise_00",
        "seq_lens": (31, 0, 63, 1),
        "heads": 4,
        "q_heads": 2,
        "k_heads": 4,
        "v_heads": 4,
        "cu_dtype": "int64",
        "use_initial_state": True,
        "checkpoint_every_n_tokens": 16,
        "l2norm": True,
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
        "store_final_state": False,
        "checkpoint_every_n_tokens": 32,
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
        "checkpoint_every_n_tokens": 48,
        "l2norm": True,
    },
    {
        "label": "pairwise_03",
        "seq_lens": (31, 0, 63, 1),
        "heads": 4,
        "q_heads": 2,
        "k_heads": 4,
        "v_heads": 4,
        "io_dtype": "float16",
        "state_dtype": "bfloat16",
        "store_final_state": False,
        "checkpoint_every_n_tokens": 16,
        "l2norm": True,
        "order_generate": False,
        "num_sms": 2,
    },
    {
        "label": "pairwise_04",
        "seq_lens": (96,),
        "heads": 4,
        "q_heads": 4,
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
        "seq_lens": (31, 0, 63, 1),
        "heads": 4,
        "q_heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "state_dtype": "bfloat16",
        "checkpoint_every_n_tokens": 48,
        "safe_gate": True,
        "beta_sigmoid": True,
        "dynamic_scheduler": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_06",
        "seq_lens": (96,),
        "heads": 4,
        "q_heads": 2,
        "k_heads": 4,
        "v_heads": 4,
        "state_dtype": "bfloat16",
        "cu_dtype": "int64",
        "use_initial_state": True,
        "store_final_state": False,
        "checkpoint_every_n_tokens": 32,
        "l2norm": True,
        "safe_gate": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_07",
        "seq_lens": (17, 65),
        "heads": 4,
        "q_heads": 2,
        "k_heads": 4,
        "v_heads": 4,
        "store_final_state": False,
        "checkpoint_every_n_tokens": 48,
        "order_generate": False,
        "split": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_08",
        "seq_lens": (31, 0, 63, 1),
        "heads": 1,
        "state_dtype": "bfloat16",
        "use_initial_state": True,
        "store_final_state": False,
        "l2norm": True,
        "safe_gate": True,
        "dynamic_scheduler": True,
    },
    {
        "label": "pairwise_09",
        "seq_lens": (31, 0, 63, 1),
        "heads": 4,
        "q_heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "cu_dtype": "int64",
        "checkpoint_every_n_tokens": 32,
        "l2norm": True,
        "beta_sigmoid": True,
        "num_sms": 2,
    },
    {
        "label": "pairwise_10",
        "seq_lens": (17, 65),
        "heads": 4,
        "q_heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "state_dtype": "bfloat16",
        "store_final_state": False,
        "checkpoint_every_n_tokens": 16,
        "num_sms": 2,
    },
    {
        "label": "pairwise_11",
        "seq_lens": (96,),
        "heads": 1,
        "state_dtype": "bfloat16",
        "store_final_state": False,
        "checkpoint_every_n_tokens": 16,
    },
    {
        "label": "pairwise_12",
        "seq_lens": (17, 65),
        "heads": 4,
        "q_heads": 2,
        "k_heads": 4,
        "v_heads": 4,
        "state_dtype": "bfloat16",
        "store_final_state": False,
        "num_sms": 2,
    },
    {
        "label": "pairwise_13",
        "seq_lens": (96,),
        "heads": 4,
        "q_heads": 4,
        "k_heads": 4,
        "v_heads": 2,
        "state_dtype": "bfloat16",
        "store_final_state": False,
        "checkpoint_every_n_tokens": 48,
        "num_sms": 2,
    },
]

BENCH_CONFIGS = [
    {
        "label": f"perf_b{batch}_s{seqlen}_h64_{mode}",
        "seq_lens": (seqlen,) * batch,
        "heads": 64,
        "l2norm": True,
        **({"store_final_state": True, "checkpoint_every_n_tokens": 16} if mode == "state" else {}),
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


_BT = 16
_DK = 128
_DV = 128
_TENSOR_MAP_BYTES = 128
_TENSOR_MAP_WORDS = 16
_TENSOR_MAP_ARRAYS = 8
_TRY_WAIT_TICKS = 10_000_000
_LOG2_E = 1.4426950408889634
_L2_NORM_EPS2 = 1.0e-24
_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"

_NORMAL_PROTOCOL = (
    (0, 48, 6, 5, 1, 128, 14),
    (88, 136, 6, 5, 1, 128, 14),
    (176, 224, 6, 5, 1, 128, 14),
    (264, 312, 6, 5, 1, 128, 14),
    (352, 400, 6, 5, 1, 128, 14),
    (440, 488, 6, 5, 1, 128, 14),
    (528, 536, 1, 2, 1, 128, 13),
    (552, None, 1, 0, 1, None, 13),
    (560, None, 1, 0, 1, None, 13),
    (568, None, 1, 0, 128, None, 13),
    (576, None, 1, 0, 128, None, 13),
    (584, None, 1, 0, 128, None, 13),
    (592, 608, 2, 2, 32, 1, 12),
    (624, None, 2, 0, 32, None, 12),
    (640, 656, 2, 2, 128, 1, 12),
    (672, None, 2, 0, 64, None, 13),
    (688, None, 4, 0, 128, None, 12),
    (720, None, 2, 0, 1, None, 13),
    (736, None, 4, 0, 1, None, 13),
    (768, None, 1, 0, 128, None, 13),
    (776, None, 1, 0, 128, None, 15),
    (784, 800, 2, 2, 128, 32, 15),
    (816, 824, 1, 1, 128, 32, 15),
    (832, 896, 8, 8, 1, 15, 15),
)

_CHECKPOINT_PROTOCOL = (
    (0, 32, 4, 3, 1, 128, 14),
    (56, 88, 4, 3, 1, 128, 14),
    (112, 144, 4, 3, 1, 128, 14),
    (168, 200, 4, 3, 1, 128, 14),
    (224, 256, 4, 3, 1, 128, 14),
    (280, 312, 4, 3, 1, 128, 14),
    (336, 344, 1, 2, 1, 128, 13),
    (360, None, 1, 0, 1, None, 13),
    (368, None, 1, 0, 1, None, 13),
    (376, None, 1, 0, 128, None, 13),
    (384, None, 1, 0, 128, None, 13),
    (392, None, 1, 0, 128, None, 13),
    (400, 416, 2, 2, 32, 1, 12),
    (432, None, 2, 0, 32, None, 12),
    (448, 464, 2, 2, 128, 1, 12),
    (480, None, 2, 0, 64, None, 13),
    (496, None, 4, 0, 128, None, 12),
    (528, None, 2, 0, 1, None, 13),
    (544, None, 4, 0, 1, None, 13),
    (576, None, 1, 0, 128, None, 13),
    (584, None, 1, 0, 128, None, 15),
    (592, 608, 2, 2, 128, 32, 15),
    (624, 640, 2, 2, 128, 32, 15),
    (656, 720, 8, 8, 1, 15, 15),
)


def _elected():
    lane = K.local_scalar("uint32")
    pred = K.local_scalar("uint32")
    K.ptx.elect_sync(lane, pred, K.uint32(0xFFFFFFFF))
    return pred == K.uint32(1)


def _copy_tensormap(src_map, dst):
    """Copy one host-encoded 128-byte TensorMap image into GMEM.

    The base images use ordinary global pointers because K's PTX instruction
    surface intentionally has no parameter-space load.  The prologue still
    publishes and patches the per-sequence descriptor exactly once, just as
    the source kernel does.
    """
    payload = K.alloc_local((4,), "uint64")
    source = K.reinterpret("uint64", src_map.ptr_to([0]))
    target = K.reinterpret("uint64", dst)
    for group in range(4):
        offset = K.uint64(group * 32)
        K.ptx.ld.global_.v4.b64(
            payload[0], payload[1], payload[2], payload[3], K.reinterpret("handle", source + offset)
        )
        for word in range(4):
            K.ptx.st.global_.b64(
                K.reinterpret("handle", target + K.uint64((group * 4 + word) * 8)), payload[word]
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


def _mul_io_pair(dst, lhs, rhs, io_dtype):
    if io_dtype == "float16":
        K.ptx.mul.f16x2(dst, lhs, rhs)
    else:
        K.ptx.mul.bf16x2(dst, lhs, rhs)


def _sub_io_pair(dst, lhs, rhs, io_dtype):
    if io_dtype == "float16":
        K.ptx.sub.f16x2(dst, lhs, rhs)
    else:
        K.ptx["sub.bf16x2"](dst, lhs, rhs)


def _movmatrix_b16(value):
    result = K.local_scalar("uint32")
    K.ptx["movmatrix.sync.aligned.m8n8.trans.b16"](result, value)
    return result


def _mma_m16n16k16(acc, a, b, io_dtype):
    opcode = (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        if io_dtype == "float16"
        else "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    )
    K.ptx[opcode](
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
    K.ptx[opcode](
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


def _fadd2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.add.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _fmul2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.mul.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _ffma2(a0, a1, b0, b1, c0, c1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.fma.rn.f32x2(
        packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1), K.cuda.make_float2(c0, c1)
    )
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


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


def _exp2(value):
    result = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(result, value)
    return result


def _rcp(value):
    result = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(result, value)
    return result


def _rsqrt(value):
    result = K.local_scalar("float32")
    K.ptx.rsqrt.approx.ftz.f32(result, value)
    return result


def _tanh(value):
    result = K.local_scalar("float32")
    K.ptx.tanh.approx.f32(result, value)
    return result


def _opaque_one():
    zero = K.local_scalar("float32")
    K.ptx.mov.b32(zero, K.uint32(0))
    return zero + K.float32(1.0)


def _raw_descriptor(arena, byte_offset, leading_bytes, stride_bytes, layout):
    shared = K.cuda.cvta_generic_to_shared(arena.ptr_to([byte_offset]))
    address = K.bitwise_and(K.shift_right(shared, K.uint32(4)), K.uint32(0x3FFF))
    fixed = (
        (((leading_bytes >> 4) & 0x3FFF) << 16) | (((stride_bytes >> 4) & 0x3FFF) << 32) | (1 << 46)
    )
    desc = K.bitwise_or(K.uint64(fixed), K.cast(address, "uint64"))
    return K.bitwise_or(desc, K.shift_left(K.uint64(layout & 7), K.uint64(61)))


def _swizzle_xor_128b(row, column, elem_bytes):
    return K.bitwise_xor(column, (row & 7) * (16 // elem_bytes))


def _raw_io_byte(base, stage, token, channel):
    segment = channel // 64
    within = channel - segment * 64
    element = segment * 1024 + token * 64 + _swizzle_xor_128b(token, within, 2)
    return base + stage * 4096 + element * 2


def _raw_f32_byte(base, stage, token, channel):
    segment = channel // 32
    within = channel - segment * 32
    element = segment * 512 + token * 32 + _swizzle_xor_128b(token, within, 4)
    return base + stage * 8192 + element * 4


def _tmem_cell(base, row, column):
    return base + column + K.shift_left(row, K.int32(16))


def _tcgen_mma_ts(
    dst, tmem_a, b_desc, idesc, *, n, k_extent, leader, b_transpose=False, accumulate=False
):
    inner_bytes = (n if b_transpose else k_extent) * 2
    swizzle_b = 128 if inner_bytes % 128 == 0 else 64 if inner_bytes % 64 == 0 else 32
    intra_b = 16 * swizzle_b if b_transpose else 32
    subtile_b = k_extent * swizzle_b if b_transpose else swizzle_b * n
    steps_b = k_extent // 16 if b_transpose else (swizzle_b // 2) // 16
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


def _make_prologue(
    *, run_order, order_generate, dynamic_scheduler, n_heads_out, checkpoints, cu_dtype
):
    cu_t = K.i64 if cu_dtype == "int64" else K.i32

    @K.kernel(warps=32, arch="sm_100a", grid=1)
    def prologue(
        base_q: K.gptr[K.i64],
        base_k: K.gptr[K.i64],
        base_v: K.gptr[K.i64],
        base_gate: K.gptr[K.i64],
        base_beta: K.gptr[K.i64],
        base_w: K.gptr[K.i64],
        base_o: K.gptr[K.i64],
        base_checkpoint: K.gptr[K.i64],
        descriptor_workspace: K.gptr[K.i64],
        cu_seqlens: K.gptr[cu_t],
        q: K.gptr[K.u8],
        k: K.gptr[K.u8],
        v: K.gptr[K.u8],
        gate: K.gptr[K.u8],
        beta: K.gptr[K.u8],
        w: K.gptr[K.u8],
        o: K.gptr[K.u8],
        checkpoint: K.gptr[K.u8],
        work_item_staging: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        work_items: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        n_batch: K.i32,
        q_row_stride_bytes: K.i32,
        k_row_stride_bytes: K.i32,
        v_row_stride_bytes: K.i32,
        gate_row_stride_bytes: K.i32,
        beta_row_stride_bytes: K.i32,
        w_row_stride_bytes: K.i32,
        o_row_stride_bytes: K.i32,
        checkpoint_row_stride_bytes: K.i32,
        checkpoint_every_n: K.i32,
    ):
        thread = K.thread_id()
        warp = K.warp_id()
        if run_order:
            order_arena = K.alloc_buffer((32_776,), K.u8, scope="shared.dyn", align=16)
            with K.If(thread == 0), K.Then():
                if dynamic_scheduler:
                    K.ptx.st.global_.s32(scheduler.ptr_to([0]), K.int32(0))

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

        maps = (base_q, base_k, base_v, base_gate, base_beta, base_w, base_o, base_checkpoint)
        bases = (q, k, v, gate, beta, w, o)
        strides = (
            q_row_stride_bytes,
            k_row_stride_bytes,
            v_row_stride_bytes,
            gate_row_stride_bytes,
            beta_row_stride_bytes,
            w_row_stride_bytes,
            o_row_stride_bytes,
        )
        for array_index in range(7):
            with K.If(warp == array_index), K.Then():
                with K.If(_elected()), K.Then():
                    with K.serial(n_batch) as batch:
                        begin = _load_cu(cu_seqlens, batch, cu_dtype)
                        end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                        slot = _descriptor_slot(descriptor_workspace, n_batch, array_index, batch)
                        _copy_tensormap(maps[array_index], slot)
                        _replace_tensormap_address(
                            slot,
                            bases[array_index].ptr_to(
                                [K.cast(begin, "int64") * strides[array_index]]
                            ),
                        )
                        _replace_tensormap_dim(slot, 2, end - begin)
                    K.ptx.fence.proxy.tensormap__generic.release.gpu()

        if checkpoints:
            with K.If(warp == 7), K.Then():
                with K.If(_elected()), K.Then():
                    checkpoint_prefix = K.local_scalar("int32", init=K.int32(0))
                    with K.serial(n_batch) as batch:
                        begin = _load_cu(cu_seqlens, batch, cu_dtype)
                        end = _load_cu(cu_seqlens, batch + 1, cu_dtype)
                        length = end - begin
                        count = K.local_scalar("int32", init=K.int32(0))
                        with K.If(length > 0), K.Then():
                            K.assign(count, (length - 1) // checkpoint_every_n + 1)
                        slot = _descriptor_slot(descriptor_workspace, n_batch, 7, batch)
                        _copy_tensormap(base_checkpoint, slot)
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


# KERNEL_SKETCH_START
def _make_main(
    *,
    num_sms,
    io_dtype,
    state_dtype,
    cu_dtype,
    use_initial_state,
    store_final_state,
    checkpoints,
    peel_cg1_first_chunk,
    l2norm,
    safe_gate,
    gate_scale_log2,
    beta_sigmoid,
    dynamic_scheduler,
    q_ratio,
    k_ratio,
    v_ratio,
    n_heads_out,
    full_tiles,
):
    io_t = K.f16 if io_dtype == "float16" else K.bf16
    state_t = K.bf16 if state_dtype == "bfloat16" else K.f32
    cu_t = K.i64 if cu_dtype == "int64" else K.i32
    protocol = _CHECKPOINT_PROTOCOL if checkpoints else _NORMAL_PROTOCOL
    arena_bytes = 211_968 if checkpoints else 203_776
    raw_stages = 3 if checkpoints else 5
    raw_ready_stages = 4 if checkpoints else 6
    checkpoint_stages = 2 if checkpoints else 1
    tmem_mailbox = 784 if checkpoints else 960
    sched_base = 800 if checkpoints else 976
    k_decay_base = 1_024
    q_decay_base = 9_216
    k_restore_base = 17_408
    intermediate_base = 25_600
    q_raw_base = 27_648
    k_raw_base = 39_936 if checkpoints else 48_128
    v_raw_base = 52_224 if checkpoints else 68_608
    gate_raw_base = 64_512 if checkpoints else 89_088
    diag_base = 89_088 if checkpoints else 130_048
    k_inv_base = 105_472 if checkpoints else 146_432
    o_base = 113_664 if checkpoints else 154_624
    beta_raw_base = 121_856 if checkpoints else 162_816
    w_raw_base = 134_144 if checkpoints else 183_296
    checkpoint_base = 146_432
    idesc_ts = 0x08040010 if io_dtype == "float16" else 0x08040490
    idesc_final = 0x08210010 if io_dtype == "float16" else 0x08210490

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=num_sms)
    def main(
        descriptor_workspace: K.gptr[K.i64],
        n_desc: K.i32,
        q: K.gptr[io_t],
        k: K.gptr[io_t],
        v: K.gptr[io_t],
        gate: K.gptr[K.f32],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        beta: K.gptr[io_t],
        w: K.gptr[io_t],
        cu_seqlens: K.gptr[cu_t],
        initial_state: K.gptr[state_t],
        o: K.gptr[io_t],
        final_state: K.gptr[state_t],
        work_items: K.gptr[K.i32],
        work_count: K.gptr[K.i32],
        scheduler: K.gptr[K.i32],
        scale: K.f32,
        checkpoint_every_n: K.i32,
    ):
        arena = K.alloc_buffer((arena_bytes,), K.u8, scope="shared.dyn", align=1024)
        K.smem_pool(base=arena)
        thread = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()

        # Declaration-ordered physical mbarrier table.  The normal source
        # compiles the checkpoint and state-read edges away entirely.
        for edge_index, edge in enumerate(protocol):
            ready_off, done_off, ready_stages, done_stages, ready_count, done_count, owner = edge
            if checkpoints or edge_index not in (20, 22):
                with K.If(warp == owner), K.Then():
                    with K.If(_elected()), K.Then():
                        for stage in range(ready_stages):
                            K.ptx.mbarrier.init.shared.b64(
                                _barrier_ptr(arena, ready_off, stage), K.uint32(ready_count)
                            )
                        if done_off is not None:
                            for stage in range(done_stages):
                                K.ptx.mbarrier.init.shared.b64(
                                    _barrier_ptr(arena, done_off, stage), K.uint32(done_count)
                                )

        # Four diagonal stages, eight 16x16 blocks per stage.
        with K.unroll(16) as diagonal_pass:
            element = thread + diagonal_pass * 512
            K.ptx.st.shared.b16(arena.ptr_to([diag_base + element * 2]), K.uint16(0))
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        total_tiles = K.local_scalar("int32")
        K.ptx.ld.global_.s32(total_tiles, work_count.ptr_to([0]))
        roles = K.specialize()
        cg0 = roles.role("cg0", warps=range(0, 8), regs=160)
        cg1 = roles.role("cg1", warps=range(8, 12), regs=136)
        super_mma = roles.role("super_mma", warps=[12], regs=56)
        tcgen = roles.role("tcgen", warps=[13], regs=56)
        tma = roles.role("tma", warps=[14], regs=56)
        epilogue = roles.role("epilogue", warps=[15], regs=56)

        # Warp 14: descriptor acquire, six input TMA rings, scheduler producer.
        with tma:
            raw = K.PipelineState(raw_stages, phase=1)
            raw_ready = K.PipelineState(raw_ready_stages, phase=0)
            sched_producer = K.PipelineState(8, phase=1)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                cstart = item[4]
                wend = item[3]
                head_q = head if q_ratio == 1 else head // q_ratio
                head_k = head if k_ratio == 1 else head // k_ratio
                head_v = head if v_ratio == 1 else head // v_ratio
                desc_q = _descriptor_slot(descriptor_workspace, n_desc, 0, batch)
                desc_k = _descriptor_slot(descriptor_workspace, n_desc, 1, batch)
                desc_v = _descriptor_slot(descriptor_workspace, n_desc, 2, batch)
                desc_gate = _descriptor_slot(descriptor_workspace, n_desc, 3, batch)
                desc_beta = _descriptor_slot(descriptor_workspace, n_desc, 4, batch)
                desc_w = _descriptor_slot(descriptor_workspace, n_desc, 5, batch)
                with K.If(_elected()), K.Then():
                    for desc in (desc_q, desc_k, desc_v, desc_gate, desc_beta, desc_w):
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc)

                with K.serial(wend - cstart, unroll=False) as local_chunk:
                    chunk = cstart + local_chunk
                    token = chunk * _BT
                    raw_stage = raw.stage
                    ready_stage = raw_ready.stage
                    with K.If(_elected()), K.Then():
                        _wait_barrier(arena, protocol[0][1], raw_stage, raw.phase)
                        _expect_tx(arena, protocol[0][0], ready_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to([q_raw_base + raw_stage * 4096 + d_coord * 32]),
                                desc_q,
                                K.int32(d_coord),
                                head_q,
                                token,
                                _barrier_ptr(arena, protocol[0][0], ready_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, protocol[1][1], raw_stage, raw.phase)
                        _expect_tx(arena, protocol[1][0], ready_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to([k_raw_base + raw_stage * 4096 + d_coord * 32]),
                                desc_k,
                                K.int32(d_coord),
                                head_k,
                                token,
                                _barrier_ptr(arena, protocol[1][0], ready_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, protocol[2][1], raw_stage, raw.phase)
                        _expect_tx(arena, protocol[2][0], ready_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to([v_raw_base + raw_stage * 4096 + d_coord * 32]),
                                desc_v,
                                K.int32(d_coord),
                                head_v,
                                token,
                                _barrier_ptr(arena, protocol[2][0], ready_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, protocol[5][1], raw_stage, raw.phase)
                        _expect_tx(arena, protocol[5][0], ready_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to([beta_raw_base + raw_stage * 4096 + d_coord * 32]),
                                desc_beta,
                                K.int32(d_coord),
                                head,
                                token,
                                _barrier_ptr(arena, protocol[5][0], ready_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, protocol[3][1], raw_stage, raw.phase)
                        _expect_tx(arena, protocol[3][0], ready_stage, 4096)
                        for d_coord in (0, 64):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to([w_raw_base + raw_stage * 4096 + d_coord * 32]),
                                desc_w,
                                K.int32(d_coord),
                                head,
                                token,
                                _barrier_ptr(arena, protocol[3][0], ready_stage),
                                K.uint64(0),
                            )
                        _wait_barrier(arena, protocol[4][1], raw_stage, raw.phase)
                        _expect_tx(arena, protocol[4][0], ready_stage, 8192)
                        for d_coord in (0, 32, 64, 96):
                            K.ptx[_TMA_G2S_3D](
                                arena.ptr_to([gate_raw_base + raw_stage * 8192 + d_coord * 64]),
                                desc_gate,
                                K.int32(d_coord),
                                head,
                                token,
                                _barrier_ptr(arena, protocol[4][0], ready_stage),
                                K.uint64(0),
                            )
                    raw.advance()
                    raw_ready.advance()

                if dynamic_scheduler:
                    _wait_barrier(
                        arena, protocol[23][1], sched_producer.stage, sched_producer.phase
                    )
                    with K.If(_elected()), K.Then():
                        ticket = K.local_scalar("uint32")
                        K.ptx.atom.global_.add.u32(ticket, scheduler.ptr_to([0]), K.uint32(1))
                        K.ptx.st.shared.u32(
                            arena.ptr_to([sched_base + sched_producer.stage * 4]),
                            K.uint32(num_sms) + ticket,
                        )
                    K.cuda.warp_sync()
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([sched_base + sched_producer.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, protocol[23][0], sched_producer.stage)
                    sched_producer.advance()
                else:
                    K.assign(tile, tile + num_sms)

        # Warp 12: register KK, strict-lower L, and three Neumann rounds.
        with super_mma:
            rhs_row = lane % 8 + K.if_then_else(lane // 16 != 0, 8, 0)
            rhs_col = K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_row = lane % 8 + K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_row = (lane & 7) + K.if_then_else((lane // 8) & 1 != 0, 8, 0)
            store_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_linear = store_row * 16 + (store_col ^ 8)
            store_swizzled = K.bitwise_xor(
                store_linear,
                K.shift_left(
                    K.bitwise_and(K.shift_right(store_linear, K.uint32(6)), K.uint32(1)),
                    K.uint32(3),
                ),
            )
            row_lo = lane // 4
            row_hi = row_lo + 8
            serial_base = K.local_scalar("int32", init=K.int32(0))
            sched_consumer = K.PipelineState(8, phase=0)
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                num_chunks = item[3] - item[4]
                with K.serial(num_chunks, unroll=False) as local_chunk:
                    serial = serial_base + local_chunk
                    decay_stage = serial % 2
                    inter_stage = serial % 2
                    _wait_barrier(arena, protocol[14][0], decay_stage, (serial // 2) & 1)

                    kk = K.alloc_local((8,), "float32")
                    for accum in range(8):
                        K.assign(kk[accum], K.float32(0.0))
                    with K.unroll(8) as key_block:
                        a = K.alloc_local((4,), "uint32")
                        b = K.alloc_local((4,), "uint32")
                        inv_channel = key_block * 16 + rhs_col
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            b[0],
                            b[1],
                            b[2],
                            b[3],
                            arena.ptr_to(
                                [_raw_io_byte(k_inv_base, decay_stage, rhs_row, inv_channel)]
                            ),
                        )
                        storage_key = K.bitwise_xor(key_block * 16 + lhs_col, K.int32(8))
                        storage_slice = storage_key // 64
                        decay_element = storage_slice * 1024 + _swizzle_xor_128b(
                            lhs_row, lhs_row * 64 + storage_key - storage_slice * 64, 2
                        )
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a[0],
                            a[1],
                            a[2],
                            a[3],
                            arena.ptr_to([k_decay_base + decay_stage * 4096 + decay_element * 2]),
                        )
                        _mma_m16n16k16(kk, a, b, io_dtype)

                    l_words = K.alloc_local((4,), "uint32")
                    rounded_l = K.alloc_local((8,), "float32")
                    tinv = K.alloc_local((8,), "float32")
                    for pair in range(4):
                        idx0 = pair * 2
                        idx1 = idx0 + 1
                        row0 = row_hi if idx0 % 4 >= 2 else row_lo
                        row1 = row_hi if idx1 % 4 >= 2 else row_lo
                        col0 = (idx0 // 4) * 8 + 2 * (lane % 4) + (idx0 & 1)
                        col1 = (idx1 // 4) * 8 + 2 * (lane % 4) + (idx1 & 1)
                        l0 = K.if_then_else(row0 > col0, kk[idx0], K.float32(0.0))
                        l1 = K.if_then_else(row1 > col1, kk[idx1], K.float32(0.0))
                        K.assign(l_words[pair], _pack_io_pair(l0, l1, io_dtype))
                        _unpack_io_pair(l_words[pair], rounded_l[idx0], rounded_l[idx1], io_dtype)
                        K.assign(
                            tinv[idx0],
                            K.if_then_else(row0 == col0, K.float32(1.0), K.float32(0.0))
                            - rounded_l[idx0],
                        )
                        K.assign(
                            tinv[idx1],
                            K.if_then_else(row1 == col1, K.float32(1.0), K.float32(0.0))
                            - rounded_l[idx1],
                        )

                    l_power = K.alloc_local((4,), "uint32")
                    l_power_t = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        K.assign(l_power[pair], l_words[pair])
                        K.assign(l_power_t[pair], _movmatrix_b16(l_power[pair]))
                    for _ in range(3):
                        square = K.alloc_local((8,), "float32")
                        for accum in range(8):
                            K.assign(square[accum], K.float32(0.0))
                        _mma_m16n16k16(square, l_power, l_power_t, io_dtype)
                        for pair in range(4):
                            K.assign(
                                l_power[pair],
                                _pack_io_pair(square[pair * 2], square[pair * 2 + 1], io_dtype),
                            )
                            K.assign(l_power_t[pair], _movmatrix_b16(l_power[pair]))

                        tinv_words = K.alloc_local((4,), "uint32")
                        for pair in range(4):
                            K.assign(
                                tinv_words[pair],
                                _pack_io_pair(tinv[pair * 2], tinv[pair * 2 + 1], io_dtype),
                            )
                        update = K.alloc_local((8,), "float32")
                        for accum in range(8):
                            K.assign(update[accum], K.float32(0.0))
                        _mma_m16n16k16(update, tinv_words, l_power_t, io_dtype)
                        for pair in range(4):
                            old_lo = K.local_scalar("float32")
                            old_hi = K.local_scalar("float32")
                            _unpack_io_pair(tinv_words[pair], old_lo, old_hi, io_dtype)
                            next_lo, next_hi = _fadd2(
                                old_lo, old_hi, update[pair * 2], update[pair * 2 + 1]
                            )
                            K.assign(tinv[pair * 2], next_lo)
                            K.assign(tinv[pair * 2 + 1], next_hi)

                    _wait_barrier(arena, protocol[12][1], inter_stage, (serial // 2 + 1) & 1)
                    tinv_out = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        K.assign(
                            tinv_out[pair],
                            _pack_io_pair(tinv[pair * 2], tinv[pair * 2 + 1], io_dtype),
                        )
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to(
                            [intermediate_base + inter_stage * 1024 + 512 + store_swizzled * 2]
                        ),
                        tinv_out[0],
                        tinv_out[1],
                        tinv_out[2],
                        tinv_out[3],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, protocol[12][0], inter_stage)
                    _arrive_barrier(arena, protocol[15][0], decay_stage)

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, protocol[23][0], sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([sched_base + sched_consumer.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, protocol[23][1], sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)

        # Warp 13: TMEM lifecycle and the six source-ordered tcgen05 chains.
        with tcgen:
            tcgen_lane = K.local_scalar("uint32")
            tcgen_leader = K.local_scalar("uint32")
            K.ptx.elect_sync(tcgen_lane, tcgen_leader, K.uint32(0xFFFFFFFF))
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                arena.ptr_to([tmem_mailbox]), K.uint32(512)
            )
            K.ptx.bar.sync(K.uint32(3), K.uint32(160))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([tmem_mailbox]))
            state_inp = K.PipelineState(1, phase=0)
            state_read = K.PipelineState(1, phase=0)
            y_inp = K.PipelineState(1, phase=0)
            u_inp = K.PipelineState(1, phase=0)
            qk_scale = K.PipelineState(4, phase=0)
            sched_consumer = K.PipelineState(8, phase=0)
            serial_base = K.local_scalar("int32", init=K.int32(0))
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                num_chunks = item[3] - item[4]
                with K.serial(num_chunks, unroll=False) as local_chunk:
                    serial = serial_base + local_chunk
                    have_state = K.bool(True) if use_initial_state else local_chunk > 0
                    decay_stage = serial % 2
                    qstate_stage = serial % 2
                    inter_stage = serial % 2
                    diag_stage = qk_scale.stage
                    k_decay_desc = _raw_descriptor(
                        arena, k_decay_base + decay_stage * 4096, 16, 1024, 2
                    )
                    q_decay_desc = _raw_descriptor(
                        arena, q_decay_base + decay_stage * 4096, 16, 1024, 2
                    )
                    k_restore_desc = _raw_descriptor(
                        arena, k_restore_base + decay_stage * 4096, 2048, 1024, 2
                    )
                    tinv_desc = _raw_descriptor(
                        arena, intermediate_base + inter_stage * 1024 + 512, 16, 256, 6
                    )
                    a_desc = _raw_descriptor(
                        arena, intermediate_base + inter_stage * 1024, 16, 256, 6
                    )

                    _wait_barrier(arena, protocol[14][0], decay_stage, (serial // 2) & 1)
                    with K.If(have_state), K.Then():
                        _wait_barrier(arena, protocol[9][0], state_inp.stage, state_inp.phase)
                        state_inp.advance()
                        _tcgen_mma_ts(
                            tmem_base + 224,
                            tmem_base + 128,
                            k_decay_desc,
                            idesc_ts,
                            n=16,
                            k_extent=128,
                            leader=tcgen_leader,
                        )
                        _tcgen_commit(arena, protocol[7][0], leader=tcgen_leader)

                    _wait_barrier(arena, protocol[16][0], diag_stage, qk_scale.phase)
                    _wait_barrier(arena, protocol[6][1], qstate_stage, (serial // 2 + 1) & 1)
                    with K.If(have_state), K.Then():
                        _tcgen_mma_ts(
                            tmem_base + 192 + qstate_stage * 16,
                            tmem_base + 128,
                            q_decay_desc,
                            idesc_ts,
                            n=16,
                            k_extent=128,
                            leader=tcgen_leader,
                        )
                    _tcgen_commit(arena, protocol[14][1], decay_stage, leader=tcgen_leader)

                    if checkpoints:
                        with K.If(have_state), K.Then():
                            _wait_barrier(
                                arena, protocol[20][0], state_read.stage, state_read.phase
                            )
                            state_read.advance()
                    with K.If(have_state), K.Then():
                        for key_block in range(8):
                            diag_desc = _raw_descriptor(
                                arena, diag_base + diag_stage * 4096 + key_block * 512, 16, 256, 6
                            )
                            _tcgen_mma_ts(
                                tmem_base + key_block * 16,
                                tmem_base + 128 + key_block * 8,
                                diag_desc,
                                idesc_ts,
                                n=16,
                                k_extent=16,
                                leader=tcgen_leader,
                            )
                    _tcgen_commit(arena, protocol[18][0], diag_stage, leader=tcgen_leader)

                    _wait_barrier(arena, protocol[12][0], inter_stage, (serial // 2) & 1)
                    _wait_barrier(arena, protocol[10][0], y_inp.stage, y_inp.phase)
                    y_inp.advance()
                    _tcgen_mma_ts(
                        tmem_base + 240,
                        tmem_base + 256,
                        tinv_desc,
                        idesc_ts,
                        n=16,
                        k_extent=16,
                        leader=tcgen_leader,
                    )
                    _tcgen_commit(arena, protocol[8][0], leader=tcgen_leader)

                    _wait_barrier(arena, protocol[11][0], u_inp.stage, u_inp.phase)
                    u_inp.advance()
                    _tcgen_mma_ts(
                        tmem_base,
                        tmem_base + 264,
                        k_restore_desc,
                        idesc_final,
                        n=128,
                        k_extent=16,
                        leader=tcgen_leader,
                        b_transpose=True,
                        accumulate=have_state,
                    )
                    _tcgen_commit(arena, protocol[17][0], decay_stage, leader=tcgen_leader)

                    _wait_barrier(arena, protocol[13][0], inter_stage, (serial // 2) & 1)
                    _tcgen_mma_ts(
                        tmem_base + 192 + qstate_stage * 16,
                        tmem_base + 264,
                        a_desc,
                        idesc_ts,
                        n=16,
                        k_extent=16,
                        leader=tcgen_leader,
                        accumulate=have_state,
                    )
                    _tcgen_commit(arena, protocol[6][0], leader=tcgen_leader)
                    _tcgen_commit(arena, protocol[12][1], inter_stage, leader=tcgen_leader)
                    qk_scale.advance()

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, protocol[23][0], sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([sched_base + sched_consumer.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, protocol[23][1], sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
            _wait_barrier(arena, protocol[19][0], 0, 0)
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                K.cast(tmem_base, "uint32"), K.uint32(512)
            )

        # Warp 15: causal register MMA and one-behind checkpoint/O TMA drain.
        with epilogue:
            rhs_row = lane % 8 + K.if_then_else(lane // 16 != 0, 8, 0)
            rhs_col = K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_row = lane % 8 + K.if_then_else((lane // 8) % 2 != 0, 8, 0)
            lhs_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_row = (lane & 7) + K.if_then_else((lane // 8) & 1 != 0, 8, 0)
            store_col = K.if_then_else(lane // 8 >= 2, 8, 0)
            store_linear = store_row * 16 + (store_col ^ 8)
            store_swizzled = K.bitwise_xor(
                store_linear,
                K.shift_left(
                    K.bitwise_and(K.shift_right(store_linear, K.uint32(6)), K.uint32(1)),
                    K.uint32(3),
                ),
            )
            row_lo = lane // 4
            row_hi = row_lo + 8
            qk_scale = K.PipelineState(4, phase=0)
            checkpoint_ready = K.PipelineState(checkpoint_stages, phase=0)
            sched_consumer = K.PipelineState(8, phase=0)
            serial_base = K.local_scalar("int32", init=K.int32(0))
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                batch = item[0]
                head = item[1]
                wstart = item[2]
                wend = item[3]
                cstart = item[4]
                num_chunks = wend - cstart
                desc_o = _descriptor_slot(descriptor_workspace, n_desc, 6, batch)
                with K.If(_elected()), K.Then():
                    K.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc_o)
                checkpoint_chunks = K.local_scalar("int32", init=K.int32(1))
                checkpoint_quot = K.local_scalar("int32", init=K.int32(0))
                checkpoint_mod = K.local_scalar("int32", init=K.int32(0))
                desc_checkpoint = _descriptor_slot(descriptor_workspace, n_desc, 7, batch)
                if checkpoints:
                    K.assign(checkpoint_chunks, checkpoint_every_n // _BT)
                    K.assign(checkpoint_quot, (cstart + 1) // checkpoint_chunks)
                    K.assign(checkpoint_mod, (cstart + 1) % checkpoint_chunks)
                    with K.If(_elected()), K.Then():
                        K.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc_checkpoint)
                    with K.If(K.And(num_chunks > 0, wstart == 0)), K.Then():
                        cp_stage = K.local_scalar("int32")
                        K.ptx.mov.b32(cp_stage, checkpoint_ready.stage)
                        _wait_barrier(arena, protocol[22][0], cp_stage, checkpoint_ready.phase)
                        checkpoint_ready.advance()
                        for value_coord in (0, 64):
                            K.ptx[_TMA_S2G_4D](
                                desc_checkpoint,
                                K.int32(value_coord),
                                K.int32(0),
                                K.int32(0),
                                head,
                                arena.ptr_to(
                                    [checkpoint_base + cp_stage * 32768 + value_coord * 256]
                                ),
                            )
                        K.ptx.cp.async_.bulk.commit_group()
                        K.ptx.cp.async_.bulk.wait_group.read(0)
                        _arrive_barrier(arena, protocol[22][1], cp_stage)

                with K.serial(num_chunks, unroll=False) as local_chunk:
                    chunk = cstart + local_chunk
                    serial = serial_base + local_chunk
                    decay_stage = serial % 2
                    inter_stage = serial % 2
                    diag_stage = qk_scale.stage
                    _wait_barrier(arena, protocol[16][0], diag_stage, qk_scale.phase)
                    a_acc = K.alloc_local((8,), "float32")
                    for accum in range(8):
                        K.assign(a_acc[accum], K.float32(0.0))
                    with K.unroll(8) as key_block:
                        a_frag = K.alloc_local((4,), "uint32")
                        b_frag = K.alloc_local((4,), "uint32")
                        inv_channel = key_block * 16 + rhs_col
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            b_frag[0],
                            b_frag[1],
                            b_frag[2],
                            b_frag[3],
                            arena.ptr_to(
                                [_raw_io_byte(k_inv_base, decay_stage, rhs_row, inv_channel)]
                            ),
                        )
                        storage_key = K.bitwise_xor(key_block * 16 + lhs_col, K.int32(8))
                        storage_slice = storage_key // 64
                        decay_element = storage_slice * 1024 + _swizzle_xor_128b(
                            lhs_row, lhs_row * 64 + storage_key - storage_slice * 64, 2
                        )
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a_frag[0],
                            a_frag[1],
                            a_frag[2],
                            a_frag[3],
                            arena.ptr_to([q_decay_base + decay_stage * 4096 + decay_element * 2]),
                        )
                        _mma_m16n16k16(a_acc, a_frag, b_frag, io_dtype)
                    a_words = K.alloc_local((4,), "uint32")
                    for pair in range(4):
                        for parity in range(2):
                            accum = pair * 2 + parity
                            row_coord = row_hi if accum % 4 >= 2 else row_lo
                            col_coord = (accum // 4) * 8 + 2 * (lane % 4) + (accum & 1)
                            K.assign(
                                a_acc[accum],
                                K.if_then_else(
                                    row_coord >= col_coord, a_acc[accum], K.float32(0.0)
                                ),
                            )
                        K.assign(
                            a_words[pair],
                            _pack_io_pair(a_acc[pair * 2], a_acc[pair * 2 + 1], io_dtype),
                        )
                    _wait_barrier(arena, protocol[12][1], inter_stage, (serial // 2 + 1) & 1)
                    K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        arena.ptr_to([intermediate_base + inter_stage * 1024 + store_swizzled * 2]),
                        a_words[0],
                        a_words[1],
                        a_words[2],
                        a_words[3],
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, protocol[13][0], inter_stage)
                    _arrive_barrier(arena, protocol[15][0], decay_stage)
                    qk_scale.advance()

                    with K.If(local_chunk > 0), K.Then():
                        output_chunk = chunk - 1
                        previous_serial = serial - 1
                        o_stage = previous_serial % 2
                        did_checkpoint = K.local_scalar("int32", init=K.int32(0))
                        cp_stage = K.local_scalar("int32", init=K.int32(0))
                        if checkpoints:
                            do_checkpoint = K.And(checkpoint_mod == 0, chunk >= wstart)
                            K.ptx.mov.b32(cp_stage, checkpoint_ready.stage)
                            with K.If(do_checkpoint), K.Then():
                                _wait_barrier(
                                    arena,
                                    protocol[22][0],
                                    checkpoint_ready.stage,
                                    checkpoint_ready.phase,
                                )
                                checkpoint_ready.advance()
                                for value_coord in (0, 64):
                                    K.ptx[_TMA_S2G_4D](
                                        desc_checkpoint,
                                        K.int32(value_coord),
                                        K.int32(0),
                                        checkpoint_quot,
                                        head,
                                        arena.ptr_to(
                                            [checkpoint_base + cp_stage * 32768 + value_coord * 256]
                                        ),
                                    )
                                K.ptx.cp.async_.bulk.commit_group()
                                K.assign(did_checkpoint, K.int32(1))
                            K.assign(checkpoint_mod, checkpoint_mod + 1)
                            with K.If(checkpoint_mod == checkpoint_chunks), K.Then():
                                K.assign(checkpoint_mod, K.int32(0))
                                K.assign(checkpoint_quot, checkpoint_quot + 1)

                        _wait_barrier(arena, protocol[21][0], o_stage, (previous_serial // 2) & 1)
                        did_o = K.local_scalar("int32", init=K.int32(0))
                        with K.If(output_chunk >= wstart), K.Then():
                            for value_coord in (0, 64):
                                K.ptx[_TMA_S2G_3D](
                                    desc_o,
                                    K.int32(value_coord),
                                    head,
                                    output_chunk * _BT,
                                    arena.ptr_to([o_base + o_stage * 4096 + value_coord * 32]),
                                )
                            K.ptx.cp.async_.bulk.commit_group()
                            K.assign(did_o, K.int32(1))
                        if checkpoints:
                            with K.If(K.And(did_checkpoint == 1, did_o == 1)), K.Then():
                                K.ptx.cp.async_.bulk.wait_group.read(1)
                                _arrive_barrier(arena, protocol[22][1], cp_stage)
                                K.ptx.cp.async_.bulk.wait_group.read(0)
                                _arrive_barrier(arena, protocol[21][1], o_stage)
                            with K.If(K.And(did_checkpoint == 1, did_o == 0)), K.Then():
                                K.ptx.cp.async_.bulk.wait_group.read(0)
                                _arrive_barrier(arena, protocol[22][1], cp_stage)
                                _arrive_barrier(arena, protocol[21][1], o_stage)
                            with K.If(did_checkpoint == 0), K.Then():
                                with K.If(did_o == 1), K.Then():
                                    K.ptx.cp.async_.bulk.wait_group.read(0)
                                _arrive_barrier(arena, protocol[21][1], o_stage)
                        else:
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                            _arrive_barrier(arena, protocol[21][1], o_stage)

                with K.If(num_chunks > 0), K.Then():
                    last_chunk = wend - 1
                    last_serial = serial_base + num_chunks - 1
                    o_stage = last_serial % 2
                    _wait_barrier(arena, protocol[21][0], o_stage, (last_serial // 2) & 1)
                    for value_coord in (0, 64):
                        K.ptx[_TMA_S2G_3D](
                            desc_o,
                            K.int32(value_coord),
                            head,
                            last_chunk * _BT,
                            arena.ptr_to([o_base + o_stage * 4096 + value_coord * 32]),
                        )
                    K.ptx.cp.async_.bulk.commit_group()
                    K.ptx.cp.async_.bulk.wait_group.read(0)
                    _arrive_barrier(arena, protocol[21][1], o_stage)

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, protocol[23][0], sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([sched_base + sched_consumer.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, protocol[23][1], sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)

        # Warps 0..7: two four-warp ping-pong CG0 groups.
        with cg0:
            cg0_warp = K.warp_id_in_role()
            group = cg0_warp // 4
            local_warp = cg0_warp % 4
            prefix_dim = local_warp * 32 + lane
            decay_row = local_warp * 4 + lane // 8
            lane_in_row = lane % 8
            sched_consumer = K.PipelineState(8, phase=0)
            serial_base = K.local_scalar("int32", init=K.int32(0))
            tile = K.local_scalar("int32", init=K.cta_id())
            with K.While(tile < total_tiles):
                item = _load_work_item(work_items, tile)
                head = item[1]
                cstart = item[4]
                wend = item[3]
                bos = item[6]
                eos = item[7]
                sequence_length = eos - bos
                num_chunks = wend - cstart
                a_log_exp = K.local_scalar("float32", init=K.float32(1.0))
                dt_value = K.local_scalar("float32", init=K.float32(0.0))
                if safe_gate:
                    with K.If(num_chunks > 0), K.Then():
                        a_value = K.local_scalar("float32")
                        K.ptx.ld.global_.f32(a_value, a_log.ptr_to([head]))
                        K.assign(a_log_exp, _exp2(a_value * K.float32(_LOG2_E)))
                        K.ptx.ld.global_.f32(
                            dt_value, dt_bias.ptr_to([K.cast(head, "int64") * 128 + prefix_dim])
                        )

                K.ptx.bar.sync(K.uint32(5), K.uint32(256))
                local_chunk = K.local_scalar("int32", init=group)
                with K.While(local_chunk < num_chunks):
                    chunk = cstart + local_chunk
                    serial = serial_base + local_chunk
                    raw_stage = serial % raw_stages
                    ready_stage = serial % raw_ready_stages
                    ready_phase = (serial // raw_ready_stages) & 1
                    decay_stage = serial % 2
                    diag_stage = serial % 4
                    _wait_barrier(arena, protocol[4][0], ready_stage, ready_phase)

                    gate_prefix = K.alloc_local((16,), "float32")
                    for row in range(16):
                        raw_gate = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            raw_gate,
                            arena.ptr_to(
                                [_raw_f32_byte(gate_raw_base, raw_stage, row, prefix_dim)]
                            ),
                        )
                        if safe_gate:
                            K.assign(raw_gate, a_log_exp * (raw_gate + dt_value))
                            sigmoid = _tanh(raw_gate * K.float32(0.5)) * K.float32(0.5)
                            K.assign(
                                raw_gate, (sigmoid + K.float32(0.5)) * K.float32(gate_scale_log2)
                            )
                            if not full_tiles:
                                with K.If(chunk * 16 + row >= sequence_length), K.Then():
                                    K.assign(raw_gate, K.float32(0.0))
                        else:
                            K.assign(raw_gate, raw_gate * K.float32(_LOG2_E))
                            if not full_tiles:
                                with K.If(chunk * 16 + row >= sequence_length), K.Then():
                                    K.assign(raw_gate, K.float32(0.0))
                        K.assign(gate_prefix[row], raw_gate)

                    prefix_acc = K.local_scalar("float32", init=K.float32(0.0))
                    for pair in range(8):
                        row0 = pair * 2
                        row1 = row0 + 1
                        prefix0, pair_sum = _fadd2(
                            prefix_acc, gate_prefix[row0], gate_prefix[row0], gate_prefix[row1]
                        )
                        prefix1 = prefix_acc + pair_sum
                        K.assign(gate_prefix[row0], prefix0)
                        K.assign(gate_prefix[row1], prefix1)
                        K.assign(prefix_acc, prefix1)
                    for row in range(16):
                        K.assign(gate_prefix[row], _exp2(gate_prefix[row]))
                        K.ptx.st.shared.f32(
                            arena.ptr_to(
                                [_raw_f32_byte(gate_raw_base, raw_stage, row, prefix_dim)]
                            ),
                            gate_prefix[row],
                        )
                    exp_last = gate_prefix[15]

                    _wait_barrier(arena, protocol[18][0], diag_stage, (serial // 4 + 1) & 1)
                    block = prefix_dim // 16
                    coord = prefix_dim - block * 16
                    storage_col = coord ^ 8
                    diagonal_linear = block * 256 + coord * 16 + storage_col
                    diagonal_swizzled = K.bitwise_xor(
                        diagonal_linear,
                        K.shift_left(
                            K.bitwise_and(K.shift_right(diagonal_linear, K.uint32(6)), K.uint32(1)),
                            K.uint32(3),
                        ),
                    )
                    K.ptx.st.shared.b16(
                        arena.ptr_to([diag_base + diag_stage * 4096 + diagonal_swizzled * 2]),
                        K.reinterpret("uint16", K.cast(exp_last, io_dtype)),
                    )
                    K.ptx.bar.sync(K.cast(1 + group, "uint32"), K.uint32(128))

                    _wait_barrier(arena, protocol[0][0], ready_stage, ready_phase)
                    _wait_barrier(arena, protocol[1][0], ready_stage, ready_phase)
                    _wait_barrier(arena, protocol[5][0], ready_stage, ready_phase)
                    raw_q = K.alloc_local((16,), "float32")
                    raw_k = K.alloc_local((16,), "float32")
                    raw_beta = K.alloc_local((16,), "float32")
                    if l2norm:
                        qk0_lo = K.local_scalar("float32", init=K.float32(0.0))
                        qk0_hi = K.local_scalar("float32", init=K.float32(0.0))
                        qk1_lo = K.local_scalar("float32", init=K.float32(0.0))
                        qk1_hi = K.local_scalar("float32", init=K.float32(0.0))
                    for dim_half in range(2):
                        dim_base = dim_half * 64 + lane_in_row * 8
                        q_words = K.alloc_local((4,), "uint32")
                        k_words = K.alloc_local((4,), "uint32")
                        beta_words = K.alloc_local((4,), "uint32")
                        K.ptx.ld.shared.v4.b32(
                            q_words[0],
                            q_words[1],
                            q_words[2],
                            q_words[3],
                            arena.ptr_to(
                                [_raw_io_byte(q_raw_base, raw_stage, decay_row, dim_base)]
                            ),
                        )
                        K.ptx.ld.shared.v4.b32(
                            k_words[0],
                            k_words[1],
                            k_words[2],
                            k_words[3],
                            arena.ptr_to(
                                [_raw_io_byte(k_raw_base, raw_stage, decay_row, dim_base)]
                            ),
                        )
                        K.ptx.ld.shared.v4.b32(
                            beta_words[0],
                            beta_words[1],
                            beta_words[2],
                            beta_words[3],
                            arena.ptr_to(
                                [_raw_io_byte(beta_raw_base, raw_stage, decay_row, dim_base)]
                            ),
                        )
                        for pair in range(4):
                            reg = dim_half * 8 + pair * 2
                            _unpack_io_pair(q_words[pair], raw_q[reg], raw_q[reg + 1], io_dtype)
                            _unpack_io_pair(k_words[pair], raw_k[reg], raw_k[reg + 1], io_dtype)
                            _unpack_io_pair(
                                beta_words[pair], raw_beta[reg], raw_beta[reg + 1], io_dtype
                            )
                            if beta_sigmoid:
                                for parity in range(2):
                                    beta_value = _tanh(
                                        raw_beta[reg + parity] * K.float32(0.5)
                                    ) * K.float32(0.5) + K.float32(0.5)
                                    rounded = _pack_io_pair(beta_value, beta_value, io_dtype)
                                    rounded_lo = K.local_scalar("float32")
                                    rounded_hi = K.local_scalar("float32")
                                    _unpack_io_pair(rounded, rounded_lo, rounded_hi, io_dtype)
                                    K.assign(raw_beta[reg + parity], rounded_lo)
                            if l2norm:
                                if pair % 2 == 0:
                                    q_lo, k_lo = _ffma2(
                                        raw_q[reg],
                                        raw_k[reg],
                                        raw_q[reg],
                                        raw_k[reg],
                                        qk0_lo,
                                        qk0_hi,
                                    )
                                    q_hi, k_hi = _ffma2(
                                        raw_q[reg + 1],
                                        raw_k[reg + 1],
                                        raw_q[reg + 1],
                                        raw_k[reg + 1],
                                        qk1_lo,
                                        qk1_hi,
                                    )
                                    K.assign(qk0_lo, q_lo)
                                    K.assign(qk0_hi, k_lo)
                                    K.assign(qk1_lo, q_hi)
                                    K.assign(qk1_hi, k_hi)
                                else:
                                    q_lo, k_lo = _ffma2(
                                        raw_q[reg],
                                        raw_k[reg],
                                        raw_q[reg],
                                        raw_k[reg],
                                        qk0_lo,
                                        qk0_hi,
                                    )
                                    q_hi, k_hi = _ffma2(
                                        raw_q[reg + 1],
                                        raw_k[reg + 1],
                                        raw_q[reg + 1],
                                        raw_k[reg + 1],
                                        qk1_lo,
                                        qk1_hi,
                                    )
                                    K.assign(qk0_lo, q_lo)
                                    K.assign(qk0_hi, k_lo)
                                    K.assign(qk1_lo, q_hi)
                                    K.assign(qk1_hi, k_hi)

                    q_inv_norm = K.local_scalar("float32", init=_opaque_one())
                    k_inv_norm = K.local_scalar("float32", init=_opaque_one())
                    if l2norm:
                        q_sum = K.local_scalar("float32", init=qk0_lo + qk1_lo)
                        k_sum = K.local_scalar("float32", init=qk0_hi + qk1_hi)
                        for delta in (4, 2, 1):
                            K.assign(q_sum, q_sum + _shuffle_xor_f32(q_sum, delta))
                            K.assign(k_sum, k_sum + _shuffle_xor_f32(k_sum, delta))
                        q_floor = K.if_then_else(
                            q_sum > K.float32(_L2_NORM_EPS2), q_sum, K.float32(_L2_NORM_EPS2)
                        )
                        k_floor = K.if_then_else(
                            k_sum > K.float32(_L2_NORM_EPS2), k_sum, K.float32(_L2_NORM_EPS2)
                        )
                        K.assign(q_inv_norm, _rsqrt(q_floor))
                        K.assign(k_inv_norm, _rsqrt(k_floor))

                    exp_g = K.alloc_local((16,), "float32")
                    exp_g_last = K.alloc_local((16,), "float32")
                    if not checkpoints:
                        exp_neg_g = K.alloc_local((16,), "float32")
                    k_inv_words = K.alloc_local((8,), "uint32")
                    for dim_half in range(2):
                        dim_base = dim_half * 64 + lane_in_row * 8
                        reg_base = dim_half * 8
                        for group4 in range(2):
                            dim4 = dim_base + group4 * 4
                            values = K.alloc_local((4,), "float32")
                            lasts = K.alloc_local((4,), "float32")
                            K.ptx.ld.shared.v4.f32(
                                values[0],
                                values[1],
                                values[2],
                                values[3],
                                arena.ptr_to(
                                    [_raw_f32_byte(gate_raw_base, raw_stage, decay_row, dim4)]
                                ),
                            )
                            K.ptx.ld.shared.v4.f32(
                                lasts[0],
                                lasts[1],
                                lasts[2],
                                lasts[3],
                                arena.ptr_to([_raw_f32_byte(gate_raw_base, raw_stage, 15, dim4)]),
                            )
                            for offset in range(4):
                                reg = reg_base + group4 * 4 + offset
                                K.assign(exp_g[reg], values[offset])
                                K.assign(exp_g_last[reg], lasts[offset])
                                if not checkpoints:
                                    K.assign(exp_neg_g[reg], _rcp(values[offset]))

                        decay_words = K.alloc_local((4,), "uint32")
                        for pair in range(4):
                            reg = reg_base + pair * 2
                            k0, k1 = _fmul2(raw_k[reg], raw_k[reg + 1], k_inv_norm, k_inv_norm)
                            kb0, kb1 = _fmul2(k0, k1, raw_beta[reg], raw_beta[reg + 1])
                            k_beta = _pack_io_pair(kb0, kb1, io_dtype)
                            exp_pair = _pack_io_pair(exp_g[reg], exp_g[reg + 1], io_dtype)
                            _mul_io_pair(decay_words[pair], k_beta, exp_pair, io_dtype)
                            if checkpoints:
                                inv_pair = _pack_io_pair(
                                    _rcp(exp_g[reg]), _rcp(exp_g[reg + 1]), io_dtype
                                )
                            else:
                                inv_pair = _pack_io_pair(
                                    exp_neg_g[reg], exp_neg_g[reg + 1], io_dtype
                                )
                            k_pair = _pack_io_pair(k0, k1, io_dtype)
                            _mul_io_pair(
                                k_inv_words[dim_half * 4 + pair], k_pair, inv_pair, io_dtype
                            )

                        if dim_half == 0:
                            _wait_barrier(
                                arena, protocol[15][0], decay_stage, (serial // 2 + 1) & 1
                            )
                            _wait_barrier(
                                arena, protocol[14][1], decay_stage, (serial // 2 + 1) & 1
                            )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to(
                                [_raw_io_byte(k_inv_base, decay_stage, decay_row, dim_base)]
                            ),
                            k_inv_words[dim_half * 4],
                            k_inv_words[dim_half * 4 + 1],
                            k_inv_words[dim_half * 4 + 2],
                            k_inv_words[dim_half * 4 + 3],
                        )
                        storage_key = K.bitwise_xor(dim_base, K.int32(8))
                        storage_slice = storage_key // 64
                        decay_element = storage_slice * 1024 + _swizzle_xor_128b(
                            decay_row, decay_row * 64 + storage_key - storage_slice * 64, 2
                        )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to([k_decay_base + decay_stage * 4096 + decay_element * 2]),
                            decay_words[0],
                            decay_words[1],
                            decay_words[2],
                            decay_words[3],
                        )

                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, protocol[14][0], decay_stage)
                    _arrive_barrier(arena, protocol[0][1], raw_stage)
                    _arrive_barrier(arena, protocol[1][1], raw_stage)
                    _arrive_barrier(arena, protocol[4][1], raw_stage)
                    _arrive_barrier(arena, protocol[5][1], raw_stage)

                    for dim_half in range(2):
                        dim_base = dim_half * 64 + lane_in_row * 8
                        reg_base = dim_half * 8
                        q_decay_words = K.alloc_local((4,), "uint32")
                        restore_words = K.alloc_local((4,), "uint32")
                        for pair in range(4):
                            reg = reg_base + pair * 2
                            q0, q1 = _fmul2(raw_q[reg], raw_q[reg + 1], q_inv_norm, q_inv_norm)
                            q_pair = _pack_io_pair(q0, q1, io_dtype)
                            exp_pair = _pack_io_pair(exp_g[reg], exp_g[reg + 1], io_dtype)
                            _mul_io_pair(q_decay_words[pair], q_pair, exp_pair, io_dtype)
                            last_pair = _pack_io_pair(
                                exp_g_last[reg], exp_g_last[reg + 1], io_dtype
                            )
                            _mul_io_pair(
                                restore_words[pair],
                                k_inv_words[dim_half * 4 + pair],
                                last_pair,
                                io_dtype,
                            )
                        storage_key = K.bitwise_xor(dim_base, K.int32(8))
                        storage_slice = storage_key // 64
                        decay_element = storage_slice * 1024 + _swizzle_xor_128b(
                            decay_row, decay_row * 64 + storage_key - storage_slice * 64, 2
                        )
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to([q_decay_base + decay_stage * 4096 + decay_element * 2]),
                            q_decay_words[0],
                            q_decay_words[1],
                            q_decay_words[2],
                            q_decay_words[3],
                        )
                        if dim_half == 0:
                            _wait_barrier(
                                arena, protocol[17][0], decay_stage, (serial // 2 + 1) & 1
                            )
                        storage_row = decay_row ^ 8
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to(
                                [_raw_io_byte(k_restore_base, decay_stage, storage_row, dim_base)]
                            ),
                            restore_words[0],
                            restore_words[1],
                            restore_words[2],
                            restore_words[3],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    _arrive_barrier(arena, protocol[16][0], diag_stage)
                    K.assign(local_chunk, local_chunk + 2)

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, protocol[23][0], sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([sched_base + sched_consumer.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, protocol[23][1], sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)

        # Warps 8..11: recurrent state owner, Y/U repack, O/checkpoint staging.
        with cg1:
            K.ptx.bar.sync(K.uint32(3), K.uint32(160))
            tmem_base = K.local_scalar("int32")
            K.ptx.ld.volatile.shared.s32(tmem_base, arena.ptr_to([tmem_mailbox]))
            tmem_col = tmem_base & 0xFFFF
            tmem_row = tmem_base >> 16
            subpartition = warp % 4
            value_base = subpartition * 32
            value_dim = value_base + lane
            frag_row = (lane // 16) * 8 + (lane & 7)
            frag_col = ((lane // 8) & 1) * 8
            row_id = tmem_row + value_base
            state_k_cursor = K.PipelineState(1, phase=0)
            u_acc_cursor = K.PipelineState(1, phase=0)
            o_acc_cursor = K.PipelineState(1, phase=0)
            k_restore_cursor = K.PipelineState(2, phase=0)
            raw_cursor = K.PipelineState(raw_stages, phase=0)
            raw_ready_cursor = K.PipelineState(raw_ready_stages, phase=0)
            checkpoint_done = K.PipelineState(checkpoint_stages, phase=1)
            sched_consumer = K.PipelineState(8, phase=0)
            serial_base = K.local_scalar("int32", init=K.int32(0))

            def stage_output(output_serial):
                qstate_stage = output_serial % 2
                o_stage = output_serial % 2
                _wait_barrier(arena, protocol[6][0], o_acc_cursor.stage, o_acc_cursor.phase)
                o_acc_cursor.advance()
                projection_col = tmem_col + 192 + qstate_stage * 16
                loaded0 = K.alloc_local((8,), "float32")
                loaded1 = K.alloc_local((8,), "float32")
                K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                    *[loaded0[i] for i in range(8)],
                    K.cast(projection_col + K.shift_left(row_id, K.int32(16)), "uint32"),
                )
                K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                    *[loaded1[i] for i in range(8)],
                    K.cast(projection_col + K.shift_left(row_id + 16, K.int32(16)), "uint32"),
                )
                if not checkpoints:
                    _wait_barrier(arena, protocol[21][1], o_stage, (output_serial // 2 + 1) & 1)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                packed0 = K.alloc_local((4,), "uint32")
                packed1 = K.alloc_local((4,), "uint32")
                for pair in range(4):
                    scaled0, scaled1 = _fmul2(
                        loaded0[pair * 2], loaded0[pair * 2 + 1], scale, scale
                    )
                    K.assign(packed0[pair], _pack_io_pair(scaled0, scaled1, io_dtype))
                    scaled0, scaled1 = _fmul2(
                        loaded1[pair * 2], loaded1[pair * 2 + 1], scale, scale
                    )
                    K.assign(packed1[pair], _pack_io_pair(scaled0, scaled1, io_dtype))
                if checkpoints:
                    _wait_barrier(arena, protocol[21][1], o_stage, (output_serial // 2 + 1) & 1)
                K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    arena.ptr_to([_raw_io_byte(o_base, o_stage, frag_row, value_base + frag_col)]),
                    packed0[0],
                    packed0[1],
                    packed0[2],
                    packed0[3],
                )
                K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    arena.ptr_to(
                        [_raw_io_byte(o_base, o_stage, frag_row, value_base + 16 + frag_col)]
                    ),
                    packed1[0],
                    packed1[1],
                    packed1[2],
                    packed1[3],
                )
                _arrive_barrier(arena, protocol[6][1], qstate_stage)
                K.ptx.fence.proxy.async_.shared__cta()
                _arrive_barrier(arena, protocol[21][0], o_stage)

            def stage_y(have_state, raw_stage, ready_stage, ready_phase):
                _wait_barrier(arena, protocol[2][0], ready_stage, ready_phase)
                raw_v0 = K.alloc_local((4,), "uint32")
                raw_v1 = K.alloc_local((4,), "uint32")
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    raw_v0[0],
                    raw_v0[1],
                    raw_v0[2],
                    raw_v0[3],
                    arena.ptr_to(
                        [_raw_io_byte(v_raw_base, raw_stage, frag_row, value_base + frag_col)]
                    ),
                )
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    raw_v1[0],
                    raw_v1[1],
                    raw_v1[2],
                    raw_v1[3],
                    arena.ptr_to(
                        [_raw_io_byte(v_raw_base, raw_stage, frag_row, value_base + 16 + frag_col)]
                    ),
                )
                state_k0 = K.alloc_local((8,), "float32")
                state_k1 = K.alloc_local((8,), "float32")
                if peel_cg1_first_chunk:
                    with K.If(have_state), K.Then():
                        _wait_barrier(
                            arena, protocol[7][0], state_k_cursor.stage, state_k_cursor.phase
                        )
                        state_k_cursor.advance()
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[state_k0[i] for i in range(8)],
                            K.cast(tmem_col + 224 + K.shift_left(row_id, K.int32(16)), "uint32"),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[state_k1[i] for i in range(8)],
                            K.cast(
                                tmem_col + 224 + K.shift_left(row_id + 16, K.int32(16)), "uint32"
                            ),
                        )
                _wait_barrier(arena, protocol[3][0], ready_stage, ready_phase)
                raw_w0 = K.alloc_local((4,), "uint32")
                raw_w1 = K.alloc_local((4,), "uint32")
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    raw_w0[0],
                    raw_w0[1],
                    raw_w0[2],
                    raw_w0[3],
                    arena.ptr_to(
                        [_raw_io_byte(w_raw_base, raw_stage, frag_row, value_base + frag_col)]
                    ),
                )
                K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    raw_w1[0],
                    raw_w1[1],
                    raw_w1[2],
                    raw_w1[3],
                    arena.ptr_to(
                        [_raw_io_byte(w_raw_base, raw_stage, frag_row, value_base + 16 + frag_col)]
                    ),
                )
                if peel_cg1_first_chunk:
                    with K.If(have_state), K.Then():
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                else:
                    with K.If(have_state), K.Then():
                        _wait_barrier(
                            arena, protocol[7][0], state_k_cursor.stage, state_k_cursor.phase
                        )
                        state_k_cursor.advance()
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[state_k0[i] for i in range(8)],
                            K.cast(tmem_col + 224 + K.shift_left(row_id, K.int32(16)), "uint32"),
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[state_k1[i] for i in range(8)],
                            K.cast(
                                tmem_col + 224 + K.shift_left(row_id + 16, K.int32(16)), "uint32"
                            ),
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()

                y0 = K.alloc_local((4,), "uint32")
                y1 = K.alloc_local((4,), "uint32")
                for reg in range(4):
                    raw_matrix = (1 - reg // 2) * 2 + (reg & 1)
                    frag_pair = (reg ^ 2) * 2
                    state_pair0 = K.local_scalar("uint32", init=K.uint32(0))
                    state_pair1 = K.local_scalar("uint32", init=K.uint32(0))
                    with K.If(have_state), K.Then():
                        K.assign(
                            state_pair0,
                            _pack_io_pair(state_k0[frag_pair], state_k0[frag_pair + 1], io_dtype),
                        )
                        K.assign(
                            state_pair1,
                            _pack_io_pair(state_k1[frag_pair], state_k1[frag_pair + 1], io_dtype),
                        )
                    product0 = K.local_scalar("uint32")
                    product1 = K.local_scalar("uint32")
                    _mul_io_pair(product0, raw_w0[raw_matrix], raw_v0[raw_matrix], io_dtype)
                    _mul_io_pair(product1, raw_w1[raw_matrix], raw_v1[raw_matrix], io_dtype)
                    _sub_io_pair(y0[reg], product0, state_pair0, io_dtype)
                    _sub_io_pair(y1[reg], product1, state_pair1, io_dtype)
                K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                    K.cast(tmem_col + 256 + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                    y0[0],
                    y0[1],
                    y0[2],
                    y0[3],
                )
                K.ptx["tcgen05.st.sync.aligned.16x128b.x2.b32"](
                    K.cast(tmem_col + 256 + K.shift_left(tmem_row + 16, K.int32(16)), "uint32"),
                    y1[0],
                    y1[1],
                    y1[2],
                    y1[3],
                )
                K.ptx.tcgen05.wait__st.sync.aligned()
                _arrive_barrier(arena, protocol[2][1], raw_stage)
                _arrive_barrier(arena, protocol[3][1], raw_stage)
                _arrive_barrier(arena, protocol[10][0])

            def stage_u():
                _wait_barrier(arena, protocol[8][0], u_acc_cursor.stage, u_acc_cursor.phase)
                u_acc_cursor.advance()
                values = K.alloc_local((16,), "float32")
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                    *[values[i] for i in range(16)],
                    K.cast(tmem_col + 240 + K.shift_left(row_id, K.int32(16)), "uint32"),
                )
                K.ptx.tcgen05.wait__ld.sync.aligned()
                packed = K.alloc_local((8,), "uint32")
                for column in range(8):
                    source_pair = column ^ 4
                    K.assign(
                        packed[column],
                        _pack_io_pair(
                            values[source_pair * 2], values[source_pair * 2 + 1], io_dtype
                        ),
                    )
                K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                    K.cast(tmem_col + 264 + K.shift_left(tmem_row, K.int32(16)), "uint32"),
                    *[packed[i] for i in range(8)],
                )
                K.ptx.tcgen05.wait__st.sync.aligned()
                _arrive_barrier(arena, protocol[11][0])

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
                sequence_length = eos - bos
                sequence_chunks = (sequence_length + 15) // 16
                num_chunks = wend - cstart
                state_index = (
                    (K.cast(batch, "int64") * n_heads_out + head) * 128 + value_dim
                ) * 128

                with K.If(num_chunks > 0), K.Then():
                    if use_initial_state:
                        for key_block in range(4):
                            state_values = K.alloc_local((32,), "float32")
                            for key in range(32):
                                value = K.local_scalar("float32", init=K.float32(0.0))
                                with K.If(cstart == 0), K.Then():
                                    K.assign(
                                        value,
                                        _load_state_value(
                                            initial_state,
                                            state_index + key_block * 32 + key,
                                            state_dtype,
                                        ),
                                    )
                                K.assign(state_values[key], value)
                            K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                                K.cast(
                                    tmem_col + key_block * 32 + K.shift_left(row_id, K.int32(16)),
                                    "uint32",
                                ),
                                *[state_values[i] for i in range(32)],
                            )
                        K.ptx.tcgen05.wait__st.sync.aligned()

                        cp_stage = K.local_scalar("int32", init=K.int32(0))
                        if checkpoints:
                            K.ptx.mov.b32(cp_stage, checkpoint_done.stage)
                            with K.If(wstart == 0), K.Then():
                                _wait_barrier(
                                    arena,
                                    protocol[22][1],
                                    checkpoint_done.stage,
                                    checkpoint_done.phase,
                                )
                                checkpoint_done.advance()
                        for key_block in range(8):
                            values = K.alloc_local((16,), "float32")
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                                *[values[i] for i in range(16)],
                                K.cast(
                                    tmem_col + key_block * 16 + K.shift_left(row_id, K.int32(16)),
                                    "uint32",
                                ),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            packed = K.alloc_local((8,), "uint32")
                            for column in range(8):
                                source_pair = column ^ 4
                                K.assign(
                                    packed[column],
                                    _pack_io_pair(
                                        values[source_pair * 2],
                                        values[source_pair * 2 + 1],
                                        io_dtype,
                                    ),
                                )
                            K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                                K.cast(
                                    tmem_col
                                    + 128
                                    + key_block * 8
                                    + K.shift_left(tmem_row, K.int32(16)),
                                    "uint32",
                                ),
                                *[packed[i] for i in range(8)],
                            )
                            if checkpoints:
                                with K.If(wstart == 0), K.Then():
                                    for half in range(2):
                                        checkpoint_words = K.alloc_local((4,), "uint32")
                                        for pair in range(4):
                                            K.assign(
                                                checkpoint_words[pair],
                                                _pack_io_pair(
                                                    values[half * 8 + pair * 2],
                                                    values[half * 8 + pair * 2 + 1],
                                                    io_dtype,
                                                ),
                                            )
                                        dk = key_block * 16 + half * 8
                                        cp_element = (
                                            (dk // 64) * 8192
                                            + value_dim * 64
                                            + _swizzle_xor_128b(value_dim, dk % 64, 2)
                                        )
                                        K.ptx.st.shared.v4.b32(
                                            arena.ptr_to(
                                                [
                                                    checkpoint_base
                                                    + cp_stage * 32768
                                                    + cp_element * 2
                                                ]
                                            ),
                                            checkpoint_words[0],
                                            checkpoint_words[1],
                                            checkpoint_words[2],
                                            checkpoint_words[3],
                                        )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        _arrive_barrier(arena, protocol[9][0])
                        if checkpoints:
                            with K.If(wstart == 0), K.Then():
                                K.ptx.fence.proxy.async_.shared__cta()
                                _arrive_barrier(arena, protocol[22][0], cp_stage)
                            _arrive_barrier(arena, protocol[20][0])
                    elif checkpoints:
                        with K.If(wstart == 0), K.Then():
                            cp_stage = K.local_scalar("int32")
                            K.ptx.mov.b32(cp_stage, checkpoint_done.stage)
                            _wait_barrier(arena, protocol[22][1], cp_stage, checkpoint_done.phase)
                            checkpoint_done.advance()
                            for key_block in range(8):
                                for half in range(2):
                                    dk = key_block * 16 + half * 8
                                    cp_element = (
                                        (dk // 64) * 8192
                                        + value_dim * 64
                                        + _swizzle_xor_128b(value_dim, dk % 64, 2)
                                    )
                                    K.ptx.st.shared.v4.b32(
                                        arena.ptr_to(
                                            [checkpoint_base + cp_stage * 32768 + cp_element * 2]
                                        ),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                    )
                            K.ptx.fence.proxy.async_.shared__cta()
                            _arrive_barrier(arena, protocol[22][0], cp_stage)

                    checkpoint_chunks = K.local_scalar("int32", init=K.int32(1))
                    checkpoint_mod = K.local_scalar("int32", init=K.int32(0))
                    if checkpoints:
                        K.assign(checkpoint_chunks, checkpoint_every_n // 16)
                        K.assign(checkpoint_mod, (cstart + 1) % checkpoint_chunks)

                    if peel_cg1_first_chunk:
                        raw_stage = raw_cursor.stage
                        ready_stage = raw_ready_cursor.stage
                        ready_phase = raw_ready_cursor.phase
                        if use_initial_state:
                            stage_y(K.bool(True), raw_stage, ready_stage, ready_phase)
                        else:
                            stage_y(K.bool(False), raw_stage, ready_stage, ready_phase)
                        stage_u()
                        _wait_barrier(
                            arena, protocol[17][0], k_restore_cursor.stage, k_restore_cursor.phase
                        )
                        k_restore_cursor.advance()
                        raw_cursor.advance()
                        raw_ready_cursor.advance()
                        recurrent_chunks = num_chunks - 1
                        first_recurrent_chunk = 1
                    else:
                        recurrent_chunks = num_chunks
                        first_recurrent_chunk = 0

                    with K.serial(recurrent_chunks, unroll=False) as recurrent_chunk:
                        local_chunk = recurrent_chunk + first_recurrent_chunk
                        serial = serial_base + local_chunk
                        raw_stage = raw_cursor.stage
                        ready_stage = raw_ready_cursor.stage
                        ready_phase = raw_ready_cursor.phase

                        def stage_recurrent_prefix():
                            do_checkpoint = K.local_scalar("bool", init=K.bool(False))
                            cp_stage = K.local_scalar("int32", init=K.int32(0))
                            if checkpoints:
                                K.assign(
                                    do_checkpoint,
                                    K.And(checkpoint_mod == 0, cstart + local_chunk >= wstart),
                                )
                                K.ptx.mov.b32(cp_stage, checkpoint_done.stage)
                                with K.If(do_checkpoint), K.Then():
                                    _wait_barrier(
                                        arena, protocol[22][1], cp_stage, checkpoint_done.phase
                                    )
                                    checkpoint_done.advance()

                            def store_state_block(key_block, values):
                                packed = K.alloc_local((8,), "uint32")
                                for column in range(8):
                                    source_pair = column ^ 4
                                    K.assign(
                                        packed[column],
                                        _pack_io_pair(
                                            values[source_pair * 2],
                                            values[source_pair * 2 + 1],
                                            io_dtype,
                                        ),
                                    )
                                K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                                    K.cast(
                                        tmem_col
                                        + 128
                                        + key_block * 8
                                        + K.shift_left(tmem_row, K.int32(16)),
                                        "uint32",
                                    ),
                                    *[packed[i] for i in range(8)],
                                )
                                if checkpoints:
                                    with K.If(do_checkpoint), K.Then():
                                        for half in range(2):
                                            cp_words = K.alloc_local((4,), "uint32")
                                            for pair in range(4):
                                                K.assign(
                                                    cp_words[pair],
                                                    _pack_io_pair(
                                                        values[half * 8 + pair * 2],
                                                        values[half * 8 + pair * 2 + 1],
                                                        io_dtype,
                                                    ),
                                                )
                                            dk = key_block * 16 + half * 8
                                            cp_element = (
                                                (dk // 64) * 8192
                                                + value_dim * 64
                                                + _swizzle_xor_128b(value_dim, dk % 64, 2)
                                            )
                                            K.ptx.st.shared.v4.b32(
                                                arena.ptr_to(
                                                    [
                                                        checkpoint_base
                                                        + cp_stage * 32768
                                                        + cp_element * 2
                                                    ]
                                                ),
                                                cp_words[0],
                                                cp_words[1],
                                                cp_words[2],
                                                cp_words[3],
                                            )

                            if checkpoints:
                                for key_block in range(8):
                                    values = K.alloc_local((16,), "float32")
                                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                                        *[values[i] for i in range(16)],
                                        K.cast(
                                            tmem_col
                                            + key_block * 16
                                            + K.shift_left(row_id, K.int32(16)),
                                            "uint32",
                                        ),
                                    )
                                    K.ptx.tcgen05.wait__ld.sync.aligned()
                                    store_state_block(key_block, values)
                            else:
                                state_values = K.alloc_local((8, 16), "float32")
                                for key_block in range(8):
                                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                                        *[state_values[key_block, i] for i in range(16)],
                                        K.cast(
                                            tmem_col
                                            + key_block * 16
                                            + K.shift_left(row_id, K.int32(16)),
                                            "uint32",
                                        ),
                                    )
                                K.ptx.tcgen05.wait__ld.sync.aligned()
                                for key_block in range(8):
                                    store_state_block(
                                        key_block, [state_values[key_block, i] for i in range(16)]
                                    )
                            K.ptx.tcgen05.wait__st.sync.aligned()
                            _arrive_barrier(arena, protocol[9][0])
                            if checkpoints:
                                _arrive_barrier(arena, protocol[20][0])
                                with K.If(do_checkpoint), K.Then():
                                    K.ptx.fence.proxy.async_.shared__cta()
                                    _arrive_barrier(arena, protocol[22][0], cp_stage)
                                K.assign(checkpoint_mod, checkpoint_mod + 1)
                                with K.If(checkpoint_mod == checkpoint_chunks), K.Then():
                                    K.assign(checkpoint_mod, K.int32(0))
                            stage_output(serial - 1)

                        if peel_cg1_first_chunk:
                            stage_recurrent_prefix()
                            stage_y(K.bool(True), raw_stage, ready_stage, ready_phase)
                        else:
                            with K.If(local_chunk > 0), K.Then():
                                stage_recurrent_prefix()
                            have_state = K.bool(True) if use_initial_state else local_chunk > 0
                            stage_y(have_state, raw_stage, ready_stage, ready_phase)
                        stage_u()
                        _wait_barrier(
                            arena, protocol[17][0], k_restore_cursor.stage, k_restore_cursor.phase
                        )
                        k_restore_cursor.advance()
                        raw_cursor.advance()
                        raw_ready_cursor.advance()

                    stage_output(serial_base + num_chunks - 1)

                    if store_final_state:
                        with K.If(wend == sequence_chunks), K.Then():
                            for key_block in range(4):
                                values = K.alloc_local((32,), "float32")
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                    *[values[i] for i in range(32)],
                                    K.cast(
                                        tmem_col
                                        + key_block * 32
                                        + K.shift_left(row_id, K.int32(16)),
                                        "uint32",
                                    ),
                                )
                                K.ptx.tcgen05.wait__ld.sync.aligned()
                                if state_dtype == "float32":
                                    for vector in range(8):
                                        key = vector * 4
                                        K.ptx.st.global_.v4.f32(
                                            final_state.ptr_to(
                                                [state_index + key_block * 32 + key]
                                            ),
                                            values[key],
                                            values[key + 1],
                                            values[key + 2],
                                            values[key + 3],
                                        )
                                else:
                                    for vector in range(4):
                                        key = vector * 8
                                        words = K.alloc_local((4,), "uint32")
                                        for pair in range(4):
                                            K.assign(
                                                words[pair],
                                                _pack_io_pair(
                                                    values[key + pair * 2],
                                                    values[key + pair * 2 + 1],
                                                    "bfloat16",
                                                ),
                                            )
                                        K.ptx.st.global_.v4.b32(
                                            final_state.ptr_to(
                                                [state_index + key_block * 32 + key]
                                            ),
                                            words[0],
                                            words[1],
                                            words[2],
                                            words[3],
                                        )
                if store_final_state:
                    with K.If(sequence_length == 0), K.Then():
                        for key in range(128):
                            value = K.local_scalar("float32", init=K.float32(0.0))
                            if use_initial_state:
                                K.assign(
                                    value,
                                    _load_state_value(
                                        initial_state, state_index + key, state_dtype
                                    ),
                                )
                            _store_state_value(final_state, state_index + key, value, state_dtype)

                K.assign(serial_base, serial_base + num_chunks)
                if dynamic_scheduler:
                    _wait_barrier(
                        arena, protocol[23][0], sched_consumer.stage, sched_consumer.phase
                    )
                    K.ptx.ld.shared.s32(tile, arena.ptr_to([sched_base + sched_consumer.stage * 4]))
                    with K.If(_elected()), K.Then():
                        _arrive_barrier(arena, protocol[23][1], sched_consumer.stage)
                    sched_consumer.advance()
                else:
                    K.assign(tile, tile + num_sms)
            _arrive_barrier(arena, protocol[19][0])

    return main


def _normalized_config(config):
    import math

    config = {key: value for key, value in config.items() if key != "label"}
    config.setdefault("seq_lens", (64,))
    config["seq_lens"] = tuple(int(value) for value in config["seq_lens"])
    config.setdefault("heads", 1)
    config.setdefault("q_heads", config["heads"])
    config.setdefault("k_heads", config["heads"])
    config.setdefault("v_heads", config["heads"])
    config.setdefault("io_dtype", "bfloat16")
    config.setdefault("state_dtype", "float32")
    config.setdefault("cu_dtype", "int32")
    config.setdefault("num_sms", 148)
    config.setdefault("scale", 1.0 / (_DK**0.5))
    config.setdefault("checkpoint_every_n_tokens", config.pop("checkpoint", 0))
    config.setdefault("gate_lower_bound", -5.0)
    config.setdefault("use_initial_state", False)
    config.setdefault("store_final_state", True)
    config.setdefault("l2norm", False)
    config.setdefault("safe_gate", False)
    config.setdefault("beta_sigmoid", False)
    config.setdefault("dynamic_scheduler", False)
    config.setdefault("split", False)
    # The source always executes order_body.  Generated uncut rows are its
    # default; scratch LPT ordering owns split rows.
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
    for name in ("q_heads", "k_heads", "v_heads"):
        value = int(config[name])
        if value <= 0 or heads % value != 0:
            raise ValueError(f"{name}={value} must be a positive divisor of heads={heads}")
    if heads != max(int(config["q_heads"]), int(config["v_heads"])):
        raise ValueError("heads must equal max(q_heads, v_heads), matching the source HO rule")
    if int(config["k_heads"]) not in (int(config["q_heads"]), int(config["v_heads"])):
        raise ValueError("k_heads must equal q_heads or v_heads")
    cadence = int(config["checkpoint_every_n_tokens"])
    if cadence < 0 or cadence % _BT != 0:
        raise ValueError("checkpoint cadence must be zero or a positive multiple of 16")
    if cadence not in (0, 16, 32, 48):
        raise ValueError("validated checkpoint cadences are 0, 16, 32, and 48")
    if config["split"] and config["order_generate"]:
        raise ValueError("split work rows require scratch ordering")
    if not config["run_order"]:
        raise ValueError("GDN2's frozen two-launch ABI always runs the order prologue")
    scale = float(config["scale"])
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("scale must be finite and nonzero")
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
        run_order=True,
        order_generate=bool(config["order_generate"]),
        dynamic_scheduler=bool(config["dynamic_scheduler"]),
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
        peel_cg1_first_chunk=(
            int(config["checkpoint_every_n_tokens"]) == 0 or len(config["seq_lens"]) <= 4
        ),
        l2norm=bool(config["l2norm"]),
        safe_gate=bool(config["safe_gate"]),
        gate_scale_log2=float(config["gate_lower_bound"]) * _LOG2_E,
        beta_sigmoid=bool(config["beta_sigmoid"]),
        dynamic_scheduler=bool(config["dynamic_scheduler"]),
        q_ratio=int(config["heads"]) // int(config["q_heads"]),
        k_ratio=int(config["heads"]) // int(config["k_heads"]),
        v_ratio=int(config["heads"]) // int(config["v_heads"]),
        n_heads_out=int(config["heads"]),
        full_tiles=all(length % _BT == 0 for length in config["seq_lens"]),
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
        work_items = torch.empty_like(base)
        staging = None if config["order_generate"] else base.clone()
        return {
            "work_items": work_items,
            "work_count": torch.tensor([base.shape[0]], dtype=torch.int32, device="cuda"),
            "staging": staging,
            "scheduler": (
                torch.zeros(1, dtype=torch.int32, device="cuda")
                if config["dynamic_scheduler"]
                else None
            ),
        }

    return {"tirx": one_side(), "source": one_side()}


def _new_outputs(torch, config, total_tokens):
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    state_t = torch.bfloat16 if config["state_dtype"] == "bfloat16" else torch.float32
    result = {
        "output": torch.full(
            (total_tokens, config["heads"], _DV), float("nan"), dtype=io_t, device="cuda"
        )
    }
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
    torch.manual_seed(20260823)
    total_tokens = sum(config["seq_lens"])
    io_t = torch.float16 if config["io_dtype"] == "float16" else torch.bfloat16
    state_t = torch.bfloat16 if config["state_dtype"] == "bfloat16" else torch.float32
    q = (0.2 * torch.randn(total_tokens, config["q_heads"], _DK, device="cuda")).to(io_t)
    k = (0.2 * torch.randn(total_tokens, config["k_heads"], _DK, device="cuda")).to(io_t)
    v = (0.2 * torch.randn(total_tokens, config["v_heads"], _DV, device="cuda")).to(io_t)
    gate = torch.empty(
        total_tokens, config["heads"], _DK, dtype=torch.float32, device="cuda"
    ).uniform_(-0.08, -0.01)
    if config["safe_gate"]:
        gate = 0.2 * torch.randn_like(gate)
    beta = (0.3 * torch.randn(total_tokens, config["heads"], _DK, device="cuda")).to(io_t)
    if not config["beta_sigmoid"]:
        beta = beta.sigmoid()
    w = torch.sigmoid(0.3 * torch.randn(total_tokens, config["heads"], _DV, device="cuda")).to(io_t)
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
        -4.0 + 0.1 * torch.randn(config["heads"], _DK, dtype=torch.float32, device="cuda")
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
    outputs = {
        "tirx": _new_outputs(torch, config, total_tokens),
        "source": _new_outputs(torch, config, total_tokens),
    }
    workspace_bytes = _TENSOR_MAP_BYTES * _TENSOR_MAP_ARRAYS * len(config["seq_lens"])
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
    total_tokens = data["q"].shape[0]

    def headed(tensor, channels, heads, box_channels):
        return _encode_tiled_map(
            tensor,
            (channels, heads, total_tokens),
            (tensor.stride(1) * tensor.element_size(), tensor.stride(0) * tensor.element_size()),
            (box_channels, 1, _BT),
        )

    output = data[side]
    maps = [
        headed(data["q"], _DK, config["q_heads"], 64),
        headed(data["k"], _DK, config["k_heads"], 64),
        headed(data["v"], _DV, config["v_heads"], 64),
        headed(data["gate"], _DK, config["heads"], 32),
        headed(data["beta"], _DK, config["heads"], 64),
        headed(data["w"], _DV, config["heads"], 64),
        headed(output["output"], _DV, config["heads"], 64),
    ]
    checkpoints = output.get("checkpoints")
    if checkpoints is None:
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
    dummy_i32 = torch.zeros(8, dtype=torch.int32, device="cuda")
    dummy_io = torch.zeros(1, dtype=io_t, device="cuda")
    dummy_state = torch.zeros(1, dtype=state_t, device="cuda")
    dummy_a_log = torch.zeros(config["heads"], dtype=torch.float32, device="cuda")
    dummy_dt_bias = torch.zeros(config["heads"], _DK, dtype=torch.float32, device="cuda")
    checkpoint = output.get("checkpoints", dummy_io)

    def launch(*, prologue_only=False):
        prologue(
            *maps,
            data["tirx_workspace"],
            data["cu_seqlens"],
            data["q"].view(torch.uint8).reshape(-1),
            data["k"].view(torch.uint8).reshape(-1),
            data["v"].view(torch.uint8).reshape(-1),
            data["gate"].view(torch.uint8).reshape(-1),
            data["beta"].view(torch.uint8).reshape(-1),
            data["w"].view(torch.uint8).reshape(-1),
            output["output"].view(torch.uint8).reshape(-1),
            checkpoint.view(torch.uint8).reshape(-1),
            work["staging"].reshape(-1) if work["staging"] is not None else dummy_i32,
            work["work_count"],
            work["work_items"].reshape(-1),
            work["scheduler"] if work["scheduler"] is not None else dummy_i32,
            len(config["seq_lens"]),
            data["q"].stride(0) * data["q"].element_size(),
            data["k"].stride(0) * data["k"].element_size(),
            data["v"].stride(0) * data["v"].element_size(),
            data["gate"].stride(0) * data["gate"].element_size(),
            data["beta"].stride(0) * data["beta"].element_size(),
            data["w"].stride(0) * data["w"].element_size(),
            output["output"].stride(0) * output["output"].element_size(),
            checkpoint.stride(0) * checkpoint.element_size() if checkpoint.ndim > 1 else 0,
            int(config["checkpoint_every_n_tokens"]),
        )
        if prologue_only:
            return
        main(
            data["tirx_workspace"],
            len(config["seq_lens"]),
            data["q"].reshape(-1),
            data["k"].reshape(-1),
            data["v"].reshape(-1),
            data["gate"].reshape(-1),
            data["a_log"] if data["a_log"] is not None else dummy_a_log,
            (
                data["dt_bias"].reshape(-1)
                if data["dt_bias"] is not None
                else dummy_dt_bias.reshape(-1)
            ),
            data["beta"].reshape(-1),
            data["w"].reshape(-1),
            data["cu_seqlens"],
            (
                data["initial_state"].reshape(-1)
                if data["initial_state"] is not None
                else dummy_state
            ),
            output["output"].reshape(-1),
            output.get("final_state", dummy_state).reshape(-1),
            work["work_items"].reshape(-1),
            work["work_count"],
            work["scheduler"] if work["scheduler"] is not None else dummy_i32,
            float(config["scale"]),
            int(config["checkpoint_every_n_tokens"]),
        )

    launch._keep_alive = (maps, dummy_i32, dummy_io, dummy_state, dummy_a_log, dummy_dt_bias)
    return launch


def _load_reference_source():
    from tirx_kernels.cudnn._reference import load_reference_module

    return load_reference_module("cudnn.linear_attention.frost.kernel.gdn2_prefill_f16")


def _source_launch(data):
    import torch

    source = _load_reference_source()
    config = data["config"]
    output = data["source"]
    work = data["work"]["source"]
    stream = int(torch.cuda.current_stream().cuda_stream)

    def launch():
        source.chunk_gdn2_sm100(
            data["q"],
            data["k"],
            data["v"],
            data["gate"],
            data["beta"],
            data["w"],
            output["output"],
            data["cu_seqlens"],
            data["initial_state"],
            output.get("final_state"),
            float(config["scale"]),
            checkpoint_every_n_tokens=int(config["checkpoint_every_n_tokens"]),
            output_state_checkpoints=output.get("checkpoints"),
            use_qk_l2norm_in_kernel=bool(config["l2norm"]),
            safe_gate=bool(config["safe_gate"]),
            gate_lower_bound=float(config["gate_lower_bound"]),
            a_log=data["a_log"],
            dt_bias=data["dt_bias"],
            use_beta_sigmoid=bool(config["beta_sigmoid"]),
            work_items=work["work_items"],
            work_count=work["work_count"],
            sched_ctr=work["scheduler"],
            work_item_scratch=work["staging"],
            tensormap_workspace=data["source_workspace"],
            stream=stream,
        )

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
            ratio = _rms_ratio(torch, data["tirx"][name], data["source"][name])
            if not math.isfinite(ratio) or ratio >= limit:
                failures[f"tirx_vs_source.{name}"] = ratio
    if failures:
        raise AssertionError(
            f"GDN2 prefill validation failed for {data['config']}: {failures}; limit={limit}"
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
