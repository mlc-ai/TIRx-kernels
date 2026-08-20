# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's STP producer-consumer vertical kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_stp.cuh.
"""

import ctypes
from typing import Any

import torch

import tirx_kernels.kern as K

from . import selective_state_update_stp_simple as _simple

KERNEL_META = {
    "name": "selective_state_update_stp_vertical",
    "category": "flashinfer",
    "compute_capability": 10,
}


_LOG2_E = _simple._LOG2_E
_LN_2 = _simple._LN_2
_FLT_LOWEST = _simple._FLT_LOWEST

_mul = _simple._mul
_add = _simple._add
_sub = _simple._sub
_fma = _simple._fma
_max = _simple._max
_min = _simple._min
_abs = _simple._abs
_exp2 = _simple._exp2
_log2 = _simple._log2
_div = _simple._div
_rcp = _simple._rcp
_prmt_5410 = _simple._prmt_5410
_mul_hi_u32 = _simple._mul_hi_u32
_mul_lo_s32 = _simple._mul_lo_s32
_add_s32 = _simple._add_s32
_lane_mask = _simple._lane_mask
_global_load_s64 = _simple._global_load_s64
_shared_load_u16 = _simple._shared_load_u16
_shared_load_u32 = _simple._shared_load_u32
_bf16_to_f32 = _simple._bf16_to_f32
_state_bits_to_f32 = _simple._state_bits_to_f32
_f32_to_state_bits = _simple._f32_to_state_bits
_f32_to_bf16 = _simple._f32_to_bf16
_load_two_byte_vector = _simple._load_two_byte_vector
_store_two_byte_vector = _simple._store_two_byte_vector

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
_BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"


def _global_load_nc_u16(buffer, index):
    out = K.alloc_local((1,), "uint16")
    K.evaluate(K.ptx.ld.global_.nc.b16(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_nc_u32(buffer, index):
    out = K.alloc_local((1,), "uint32")
    K.evaluate(K.ptx.ld.global_.nc.b32(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_nc_s32(buffer, index):
    out = K.alloc_local((1,), "int32")
    K.evaluate(K.ptx.ld.global_.nc.s32(out[0], buffer.ptr_to([index])))
    return out[0]


def _global_load_nc_s64(buffer, index):
    out = K.alloc_local((1,), "int64")
    K.evaluate(K.ptx.ld.global_.nc.s64(out[0], buffer.ptr_to([index])))
    return out[0]


def _load_weight_nc(buffer, index, dtype: str):
    if dtype == "float32":
        return K.reinterpret("float32", _global_load_nc_u32(buffer, index))
    return _bf16_to_f32(_global_load_nc_u16(buffer, index))


def _mbarrier_arrive_wait(barrier):
    token = K.alloc_local((1,), "uint64")
    done = K.alloc_local((1,), "uint32")
    K.evaluate(K.ptx.mbarrier.arrive.shared__cta.b64(token[0], barrier, K.uint32(1)))
    with K.While(True):
        K.evaluate(K.ptx.mbarrier.try_wait.shared__cta.b64(done[0], barrier, token[0]))
        with K.If(done[0] != K.uint32(0)):
            with K.Then():
                K.Break()


def _mbarrier_arrive(barrier):
    K.evaluate(K.ptx.mbarrier.arrive.shared__cta.b64(barrier, K.uint32(1)))


def _mbarrier_expect_tx(barrier, num_bytes):
    K.evaluate(
        K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(barrier, K.uint32(num_bytes))
    )


def _bulk_g2s(dst, source, source_index, num_bytes, barrier):
    K.evaluate(K.ptx[_BULK_G2S](dst, source.ptr_to([source_index]), K.uint32(num_bytes), barrier))


def _tma_g2s(dst, tensor_state, d, head, batch, barrier):
    K.evaluate(
        K.ptx[_TMA_G2S_4D](
            dst,
            K.address_of(tensor_state),
            K.int32(0),
            K.cast(d, "int32"),
            K.cast(head, "int32"),
            K.cast(batch, "int32"),
            barrier,
        )
    )


def _tma_s2g(src, tensor_state, d, head, batch):
    K.evaluate(
        K.ptx[_TMA_S2G_4D](
            K.address_of(tensor_state),
            K.int32(0),
            K.cast(d, "int32"),
            K.cast(head, "int32"),
            K.cast(batch, "int32"),
            src,
        )
    )


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "label": label,
        "batch": 64,
        "nheads": 64,
        "dim": 64,
        "dstate": 128,
        "ngroups": 8,
        "input_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "weight_dtype": "float32",
        "matrix_a_dtype": "float32",
        "index_dtype": "int64",
        "has_state_indices": True,
        "has_dst_indices": False,
        "index_rank": 1,
        "has_z": False,
        "has_d": True,
        "has_dt_bias": True,
        "dt_softplus": True,
        "update_state": True,
        "state_stride_factor": 1,
        "pad_every": 0,
        "use_out_tensor": True,
        "philox_rounds": 0,
        "seed": 0,
    }
    config.update(overrides)
    return config


# Every row changes a source branch or compile-time specialization.  Batch=1
# exercises the same code shape and remains in correctness coverage only.
BENCH_CONFIGS = [
    _case("b64_h64_d64_s128_r8_base"),
    _case("b64_h8_d64_s128_r1", nheads=8),
    _case("b64_h64_d128_s128_r8", dim=128),
    _case("b64_h64_d64_s64_r8", dstate=64),
    _case("b64_h64_d64_s96_r8", dstate=96),
    _case("b64_h64_d64_s256_r8", dstate=256),
    _case("b64_h64_d64_s128_r8_statef16", state_dtype="float16"),
    _case("b64_h64_d64_s128_r8_statef32", state_dtype="float32"),
    _case("b64_h64_d64_s128_r8_weightbf16", weight_dtype="bfloat16"),
    _case("b64_h64_d64_s128_r1", ngroups=64),
    _case("b64_h64_d64_s128_r2", ngroups=32),
    _case("b64_h64_d64_s128_r4", ngroups=16),
    _case("b64_h64_d64_s128_r16", ngroups=4),
    _case("b64_h64_d64_s128_r32", ngroups=2),
    _case("b64_h64_d64_s128_r64", ngroups=1),
    _case("b64_h64_d64_s128_r8_z", has_z=True),
    _case("b64_h64_d64_s128_r8_no_dt_bias", has_dt_bias=False),
    _case("b64_h64_d64_s128_r8_no_softplus", dt_softplus=False),
    _case("b64_h64_d64_s128_r8_no_update", update_state=False),
    _case("b64_h64_d64_s128_r8_no_indices", has_state_indices=False, index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_indices_i32", index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_stride2", state_stride_factor=2),
    _case("b64_h64_d64_s128_r8_dst2d", has_dst_indices=True, index_rank=2, index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_pad4", pad_every=4, index_dtype="int32"),
    _case("b64_h64_d64_s128_r8_int16", state_dtype="int16"),
    _case("b64_h64_d64_s128_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s64_r8_philox10", dstate=64, state_dtype="float16", philox_rounds=10, seed=42
    ),
]


# The public FlashInfer wrapper requires D, so the nullable-D branch is
# correctness-only.  DIM=64/128 is the reviewed vertical dispatch domain.
CONFIGS = [dict(config) for config in BENCH_CONFIGS] + [
    _case("b1_h64_d64_s128_r8", batch=1),
    _case("b64_h64_d64_s128_r8_no_d", has_d=False),
    _case("b64_h64_d64_s128_r8_out_allocated", use_out_tensor=False),
    *[
        _case(
            f"b{batch}_h64_d64_s128_r8_dst1d",
            batch=batch,
            has_dst_indices=True,
            index_dtype="int32",
        )
        for batch in (1, 4, 32, 64)
    ],
    *[
        _case(
            f"b{batch}_h64_d64_s128_r8_dst2d_correctness",
            batch=batch,
            has_dst_indices=True,
            index_rank=2,
            index_dtype="int32",
        )
        for batch in (1, 16)
    ],
    _case("b1_h64_d64_s128_r8_int16", batch=1, state_dtype="int16"),
    _case("b64_h8_d64_s128_r1_int16", nheads=8, state_dtype="int16"),
    _case("b64_h64_d128_s128_r8_int16", dim=128, state_dtype="int16"),
    _case("b64_h64_d64_s64_r8_int16", dstate=64, state_dtype="int16"),
    _case("b64_h64_d64_s256_r8_int16", dstate=256, state_dtype="int16"),
    _case("b64_h64_d64_s128_r8_int16_weightbf16", state_dtype="int16", weight_dtype="bfloat16"),
]


def _philox4x32(random_words, random_seed, random_offset, *, PHILOX_ROUNDS):
    c0 = K.local_scalar("uint32")
    K.assign(c0, K.cast(random_offset, "uint32"))
    c1 = K.local_scalar("uint32")
    K.assign(c1, K.cast(K.shift_right(K.cast(random_offset, "uint64"), K.uint64(32)), "uint32"))
    c2 = K.local_scalar("uint32")
    K.assign(c2, 0)
    c3 = K.local_scalar("uint32")
    K.assign(c3, 0)
    k0 = K.local_scalar("uint32")
    K.assign(k0, K.cast(K.reinterpret("uint64", random_seed), "uint32"))
    k1 = K.local_scalar("uint32")
    K.assign(
        k1, K.cast(K.shift_right(K.reinterpret("uint64", random_seed), K.uint64(32)), "uint32")
    )
    with K.unroll(PHILOX_ROUNDS) as _round:
        old_c0 = K.local_scalar("uint32")
        K.assign(old_c0, c0)
        old_c2 = K.local_scalar("uint32")
        K.assign(old_c2, c2)
        hi_b = K.local_scalar("uint32")
        K.assign(hi_b, _mul_hi_u32(K.uint32(0xCD9E8D57), old_c2))
        next_c0 = K.local_scalar("uint32")
        K.assign(next_c0, K.bitwise_xor(K.bitwise_xor(hi_b, c1), k0))
        hi_a = K.local_scalar("uint32")
        K.assign(hi_a, _mul_hi_u32(K.uint32(0xD2511F53), old_c0))
        next_c2 = K.local_scalar("uint32")
        K.assign(next_c2, K.bitwise_xor(K.bitwise_xor(hi_a, c3), k1))
        next_c1_s = K.local_scalar("int32")
        K.assign(next_c1_s, _mul_lo_s32(K.int32(-845247145), K.reinterpret("int32", old_c2)))
        next_c3_s = K.local_scalar("int32")
        K.assign(next_c3_s, _mul_lo_s32(K.int32(-766435501), K.reinterpret("int32", old_c0)))
        next_k0_s = K.local_scalar("int32")
        K.assign(next_k0_s, _add_s32(K.reinterpret("int32", k0), K.int32(-1640531527)))
        next_k1_s = K.local_scalar("int32")
        K.assign(next_k1_s, _add_s32(K.reinterpret("int32", k1), K.int32(-1150833019)))
        K.assign(c0, next_c0)
        K.assign(c1, K.reinterpret("uint32", next_c1_s))
        K.assign(c2, next_c2)
        K.assign(c3, K.reinterpret("uint32", next_c3_s))
        K.assign(k0, K.reinterpret("uint32", next_k0_s))
        K.assign(k1, K.reinterpret("uint32", next_k1_s))
    K.ptx.mov.b32(random_words[0], c0)
    K.ptx.mov.b32(random_words[1], c1)
    K.ptx.mov.b32(random_words[2], c2)
    K.ptx.mov.b32(random_words[3], c3)


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    ngroups = int(kwargs["ngroups"])
    state_dtype = str(kwargs["state_dtype"])
    state_stride_factor = int(kwargs.get("state_stride_factor", 1))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    if str(kwargs.get("input_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("vertical STP is scoped to bfloat16 input")
    if str(kwargs.get("matrix_a_dtype", "float32")) != "float32":
        raise ValueError("vertical STP is scoped to float32 matrix A")
    if dim not in (64, 128):
        raise ValueError("vertical STP dispatch requires dim in {64, 128}")
    if dstate not in (64, 96, 128, 256):
        raise ValueError("vertical STP requires dstate in {64, 96, 128, 256}")
    if nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by ngroups")
    if state_stride_factor < 1:
        raise ValueError("state_stride_factor must be positive")

    state_slots = max((2 if has_dst_indices else 1) * batch + 8, 16)
    state_stride = nheads * dim * dstate * state_stride_factor
    scale_stride = nheads * dim
    index_elements = batch * (2 if int(kwargs.get("index_rank", 1)) == 2 else 1)
    scale_state = state_dtype == "int16"
    state_bytes = 4 if state_dtype == "float32" else 2
    state_values_per_bank = 4 // state_bytes
    state_iterations = (dstate + 32 * state_values_per_bank - 1) // (32 * state_values_per_bank)
    philox_rounds = int(kwargs.get("philox_rounds", 0))
    if scale_state and dstate not in (64, 128, 256):
        raise ValueError("int16 vertical specializations require dstate in {64, 128, 256}")
    if philox_rounds not in (0, 10):
        raise ValueError("vertical stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and (state_dtype != "float16" or dstate not in (64, 128)):
        raise ValueError("philox10 is scoped to float16 state with dstate 64 or 128")

    state_stage_values = 16 * dstate
    state_stage_bytes = state_stage_values * state_bytes
    has_z = bool(kwargs.get("has_z", False))
    input_bytes = (
        dim * 2
        + dstate * 2
        + dstate * 2
        + (dim * 2 if has_z else 0)
        + (dim * 4 if scale_state else 0)
    )

    # These are the four 16-byte batch-stride preconditions enforced by the
    # frozen common host helper before the vertical launch.
    for name, stride_bytes in (
        ("x", nheads * dim * 2),
        ("z", nheads * dim * 2),
        ("B", ngroups * dstate * 2),
        ("C", ngroups * dstate * 2),
    ):
        if stride_bytes % 16 != 0:
            raise ValueError(f"{name} batch stride must be 16-byte aligned, got {stride_bytes}")

    return {
        "BATCH": batch,
        "NHEADS": nheads,
        "DIM": dim,
        "DSTATE": dstate,
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": str(kwargs["weight_dtype"]),
        "INDEX_DTYPE": str(kwargs["index_dtype"]),
        "STATE_ELEMENTS": state_slots * state_stride,
        "SCALE_ELEMENTS": state_slots * scale_stride if scale_state else 1,
        "X_ELEMENTS": batch * nheads * dim,
        "DT_ELEMENTS": batch * nheads,
        "BC_ELEMENTS": batch * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "HAS_STATE_INDICES": bool(kwargs.get("has_state_indices", True)),
        "HAS_DST_INDICES": has_dst_indices,
        "HAS_Z": has_z,
        "HAS_D": bool(kwargs.get("has_d", True)),
        "HAS_DT_BIAS": bool(kwargs.get("has_dt_bias", True)),
        "SCALE_STATE": scale_state,
        "PHILOX_ROUNDS": philox_rounds,
        "STATE_BYTES": state_bytes,
        "STATE_VALUES_PER_BANK": state_values_per_bank,
        "STATE_ITERATIONS": state_iterations,
        "NEW_STATE_COUNT": dstate // 32 if scale_state else 1,
        "STATE_STAGE_VALUES": state_stage_values,
        "STATE_STAGE_BYTES": state_stage_bytes,
        "INPUT_BYTES": input_bytes,
    }


def get_kernel(**kwargs: Any):
    """Build the K entry for one vertical specialization."""
    spec = _specialization(kwargs)
    DIM = spec["DIM"]
    DSTATE = spec["DSTATE"]
    STATE_DTYPE = spec["STATE_DTYPE"]
    WEIGHT_DTYPE = spec["WEIGHT_DTYPE"]
    INDEX_DTYPE = spec["INDEX_DTYPE"]
    HAS_STATE_INDICES = spec["HAS_STATE_INDICES"]
    HAS_DST_INDICES = spec["HAS_DST_INDICES"]
    HAS_Z = spec["HAS_Z"]
    HAS_D = spec["HAS_D"]
    HAS_DT_BIAS = spec["HAS_DT_BIAS"]
    SCALE_STATE = spec["SCALE_STATE"]
    PHILOX_ROUNDS = spec["PHILOX_ROUNDS"]
    STATE_BYTES = spec["STATE_BYTES"]
    STATE_VALUES_PER_BANK = spec["STATE_VALUES_PER_BANK"]
    STATE_ITERATIONS = spec["STATE_ITERATIONS"]
    NEW_STATE_COUNT = spec["NEW_STATE_COUNT"]
    STATE_STAGE_VALUES = spec["STATE_STAGE_VALUES"]
    STATE_STAGE_BYTES = spec["STATE_STAGE_BYTES"]
    INPUT_BYTES = spec["INPUT_BYTES"]

    @K.kernel(warps=5, arch="sm_100a", grid=(spec["BATCH"], spec["NHEADS"]))
    def selective_state_update_stp_vertical(
        tensor_state: K.TensorMap,
        state: K.gptr[spec["STATE_DTYPE"]],
        state_scale: K.gptr[K.f32],
        x: K.gptr[K.bf16],
        dt: K.gptr[spec["WEIGHT_DTYPE"]],
        matrix_a: K.gptr[K.f32],
        matrix_b: K.gptr[K.bf16],
        matrix_c: K.gptr[K.bf16],
        d_weight: K.gptr[spec["WEIGHT_DTYPE"]],
        z: K.gptr[K.bf16],
        dt_bias: K.gptr[spec["WEIGHT_DTYPE"]],
        state_indices: K.gptr[spec["INDEX_DTYPE"]],
        dst_indices: K.gptr[spec["INDEX_DTYPE"]],
        rand_seed: K.gptr[K.i64],
        output: K.gptr[K.bf16],
        state_stride_batch: K.i64,
        state_scale_stride_batch: K.i64,
        x_stride_batch: K.i64,
        dt_stride_batch: K.i64,
        b_stride_batch: K.i64,
        c_stride_batch: K.i64,
        z_stride_batch: K.i64,
        out_stride_batch: K.i64,
        state_indices_stride_batch: K.i64,
        dst_indices_stride_batch: K.i64,
        nheads_runtime: K.i32,
        ngroups_runtime: K.i32,
        dt_softplus: K.i32,
        update_state: K.i32,
        pad_slot_id: K.i32,
    ):
        batch_i, head = K.cta_id()
        smem = K.smem_pool()
        s_state = smem.alloc((3 * spec["STATE_STAGE_VALUES"],), spec["STATE_DTYPE"], align=128)
        s_x = smem.alloc((spec["DIM"],), K.bf16, align=16)
        s_z = smem.alloc((spec["DIM"],), K.bf16, align=16)
        s_b = smem.alloc((spec["DSTATE"],), K.bf16, align=16)
        s_c = smem.alloc((spec["DSTATE"],), K.bf16, align=16)
        s_out = smem.alloc((spec["DIM"],), K.f32, align=4)
        s_scale = smem.alloc((spec["DIM"],), K.f32, align=128) if spec["SCALE_STATE"] else s_out
        empty = K.MBarrier(smem, 3)
        full = K.MBarrier(smem, 3)
        consumers_ready = K.MBarrier(smem, 1)
        empty.init(129)
        full.init(129)
        consumers_ready.init(128)
        K.cuda.cta_sync()

        roles = K.specialize()
        consumer = roles.role("consumer", warps=range(4))
        producer = roles.role("producer", warps=[4])

        empty_barriers_buf = empty.buf
        full_barriers_buf = full.buf
        consumers_ready_buf = consumers_ready.buf

        def producer_pipeline(
            group, state_batch, dst_state_batch, READ_STATE: K.constexpr, WRITE_STATE: K.constexpr
        ):
            # Phase 1, stage 0: vector inputs and the first optional state tile share
            # one full barrier transaction, exactly as in producer_func_vertical.
            _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([0]))
            _bulk_g2s(
                s_x.ptr_to([0]),
                x,
                K.cast(batch_i, "int64") * x_stride_batch + head * DIM,
                DIM * 2,
                full_barriers_buf.ptr_to([0]),
            )
            _bulk_g2s(
                s_b.ptr_to([0]),
                matrix_b,
                K.cast(batch_i, "int64") * b_stride_batch + group * DSTATE,
                DSTATE * 2,
                full_barriers_buf.ptr_to([0]),
            )
            _bulk_g2s(
                s_c.ptr_to([0]),
                matrix_c,
                K.cast(batch_i, "int64") * c_stride_batch + group * DSTATE,
                DSTATE * 2,
                full_barriers_buf.ptr_to([0]),
            )
            if HAS_Z:
                _bulk_g2s(
                    s_z.ptr_to([0]),
                    z,
                    K.cast(batch_i, "int64") * z_stride_batch + head * DIM,
                    DIM * 2,
                    full_barriers_buf.ptr_to([0]),
                )
            if SCALE_STATE:
                _bulk_g2s(
                    s_scale.ptr_to([0]),
                    state_scale,
                    state_batch * state_scale_stride_batch + head * DIM,
                    DIM * 4,
                    full_barriers_buf.ptr_to([0]),
                )
            if READ_STATE:
                _tma_g2s(
                    s_state.ptr_to([0]),
                    tensor_state,
                    0,
                    head,
                    state_batch,
                    full_barriers_buf.ptr_to([0]),
                )
                _mbarrier_expect_tx(full_barriers_buf.ptr_to([0]), STATE_STAGE_BYTES + INPUT_BYTES)
            else:
                _mbarrier_expect_tx(full_barriers_buf.ptr_to([0]), INPUT_BYTES)

            # Phase 1, stages 1 and 2: state-only fill.
            with K.unroll(1, 3) as fill_iter:
                fill_stage: K.int32 = fill_iter
                fill_d: K.int32 = fill_iter * 16
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([fill_stage]))
                if READ_STATE:
                    _tma_g2s(
                        s_state.ptr_to([fill_stage * STATE_STAGE_VALUES]),
                        tensor_state,
                        fill_d,
                        head,
                        state_batch,
                        full_barriers_buf.ptr_to([fill_stage]),
                    )
                    _mbarrier_expect_tx(full_barriers_buf.ptr_to([fill_stage]), STATE_STAGE_BYTES)
                else:
                    _mbarrier_arrive(full_barriers_buf.ptr_to([fill_stage]))

            # Phase 2: every reused stage is stored before its next load.
            with K.unroll(DIM // 16 - 3) as steady_iter:
                steady_stage: K.int32 = (3 + steady_iter) % 3
                d_read: K.int32 = (3 + steady_iter) * 16
                d_write: K.int32 = steady_iter * 16
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([steady_stage]))
                if READ_STATE or WRITE_STATE:
                    K.evaluate(K.ptx.fence.proxy.async_.shared__cta())
                    if WRITE_STATE:
                        _tma_s2g(
                            s_state.ptr_to([steady_stage * STATE_STAGE_VALUES]),
                            tensor_state,
                            d_write,
                            head,
                            dst_state_batch,
                        )
                        K.evaluate(K.ptx.cp.async_.bulk.commit_group())
                        K.evaluate(K.ptx.cp.async_.bulk.wait_group.read(0))
                    if READ_STATE:
                        _tma_g2s(
                            s_state.ptr_to([steady_stage * STATE_STAGE_VALUES]),
                            tensor_state,
                            d_read,
                            head,
                            state_batch,
                            full_barriers_buf.ptr_to([steady_stage]),
                        )
                        _mbarrier_expect_tx(
                            full_barriers_buf.ptr_to([steady_stage]), STATE_STAGE_BYTES
                        )
                    else:
                        _mbarrier_arrive(full_barriers_buf.ptr_to([steady_stage]))
                else:
                    _mbarrier_arrive(full_barriers_buf.ptr_to([steady_stage]))

            # Phase 3: wait for and drain the final three shared state tiles.
            with K.unroll(3) as drain_iter:
                drain_stage: K.int32 = (DIM // 16 + drain_iter) % 3
                drain_d: K.int32 = (DIM // 16 - 3 + drain_iter) * 16
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([drain_stage]))
                if WRITE_STATE:
                    K.evaluate(K.ptx.fence.proxy.async_.shared__cta())
                    _tma_s2g(
                        s_state.ptr_to([drain_stage * STATE_STAGE_VALUES]),
                        tensor_state,
                        drain_d,
                        head,
                        dst_state_batch,
                    )
                    K.evaluate(K.ptx.cp.async_.bulk.commit_group())
                    K.evaluate(K.ptx.cp.async_.bulk.wait_group.read(0))

        def consumer_pipeline(
            lane,
            warp,
            d_value,
            dt_value,
            da_value,
            lane_indicator,
            random_seed,
            state_ptr_offset,
            USE_STATE_CACHE: K.constexpr,
        ):
            d_begin = K.local_scalar("int32")
            K.assign(d_begin, 0)
            state_pipe = K.PipelineState(3, phase=0)
            with K.While(d_begin < DIM):
                _mbarrier_arrive_wait(full_barriers_buf.ptr_to([state_pipe.stage]))
                with K.unroll(4) as row_iter:
                    dd: K.int32 = warp + row_iter * 4
                    row_d: K.int32 = d_begin + dd
                    x_value: K.float32 = _bf16_to_f32(_shared_load_u16(s_x, row_d))
                    d_times_x: K.float32 = _mul(d_value, x_value)
                    out_value = K.local_scalar("float32")
                    K.assign(out_value, _mul(d_times_x, lane_indicator))
                    decode_scale = K.local_scalar("float32")
                    K.assign(decode_scale, 1.0)
                    new_state_max = K.local_scalar("float32")
                    K.assign(new_state_max, K.float32(_FLT_LOWEST))
                    if SCALE_STATE:
                        K.assign(
                            decode_scale, K.reinterpret("float32", _shared_load_u32(s_scale, row_d))
                        )
                    new_states = K.alloc_local((NEW_STATE_COUNT,), "float32")

                    with K.serial(STATE_ITERATIONS) as state_iter:
                        state_i: K.int32 = (state_iter * 32 + lane) * STATE_VALUES_PER_BANK
                        with K.If(state_i < DSTATE):
                            with K.Then():
                                state_index: K.int32 = (
                                    state_pipe.stage * STATE_STAGE_VALUES + dd * DSTATE + state_i
                                )
                                if STATE_BYTES == 2:
                                    r_state = _load_two_byte_vector(
                                        s_state, state_index, STATE_VALUES_PER_BANK, "shared"
                                    )
                                    b_bits = _load_two_byte_vector(
                                        s_b, state_i, STATE_VALUES_PER_BANK, "shared"
                                    )
                                    c_bits = _load_two_byte_vector(
                                        s_c, state_i, STATE_VALUES_PER_BANK, "shared"
                                    )
                                    random_words = K.alloc_local((4,), "uint32")
                                    sr_raw = K.alloc_local((STATE_VALUES_PER_BANK,), "uint32")
                                    if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                                        random_offset: K.uint64 = K.cast(
                                            state_ptr_offset + row_d * DSTATE + state_i, "uint64"
                                        )
                                        _philox4x32(
                                            random_words,
                                            random_seed,
                                            random_offset,
                                            PHILOX_ROUNDS=PHILOX_ROUNDS,
                                        )
                                    with K.unroll(STATE_VALUES_PER_BANK) as e:
                                        state_value = K.local_scalar("float32")
                                        K.assign(state_value, 0.0)
                                        if USE_STATE_CACHE:
                                            K.assign(
                                                state_value,
                                                _state_bits_to_f32(r_state[e], STATE_DTYPE),
                                            )
                                            if SCALE_STATE:
                                                K.assign(
                                                    state_value, _mul(state_value, decode_scale)
                                                )
                                        b_value: K.float32 = _bf16_to_f32(b_bits[e])
                                        c_value: K.float32 = _bf16_to_f32(c_bits[e])
                                        db_value: K.float32 = _mul(b_value, dt_value)
                                        db_x: K.float32 = _mul(db_value, x_value)
                                        new_state: K.float32 = _fma(state_value, da_value, db_x)
                                        if SCALE_STATE:
                                            magnitude: K.float32 = _abs(new_state)
                                            K.assign(new_state_max, _max(new_state_max, magnitude))
                                            K.ptx.mov.b32(
                                                new_states[state_iter * STATE_VALUES_PER_BANK + e],
                                                new_state,
                                            )
                                        else:
                                            if PHILOX_ROUNDS > 0:
                                                random13: K.uint32 = K.bitwise_and(
                                                    random_words[e], K.uint32(0x1FFF)
                                                )
                                                K.evaluate(
                                                    K.ptx.cvt.rs.f16x2.f32(
                                                        sr_raw[e],
                                                        K.float32(0.0),
                                                        new_state,
                                                        random13,
                                                    )
                                                )
                                            else:
                                                K.ptx.mov.b16(
                                                    r_state[e],
                                                    _f32_to_state_bits(new_state, STATE_DTYPE),
                                                )
                                        K.assign(out_value, _fma(new_state, c_value, out_value))

                                    if not SCALE_STATE:
                                        if PHILOX_ROUNDS > 0:
                                            packed_sr: K.uint32 = _prmt_5410(sr_raw[0], sr_raw[1])
                                            K.evaluate(
                                                K.ptx.st.shared.b32(
                                                    s_state.ptr_to([state_index]), packed_sr
                                                )
                                            )
                                        else:
                                            _store_two_byte_vector(
                                                s_state,
                                                state_index,
                                                r_state,
                                                STATE_VALUES_PER_BANK,
                                                "shared",
                                            )
                                else:
                                    state_word: K.uint32 = _shared_load_u32(s_state, state_index)
                                    state_value = K.local_scalar("float32")
                                    K.assign(state_value, 0.0)
                                    if USE_STATE_CACHE:
                                        K.assign(state_value, K.reinterpret("float32", state_word))
                                    b_value: K.float32 = _bf16_to_f32(
                                        _shared_load_u16(s_b, state_i)
                                    )
                                    c_value: K.float32 = _bf16_to_f32(
                                        _shared_load_u16(s_c, state_i)
                                    )
                                    db_value: K.float32 = _mul(b_value, dt_value)
                                    db_x: K.float32 = _mul(db_value, x_value)
                                    new_state: K.float32 = _fma(state_value, da_value, db_x)
                                    K.assign(out_value, _fma(new_state, c_value, out_value))
                                    K.evaluate(
                                        K.ptx.st.shared.b32(
                                            s_state.ptr_to([state_index]),
                                            K.reinterpret("uint32", new_state),
                                        )
                                    )

                    with K.unroll(5) as delta_i:
                        delta: K.int32 = K.shift_right(K.int32(16), delta_i)
                        peer_out: K.float32 = K.cuda.__shfl_down_sync(
                            K.uint32(0xFFFFFFFF), out_value, delta, 32
                        )
                        K.assign(out_value, _add(out_value, peer_out))
                    with K.If(lane == 0):
                        with K.Then():
                            K.evaluate(
                                K.ptx.st.shared.b32(
                                    s_out.ptr_to([row_d]), K.reinterpret("uint32", out_value)
                                )
                            )

                    if SCALE_STATE and USE_STATE_CACHE:
                        with K.unroll(5) as delta_i:
                            delta: K.int32 = K.shift_right(K.int32(16), delta_i)
                            peer_max: K.float32 = K.cuda.__shfl_down_sync(
                                K.uint32(0xFFFFFFFF), new_state_max, delta, 32
                            )
                            K.assign(new_state_max, _max(new_state_max, peer_max))
                        # Unlike the simple kernel, the frozen vertical source has no
                        # standalone __syncwarp between max reduction and broadcast.
                        K.assign(
                            new_state_max,
                            K.cuda.__shfl_sync(K.uint32(0xFFFFFFFF), new_state_max, 0, 32),
                        )
                        encode_scale = K.local_scalar("float32")
                        K.assign(encode_scale, 1.0)
                        with K.If(new_state_max != K.float32(0.0)):
                            with K.Then():
                                K.assign(encode_scale, _div(K.float32(32767.0), new_state_max))
                        new_decode_scale: K.float32 = _rcp(encode_scale)
                        with K.serial(STATE_ITERATIONS) as state_iter:
                            state_i: K.int32 = (state_iter * 32 + lane) * STATE_VALUES_PER_BANK
                            with K.If(state_i < DSTATE):
                                with K.Then():
                                    quantized = K.alloc_local((STATE_VALUES_PER_BANK,), "int32")
                                    with K.unroll(STATE_VALUES_PER_BANK) as e:
                                        scaled: K.float32 = _mul(
                                            new_states[state_iter * STATE_VALUES_PER_BANK + e],
                                            encode_scale,
                                        )
                                        clipped_low: K.float32 = _max(scaled, K.float32(-32767.0))
                                        clipped: K.float32 = _min(clipped_low, K.float32(32767.0))
                                        K.evaluate(K.ptx.cvt.rni.ftz.s32.f32(quantized[e], clipped))
                                    packed_i16: K.uint32 = _prmt_5410(
                                        K.reinterpret("uint32", quantized[0]),
                                        K.reinterpret("uint32", quantized[1]),
                                    )
                                    state_index: K.int32 = (
                                        state_pipe.stage * STATE_STAGE_VALUES
                                        + dd * DSTATE
                                        + state_i
                                    )
                                    K.evaluate(
                                        K.ptx.st.shared.b32(
                                            s_state.ptr_to([state_index]), packed_i16
                                        )
                                    )
                        with K.If(lane == 0):
                            with K.Then():
                                K.evaluate(
                                    K.ptx.st.shared.b32(
                                        s_scale.ptr_to([row_d]),
                                        K.reinterpret("uint32", new_decode_scale),
                                    )
                                )

                K.evaluate(K.ptx.fence.proxy.async_.shared__cta())
                _mbarrier_arrive(empty_barriers_buf.ptr_to([state_pipe.stage]))
                K.assign(d_begin, d_begin + 16)
                state_pipe.advance()

        def run_role(ROLE: K.constexpr):
            flat_tid: K.int32 = K.thread_id()
            # TIRX_TRANSCRIBE_START selective_state_update_stp_vertical

            random_seed = K.local_scalar("int64")
            K.assign(random_seed, 0)
            if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                K.assign(random_seed, _global_load_s64(rand_seed, 0))

            lane: K.int32 = _lane_mask(flat_tid)
            warp: K.int32 = flat_tid >> 5
            group: K.int32 = head // (nheads_runtime // ngroups_runtime)

            state_batch = K.local_scalar("int64")
            if HAS_STATE_INDICES:
                if INDEX_DTYPE == "int32":
                    K.assign(
                        state_batch,
                        K.cast(
                            _global_load_nc_s32(
                                state_indices, batch_i * state_indices_stride_batch
                            ),
                            "int64",
                        ),
                    )
                else:
                    K.assign(
                        state_batch,
                        _global_load_nc_s64(state_indices, batch_i * state_indices_stride_batch),
                    )
            else:
                K.assign(state_batch, K.cast(batch_i, "int64"))

            dst_state_batch = K.local_scalar("int64")
            if HAS_DST_INDICES:
                if INDEX_DTYPE == "int32":
                    K.assign(
                        dst_state_batch,
                        K.cast(
                            _global_load_nc_s32(dst_indices, batch_i * dst_indices_stride_batch),
                            "int64",
                        ),
                    )
                else:
                    K.assign(
                        dst_state_batch,
                        _global_load_nc_s64(dst_indices, batch_i * dst_indices_stride_batch),
                    )
            else:
                K.assign(dst_state_batch, state_batch)

            state_ptr_offset: K.int64 = state_batch * state_stride_batch + K.cast(
                head * DIM * DSTATE, "int64"
            )
            scale_head_offset: K.int64 = state_batch * state_scale_stride_batch + K.cast(
                head * DIM, "int64"
            )
            dst_scale_head_offset: K.int64 = dst_state_batch * state_scale_stride_batch + K.cast(
                head * DIM, "int64"
            )

            K.evaluate(state.data)

            if ROLE == "producer":
                read_state: K.bool = state_batch != K.cast(pad_slot_id, "int64")
                write_state: K.bool = K.And(read_state, update_state != 0)
                with K.If(lane == 0):
                    with K.Then():
                        with K.If(read_state):
                            with K.Then():
                                with K.If(write_state):
                                    with K.Then():
                                        producer_pipeline(
                                            group,
                                            state_batch,
                                            dst_state_batch,
                                            READ_STATE=True,
                                            WRITE_STATE=True,
                                        )
                                    with K.Else():
                                        producer_pipeline(
                                            group,
                                            state_batch,
                                            dst_state_batch,
                                            READ_STATE=True,
                                            WRITE_STATE=False,
                                        )
                            with K.Else():
                                producer_pipeline(
                                    group,
                                    state_batch,
                                    dst_state_batch,
                                    READ_STATE=False,
                                    WRITE_STATE=False,
                                )
            else:
                with K.unroll(3) as arrive_stage:
                    _mbarrier_arrive(empty_barriers_buf.ptr_to([arrive_stage]))

                a_value: K.float32 = K.reinterpret("float32", _global_load_nc_u32(matrix_a, head))
                d_value = K.local_scalar("float32")
                K.assign(d_value, 0.0)
                if HAS_D:
                    K.assign(d_value, _load_weight_nc(d_weight, head, WEIGHT_DTYPE))
                dt_value = K.local_scalar("float32")
                K.assign(
                    dt_value,
                    _load_weight_nc(
                        dt, K.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE
                    ),
                )
                if HAS_DT_BIAS:
                    bias_value: K.float32 = _load_weight_nc(dt_bias, head, WEIGHT_DTYPE)
                    K.assign(dt_value, _add(dt_value, bias_value))
                with K.If(dt_softplus != 0):
                    with K.Then():
                        with K.If(dt_value <= K.float32(20.0)):
                            with K.Then():
                                exp_arg: K.float32 = _mul(dt_value, K.float32(_LOG2_E))
                                exp_value: K.float32 = _exp2(exp_arg)
                                one_plus_exp: K.float32 = _add(K.float32(1.0), exp_value)
                                log_value: K.float32 = _log2(one_plus_exp)
                                K.assign(dt_value, _mul(log_value, K.float32(_LN_2)))
                da_arg: K.float32 = _mul(a_value, dt_value)
                da_exp_arg: K.float32 = _mul(da_arg, K.float32(_LOG2_E))
                da_value: K.float32 = _exp2(da_exp_arg)
                lane_indicator: K.float32 = K.if_then_else(
                    lane == 0, K.float32(1.0), K.float32(0.0)
                )

                with K.If(state_batch != K.cast(pad_slot_id, "int64")):
                    with K.Then():
                        consumer_pipeline(
                            lane,
                            warp,
                            d_value,
                            dt_value,
                            da_value,
                            lane_indicator,
                            random_seed,
                            state_ptr_offset,
                            USE_STATE_CACHE=True,
                        )
                    with K.Else():
                        consumer_pipeline(
                            lane,
                            warp,
                            d_value,
                            dt_value,
                            da_value,
                            lane_indicator,
                            random_seed,
                            state_ptr_offset,
                            USE_STATE_CACHE=False,
                        )

                _mbarrier_arrive_wait(consumers_ready_buf.ptr_to([0]))
                row_d: K.int32 = warp * 32 + lane
                with K.If(row_d < DIM):
                    with K.Then():
                        out_value = K.local_scalar("float32")
                        K.assign(
                            out_value, K.reinterpret("float32", _shared_load_u32(s_out, row_d))
                        )
                        if HAS_Z:
                            z_value: K.float32 = _bf16_to_f32(_shared_load_u16(s_z, row_d))
                            neg_z: K.float32 = _sub(K.float32(0.0), z_value)
                            z_exp_arg: K.float32 = _mul(neg_z, K.float32(_LOG2_E))
                            exp_neg_z: K.float32 = _exp2(z_exp_arg)
                            denominator: K.float32 = _add(K.float32(1.0), exp_neg_z)
                            sigmoid_z: K.float32 = _div(K.float32(1.0), denominator)
                            silu_z: K.float32 = _mul(z_value, sigmoid_z)
                            K.assign(out_value, _mul(out_value, silu_z))
                        output_bits: K.uint16 = _f32_to_bf16(out_value)
                        K.evaluate(
                            K.ptx.st.global_.b16(
                                output.ptr_to(
                                    [
                                        K.cast(batch_i, "int64") * out_stride_batch
                                        + head * DIM
                                        + row_d
                                    ]
                                ),
                                output_bits,
                            )
                        )
                with K.If(
                    K.And(
                        K.And(SCALE_STATE, update_state != 0),
                        state_batch != K.cast(pad_slot_id, "int64"),
                    )
                ):
                    with K.Then():
                        with K.If(row_d < DIM):
                            with K.Then():
                                scale_bits: K.uint32 = _shared_load_u32(s_scale, row_d)
                                K.evaluate(
                                    K.ptx.st.global_.b32(
                                        state_scale.ptr_to([dst_scale_head_offset + row_d]),
                                        scale_bits,
                                    )
                                )

        with producer:
            run_role("producer")
        with consumer:
            run_role("consumer")

    return selective_state_update_stp_vertical.func


class _AlignedTensorMap:
    """Host storage for one 128-byte-aligned TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 128)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 127) & ~127)


def _encode_state_tensor_map(
    state: torch.Tensor, spec: dict[str, Any], state_stride: int
) -> _AlignedTensorMap:
    import tvm

    if int(state.data_ptr()) % 128:
        raise ValueError("vertical state TensorMap base must be 128-byte aligned")
    descriptor = _AlignedTensorMap()
    dstate = spec["DSTATE"]
    dim = spec["DIM"]
    nheads = spec["NHEADS"]
    state_slots = spec["STATE_ELEMENTS"] // state_stride
    state_bytes = spec["STATE_BYTES"]
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        "uint16" if spec["STATE_DTYPE"] == "int16" else spec["STATE_DTYPE"],
        4,
        ctypes.c_void_p(int(state.data_ptr())),
        dstate,
        dim,
        nheads,
        state_slots,
        dstate * state_bytes,
        dstate * dim * state_bytes,
        state_stride * state_bytes,
        dstate,
        16,
        1,
        1,
        1,
        1,
        1,
        1,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        0,  # CU_TENSOR_MAP_SWIZZLE_NONE
        2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create independent mutable TIRx/source cases and the state TensorMap."""
    case = _simple.prepare_data(**kwargs)
    spec = _specialization(kwargs)
    case["spec"] = spec
    case["tensor_state"] = _encode_state_tensor_map(
        case["tirx_state_raw"], spec, case["state_stride"]
    )
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    kwargs = case["kwargs"]
    spec = case["spec"]
    nheads, dim = spec["NHEADS"], spec["DIM"]
    ngroups, dstate = int(kwargs["ngroups"]), spec["DSTATE"]
    has_state_indices = bool(kwargs.get("has_state_indices", True))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    return (
        case["tensor_state"].ptr,
        case["tirx_state_raw"],
        case["tirx_scale_raw"],
        case["x"].reshape(-1),
        case["dt_base"].reshape(-1),
        case["matrix_a_base"],
        case["matrix_b"].reshape(-1),
        case["matrix_c"].reshape(-1),
        case["d_base"],
        case["z"].reshape(-1),
        case["bias_base"],
        (case["state_indices_flat"] if has_state_indices else case["dummy_index"]),
        (case["dst_indices_flat"] if has_dst_indices else case["dummy_index"]),
        case["seed"],
        case["tirx_output"].reshape(-1),
        case["state_stride"],
        case["scale_stride"] if spec["SCALE_STATE"] else 0,
        nheads * dim,
        nheads,
        ngroups * dstate,
        ngroups * dstate,
        nheads * dim,
        nheads * dim,
        case["state_index_stride"] if has_state_indices else 1,
        case["dst_index_stride"] if has_dst_indices else 0,
        nheads,
        ngroups,
        int(bool(kwargs.get("dt_softplus", False))),
        int(bool(kwargs.get("update_state", True))),
        case["pad_slot_id"],
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    kwargs = case["kwargs"]
    spec = case["spec"]
    oracle = _simple._load_oracle()
    state_view = _simple._view_state(case["reference_state_raw"], spec, case["state_stride"])
    state_scale = (
        _simple._view_scale(case["reference_scale_raw"], spec, case["scale_stride"])
        if spec["SCALE_STATE"]
        else None
    )
    source_out = case["reference_output"] if bool(kwargs.get("use_out_tensor", True)) else None
    result = oracle(
        state_view,
        case["x"],
        case["dt_view"],
        case["matrix_a_view"],
        case["matrix_b"],
        case["matrix_c"],
        case["d_view"],
        z=case["z"] if bool(kwargs.get("has_z", False)) else None,
        dt_bias=(case["bias_view"] if bool(kwargs.get("has_dt_bias", True)) else None),
        dt_softplus=bool(kwargs.get("dt_softplus", False)),
        state_batch_indices=(
            case["state_indices"] if bool(kwargs.get("has_state_indices", True)) else None
        ),
        dst_state_batch_indices=(
            case["dst_indices"] if bool(kwargs.get("has_dst_indices", False)) else None
        ),
        pad_slot_id=case["pad_slot_id"],
        state_scale=state_scale,
        out=source_out,
        disable_state_update=not bool(kwargs.get("update_state", True)),
        rand_seed=case["seed"] if spec["PHILOX_ROUNDS"] else None,
        philox_rounds=spec["PHILOX_ROUNDS"],
        algorithm="vertical",
    )
    if source_out is None:
        case["reference_output"].copy_(result)
    return result


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    _run_reference(case)
    torch.cuda.synchronize()
    _simple._assert_case_close(case)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = dict(prepared["config"])
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    rounds = int(kwargs.pop("rounds", 5))
    cooldown_s = float(kwargs.pop("cooldown_s", 1.0))
    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    args = _tirx_args(case)

    def source_builder():
        executable(*args)
        _run_reference(case)
        torch.cuda.synchronize()
        _simple._assert_case_close(case)
        for _ in range(2):
            _run_reference(case)
        torch.cuda.synchronize()

        def launch():
            _run_reference(case)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cuda": source_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
