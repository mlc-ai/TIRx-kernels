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

import tirx_kernels.tirx_lite as txl

from . import selective_state_update_stp_simple as _simple

KERNEL_META = {
    "name": "selective_state_update_stp_vertical",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}


_LOG2_E = _simple._LOG2_E
_LN_2 = _simple._LN_2
_FLT_LOWEST = _simple._FLT_LOWEST

_lane_mask = _simple._lane_mask
_state_bits_to_f32 = _simple._state_bits_to_f32
_f32_to_state_bits = _simple._f32_to_state_bits
_load_two_byte_vector = _simple._load_two_byte_vector
_store_two_byte_vector = _simple._store_two_byte_vector

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
_BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"


def _load_weight_nc(buffer, index, dtype: str):
    if dtype == "float32":
        _t1 = txl.local_scalar("uint32")
        txl.ptx.ld.global_.nc.b32(_t1, buffer.ptr_to([index]))
        return txl.reinterpret("float32", _t1)
    bf16_f32_0 = txl.local_scalar("float32")
    _t2 = txl.local_scalar("uint16")
    txl.ptx.ld.global_.nc.b16(_t2, buffer.ptr_to([index]))
    txl.ptx.cvt.f32.bf16(bf16_f32_0, txl.cast(_t2, "uint16"))
    return bf16_f32_0


def _mbarrier_arrive_wait(barrier):
    token = txl.local_scalar("uint64")
    done = txl.local_scalar("uint32")
    txl.ptx.mbarrier.arrive.shared__cta.b64(token, barrier, txl.uint32(1))
    with txl.While(True):
        txl.ptx.mbarrier.try_wait.shared__cta.b64(done, barrier, token)
        with txl.If(done != txl.uint32(0)), txl.Then():
            txl.Break()


def _tma_g2s(dst, tensor_state, d, head, batch, barrier):
    txl.ptx[_TMA_G2S_4D](
        dst,
        txl.address_of(tensor_state),
        txl.int32(0),
        txl.cast(d, "int32"),
        txl.cast(head, "int32"),
        txl.cast(batch, "int32"),
        barrier,
    )


def _tma_s2g(src, tensor_state, d, head, batch):
    txl.ptx[_TMA_S2G_4D](
        txl.address_of(tensor_state),
        txl.int32(0),
        txl.cast(d, "int32"),
        txl.cast(head, "int32"),
        txl.cast(batch, "int32"),
        src,
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
    c0 = txl.local_scalar("uint32", init=txl.cast(random_offset, "uint32"))
    c1 = txl.local_scalar(
        "uint32",
        init=txl.cast(txl.shift_right(txl.cast(random_offset, "uint64"), txl.uint64(32)), "uint32"),
    )
    c2 = txl.local_scalar("uint32", init=0)
    c3 = txl.local_scalar("uint32", init=0)
    k0 = txl.local_scalar("uint32", init=txl.cast(txl.reinterpret("uint64", random_seed), "uint32"))
    k1 = txl.local_scalar(
        "uint32",
        init=txl.cast(txl.shift_right(txl.reinterpret("uint64", random_seed), txl.uint64(32)), "uint32"),
    )
    with txl.unroll(PHILOX_ROUNDS) as _round:
        old_c0 = txl.local_scalar("uint32", init=c0)
        old_c2 = txl.local_scalar("uint32", init=c2)
        hi_b = txl.local_scalar("uint32")
        txl.ptx["mul.hi.u32"](hi_b, txl.uint32(0xCD9E8D57), old_c2)
        next_c0 = txl.local_scalar("uint32", init=txl.bitwise_xor(txl.bitwise_xor(hi_b, c1), k0))
        hi_a = txl.local_scalar("uint32")
        txl.ptx["mul.hi.u32"](hi_a, txl.uint32(0xD2511F53), old_c0)
        next_c2 = txl.local_scalar("uint32", init=txl.bitwise_xor(txl.bitwise_xor(hi_a, c3), k1))
        next_c1_s = txl.local_scalar("int32")
        txl.ptx["mul.lo.s32"](next_c1_s, txl.int32(-845247145), txl.reinterpret("int32", old_c2))
        next_c3_s = txl.local_scalar("int32")
        txl.ptx["mul.lo.s32"](next_c3_s, txl.int32(-766435501), txl.reinterpret("int32", old_c0))
        next_k0_s = txl.local_scalar("int32")
        txl.ptx["add.s32"](next_k0_s, txl.reinterpret("int32", k0), txl.int32(-1640531527))
        next_k1_s = txl.local_scalar("int32")
        txl.ptx["add.s32"](next_k1_s, txl.reinterpret("int32", k1), txl.int32(-1150833019))
        txl.assign(c0, next_c0)
        txl.assign(c1, txl.reinterpret("uint32", next_c1_s))
        txl.assign(c2, next_c2)
        txl.assign(c3, txl.reinterpret("uint32", next_c3_s))
        txl.assign(k0, txl.reinterpret("uint32", next_k0_s))
        txl.assign(k1, txl.reinterpret("uint32", next_k1_s))
    txl.ptx.mov.b32(random_words[0], c0)
    txl.ptx.mov.b32(random_words[1], c1)
    txl.ptx.mov.b32(random_words[2], c2)
    txl.ptx.mov.b32(random_words[3], c3)


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

    @txl.kernel(warps=5, arch="sm_100a", grid=(spec["BATCH"], spec["NHEADS"]))
    def selective_state_update_stp_vertical(
        tensor_state: txl.TensorMap,
        state: txl.gptr[spec["STATE_DTYPE"]],
        state_scale: txl.gptr[txl.f32],
        x: txl.gptr[txl.bf16],
        dt: txl.gptr[spec["WEIGHT_DTYPE"]],
        matrix_a: txl.gptr[txl.f32],
        matrix_b: txl.gptr[txl.bf16],
        matrix_c: txl.gptr[txl.bf16],
        d_weight: txl.gptr[spec["WEIGHT_DTYPE"]],
        z: txl.gptr[txl.bf16],
        dt_bias: txl.gptr[spec["WEIGHT_DTYPE"]],
        state_indices: txl.gptr[spec["INDEX_DTYPE"]],
        dst_indices: txl.gptr[spec["INDEX_DTYPE"]],
        rand_seed: txl.gptr[txl.i64],
        output: txl.gptr[txl.bf16],
        state_stride_batch: txl.i64,
        state_scale_stride_batch: txl.i64,
        x_stride_batch: txl.i64,
        dt_stride_batch: txl.i64,
        b_stride_batch: txl.i64,
        c_stride_batch: txl.i64,
        z_stride_batch: txl.i64,
        out_stride_batch: txl.i64,
        state_indices_stride_batch: txl.i64,
        dst_indices_stride_batch: txl.i64,
        nheads_runtime: txl.i32,
        ngroups_runtime: txl.i32,
        dt_softplus: txl.i32,
        update_state: txl.i32,
        pad_slot_id: txl.i32,
    ):
        batch_i, head = txl.cta_id()
        smem = txl.smem_pool()
        s_state = smem.alloc((3 * spec["STATE_STAGE_VALUES"],), spec["STATE_DTYPE"], align=128)
        s_x = smem.alloc((spec["DIM"],), txl.bf16, align=16)
        s_z = smem.alloc((spec["DIM"],), txl.bf16, align=16)
        s_b = smem.alloc((spec["DSTATE"],), txl.bf16, align=16)
        s_c = smem.alloc((spec["DSTATE"],), txl.bf16, align=16)
        s_out = smem.alloc((spec["DIM"],), txl.f32, align=4)
        s_scale = smem.alloc((spec["DIM"],), txl.f32, align=128) if spec["SCALE_STATE"] else s_out
        empty = txl.MBarrier(smem, 3)
        full = txl.MBarrier(smem, 3)
        consumers_ready = txl.MBarrier(smem, 1)
        empty.init(129)
        full.init(129)
        consumers_ready.init(128)
        txl.cuda.cta_sync()

        roles = txl.specialize()
        consumer = roles.role("consumer", warps=range(4))
        producer = roles.role("producer", warps=[4])

        empty_barriers_buf = empty.buf
        full_barriers_buf = full.buf
        consumers_ready_buf = consumers_ready.buf

        def producer_pipeline(
            group, state_batch, dst_state_batch, READ_STATE: txl.constexpr, WRITE_STATE: txl.constexpr
        ):
            # Phase 1, stage 0: vector inputs and the first optional state tile share
            # one full barrier transaction, exactly as in producer_func_vertical.
            _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([0]))
            txl.ptx[_BULK_G2S](
                s_x.ptr_to([0]),
                x.ptr_to([(txl.cast(batch_i, "int64") * x_stride_batch + head * DIM)]),
                txl.uint32(DIM * 2),
                full_barriers_buf.ptr_to([0]),
            )
            txl.ptx[_BULK_G2S](
                s_b.ptr_to([0]),
                matrix_b.ptr_to([(txl.cast(batch_i, "int64") * b_stride_batch + group * DSTATE)]),
                txl.uint32(DSTATE * 2),
                full_barriers_buf.ptr_to([0]),
            )
            txl.ptx[_BULK_G2S](
                s_c.ptr_to([0]),
                matrix_c.ptr_to([(txl.cast(batch_i, "int64") * c_stride_batch + group * DSTATE)]),
                txl.uint32(DSTATE * 2),
                full_barriers_buf.ptr_to([0]),
            )
            if HAS_Z:
                txl.ptx[_BULK_G2S](
                    s_z.ptr_to([0]),
                    z.ptr_to([(txl.cast(batch_i, "int64") * z_stride_batch + head * DIM)]),
                    txl.uint32(DIM * 2),
                    full_barriers_buf.ptr_to([0]),
                )
            if SCALE_STATE:
                txl.ptx[_BULK_G2S](
                    s_scale.ptr_to([0]),
                    state_scale.ptr_to([(state_batch * state_scale_stride_batch + head * DIM)]),
                    txl.uint32(DIM * 4),
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
                txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                    full_barriers_buf.ptr_to([0]), txl.uint32(STATE_STAGE_BYTES + INPUT_BYTES)
                )
            else:
                txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                    full_barriers_buf.ptr_to([0]), txl.uint32(INPUT_BYTES)
                )

            # Phase 1, stages 1 and 2: state-only fill.
            with txl.unroll(1, 3) as fill_iter:
                fill_stage: txl.int32 = fill_iter
                fill_d: txl.int32 = fill_iter * 16
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
                    txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                        full_barriers_buf.ptr_to([fill_stage]), txl.uint32(STATE_STAGE_BYTES)
                    )
                else:
                    txl.ptx.mbarrier.arrive.shared__cta.b64(
                        full_barriers_buf.ptr_to([fill_stage]), txl.uint32(1)
                    )

            # Phase 2: every reused stage is stored before its next load.
            with txl.unroll(DIM // 16 - 3) as steady_iter:
                steady_stage: txl.int32 = (3 + steady_iter) % 3
                d_read: txl.int32 = (3 + steady_iter) * 16
                d_write: txl.int32 = steady_iter * 16
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([steady_stage]))
                if READ_STATE or WRITE_STATE:
                    txl.ptx.fence.proxy.async_.shared__cta()
                    if WRITE_STATE:
                        _tma_s2g(
                            s_state.ptr_to([steady_stage * STATE_STAGE_VALUES]),
                            tensor_state,
                            d_write,
                            head,
                            dst_state_batch,
                        )
                        txl.ptx.cp.async_.bulk.commit_group()
                        txl.ptx.cp.async_.bulk.wait_group.read(0)
                    if READ_STATE:
                        _tma_g2s(
                            s_state.ptr_to([steady_stage * STATE_STAGE_VALUES]),
                            tensor_state,
                            d_read,
                            head,
                            state_batch,
                            full_barriers_buf.ptr_to([steady_stage]),
                        )
                        txl.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                            full_barriers_buf.ptr_to([steady_stage]), txl.uint32(STATE_STAGE_BYTES)
                        )
                    else:
                        txl.ptx.mbarrier.arrive.shared__cta.b64(
                            full_barriers_buf.ptr_to([steady_stage]), txl.uint32(1)
                        )
                else:
                    txl.ptx.mbarrier.arrive.shared__cta.b64(
                        full_barriers_buf.ptr_to([steady_stage]), txl.uint32(1)
                    )

            # Phase 3: wait for and drain the final three shared state tiles.
            with txl.unroll(3) as drain_iter:
                drain_stage: txl.int32 = (DIM // 16 + drain_iter) % 3
                drain_d: txl.int32 = (DIM // 16 - 3 + drain_iter) * 16
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([drain_stage]))
                if WRITE_STATE:
                    txl.ptx.fence.proxy.async_.shared__cta()
                    _tma_s2g(
                        s_state.ptr_to([drain_stage * STATE_STAGE_VALUES]),
                        tensor_state,
                        drain_d,
                        head,
                        dst_state_batch,
                    )
                    txl.ptx.cp.async_.bulk.commit_group()
                    txl.ptx.cp.async_.bulk.wait_group.read(0)

        def consumer_pipeline(
            lane,
            warp,
            d_value,
            dt_value,
            da_value,
            lane_indicator,
            random_seed,
            state_ptr_offset,
            USE_STATE_CACHE: txl.constexpr,
        ):
            d_begin = txl.local_scalar("int32", init=0)
            state_pipe = txl.PipelineState(3, phase=0)
            with txl.While(d_begin < DIM):
                _mbarrier_arrive_wait(full_barriers_buf.ptr_to([state_pipe.stage]))
                with txl.unroll(4) as row_iter:
                    dd: txl.int32 = warp + row_iter * 4
                    row_d: txl.int32 = d_begin + dd
                    sload_0 = txl.local_scalar("uint16")
                    txl.ptx.ld.shared.b16(sload_0, s_x.ptr_to([row_d]))
                    bf16_f32_1 = txl.local_scalar("float32")
                    txl.ptx.cvt.f32.bf16(bf16_f32_1, txl.cast(sload_0, "uint16"))
                    x_value: txl.float32 = bf16_f32_1
                    mul_0 = txl.local_scalar("float32")
                    txl.ptx["mul.ftz.f32"](mul_0, d_value, x_value)
                    d_times_x: txl.float32 = mul_0
                    out_value = txl.local_scalar("float32")
                    txl.ptx["mul.ftz.f32"](out_value, d_times_x, lane_indicator)
                    decode_scale = txl.local_scalar("float32", init=1.0)
                    new_state_max = txl.local_scalar("float32", init=txl.float32(_FLT_LOWEST))
                    if SCALE_STATE:
                        sload_1 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_1, s_scale.ptr_to([row_d]))
                        txl.assign(decode_scale, txl.reinterpret("float32", sload_1))
                    new_states = txl.alloc_local((NEW_STATE_COUNT,), "float32")

                    with txl.serial(STATE_ITERATIONS) as state_iter:
                        state_i: txl.int32 = (state_iter * 32 + lane) * STATE_VALUES_PER_BANK
                        with txl.If(state_i < DSTATE), txl.Then():
                            state_index: txl.int32 = (
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
                                random_words = txl.alloc_local((4,), "uint32")
                                sr_raw = txl.alloc_local((STATE_VALUES_PER_BANK,), "uint32")
                                if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                                    random_offset: txl.uint64 = txl.cast(
                                        state_ptr_offset + row_d * DSTATE + state_i, "uint64"
                                    )
                                    _philox4x32(
                                        random_words,
                                        random_seed,
                                        random_offset,
                                        PHILOX_ROUNDS=PHILOX_ROUNDS,
                                    )
                                with txl.unroll(STATE_VALUES_PER_BANK) as e:
                                    state_value = txl.local_scalar("float32", init=0.0)
                                    if USE_STATE_CACHE:
                                        txl.assign(
                                            state_value, _state_bits_to_f32(r_state[e], STATE_DTYPE)
                                        )
                                        if SCALE_STATE:
                                            txl.ptx["mul.ftz.f32"](
                                                state_value, state_value, decode_scale
                                            )
                                    bf16_f32_2 = txl.local_scalar("float32")
                                    txl.ptx.cvt.f32.bf16(bf16_f32_2, txl.cast(b_bits[e], "uint16"))
                                    b_value: txl.float32 = bf16_f32_2
                                    bf16_f32_3 = txl.local_scalar("float32")
                                    txl.ptx.cvt.f32.bf16(bf16_f32_3, txl.cast(c_bits[e], "uint16"))
                                    c_value: txl.float32 = bf16_f32_3
                                    mul_1 = txl.local_scalar("float32")
                                    txl.ptx["mul.ftz.f32"](mul_1, b_value, dt_value)
                                    db_value: txl.float32 = mul_1
                                    mul_2 = txl.local_scalar("float32")
                                    txl.ptx["mul.ftz.f32"](mul_2, db_value, x_value)
                                    db_x: txl.float32 = mul_2
                                    fma_0 = txl.local_scalar("float32")
                                    txl.ptx["fma.rn.ftz.f32"](fma_0, state_value, da_value, db_x)
                                    new_state: txl.float32 = fma_0
                                    if SCALE_STATE:
                                        abs_0 = txl.local_scalar("float32")
                                        txl.ptx["abs.ftz.f32"](abs_0, new_state)
                                        magnitude: txl.float32 = abs_0
                                        txl.ptx["max.ftz.f32"](
                                            new_state_max, new_state_max, magnitude
                                        )
                                        txl.ptx.mov.b32(
                                            new_states[state_iter * STATE_VALUES_PER_BANK + e],
                                            new_state,
                                        )
                                    elif PHILOX_ROUNDS > 0:
                                        random13: txl.uint32 = txl.bitwise_and(
                                            random_words[e], txl.uint32(0x1FFF)
                                        )
                                        txl.ptx.cvt.rs.f16x2.f32(
                                            sr_raw[e], txl.float32(0.0), new_state, random13
                                        )
                                    else:
                                        txl.ptx.mov.b16(
                                            r_state[e], _f32_to_state_bits(new_state, STATE_DTYPE)
                                        )
                                    txl.ptx["fma.rn.ftz.f32"](
                                        out_value, new_state, c_value, out_value
                                    )

                                if not SCALE_STATE:
                                    if PHILOX_ROUNDS > 0:
                                        prmt_0 = txl.local_scalar("uint32")
                                        txl.ptx["prmt.b32"](
                                            prmt_0,
                                            txl.cast(sr_raw[0], "uint32"),
                                            txl.cast(sr_raw[1], "uint32"),
                                            txl.uint32(0x5410),
                                        )
                                        packed_sr: txl.uint32 = prmt_0
                                        txl.ptx.st.shared.b32(
                                            s_state.ptr_to([state_index]), packed_sr
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
                                sload_2 = txl.local_scalar("uint32")
                                txl.ptx.ld.shared.b32(sload_2, s_state.ptr_to([state_index]))
                                state_word: txl.uint32 = sload_2
                                state_value = txl.local_scalar("float32", init=0.0)
                                if USE_STATE_CACHE:
                                    txl.assign(state_value, txl.reinterpret("float32", state_word))
                                sload_3 = txl.local_scalar("uint16")
                                txl.ptx.ld.shared.b16(sload_3, s_b.ptr_to([state_i]))
                                bf16_f32_4 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_4, txl.cast(sload_3, "uint16"))
                                b_value: txl.float32 = bf16_f32_4
                                sload_4 = txl.local_scalar("uint16")
                                txl.ptx.ld.shared.b16(sload_4, s_c.ptr_to([state_i]))
                                bf16_f32_5 = txl.local_scalar("float32")
                                txl.ptx.cvt.f32.bf16(bf16_f32_5, txl.cast(sload_4, "uint16"))
                                c_value: txl.float32 = bf16_f32_5
                                mul_3 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](mul_3, b_value, dt_value)
                                db_value: txl.float32 = mul_3
                                mul_4 = txl.local_scalar("float32")
                                txl.ptx["mul.ftz.f32"](mul_4, db_value, x_value)
                                db_x: txl.float32 = mul_4
                                fma_1 = txl.local_scalar("float32")
                                txl.ptx["fma.rn.ftz.f32"](fma_1, state_value, da_value, db_x)
                                new_state: txl.float32 = fma_1
                                txl.ptx["fma.rn.ftz.f32"](out_value, new_state, c_value, out_value)
                                txl.ptx.st.shared.b32(
                                    s_state.ptr_to([state_index]),
                                    txl.reinterpret("uint32", new_state),
                                )

                    with txl.unroll(5) as delta_i:
                        delta: txl.int32 = txl.shift_right(txl.int32(16), delta_i)
                        txl.ptx["add.ftz.f32"](
                            out_value, out_value, _simple._shfl_down_f32(out_value, delta)
                        )
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.shared.b32(
                            s_out.ptr_to([row_d]), txl.reinterpret("uint32", out_value)
                        )

                    if SCALE_STATE and USE_STATE_CACHE:
                        with txl.unroll(5) as delta_i:
                            delta: txl.int32 = txl.shift_right(txl.int32(16), delta_i)
                            txl.ptx["max.ftz.f32"](
                                new_state_max,
                                new_state_max,
                                _simple._shfl_down_f32(new_state_max, delta),
                            )
                        # Unlike the simple kernel, the frozen vertical source has no
                        # standalone __syncwarp between max reduction and broadcast.
                        txl.assign(new_state_max, _simple._shfl_idx_f32(new_state_max, txl.int32(0)))
                        encode_scale = txl.local_scalar("float32", init=1.0)
                        with txl.If(new_state_max != txl.float32(0.0)), txl.Then():
                            txl.ptx["div.approx.ftz.f32"](
                                encode_scale, txl.float32(32767.0), new_state_max
                            )
                        rcp_0 = txl.local_scalar("float32")
                        txl.ptx["rcp.approx.ftz.f32"](rcp_0, encode_scale)
                        new_decode_scale: txl.float32 = rcp_0
                        with txl.serial(STATE_ITERATIONS) as state_iter:
                            state_i: txl.int32 = (state_iter * 32 + lane) * STATE_VALUES_PER_BANK
                            with txl.If(state_i < DSTATE), txl.Then():
                                quantized = txl.alloc_local((STATE_VALUES_PER_BANK,), "int32")
                                with txl.unroll(STATE_VALUES_PER_BANK) as e:
                                    mul_5 = txl.local_scalar("float32")
                                    txl.ptx["mul.ftz.f32"](
                                        mul_5,
                                        new_states[state_iter * STATE_VALUES_PER_BANK + e],
                                        encode_scale,
                                    )
                                    scaled: txl.float32 = mul_5
                                    max_0 = txl.local_scalar("float32")
                                    txl.ptx["max.ftz.f32"](max_0, scaled, txl.float32(-32767.0))
                                    clipped_low: txl.float32 = max_0
                                    min_0 = txl.local_scalar("float32")
                                    txl.ptx["min.ftz.f32"](min_0, clipped_low, txl.float32(32767.0))
                                    clipped: txl.float32 = min_0
                                    txl.ptx.cvt.rni.ftz.s32.f32(quantized[e], clipped)
                                prmt_1 = txl.local_scalar("uint32")
                                txl.ptx["prmt.b32"](
                                    prmt_1,
                                    txl.cast(txl.reinterpret("uint32", quantized[0]), "uint32"),
                                    txl.cast(txl.reinterpret("uint32", quantized[1]), "uint32"),
                                    txl.uint32(0x5410),
                                )
                                packed_i16: txl.uint32 = prmt_1
                                state_index: txl.int32 = (
                                    state_pipe.stage * STATE_STAGE_VALUES + dd * DSTATE + state_i
                                )
                                txl.ptx.st.shared.b32(s_state.ptr_to([state_index]), packed_i16)
                        with txl.If(lane == 0), txl.Then():
                            txl.ptx.st.shared.b32(
                                s_scale.ptr_to([row_d]), txl.reinterpret("uint32", new_decode_scale)
                            )

                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx.mbarrier.arrive.shared__cta.b64(
                    empty_barriers_buf.ptr_to([state_pipe.stage]), txl.uint32(1)
                )
                txl.assign(d_begin, d_begin + 16)
                state_pipe.advance()

        def run_role(ROLE: txl.constexpr):
            flat_tid: txl.int32 = txl.thread_id()
            # TIRX_TRANSCRIBE_START selective_state_update_stp_vertical

            random_seed = txl.local_scalar("int64", init=0)
            if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                txl.ptx.ld.global_.s64(random_seed, rand_seed.ptr_to([0]))

            lane: txl.int32 = _lane_mask(flat_tid)
            warp: txl.int32 = flat_tid >> 5
            group: txl.int32 = head // (nheads_runtime // ngroups_runtime)

            state_batch = txl.local_scalar("int64")
            if HAS_STATE_INDICES:
                if INDEX_DTYPE == "int32":
                    _t3 = txl.local_scalar("int32")
                    txl.ptx.ld.global_.nc.s32(
                        _t3, state_indices.ptr_to([(batch_i * state_indices_stride_batch)])
                    )
                    txl.assign(state_batch, txl.cast(_t3, "int64"))
                else:
                    _t4 = txl.local_scalar("int64")
                    txl.ptx.ld.global_.nc.s64(
                        _t4, state_indices.ptr_to([(batch_i * state_indices_stride_batch)])
                    )
                    txl.assign(state_batch, _t4)
            else:
                txl.assign(state_batch, txl.cast(batch_i, "int64"))

            dst_state_batch = txl.local_scalar("int64")
            if HAS_DST_INDICES:
                if INDEX_DTYPE == "int32":
                    _t5 = txl.local_scalar("int32")
                    txl.ptx.ld.global_.nc.s32(
                        _t5, dst_indices.ptr_to([(batch_i * dst_indices_stride_batch)])
                    )
                    txl.assign(dst_state_batch, txl.cast(_t5, "int64"))
                else:
                    _t6 = txl.local_scalar("int64")
                    txl.ptx.ld.global_.nc.s64(
                        _t6, dst_indices.ptr_to([(batch_i * dst_indices_stride_batch)])
                    )
                    txl.assign(dst_state_batch, _t6)
            else:
                txl.assign(dst_state_batch, state_batch)

            state_ptr_offset: txl.int64 = state_batch * state_stride_batch + txl.cast(
                head * DIM * DSTATE, "int64"
            )
            scale_head_offset: txl.int64 = state_batch * state_scale_stride_batch + txl.cast(
                head * DIM, "int64"
            )
            dst_scale_head_offset: txl.int64 = dst_state_batch * state_scale_stride_batch + txl.cast(
                head * DIM, "int64"
            )

            txl.keep_alive(state.data)

            if ROLE == "producer":
                read_state: txl.bool = state_batch != txl.cast(pad_slot_id, "int64")
                write_state: txl.bool = txl.And(read_state, update_state != 0)
                with txl.If(lane == 0), txl.Then():
                    with txl.If(read_state):
                        with txl.Then():
                            with txl.If(write_state):
                                with txl.Then():
                                    producer_pipeline(
                                        group,
                                        state_batch,
                                        dst_state_batch,
                                        READ_STATE=True,
                                        WRITE_STATE=True,
                                    )
                                with txl.Else():
                                    producer_pipeline(
                                        group,
                                        state_batch,
                                        dst_state_batch,
                                        READ_STATE=True,
                                        WRITE_STATE=False,
                                    )
                        with txl.Else():
                            producer_pipeline(
                                group,
                                state_batch,
                                dst_state_batch,
                                READ_STATE=False,
                                WRITE_STATE=False,
                            )
            else:
                with txl.unroll(3) as arrive_stage:
                    txl.ptx.mbarrier.arrive.shared__cta.b64(
                        empty_barriers_buf.ptr_to([arrive_stage]), txl.uint32(1)
                    )

                _t7 = txl.local_scalar("uint32")
                txl.ptx.ld.global_.nc.b32(_t7, matrix_a.ptr_to([head]))
                a_value: txl.float32 = txl.reinterpret("float32", _t7)
                d_value = txl.local_scalar("float32", init=0.0)
                if HAS_D:
                    txl.assign(d_value, _load_weight_nc(d_weight, head, WEIGHT_DTYPE))
                dt_value = txl.local_scalar("float32")
                txl.assign(
                    dt_value,
                    _load_weight_nc(
                        dt, txl.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE
                    ),
                )
                if HAS_DT_BIAS:
                    bias_value: txl.float32 = _load_weight_nc(dt_bias, head, WEIGHT_DTYPE)
                    txl.ptx["add.ftz.f32"](dt_value, dt_value, bias_value)
                with txl.If(dt_softplus != 0), txl.Then():
                    with txl.If(dt_value <= txl.float32(20.0)), txl.Then():
                        mul_6 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_6, dt_value, txl.float32(_LOG2_E))
                        exp_arg: txl.float32 = mul_6
                        exp2_0 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_0, exp_arg)
                        exp_value: txl.float32 = exp2_0
                        add_0 = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](add_0, txl.float32(1.0), exp_value)
                        one_plus_exp: txl.float32 = add_0
                        log2_0 = txl.local_scalar("float32")
                        txl.ptx["lg2.approx.ftz.f32"](log2_0, one_plus_exp)
                        log_value: txl.float32 = log2_0
                        txl.ptx["mul.ftz.f32"](dt_value, log_value, txl.float32(_LN_2))
                mul_7 = txl.local_scalar("float32")
                txl.ptx["mul.ftz.f32"](mul_7, a_value, dt_value)
                da_arg: txl.float32 = mul_7
                mul_8 = txl.local_scalar("float32")
                txl.ptx["mul.ftz.f32"](mul_8, da_arg, txl.float32(_LOG2_E))
                da_exp_arg: txl.float32 = mul_8
                exp2_1 = txl.local_scalar("float32")
                txl.ptx["ex2.approx.ftz.f32"](exp2_1, da_exp_arg)
                da_value: txl.float32 = exp2_1
                lane_indicator: txl.float32 = txl.if_then_else(
                    lane == 0, txl.float32(1.0), txl.float32(0.0)
                )

                with txl.If(state_batch != txl.cast(pad_slot_id, "int64")):
                    with txl.Then():
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
                    with txl.Else():
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
                row_d: txl.int32 = warp * 32 + lane
                with txl.If(row_d < DIM), txl.Then():
                    out_value = txl.local_scalar("float32")
                    sload_5 = txl.local_scalar("uint32")
                    txl.ptx.ld.shared.b32(sload_5, s_out.ptr_to([row_d]))
                    txl.assign(out_value, txl.reinterpret("float32", sload_5))
                    if HAS_Z:
                        sload_6 = txl.local_scalar("uint16")
                        txl.ptx.ld.shared.b16(sload_6, s_z.ptr_to([row_d]))
                        bf16_f32_6 = txl.local_scalar("float32")
                        txl.ptx.cvt.f32.bf16(bf16_f32_6, txl.cast(sload_6, "uint16"))
                        z_value: txl.float32 = bf16_f32_6
                        sub_0 = txl.local_scalar("float32")
                        txl.ptx["sub.ftz.f32"](sub_0, txl.float32(0.0), z_value)
                        neg_z: txl.float32 = sub_0
                        mul_9 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_9, neg_z, txl.float32(_LOG2_E))
                        z_exp_arg: txl.float32 = mul_9
                        exp2_2 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_2, z_exp_arg)
                        exp_neg_z: txl.float32 = exp2_2
                        add_1 = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](add_1, txl.float32(1.0), exp_neg_z)
                        denominator: txl.float32 = add_1
                        div_0 = txl.local_scalar("float32")
                        txl.ptx["div.approx.ftz.f32"](div_0, txl.float32(1.0), denominator)
                        sigmoid_z: txl.float32 = div_0
                        mul_10 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_10, z_value, sigmoid_z)
                        silu_z: txl.float32 = mul_10
                        txl.ptx["mul.ftz.f32"](out_value, out_value, silu_z)
                    f32_bf16_0 = txl.local_scalar("uint16")
                    txl.ptx.cvt.rn.bf16.f32(f32_bf16_0, out_value)
                    output_bits: txl.uint16 = f32_bf16_0
                    txl.ptx.st.global_.b16(
                        output.ptr_to(
                            [txl.cast(batch_i, "int64") * out_stride_batch + head * DIM + row_d]
                        ),
                        output_bits,
                    )
                with (
                    txl.If(
                        txl.And(
                            txl.And(SCALE_STATE, update_state != 0),
                            state_batch != txl.cast(pad_slot_id, "int64"),
                        )
                    ),
                    txl.Then(),
                ):
                    with txl.If(row_d < DIM), txl.Then():
                        sload_7 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_7, s_scale.ptr_to([row_d]))
                        scale_bits: txl.uint32 = sload_7
                        txl.ptx.st.global_.b32(
                            state_scale.ptr_to([dst_scale_head_offset + row_d]), scale_bits
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
