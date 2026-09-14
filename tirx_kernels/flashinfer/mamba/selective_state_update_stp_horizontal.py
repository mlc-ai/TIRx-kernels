# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Integration scaffold for FlashInfer's STP producer-consumer horizontal kernel.

Upstream source: include/flashinfer/mamba/kernel_selective_state_update_stp.cuh.
"""

import ctypes
from typing import Any

import torch

import tirx_kernels.tirx_lite as txl

from . import selective_state_update_stp_simple as _simple

KERNEL_META = {
    "name": "selective_state_update_stp_horizontal",
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

_state_bits_to_f32 = _simple._state_bits_to_f32
_f32_to_state_bits = _simple._f32_to_state_bits
_load_two_byte_vector = _simple._load_two_byte_vector
_store_two_byte_vector = _simple._store_two_byte_vector

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"


def _load_weight(buffer, index, dtype: str):
    if dtype == "float32":
        gload_0 = txl.local_scalar("uint32")
        txl.ptx.ld.global_.b32(gload_0, buffer.ptr_to([index]))
        return txl.reinterpret("float32", gload_0)
    gload_1 = txl.local_scalar("uint16")
    txl.ptx.ld.global_.b16(gload_1, buffer.ptr_to([index]))
    bf16_f32_0 = txl.local_scalar("float32")
    txl.ptx.cvt.f32.bf16(bf16_f32_0, txl.cast(gload_1, "uint16"))
    return bf16_f32_0


def _lane_mask(raw_lane):
    return txl.cast(txl.bitwise_and(txl.cast(raw_lane, "uint32"), txl.uint32(31)), "int32")


def _copy_bf16x8_g2s(source, source_index, destination, destination_index):
    words = txl.alloc_local((4,), "uint32")
    txl.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], source.ptr_to([source_index]))
    txl.ptx.st.shared.v4.b32(
        destination.ptr_to([destination_index]), words[0], words[1], words[2], words[3]
    )


def _mbarrier_arrive_wait(barrier):
    token = txl.local_scalar("uint64")
    done = txl.local_scalar("uint32")
    txl.ptx.mbarrier.arrive.shared__cta.b64(token, barrier, txl.uint32(1))
    with txl.While(True):
        txl.ptx.mbarrier.try_wait.shared__cta.b64(done, barrier, token)
        with txl.If(done != txl.uint32(0)), txl.Then():
            txl.Break()


def _tma_g2s_horizontal(dst, tensor_state, column, head, state_batch, barrier):
    txl.ptx[_TMA_G2S_4D](
        dst,
        txl.address_of(tensor_state),
        txl.cast(column, "int32"),
        txl.int32(0),
        txl.cast(head, "int32"),
        txl.cast(state_batch, "int32"),
        barrier,
    )


def _tma_s2g_horizontal(src, tensor_state, column, head, state_batch):
    txl.ptx[_TMA_S2G_4D](
        txl.address_of(tensor_state),
        txl.cast(column, "int32"),
        txl.int32(0),
        txl.cast(head, "int32"),
        txl.cast(state_batch, "int32"),
        src,
    )


def _philox4x32_horizontal(random_words, random_seed, random_offset, *, PHILOX_ROUNDS):
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
        next_c1 = txl.local_scalar("uint32", init=old_c2 * txl.uint32(0xCD9E8D57))
        next_c3 = txl.local_scalar("uint32", init=old_c0 * txl.uint32(0xD2511F53))
        txl.assign(c0, next_c0)
        txl.assign(c1, next_c1)
        txl.assign(c2, next_c2)
        txl.assign(c3, next_c3)
        txl.assign(k0, k0 + txl.uint32(0x9E3779B9))
        txl.assign(k1, k1 + txl.uint32(0xBB67AE85))
    txl.ptx.mov.b32(random_words[0], c0)
    txl.ptx.mov.b32(random_words[1], c1)
    txl.ptx.mov.b32(random_words[2], c2)
    txl.ptx.mov.b32(random_words[3], c3)


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


# Performance rows vary every source branch or compile-time specialization that
# is meaningful for the explicit horizontal oracle.  Batch=1 and nullable-D are
# kept in CONFIGS as correctness-only rows.
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
    _case("b64_h64_d64_s128_r8_philox10", state_dtype="float16", philox_rounds=10, seed=42),
    _case(
        "b64_h64_d64_s64_r8_philox10", dstate=64, state_dtype="float16", philox_rounds=10, seed=42
    ),
]


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
]


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    batch = int(kwargs["batch"])
    nheads = int(kwargs["nheads"])
    dim = int(kwargs["dim"])
    dstate = int(kwargs["dstate"])
    ngroups = int(kwargs["ngroups"])
    state_dtype = str(kwargs["state_dtype"])
    weight_dtype = str(kwargs["weight_dtype"])
    index_dtype = str(kwargs["index_dtype"])
    state_stride_factor = int(kwargs.get("state_stride_factor", 1))
    has_dst_indices = bool(kwargs.get("has_dst_indices", False))
    if str(kwargs.get("input_dtype", "bfloat16")) != "bfloat16":
        raise ValueError("horizontal STP is scoped to bfloat16 input")
    if str(kwargs.get("matrix_a_dtype", "float32")) != "float32":
        raise ValueError("horizontal STP is scoped to float32 matrix A")
    if state_dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("horizontal STP supports bfloat16, float16, or float32 state")
    if weight_dtype not in ("float32", "bfloat16"):
        raise ValueError("horizontal STP supports float32 or bfloat16 weights")
    if index_dtype not in ("int32", "int64"):
        raise ValueError("horizontal STP supports int32 or int64 indices")
    if dim not in (64, 128):
        raise ValueError("horizontal STP requires dim in {64, 128}")
    if dstate not in (64, 96, 128, 256):
        raise ValueError("horizontal STP requires dstate in {64, 96, 128, 256}")
    if nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by ngroups")
    heads_group_ratio = nheads // ngroups
    if heads_group_ratio not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("horizontal STP group ratio must be a supported power of two")
    if state_stride_factor < 1:
        raise ValueError("state_stride_factor must be positive")

    state_bytes = 4 if state_dtype == "float32" else 2
    stage_cols = 64 // state_bytes
    if dstate % stage_cols:
        raise ValueError("dstate must be divisible by the horizontal stage width")
    total_stages = dstate // stage_cols
    num_stages = min(4, total_stages)
    consumer_warps = (dim // 64) * 4
    state_values_per_bank = 4 // state_bytes
    state_stage_values = dim * stage_cols
    state_stage_bytes = state_stage_values * state_bytes
    items_per_thread = stage_cols // 2

    state_slots = max((2 if has_dst_indices else 1) * batch + 8, 16)
    state_stride = nheads * dim * dstate * state_stride_factor
    index_elements = batch * (2 if int(kwargs.get("index_rank", 1)) == 2 else 1)
    philox_rounds = int(kwargs.get("philox_rounds", 0))
    if philox_rounds not in (0, 10):
        raise ValueError("horizontal stochastic rounding supports philox_rounds in {0, 10}")
    if philox_rounds and (state_dtype != "float16" or dstate not in (64, 128)):
        raise ValueError("philox10 is scoped to float16 state with dstate 64 or 128")

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
        "HEADS_GROUP_RATIO": heads_group_ratio,
        "CONSUMER_WARPS": consumer_warps,
        "NUM_WARPS": consumer_warps + 1,
        "MIN_BLOCKS_PER_SM": 1 if dim == 128 else 9,
        "STATE_DTYPE": state_dtype,
        "WEIGHT_DTYPE": weight_dtype,
        "INDEX_DTYPE": index_dtype,
        "STATE_ELEMENTS": state_slots * state_stride,
        "SCALE_ELEMENTS": 1,
        "X_ELEMENTS": batch * nheads * dim,
        "DT_ELEMENTS": batch * nheads,
        "BC_ELEMENTS": batch * ngroups * dstate,
        "INDEX_ELEMENTS": max(index_elements, 1),
        "HAS_STATE_INDICES": bool(kwargs.get("has_state_indices", True)),
        "HAS_DST_INDICES": has_dst_indices,
        "HAS_Z": bool(kwargs.get("has_z", False)),
        "HAS_D": bool(kwargs.get("has_d", True)),
        "HAS_DT_BIAS": bool(kwargs.get("has_dt_bias", True)),
        "SCALE_STATE": False,
        "PHILOX_ROUNDS": philox_rounds,
        "STATE_BYTES": state_bytes,
        "STATE_VALUES_PER_BANK": state_values_per_bank,
        "STAGE_COLS": stage_cols,
        "NUM_STAGES": num_stages,
        "ITEMS_PER_THREAD": items_per_thread,
        "STATE_STAGE_VALUES": state_stage_values,
        "STATE_STAGE_BYTES": state_stage_bytes,
    }


def get_kernel(**kwargs: Any):
    """Build the K entry for one horizontal specialization."""
    spec = _specialization(kwargs)
    DIM = spec["DIM"]
    DSTATE = spec["DSTATE"]
    HEADS_GROUP_RATIO = spec["HEADS_GROUP_RATIO"]
    STATE_DTYPE = spec["STATE_DTYPE"]
    WEIGHT_DTYPE = spec["WEIGHT_DTYPE"]
    INDEX_DTYPE = spec["INDEX_DTYPE"]
    HAS_STATE_INDICES = spec["HAS_STATE_INDICES"]
    HAS_DST_INDICES = spec["HAS_DST_INDICES"]
    HAS_Z = spec["HAS_Z"]
    HAS_D = spec["HAS_D"]
    HAS_DT_BIAS = spec["HAS_DT_BIAS"]
    PHILOX_ROUNDS = spec["PHILOX_ROUNDS"]
    STATE_BYTES = spec["STATE_BYTES"]
    STATE_VALUES_PER_BANK = spec["STATE_VALUES_PER_BANK"]
    STAGE_COLS = spec["STAGE_COLS"]
    NUM_STAGES = spec["NUM_STAGES"]
    ITEMS_PER_THREAD = spec["ITEMS_PER_THREAD"]
    STATE_STAGE_VALUES = spec["STATE_STAGE_VALUES"]
    STATE_STAGE_BYTES = spec["STATE_STAGE_BYTES"]

    @txl.kernel(
        warps=spec["NUM_WARPS"],
        arch="sm_100a",
        min_blocks_per_sm=spec["MIN_BLOCKS_PER_SM"],
        grid=(spec["BATCH"], spec["NHEADS"]),
    )
    def selective_state_update_stp_horizontal(
        tensor_state: txl.TensorMap,
        state: txl.gptr[spec["STATE_DTYPE"]],
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
        x_stride_batch: txl.i64,
        dt_stride_batch: txl.i64,
        b_stride_batch: txl.i64,
        c_stride_batch: txl.i64,
        z_stride_batch: txl.i64,
        out_stride_batch: txl.i64,
        state_indices_stride_batch: txl.i64,
        dst_indices_stride_batch: txl.i64,
        dt_softplus: txl.i32,
        update_state: txl.i32,
        pad_slot_id: txl.i32,
    ):
        batch_i, head = txl.cta_id()
        smem = txl.smem_pool()
        s_state = smem.alloc(
            (spec["NUM_STAGES"] * spec["STATE_STAGE_VALUES"],), spec["STATE_DTYPE"], align=128
        )
        s_b = smem.alloc((spec["DSTATE"],), txl.bf16, align=16)
        s_c = smem.alloc((spec["DSTATE"],), txl.bf16, align=16)
        empty = txl.MBarrier(smem, spec["NUM_STAGES"])
        full = txl.MBarrier(smem, spec["NUM_STAGES"])
        consumers_ready = txl.MBarrier(smem, 1)
        arrival_count = 1 + spec["CONSUMER_WARPS"] * 32
        empty.init(arrival_count)
        full.init(arrival_count)
        consumers_ready.init(spec["CONSUMER_WARPS"] * 32)
        txl.cuda.cta_sync()

        roles = txl.specialize()
        consumer = roles.role("consumer", warps=list(range(spec["CONSUMER_WARPS"])))
        producer = roles.role("producer", warps=[spec["CONSUMER_WARPS"]])

        empty_barriers_buf = empty.buf
        full_barriers_buf = full.buf
        consumers_ready_buf = consumers_ready.buf

        def producer_pipeline(
            state_batch, dst_state_batch, READ_STATE: txl.constexpr, WRITE_STATE: txl.constexpr
        ):
            with txl.unroll(NUM_STAGES) as fill_iter:
                fill_stage: txl.int32 = fill_iter
                fill_column: txl.int32 = fill_iter * STAGE_COLS
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([fill_stage]))
                if READ_STATE:
                    _tma_g2s_horizontal(
                        s_state.ptr_to([fill_stage * STATE_STAGE_VALUES]),
                        tensor_state,
                        fill_column,
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

            with txl.unroll(DSTATE // STAGE_COLS - NUM_STAGES) as steady_iter:
                steady_stage: txl.int32 = (NUM_STAGES + steady_iter) % NUM_STAGES
                read_column: txl.int32 = (NUM_STAGES + steady_iter) * STAGE_COLS
                write_column: txl.int32 = steady_iter * STAGE_COLS
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([steady_stage]))
                if READ_STATE or WRITE_STATE:
                    txl.ptx.fence.proxy.async_.shared__cta()
                    if WRITE_STATE:
                        _tma_s2g_horizontal(
                            s_state.ptr_to([steady_stage * STATE_STAGE_VALUES]),
                            tensor_state,
                            write_column,
                            head,
                            dst_state_batch,
                        )
                        txl.ptx.cp.async_.bulk.commit_group()
                        txl.ptx.cp.async_.bulk.wait_group.read(0)
                    if READ_STATE:
                        _tma_g2s_horizontal(
                            s_state.ptr_to([steady_stage * STATE_STAGE_VALUES]),
                            tensor_state,
                            read_column,
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

            with txl.unroll(NUM_STAGES) as drain_iter:
                drain_stage: txl.int32 = (
                    NUM_STAGES + (DSTATE // STAGE_COLS - NUM_STAGES) + drain_iter
                ) % NUM_STAGES
                write_column: txl.int32 = (
                    DSTATE // STAGE_COLS - NUM_STAGES + drain_iter
                ) * STAGE_COLS
                _mbarrier_arrive_wait(empty_barriers_buf.ptr_to([drain_stage]))
                if WRITE_STATE:
                    txl.ptx.fence.proxy.async_.shared__cta()
                    _tma_s2g_horizontal(
                        s_state.ptr_to([drain_stage * STATE_STAGE_VALUES]),
                        tensor_state,
                        write_column,
                        head,
                        dst_state_batch,
                    )
                    txl.ptx.cp.async_.bulk.commit_group()
                    txl.ptx.cp.async_.bulk.wait_group.read(0)

        def dispatch_producer(state_batch, dst_state_batch):
            read_state: txl.bool = state_batch != txl.cast(pad_slot_id, "int64")
            with txl.If(read_state):
                with txl.Then():
                    with txl.If(update_state != 0):
                        with txl.Then():
                            producer_pipeline(
                                state_batch, dst_state_batch, READ_STATE=True, WRITE_STATE=True
                            )
                        with txl.Else():
                            producer_pipeline(
                                state_batch, dst_state_batch, READ_STATE=True, WRITE_STATE=False
                            )
                with txl.Else():
                    producer_pipeline(
                        state_batch, dst_state_batch, READ_STATE=False, WRITE_STATE=False
                    )

        def consumer_pipeline(
            out_accum,
            d,
            member,
            row_group,
            a_value,
            dt_value,
            x_value,
            random_seed,
            state_ptr_offset,
            USE_STATE_CACHE: txl.constexpr,
        ):
            mul_0 = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](mul_0, a_value, dt_value)
            a_dt: txl.float32 = mul_0
            mul_1 = txl.local_scalar("float32")
            txl.ptx["mul.ftz.f32"](mul_1, a_dt, txl.float32(_LOG2_E))
            a_dt_exp_arg: txl.float32 = mul_1
            exp2_0 = txl.local_scalar("float32")
            txl.ptx["ex2.approx.ftz.f32"](exp2_0, a_dt_exp_arg)
            d_a: txl.float32 = exp2_0
            padded_state_d_a = txl.local_scalar("float32", init=0.0)
            if not USE_STATE_CACHE:
                txl.ptx["mul.ftz.f32"](padded_state_d_a, d_a, txl.float32(0.0))

            out_value = txl.local_scalar("float32", init=0.0)
            random_words = txl.alloc_local((4,), "uint32")
            i_begin = txl.local_scalar("int32", init=0)
            state_pipe = txl.PipelineState(NUM_STAGES, phase=0)
            with txl.While(i_begin < DSTATE):
                _mbarrier_arrive_wait(full_barriers_buf.ptr_to([state_pipe.stage]))
                with txl.unroll(ITEMS_PER_THREAD // STATE_VALUES_PER_BANK) as item_iter:
                    item: txl.int32 = item_iter * STATE_VALUES_PER_BANK
                    base_column: txl.int32 = item + member * ITEMS_PER_THREAD
                    sequence_index: txl.int32 = row_group * STAGE_COLS + base_column
                    bank_cycle: txl.int32 = (sequence_index // STATE_VALUES_PER_BANK) // 32
                    ii: txl.int32 = (base_column + STATE_VALUES_PER_BANK * bank_cycle) % STAGE_COLS
                    state_column: txl.int32 = i_begin + ii
                    state_index: txl.int32 = (
                        state_pipe.stage * STATE_STAGE_VALUES + d * STAGE_COLS + ii
                    )

                    if STATE_BYTES == 2:
                        r_state = _load_two_byte_vector(
                            s_state, state_index, STATE_VALUES_PER_BANK, "shared"
                        )
                        b_bits = _load_two_byte_vector(
                            s_b, state_column, STATE_VALUES_PER_BANK, "shared"
                        )
                        c_bits = _load_two_byte_vector(
                            s_c, state_column, STATE_VALUES_PER_BANK, "shared"
                        )
                        with txl.If(txl.And(PHILOX_ROUNDS > 0, item_iter % 2 == 0)), txl.Then():
                            random_offset: txl.int64 = state_ptr_offset + txl.cast(
                                d * DSTATE + state_column, "int64"
                            )
                            _philox4x32_horizontal(
                                random_words,
                                random_seed,
                                random_offset,
                                PHILOX_ROUNDS=PHILOX_ROUNDS,
                            )
                        sr_raw = txl.alloc_local((STATE_VALUES_PER_BANK,), "uint32")
                        with txl.unroll(STATE_VALUES_PER_BANK) as e:
                            state_value = txl.local_scalar("float32", init=0.0)
                            if USE_STATE_CACHE:
                                txl.assign(state_value, _state_bits_to_f32(r_state[e], STATE_DTYPE))
                            bf16_f32_1 = txl.local_scalar("float32")
                            txl.ptx.cvt.f32.bf16(bf16_f32_1, txl.cast(b_bits[e], "uint16"))
                            b_value: txl.float32 = bf16_f32_1
                            bf16_f32_2 = txl.local_scalar("float32")
                            txl.ptx.cvt.f32.bf16(bf16_f32_2, txl.cast(c_bits[e], "uint16"))
                            c_value: txl.float32 = bf16_f32_2
                            mul_2 = txl.local_scalar("float32")
                            txl.ptx["mul.ftz.f32"](mul_2, b_value, dt_value)
                            d_b: txl.float32 = mul_2
                            state_d_a = txl.local_scalar("float32", init=padded_state_d_a)
                            if USE_STATE_CACHE:
                                txl.ptx["mul.ftz.f32"](state_d_a, state_value, d_a)
                            fma_0 = txl.local_scalar("float32")
                            txl.ptx["fma.rn.ftz.f32"](fma_0, x_value, d_b, state_d_a)
                            new_state: txl.float32 = fma_0
                            if PHILOX_ROUNDS > 0:
                                random13: txl.uint32 = txl.bitwise_and(
                                    random_words[(item + e) % 4], txl.uint32(0x1FFF)
                                )
                                txl.ptx.cvt.rs.f16x2.f32(
                                    sr_raw[e], txl.float32(0.0), new_state, random13
                                )
                            else:
                                txl.ptx.mov.b16(
                                    r_state[e], _f32_to_state_bits(new_state, STATE_DTYPE)
                                )
                            txl.ptx["fma.rn.ftz.f32"](out_value, c_value, new_state, out_value)

                        if PHILOX_ROUNDS > 0:
                            prmt_0 = txl.local_scalar("uint32")
                            txl.ptx["prmt.b32"](
                                prmt_0,
                                txl.cast(sr_raw[0], "uint32"),
                                txl.cast(sr_raw[1], "uint32"),
                                txl.uint32(0x5410),
                            )
                            packed_state: txl.uint32 = prmt_0
                            txl.ptx.st.shared.b32(s_state.ptr_to([state_index]), packed_state)
                        else:
                            _store_two_byte_vector(
                                s_state, state_index, r_state, STATE_VALUES_PER_BANK, "shared"
                            )
                    else:
                        sload_0 = txl.local_scalar("uint32")
                        txl.ptx.ld.shared.b32(sload_0, s_state.ptr_to([state_index]))
                        state_word: txl.uint32 = sload_0
                        state_value = txl.local_scalar("float32", init=0.0)
                        if USE_STATE_CACHE:
                            txl.assign(state_value, txl.reinterpret("float32", state_word))
                        sload_1 = txl.local_scalar("uint16")
                        txl.ptx.ld.shared.b16(sload_1, s_b.ptr_to([state_column]))
                        bf16_f32_3 = txl.local_scalar("float32")
                        txl.ptx.cvt.f32.bf16(bf16_f32_3, txl.cast(sload_1, "uint16"))
                        b_value: txl.float32 = bf16_f32_3
                        sload_2 = txl.local_scalar("uint16")
                        txl.ptx.ld.shared.b16(sload_2, s_c.ptr_to([state_column]))
                        bf16_f32_4 = txl.local_scalar("float32")
                        txl.ptx.cvt.f32.bf16(bf16_f32_4, txl.cast(sload_2, "uint16"))
                        c_value: txl.float32 = bf16_f32_4
                        mul_3 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_3, b_value, dt_value)
                        d_b: txl.float32 = mul_3
                        state_d_a = txl.local_scalar("float32", init=padded_state_d_a)
                        if USE_STATE_CACHE:
                            txl.ptx["mul.ftz.f32"](state_d_a, state_value, d_a)
                        fma_1 = txl.local_scalar("float32")
                        txl.ptx["fma.rn.ftz.f32"](fma_1, x_value, d_b, state_d_a)
                        new_state: txl.float32 = fma_1
                        txl.ptx["fma.rn.ftz.f32"](out_value, c_value, new_state, out_value)
                        txl.ptx.st.shared.b32(
                            s_state.ptr_to([state_index]), txl.reinterpret("uint32", new_state)
                        )

                txl.ptx.mbarrier.arrive.shared__cta.b64(
                    empty_barriers_buf.ptr_to([state_pipe.stage]), txl.uint32(1)
                )
                txl.assign(i_begin, i_begin + STAGE_COLS)
                state_pipe.advance()

            txl.assign(out_accum[0], out_value)

        def run_role(ROLE: txl.constexpr):
            # TIRX_TRANSCRIBE_START selective_state_update_stp_horizontal

            flat_tid: txl.int32 = txl.tid_in_role()
            random_seed = txl.local_scalar("int64", init=0)
            if PHILOX_ROUNDS > 0:
                _t1 = txl.local_scalar("int64")
                txl.ptx.ld.global_.b64(_t1, rand_seed.ptr_to([0]))
                txl.assign(random_seed, _t1)

            lane: txl.int32 = _lane_mask(flat_tid)
            warp: txl.int32 = flat_tid >> 5
            group: txl.int32 = head // HEADS_GROUP_RATIO

            state_batch = txl.local_scalar("int64")
            if HAS_STATE_INDICES:
                if INDEX_DTYPE == "int32":
                    _t2 = txl.local_scalar("int64")
                    txl.ptx.ld.global_.s32(
                        _t2, state_indices.ptr_to([(batch_i * state_indices_stride_batch)])
                    )
                    txl.assign(state_batch, _t2)
                else:
                    _t3 = txl.local_scalar("int64")
                    txl.ptx.ld.global_.b64(
                        _t3, state_indices.ptr_to([(batch_i * state_indices_stride_batch)])
                    )
                    txl.assign(state_batch, _t3)
            else:
                txl.assign(state_batch, txl.cast(batch_i, "int64"))

            dst_state_batch_i32 = txl.local_scalar("int32", init=0)
            dst_state_batch_i64 = txl.local_scalar("int64", init=state_batch)
            if HAS_DST_INDICES:
                if INDEX_DTYPE == "int32":
                    _t4 = txl.local_scalar("int32")
                    txl.ptx.ld.global_.b32(
                        _t4, dst_indices.ptr_to([(batch_i * dst_indices_stride_batch)])
                    )
                    txl.assign(dst_state_batch_i32, _t4)
                else:
                    _t5 = txl.local_scalar("int64")
                    txl.ptx.ld.global_.b64(
                        _t5, dst_indices.ptr_to([(batch_i * dst_indices_stride_batch)])
                    )
                    txl.assign(dst_state_batch_i64, _t5)

            state_ptr_offset: txl.int64 = state_batch * state_stride_batch + txl.cast(
                head * DIM * DSTATE, "int64"
            )

            txl.keep_alive(state.data)

            if ROLE == "producer":
                with txl.If(txl.cuda.elect_sync()), txl.Then():
                    if HAS_DST_INDICES and INDEX_DTYPE == "int32":
                        dispatch_producer(state_batch, dst_state_batch_i32)
                    else:
                        dispatch_producer(state_batch, dst_state_batch_i64)
            else:
                with txl.unroll(NUM_STAGES) as arrive_stage:
                    txl.ptx.mbarrier.arrive.shared__cta.b64(
                        empty_barriers_buf.ptr_to([arrive_stage]), txl.uint32(1)
                    )

                gload_2 = txl.local_scalar("uint32")
                txl.ptx.ld.global_.b32(gload_2, matrix_a.ptr_to([head]))
                a_value: txl.float32 = txl.reinterpret("float32", gload_2)
                d_value = txl.local_scalar("float32", init=0.0)
                if HAS_D:
                    txl.assign(d_value, _load_weight(d_weight, head, WEIGHT_DTYPE))
                dt_value = txl.local_scalar("float32")
                txl.assign(
                    dt_value,
                    _load_weight(
                        dt, txl.cast(batch_i, "int64") * dt_stride_batch + head, WEIGHT_DTYPE
                    ),
                )
                if HAS_DT_BIAS:
                    bias_value: txl.float32 = _load_weight(dt_bias, head, WEIGHT_DTYPE)
                    txl.ptx["add.ftz.f32"](dt_value, dt_value, bias_value)
                with txl.If(dt_softplus != 0), txl.Then():
                    with txl.If(dt_value <= txl.float32(20.0)), txl.Then():
                        mul_4 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_4, dt_value, txl.float32(_LOG2_E))
                        dt_exp_arg: txl.float32 = mul_4
                        exp2_1 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_1, dt_exp_arg)
                        dt_exp: txl.float32 = exp2_1
                        add_0 = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](add_0, txl.float32(1.0), dt_exp)
                        dt_one_plus: txl.float32 = add_0
                        log2_0 = txl.local_scalar("float32")
                        txl.ptx["lg2.approx.ftz.f32"](log2_0, dt_one_plus)
                        dt_log2: txl.float32 = log2_0
                        txl.ptx["mul.ftz.f32"](dt_value, dt_log2, txl.float32(_LN_2))

                with txl.If(warp == 0):
                    with txl.Then():
                        b_column = txl.local_scalar("int32", init=lane * 8)
                        with txl.While(b_column < DSTATE):
                            _copy_bf16x8_g2s(
                                matrix_b,
                                txl.cast(batch_i, "int64") * b_stride_batch
                                + group * DSTATE
                                + b_column,
                                s_b,
                                b_column,
                            )
                            txl.assign(b_column, b_column + 32 * 8)
                    with txl.Else():
                        with txl.If(warp == 1), txl.Then():
                            c_column = txl.local_scalar("int32", init=lane * 8)
                            with txl.While(c_column < DSTATE):
                                _copy_bf16x8_g2s(
                                    matrix_c,
                                    txl.cast(batch_i, "int64") * c_stride_batch
                                    + group * DSTATE
                                    + c_column,
                                    s_c,
                                    c_column,
                                )
                                txl.assign(c_column, c_column + 32 * 8)

                row_group: txl.int32 = lane % 16
                member: txl.int32 = lane // 16
                d: txl.int32 = warp * 16 + row_group
                gload_3 = txl.local_scalar("uint16")
                txl.ptx.ld.global_.b16(
                    gload_3, x.ptr_to([txl.cast(batch_i, "int64") * x_stride_batch + head * DIM + d])
                )
                bf16_f32_5 = txl.local_scalar("float32")
                txl.ptx.cvt.f32.bf16(bf16_f32_5, txl.cast(gload_3, "uint16"))
                x_value: txl.float32 = bf16_f32_5
                z_value = txl.local_scalar("float32", init=0.0)
                if HAS_Z:
                    gload_4 = txl.local_scalar("uint16")
                    txl.ptx.ld.global_.b16(
                        gload_4,
                        z.ptr_to([txl.cast(batch_i, "int64") * z_stride_batch + head * DIM + d]),
                    )
                    bf16_f32_6 = txl.local_scalar("float32")
                    txl.ptx.cvt.f32.bf16(bf16_f32_6, txl.cast(gload_4, "uint16"))
                    txl.assign(z_value, bf16_f32_6)

                _mbarrier_arrive_wait(consumers_ready_buf.ptr_to([0]))
                out_accum = txl.alloc_local((1,), "float32")
                txl.assign(out_accum[0], 0.0)
                with txl.If(state_batch != txl.cast(pad_slot_id, "int64")):
                    with txl.Then():
                        consumer_pipeline(
                            out_accum,
                            d,
                            member,
                            row_group,
                            a_value,
                            dt_value,
                            x_value,
                            random_seed,
                            state_ptr_offset,
                            USE_STATE_CACHE=True,
                        )
                    with txl.Else():
                        consumer_pipeline(
                            out_accum,
                            d,
                            member,
                            row_group,
                            a_value,
                            dt_value,
                            x_value,
                            random_seed,
                            state_ptr_offset,
                            USE_STATE_CACHE=False,
                        )

                out_value = txl.local_scalar("float32", init=out_accum[0])
                txl.ptx["add.ftz.f32"](
                    out_value, out_value, _simple._shfl_down_f32(out_value, txl.int32(16))
                )
                with txl.If(member == 0), txl.Then():
                    txl.ptx["fma.rn.ftz.f32"](out_value, d_value, x_value, out_value)
                    if HAS_Z:
                        sub_0 = txl.local_scalar("float32")
                        txl.ptx["sub.ftz.f32"](sub_0, txl.float32(0.0), z_value)
                        neg_z: txl.float32 = sub_0
                        mul_5 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_5, neg_z, txl.float32(_LOG2_E))
                        z_exp_arg: txl.float32 = mul_5
                        exp2_2 = txl.local_scalar("float32")
                        txl.ptx["ex2.approx.ftz.f32"](exp2_2, z_exp_arg)
                        exp_neg_z: txl.float32 = exp2_2
                        add_1 = txl.local_scalar("float32")
                        txl.ptx["add.ftz.f32"](add_1, txl.float32(1.0), exp_neg_z)
                        denominator: txl.float32 = add_1
                        div_0 = txl.local_scalar("float32")
                        txl.ptx["div.approx.ftz.f32"](div_0, txl.float32(1.0), denominator)
                        sigmoid_z: txl.float32 = div_0
                        mul_6 = txl.local_scalar("float32")
                        txl.ptx["mul.ftz.f32"](mul_6, z_value, sigmoid_z)
                        silu_z: txl.float32 = mul_6
                        txl.ptx["mul.ftz.f32"](out_value, out_value, silu_z)
                    f32_bf16_0 = txl.local_scalar("uint16")
                    txl.ptx.cvt.rn.bf16.f32(f32_bf16_0, out_value)
                    output_bits: txl.uint16 = f32_bf16_0
                    txl.ptx.st.global_.b16(
                        output.ptr_to(
                            [txl.cast(batch_i, "int64") * out_stride_batch + head * DIM + d]
                        ),
                        output_bits,
                    )

        with consumer:
            run_role("consumer")
        with producer:
            run_role("producer")

    return selective_state_update_stp_horizontal.func


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
        raise ValueError("horizontal state TensorMap base must be 128-byte aligned")
    descriptor = _AlignedTensorMap()
    dstate = spec["DSTATE"]
    dim = spec["DIM"]
    nheads = spec["NHEADS"]
    state_slots = spec["STATE_ELEMENTS"] // state_stride
    state_bytes = spec["STATE_BYTES"]
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        spec["STATE_DTYPE"],
        4,
        ctypes.c_void_p(int(state.data_ptr())),
        dstate,
        dim,
        nheads,
        state_slots,
        dstate * state_bytes,
        dstate * dim * state_bytes,
        state_stride * state_bytes,
        spec["STAGE_COLS"],
        dim,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    return descriptor


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create independent mutable TIRx/source cases and the state TensorMap."""
    case = _simple.prepare_data(**kwargs)
    spec = _specialization(kwargs)
    case["spec"] = spec
    for name, tensor in (
        ("x", case["x"]),
        ("z", case["z"]),
        ("B", case["matrix_b"]),
        ("C", case["matrix_c"]),
    ):
        if int(tensor.data_ptr()) % 16:
            raise ValueError(f"horizontal {name} base must be 16-byte aligned")
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
        case["x"].reshape(-1),
        case["dt_base"].reshape(-1),
        case["matrix_a_base"],
        case["matrix_b"].reshape(-1),
        case["matrix_c"].reshape(-1),
        case["d_base"],
        case["z"].reshape(-1),
        case["bias_base"],
        case["state_indices_flat"] if has_state_indices else case["dummy_index"],
        case["dst_indices_flat"] if has_dst_indices else case["dummy_index"],
        case["seed"],
        case["tirx_output"].reshape(-1),
        case["state_stride"],
        nheads * dim,
        nheads,
        ngroups * dstate,
        ngroups * dstate,
        nheads * dim,
        nheads * dim,
        case["state_index_stride"] if has_state_indices else 1,
        case["dst_index_stride"] if has_dst_indices else 0,
        int(bool(kwargs.get("dt_softplus", False))),
        int(bool(kwargs.get("update_state", True))),
        case["pad_slot_id"],
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    kwargs = case["kwargs"]
    spec = case["spec"]
    oracle = _simple._load_oracle()
    state_view = _simple._view_state(case["reference_state_raw"], spec, case["state_stride"])
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
        dt_bias=case["bias_view"] if bool(kwargs.get("has_dt_bias", True)) else None,
        dt_softplus=bool(kwargs.get("dt_softplus", False)),
        state_batch_indices=(
            case["state_indices"] if bool(kwargs.get("has_state_indices", True)) else None
        ),
        dst_state_batch_indices=(
            case["dst_indices"] if bool(kwargs.get("has_dst_indices", False)) else None
        ),
        pad_slot_id=case["pad_slot_id"],
        out=source_out,
        disable_state_update=not bool(kwargs.get("update_state", True)),
        rand_seed=case["seed"] if spec["PHILOX_ROUNDS"] else None,
        philox_rounds=spec["PHILOX_ROUNDS"],
        algorithm="horizontal",
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
