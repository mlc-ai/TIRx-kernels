# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ d7e4dba3),
# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 two-CTA FlashAttention backward kernel using raw CUDA/PTX wrappers.

This is a TIRx port of Dao-AILab/flash-attention's
``flash_attn/cute/flash_bwd_sm100.py`` at commit
``d7e4dba3e568106b0f1b6323b07c1272f53679b3`` (2026-08-05).  The retained
specialization is dense MHA with equal query/key lengths, fp16 inputs,
head-dimension 128, and causal or noncausal masking.

The core uses public ``T.cuda`` and ``T.ptx`` wrappers for Tensor Maps, TMA,
DSMEM, mbarriers, tcgen05/TMEM, and scalar/vector PTX.  It intentionally does
not depend on TIRx tile primitives.
"""

import copy
import math
from functools import cache

import torch

import tvm
from tirx_kernels.flashmla.utils._ir_builder import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.ir.type import PointerType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.layout import ComposeLayout, S, TileLayout
from tvm.tirx.script.builder.ir import name_meta_class_value

_BUILDER_MISSING = object()


def _builder_enter(frame):
    frames = frame.frames if hasattr(frame, "frames") else [frame]
    prim_func = next(
        candidate
        for candidate in reversed(IRBuilder.current().frames)
        if type(candidate).__name__ == "PrimFuncFrame"
    )
    for item in frames:
        prim_func.add_callback(lambda item=item: item.__exit__(None, None, None))
        item.__enter__()


def _builder_emit(value):
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if isinstance(value, IRBuilderFrame) or (
        hasattr(value, "frames") and hasattr(value, "__enter__")
    ):
        _builder_enter(value)
    elif tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)
    elif isinstance(value, int | bool):
        T.evaluate(tvm.tirx.const(value))


def _builder_alloc_scalar(name, dtype):
    scalar = T.local_scalar(dtype)
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def _builder_scalar(name, value, dtype):
    scalar = _builder_alloc_scalar(name, dtype)
    T.buffer_store(scalar.buffer, value, scalar.indices)
    return scalar


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_assign(name, value, previous=_BUILDER_MISSING):
    if isinstance(value, I.meta_var):
        return value.value
    if previous is not _BUILDER_MISSING:
        if isinstance(previous, T.scalar_wrapper | tvm.tirx.expr.BufferLoad):
            target = previous.scalar if isinstance(previous, T.scalar_wrapper) else previous
            T.buffer_store(target.buffer, value, target.indices)
            return target
        if (
            is_buffer_var(previous)
            and len(previous.ty.shape) == 1
            and bool(previous.ty.shape[0] == 1)
        ):
            try:
                T.buffer_store(previous, value, [0])
                return previous
            except TypeError:
                pass
    if isinstance(value, PipelineState):
        IRBuilder.name(f"{name}_stage", value.stage.buffer)
        IRBuilder.name(f"{name}_phase", value.phase.buffer)
        return value
    if getattr(type(value), "_is_meta_class", False):
        name_meta_class_value(name, value)
        return value
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _builder_assign(f"{name}_{index}", item)
        return value
    if is_buffer_var(value) or isinstance(value, IterVar | Layout):
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Var):
        if isinstance(value.ty, PointerType):
            return _builder_bind(name, value, value.ty)
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Expr) and isinstance(getattr(value, "ty", None), PointerType):
        return _builder_bind(name, value, value.ty)
    if isinstance(value, tvm.ir.Expr) and tvm.ir.is_prim_expr(value):
        return _builder_scalar(name, value, str(value.ty.dtype))
    return value


def _builder_assign_many(names, values, previous):
    return tuple(
        _builder_assign(name, value, old) for name, value, old in zip(names, values, previous)
    )


IKET_EVENT_NAMES = (
    "dq-reduce",
    "ds-exchange",
    "dkv-epilogue",
    "mma-dk",
    "mma-dp",
    "mma-dq-alias-wait",
    "mma-dq-issue",
    "mma-dq-ready-wait",
    "mma-dv",
    "mma-s",
    "softmax-ds",
    "softmax-p",
    "tma-wait-a",
    "tma-wait-dpsum",
    "tma-wait-lse",
    "tma-wait-q",
    "tma-wait-qcol",
    "tma-prefetch",
    "dq-reduce-stage",
)


def tma_shared_layout(dtype, shape):
    """Construct a public 128-byte-swizzled tcgen05/TMA SMEM layout."""
    bits = tvm.DataType(dtype).bits
    per_element = (128 // bits).bit_length() - 1
    swizzle_len = 3
    atom_len = 3
    period = 1 << (per_element + swizzle_len + atom_len)
    atom_shape = [1] * (len(shape) - 2) + [8, 1024 // bits]
    layout = ComposeLayout(per_element, swizzle_len, atom_len, TileLayout(S[(period,)]))
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


# ---------------------------------------------------------------------------
# Preprocessing kernels
# ---------------------------------------------------------------------------


def build_preprocess(B, S, H, D):
    """Build dPsum/LSE-log2 and clear dQ accumulation in one pass.

    A 256-thread block covers 128 rows. Sixteen lanes cooperatively load each
    row, reduce with width-16 shuffles, and clear the matching dQ slice.  This
    matches the official kernel's copy topology: sixteen rows are processed in
    parallel and each thread owns 64 elements from each input tile.
    """
    if S % 128:
        raise ValueError("the SM100 backward preprocess requires seq_len divisible by 128")
    THREADS_PER_ROW = 16
    ELEMS_PER_THREAD = D // THREADS_PER_ROW
    BLOCK = 256
    ROWS_PER_WAVE = BLOCK // THREADS_PER_ROW
    ROWS_PER_BLOCK = 128
    ROW_ITERS = ROWS_PER_BLOCK // ROWS_PER_WAVE
    NBLK = S // ROWS_PER_BLOCK
    LOG2_E = math.log2(math.e)

    def dot_f16x8(lhs, rhs):
        lhs_words = T.alloc_local((4,), "uint32")
        rhs_words = T.alloc_local((4,), "uint32")
        lhs_halves = T.alloc_local((2,), "uint16")
        rhs_halves = T.alloc_local((2,), "uint16")
        lhs_value = T.alloc_local((2,), "float32")
        rhs_value = T.alloc_local((2,), "float32")
        result = T.alloc_local((1,), "float32")
        T.evaluate(
            T.ptx.ld.global_.nc.v4.b32(lhs_words[0], lhs_words[1], lhs_words[2], lhs_words[3], lhs)
        )
        T.evaluate(
            T.ptx.ld.global_.nc.v4.b32(rhs_words[0], rhs_words[1], rhs_words[2], rhs_words[3], rhs)
        )
        T.buffer_store(result, T.float32(0), 0)
        for pair in range(4):
            T.evaluate(T.ptx.mov.b32(lhs_halves[0], lhs_halves[1], lhs_words[pair]))
            for element in range(2):
                T.evaluate(T.ptx.cvt.f32.f16(lhs_value[element], lhs_halves[element]))
            T.evaluate(T.ptx.mov.b32(rhs_halves[0], rhs_halves[1], rhs_words[pair]))
            for element in range(2):
                T.evaluate(T.ptx.cvt.f32.f16(rhs_value[element], rhs_halves[element]))
            for element in range(2):
                # CUDA's default fast-math path lowers the original fmaf chain
                # with FTZ; spelling it here preserves the final instruction schedule.
                T.evaluate(
                    T.ptx.fma.rn.ftz.f32(
                        result[0], lhs_value[element], rhs_value[element], result[0]
                    )
                )
        return result[0]

    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("preprocess_kernel")
            dO_buf = T.arg("dO_buf", T.Buffer((B, S, H, D), "float16"))
            O_buf = T.arg("O_buf", T.Buffer((B, S, H, D), "float16"))
            LSE_buf = T.arg("LSE_buf", T.Buffer((B, H, S), "float32"))
            dpsum_out = T.arg("dpsum_out", T.Buffer((B, H, S), "float32"))
            LSE_log2_out = T.arg("LSE_log2_out", T.Buffer((B, H, S), "float32"))
            dQ_accum = T.arg("dQ_accum", T.Buffer((B, H, S, D), "float32"))
            T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
            with T.thread_binding(NBLK, thread="blockIdx.x") as bx:
                IRBuilder.name("bx", bx)
                with T.thread_binding(H, thread="blockIdx.y") as by:
                    IRBuilder.name("by", by)
                    with T.thread_binding(B, thread="blockIdx.z") as bz:
                        IRBuilder.name("bz", bz)
                        with T.thread_binding(BLOCK, thread="threadIdx.x") as tx:
                            IRBuilder.name("tx", tx)
                            col_in_row = _builder_scalar(
                                "col_in_row", tx % THREADS_PER_ROW, "int32"
                            )
                            row_in_wave = _builder_scalar(
                                "row_in_wave", tx // THREADS_PER_ROW, "int32"
                            )
                            d_start = _builder_scalar(
                                "d_start", col_in_row * ELEMS_PER_THREAD, "int32"
                            )

                            # Match the official dependency shape: start the
                            # independent LSE load before the O/dO dot products
                            # and retain it until the final scalar conversion.
                            # This lets its GMEM latency overlap the eight row
                            # iterations below.
                            lse_for_log2 = _builder_scalar("lse_for_log2", T.float32(0), "float32")
                            with T.If(tx < ROWS_PER_BLOCK):
                                with T.Then():
                                    lse_s = _builder_scalar(
                                        "lse_s", bx * ROWS_PER_BLOCK + tx, "int32"
                                    )
                                    T.buffer_store(
                                        lse_for_log2.buffer,
                                        LSE_buf[bz, by, lse_s],
                                        lse_for_log2.indices,
                                    )

                            with T.unroll(ROW_ITERS) as row_iter:
                                IRBuilder.name("row_iter", row_iter)
                                s = _builder_scalar(
                                    "s",
                                    bx * ROWS_PER_BLOCK + row_iter * ROWS_PER_WAVE + row_in_wave,
                                    "int32",
                                )
                                acc = _builder_scalar(
                                    "acc",
                                    dot_f16x8(
                                        T.address_of(dO_buf[bz, s, by, d_start]),
                                        T.address_of(O_buf[bz, s, by, d_start]),
                                    ),
                                    "float32",
                                )
                                with T.unroll(ELEMS_PER_THREAD // 4) as chunk:
                                    IRBuilder.name("chunk", chunk)
                                    with T.vectorized(4) as d:
                                        IRBuilder.name("d", d)
                                        T.buffer_store(
                                            dQ_accum,
                                            T.float32(0),
                                            [bz, by, s, d_start + chunk * 4 + d],
                                        )

                                for delta in (8, 4, 2, 1):
                                    T.buffer_store(
                                        acc.buffer,
                                        acc
                                        + T.cuda.__shfl_xor_sync(
                                            T.uint32(0xFFFFFFFF), acc, delta, THREADS_PER_ROW
                                        ),
                                        acc.indices,
                                    )
                                with T.If(col_in_row == 0):
                                    with T.Then():
                                        T.buffer_store(dpsum_out, acc, [bz, by, s])

                            # LSE conversion has one independent scalar per row;
                            # spread the 128 rows across 128 threads instead of
                            # serializing eight conversions on each row leader.
                            with T.If(tx < ROWS_PER_BLOCK):
                                with T.Then():
                                    lse_s = _builder_scalar(
                                        "lse_s", bx * ROWS_PER_BLOCK + tx, "int32"
                                    )
                                    T.buffer_store(
                                        LSE_log2_out,
                                        T.if_then_else(
                                            lse_for_log2 == T.float32(-float("inf")),
                                            T.float32(0),
                                            lse_for_log2 * T.float32(LOG2_E),
                                        ),
                                        [bz, by, lse_s],
                                    )

    mod = tvm.IRModule({"main": builder.get()})
    from tirx_kernels.runner import cuda_target

    return tvm.compile(mod, target=cuda_target())


def build_cast_f32_to_f16(B, S, H, D, scale):
    """Scale and transpose the head-major dQ accumulation to fp16."""
    GROUP_WIDTH = 4
    GROUPS_PER_THREAD = 4
    BLOCK = 256
    GROUPS_PER_BLOCK = BLOCK * GROUPS_PER_THREAD
    num_groups = B * S * H * (D // GROUP_WIDTH)
    NBLK = (num_groups + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK

    def scale_cast_f32x4_f16x4(dst_ptr, src_ptr, scale_value):
        source = T.alloc_local((4,), "float32")
        scaled = T.alloc_local((4,), "float32")
        packed = T.alloc_local((2,), "uint32")
        T.evaluate(T.ptx.ld.global_.v4.f32(source[0], source[1], source[2], source[3], src_ptr))
        for element in range(4):
            T.evaluate(T.ptx.mul.rn.f32(scaled[element], source[element], scale_value))
        T.evaluate(T.ptx.cvt.rn.f16x2.f32(packed[0], scaled[1], scaled[0]))
        T.evaluate(T.ptx.cvt.rn.f16x2.f32(packed[1], scaled[3], scaled[2]))
        return T.ptx.st.global_.v2.b32(dst_ptr, packed[0], packed[1])

    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("cast_kernel")
            src = T.arg("src", T.Buffer((B, H, S, D), "float32"))
            dst = T.arg("dst", T.Buffer((B, S, H, D), "float16"))
            T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
            with T.thread_binding(NBLK, thread="blockIdx.x") as bx:
                IRBuilder.name("bx", bx)
                with T.thread_binding(BLOCK, thread="threadIdx.x") as tx:
                    IRBuilder.name("tx", tx)
                    with T.unroll(GROUPS_PER_THREAD) as e:
                        IRBuilder.name("e", e)
                        group = _builder_scalar(
                            "group", bx * GROUPS_PER_BLOCK + e * BLOCK + tx, "int32"
                        )
                        with T.If(group < num_groups):
                            with T.Then():
                                d_group = _builder_scalar(
                                    "d_group", group % (D // GROUP_WIDTH), "int32"
                                )
                                h = _builder_scalar("h", group // (D // GROUP_WIDTH) % H, "int32")
                                s = _builder_scalar(
                                    "s", group // (D // GROUP_WIDTH * H) % S, "int32"
                                )
                                b = _builder_scalar(
                                    "b", group // (D // GROUP_WIDTH * H * S), "int32"
                                )
                                d = _builder_scalar("d", d_group * GROUP_WIDTH, "int32")
                                # The raw tcgen05 dQ accumulator uses the physical
                                # 128x128 C-fragment bit layout.  Decode that internal
                                # layout while producing the public sequence-major dQ.
                                s_in_block = _builder_scalar("s_in_block", s % 128, "int32")
                                src_s = _builder_scalar(
                                    "src_s",
                                    s // 128 * 128
                                    + T.bitwise_and(T.shift_right(s_in_block, 5), 1)
                                    + T.shift_left(T.bitwise_and(T.shift_right(d, 6), 1), 1)
                                    + T.shift_left(T.bitwise_and(T.shift_right(d, 2), 15), 2)
                                    + T.shift_left(
                                        T.bitwise_and(T.shift_right(s_in_block, 6), 1), 6
                                    ),
                                    "int32",
                                )
                                src_d = _builder_scalar(
                                    "src_d", T.shift_left(T.bitwise_and(s_in_block, 31), 2), "int32"
                                )
                                _builder_emit(
                                    scale_cast_f32x4_f16x4(
                                        T.address_of(dst[b, s, h, d]),
                                        T.address_of(src[b, h, src_s, src_d]),
                                        T.float32(scale),
                                    )
                                )

    mod = tvm.IRModule({"main": builder.get()})
    from tirx_kernels.runner import cuda_target

    return tvm.compile(mod, target=cuda_target())


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


def build_kernel(
    BATCH: int,
    HEADS_PER_BATCH: int,
    SEQ_LEN: int,
    HEAD_DIM: int = 128,
    *,
    causal: bool = False,
    attention_scale: float | None = None,
    sm_count: int = 148,
):
    if HEAD_DIM != 128:
        raise ValueError("the SM100 2-CTA backward kernel currently requires head_dim=128")
    if SEQ_LEN % 256:
        raise ValueError("the SM100 2-CTA backward kernel requires seq_len divisible by 256")
    if sm_count < 2:
        raise ValueError("the SM100 2-CTA backward kernel requires at least two SMs")
    f16 = tvm.DataType("float16")
    f32 = tvm.DataType("float32")

    # Leave the first KiB to the barriers allocated before the matrix payloads.
    # TCGEN descriptors derive their address field from an allocated shared view
    # below; this is a pool-relative choice, not an architectural shared address.
    POOL_Q_ROW = 1024
    MATRIX_DESC_F16_SS_LDO_1024 = 0x4000404004000000
    MATRIX_DESC_F16_SS_LDO_512 = 0x4000404002000000
    MATRIX_DESC_F16_TS = 0x4000404000000000

    def matrix_desc_from_anchor(layout_bits, anchor_start, byte_delta):
        # layout_bits deliberately has a zero start-address field.  Authority
        # comes from anchor_start, which is derived from a real shared view.
        start = T.bitwise_and(anchor_start + T.cast(byte_delta // 16, "uint64"), T.uint64(0x3FFF))
        return T.bitwise_or(T.uint64(layout_bits), start)

    def copy_128b(dst, value):
        # The dialect takes the payload as one 128-bit register; callers pass a
        # uint128 view element rather than a pointer into their local buffer.
        T.evaluate(T.ptx.st.weak.shared__cta.b128(dst, value))

    def pointer_offset(ptr, offset):
        return T.ptr_byte_offset(ptr, offset * 2, "float16")

    def pointer_offset_f32(ptr, offset):
        return T.ptr_byte_offset(ptr, offset * 4, "float32")

    # kind::f16 = fp16 A/B into an fp32 accumulator. The .ss and .ts table
    # entries share this chain and are told apart by the A operand's dtype:
    # u64 is a shared-memory descriptor, u32 a TMEM address.
    _MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
    # cta_group::2 takes an 8-lane disable-output-lane vector; nothing is
    # disabled here, but the operands are not optional.
    _MMA_KEEP_ALL_LANES = (0, 0, 0, 0, 0, 0, 0, 0)

    def mma_ss_one(d, accumulate, a_desc, b_desc, instruction):
        T.evaluate(
            T.ptx[_MMA_F16](
                T.uint32(d),
                T.uint64(a_desc),
                T.uint64(b_desc),
                T.uint32(instruction),
                *_MMA_KEEP_ALL_LANES,
                T.ptx.pred(accumulate),
            )
        )

    def mma_ss8(d, accumulate, a_base, b_base):
        mma_ss_one(d, accumulate, a_base, b_base, 270532624)
        mma_ss_one(d, 1, a_base + 2, b_base + 2, 270532624)
        mma_ss_one(d, 1, a_base + 4, b_base + 4, 270532624)
        mma_ss_one(d, 1, a_base + 6, b_base + 6, 270532624)
        mma_ss_one(d, 1, a_base + 1024, b_base + 512, 270532624)
        mma_ss_one(d, 1, a_base + 1026, b_base + 514, 270532624)
        mma_ss_one(d, 1, a_base + 1028, b_base + 516, 270532624)
        mma_ss_one(d, 1, a_base + 1030, b_base + 518, 270532624)

    def mma_ts_one(d, a, accumulate, b_desc):
        T.evaluate(
            T.ptx[_MMA_F16](
                T.uint32(d),
                T.uint32(a),
                T.uint64(b_desc),
                T.uint32(270598160),
                *_MMA_KEEP_ALL_LANES,
                T.ptx.pred(accumulate),
            )
        )

    def mma_ts8(d, a, accumulate, b_base):
        mma_ts_one(d, a, accumulate, b_base)
        mma_ts_one(d, a + 8, 1, b_base + 128)
        mma_ts_one(d, a + 16, 1, b_base + 256)
        mma_ts_one(d, a + 24, 1, b_base + 384)
        mma_ts_one(d, a + 32, 1, b_base + 512)
        mma_ts_one(d, a + 40, 1, b_base + 640)
        mma_ts_one(d, a + 48, 1, b_base + 768)
        mma_ts_one(d, a + 56, 1, b_base + 896)

    def mma_s(d, accumulate, desc_k_row, desc_q_row):
        mma_ss8(d, accumulate, desc_k_row, desc_q_row)

    def mma_dp(d, accumulate, desc_v_row, desc_do_row):
        mma_ss8(d, accumulate, desc_v_row, desc_do_row)

    def mma_dv(d, a, accumulate, desc_do_col):
        mma_ts8(d, a, accumulate, desc_do_col)

    def mma_dk(d, a, accumulate, desc_q_col):
        mma_ts8(d, a, accumulate, desc_q_col)

    def mma_dq(d, accumulate, desc_ds_exch, desc_k_col):
        mma_ss_one(d, accumulate, desc_ds_exch, desc_k_col, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 128, desc_k_col + 128, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 256, desc_k_col + 256, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 384, desc_k_col + 384, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 512, desc_k_col + 512, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 640, desc_k_col + 640, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 768, desc_k_col + 768, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 896, desc_k_col + 896, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1024, desc_k_col + 1024, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1152, desc_k_col + 1152, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1280, desc_k_col + 1280, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1408, desc_k_col + 1408, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1536, desc_k_col + 1536, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1664, desc_k_col + 1664, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1792, desc_k_col + 1792, 136413200)
        mma_ss_one(d, 1, desc_ds_exch + 1920, desc_k_col + 1920, 136413200)

    def cast_f32x2_to_f16x2(dst, src):
        T.evaluate(T.cuda.float22half2(dst, src))

    def fma_scale_sub_f32x2(scores, scale, lse):
        neg_lse = T.alloc_local((1,), "uint64")
        result = T.alloc_local((1,), "uint64")
        sign_mask = T.bitwise_or(
            T.shift_left(T.uint64(0x80000000), T.uint64(32)), T.uint64(0x80000000)
        )
        T.evaluate(T.ptx.xor.b64(neg_lse[0], lse, sign_mask))
        T.evaluate(T.ptx.fma.rn.f32x2(result[0], scores, scale, neg_lse[0]))
        return result[0]

    # The dialect takes one composed TMEM address instead of a base plus
    # row/col; get_tmem_addr packs them the same way the old operands did.
    def _tmem_addr(base, row, col):
        return T.cuda.get_tmem_addr(T.uint32(base), row, col)

    def tmem_load_64(dst, dst_offset, base, row, col):
        return T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"](
            *[dst[dst_offset + i] for i in range(64)], _tmem_addr(base, row, col)
        )

    def tmem_load_32(dst, dst_offset, base, row, col):
        return T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
            *[dst[dst_offset + i] for i in range(32)], _tmem_addr(base, row, col)
        )

    def tmem_store_32(src, src_offset, base, row, col):
        return T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
            _tmem_addr(base, row, col), *[src[src_offset + i] for i in range(32)]
        )

    # 2-SM cluster TMA load: unicast (no .multicast::cluster) and no cache
    # policy, so only the completion and cta_group tokens ride the chain. Note
    # the mbarrier operand now follows the coordinates.
    _TMA_G2S_2SM = (
        "cp.async.bulk.tensor.{dim}d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
    )
    # Stores put the tensor map and coordinates first and the shared source last.
    _TMA_S2G = "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"
    _TMA_S2G_REDUCE = (
        "cp.reduce.async.bulk.tensor.{dim}d.global.shared::cta.{redop}.tile.bulk_group"
    )
    # The destination is this CTA's own shared memory.
    _BULK_G2S_CTA = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    # shared::cta -> peer CTA's shared::cluster window (DSMEM push).
    _BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
    # pair_mask names both CTAs of the pair, so the commit is the multicast form.
    _TCGEN05_COMMIT = (
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
    )

    def tma_g2s(dim, dst_ptr, mbar, tensormap_addr, *coords):
        return T.ptx[_TMA_G2S_2SM.format(dim=dim)](dst_ptr, tensormap_addr, *coords, mbar)

    def tma_s2g(dim, src_ptr, tensormap_addr, *coords):
        return T.ptx[_TMA_S2G.format(dim=dim)](tensormap_addr, *coords, src_ptr)

    def tma_s2g_reduce(dim, src_ptr, tensormap_addr, redop, *coords):
        chain = _TMA_S2G_REDUCE.format(dim=dim, redop=redop)
        return T.ptx[chain](tensormap_addr, *coords, src_ptr)

    def bulk_g2s_cta(dst_ptr, src_ptr, num_bytes, mbar):
        return T.ptx[_BULK_G2S_CTA](dst_ptr, src_ptr, T.uint32(num_bytes), mbar)

    def tcgen05_commit(mbar_ptr, cta_mask):
        return T.ptx[_TCGEN05_COMMIT](mbar_ptr, T.Cast("uint16", cta_mask))

    def tmem_store_16(src, src_offset, base, row, col):
        return T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
            _tmem_addr(base, row, col), *[src[src_offset + i] for i in range(16)]
        )

    @T.meta_class
    class RowiseSwizzleOffset:
        def __init__(self, swizzle_len, atom_len, per_element, row_base, prefix="row_sw"):
            self.swizzle_len = swizzle_len
            self.atom_len = atom_len
            self.per_element = per_element
            self.row_base = row_base
            self.signed_strides = T.alloc_buffer([self.atom_len], "int32", scope="local")
            self.n_dim = self.swizzle_len + 1
            self.shape = [2] * self.n_dim
            self.shape[-1] = 1 << self.per_element

        def init(self):
            with T.unroll(self.swizzle_len) as i:
                y_i = T.meta_var(self.row_base & (1 << i))
                stride_i = T.meta_var(1 << (i + self.per_element))
                T.buffer_store(
                    self.signed_strides,
                    T.if_then_else(y_i.value > 0, stride_i.value * -1, stride_i.value),
                    [i],
                )
            with T.unroll(self.swizzle_len, self.atom_len) as i:
                stride_i = T.meta_var(1 << (i + self.per_element))
                T.buffer_store(self.signed_strides, stride_i.value, [i])

        def apply(self, offset):
            offset_layout = TileLayout(
                S[
                    self.shape : [
                        self.signed_strides[self.swizzle_len - 1 - i] for i in range(self.atom_len)
                    ]
                    + [0]
                ]
            )
            return offset_layout.apply(offset)["m"]

    NUM_HEADS = BATCH * HEADS_PER_BATCH

    CTA_GROUP = 2
    CLUSTER_M, CLUSTER_N = 2, 1
    CLUSTER_SIZE = CLUSTER_M * CLUSTER_N
    BLK_M = 128
    BLK_N = 256  # doubled: 2 CTAs x 128 rows each
    CTA_N = BLK_N // CTA_GROUP  # 128 per CTA
    MMA_N = 128  # TMEM output cols (unchanged)
    B_N = BLK_M // CTA_GROUP  # 64: per-CTA B rows for row-split
    B_N_COL = HEAD_DIM // CTA_GROUP  # 64: per-CTA B cols for col-split
    EPI_N = 64
    STRIP_SIZE = 64
    TMEM_LD_N = 64
    DQ_RED_N = 8
    DQ_STAGES = 4
    DQ_M_PER_CTA = 64  # Phase E Layout B: 64 rows per CTA
    DQ_ROWS_PER_STAGE = DQ_RED_N
    DQ_REDUCE_ITERS = DQ_M_PER_CTA // DQ_ROWS_PER_STAGE

    NUM_M_TILES = SEQ_LEN // BLK_M
    NUM_N_TILES = SEQ_LEN // BLK_N  # halved vs v0
    STRIPS = HEAD_DIM // STRIP_SIZE  # 2

    softmax_scale = 1.0 / math.sqrt(HEAD_DIM) if attention_scale is None else float(attention_scale)
    log2e = 1.4426950408889634
    scale_log2 = softmax_scale * log2e  # precomputed compile-time constant

    WG_NUMBER = 4
    DTYPE_SIZE = 2
    # Keep the public specialization argument even though the current upstream
    # kernel uses SingleTileScheduler for both causal and dense launches.
    _ = sm_count

    # TMA byte counts
    CTA_N_BYTES = CTA_N * HEAD_DIM * DTYPE_SIZE  # 32KB per CTA's K or V load
    Q_ROW_BYTES = B_N * HEAD_DIM * DTYPE_SIZE  # 16KB per CTA's Q row-split
    Q_COL_BYTES = BLK_M * B_N_COL * DTYPE_SIZE  # 16KB per CTA's Q col-split
    LSE_BYTES = BLK_M * 4  # 512 bytes (fp32)
    DPSUM_BYTES = BLK_M * 4  # 512 bytes (fp32)

    # SMEM layouts
    kv_layout = tma_shared_layout(f16, (CTA_N, HEAD_DIM))
    q_row_layout = tma_shared_layout(f16, (1, B_N, HEAD_DIM))
    q_col_layout = tma_shared_layout(f16, (BLK_M, B_N_COL))
    do_row_layout = tma_shared_layout(f16, (B_N, HEAD_DIM))
    do_col_layout = tma_shared_layout(f16, (BLK_M, B_N_COL))
    # dS for Phase E after DSMEM exchange: [BLK_N, B_N] per CTA = [256, 64]
    ds_exchange_layout = tma_shared_layout(f16, (BLK_N, B_N))
    # dS staging buffer: holds local N half x peer's M strip, for DSMEM send
    ds_stage_layout = tma_shared_layout(f16, (CTA_N, B_N))
    # K col-split for Phase E after DSMEM exchange: [BLK_N, B_N_COL] = [256, 64]
    k_col_layout = tma_shared_layout(f16, (BLK_N, B_N_COL))
    epi_layout_2 = tma_shared_layout(f16, (2, CTA_N, EPI_N))
    dq_red_layout = TileLayout(S[(DQ_STAGES, BLK_M, DQ_RED_N) : (BLK_M * DQ_RED_N, DQ_RED_N, 1)])

    # Current FA4 D=128 TMEM packing.  dQ aliases the upper half of S/P;
    # dP and dS alias each other after the compute warps drain dP.
    TMEM_OFF_A = 0  # S/P
    TMEM_OFF_DQ = MMA_N // 2  # dQ (64), aliases S/P
    TMEM_OFF_B = MMA_N  # dV accumulator (128)
    TMEM_OFF_DP = 2 * MMA_N  # dP/dS (256)
    TMEM_OFF_C = 3 * MMA_N  # dK accumulator (384)
    iket = IketProfiler()

    # fmt: off
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("kernel")
            Q_g = T.arg("Q_g", T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16))
            K_g = T.arg("K_g", T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16))
            V_g = T.arg("V_g", T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16))
            dO_g = T.arg("dO_g", T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16))
            LSE_g = T.arg("LSE_g", T.Buffer((BATCH, HEADS_PER_BATCH, SEQ_LEN), f32))
            dpsum_g = T.arg("dpsum_g", T.Buffer((BATCH, HEADS_PER_BATCH, SEQ_LEN), f32))
            dK_g = T.arg("dK_g", T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16))
            dV_g = T.arg("dV_g", T.Buffer((BATCH, SEQ_LEN, HEADS_PER_BATCH, HEAD_DIM), f16))
            dQ_acc_g = T.arg("dQ_acc_g", T.Buffer((BATCH, HEADS_PER_BATCH, SEQ_LEN, HEAD_DIM), f32))
            q_row_tensormap = _builder_bind("q_row_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', q_row_tensormap, 'float16', 5, Q_g.data, HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM // 2, B_N, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            q_col_tensormap = _builder_bind("q_col_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', q_col_tensormap, 'float16', 4, Q_g.data, HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, B_N_COL, BLK_M, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            k_row_tensormap = _builder_bind("k_row_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', k_row_tensormap, 'float16', 5, K_g.data, HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM // 2, CTA_N, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            k_col_tensormap = _builder_bind("k_col_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', k_col_tensormap, 'float16', 4, K_g.data, HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, B_N_COL, BLK_N, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            v_row_tensormap = _builder_bind("v_row_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', v_row_tensormap, 'float16', 5, V_g.data, HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM // 2, CTA_N, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            do_row_tensormap = _builder_bind("do_row_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', do_row_tensormap, 'float16', 5, dO_g.data, HEAD_DIM // 2, SEQ_LEN, 2, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM // 2, B_N, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            do_col_tensormap = _builder_bind("do_col_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', do_col_tensormap, 'float16', 4, dO_g.data, HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, B_N_COL, BLK_M, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            lse_tensormap = _builder_bind("lse_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', lse_tensormap, 'float32', 3, LSE_g.data, SEQ_LEN, HEADS_PER_BATCH, BATCH, SEQ_LEN * 4, HEADS_PER_BATCH * SEQ_LEN * 4, BLK_M, 1, 1, 1, 1, 1, 0, 0, 2, 0))
            dpsum_tensormap = _builder_bind("dpsum_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', dpsum_tensormap, 'float32', 3, dpsum_g.data, SEQ_LEN, HEADS_PER_BATCH, BATCH, SEQ_LEN * 4, HEADS_PER_BATCH * SEQ_LEN * 4, BLK_M, 1, 1, 1, 1, 1, 0, 0, 2, 0))
            dk_tensormap = _builder_bind("dk_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', dk_tensormap, 'float16', 4, dK_g.data, HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, EPI_N, CTA_N, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            dv_tensormap = _builder_bind("dv_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', dv_tensormap, 'float16', 4, dV_g.data, HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH, HEADS_PER_BATCH * HEAD_DIM * 2, HEAD_DIM * 2, SEQ_LEN * HEADS_PER_BATCH * HEAD_DIM * 2, EPI_N, CTA_N, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0))
            dq_tensormap = _builder_bind("dq_tensormap", T.tvm_stack_alloca('tensormap', 1), T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', dq_tensormap, 'float32', 4, dQ_acc_g.data, HEAD_DIM, SEQ_LEN, HEADS_PER_BATCH, BATCH, HEAD_DIM * 4, SEQ_LEN * HEAD_DIM * 4, HEADS_PER_BATCH * SEQ_LEN * HEAD_DIM * 4, HEAD_DIM, DQ_ROWS_PER_STAGE, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0))
            _builder_emit(T.device_entry())
            cluster_rank_ = _builder_assign("cluster_rank_", T.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE]), locals().get("cluster_rank_", _BUILDER_MISSING))
            bx, by = _builder_assign_many(('bx', 'by'), T.cta_id([NUM_N_TILES * CLUSTER_SIZE, NUM_HEADS]), (locals().get("bx", _BUILDER_MISSING), locals().get("by", _BUILDER_MISSING),))
            wg_id = _builder_assign("wg_id", T.warpgroup_id([WG_NUMBER]), locals().get("wg_id", _BUILDER_MISSING))
            warp_id = _builder_assign("warp_id", T.warp_id_in_wg([4]), locals().get("warp_id", _BUILDER_MISSING))
            lane_id = _builder_assign("lane_id", T.lane_id([32]), locals().get("lane_id", _BUILDER_MISSING))
            id_in_pair = _builder_bind("id_in_pair", cluster_rank_ % CTA_GROUP, None)
            pair_leader_rank = _builder_bind("pair_leader_rank", cluster_rank_ - id_in_pair, None)
            pool = _builder_assign("pool", T.SMEMPool(), locals().get("pool", _BUILDER_MISSING))
            tmem_addr = _builder_assign("tmem_addr", pool.alloc((1,), 'uint32'), locals().get("tmem_addr", _BUILDER_MISSING))
            tmem_dealloc_mbar = _builder_assign("tmem_dealloc_mbar", MBarrier(pool, 1, leader=(wg_id == 3) & (warp_id == 0) & (lane_id == 0)), locals().get("tmem_dealloc_mbar", _BUILDER_MISSING))
            tma_kv = _builder_assign("tma_kv", TMABar(pool, 1), locals().get("tma_kv", _BUILDER_MISSING))
            tma_a = _builder_assign("tma_a", TMABar(pool, 1), locals().get("tma_a", _BUILDER_MISSING))
            tma_q = _builder_assign("tma_q", TMABar(pool, 1), locals().get("tma_q", _BUILDER_MISSING))
            tma_lse = _builder_assign("tma_lse", TMABar(pool, 1), locals().get("tma_lse", _BUILDER_MISSING))
            tma_dpsum = _builder_assign("tma_dpsum", TMABar(pool, 1), locals().get("tma_dpsum", _BUILDER_MISSING))
            tma_qcol = _builder_assign("tma_qcol", TMABar(pool, 1), locals().get("tma_qcol", _BUILDER_MISSING))
            mma2wg0_s = _builder_assign("mma2wg0_s", TCGen05Bar(pool, 1), locals().get("mma2wg0_s", _BUILDER_MISSING))
            mma2wg0_dp = _builder_assign("mma2wg0_dp", TCGen05Bar(pool, 1), locals().get("mma2wg0_dp", _BUILDER_MISSING))
            mma2wg0_dq = _builder_assign("mma2wg0_dq", TCGen05Bar(pool, 1), locals().get("mma2wg0_dq", _BUILDER_MISSING))
            ds_exch_mbar = _builder_assign("ds_exch_mbar", MBarrier(pool, 1), locals().get("ds_exch_mbar", _BUILDER_MISSING))
            ds_exch_consumed = _builder_assign("ds_exch_consumed", MBarrier(pool, 1), locals().get("ds_exch_consumed", _BUILDER_MISSING))
            wg02mma = _builder_assign("wg02mma", MBarrier(pool, 1), locals().get("wg02mma", _BUILDER_MISSING))
            wg02mma_tmem = _builder_assign("wg02mma_tmem", MBarrier(pool, 1), locals().get("wg02mma_tmem", _BUILDER_MISSING))
            strip_ready = _builder_assign("strip_ready", MBarrier(pool, 1), locals().get("strip_ready", _BUILDER_MISSING))
            s_tmem_consumed = _builder_assign("s_tmem_consumed", MBarrier(pool, 1), locals().get("s_tmem_consumed", _BUILDER_MISSING))
            buf_a_consumed = _builder_assign("buf_a_consumed", MBarrier(pool, 1), locals().get("buf_a_consumed", _BUILDER_MISSING))
            q_consumed = _builder_assign("q_consumed", MBarrier(pool, 1), locals().get("q_consumed", _BUILDER_MISSING))
            lse_consumed = _builder_assign("lse_consumed", MBarrier(pool, 1), locals().get("lse_consumed", _BUILDER_MISSING))
            dpsum_consumed = _builder_assign("dpsum_consumed", MBarrier(pool, 1), locals().get("dpsum_consumed", _BUILDER_MISSING))
            qcol_consumed = _builder_assign("qcol_consumed", MBarrier(pool, 1), locals().get("qcol_consumed", _BUILDER_MISSING))
            dq_tmem_free = _builder_assign("dq_tmem_free", MBarrier(pool, 1), locals().get("dq_tmem_free", _BUILDER_MISSING))
            dv_done = _builder_assign("dv_done", TCGen05Bar(pool, 1), locals().get("dv_done", _BUILDER_MISSING))
            dk_done = _builder_assign("dk_done", TCGen05Bar(pool, 1), locals().get("dk_done", _BUILDER_MISSING))
            _builder_emit(pool.move_base_to(POOL_Q_ROW))
            Q_row = _builder_assign("Q_row", pool.alloc((1, B_N, HEAD_DIM), f16, layout=q_row_layout), locals().get("Q_row", _BUILDER_MISSING))
            K_smem = _builder_assign("K_smem", pool.alloc((CTA_N, HEAD_DIM), f16, layout=kv_layout), locals().get("K_smem", _BUILDER_MISSING))
            V_smem = _builder_assign("V_smem", pool.alloc((CTA_N, HEAD_DIM), f16, layout=kv_layout), locals().get("V_smem", _BUILDER_MISSING))
            dO_row = _builder_assign("dO_row", pool.alloc((B_N, HEAD_DIM), f16, layout=do_row_layout), locals().get("dO_row", _BUILDER_MISSING))
            Q_col = _builder_assign("Q_col", pool.alloc((BLK_M, B_N_COL), f16, layout=q_col_layout), locals().get("Q_col", _BUILDER_MISSING))
            dO_col = _builder_assign("dO_col", pool.alloc((BLK_M, B_N_COL), f16, layout=do_col_layout), locals().get("dO_col", _BUILDER_MISSING))
            dS_send = _builder_assign("dS_send", pool.alloc((CTA_N, B_N), f16, layout=ds_stage_layout), locals().get("dS_send", _BUILDER_MISSING))
            K_col = _builder_assign("K_col", pool.alloc((BLK_N, B_N_COL), f16, layout=k_col_layout), locals().get("K_col", _BUILDER_MISSING))
            dS_exch = _builder_assign("dS_exch", pool.alloc((BLK_N, B_N), f16, layout=ds_exchange_layout), locals().get("dS_exch", _BUILDER_MISSING))
            sLSE = _builder_assign("sLSE", pool.alloc((1, BLK_M), f32, layout=TileLayout(S[(1, BLK_M):(BLK_M, 1)])), locals().get("sLSE", _BUILDER_MISSING))
            sDPsum = _builder_assign("sDPsum", pool.alloc((BLK_M,), f32, layout=TileLayout(S[(BLK_M,):(1,)])), locals().get("sDPsum", _BUILDER_MISSING))
            dQ_smem = _builder_assign("dQ_smem", pool.alloc((DQ_STAGES, BLK_M, DQ_RED_N), f32, layout=dq_red_layout, align=1024), locals().get("dQ_smem", _BUILDER_MISSING))
            _builder_emit(pool.move_base_to(K_smem.elem_offset * DTYPE_SIZE))
            dK_epi = _builder_assign("dK_epi", pool.alloc((2, CTA_N, EPI_N), f16, layout=epi_layout_2), locals().get("dK_epi", _BUILDER_MISSING))
            _builder_emit(pool.move_base_to(V_smem.elem_offset * DTYPE_SIZE))
            dV_epi = _builder_assign("dV_epi", pool.alloc((2, CTA_N, EPI_N), f16, layout=epi_layout_2), locals().get("dV_epi", _BUILDER_MISSING))
            _builder_emit(pool.commit())
            desc_k_row = _builder_alloc_scalar("desc_k_row", "uint64")
            desc_q_row = _builder_alloc_scalar("desc_q_row", "uint64")
            desc_v_row = _builder_alloc_scalar("desc_v_row", "uint64")
            desc_do_row = _builder_alloc_scalar("desc_do_row", "uint64")
            desc_q_col = _builder_alloc_scalar("desc_q_col", "uint64")
            desc_do_col = _builder_alloc_scalar("desc_do_col", "uint64")
            desc_k_col = _builder_alloc_scalar("desc_k_col", "uint64")
            desc_ds_exch = _builder_alloc_scalar("desc_ds_exch", "uint64")
            _builder_emit(tma_kv.init(1))
            _builder_emit(tma_a.init(1))
            _builder_emit(tma_q.init(1))
            _builder_emit(tma_lse.init(1))
            _builder_emit(tma_dpsum.init(1))
            _builder_emit(tma_qcol.init(1))
            _builder_emit(mma2wg0_s.init(1))
            _builder_emit(mma2wg0_dp.init(1))
            _builder_emit(mma2wg0_dq.init(1))
            _builder_emit(wg02mma.init(CTA_GROUP))
            _builder_emit(wg02mma_tmem.init(8 * CTA_GROUP))
            _builder_emit(strip_ready.init(8 * CTA_GROUP))
            _builder_emit(s_tmem_consumed.init(8 * CTA_GROUP))
            _builder_emit(buf_a_consumed.init(1))
            _builder_emit(q_consumed.init(1))
            _builder_emit(lse_consumed.init(8))
            _builder_emit(dpsum_consumed.init(8))
            _builder_emit(qcol_consumed.init(1))
            _builder_emit(dq_tmem_free.init(4 * CTA_GROUP))
            _builder_emit(ds_exch_mbar.init(1))
            _builder_emit(ds_exch_consumed.init(1))
            _builder_emit(dv_done.init(1))
            _builder_emit(dk_done.init(1))
            _builder_emit(tmem_dealloc_mbar.init(32))
            pair_mask = _builder_alloc_scalar("pair_mask", "int32")
            pair_mask = _builder_assign("pair_mask", 1 << pair_leader_rank | 1 << pair_leader_rank + 1, locals().get("pair_mask", _BUILDER_MISSING))
            _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(T.ptx.barrier.cluster.arrive.relaxed())
            _builder_emit(T.ptx.barrier.cluster.wait())
            tma_kv_cta0 = _builder_assign("tma_kv_cta0", tma_kv.remote_view(pair_leader_rank), locals().get("tma_kv_cta0", _BUILDER_MISSING))
            tma_q_cta0 = _builder_assign("tma_q_cta0", tma_q.remote_view(pair_leader_rank), locals().get("tma_q_cta0", _BUILDER_MISSING))
            tma_qcol_cta0 = _builder_assign("tma_qcol_cta0", tma_qcol.remote_view(pair_leader_rank), locals().get("tma_qcol_cta0", _BUILDER_MISSING))
            tma_a_cta0 = _builder_assign("tma_a_cta0", tma_a.remote_view(pair_leader_rank), locals().get("tma_a_cta0", _BUILDER_MISSING))
            n_tile_idx = bx // CLUSTER_SIZE
            head_flat = by
            b_idx = head_flat // HEADS_PER_BATCH
            h_idx = head_flat % HEADS_PER_BATCH
            n_st = n_tile_idx * BLK_N
            n_st_cta = n_st + id_in_pair * CTA_N
            m_tile_start = n_tile_idx * (BLK_N // BLK_M) if causal else 0
            num_m_tiles_this_n = NUM_M_TILES - m_tile_start
            m_st_first = m_tile_start * BLK_M
            with T.If(wg_id == 3):
                with T.Then():
                    _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(104))
                    with T.If(warp_id == 0):
                        with T.Then():
                            _builder_emit(T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(tmem_addr), T.uint32(512)))
                            _builder_emit(T.ptx.barrier.sync(T.uint32(5), 416))
                    with T.If(warp_id == 1):
                        with T.Then():
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_row_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_col_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_row_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_col_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(v_row_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(do_row_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(do_col_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(dk_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(dv_tensormap))))
                                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(dq_tensormap))))
                            q_cons_ph = _builder_assign("q_cons_ph", PipelineState(1), locals().get("q_cons_ph", _BUILDER_MISSING))
                            _builder_emit(q_cons_ph.init(1))
                            lse_cons_ph = _builder_assign("lse_cons_ph", PipelineState(1), locals().get("lse_cons_ph", _BUILDER_MISSING))
                            _builder_emit(lse_cons_ph.init(1))
                            qcol_cons_ph = _builder_assign("qcol_cons_ph", PipelineState(1), locals().get("qcol_cons_ph", _BUILDER_MISSING))
                            _builder_emit(qcol_cons_ph.init(1))
                            a_cons_ph = _builder_assign("a_cons_ph", PipelineState(1), locals().get("a_cons_ph", _BUILDER_MISSING))
                            _builder_emit(a_cons_ph.init(1))
                            dpsum_cons_ph = _builder_assign("dpsum_cons_ph", PipelineState(1), locals().get("dpsum_cons_ph", _BUILDER_MISSING))
                            _builder_emit(dpsum_cons_ph.init(1))
                            K_COL_BYTES = BLK_N * B_N_COL * DTYPE_SIZE
                            KV_TOTAL_BYTES = (CTA_N_BYTES * 2 + K_COL_BYTES) * CTA_GROUP
                            Q_BATCH_BYTES = Q_ROW_BYTES * CTA_GROUP
                            QCOL_BATCH_BYTES = Q_COL_BYTES * CTA_GROUP
                            DO_BATCH_BYTES = (Q_ROW_BYTES + Q_COL_BYTES) * CTA_GROUP

                            def tma_n_tile():
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        _builder_emit(tma_g2s(5, K_smem.ptr_to([0, 0]), tma_kv_cta0.ptr_to([0]), T.address_of(k_row_tensormap), 0, n_st_cta, 0, h_idx, b_idx))
                                        _builder_emit(tma_g2s(5, V_smem.ptr_to([0, 0]), tma_kv_cta0.ptr_to([0]), T.address_of(v_row_tensormap), 0, n_st_cta, 0, h_idx, b_idx))
                                        k_col_col_st = id_in_pair * B_N_COL
                                        _builder_emit(tma_g2s(4, K_col.ptr_to([0, 0]), tma_kv_cta0.ptr_to([0]), T.address_of(k_col_tensormap), k_col_col_st, n_st, h_idx, b_idx))
                                        with T.If(id_in_pair == 0):
                                            with T.Then():
                                                _builder_emit(tma_kv.arrive(0, KV_TOTAL_BYTES))
                                tma_prefetch_token = _builder_assign("tma_prefetch_token", iket.range_start('tma-prefetch'), locals().get("tma_prefetch_token", _BUILDER_MISSING))
                                tma_wait_q_token = _builder_assign("tma_wait_q_token", iket.range_start('tma-wait-q'), locals().get("tma_wait_q_token", _BUILDER_MISSING))
                                _builder_emit(q_consumed.wait(0, q_cons_ph.phase))
                                _builder_emit(iket.range_end(tma_wait_q_token))
                                q_row_st = m_st_first + id_in_pair * B_N
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        _builder_emit(tma_g2s(5, Q_row.ptr_to([0, 0, 0]), tma_q_cta0.ptr_to([0]), T.address_of(q_row_tensormap), 0, q_row_st, 0, h_idx, b_idx))
                                        with T.If(id_in_pair == 0):
                                            with T.Then():
                                                _builder_emit(tma_q.arrive(0, Q_BATCH_BYTES))
                                _builder_emit(q_cons_ph.advance())
                                tma_wait_lse_token = _builder_assign("tma_wait_lse_token", iket.range_start('tma-wait-lse'), locals().get("tma_wait_lse_token", _BUILDER_MISSING))
                                _builder_emit(lse_consumed.wait(0, lse_cons_ph.phase))
                                _builder_emit(iket.range_end(tma_wait_lse_token))
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        _builder_emit(tma_lse.arrive(0, LSE_BYTES))
                                        _builder_emit(bulk_g2s_cta(sLSE.ptr_to([0, 0]), LSE_g.ptr_to([b_idx, h_idx, m_st_first]), LSE_BYTES, tma_lse.ptr_to([0])))
                                _builder_emit(lse_cons_ph.advance())
                                tma_wait_a_token = _builder_assign("tma_wait_a_token", iket.range_start('tma-wait-a'), locals().get("tma_wait_a_token", _BUILDER_MISSING))
                                _builder_emit(buf_a_consumed.wait(0, a_cons_ph.phase))
                                _builder_emit(iket.range_end(tma_wait_a_token))
                                do_col_st = id_in_pair * B_N_COL
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        _builder_emit(tma_g2s(5, dO_row.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]), T.address_of(do_row_tensormap), 0, q_row_st, 0, h_idx, b_idx))
                                        _builder_emit(tma_g2s(4, dO_col.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]), T.address_of(do_col_tensormap), do_col_st, m_st_first, h_idx, b_idx))
                                        with T.If(id_in_pair == 0):
                                            with T.Then():
                                                _builder_emit(tma_a.arrive(0, DO_BATCH_BYTES))
                                _builder_emit(a_cons_ph.advance())
                                tma_wait_dpsum_token = _builder_assign("tma_wait_dpsum_token", iket.range_start('tma-wait-dpsum'), locals().get("tma_wait_dpsum_token", _BUILDER_MISSING))
                                _builder_emit(dpsum_consumed.wait(0, dpsum_cons_ph.phase))
                                _builder_emit(iket.range_end(tma_wait_dpsum_token))
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        _builder_emit(tma_dpsum.arrive(0, DPSUM_BYTES))
                                        _builder_emit(bulk_g2s_cta(sDPsum.ptr_to([0]), dpsum_g.ptr_to([b_idx, h_idx, m_st_first]), DPSUM_BYTES, tma_dpsum.ptr_to([0])))
                                _builder_emit(dpsum_cons_ph.advance())
                                _builder_emit(iket.range_end(tma_prefetch_token))
                                with T.serial(num_m_tiles_this_n - 1, annotations={'disable_unroll': True}) as i_m:
                                    IRBuilder.name("i_m", i_m)
                                    m_st_next = (m_tile_start + i_m + 1) * BLK_M
                                    q_row_st_next = m_st_next + id_in_pair * B_N
                                    q_col_st_next = id_in_pair * B_N_COL
                                    tma_prefetch_token = _builder_assign("tma_prefetch_token", iket.range_start('tma-prefetch'), locals().get("tma_prefetch_token", _BUILDER_MISSING))
                                    m_st_qcol = m_st_next - BLK_M
                                    tma_wait_qcol_token = _builder_assign("tma_wait_qcol_token", iket.range_start('tma-wait-qcol'), locals().get("tma_wait_qcol_token", _BUILDER_MISSING))
                                    _builder_emit(qcol_consumed.wait(0, qcol_cons_ph.phase))
                                    _builder_emit(iket.range_end(tma_wait_qcol_token))
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(tma_g2s(4, Q_col.ptr_to([0, 0]), tma_qcol_cta0.ptr_to([0]), T.address_of(q_col_tensormap), q_col_st_next, m_st_qcol, h_idx, b_idx))
                                            with T.If(id_in_pair == 0):
                                                with T.Then():
                                                    _builder_emit(tma_qcol.arrive(0, QCOL_BATCH_BYTES))
                                    _builder_emit(qcol_cons_ph.advance())
                                    tma_wait_q_token = _builder_assign("tma_wait_q_token", iket.range_start('tma-wait-q'), locals().get("tma_wait_q_token", _BUILDER_MISSING))
                                    _builder_emit(q_consumed.wait(0, q_cons_ph.phase))
                                    _builder_emit(iket.range_end(tma_wait_q_token))
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(tma_g2s(5, Q_row.ptr_to([0, 0, 0]), tma_q_cta0.ptr_to([0]), T.address_of(q_row_tensormap), 0, q_row_st_next, 0, h_idx, b_idx))
                                            with T.If(id_in_pair == 0):
                                                with T.Then():
                                                    _builder_emit(tma_q.arrive(0, Q_BATCH_BYTES))
                                    _builder_emit(q_cons_ph.advance())
                                    tma_wait_lse_token = _builder_assign("tma_wait_lse_token", iket.range_start('tma-wait-lse'), locals().get("tma_wait_lse_token", _BUILDER_MISSING))
                                    _builder_emit(lse_consumed.wait(0, lse_cons_ph.phase))
                                    _builder_emit(iket.range_end(tma_wait_lse_token))
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(tma_lse.arrive(0, LSE_BYTES))
                                            _builder_emit(bulk_g2s_cta(sLSE.ptr_to([0, 0]), LSE_g.ptr_to([b_idx, h_idx, m_st_next]), LSE_BYTES, tma_lse.ptr_to([0])))
                                    _builder_emit(lse_cons_ph.advance())
                                    tma_wait_a_token = _builder_assign("tma_wait_a_token", iket.range_start('tma-wait-a'), locals().get("tma_wait_a_token", _BUILDER_MISSING))
                                    _builder_emit(buf_a_consumed.wait(0, a_cons_ph.phase))
                                    _builder_emit(iket.range_end(tma_wait_a_token))
                                    do_col_st_next = id_in_pair * B_N_COL
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(tma_g2s(5, dO_row.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]), T.address_of(do_row_tensormap), 0, q_row_st_next, 0, h_idx, b_idx))
                                            _builder_emit(tma_g2s(4, dO_col.ptr_to([0, 0]), tma_a_cta0.ptr_to([0]), T.address_of(do_col_tensormap), do_col_st_next, m_st_next, h_idx, b_idx))
                                            with T.If(id_in_pair == 0):
                                                with T.Then():
                                                    _builder_emit(tma_a.arrive(0, DO_BATCH_BYTES))
                                    _builder_emit(a_cons_ph.advance())
                                    tma_wait_dpsum_token = _builder_assign("tma_wait_dpsum_token", iket.range_start('tma-wait-dpsum'), locals().get("tma_wait_dpsum_token", _BUILDER_MISSING))
                                    _builder_emit(dpsum_consumed.wait(0, dpsum_cons_ph.phase))
                                    _builder_emit(iket.range_end(tma_wait_dpsum_token))
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(tma_dpsum.arrive(0, DPSUM_BYTES))
                                            _builder_emit(bulk_g2s_cta(sDPsum.ptr_to([0]), dpsum_g.ptr_to([b_idx, h_idx, m_st_next]), DPSUM_BYTES, tma_dpsum.ptr_to([0])))
                                    _builder_emit(dpsum_cons_ph.advance())
                                    _builder_emit(iket.range_end(tma_prefetch_token))
                                tma_prefetch_token = _builder_assign("tma_prefetch_token", iket.range_start('tma-prefetch'), locals().get("tma_prefetch_token", _BUILDER_MISSING))
                                tma_wait_qcol_token = _builder_assign(
                                    "tma_wait_qcol_token",
                                    iket.range_start('tma-wait-qcol'),
                                    _BUILDER_MISSING,
                                )
                                _builder_emit(qcol_consumed.wait(0, qcol_cons_ph.phase))
                                _builder_emit(iket.range_end(tma_wait_qcol_token))
                                m_st_qcol_tail = (m_tile_start + num_m_tiles_this_n - 1) * BLK_M
                                q_col_st_tail = id_in_pair * B_N_COL
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        _builder_emit(tma_g2s(4, Q_col.ptr_to([0, 0]), tma_qcol_cta0.ptr_to([0]), T.address_of(q_col_tensormap), q_col_st_tail, m_st_qcol_tail, h_idx, b_idx))
                                        with T.If(id_in_pair == 0):
                                            with T.Then():
                                                _builder_emit(tma_qcol.arrive(0, QCOL_BATCH_BYTES))
                                _builder_emit(qcol_cons_ph.advance())
                                _builder_emit(iket.range_end(tma_prefetch_token))
                            _builder_emit(tma_n_tile())
                        with T.Else():
                            with T.If(T.And(warp_id == 0, id_in_pair == 0)):
                                with T.Then():
                                    q_row_addr = _builder_scalar("q_row_addr", T.cuda.cvta_generic_to_shared(Q_row.ptr_to([0, 0, 0])), "uint32")
                                    q_row_start = _builder_scalar("q_row_start", T.cast(T.shift_right(q_row_addr, T.uint32(4)), 'uint64'), "uint64")
                                    desc_k_row = _builder_assign("desc_k_row", matrix_desc_from_anchor(MATRIX_DESC_F16_SS_LDO_1024, q_row_start, (K_smem.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_k_row", _BUILDER_MISSING))
                                    if not causal:
                                        desc_q_row = _builder_assign("desc_q_row", matrix_desc_from_anchor(MATRIX_DESC_F16_SS_LDO_512, q_row_start, 0), locals().get("desc_q_row", _BUILDER_MISSING))
                                    desc_v_row = _builder_assign("desc_v_row", matrix_desc_from_anchor(MATRIX_DESC_F16_SS_LDO_1024, q_row_start, (V_smem.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_v_row", _BUILDER_MISSING))
                                    desc_do_row = _builder_assign("desc_do_row", matrix_desc_from_anchor(MATRIX_DESC_F16_SS_LDO_512, q_row_start, (dO_row.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_do_row", _BUILDER_MISSING))
                                    desc_q_col = _builder_assign("desc_q_col", matrix_desc_from_anchor(MATRIX_DESC_F16_TS, q_row_start, (Q_col.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_q_col", _BUILDER_MISSING))
                                    desc_do_col = _builder_assign("desc_do_col", matrix_desc_from_anchor(MATRIX_DESC_F16_TS, q_row_start, (dO_col.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_do_col", _BUILDER_MISSING))
                                    desc_k_col = _builder_assign("desc_k_col", matrix_desc_from_anchor(MATRIX_DESC_F16_TS, q_row_start, (K_col.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_k_col", _BUILDER_MISSING))
                                    desc_ds_exch = _builder_assign("desc_ds_exch", matrix_desc_from_anchor(MATRIX_DESC_F16_TS, q_row_start, (dS_exch.elem_offset - Q_row.elem_offset) * DTYPE_SIZE), locals().get("desc_ds_exch", _BUILDER_MISSING))
                                    kv_ph = _builder_assign("kv_ph", PipelineState(1), locals().get("kv_ph", _BUILDER_MISSING))
                                    _builder_emit(kv_ph.init(0))
                                    q_ph = _builder_assign("q_ph", PipelineState(1), locals().get("q_ph", _BUILDER_MISSING))
                                    _builder_emit(q_ph.init(0))
                                    qcol_ph = _builder_assign("qcol_ph", PipelineState(1), locals().get("qcol_ph", _BUILDER_MISSING))
                                    _builder_emit(qcol_ph.init(0))
                                    a_ph = _builder_assign("a_ph", PipelineState(1), locals().get("a_ph", _BUILDER_MISSING))
                                    _builder_emit(a_ph.init(0))
                                    wg0_ph = _builder_assign("wg0_ph", PipelineState(1), locals().get("wg0_ph", _BUILDER_MISSING))
                                    _builder_emit(wg0_ph.init(0))
                                    wg0_smem_ph = _builder_assign("wg0_smem_ph", PipelineState(1), locals().get("wg0_smem_ph", _BUILDER_MISSING))
                                    _builder_emit(wg0_smem_ph.init(0))
                                    strip_ready_ph = _builder_assign("strip_ready_ph", PipelineState(1), locals().get("strip_ready_ph", _BUILDER_MISSING))
                                    _builder_emit(strip_ready_ph.init(0))
                                    s_tmem_consumed_ph = _builder_assign("s_tmem_consumed_ph", PipelineState(1), locals().get("s_tmem_consumed_ph", _BUILDER_MISSING))
                                    _builder_emit(s_tmem_consumed_ph.init(0))
                                    dq_tmem_free_ph = _builder_assign("dq_tmem_free_ph", PipelineState(1), locals().get("dq_tmem_free_ph", _BUILDER_MISSING))
                                    _builder_emit(dq_tmem_free_ph.init(1))
                                    accum_var = _builder_alloc_scalar("accum_var", "int32")
                                    accum_dv = _builder_alloc_scalar("accum_dv", "int32")
                                    accum_dk = _builder_alloc_scalar("accum_dk", "int32")

                                    def mma_n_tile():
                                        nonlocal accum_var, accum_dv, accum_dk

                                        _builder_emit(tma_kv.wait(0, kv_ph.phase))
                                        _builder_emit(kv_ph.advance())
                                        accum_dv = _builder_assign("accum_dv", 0, accum_dv)
                                        accum_dk = _builder_assign("accum_dk", 0, accum_dk)
                                        mma_s_token = _builder_assign("mma_s_token", iket.range_start('mma-s'), locals().get("mma_s_token", _BUILDER_MISSING))
                                        _builder_emit(tma_q.wait(0, q_ph.phase))
                                        _builder_emit(q_ph.advance())
                                        accum_var = _builder_assign("accum_var", 0, accum_var)
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                if causal:
                                                    q_row_issue_addr = _builder_scalar("q_row_issue_addr", T.cuda.cvta_generic_to_shared(Q_row.ptr_to([0, 0, 0])), "uint32")
                                                    q_row_issue_start = _builder_scalar("q_row_issue_start", T.cast(T.shift_right(q_row_issue_addr, T.uint32(4)), 'uint64'), "uint64")
                                                    desc_q_row_issue = _builder_scalar("desc_q_row_issue", matrix_desc_from_anchor(MATRIX_DESC_F16_SS_LDO_512, q_row_issue_start, 0), "uint64")
                                                    _builder_emit(mma_s(TMEM_OFF_A, accum_var, desc_k_row, desc_q_row_issue))
                                                else:
                                                    _builder_emit(mma_s(TMEM_OFF_A, accum_var, desc_k_row, desc_q_row))
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma2wg0_s.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask))
                                                _builder_emit(tcgen05_commit(q_consumed.ptr_to([0]), pair_mask))
                                        _builder_emit(iket.range_end(mma_s_token))
                                        mma_dp_token = _builder_assign("mma_dp_token", iket.range_start('mma-dp'), locals().get("mma_dp_token", _BUILDER_MISSING))
                                        _builder_emit(tma_a.wait(0, a_ph.phase))
                                        _builder_emit(a_ph.advance())
                                        accum_var = _builder_assign("accum_var", 0, accum_var)
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma_dp(TMEM_OFF_DP, accum_var, desc_v_row, desc_do_row))
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma2wg0_dp.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask))
                                        _builder_emit(iket.range_end(mma_dp_token))
                                        mma_dv_token = _builder_assign("mma_dv_token", iket.range_start('mma-dv'), locals().get("mma_dv_token", _BUILDER_MISSING))
                                        _builder_emit(strip_ready.wait(0, strip_ready_ph.phase))
                                        _builder_emit(strip_ready_ph.advance())
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma_dv(TMEM_OFF_B, TMEM_OFF_A * 2, accum_dv, desc_do_col))
                                        accum_dv = _builder_assign("accum_dv", 1, accum_dv)
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(tcgen05_commit(buf_a_consumed.ptr_to([0]), pair_mask))
                                        _builder_emit(iket.range_end(mma_dv_token))
                                        with T.serial(num_m_tiles_this_n - 1, annotations={'disable_unroll': True}) as i_m_inner:
                                            IRBuilder.name("i_m_inner", i_m_inner)
                                            i_m = m_tile_start + i_m_inner + 1
                                            mma_s_token = _builder_assign("mma_s_token", iket.range_start('mma-s'), locals().get("mma_s_token", _BUILDER_MISSING))
                                            _builder_emit(tma_q.wait(0, q_ph.phase))
                                            _builder_emit(q_ph.advance())
                                            _builder_emit(dq_tmem_free.wait(0, dq_tmem_free_ph.phase))
                                            _builder_emit(dq_tmem_free_ph.advance())
                                            accum_var = _builder_assign("accum_var", 0, accum_var)
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    if causal:
                                                        q_row_issue_addr = _builder_scalar("q_row_issue_addr", T.cuda.cvta_generic_to_shared(Q_row.ptr_to([0, 0, 0])), "uint32")
                                                        q_row_issue_start = _builder_scalar("q_row_issue_start", T.cast(T.shift_right(q_row_issue_addr, T.uint32(4)), 'uint64'), "uint64")
                                                        desc_q_row_issue = _builder_scalar("desc_q_row_issue", matrix_desc_from_anchor(MATRIX_DESC_F16_SS_LDO_512, q_row_issue_start, 0), "uint64")
                                                        _builder_emit(mma_s(TMEM_OFF_A, accum_var, desc_k_row, desc_q_row_issue))
                                                    else:
                                                        _builder_emit(mma_s(TMEM_OFF_A, accum_var, desc_k_row, desc_q_row))
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma2wg0_s.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask))
                                                    _builder_emit(tcgen05_commit(q_consumed.ptr_to([0]), pair_mask))
                                            _builder_emit(iket.range_end(mma_s_token))
                                            mma_dk_token = _builder_assign("mma_dk_token", iket.range_start('mma-dk'), locals().get("mma_dk_token", _BUILDER_MISSING))
                                            _builder_emit(wg02mma_tmem.wait(0, wg0_ph.phase))
                                            _builder_emit(wg0_ph.advance())
                                            _builder_emit(tma_qcol.wait(0, qcol_ph.phase))
                                            _builder_emit(qcol_ph.advance())
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma_dk(TMEM_OFF_C, TMEM_OFF_DP, accum_dk, desc_q_col))
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(tcgen05_commit(qcol_consumed.ptr_to([0]), pair_mask))
                                            accum_dk = _builder_assign("accum_dk", 1, accum_dk)
                                            _builder_emit(iket.range_end(mma_dk_token))
                                            mma_dp_token = _builder_assign("mma_dp_token", iket.range_start('mma-dp'), locals().get("mma_dp_token", _BUILDER_MISSING))
                                            _builder_emit(tma_a.wait(0, a_ph.phase))
                                            _builder_emit(a_ph.advance())
                                            accum_var = _builder_assign("accum_var", 0, accum_var)
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma_dp(TMEM_OFF_DP, accum_var, desc_v_row, desc_do_row))
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma2wg0_dp.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask))
                                            _builder_emit(iket.range_end(mma_dp_token))
                                            mma_dq_ready_token = _builder_assign("mma_dq_ready_token", iket.range_start('mma-dq-ready-wait'), locals().get("mma_dq_ready_token", _BUILDER_MISSING))
                                            _builder_emit(wg02mma.wait(0, wg0_smem_ph.phase))
                                            _builder_emit(wg0_smem_ph.advance())
                                            _builder_emit(iket.range_end(mma_dq_ready_token))
                                            mma_dq_alias_token = _builder_assign("mma_dq_alias_token", iket.range_start('mma-dq-alias-wait'), locals().get("mma_dq_alias_token", _BUILDER_MISSING))
                                            _builder_emit(s_tmem_consumed.wait(0, s_tmem_consumed_ph.phase))
                                            _builder_emit(s_tmem_consumed_ph.advance())
                                            _builder_emit(iket.range_end(mma_dq_alias_token))
                                            mma_dq_issue_token = _builder_assign("mma_dq_issue_token", iket.range_start('mma-dq-issue'), locals().get("mma_dq_issue_token", _BUILDER_MISSING))
                                            accum_var = _builder_assign("accum_var", 0, accum_var)
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma_dq(TMEM_OFF_DQ, accum_var, desc_ds_exch, desc_k_col))
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma2wg0_dq.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask))
                                                    _builder_emit(tcgen05_commit(ds_exch_consumed.ptr_to([0]), pair_mask))
                                            _builder_emit(iket.range_end(mma_dq_issue_token))
                                            mma_dv_token = _builder_assign("mma_dv_token", iket.range_start('mma-dv'), locals().get("mma_dv_token", _BUILDER_MISSING))
                                            _builder_emit(strip_ready.wait(0, strip_ready_ph.phase))
                                            _builder_emit(strip_ready_ph.advance())
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(mma_dv(TMEM_OFF_B, TMEM_OFF_A * 2, accum_dv, desc_do_col))
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(tcgen05_commit(buf_a_consumed.ptr_to([0]), pair_mask))
                                            _builder_emit(iket.range_end(mma_dv_token))
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(tcgen05_commit(dv_done.ptr_to([0]), pair_mask))
                                        mma_dk_token = _builder_assign(
                                            "mma_dk_token",
                                            iket.range_start('mma-dk'),
                                            _BUILDER_MISSING,
                                        )
                                        _builder_emit(wg02mma_tmem.wait(0, wg0_ph.phase))
                                        _builder_emit(wg0_ph.advance())
                                        _builder_emit(tma_qcol.wait(0, qcol_ph.phase))
                                        _builder_emit(qcol_ph.advance())
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma_dk(TMEM_OFF_C, TMEM_OFF_DP, accum_dk, desc_q_col))
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(tcgen05_commit(qcol_consumed.ptr_to([0]), pair_mask))
                                                _builder_emit(tcgen05_commit(dk_done.ptr_to([0]), pair_mask))
                                        _builder_emit(iket.range_end(mma_dk_token))
                                        _builder_emit(dq_tmem_free.wait(0, dq_tmem_free_ph.phase))
                                        _builder_emit(dq_tmem_free_ph.advance())
                                        mma_dq_ready_token = _builder_assign(
                                            "mma_dq_ready_token",
                                            iket.range_start('mma-dq-ready-wait'),
                                            _BUILDER_MISSING,
                                        )
                                        _builder_emit(wg02mma.wait(0, wg0_smem_ph.phase))
                                        _builder_emit(wg0_smem_ph.advance())
                                        _builder_emit(iket.range_end(mma_dq_ready_token))
                                        mma_dq_issue_token = _builder_assign(
                                            "mma_dq_issue_token",
                                            iket.range_start('mma-dq-issue'),
                                            _BUILDER_MISSING,
                                        )
                                        accum_var = _builder_assign("accum_var", 0, accum_var)
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma_dq(TMEM_OFF_DQ, accum_var, desc_ds_exch, desc_k_col))
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(mma2wg0_dq.arrive(0, cta_group=CTA_GROUP, cta_mask=pair_mask))
                                                _builder_emit(tcgen05_commit(ds_exch_consumed.ptr_to([0]), pair_mask))
                                        _builder_emit(iket.range_end(mma_dq_issue_token))
                                        _builder_emit(dq_tmem_free.wait(0, dq_tmem_free_ph.phase))
                                    _builder_emit(mma_n_tile())
                                with T.Else():
                                    with T.If(warp_id == 2):
                                        with T.Then():
                                            relay_ph = _builder_assign("relay_ph", PipelineState(1), locals().get("relay_ph", _BUILDER_MISSING))
                                            _builder_emit(relay_ph.init(0))

                                            def relay_n_tile():
                                                with T.serial(num_m_tiles_this_n, annotations={'disable_unroll': True}) as _:
                                                    IRBuilder.name("_", _)
                                                    _builder_emit(ds_exch_mbar.wait(0, relay_ph.phase))
                                                    _builder_emit(relay_ph.advance())
                                                    with T.If(T.cuda.elect_sync()):
                                                        with T.Then():
                                                            _builder_emit(wg02mma.arrive(0, remote=0, pred=True))
                                            _builder_emit(relay_n_tile())
                    with T.If(warp_id == 0):
                        with T.Then():
                            _builder_emit(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned())
                            _builder_emit(T.ptx.bar.sync(T.uint32(5), 416))
                            _builder_emit(tmem_dealloc_mbar.arrive(0, remote=1 - id_in_pair, pred=True))
                            _builder_emit(tmem_dealloc_mbar.wait(0, 0))
                            tmem_dealloc_addr = _builder_alloc_scalar("tmem_dealloc_addr", "uint32")
                            _builder_emit(T.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_addr.ptr_to([0])))
                            _builder_emit(T.ptx['tcgen05.dealloc.cta_group::2.sync.aligned.b32'](tmem_dealloc_addr, T.uint32(512)))
                with T.Else():
                    with T.If((wg_id >= 1) & (wg_id <= 2)):
                        with T.Then():
                            _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(136))
                            _builder_emit(T.ptx.barrier.sync(T.uint32(5), 416))
                            compute_wg = wg_id - 1
                            gemm_s_ph = _builder_assign("gemm_s_ph", PipelineState(1), locals().get("gemm_s_ph", _BUILDER_MISSING))
                            _builder_emit(gemm_s_ph.init(0))
                            lse_ph = _builder_assign("lse_ph", PipelineState(1), locals().get("lse_ph", _BUILDER_MISSING))
                            _builder_emit(lse_ph.init(0))
                            ds_exch_ph = _builder_assign("ds_exch_ph", PipelineState(1), locals().get("ds_exch_ph", _BUILDER_MISSING))
                            _builder_emit(ds_exch_ph.init(0))
                            gemm_dp_ph = _builder_assign("gemm_dp_ph", PipelineState(1), locals().get("gemm_dp_ph", _BUILDER_MISSING))
                            _builder_emit(gemm_dp_ph.init(0))
                            dpsum_ph = _builder_assign("dpsum_ph", PipelineState(1), locals().get("dpsum_ph", _BUILDER_MISSING))
                            _builder_emit(dpsum_ph.init(0))
                            dv_done_ph = _builder_assign("dv_done_ph", PipelineState(1), locals().get("dv_done_ph", _BUILDER_MISSING))
                            _builder_emit(dv_done_ph.init(0))
                            dk_done_ph = _builder_assign("dk_done_ph", PipelineState(1), locals().get("dk_done_ph", _BUILDER_MISSING))
                            _builder_emit(dk_done_ph.init(0))
                            ds_exch_consumed_ph = _builder_assign("ds_exch_consumed_ph", PipelineState(1), locals().get("ds_exch_consumed_ph", _BUILDER_MISSING))
                            _builder_emit(ds_exch_consumed_ph.init(1))
                            dS_sw = _builder_assign("dS_sw", RowiseSwizzleOffset(3, 3, 3, warp_id * 32 + lane_id, prefix='dS_sw'), locals().get("dS_sw", _BUILDER_MISSING))
                            _builder_emit(dS_sw.init())
                            epi_sw = _builder_assign("epi_sw", RowiseSwizzleOffset(3, 3, 3, warp_id * 32 + lane_id, prefix='epi_sw'), locals().get("epi_sw", _BUILDER_MISSING))
                            _builder_emit(epi_sw.init())
                            strip_off = compute_wg * STRIP_SIZE

                            def softmax_n_tile():
                                with T.serial(num_m_tiles_this_n, annotations={'disable_unroll': True}) as i_m_inner:
                                    IRBuilder.name("i_m_inner", i_m_inner)
                                    i_m = m_tile_start + i_m_inner
                                    m_st_val = i_m * BLK_M
                                    row_local = warp_id * 32 + lane_id
                                    softmax_p_token = _builder_assign("softmax_p_token", iket.range_start('softmax-p'), locals().get("softmax_p_token", _BUILDER_MISSING))
                                    _builder_emit(tma_lse.wait(0, lse_ph.phase))
                                    _builder_emit(mma2wg0_s.wait(0, gemm_s_ph.phase))
                                    _builder_emit(gemm_s_ph.advance())
                                    S_strip = _builder_assign("S_strip", T.alloc_local((STRIP_SIZE,), f32), locals().get("S_strip", _BUILDER_MISSING))
                                    tmem_s_col = TMEM_OFF_A + strip_off
                                    with T.unroll(STRIP_SIZE // 32) as stage:
                                        IRBuilder.name("stage", stage)
                                        _builder_emit(tmem_load_32(S_strip, stage * 32, TMEM_OFF_A, 0, tmem_s_col + stage * 32))
                                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                    with T.If((i_m_inner > 0) & (lane_id == 0)):
                                        with T.Then():
                                            _builder_emit(s_tmem_consumed.arrive(0, remote=0, pred=True))
                                    P_f16 = _builder_assign("P_f16", T.alloc_local((STRIP_SIZE,), f16), locals().get("P_f16", _BUILDER_MISSING))
                                    P_f16_u32 = _builder_assign("P_f16_u32", P_f16.view('uint32'), locals().get("P_f16_u32", _BUILDER_MISSING))
                                    tmem_p_col = TMEM_OFF_A * 2 + strip_off
                                    with T.unroll(STRIP_SIZE // 32) as stage:
                                        IRBuilder.name("stage", stage)
                                        with T.unroll(32 // 2) as j_inner:
                                            IRBuilder.name("j_inner", j_inner)
                                            j = stage * (32 // 2) + j_inner
                                            lse_pair_0 = _builder_alloc_scalar("lse_pair_0", "float32")
                                            lse_pair_1 = _builder_alloc_scalar("lse_pair_1", "float32")
                                            _builder_emit(T.ptx.ld.shared.v2.f32(lse_pair_0, lse_pair_1, sLSE.ptr_to([0, strip_off + 2 * j])))
                                            scaled_pair = _builder_bind("scaled_pair", fma_scale_sub_f32x2(T.cuda.make_float2(S_strip[2 * j], S_strip[2 * j + 1]), T.cuda.make_float2(T.float32(scale_log2), T.float32(scale_log2)), T.cuda.make_float2(lse_pair_0, lse_pair_1)), None)
                                            T.buffer_store(S_strip, T.cuda.float2_x(scaled_pair), [2 * j])
                                            T.buffer_store(S_strip, T.cuda.float2_y(scaled_pair), [2 * j + 1])
                                            _builder_emit(T.ptx.ex2.approx.ftz.f32(S_strip[2 * j], S_strip[2 * j]))
                                            _builder_emit(T.ptx.ex2.approx.ftz.f32(S_strip[2 * j + 1], S_strip[2 * j + 1]))
                                        if causal:
                                            with T.If(i_m < m_tile_start + BLK_N // BLK_M):
                                                with T.Then():
                                                    key_idx = n_st_cta + row_local
                                                    with T.unroll(32 // 2) as j_inner:
                                                        IRBuilder.name("j_inner", j_inner)
                                                        j = stage * (32 // 2) + j_inner
                                                        query_idx_0 = m_st_val + strip_off + 2 * j
                                                        query_idx_1 = query_idx_0 + 1
                                                        T.buffer_store(S_strip, T.if_then_else(query_idx_0 >= key_idx, S_strip[2 * j], T.float32(0)), [2 * j])
                                                        T.buffer_store(S_strip, T.if_then_else(query_idx_1 >= key_idx, S_strip[2 * j + 1], T.float32(0)), [2 * j + 1])
                                        with T.unroll(32 // 2) as j_inner:
                                            IRBuilder.name("j_inner", j_inner)
                                            j = stage * (32 // 2) + j_inner
                                            _builder_emit(cast_f32x2_to_f16x2(T.address_of(P_f16[2 * j]), T.address_of(S_strip[2 * j])))
                                        with T.If(stage == 0):
                                            with T.Then():
                                                _builder_emit(T.ptx.bar.sync(T.uint32(8), 256))
                                        _builder_emit(tmem_store_16(P_f16_u32, stage * 16, TMEM_OFF_A, 0, (tmem_p_col + stage * 32) // 2))
                                    _builder_emit(lse_ph.advance())
                                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                    _builder_emit(T.ptx.bar.sync(T.uint32(8), 256))
                                    with T.If(lane_id == 0):
                                        with T.Then():
                                            _builder_emit(lse_consumed.arrive(0))
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(strip_ready.arrive(0, remote=0, pred=True))
                                    _builder_emit(iket.range_end(softmax_p_token))
                                    softmax_ds_token = _builder_assign("softmax_ds_token", iket.range_start('softmax-ds'), locals().get("softmax_ds_token", _BUILDER_MISSING))
                                    _builder_emit(tma_dpsum.wait(0, dpsum_ph.phase))
                                    _builder_emit(mma2wg0_dp.wait(0, gemm_dp_ph.phase))
                                    _builder_emit(gemm_dp_ph.advance())
                                    dP_strip = _builder_assign("dP_strip", T.alloc_local((STRIP_SIZE,), f32), locals().get("dP_strip", _BUILDER_MISSING))
                                    dP_pairs = _builder_assign("dP_pairs", dP_strip.view('uint64'), locals().get("dP_pairs", _BUILDER_MISSING))
                                    tmem_dp_col = TMEM_OFF_DP + strip_off
                                    with T.unroll(STRIP_SIZE // 32) as stage:
                                        IRBuilder.name("stage", stage)
                                        _builder_emit(tmem_load_32(dP_strip, stage * 32, TMEM_OFF_A, 0, tmem_dp_col + stage * 32))
                                    with T.unroll(STRIP_SIZE // 2) as j:
                                        IRBuilder.name("j", j)
                                        dpsum_pair_0 = _builder_alloc_scalar("dpsum_pair_0", "float32")
                                        dpsum_pair_1 = _builder_alloc_scalar("dpsum_pair_1", "float32")
                                        _builder_emit(T.ptx.ld.shared.v2.f32(dpsum_pair_0, dpsum_pair_1, sDPsum.ptr_to([strip_off + 2 * j])))
                                        _builder_emit(T.ptx.sub.rn.ftz.f32x2(dP_pairs[j], T.cuda.make_float2(dP_strip[2 * j], dP_strip[2 * j + 1]), T.cuda.make_float2(dpsum_pair_0, dpsum_pair_1)))
                                        _builder_emit(T.ptx.mul.rn.ftz.f32x2(dP_pairs[j], T.cuda.make_float2(S_strip[2 * j], S_strip[2 * j + 1]), T.cuda.make_float2(dP_strip[2 * j], dP_strip[2 * j + 1])))
                                    _builder_emit(dpsum_ph.advance())
                                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                    dS_full_f16 = _builder_assign("dS_full_f16", T.alloc_local((STRIP_SIZE,), f16), locals().get("dS_full_f16", _BUILDER_MISSING))
                                    with T.unroll(STRIP_SIZE // 2) as j:
                                        IRBuilder.name("j", j)
                                        _builder_emit(cast_f32x2_to_f16x2(T.address_of(dS_full_f16[2 * j]), T.address_of(dP_strip[2 * j])))
                                    _builder_emit(T.ptx.bar.sync(T.uint32(8), 256))
                                    tmem_ds_col = TMEM_OFF_DP * 2 + strip_off
                                    dS_f16_u32 = _builder_assign("dS_f16_u32", dS_full_f16.view('uint32'), locals().get("dS_f16_u32", _BUILDER_MISSING))
                                    _builder_emit(tmem_store_32(dS_f16_u32, 0, TMEM_OFF_A, 0, tmem_ds_col // 2))
                                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(wg02mma_tmem.arrive(0, remote=0, pred=True))
                                    _builder_emit(iket.range_end(softmax_ds_token))
                                    ds_exchange_token = _builder_assign("ds_exchange_token", iket.range_start('ds-exchange'), locals().get("ds_exchange_token", _BUILDER_MISSING))
                                    _builder_emit(ds_exch_consumed.wait(0, ds_exch_consumed_ph.phase))
                                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                    _builder_emit(ds_exch_consumed_ph.advance())
                                    with T.If(compute_wg == id_in_pair):
                                        with T.Then():
                                            ds_row_base = id_in_pair * CTA_N
                                            ds_row_st = _builder_alloc_scalar("ds_row_st", "int32")
                                            ds_row_st = _builder_assign("ds_row_st", (ds_row_base + row_local) * B_N + (ds_row_base + row_local & 7) * 8, locals().get("ds_row_st", _BUILDER_MISSING))
                                            with T.unroll(STRIP_SIZE // 8) as ni:
                                                IRBuilder.name("ni", ni)
                                                _builder_emit(copy_128b(pointer_offset(dS_exch.ptr_to([0, 0]), ds_row_st + dS_sw.apply(ni * 8)), dS_full_f16.view('uint128')[ni]))
                                        with T.Else():
                                            stage_row_st = _builder_alloc_scalar("stage_row_st", "int32")
                                            stage_row_st = _builder_assign("stage_row_st", row_local * B_N + (row_local & 7) * 8, locals().get("stage_row_st", _BUILDER_MISSING))
                                            with T.unroll(STRIP_SIZE // 8) as ni:
                                                IRBuilder.name("ni", ni)
                                                _builder_emit(copy_128b(pointer_offset(dS_send.ptr_to([0, 0]), stage_row_st + dS_sw.apply(ni * 8)), dS_full_f16.view('uint128')[ni]))
                                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                    _builder_emit(T.ptx.bar.sync(T.uint32(8), 256))
                                    with T.If(lane_id == 0):
                                        with T.Then():
                                            _builder_emit(dpsum_consumed.arrive(0))
                                    with T.If((compute_wg != id_in_pair) & (warp_id == 0) & (lane_id == 0)):
                                        with T.Then():
                                            peer_cta = _builder_alloc_scalar("peer_cta", "int32")
                                            peer_cta = _builder_assign("peer_cta", 1 - id_in_pair, locals().get("peer_cta", _BUILDER_MISSING))
                                            ds_copy_bytes = CTA_N * B_N * DTYPE_SIZE
                                            remote_mbar = _builder_assign("remote_mbar", T.alloc_local([1], 'uint32'), locals().get("remote_mbar", _BUILDER_MISSING))
                                            _builder_emit(T.ptx.mapa.shared__cluster.u32(remote_mbar[0], T.cuda.cvta_generic_to_shared(ds_exch_mbar.ptr_to([0])), T.uint32(peer_cta)))
                                            remote_dst = _builder_assign("remote_dst", T.alloc_local([1], 'uint32'), locals().get("remote_dst", _BUILDER_MISSING))
                                            _builder_emit(T.ptx.mapa.shared__cluster.u32(remote_dst[0], T.cuda.cvta_generic_to_shared(dS_exch.ptr_to([id_in_pair * CTA_N, 0])), T.uint32(peer_cta)))
                                            _builder_emit(T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(remote_mbar[0], T.uint32(ds_copy_bytes), pred=True))
                                            _builder_emit(T.ptx[_BULK_S2C](remote_dst[0], dS_send.ptr_to([0, 0]), T.uint32(ds_copy_bytes), remote_mbar[0]))
                                    _builder_emit(iket.range_end(ds_exchange_token))
                                dkv_epilogue_token = _builder_assign("dkv_epilogue_token", iket.range_start('dkv-epilogue'), locals().get("dkv_epilogue_token", _BUILDER_MISSING))
                                _builder_emit(dv_done.wait(0, dv_done_ph.phase))
                                _builder_emit(dv_done_ph.advance())
                                dv_epi_strip = _builder_assign("dv_epi_strip", T.alloc_local((EPI_N,), f32), locals().get("dv_epi_strip", _BUILDER_MISSING))
                                dv_epi_f16 = _builder_assign("dv_epi_f16", T.alloc_local((EPI_N,), f16), locals().get("dv_epi_f16", _BUILDER_MISSING))
                                _builder_emit(tmem_load_64(dv_epi_strip, 0, TMEM_OFF_A, 0, TMEM_OFF_B + compute_wg * EPI_N))
                                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                with T.unroll(EPI_N // 2) as j:
                                    IRBuilder.name("j", j)
                                    _builder_emit(cast_f32x2_to_f16x2(T.address_of(dv_epi_f16[2 * j]), T.address_of(dv_epi_strip[2 * j])))
                                epi_row_st = _builder_alloc_scalar("epi_row_st", "int32")
                                epi_row_st = _builder_assign("epi_row_st", (warp_id * 32 + lane_id) * EPI_N + (warp_id * 32 + lane_id & 7) * 8, locals().get("epi_row_st", _BUILDER_MISSING))
                                with T.unroll(EPI_N // 8) as ni:
                                    IRBuilder.name("ni", ni)
                                    _builder_emit(copy_128b(pointer_offset(dV_epi.ptr_to([compute_wg, 0, 0]), epi_row_st + epi_sw.apply(ni * 8)), dv_epi_f16.view('uint128')[ni]))
                                _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                _builder_emit(T.ptx.bar.sync(T.uint32(wg_id + 10), 128))
                                with T.If((warp_id == 0) & (lane_id == 0)):
                                    with T.Then():
                                        _builder_emit(tma_s2g(4, dV_epi.ptr_to([compute_wg, 0, 0]), T.address_of(dv_tensormap), compute_wg * EPI_N, n_st_cta, h_idx, b_idx))
                                with T.If(warp_id == 0):
                                    with T.Then():
                                        _builder_emit(T.ptx.bar.arrive(T.uint32(wg_id + 10), 160))
                                _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                _builder_emit(T.ptx.bar.sync(T.uint32(wg_id + 10), 160))
                                _builder_emit(dk_done.wait(0, dk_done_ph.phase))
                                _builder_emit(dk_done_ph.advance())
                                dk_epi_strip = _builder_assign("dk_epi_strip", T.alloc_local((EPI_N,), f32), locals().get("dk_epi_strip", _BUILDER_MISSING))
                                dk_epi_pairs = _builder_assign("dk_epi_pairs", dk_epi_strip.view('uint64'), locals().get("dk_epi_pairs", _BUILDER_MISSING))
                                dk_epi_f16 = _builder_assign("dk_epi_f16", T.alloc_local((EPI_N,), f16), locals().get("dk_epi_f16", _BUILDER_MISSING))
                                _builder_emit(tmem_load_64(dk_epi_strip, 0, TMEM_OFF_A, 0, TMEM_OFF_C + compute_wg * EPI_N))
                                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                with T.unroll(EPI_N // 2) as j:
                                    IRBuilder.name("j", j)
                                    _builder_emit(T.ptx.mul.rn.ftz.f32x2(dk_epi_pairs[j], T.cuda.make_float2(dk_epi_strip[2 * j], dk_epi_strip[2 * j + 1]), T.cuda.make_float2(T.float32(softmax_scale), T.float32(softmax_scale))))
                                    _builder_emit(cast_f32x2_to_f16x2(T.address_of(dk_epi_f16[2 * j]), T.address_of(dk_epi_strip[2 * j])))
                                with T.unroll(EPI_N // 8) as ni:
                                    IRBuilder.name("ni", ni)
                                    _builder_emit(copy_128b(pointer_offset(dK_epi.ptr_to([compute_wg, 0, 0]), epi_row_st + epi_sw.apply(ni * 8)), dk_epi_f16.view('uint128')[ni]))
                                _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                _builder_emit(T.ptx.bar.sync(T.uint32(wg_id + 10), 128))
                                with T.If((warp_id == 0) & (lane_id == 0)):
                                    with T.Then():
                                        _builder_emit(tma_s2g(4, dK_epi.ptr_to([compute_wg, 0, 0]), T.address_of(dk_tensormap), compute_wg * EPI_N, n_st_cta, h_idx, b_idx))
                                with T.If(warp_id == 0):
                                    with T.Then():
                                        _builder_emit(T.ptx.bar.arrive(T.uint32(wg_id + 10), 160))
                                _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                _builder_emit(T.ptx.bar.sync(T.uint32(wg_id + 10), 160))
                                with T.If((warp_id == 0) & (lane_id == 0)):
                                    with T.Then():
                                        _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                                _builder_emit(iket.range_end(dkv_epilogue_token))
                            _builder_emit(softmax_n_tile())
                            _builder_emit(T.ptx.bar.arrive(T.uint32(5), 416))
                        with T.Else():
                            _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(136))
                            _builder_emit(T.ptx.barrier.sync(T.uint32(5), 416))
                            gemm_dq_ph_wg3 = _builder_assign("gemm_dq_ph_wg3", PipelineState(1), locals().get("gemm_dq_ph_wg3", _BUILDER_MISSING))
                            _builder_emit(gemm_dq_ph_wg3.init(0))

                            def wg3_n_tile():
                                with T.serial(num_m_tiles_this_n, annotations={'disable_unroll': True}) as i_m_inner:
                                    IRBuilder.name("i_m_inner", i_m_inner)
                                    i_m = m_tile_start + i_m_inner
                                    m_st_val = i_m * BLK_M
                                    row_local = warp_id * 32 + lane_id
                                    dq_reduce_token = _builder_assign("dq_reduce_token", iket.range_start('dq-reduce'), locals().get("dq_reduce_token", _BUILDER_MISSING))
                                    _builder_emit(mma2wg0_dq.wait(0, gemm_dq_ph_wg3.phase))
                                    _builder_emit(gemm_dq_ph_wg3.advance())
                                    dQ_full = _builder_assign("dQ_full", T.alloc_local((64,), f32), locals().get("dQ_full", _BUILDER_MISSING))
                                    _builder_emit(tmem_load_32(dQ_full, 32, TMEM_OFF_DQ, warp_id * 32, 0))
                                    _builder_emit(tmem_load_32(dQ_full, 0, TMEM_OFF_DQ, warp_id * 32, 32))
                                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                    _builder_emit(T.cuda.warp_sync())
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(dq_tmem_free.arrive(0, remote=0, pred=True))
                                    m_st_cta = m_st_val + id_in_pair * DQ_M_PER_CTA
                                    dq_reduce_elected = _builder_bind("dq_reduce_elected", T.cuda.elect_sync(), None)
                                    with T.unroll(DQ_REDUCE_ITERS) as stage:
                                        IRBuilder.name("stage", stage)
                                        dq_reduce_stage_token = _builder_assign("dq_reduce_stage_token", iket.range_start('dq-reduce-stage'), locals().get("dq_reduce_stage_token", _BUILDER_MISSING))
                                        smem_slot = stage % DQ_STAGES
                                        dq_stage_st = _builder_assign("dq_stage_st", smem_slot * BLK_M * DQ_RED_N, locals().get("dq_stage_st", _BUILDER_MISSING))
                                        dq_reg_st = (stage + DQ_REDUCE_ITERS // 2) % DQ_REDUCE_ITERS * DQ_RED_N
                                        with T.unroll(DQ_RED_N // 4) as chunk:
                                            IRBuilder.name("chunk", chunk)
                                            _builder_emit(copy_128b(pointer_offset_f32(dQ_smem.ptr_to([0, 0, 0]), dq_stage_st + chunk * BLK_M * 4 + row_local * 4), dQ_full.view('uint128')[(dq_reg_st + chunk * 4) // 4]))
                                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                        _builder_emit(T.cuda.warpgroup_sync(4))
                                        with T.If(warp_id == 0):
                                            with T.Then():
                                                with T.If(dq_reduce_elected):
                                                    with T.Then():
                                                        _builder_emit(tma_s2g_reduce(4, dQ_smem.ptr_to([smem_slot, 0, 0]), T.address_of(dq_tensormap), 'add', 0, m_st_cta + stage * DQ_ROWS_PER_STAGE, h_idx, b_idx))
                                                _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                                                _builder_emit(T.ptx.cp.async_.bulk.wait_group(DQ_STAGES - 1))
                                        _builder_emit(T.cuda.warpgroup_sync(4))
                                        _builder_emit(iket.range_end(dq_reduce_stage_token))
                                    _builder_emit(iket.range_end(dq_reduce_token))
                                with T.If(warp_id == 0):
                                    with T.Then():
                                        _builder_emit(T.ptx.cp.async_.bulk.wait_group(0))
                                _builder_emit(T.cuda.warpgroup_sync(4))
                            _builder_emit(wg3_n_tile())
                            _builder_emit(T.ptx.bar.arrive(T.uint32(5), 416))
    # fmt: on
    return builder.get()


# ---------------------------------------------------------------------------
# setup() — compile all kernels, run once, return kernel_fn
# ---------------------------------------------------------------------------


@cache
def _compile_pipeline(B: int, H: int, S: int, D: int, causal: bool, attention_scale: float):
    from tirx_kernels.runner import cuda_target

    preprocess_ex = build_preprocess(B, S, H, D)
    cast_ex = build_cast_f32_to_f16(B, S, H, D, attention_scale)
    target = cuda_target()
    with target:
        kernel_func = build_kernel(B, H, S, D, causal=causal, attention_scale=attention_scale)
        kernel_mod = tvm.IRModule({"main": kernel_func})
        kernel_ex = tvm.compile(kernel_mod, target=target, tir_pipeline="tirx")
    return preprocess_ex, kernel_ex, cast_ex


def setup(data, B, H, S, D, *, executables=None):
    """Prepare GPU data and return one full backward-pipeline launch."""
    Q = data["Q"]
    K = data["K"]
    V = data["V"]
    O = data["O"]
    dO = data["dO"]
    LSE = data["LSE"]
    causal = bool(data.get("causal", False))
    attention_scale = float(data.get("softmax_scale", 1.0 / math.sqrt(D)))

    dpsum = torch.empty(B, H, S, dtype=torch.float32, device="cuda")
    LSE_log2 = torch.empty_like(LSE)
    dQ_acc = torch.empty(B, H, S, D, dtype=torch.float32, device="cuda")
    dK = data["dK"]
    dV = data["dV"]
    dQ = data["dQ"]
    if executables is None:
        executables = _compile_pipeline(B, H, S, D, causal, attention_scale)
    preprocess_ex, kernel_ex, cast_ex = executables

    def run_all():
        preprocess_ex(dO, O, LSE, dpsum, LSE_log2, dQ_acc)
        kernel_ex(Q, K, V, dO, LSE_log2, dpsum, dK, dV, dQ_acc)
        cast_ex(dQ_acc, dQ)

    run_all()

    data["dQ"] = dQ
    data["dK"] = dK
    data["dV"] = dV

    def kernel_fn():
        run_all()

    return kernel_fn


KERNEL_META = {
    "name": "flash_attention_backward_sm100",
    "category": "flashattention",
    "compute_capability": 10,
}

CONFIGS = [
    {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": 16,
        "head_dim": 128,
        "is_causal": is_causal,
        "label": (f"b{batch_size}_s{seq_len}_h16_{'causal' if is_causal else 'noncausal'}"),
    }
    for batch_size, seq_len, is_causal in (
        (1, 2048, True),
        (1, 4096, True),
        (2, 4096, True),
        (1, 8192, True),
        (1, 8192, False),
        # flash-attn's own sweep has no fixed config list -- benchmark_attn.py
        # takes every shape from the command line -- so "the official shape" is
        # its default: --headdim (128,128), --seqlen 8192, --nheads unset (16
        # for head_dim <= 192), --total-seqlen 32k giving batch 32k // 8192,
        # and --causal both. These two rows are that default. We still generate
        # fp16 where it generates bf16.
        (4, 8192, True),
        (4, 8192, False),
    )
]


def get_kernel(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Return the raw backward core PrimFunc for the registry configuration."""
    return build_kernel(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        causal=is_causal,
        attention_scale=1.0 / math.sqrt(head_dim),
    )


def _prepare_official_workload(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int, is_causal: bool
):
    """Create saved forward tensors and the current FA4 backward reference."""
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    torch.manual_seed(0)
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    k = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    v = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.5).half()
    dout = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.25).half()
    scale = 1.0 / math.sqrt(head_dim)
    with torch.no_grad():
        out, lse = _flash_attn_fwd(
            q=q, k=k, v=v, softmax_scale=scale, causal=is_causal, return_lse=True
        )[:2]
        expected = _flash_attn_bwd(
            q=q, k=k, v=v, out=out, dout=dout, lse=lse, softmax_scale=scale, causal=is_causal
        )
    data = {
        "Q": q,
        "K": k,
        "V": v,
        "O": out,
        "dO": dout,
        "LSE": lse,
        "dQ": torch.empty_like(q),
        "dK": torch.empty_like(k),
        "dV": torch.empty_like(v),
        "causal": is_causal,
        "softmax_scale": scale,
    }
    return data, expected


def run_test(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Compile the three-kernel pipeline and compare it with current FA4."""
    data, expected = _prepare_official_workload(batch_size, seq_len, num_heads, head_dim, is_causal)
    setup(data, batch_size, num_heads, seq_len, head_dim)
    torch.cuda.synchronize()
    for name, actual, reference in zip(
        ("dQ", "dK", "dV"), (data["dQ"], data["dK"], data["dV"]), expected, strict=True
    ):
        if not torch.isfinite(actual).all():
            raise AssertionError(f"{name} contains a non-finite value")
        matched = torch.isclose(actual, reference, rtol=0.1, atol=0.1).float().mean()
        if matched.item() < 0.995:
            raise AssertionError(f"{name} matched ratio {matched.item():.6f} is below 0.995")


def prepare_bench(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    **kwargs,
):
    """Compile the preprocess/core/cast pipeline before CUDA setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    scale = 1.0 / math.sqrt(head_dim)
    state = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "is_causal": is_causal,
            **kwargs,
        },
        "executables": _compile_pipeline(
            batch_size, num_heads, seq_len, head_dim, is_causal, scale
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark the full preprocess/core/cast pipeline against current FA4."""
    from flash_attn.cute.interface import _flash_attn_bwd

    from tirx_kernels.runner import bench

    config = {**prepared["config"], **kwargs}
    batch_size = config.pop("batch_size")
    seq_len = config.pop("seq_len")
    num_heads = config.pop("num_heads")
    head_dim = config.pop("head_dim")
    is_causal = config.pop("is_causal")
    data, _ = _prepare_official_workload(batch_size, seq_len, num_heads, head_dim, is_causal)
    candidate = setup(
        data, batch_size, num_heads, seq_len, head_dim, executables=prepared["executables"]
    )

    def official_factory():
        def run():
            return _flash_attn_bwd(
                q=data["Q"],
                k=data["K"],
                v=data["V"],
                out=data["O"],
                dout=data["dO"],
                lse=data["LSE"],
                softmax_scale=data["softmax_scale"],
                causal=data["causal"],
            )

        return run

    return bench(
        {"tir": candidate},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": official_factory},
        **config,
    )


def run_bench(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int = 128,
    is_causal: bool = False,
    warmup=None,
    repeat=None,
    timer=None,
    **kwargs,
):
    rounds = kwargs.pop("rounds", None)
    cooldown_s = kwargs.pop("cooldown_s", None)
    protocol = {"warmup": warmup, "repeat": repeat, "timer": timer}
    if rounds is not None:
        protocol["rounds"] = rounds
    if cooldown_s is not None:
        protocol["cooldown_s"] = cooldown_s
    return prepare_bench(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        **kwargs,
    ).run_gpu(**protocol)
