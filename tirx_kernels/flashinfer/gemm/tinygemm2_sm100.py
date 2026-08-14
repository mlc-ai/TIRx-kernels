# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer TinyGEMM2 BF16 kernel port for SM100.

Upstream source: csrc/tinygemm2_sm100.cu.
"""

from __future__ import annotations

import ctypes
import hashlib
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

import tvm
from tvm.ir.type import PointerType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value

KERNEL_META = {"name": "tinygemm2_sm100", "category": "flashinfer", "compute_capability": 10}

CONFIGS = [
    {"label": "b1_o128_k720", "B": 1, "O": 128, "K": 720},
    {"label": "b2_o16_k256", "B": 2, "O": 16, "K": 256},
    {"label": "b4_o2880_k2880", "B": 4, "O": 2880, "K": 2880},
    {"label": "b7_o128_k4096", "B": 7, "O": 128, "K": 4096},
    {"label": "b8_o1024_k1024", "B": 8, "O": 1024, "K": 1024},
    {"label": "b13_o1024_k2048", "B": 13, "O": 1024, "K": 2048},
    {"label": "b16_o2880_k2880", "B": 16, "O": 2880, "K": 2880},
    {"label": "b64_o4096_k3072", "B": 64, "O": 4096, "K": 3072},
]

BENCH_CONFIGS = CONFIGS

THREADS = 384
WT_OFF = 1024
WT_STAGE_BYTES = 4 * 2048
ACT_STAGE_BYTES = 4 * 1024
RED_BYTES = 2048
BIAS_BYTES = 32
SOURCE_SHA256 = "ea4d87f058b269e2f15d04f945849d7b26604c5b76eed55a06eb8e08d7bc891d"
_TMA_G2S_2D = "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"

_BUILDER_MISSING = object()


def _builder_runtime_condition(value):
    return value


def _builder_enter(frame):
    """Enter a parser-flat builder frame until the enclosing PrimFunc exits."""
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
    """Emit the expression-statement cases handled implicitly by TVMScript."""
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


def _builder_buffer(name, shape, dtype):
    buffer = T.alloc_local(shape, dtype)
    IRBuilder.name(name, buffer)
    return buffer


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_assign(name, value, previous=_BUILDER_MISSING):
    """Match parser assignment: meta unwrap, resource name, or mutable scalar."""
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


def _select_stage(B: int, O: int, K: int, num_sms: int) -> int:
    """Mirror FlashInfer's B200 stage-ring dispatch predicate."""
    total_ctas = (O + 15) // 16 * ((B + 7) // 8)
    return 4 if K <= 1024 or total_ctas > 2 * num_sms else 8


def _validate_problem(B: int, O: int, K: int) -> None:
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if K < 64:
        raise ValueError(f"K must be at least 64, got {K}")
    if O < 16 or O % 16:
        raise ValueError(f"O must be a positive multiple of 16, got {O}")
    i32_max = (1 << 31) - 1
    if B > i32_max or O > i32_max or K > i32_max:
        raise ValueError("B, O, and K must fit signed int32")


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("TinyGEMM2 SM100 requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise SkipTest(
            f"TinyGEMM2 SM100 requires SM100/B200, got sm_{capability[0]}{capability[1]}"
        )


def _make_warp_uniform(value):
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), value, 0, 32)


def _mbarrier_wait(smem_raw, byte_offset, phase):
    T.evaluate(T.cuda.mbarrier_wait(smem_raw.ptr_to([byte_offset]), phase))


def _mbarrier_wait_address(address, phase):
    T.evaluate(T.cuda.mbarrier_wait(address, phase))


def _mbarrier_arrive(smem_raw, byte_offset):
    return T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(smem_raw.ptr_to([byte_offset]))


def _mbarrier_arrive_address(address):
    return T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(address)


def _mbarrier_expect_tx(smem_raw, byte_offset, num_bytes):
    return T.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
        smem_raw.ptr_to([byte_offset]), T.uint32(num_bytes)
    )


def _tma_2d_g2s(smem_raw, dst_offset, tensor_map, x, y, barrier_offset):
    return T.ptx[_TMA_G2S_2D](
        smem_raw.ptr_to([dst_offset]),
        T.address_of(tensor_map),
        x,
        y,
        smem_raw.ptr_to([barrier_offset]),
    )


def _mma_bf16(accum, a_frag, b_frag):
    return T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        accum[0],
        accum[1],
        accum[2],
        accum[3],
        a_frag[0],
        a_frag[1],
        a_frag[2],
        a_frag[3],
        b_frag[0],
        b_frag[1],
        accum[0],
        accum[1],
        accum[2],
        accum[3],
    )


def _tinygemm2_sm100(*, STAGES, USE_PDL, GRID_X, GRID_Y):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_tinygemm2_sm100")
            a_tmap_wt = T.arg("a_tmap_wt", T.TensorMap())
            b_tmap_act = T.arg("b_tmap_act", T.TensorMap())
            c_output_h = T.arg("c_output_h", T.handle())
            d_bias_h = T.arg("d_bias_h", T.handle())
            a_M = T.arg("a_M", T.int32())
            b_N = T.arg("b_N", T.int32())
            c_K = T.arg("c_K", T.int32())
            c_output = _builder_assign(
                "c_output",
                T.match_buffer(c_output_h, (b_N * a_M,), "bfloat16"),
                locals().get("c_output", _BUILDER_MISSING),
            )
            d_bias = _builder_assign(
                "d_bias",
                T.match_buffer(d_bias_h, (a_M,), "bfloat16"),
                locals().get("d_bias", _BUILDER_MISSING),
            )
            _builder_emit(T.device_entry())
            _builder_emit(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            tid_u32 = _builder_assign(
                "tid_u32",
                T.thread_id([THREADS], dtype="uint32"),
                locals().get("tid_u32", _BUILDER_MISSING),
            )
            tid = _builder_scalar("tid", T.cast(tid_u32, "int32"), "int32")
            warp = _builder_assign(
                "warp", _make_warp_uniform(tid // 32), locals().get("warp", _BUILDER_MISSING)
            )
            lane = _builder_assign("lane", tid % 32, locals().get("lane", _BUILDER_MISSING))
            lane_u32 = _builder_scalar("lane_u32", tid_u32 % T.uint32(32), "uint32")
            block_m, block_n = _builder_assign_many(
                ("block_m", "block_n"),
                T.cta_id([GRID_X, GRID_Y]),
                (
                    locals().get("block_m", _BUILDER_MISSING),
                    locals().get("block_n", _BUILDER_MISSING),
                ),
            )
            act_off = 33792 if STAGES == 4 else 66560
            red_off = 50176 if STAGES == 4 else 99328
            bias_off = 52224 if STAGES == 4 else 101376
            smem_total = 52352 if STAGES == 4 else 101504
            act_ready_off = 32 if STAGES == 4 else 64
            consumed_off = 64 if STAGES == 4 else 128
            pool = _builder_assign("pool", T.SMEMPool(), locals().get("pool", _BUILDER_MISSING))
            smem_raw = _builder_assign(
                "smem_raw",
                pool.alloc((smem_total,), "uint8", align=1024),
                locals().get("smem_raw", _BUILDER_MISSING),
            )
            smem_bias = _builder_assign(
                "smem_bias",
                T.decl_buffer(
                    (BIAS_BYTES // 2,),
                    "bfloat16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=bias_off,
                    align=2,
                ),
                locals().get("smem_bias", _BUILDER_MISSING),
            )
            _builder_emit(pool.commit())
            with T.If(tid == 0):
                with T.Then():
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(a_tmap_wt))))
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(b_tmap_act))))
            with T.If(warp == 0):
                with T.Then():
                    leader = _builder_assign(
                        "leader", T.cuda.elect_sync(), locals().get("leader", _BUILDER_MISSING)
                    )
                    with T.unroll(STAGES) as stage_init:
                        IRBuilder.name("stage_init", stage_init)
                        _builder_emit(
                            T.ptx.mbarrier.init.shared__cta.b64(
                                smem_raw.ptr_to([stage_init * 8]), T.uint32(1), pred=leader
                            )
                        )
                    with T.unroll(STAGES) as stage_init:
                        IRBuilder.name("stage_init", stage_init)
                        _builder_emit(
                            T.ptx.mbarrier.init.shared__cta.b64(
                                smem_raw.ptr_to([act_ready_off + stage_init * 8]),
                                T.uint32(1),
                                pred=leader,
                            )
                        )
                    with T.unroll(STAGES) as stage_init:
                        IRBuilder.name("stage_init", stage_init)
                        _builder_emit(
                            T.ptx.mbarrier.init.shared__cta.b64(
                                smem_raw.ptr_to([consumed_off + stage_init * 8]),
                                T.uint32(32),
                                pred=leader,
                            )
                        )
                    _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(T.ptx.bar.sync(T.uint32(0)))
            _builder_emit(T.ptx.bar.sync(T.uint32(0)))
            with T.If(warp <= 3):
                with T.Then():
                    k_loops_c = _builder_scalar("k_loops_c", T.truncdiv(c_K + 1023, 1024), "int32")
                    mib_c = _builder_scalar("mib_c", block_m * 16, "int32")
                    ni_c = _builder_scalar("ni_c", block_n * 8, "int32")
                    smem_addr_c = _builder_scalar(
                        "smem_addr_c", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), "uint32"
                    )
                    smem_wt_addr_c = _builder_scalar(
                        "smem_wt_addr_c", smem_addr_c + T.uint32(WT_OFF), "uint32"
                    )
                    smem_act_addr_c = _builder_scalar(
                        "smem_act_addr_c", smem_addr_c + T.uint32(act_off), "uint32"
                    )
                    with T.If(tid < 16):
                        with T.Then():
                            T.buffer_store(smem_bias, d_bias[mib_c + tid], [tid])
                    accum = _builder_assign(
                        "accum",
                        T.alloc_local((4,), "float32", align=4),
                        locals().get("accum", _BUILDER_MISSING),
                    )
                    with T.unroll(4) as z:
                        IRBuilder.name("z", z)
                        T.buffer_store(accum, T.float32(0), [z])
                    lane_div8 = _builder_scalar("lane_div8", lane_u32 // T.uint32(8), "uint32")
                    lane_mod8 = _builder_scalar("lane_mod8", lane_u32 % T.uint32(8), "uint32")
                    row_wt = _builder_scalar(
                        "row_wt", lane_mod8 + lane_div8 % T.uint32(2) * T.uint32(8), "uint32"
                    )
                    col_off_wt = _builder_scalar("col_off_wt", lane_div8 // T.uint32(2), "uint32")
                    row_act = _builder_scalar("row_act", lane_mod8, "uint32")

                    def compute_iter(ki):
                        if STAGES == 4:
                            stage_c = _builder_scalar("stage_c", T.cast(warp, "uint32"), "uint32")
                            phase_c = _builder_scalar("phase_c", ki & T.uint32(1), "uint32")
                        else:
                            stage_c = _builder_scalar(
                                "stage_c",
                                T.cast(warp, "uint32") + T.uint32(4) * (ki % T.uint32(2)),
                                "uint32",
                            )
                            phase_c = _builder_scalar(
                                "phase_c", ki // T.uint32(2) & T.uint32(1), "uint32"
                            )
                        _builder_emit(
                            _mbarrier_wait_address(smem_addr_c + stage_c * T.uint32(8), phase_c)
                        )
                        _builder_emit(
                            _mbarrier_wait_address(
                                smem_addr_c + T.uint32(act_ready_off) + stage_c * T.uint32(8),
                                phase_c,
                            )
                        )
                        with T.unroll(4) as su:
                            IRBuilder.name("su", su)
                            base_wt = _builder_scalar(
                                "base_wt",
                                smem_wt_addr_c
                                + (stage_c * T.uint32(4) + T.uint32(su)) * T.uint32(2048),
                                "uint32",
                            )
                            base_act = _builder_scalar(
                                "base_act",
                                smem_act_addr_c
                                + (stage_c * T.uint32(4) + T.uint32(su)) * T.uint32(1024),
                                "uint32",
                            )
                            with T.unroll(4) as kii:
                                IRBuilder.name("kii", kii)
                                a_frag = _builder_assign(
                                    "a_frag",
                                    T.alloc_local((4,), "uint32", align=4),
                                    locals().get("a_frag", _BUILDER_MISSING),
                                )
                                b_frag = _builder_assign(
                                    "b_frag",
                                    T.alloc_local((2,), "uint32", align=4),
                                    locals().get("b_frag", _BUILDER_MISSING),
                                )
                                col_w = _builder_scalar(
                                    "col_w", T.uint32(2 * kii) + col_off_wt, "uint32"
                                )
                                col_sw_w = _builder_scalar(
                                    "col_sw_w", row_wt % T.uint32(8) ^ col_w, "uint32"
                                )
                                _builder_emit(
                                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                                        a_frag[0],
                                        a_frag[1],
                                        a_frag[2],
                                        a_frag[3],
                                        base_wt + row_wt * T.uint32(128) + col_sw_w * T.uint32(16),
                                    )
                                )
                                col_a = _builder_scalar(
                                    "col_a", T.uint32(2 * kii) + lane_div8, "uint32"
                                )
                                col_sw_a = _builder_scalar(
                                    "col_sw_a", row_act % T.uint32(8) ^ col_a, "uint32"
                                )
                                _builder_emit(
                                    T.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(
                                        b_frag[0],
                                        b_frag[1],
                                        base_act
                                        + row_act * T.uint32(128)
                                        + col_sw_a * T.uint32(16),
                                    )
                                )
                                _builder_emit(_mma_bf16(accum, a_frag, b_frag))
                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                        _builder_emit(
                            _mbarrier_arrive_address(
                                smem_addr_c + T.uint32(consumed_off) + stage_c * T.uint32(8)
                            )
                        )

                    with T.serial(0, k_loops_c, unroll=2, dtype="uint32") as ki:
                        IRBuilder.name("ki", ki)
                        _builder_emit(compute_iter(ki))
                    accum_bits = _builder_assign(
                        "accum_bits",
                        T.alloc_local((4,), "uint32", align=4),
                        locals().get("accum_bits", _BUILDER_MISSING),
                    )
                    with T.unroll(4) as z:
                        IRBuilder.name("z", z)
                        T.buffer_store(accum_bits, T.reinterpret("uint32", accum[z]), [z])
                    _builder_emit(
                        T.ptx.st.shared.v4.b32(
                            smem_addr_c + T.uint32(red_off) + tid_u32 * T.uint32(16),
                            accum_bits[0],
                            accum_bits[1],
                            accum_bits[2],
                            accum_bits[3],
                        )
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(2), T.uint32(THREADS)))
                    with T.If(warp == 0):
                        with T.Then():
                            part_bits = _builder_assign(
                                "part_bits",
                                T.alloc_local((12,), "uint32", align=4),
                                locals().get("part_bits", _BUILDER_MISSING),
                            )
                            part = _builder_assign(
                                "part",
                                part_bits.view("float32"),
                                locals().get("part", _BUILDER_MISSING),
                            )
                            with T.unroll(3) as other_warp:
                                IRBuilder.name("other_warp", other_warp)
                                _builder_emit(
                                    T.ptx.ld.shared.v4.b32(
                                        part_bits[other_warp * 4],
                                        part_bits[other_warp * 4 + 1],
                                        part_bits[other_warp * 4 + 2],
                                        part_bits[other_warp * 4 + 3],
                                        smem_addr_c
                                        + T.uint32(red_off)
                                        + (T.uint32(32 + other_warp * 32) + tid_u32) * T.uint32(16),
                                    )
                                )
                            with T.unroll(4) as z:
                                IRBuilder.name("z", z)
                                _builder_emit(T.ptx["add.ftz.f32"](accum[z], accum[z], part[z]))
                                _builder_emit(T.ptx["add.ftz.f32"](accum[z], accum[z], part[4 + z]))
                                _builder_emit(T.ptx["add.ftz.f32"](accum[z], accum[z], part[8 + z]))
                            tm = _builder_scalar("tm", mib_c + lane // 4, "int32")
                            tn = _builder_scalar("tn", ni_c + 2 * (lane % 4), "int32")
                            bias_lo = _builder_scalar(
                                "bias_lo", T.cast(smem_bias[lane // 4], "float32"), "float32"
                            )
                            bias_hi = _builder_scalar(
                                "bias_hi", T.cast(smem_bias[lane // 4 + 8], "float32"), "float32"
                            )
                            out_frag = _builder_assign(
                                "out_frag",
                                T.alloc_local((4,), "float32", align=4),
                                locals().get("out_frag", _BUILDER_MISSING),
                            )
                            _builder_emit(T.ptx["add.ftz.f32"](out_frag[0], accum[0], bias_lo))
                            _builder_emit(T.ptx["add.ftz.f32"](out_frag[1], accum[1], bias_lo))
                            _builder_emit(T.ptx["add.ftz.f32"](out_frag[2], accum[2], bias_hi))
                            _builder_emit(T.ptx["add.ftz.f32"](out_frag[3], accum[3], bias_hi))
                            out_base = _builder_scalar("out_base", tn * a_M + tm, "int32")
                            out_next = _builder_scalar("out_next", out_base + a_M, "int32")
                            with T.If(tn < b_N):
                                with T.Then():
                                    with T.If(tm < a_M):
                                        with T.Then():
                                            T.buffer_store(
                                                c_output,
                                                T.cast(out_frag[0], "bfloat16"),
                                                [out_base],
                                            )
                            with T.If(tn + 1 < b_N):
                                with T.Then():
                                    with T.If(tm < a_M):
                                        with T.Then():
                                            T.buffer_store(
                                                c_output,
                                                T.cast(out_frag[1], "bfloat16"),
                                                [out_next],
                                            )
                            with T.If(tn < b_N):
                                with T.Then():
                                    with T.If(tm + 8 < a_M):
                                        with T.Then():
                                            T.buffer_store(
                                                c_output,
                                                T.cast(out_frag[2], "bfloat16"),
                                                [out_base + 8],
                                            )
                            with T.If(tn + 1 < b_N):
                                with T.Then():
                                    with T.If(tm + 8 < a_M):
                                        with T.Then():
                                            T.buffer_store(
                                                c_output,
                                                T.cast(out_frag[3], "bfloat16"),
                                                [out_next + 8],
                                            )
                with T.Else():
                    with T.If(T.And(T.int32(4) <= warp, warp <= 7)):
                        with T.Then():
                            k_loops_w = _builder_scalar(
                                "k_loops_w", T.truncdiv(c_K + 1023, 1024), "int32"
                            )
                            mib_w = _builder_scalar("mib_w", block_m * 16, "int32")
                            wslot = _builder_scalar(
                                "wslot", T.cast(warp, "uint32") % T.uint32(4), "uint32"
                            )
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    with T.serial(0, k_loops_w, unroll=False, dtype="uint32") as ki:
                                        IRBuilder.name("ki", ki)
                                        if STAGES == 4:
                                            stage_w = _builder_scalar("stage_w", wslot, "uint32")
                                            phase_w = _builder_scalar(
                                                "phase_w", ki & T.uint32(1), "uint32"
                                            )
                                        else:
                                            stage_w = _builder_scalar(
                                                "stage_w",
                                                wslot + T.uint32(4) * (ki % T.uint32(2)),
                                                "uint32",
                                            )
                                            phase_w = _builder_scalar(
                                                "phase_w", ki // T.uint32(2) & T.uint32(1), "uint32"
                                            )
                                        k_base_w = _builder_scalar(
                                            "k_base_w",
                                            T.cast(
                                                (ki * T.uint32(4) + wslot) * T.uint32(256), "int32"
                                            ),
                                            "int32",
                                        )
                                        _builder_emit(
                                            _mbarrier_wait(
                                                smem_raw,
                                                T.uint32(consumed_off) + stage_w * T.uint32(8),
                                                phase_w ^ T.uint32(1),
                                            )
                                        )
                                        _builder_emit(
                                            _mbarrier_expect_tx(
                                                smem_raw, stage_w * T.uint32(8), 8192
                                            )
                                        )
                                        with T.unroll(4) as box:
                                            IRBuilder.name("box", box)
                                            _builder_emit(
                                                _tma_2d_g2s(
                                                    smem_raw,
                                                    T.uint32(WT_OFF)
                                                    + (stage_w * T.uint32(4) + T.uint32(box))
                                                    * T.uint32(2048),
                                                    a_tmap_wt,
                                                    k_base_w + box * 64,
                                                    mib_w,
                                                    stage_w * T.uint32(8),
                                                )
                                            )
                                    if STAGES == 8:
                                        dki_w = _builder_scalar(
                                            "dki_w", T.cast(k_loops_w, "uint32"), "uint32"
                                        )
                                        dstage_w = _builder_scalar(
                                            "dstage_w",
                                            wslot + T.uint32(4) * (dki_w % T.uint32(2)),
                                            "uint32",
                                        )
                                        dphase_w = _builder_scalar(
                                            "dphase_w", dki_w // T.uint32(2) & T.uint32(1), "uint32"
                                        )
                                        _builder_emit(
                                            _mbarrier_wait(
                                                smem_raw,
                                                T.uint32(consumed_off) + dstage_w * T.uint32(8),
                                                dphase_w ^ T.uint32(1),
                                            )
                                        )
                            _builder_emit(T.ptx.bar.sync(T.uint32(2), T.uint32(THREADS)))
                        with T.Else():
                            with T.If(T.And(T.int32(8) <= warp, warp <= 11)):
                                with T.Then():
                                    k_loops_a = _builder_scalar(
                                        "k_loops_a", T.truncdiv(c_K + 1023, 1024), "int32"
                                    )
                                    ni_a = _builder_scalar("ni_a", block_n * 8, "int32")
                                    aslot = _builder_scalar(
                                        "aslot", T.cast(warp, "uint32") % T.uint32(4), "uint32"
                                    )
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            if USE_PDL:
                                                _builder_emit(
                                                    T.evaluate(T.ptx.griddepcontrol.wait())
                                                )
                                                _builder_emit(
                                                    T.ptx.griddepcontrol.launch_dependents()
                                                )
                                            with T.serial(
                                                0, k_loops_a, unroll=False, dtype="uint32"
                                            ) as ki:
                                                IRBuilder.name("ki", ki)
                                                if STAGES == 4:
                                                    stage_a = _builder_scalar(
                                                        "stage_a", aslot, "uint32"
                                                    )
                                                    phase_a = _builder_scalar(
                                                        "phase_a", ki & T.uint32(1), "uint32"
                                                    )
                                                else:
                                                    stage_a = _builder_scalar(
                                                        "stage_a",
                                                        aslot + T.uint32(4) * (ki % T.uint32(2)),
                                                        "uint32",
                                                    )
                                                    phase_a = _builder_scalar(
                                                        "phase_a",
                                                        ki // T.uint32(2) & T.uint32(1),
                                                        "uint32",
                                                    )
                                                k_base_a = _builder_scalar(
                                                    "k_base_a",
                                                    T.cast(
                                                        (ki * T.uint32(4) + aslot) * T.uint32(256),
                                                        "int32",
                                                    ),
                                                    "int32",
                                                )
                                                _builder_emit(
                                                    _mbarrier_wait(
                                                        smem_raw,
                                                        T.uint32(consumed_off)
                                                        + stage_a * T.uint32(8),
                                                        phase_a ^ T.uint32(1),
                                                    )
                                                )
                                                _builder_emit(
                                                    _mbarrier_expect_tx(
                                                        smem_raw,
                                                        T.uint32(act_ready_off)
                                                        + stage_a * T.uint32(8),
                                                        4096,
                                                    )
                                                )
                                                with T.unroll(4) as box:
                                                    IRBuilder.name("box", box)
                                                    _builder_emit(
                                                        _tma_2d_g2s(
                                                            smem_raw,
                                                            T.uint32(act_off)
                                                            + (
                                                                stage_a * T.uint32(4)
                                                                + T.uint32(box)
                                                            )
                                                            * T.uint32(1024),
                                                            b_tmap_act,
                                                            k_base_a + box * 64,
                                                            ni_a,
                                                            T.uint32(act_ready_off)
                                                            + stage_a * T.uint32(8),
                                                        )
                                                    )
                                            if STAGES == 8:
                                                dki_a = _builder_scalar(
                                                    "dki_a", T.cast(k_loops_a, "uint32"), "uint32"
                                                )
                                                dstage_a = _builder_scalar(
                                                    "dstage_a",
                                                    aslot + T.uint32(4) * (dki_a % T.uint32(2)),
                                                    "uint32",
                                                )
                                                dphase_a = _builder_scalar(
                                                    "dphase_a",
                                                    dki_a // T.uint32(2) & T.uint32(1),
                                                    "uint32",
                                                )
                                                _builder_emit(
                                                    _mbarrier_wait(
                                                        smem_raw,
                                                        T.uint32(consumed_off)
                                                        + dstage_a * T.uint32(8),
                                                        dphase_a ^ T.uint32(1),
                                                    )
                                                )
                                    _builder_emit(T.ptx.bar.sync(T.uint32(2), T.uint32(THREADS)))
    return builder.get()


def get_kernel(
    B: int,
    O: int,
    K: int,
    *,
    stage: int | None = None,
    use_pdl: bool = False,
    num_sms: int | None = None,
):
    _validate_problem(B, O, K)
    if stage is None:
        if num_sms is None:
            if not torch.cuda.is_available():
                raise ValueError("num_sms is required for automatic dispatch without CUDA")
            num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        stage = _select_stage(B, O, K, num_sms)
    if stage not in (4, 8):
        raise ValueError(f"stage must be 4 or 8, got {stage}")

    launch_params = ["blockIdx.x", "blockIdx.y", "threadIdx.x"]
    if use_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return _tinygemm2_sm100(
        STAGES=stage, USE_PDL=use_pdl, GRID_X=(O + 15) // 16, GRID_Y=(B + 7) // 8
    ).with_attr("tirx.kernel_launch_params", launch_params)


class _AlignedTensorMap:
    """Host storage for one 128-byte-aligned TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 128)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 127) & ~127)


def _encode_tensor_map(
    tensor: torch.Tensor,
    *,
    global_dims: tuple[int, int],
    global_stride_bytes: int,
    box_dims: tuple[int, int],
) -> _AlignedTensorMap:
    import tvm

    descriptor = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        "bfloat16",
        2,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        global_stride_bytes,
        *box_dims,
        1,
        1,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        3,  # CU_TENSOR_MAP_SWIZZLE_128B
        0,  # CU_TENSOR_MAP_L2_PROMOTION_NONE
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def _build_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    B, O, K = case["B"], case["O"], case["K"]
    return {
        "weight": _encode_tensor_map(
            case["weight"], global_dims=(K, O), global_stride_bytes=K * 2, box_dims=(64, 16)
        ),
        "input": _encode_tensor_map(
            case["input"], global_dims=(K, B), global_stride_bytes=K * 2, box_dims=(64, 8)
        ),
    }


def prepare_data(
    B: int, O: int, K: int, *, seed: int = 0, device: str | torch.device = "cuda"
) -> dict[str, Any]:
    _validate_problem(B, O, K)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("TinyGEMM2 SM100 data preparation requires CUDA")
    generator = torch.Generator(device=device).manual_seed(seed)
    input_tensor = (
        torch.randn((B, K), generator=generator, device=device, dtype=torch.float32) / 8
    ).bfloat16()
    weight = (
        torch.randn((O, K), generator=generator, device=device, dtype=torch.float32) / 8
    ).bfloat16()
    bias = torch.randn((O,), generator=generator, device=device, dtype=torch.float32).bfloat16()
    out = torch.zeros((B, O), dtype=torch.bfloat16, device=device)
    case = {
        "B": B,
        "O": O,
        "K": K,
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "out": out,
    }
    case["tensor_maps"] = _build_tensor_maps(case)
    return case


def _tirx_args(case: dict[str, Any], output: torch.Tensor | None = None) -> tuple[Any, ...]:
    maps = case["tensor_maps"]
    return (
        maps["weight"].ptr,
        maps["input"].ptr,
        (case["out"] if output is None else output).view(-1),
        case["bias"],
        case["O"],
        case["B"],
        case["K"],
    )


@lru_cache(maxsize=1)
def _flashinfer_tinygemm2_spec():
    import flashinfer
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_jit_spec, sm100a_nvcc_flags

    filename = "tinygemm2_sm100.cu"
    candidates = (
        Path(flashinfer.__file__).resolve().parents[1] / "csrc" / filename,
        jit_env.FLASHINFER_CSRC_DIR / filename,
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise RuntimeError(
            "FlashInfer TinyGEMM2 frozen source is unavailable; checked "
            + ", ".join(map(str, candidates))
        )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(
            "FlashInfer TinyGEMM2 source does not match the frozen oracle: "
            f"{source} sha256={source_hash}"
        )
    return gen_jit_spec(
        "tinygemm2_sm100",
        [source],
        extra_cuda_cflags=[*sm100a_nvcc_flags, "-gencode=arch=compute_103a,code=sm_103a"],
        extra_include_paths=[source.parent, source.parent.parent / "include"],
    )


@lru_cache(maxsize=1)
def _load_flashinfer_module():
    return _flashinfer_tinygemm2_spec().build_and_load()


def _flashinfer_variant(stage: int, use_pdl: bool):
    suffix = "_pdl" if use_pdl else ""
    return getattr(_load_flashinfer_module(), f"stage{stage}{suffix}_op")


@cache
def _compile_executable(B: int, O: int, stage: int, use_pdl: bool):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(get_kernel(B, O, 1024, stage=stage, use_pdl=use_pdl))


def _run_tirx(case: dict[str, Any], stage: int, use_pdl: bool, output: torch.Tensor) -> None:
    executable = _compile_executable(case["B"], case["O"], stage, use_pdl)
    executable(*_tirx_args(case, output))


def _run_flashinfer(case: dict[str, Any], stage: int, use_pdl: bool, output: torch.Tensor) -> None:
    _flashinfer_variant(stage, use_pdl)(case["input"], case["weight"], case["bias"], output)


def run_test(B: int, O: int, K: int) -> None:
    _require_sm100()
    case = prepare_data(B, O, K)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    stage = _select_stage(B, O, K, num_sms)
    linear_ref = F.linear(
        case["input"].float(), case["weight"].float(), case["bias"].float()
    ).bfloat16()

    for use_pdl in (False, True):
        tirx_out = torch.zeros_like(case["out"])
        flashinfer_out = torch.zeros_like(case["out"])
        _run_tirx(case, stage, use_pdl, tirx_out)
        _run_flashinfer(case, stage, use_pdl, flashinfer_out)
        torch.cuda.synchronize()
        if not torch.equal(tirx_out, flashinfer_out):
            differing = int((tirx_out != flashinfer_out).sum().item())
            max_diff = float((tirx_out.float() - flashinfer_out.float()).abs().max().item())
            raise AssertionError(
                f"TinyGEMM2 bitwise mismatch for B={B}, O={O}, K={K}, "
                f"stage={stage}, use_pdl={use_pdl}: {differing} elements, "
                f"max_abs_diff={max_diff}"
            )
        torch.testing.assert_close(tirx_out.float(), linear_ref.float(), atol=1e-2, rtol=1e-2)


def prepare_bench(B: int, O: int, K: int):
    """Compile the hardware-profile dispatch before CUDA initialization."""
    from tirx_kernels.runner import hardware_num_sms, prepared_gpu_benchmark

    stage = _select_stage(B, O, K, hardware_num_sms())
    state = {
        "B": B,
        "O": O,
        "K": K,
        "stage": stage,
        "executable": _compile_executable(B, O, stage, False),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 5,
    cooldown_s: float = 1.0,
) -> dict[str, Any]:
    _require_sm100()
    from tirx_kernels.runner import bench

    B, O, K = prepared["B"], prepared["O"], prepared["K"]
    stage = prepared["stage"]
    executable = prepared["executable"]
    case = prepare_data(B, O, K)
    args = _tirx_args(case)

    reference_out = torch.zeros_like(case["out"])
    _run_flashinfer(case, stage, False, reference_out)
    executable(*args)
    torch.cuda.synchronize()
    if not torch.equal(case["out"], reference_out):
        raise AssertionError("TinyGEMM2 benchmark preflight failed bitwise validation")

    def _flashinfer_builder():
        output = torch.empty_like(case["out"])
        op = _flashinfer_variant(stage, False)
        op(case["input"], case["weight"], case["bias"], output)
        torch.cuda.synchronize()

        def launch():
            op(case["input"], case["weight"], case["bias"], output)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_sm100": _flashinfer_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    B: int,
    O: int,
    K: int,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 5,
    cooldown_s: float = 1.0,
) -> dict[str, Any]:
    return prepare_bench(B, O, K).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
