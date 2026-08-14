# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""The MegaMoE kernel body: the dispatch, GEMM, and combine megakernel.

Configuration and layout derivation live in :mod:`.spec`; nothing here reaches
back into :mod:`.data`.

GUARD: the ``@T.prim_func def mega_moe`` name and the ``"main"`` global var are
part of the emitted CUDA symbol -- never rename them.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value

from .spec import (
    MegaMoeConfig,
    _align_up,
    _ceil_div,
    get_deepgemm_launch_config,
    get_deepgemm_symm_buffer_layout,
    get_deepgemm_workspace_layout,
    get_tirx_launch_param_tags,
)

__all__ = ["get_kernel"]

# ---------------------------------------------------------------------------
# PTX wrappers
#
# Pure functions of their arguments: each names one instruction (or one small
# instruction pair) so the kernel body below reads as the upstream CUDA does.
# Nothing here depends on the launch configuration, so none of it belongs in
# the builder's closure.
# ---------------------------------------------------------------------------


def load_global_s64(dst, address):
    return T.ptx.ld.global_.s64(dst, address)


def load_global_u64(dst, address):
    return T.ptx.ld.global_.u64(dst, address)


def load_global_u32(dst, address):
    return T.ptx.ld.global_.u32(dst, address)


def store_global_u64(address, value):
    return T.ptx.st.global_.u64(address, value)


def store_global_u32(address, value):
    return T.ptx.st.global_.u32(address, value)


def store_global_u8(address, value):
    return T.ptx.st.global_.u8(address, value)


# Destination-first, mirroring the PTX these wrap: the caller declares the
# register and passes it in.
def load_acq_sys_s32(dst, address):
    return T.ptx.ld.acquire.sys.global_.s32(dst, address)


def atomic_add_rel_u32(dst, address, value):
    return T.ptx.atom.release.gpu.global_.add.u32(dst, address, value)


def load_acq_u32(dst, address):
    return T.ptx.ld.acquire.gpu.global_.b32(dst, address)


def grid_sync_done_u32(new_value, old_value):
    return T.cast(
        T.bitwise_and(T.bitwise_xor(new_value, old_value), T.uint32(0x80000000)) != T.uint32(0),
        "uint32",
    )


def load_f32(dst, address):
    # ptx destinations are declared registers, so the helper writes into
    # one the caller owns rather than returning a value.
    return T.ptx.ld.global_.f32(dst, address)


def load_shared_u32(dst, address):
    return T.ptx.ld.shared.u32(dst, address)


def uint32_bits_to_float(bits):
    return T.cuda.uint_as_float(bits)


def float_bits(x):
    return T.cuda.float_as_uint(x)


def sync_unaligned(barrier_idx, num_threads):
    return T.ptx.barrier.sync(T.uint32(barrier_idx), T.uint32(num_threads))


def prefetch_tensormap(tensor_map):
    return T.ptx.prefetch.tensormap(T.address_of(tensor_map))


def lds128(src_ptr, dst, base=0):
    return T.ptx.ld.shared.v4.u32(dst[base], dst[base + 1], dst[base + 2], dst[base + 3], src_ptr)


def mbarrier_arrive_and_set_tx(barrier_ptr, num_bytes):
    return T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier_ptr, T.uint32(num_bytes))


def mbarrier_wait_phase(barrier_ptr, phase):
    return T.cuda.mbarrier_wait(barrier_ptr, phase)


def replace_smem_desc_addr(desc, smem_ptr):
    start_addr = T.cast(
        T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)
        ),
        "uint64",
    )
    return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)


#: Bulk global -> shared copy, completion signalled on an mbarrier.
_bulk_g2s_chain = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"

#: Bulk shared -> global copy, retired through the bulk commit group.
_bulk_s2g_chain = "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"

#: `cute::TMA::CacheHintSm90::EVICT_FIRST`. Ring-buffer traffic is streamed
#: through L2 once, so it should be the first thing evicted.
_evict_first_policy = T.uint64(0x12F0000000000000)


def tma_load_1d(dst_ptr, src_ptr, barrier_ptr, num_bytes):
    return T.ptx[_bulk_g2s_chain](
        dst_ptr, src_ptr, T.cast(num_bytes, "uint32"), barrier_ptr, _evict_first_policy
    )


def tma_store_1d(dst_ptr, src_ptr, num_bytes):
    return T.ptx[_bulk_s2g_chain](
        dst_ptr, src_ptr, T.cast(num_bytes, "uint32"), _evict_normal_policy
    )


def tma_store_fence():
    return T.ptx.fence.proxy.async_.shared__cta()


def fence_barrier_init():
    return T.ptx.fence.mbarrier_init.release.cluster()


def tma_store_arrive():
    return T.ptx.cp.async_.bulk.commit_group()


def tma_store_wait(num_prior_groups):
    if num_prior_groups == 0:
        return T.ptx.cp.async_.bulk.wait_group(0)
    if num_prior_groups == 1:
        return T.ptx.cp.async_.bulk.wait_group(1)
    raise ValueError("Unsupported TMA store wait distance")


def tma_store_2d(src, tensormap, coord0, coord1):
    return T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
        T.address_of(tensormap), coord0, coord1, src
    )


#: Unicast (cta_mask=1) at cta_group::2 scope with the evict-normal L2 policy;
#: the mbarrier arrives as a precomputed leader-CTA shared address.
_sm100_2sm_load_chain = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)

#: `cute::TMA::CacheHintSm100::EVICT_NORMAL`. Weights and activations are
#: re-read across blocks, so they keep the default L2 residency.
_evict_normal_policy = T.uint64(1152921504606846976)


def sm100_tma_2sm_load_2d_addr(dst, mbar, tensormap_addr, coord0, coord1):
    mbar_addr = T.cuda.sm100_2sm_leader_smem_addr(mbar)
    T.evaluate(
        T.ptx[_sm100_2sm_load_chain](
            dst, tensormap_addr, coord0, coord1, mbar_addr, _evict_normal_policy
        )
    )


def stg128_symm(peer_base, byte_offset, r0, r1, r2, r3):
    return T.ptx.st.global_.v4.b32(peer_ptr(peer_base, byte_offset), r0, r1, r2, r3)


def ptr_to_u64(ptr):
    return T.reinterpret("uint64", ptr)


def peer_ptr(peer_base, byte_offset):
    return T.reinterpret("handle", peer_base + byte_offset)


def peer_store_u32(peer_base, byte_offset, value):
    return T.ptx.st.global_.u32(peer_ptr(peer_base, byte_offset), value)


def peer_store_u64(peer_base, byte_offset, value):
    return T.ptx.st.global_.u64(peer_ptr(peer_base, byte_offset), value)


def st_shared_bulk(ptr, num_bytes):
    return T.ptx.st_bulk.weak.shared__cta(ptr, T.cast(num_bytes, "uint64"))


def peer_atomic_add_u64(dst, peer_base, byte_offset, value):
    return T.ptx.atom.sys.global_.add.u64(dst, peer_ptr(peer_base, byte_offset), value)


def peer_red_add_rel_sys_s32(peer_base, byte_offset, value):
    return T.ptx.red.release.sys.global_.add.s32(peer_ptr(peer_base, byte_offset), value)


def peer_load_u32(dst, peer_base, byte_offset):
    return T.ptx.ld.global_.u32(dst, peer_ptr(peer_base, byte_offset))


def peer_load_f32(dst, peer_base, byte_offset):
    return T.ptx.ld.global_.f32(dst, peer_ptr(peer_base, byte_offset))


def tma_load_1d_symm(dst_ptr, peer_base, byte_offset, barrier_ptr, num_bytes):
    return T.ptx[_bulk_g2s_chain](
        dst_ptr,
        peer_ptr(peer_base, byte_offset),
        T.cast(num_bytes, "uint32"),
        barrier_ptr,
        _evict_first_policy,
    )


def ballot_sync(mask, pred):
    return T.cuda.ballot_sync(mask, pred)


def ffs_u32(value):
    return T.cuda.ffs_u32(value)


def reduce_add_sync_u32(mask, value):
    return T.cuda.reduce_add_sync_u32(mask, value)


def red_add_gpu_u32(address, value):
    return T.ptx.red.gpu.global_.add.u32(address, value)


def cuda_clock64():
    return T.cuda.clock64()


def bf16x2_lo(packed):
    return T.cast(T.bitwise_and(packed, T.uint32(0xFFFF)), "uint16")


def bf16x2_hi(packed):
    return T.cast(T.bitwise_and(T.shift_right(packed, T.uint32(16)), T.uint32(0xFFFF)), "uint16")


def cast_into_bf16_and_pack(v0, v1):
    return T.cuda.float22bfloat162_rn(v0, v1)


def make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
    runtime_desc = T.bitwise_and(desc, T.uint32(0x9FFFFFCF))
    runtime_desc = T.bitwise_or(runtime_desc, T.shift_left(T.cast(sfa_id, "uint32"), T.uint32(29)))
    runtime_desc = T.bitwise_or(runtime_desc, T.shift_left(T.cast(sfb_id, "uint32"), T.uint32(4)))
    return runtime_desc


def st_async_cluster_task_info(dst_ptr, bar_ptr, dst_cta_idx, task_info_regs):
    mapped_bar = T.alloc_local((1,), "uint32")
    mapped_dst = T.alloc_local((1,), "uint32")
    mapped_dst_hi = T.alloc_local((1,), "uint32")
    cta = T.cast(dst_cta_idx, "uint32")
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(mapped_bar[0], T.cuda.cvta_generic_to_shared(bar_ptr), cta)
    )
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(mapped_dst[0], T.cuda.cvta_generic_to_shared(dst_ptr), cta)
    )
    T.evaluate(
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            mapped_dst[0],
            task_info_regs[0],
            task_info_regs[1],
            task_info_regs[2],
            task_info_regs[3],
            mapped_bar[0],
        )
    )
    T.evaluate(T.ptx.add.u32(mapped_dst_hi[0], mapped_dst[0], T.uint32(16)))
    return T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
        mapped_dst_hi[0],
        task_info_regs[4],
        task_info_regs[5],
        task_info_regs[6],
        task_info_regs[7],
        mapped_bar[0],
    )


def atomic_add_u32(dst, address, value):
    return T.ptx.atom.global_.add.u32(dst, address, value)


def load_volatile_u32(dst, address):
    return T.ptx.ld.volatile.global_.u32(dst, address)

_BUILDER_MISSING = object()
_BUILDER_DECL_SCOPES = {}
_BUILDER_SCOPE_TOKEN = ContextVar("mega_moe_builder_scope_token", default=None)


@contextmanager
def _builder_context(builder):
    scope_token = object()
    reset_token = _BUILDER_SCOPE_TOKEN.set(scope_token)
    try:
        with builder:
            yield
    finally:
        stale = [key for key in _BUILDER_DECL_SCOPES if key[0] is scope_token]
        for key in stale:
            del _BUILDER_DECL_SCOPES[key]
        _BUILDER_SCOPE_TOKEN.reset(reset_token)


def _builder_scope_key():
    scope_token = _BUILDER_SCOPE_TOKEN.get()
    if scope_token is None:
        raise RuntimeError("Builder scope tracking requires _builder_context()")
    return scope_token, tuple(IRBuilder.current().frames)


def _builder_record_scope(value):
    builder_id, frames = _builder_scope_key()
    _BUILDER_DECL_SCOPES[builder_id, id(value)] = (value, frames)
    return value


def _builder_visible(previous):
    builder_id, frames = _builder_scope_key()
    record = _BUILDER_DECL_SCOPES.get((builder_id, id(previous)))
    if record is None:
        return False
    declared_value, declared = record
    if declared_value is not previous:
        return False
    control_frames = {"IfFrame", "ThenFrame", "ElseFrame", "ForFrame", "WhileFrame"}
    if not any(
        type(frame).__name__ in control_frames
        for frame in IRBuilder.current().frames[: len(declared)]
    ):
        return True
    return len(frames) >= len(declared) and all(
        current.same_as(expected) for current, expected in zip(frames[: len(declared)], declared)
    )


def _builder_runtime_condition(value):
    return value


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
    return _builder_record_scope(scalar.scalar)


def _builder_scalar(name, value, dtype):
    scalar = _builder_alloc_scalar(name, dtype)
    T.buffer_store(scalar.buffer, value, scalar.indices)
    return scalar


def _builder_buffer(name, shape, dtype):
    buffer = T.alloc_local(shape, dtype)
    IRBuilder.name(name, buffer)
    return _builder_record_scope(buffer)


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return _builder_record_scope(result)


def _builder_assign(name, value, previous=_BUILDER_MISSING):
    if isinstance(value, I.meta_var):
        return value.value
    if previous is not _BUILDER_MISSING and _builder_visible(previous):
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
        return _builder_record_scope(value)
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _builder_assign(f"{name}_{index}", item)
        return value
    if is_buffer_var(value) or isinstance(value, IterVar | Layout):
        IRBuilder.name(name, value)
        return _builder_record_scope(value)
    if isinstance(value, tvm.ir.Var):
        if isinstance(value.ty, tvm.ir.PointerType):
            return _builder_bind(name, value, value.ty)
        IRBuilder.name(name, value)
        return _builder_record_scope(value)
    if isinstance(value, tvm.ir.Expr) and isinstance(
        getattr(value, "ty", None), tvm.ir.PointerType
    ):
        return _builder_bind(name, value, value.ty)
    if isinstance(value, tvm.ir.Expr) and tvm.ir.is_prim_expr(value):
        return _builder_scalar(name, value, str(value.ty.dtype))
    if isinstance(value, tvm.tirx.expr.ExprOp):
        return _builder_scalar(name, value, "bool")
    return value


def _builder_assign_many(names, values, previous):
    return tuple(
        _builder_assign(name, value, old) for name, value, old in zip(names, values, previous)
    )

def get_kernel(
    *,
    num_processes: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    activation_clamp: float = 10.0,
    fast_math: int = 1,
    collect_stats: bool = False,
    emit_nvl_barrier_timeout_printf: bool = True,
):
    from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    runtime_config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    kernel_config = get_deepgemm_launch_config(runtime_config)
    workspace_layout = get_deepgemm_workspace_layout(runtime_config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(runtime_config)
    num_experts_per_rank = num_experts // num_processes
    num_experts_per_lane = (num_experts_per_rank + 31) // 32
    num_ranks_per_lane = (num_processes + 31) // 32
    num_warps_per_warpgroup = 4
    num_l1_block_ns = (intermediate_hidden * 2) // kernel_config.block_n
    num_l2_block_ns = hidden // kernel_config.block_n
    # Dynamic task scheduler constants (scheduler/mega_moe.cuh, upstream 559d79f)
    num_l1_clusters = num_l1_block_ns // 2
    num_l2_clusters = num_l2_block_ns // 2
    num_sched_clusters = kernel_config.num_sms // 2
    num_schedule_stages = 2
    num_schedule_consumer_threads = 2 * kernel_config.num_epilogue_threads
    task_info_bytes = 32
    sched_l1_waves_done = 0xFFFFFFFF
    # `get_num_l1_warmup_waves`: only the interleave term depends on the runtime
    # total-M-block count; fold the rest to compile-time constants.
    sched_num_first_l2_wave_m_blocks = _ceil_div(num_sched_clusters, num_l2_clusters)
    sched_l1_warmup_first_l2_wave = _ceil_div(
        sched_num_first_l2_wave_m_blocks * num_l1_clusters, num_sched_clusters
    )
    sched_interleave_cluster_diff = (
        num_l1_clusters - num_l2_clusters if num_l1_clusters > num_l2_clusters else 0
    )
    num_ring_tokens = workspace_layout.num_ring_tokens
    if num_ring_tokens % kernel_config.block_m != 0:
        raise ValueError("MegaMoE ring capacity must be divisible by BLOCK_M")
    num_ring_blocks = num_ring_tokens // kernel_config.block_m
    l1_out_block_n = kernel_config.l1_out_block_n
    sf_block_m = kernel_config.sf_block_m
    umma_m = 256
    umma_n = kernel_config.block_m
    umma_block_k = 128
    umma_k = 32
    num_sfa_utccp_chunks = sf_block_m // 128
    num_sfb_utccp_chunks = kernel_config.block_n // 128
    num_epilogue_stages = 2
    num_tma_store_stages = 2
    use_more_epilogue_registers = num_experts_per_rank <= 64
    num_dispatch_registers = 48 if use_more_epilogue_registers else 96
    num_non_epilogue_registers = 40 if use_more_epilogue_registers else 88
    num_epilogue_registers = 208 if use_more_epilogue_registers else 160
    num_accum_tmem_cols = kernel_config.block_m * num_epilogue_stages
    num_sfa_tmem_cols = sf_block_m // 32
    num_sfb_tmem_cols = kernel_config.block_n // 32
    num_tmem_cols = 32
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 32:
        num_tmem_cols = 64
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 64:
        num_tmem_cols = 128
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 128:
        num_tmem_cols = 256
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 256:
        num_tmem_cols = 512

    if not 32 <= num_tmem_cols <= 512:
        raise ValueError("Invalid tensor memory columns")

    tcgen05_cta_mask = (1 << 2) - 1


    def scale_pack_fp8x4_e4m3(out, upper, lower, v0, v1, v2, v3, sf_inv_x, sf_inv_y):
        sf_inv_pair = _builder_scalar(
            "sf_inv_pair", T.cuda.make_float2(sf_inv_x, sf_inv_y), "uint64"
        )
        T.buffer_store(upper, T.cuda.fmul2_rn(T.cuda.make_float2(v0, v1), sf_inv_pair), [0])
        T.buffer_store(lower, T.cuda.fmul2_rn(T.cuda.make_float2(v2, v3), sf_inv_pair), [0])
        T.buffer_store(
            out,
            T.cuda.fp8x4_e4m3_from_float4(
                T.cuda.float2_x(upper[0]),
                T.cuda.float2_y(upper[0]),
                T.cuda.float2_x(lower[0]),
                T.cuda.float2_y(lower[0]),
            ),
            [0],
        )


    def red_add_gpu_s32(address, value):
        return T.ptx.red.gpu.global_.add.s32(address, value)

    def load_volatile_u64(dst, address):
        return T.ptx.ld.volatile.global_.u64(dst, address)







    def store_global_f32(address, value):
        return T.ptx.st.global_.f32(address, value)

    # Destination-first, mirroring the PTX these wrap: the caller declares the
    # register and passes it in.





    def load_shared_u64(dst, address):
        return T.ptx.ld.shared.u64(dst, address)




    def get_e4m3_sf_and_sf_inv(
        sf, sf_inv, scaled_pair, scaled_values, scaled_bits, scale_exponents, amax_x, amax_y
    ):
        T.buffer_store(
            scaled_pair,
            T.cuda.fmul2_rn(
                T.cuda.make_float2(amax_x, amax_y),
                T.cuda.make_float2(T.float32(1.0 / 448.0), T.float32(1.0 / 448.0)),
            ),
            [0],
        )
        T.buffer_store(scaled_values, T.cuda.float2_x(scaled_pair[0]), [0])
        T.buffer_store(scaled_values, T.cuda.float2_y(scaled_pair[0]), [1])
        with T.unroll(0, 2) as dim:
            IRBuilder.name("dim", dim)
            T.buffer_store(scaled_bits, float_bits(scaled_values[dim]), [dim])
            T.buffer_store(
                scale_exponents,
                T.cast(T.shift_right(scaled_bits[dim], T.uint32(23)), "int32")
                - T.int32(127)
                + T.Select(
                    T.bitwise_and(scaled_bits[dim], T.uint32((1 << 23) - 1)) != T.uint32(0),
                    T.int32(1),
                    T.int32(0),
                ),
                [dim],
            )
            T.buffer_store(
                sf,
                uint32_bits_to_float(
                    T.shift_left(
                        T.cast(scale_exponents[dim] + T.int32(127), "uint32"), T.uint32(23)
                    )
                ),
                [dim],
            )
            T.buffer_store(
                sf_inv,
                uint32_bits_to_float(
                    T.shift_left(
                        T.cast(T.int32(127) - scale_exponents[dim], "uint32"), T.uint32(23)
                    )
                ),
                [dim],
            )

    kernel_activation_clamp = float(activation_clamp)
    kernel_fast_math = bool(fast_math)
    kernel_collect_stats = bool(collect_stats)
    kernel_emit_nvl_barrier_timeout_printf = bool(emit_nvl_barrier_timeout_printf)
    use_activation_clamp = math.isfinite(kernel_activation_clamp)

    def warp_reduce_max_4(values, atom_idx, dim):
        T.buffer_store(
            values,
            T.max(
                values[atom_idx, dim],
                T.tvm_warp_shuffle_xor(0xFFFFFFFF, values[atom_idx, dim], 4, 32, 32),
            ),
            [atom_idx, dim],
        )
        T.buffer_store(
            values,
            T.max(
                values[atom_idx, dim],
                T.tvm_warp_shuffle_xor(0xFFFFFFFF, values[atom_idx, dim], 8, 32, 32),
            ),
            [atom_idx, dim],
        )
        T.buffer_store(
            values,
            T.max(
                values[atom_idx, dim],
                T.tvm_warp_shuffle_xor(0xFFFFFFFF, values[atom_idx, dim], 16, 32, 32),
            ),
            [atom_idx, dim],
        )


    def transform_sf_token_idx(token_idx_in_expert):
        token_idx_u32 = T.cast(token_idx_in_expert, "uint32")
        idx = token_idx_u32 % T.uint32(kernel_config.block_m)
        return T.cast(
            token_idx_u32 // T.uint32(kernel_config.block_m) * T.uint32(sf_block_m)
            + T.bitwise_and(idx, T.uint32(0xFFFFFF80))
            + T.shift_left(T.bitwise_and(idx, T.uint32(31)), T.uint32(2))
            + T.bitwise_and(T.shift_right(idx, T.uint32(5)), T.uint32(3)),
            "int32",
        )




    def sts128(dst_ptr, r0, r1, r2, r3):
        return T.ptx.st.shared.v4.b32(dst_ptr, r0, r1, r2, r3)




    def shared_addr_u32(ptr):
        return T.cuda.cvta_generic_to_shared(ptr)


    _bulk_g2s_chain = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    _bulk_s2g_chain = "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"
    _evict_first_policy = T.uint64(0x12F0000000000000)








    # Unicast (cta_mask=1) at cta_group::2 scope with the evict-normal L2
    # policy; the mbarrier arrives as a precomputed leader-CTA shared address.
    _sm100_2sm_load_chain = (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
    )
    _evict_normal_policy = T.uint64(1152921504606846976)



    def sm100_tma_2sm_load_2d_select(
        dst, mbar, tensor_map_l1, tensor_map_l2, block_phase_value, coord0, coord1
    ):
        sm100_tma_2sm_load_2d_addr(
            dst,
            mbar,
            T.Select(
                block_phase_value == 1, T.address_of(tensor_map_l1), T.address_of(tensor_map_l2)
            ),
            coord0,
            coord1,
        )






    def st_shared_u32(ptr, value):
        return T.ptx.st.shared.u32(ptr, value)

    def st_shared_u64(ptr, value):
        return T.ptx.st.shared.u64(ptr, value)










    def reduce_min_sync_u32(mask, value):
        return T.cuda.reduce_min_sync_u32(mask, value)

    def fns_b32(dst, mask, base, offset):
        return T.ptx.fns.b32(dst, mask, base, offset)





    def stmatrix_fp8x4_trans(smem_ptr, word):
        return T.ptx.stmatrix.sync.aligned.m16n8.x1.trans.shared.b8(smem_ptr, word)


    def activation_pair_store(out, atom_idx, pair_idx, gate0, gate1, up0, up1, weight0, weight1):
        bf16_gate = _builder_scalar("bf16_gate", cast_into_bf16_and_pack(gate0, gate1), "uint32")
        bf16_up = _builder_scalar("bf16_up", cast_into_bf16_and_pack(up0, up1), "uint32")

        if use_activation_clamp:
            activation_clamp_value = _builder_scalar(
                "activation_clamp_value", T.float32(kernel_activation_clamp), "float32"
            )
            clamp_pos = _builder_scalar(
                "clamp_pos",
                cast_into_bf16_and_pack(activation_clamp_value, activation_clamp_value),
                "uint32",
            )
            clamp_neg = _builder_scalar(
                "clamp_neg",
                cast_into_bf16_and_pack(-activation_clamp_value, -activation_clamp_value),
                "uint32",
            )
            T.buffer_store(bf16_gate.buffer, T.cuda.hmin2(bf16_gate, clamp_pos), bf16_gate.indices)
            T.buffer_store(bf16_up.buffer, T.cuda.hmax2(bf16_up, clamp_neg), bf16_up.indices)
            T.buffer_store(bf16_up.buffer, T.cuda.hmin2(bf16_up, clamp_pos), bf16_up.indices)

        gate = _builder_scalar("gate", T.cuda.bfloat1622float2(bf16_gate), "uint64")
        gate_x = _builder_scalar("gate_x", T.cuda.float2_x(gate), "float32")
        gate_y = _builder_scalar("gate_y", T.cuda.float2_y(gate), "float32")
        neg_gate_exp = _builder_scalar(
            "neg_gate_exp", T.cuda.make_float2(T.exp(-gate_x), T.exp(-gate_y)), "uint64"
        )
        denom = _builder_scalar(
            "denom",
            T.cuda.fadd2_rn(T.cuda.make_float2(T.float32(1.0), T.float32(1.0)), neg_gate_exp),
            "uint64",
        )
        if kernel_fast_math:
            rcp_x = _builder_alloc_scalar("rcp_x", "float32")
            rcp_y = _builder_alloc_scalar("rcp_y", "float32")
            _builder_emit(T.ptx.rcp.approx.ftz.f32(rcp_x, T.cuda.float2_x(denom)))
            _builder_emit(T.ptx.rcp.approx.ftz.f32(rcp_y, T.cuda.float2_y(denom)))
            T.buffer_store(
                gate.buffer, T.cuda.fmul2_rn(gate, T.cuda.make_float2(rcp_x, rcp_y)), gate.indices
            )
        else:
            T.buffer_store(
                gate.buffer,
                T.cuda.make_float2(
                    gate_x / T.cuda.float2_x(denom), gate_y / T.cuda.float2_y(denom)
                ),
                gate.indices,
            )

        up = _builder_scalar("up", T.cuda.bfloat1622float2(bf16_up), "uint64")
        weights = _builder_scalar("weights", T.cuda.make_float2(weight0, weight1), "uint64")
        result = _builder_scalar(
            "result", T.cuda.fmul2_rn(T.cuda.fmul2_rn(gate, up), weights), "uint64"
        )
        T.buffer_store(out, T.cuda.float2_x(result), [atom_idx, pair_idx, 0])
        T.buffer_store(out, T.cuda.float2_y(result), [atom_idx, pair_idx, 1])


    def advance_umma_desc_lo(desc, base_lo, mn_offset, k_offset):
        return T.bitwise_or(
            T.bitwise_and(desc, T.shift_left(T.uint64(0xFFFFFFFF), T.uint64(32))),
            T.cast(base_lo + T.cast((mn_offset + k_offset) // f128_bytes, "uint32"), "uint64"),
        )

    def scheduler_get_num_tokens(
        expert_idx, lane_idx, stored_num_tokens_per_expert, selected_num_tokens
    ):
        T.buffer_store(selected_num_tokens, T.int32(0), [0])
        with T.serial(0, num_experts_per_lane) as expert_lane_idx:
            IRBuilder.name("expert_lane_idx", expert_lane_idx)
            T.buffer_store(
                selected_num_tokens,
                T.Select(
                    expert_idx == expert_lane_idx * 32 + lane_idx,
                    T.cast(stored_num_tokens_per_expert[expert_lane_idx], "int32"),
                    selected_num_tokens[0],
                ),
                [0],
            )
        expert_lane_idx_u32 = _builder_scalar(
            "expert_lane_idx_u32", T.cast(expert_idx, "uint32") % T.uint32(32), "uint32"
        )
        T.buffer_store(
            selected_num_tokens,
            T.tvm_warp_shuffle(
                T.uint32(0xFFFFFFFF),
                selected_num_tokens[0],
                T.cast(expert_lane_idx_u32, "int32"),
                32,
                32,
            ),
            [0],
        )

    def scheduler_get_pool_block_offset(
        expert_idx, lane_idx, stored_num_tokens_per_expert, pool_block_offset_sum
    ):
        T.buffer_store(pool_block_offset_sum, T.int32(0), [0])
        with T.serial(0, num_experts_per_lane) as expert_lane_idx:
            IRBuilder.name("expert_lane_idx", expert_lane_idx)
            expert_num_blocks_u32 = _builder_scalar(
                "expert_num_blocks_u32",
                (
                    stored_num_tokens_per_expert[expert_lane_idx]
                    + T.uint32(kernel_config.block_m - 1)
                )
                // T.uint32(kernel_config.block_m),
                "uint32",
            )
            T.buffer_store(
                pool_block_offset_sum,
                pool_block_offset_sum[0]
                + T.Select(
                    expert_lane_idx * 32 + lane_idx < expert_idx,
                    T.cast(expert_num_blocks_u32, "int32"),
                    T.int32(0),
                ),
                [0],
            )
        T.buffer_store(
            pool_block_offset_sum,
            T.cast(
                reduce_add_sync_u32(
                    T.uint32(0xFFFFFFFF), T.cast(pool_block_offset_sum[0], "uint32")
                ),
                "int32",
            ),
            [0],
        )




    def symm_rank_offset_arg_expr(symm_rank_offsets, mapped_rank_idx):
        if num_processes == 1:
            return symm_rank_offsets[0]
        mapped_rank_idx_u32 = T.cast(mapped_rank_idx, "uint32")
        rank_offset = symm_rank_offsets[0]
        for rank in range(1, num_processes):
            rank_offset = T.Select(
                mapped_rank_idx_u32 == T.uint32(rank), symm_rank_offsets[rank], rank_offset
            )
        return rank_offset

    def load_symm_rank_base(dst, smem_symm_rank_bases, mapped_rank_idx):
        if num_processes > 1:
            return load_shared_u64(
                dst, smem_symm_rank_bases.ptr_to([T.cast(mapped_rank_idx, "int32")])
            )

    sm100_smem_capacity = 232448
    shared_alignment = 1024
    f32_bytes = 4
    f128_bytes = 16
    num_epilogue_wgs = kernel_config.num_epilogue_warps // 4
    wg_block_m = kernel_config.block_m // num_epilogue_wgs
    atom_m = 8
    num_atoms_per_store = kernel_config.store_block_m // atom_m
    num_rows_per_warp = kernel_config.store_block_m // 8
    num_bank_group_bytes = 16
    num_hidden_bytes = hidden * 2
    num_elems_per_uint4 = 4
    num_chunk_slots = 3
    num_max_registers_for_buffer = 128
    swizzle_cd_mode = 128
    a_desc_sdo = 8 * umma_block_k // f128_bytes
    b_desc_sdo = 8 * umma_block_k // f128_bytes
    sf_desc_sdo = 8 * 4 * f32_bytes // f128_bytes
    smem_expert_count_size = _align_up(num_experts * 4, shared_alignment)
    num_pull_chunks = hidden // kernel_config.num_bytes_per_pull
    if num_pull_chunks * kernel_config.num_bytes_per_pull != hidden:
        raise ValueError("MegaMoE pull chunk size must divide hidden")
    smem_send_buffer_size = _align_up(
        kernel_config.num_bytes_per_pull * kernel_config.num_dispatch_warps, shared_alignment
    )
    smem_dispatch_size = smem_expert_count_size + smem_send_buffer_size
    smem_cd_l1_size = (
        (kernel_config.num_epilogue_warps // 4)
        * kernel_config.store_block_m
        * l1_out_block_n
        * num_tma_store_stages
    )
    smem_cd_l2_size = (
        (kernel_config.num_epilogue_warps // 4)
        * kernel_config.store_block_m
        * kernel_config.block_n
        * 2
    )
    smem_cd_size = _align_up(max(smem_cd_l1_size, smem_cd_l2_size), shared_alignment)
    smem_a_size_per_stage = kernel_config.load_block_m * kernel_config.block_k
    smem_b_size_per_stage = kernel_config.load_block_n * kernel_config.block_k
    sf_smem_outer_dim = kernel_config.block_k // 128
    smem_sfa_size_per_stage = sf_block_m * (kernel_config.block_k // 32)
    smem_sfb_size_per_stage = kernel_config.block_n * (kernel_config.block_k // 32)
    full_a_expect_tx_leader_bytes: int = smem_a_size_per_stage * 2 + (smem_sfa_size_per_stage * 2)
    full_b_expect_tx_leader_bytes: int = smem_b_size_per_stage + (smem_sfb_size_per_stage * 2)
    smem_amax_reduction_size = kernel_config.store_block_m * kernel_config.num_epilogue_warps * 4
    smem_tmem_ptr_size = 4
    smem_per_stage = (
        smem_a_size_per_stage
        + smem_b_size_per_stage
        + smem_sfa_size_per_stage
        + smem_sfb_size_per_stage
        + 16
    )
    smem_task_info_size = num_schedule_stages * task_info_bytes
    smem_fixed = (
        smem_dispatch_size
        + smem_cd_size
        + smem_amax_reduction_size
        + (
            kernel_config.num_dispatch_warps
            + num_epilogue_stages * 2
            + kernel_config.num_epilogue_warps * 2
            + num_schedule_stages * 2
        )
        * 8
        + smem_task_info_size
        + smem_tmem_ptr_size
    )
    num_stages = (sm100_smem_capacity - smem_fixed) // smem_per_stage
    if num_stages < 2:
        raise ValueError("MegaMoE requires at least two pipeline stages")
    if (
        smem_cd_size % shared_alignment != 0
        or smem_a_size_per_stage % shared_alignment != 0
        or smem_b_size_per_stage % shared_alignment != 0
    ):
        raise ValueError("Shared memory of CD/A/B must be aligned to 1024 bytes")
    if num_stages > 32:
        raise ValueError("Too many stages")
    if (
        num_dispatch_registers * kernel_config.num_dispatch_threads
        + num_non_epilogue_registers * kernel_config.num_non_epilogue_threads
        + num_epilogue_registers * kernel_config.num_epilogue_threads
        > 64512
    ):
        raise ValueError("Too many registers")
    smem_a_layout = mma_shared_layout(
        "int8",
        SwizzleMode.SWIZZLE_128B_ATOM,
        (num_stages, kernel_config.load_block_m, kernel_config.block_k),
    )
    smem_b_layout = mma_shared_layout(
        "uint8",
        SwizzleMode.SWIZZLE_128B_ATOM,
        (num_stages, kernel_config.load_block_n, kernel_config.block_k),
    )
    num_total_barriers = (
        kernel_config.num_dispatch_warps
        + num_stages * 2
        + num_epilogue_stages * 2
        + kernel_config.num_epilogue_warps * 2
        + num_schedule_stages * 2
    )
    dispatch_barrier_base = 0
    full_barrier_base = dispatch_barrier_base + kernel_config.num_dispatch_warps
    empty_barrier_base = full_barrier_base + num_stages
    tmem_full_barrier_base = empty_barrier_base + num_stages
    tmem_empty_barrier_base = tmem_full_barrier_base + num_epilogue_stages
    combine_barrier_base = tmem_empty_barrier_base + num_epilogue_stages
    task_info_full_barrier_base = combine_barrier_base + kernel_config.num_epilogue_warps * 2
    task_info_empty_barrier_base = task_info_full_barrier_base + num_schedule_stages
    dispatch_sync_barrier_idx = 0
    dispatch_with_epilogue_sync_barrier_idx = 1
    epilogue_full_sync_barrier_idx = 2
    epilogue_wg_sync_barrier_start_idx = 3
    dispatch_with_load_a_sync_barrier_idx = epilogue_wg_sync_barrier_start_idx + num_epilogue_wgs
    before_dispatch_pull_barrier_tag = 1
    before_combine_reduce_barrier_tag = 2
    after_workspace_clean_barrier_tag = 3
    nvlink_barrier_timeout_seconds = int(os.environ.get("TIRX_MEGAMOE_NVL_TIMEOUT_SECONDS", "300"))
    if nvlink_barrier_timeout_seconds <= 0:
        raise ValueError("TIRX_MEGAMOE_NVL_TIMEOUT_SECONDS must be positive")
    num_nvlink_barrier_timeout_cycles = nvlink_barrier_timeout_seconds * 2000000000
    dispatch_grid_sync_index = 0
    epilogue_grid_sync_index = 1
    smem_expert_count_offset = 0
    smem_send_buffer_offset = smem_expert_count_offset + smem_expert_count_size
    smem_gemm_base_offset = smem_send_buffer_offset + smem_send_buffer_size
    smem_cd_offset = smem_gemm_base_offset
    smem_a_offset = smem_cd_offset + smem_cd_size
    smem_b_offset = smem_a_offset + num_stages * smem_a_size_per_stage
    smem_sfa_offset = smem_b_offset + num_stages * smem_b_size_per_stage
    smem_sfb_offset = smem_sfa_offset + num_stages * smem_sfa_size_per_stage
    smem_amax_reduction_offset = smem_sfb_offset + num_stages * smem_sfb_size_per_stage
    # `task_info_t task_infos[kNumScheduleStages]` (alignas(16)) sits between the
    # amax buffer and the barriers, mirroring the SharedStorage member order.
    smem_task_info_offset = _align_up(smem_amax_reduction_offset + smem_amax_reduction_size, 16)
    smem_barrier_offset = smem_task_info_offset + smem_task_info_size
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_symm_rank_bases_offset = _align_up(smem_tmem_ptr_offset + smem_tmem_ptr_size, 8)
    smem_symm_rank_bases_size = num_processes * 8 if num_processes > 1 else 0
    smem_total_bytes = (
        smem_symm_rank_bases_offset + smem_symm_rank_bases_size
        if num_processes > 1
        else smem_tmem_ptr_offset + smem_tmem_ptr_size
    )
    num_chunks = (
        1
        if num_chunk_slots * kernel_config.num_epilogue_warps * num_hidden_bytes
        <= smem_barrier_offset
        and hidden <= 32 * num_max_registers_for_buffer
        else 2
    )
    num_chunk_bytes = num_hidden_bytes // num_chunks
    num_chunk_uint4 = num_chunk_bytes // 16
    num_uint4_per_lane = num_chunk_uint4 // 32
    if hidden % num_chunks != 0:
        raise ValueError("Hidden must be divisible by number of chunks")
    if num_chunk_slots * kernel_config.num_epilogue_warps * num_chunk_bytes > smem_barrier_offset:
        raise ValueError("Hidden is too large")
    if num_chunk_bytes % 16 != 0:
        raise ValueError("Combine chunk must be TMA-aligned (16 bytes)")
    if num_chunk_bytes % 16 != 0:
        raise ValueError("Combine chunk must be divisible by 16 bytes")
    if num_chunk_uint4 % 32 != 0:
        raise ValueError("Combine chunk must be a multiple of 32 16-byte elements (one per lane)")
    if num_topk > 32:
        raise ValueError("Top-k must fit in a single warp")

    builder = IRBuilder()
    with _builder_context(builder):
        with T.prim_func():
            T.func_name("mega_moe")
            y_ptr = T.arg("y_ptr", T.handle())
            cumulative_local_expert_recv_stats_ptr = T.arg(
                "cumulative_local_expert_recv_stats_ptr", T.handle()
            )
            symm_buffer_ptr = T.arg("symm_buffer_ptr", T.handle())
            symm_rank_offset_0 = T.arg("symm_rank_offset_0", T.int64())
            symm_rank_offset_1 = T.arg("symm_rank_offset_1", T.int64())
            symm_rank_offset_2 = T.arg("symm_rank_offset_2", T.int64())
            symm_rank_offset_3 = T.arg("symm_rank_offset_3", T.int64())
            symm_rank_offset_4 = T.arg("symm_rank_offset_4", T.int64())
            symm_rank_offset_5 = T.arg("symm_rank_offset_5", T.int64())
            symm_rank_offset_6 = T.arg("symm_rank_offset_6", T.int64())
            symm_rank_offset_7 = T.arg("symm_rank_offset_7", T.int64())
            symm_rank_offset_8 = T.arg("symm_rank_offset_8", T.int64())
            symm_rank_offset_9 = T.arg("symm_rank_offset_9", T.int64())
            symm_rank_offset_10 = T.arg("symm_rank_offset_10", T.int64())
            symm_rank_offset_11 = T.arg("symm_rank_offset_11", T.int64())
            symm_rank_offset_12 = T.arg("symm_rank_offset_12", T.int64())
            symm_rank_offset_13 = T.arg("symm_rank_offset_13", T.int64())
            symm_rank_offset_14 = T.arg("symm_rank_offset_14", T.int64())
            symm_rank_offset_15 = T.arg("symm_rank_offset_15", T.int64())
            symm_rank_offset_16 = T.arg("symm_rank_offset_16", T.int64())
            symm_rank_offset_17 = T.arg("symm_rank_offset_17", T.int64())
            symm_rank_offset_18 = T.arg("symm_rank_offset_18", T.int64())
            symm_rank_offset_19 = T.arg("symm_rank_offset_19", T.int64())
            symm_rank_offset_20 = T.arg("symm_rank_offset_20", T.int64())
            symm_rank_offset_21 = T.arg("symm_rank_offset_21", T.int64())
            symm_rank_offset_22 = T.arg("symm_rank_offset_22", T.int64())
            symm_rank_offset_23 = T.arg("symm_rank_offset_23", T.int64())
            symm_rank_offset_24 = T.arg("symm_rank_offset_24", T.int64())
            symm_rank_offset_25 = T.arg("symm_rank_offset_25", T.int64())
            symm_rank_offset_26 = T.arg("symm_rank_offset_26", T.int64())
            symm_rank_offset_27 = T.arg("symm_rank_offset_27", T.int64())
            symm_rank_offset_28 = T.arg("symm_rank_offset_28", T.int64())
            symm_rank_offset_29 = T.arg("symm_rank_offset_29", T.int64())
            symm_rank_offset_30 = T.arg("symm_rank_offset_30", T.int64())
            symm_rank_offset_31 = T.arg("symm_rank_offset_31", T.int64())
            symm_rank_offset_32 = T.arg("symm_rank_offset_32", T.int64())
            symm_rank_offset_33 = T.arg("symm_rank_offset_33", T.int64())
            symm_rank_offset_34 = T.arg("symm_rank_offset_34", T.int64())
            symm_rank_offset_35 = T.arg("symm_rank_offset_35", T.int64())
            symm_rank_offset_36 = T.arg("symm_rank_offset_36", T.int64())
            symm_rank_offset_37 = T.arg("symm_rank_offset_37", T.int64())
            symm_rank_offset_38 = T.arg("symm_rank_offset_38", T.int64())
            symm_rank_offset_39 = T.arg("symm_rank_offset_39", T.int64())
            symm_rank_offset_40 = T.arg("symm_rank_offset_40", T.int64())
            symm_rank_offset_41 = T.arg("symm_rank_offset_41", T.int64())
            symm_rank_offset_42 = T.arg("symm_rank_offset_42", T.int64())
            symm_rank_offset_43 = T.arg("symm_rank_offset_43", T.int64())
            symm_rank_offset_44 = T.arg("symm_rank_offset_44", T.int64())
            symm_rank_offset_45 = T.arg("symm_rank_offset_45", T.int64())
            symm_rank_offset_46 = T.arg("symm_rank_offset_46", T.int64())
            symm_rank_offset_47 = T.arg("symm_rank_offset_47", T.int64())
            symm_rank_offset_48 = T.arg("symm_rank_offset_48", T.int64())
            symm_rank_offset_49 = T.arg("symm_rank_offset_49", T.int64())
            symm_rank_offset_50 = T.arg("symm_rank_offset_50", T.int64())
            symm_rank_offset_51 = T.arg("symm_rank_offset_51", T.int64())
            symm_rank_offset_52 = T.arg("symm_rank_offset_52", T.int64())
            symm_rank_offset_53 = T.arg("symm_rank_offset_53", T.int64())
            symm_rank_offset_54 = T.arg("symm_rank_offset_54", T.int64())
            symm_rank_offset_55 = T.arg("symm_rank_offset_55", T.int64())
            symm_rank_offset_56 = T.arg("symm_rank_offset_56", T.int64())
            symm_rank_offset_57 = T.arg("symm_rank_offset_57", T.int64())
            symm_rank_offset_58 = T.arg("symm_rank_offset_58", T.int64())
            symm_rank_offset_59 = T.arg("symm_rank_offset_59", T.int64())
            symm_rank_offset_60 = T.arg("symm_rank_offset_60", T.int64())
            symm_rank_offset_61 = T.arg("symm_rank_offset_61", T.int64())
            symm_rank_offset_62 = T.arg("symm_rank_offset_62", T.int64())
            symm_rank_offset_63 = T.arg("symm_rank_offset_63", T.int64())
            symm_rank_offset_64 = T.arg("symm_rank_offset_64", T.int64())
            symm_rank_offset_65 = T.arg("symm_rank_offset_65", T.int64())
            symm_rank_offset_66 = T.arg("symm_rank_offset_66", T.int64())
            symm_rank_offset_67 = T.arg("symm_rank_offset_67", T.int64())
            symm_rank_offset_68 = T.arg("symm_rank_offset_68", T.int64())
            symm_rank_offset_69 = T.arg("symm_rank_offset_69", T.int64())
            symm_rank_offset_70 = T.arg("symm_rank_offset_70", T.int64())
            symm_rank_offset_71 = T.arg("symm_rank_offset_71", T.int64())
            tensor_map_l1_acts = T.arg("tensor_map_l1_acts", T.TensorMap())
            tensor_map_l1_acts_sf = T.arg("tensor_map_l1_acts_sf", T.TensorMap())
            tensor_map_l1_weights = T.arg("tensor_map_l1_weights", T.TensorMap())
            tensor_map_l1_weights_sf = T.arg("tensor_map_l1_weights_sf", T.TensorMap())
            tensor_map_l1_output = T.arg("tensor_map_l1_output", T.TensorMap())
            tensor_map_l2_acts = T.arg("tensor_map_l2_acts", T.TensorMap())
            tensor_map_l2_acts_sf = T.arg("tensor_map_l2_acts_sf", T.TensorMap())
            tensor_map_l2_weights = T.arg("tensor_map_l2_weights", T.TensorMap())
            tensor_map_l2_weights_sf = T.arg("tensor_map_l2_weights_sf", T.TensorMap())
            num_tokens = T.arg("num_tokens", T.int32())
            rank_idx = T.arg("rank_idx", T.int32())
            y = _builder_assign(
                "y",
                T.match_buffer(y_ptr, (num_tokens, hidden), "bfloat16"),
                locals().get("y", _BUILDER_MISSING),
            )
            cumulative_local_expert_recv_stats = _builder_assign(
                "cumulative_local_expert_recv_stats",
                T.match_buffer(
                    cumulative_local_expert_recv_stats_ptr, (num_experts_per_rank,), "int32"
                ),
                locals().get("cumulative_local_expert_recv_stats", _BUILDER_MISSING),
            )
            symm_buffer = _builder_assign(
                "symm_buffer",
                T.match_buffer(symm_buffer_ptr, (symm_buffer_layout.total_bytes,), "int8"),
                locals().get("symm_buffer", _BUILDER_MISSING),
            )
            _builder_emit(T.device_entry())
            _builder_emit(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            symm_rank_offsets = (
                symm_rank_offset_0,
                symm_rank_offset_1,
                symm_rank_offset_2,
                symm_rank_offset_3,
                symm_rank_offset_4,
                symm_rank_offset_5,
                symm_rank_offset_6,
                symm_rank_offset_7,
                symm_rank_offset_8,
                symm_rank_offset_9,
                symm_rank_offset_10,
                symm_rank_offset_11,
                symm_rank_offset_12,
                symm_rank_offset_13,
                symm_rank_offset_14,
                symm_rank_offset_15,
                symm_rank_offset_16,
                symm_rank_offset_17,
                symm_rank_offset_18,
                symm_rank_offset_19,
                symm_rank_offset_20,
                symm_rank_offset_21,
                symm_rank_offset_22,
                symm_rank_offset_23,
                symm_rank_offset_24,
                symm_rank_offset_25,
                symm_rank_offset_26,
                symm_rank_offset_27,
                symm_rank_offset_28,
                symm_rank_offset_29,
                symm_rank_offset_30,
                symm_rank_offset_31,
                symm_rank_offset_32,
                symm_rank_offset_33,
                symm_rank_offset_34,
                symm_rank_offset_35,
                symm_rank_offset_36,
                symm_rank_offset_37,
                symm_rank_offset_38,
                symm_rank_offset_39,
                symm_rank_offset_40,
                symm_rank_offset_41,
                symm_rank_offset_42,
                symm_rank_offset_43,
                symm_rank_offset_44,
                symm_rank_offset_45,
                symm_rank_offset_46,
                symm_rank_offset_47,
                symm_rank_offset_48,
                symm_rank_offset_49,
                symm_rank_offset_50,
                symm_rank_offset_51,
                symm_rank_offset_52,
                symm_rank_offset_53,
                symm_rank_offset_54,
                symm_rank_offset_55,
                symm_rank_offset_56,
                symm_rank_offset_57,
                symm_rank_offset_58,
                symm_rank_offset_59,
                symm_rank_offset_60,
                symm_rank_offset_61,
                symm_rank_offset_62,
                symm_rank_offset_63,
                symm_rank_offset_64,
                symm_rank_offset_65,
                symm_rank_offset_66,
                symm_rank_offset_67,
                symm_rank_offset_68,
                symm_rank_offset_69,
                symm_rank_offset_70,
                symm_rank_offset_71,
            )
            sym_buffer_base = _builder_assign(
                "sym_buffer_base",
                ptr_to_u64(symm_buffer.ptr_to([0])),
                locals().get("sym_buffer_base", _BUILDER_MISSING),
            )
            thread_idx = _builder_assign(
                "thread_idx",
                T.thread_id([kernel_config.num_threads_per_cta]),
                locals().get("thread_idx", _BUILDER_MISSING),
            )
            cta_idx_in_cluster = _builder_assign(
                "cta_idx_in_cluster",
                T.cta_id_in_cluster([kernel_config.num_ctas_per_cluster]),
                locals().get("cta_idx_in_cluster", _BUILDER_MISSING),
            )
            sm_idx = _builder_assign(
                "sm_idx",
                T.cta_id([kernel_config.num_sms]),
                locals().get("sm_idx", _BUILDER_MISSING),
            )
            wg_id = _builder_assign(
                "wg_id",
                T.warpgroup_id([kernel_config.num_warpgroups_per_cta]),
                locals().get("wg_id", _BUILDER_MISSING),
            )
            warp_id = _builder_assign(
                "warp_id",
                T.warp_id_in_wg([num_warps_per_warpgroup]),
                locals().get("warp_id", _BUILDER_MISSING),
            )
            lane_idx = _builder_assign(
                "lane_idx", T.lane_id([32]), locals().get("lane_idx", _BUILDER_MISSING)
            )
            flat_warp_idx = _builder_assign(
                "flat_warp_idx",
                wg_id * num_warps_per_warpgroup + warp_id,
                locals().get("flat_warp_idx", _BUILDER_MISSING),
            )
            with T.If(flat_warp_idx == 0):
                with T.Then():
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l1_acts)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l1_acts_sf)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l1_weights)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l1_weights_sf)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l1_output)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l2_acts)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l2_acts_sf)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l2_weights)))
                    _builder_emit(T.evaluate(prefetch_tensormap(tensor_map_l2_weights_sf)))
            input_topk_idx_data = _builder_bind(
                "input_topk_idx_data",
                T.reinterpret(
                    PointerType(PrimType("int64")),
                    symm_buffer.ptr_to([symm_buffer_layout.input_topk_idx_offset]),
                ),
                None,
            )
            l1_acts_sf_data = _builder_bind(
                "l1_acts_sf_data",
                T.reinterpret(
                    PointerType(PrimType("int32")),
                    symm_buffer.ptr_to([symm_buffer_layout.l1_sf_offset]),
                ),
                None,
            )
            workspace_expert_send_count_data = _builder_bind(
                "workspace_expert_send_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint64")),
                    symm_buffer.ptr_to([workspace_layout.expert_send_count_offset]),
                ),
                None,
            )
            workspace_grid_sync_count_data = _builder_bind(
                "workspace_grid_sync_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.barrier_offset]),
                ),
                None,
            )
            workspace_nvl_barrier_counter_data = _builder_bind(
                "workspace_nvl_barrier_counter_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.barrier_offset + 16]),
                ),
                None,
            )
            workspace_nvl_barrier_signal_data = _builder_bind(
                "workspace_nvl_barrier_signal_data",
                T.reinterpret(
                    PointerType(PrimType("int32")),
                    symm_buffer.ptr_to([workspace_layout.barrier_offset + 20]),
                ),
                None,
            )
            workspace_expert_recv_count_data = _builder_bind(
                "workspace_expert_recv_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint64")),
                    symm_buffer.ptr_to([workspace_layout.expert_recv_count_offset]),
                ),
                None,
            )
            workspace_expert_recv_count_sum_data = _builder_bind(
                "workspace_expert_recv_count_sum_data",
                T.reinterpret(
                    PointerType(PrimType("uint64")),
                    symm_buffer.ptr_to([workspace_layout.expert_recv_count_sum_offset]),
                ),
                None,
            )
            workspace_src_token_topk_idx_data = _builder_bind(
                "workspace_src_token_topk_idx_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.src_token_topk_idx_offset]),
                ),
                None,
            )
            workspace_token_src_metadata_data = _builder_bind(
                "workspace_token_src_metadata_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.token_src_metadata_offset]),
                ),
                None,
            )
            workspace_l1_full_count_data = _builder_bind(
                "workspace_l1_full_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.l1_full_count_offset]),
                ),
                None,
            )
            workspace_l1_empty_count_data = _builder_bind(
                "workspace_l1_empty_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.l1_empty_count_offset]),
                ),
                None,
            )
            workspace_l2_full_count_data = _builder_bind(
                "workspace_l2_full_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.l2_full_count_offset]),
                ),
                None,
            )
            workspace_l2_empty_count_data = _builder_bind(
                "workspace_l2_empty_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.l2_empty_count_offset]),
                ),
                None,
            )
            workspace_l1_task_count_data = _builder_bind(
                "workspace_l1_task_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.l1_task_count_offset]),
                ),
                None,
            )
            workspace_l2_task_count_data = _builder_bind(
                "workspace_l2_task_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.l2_task_count_offset]),
                ),
                None,
            )
            workspace_shared_l1_task_count_data = _builder_bind(
                "workspace_shared_l1_task_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.shared_l1_task_count_offset]),
                ),
                None,
            )
            workspace_shared_l2_task_count_data = _builder_bind(
                "workspace_shared_l2_task_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.shared_l2_task_count_offset]),
                ),
                None,
            )
            workspace_shared_l2_full_count_data = _builder_bind(
                "workspace_shared_l2_full_count_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")),
                    symm_buffer.ptr_to([workspace_layout.shared_l2_full_count_offset]),
                ),
                None,
            )
            l1_topk_weights_data = _builder_bind(
                "l1_topk_weights_data",
                T.reinterpret(
                    PointerType(PrimType("float32")),
                    symm_buffer.ptr_to([symm_buffer_layout.l1_topk_weights_offset]),
                ),
                None,
            )
            l2_acts_sf_data = _builder_bind(
                "l2_acts_sf_data",
                T.reinterpret(
                    PointerType(PrimType("int32")),
                    symm_buffer.ptr_to([symm_buffer_layout.l2_sf_offset]),
                ),
                None,
            )
            combine_tokens_data = _builder_bind(
                "combine_tokens_data",
                T.reinterpret(
                    PointerType(PrimType("uint16")),
                    symm_buffer.ptr_to([symm_buffer_layout.combine_token_offset]),
                ),
                None,
            )
            input_topk_idx = _builder_assign(
                "input_topk_idx",
                T.decl_buffer(
                    (workspace_layout.num_max_tokens_per_rank, num_topk),
                    "int64",
                    data=input_topk_idx_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("input_topk_idx", _BUILDER_MISSING),
            )
            l1_acts = _builder_assign(
                "l1_acts",
                T.decl_buffer(
                    (num_ring_tokens, hidden),
                    "int8",
                    data=symm_buffer.data,
                    scope="global",
                    elem_offset=symm_buffer_layout.l1_token_offset,
                ),
                locals().get("l1_acts", _BUILDER_MISSING),
            )
            l1_acts_sf = _builder_assign(
                "l1_acts_sf",
                T.decl_buffer(
                    (hidden // 128, workspace_layout.num_sf_ring_tokens),
                    "int32",
                    data=symm_buffer.data,
                    scope="global",
                    elem_offset=symm_buffer_layout.l1_sf_offset // 4,
                ),
                locals().get("l1_acts_sf", _BUILDER_MISSING),
            )
            workspace_expert_send_count = _builder_assign(
                "workspace_expert_send_count",
                T.decl_buffer(
                    (num_experts,),
                    "uint64",
                    data=workspace_expert_send_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_expert_send_count", _BUILDER_MISSING),
            )
            workspace_grid_sync_count = _builder_assign(
                "workspace_grid_sync_count",
                T.decl_buffer(
                    (4,),
                    "uint32",
                    data=workspace_grid_sync_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_grid_sync_count", _BUILDER_MISSING),
            )
            workspace_nvl_barrier_counter = _builder_assign(
                "workspace_nvl_barrier_counter",
                T.decl_buffer(
                    (1,),
                    "uint32",
                    data=workspace_nvl_barrier_counter_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_nvl_barrier_counter", _BUILDER_MISSING),
            )
            workspace_nvl_barrier_signal = _builder_assign(
                "workspace_nvl_barrier_signal",
                T.decl_buffer(
                    (2,),
                    "int32",
                    data=workspace_nvl_barrier_signal_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_nvl_barrier_signal", _BUILDER_MISSING),
            )
            workspace_expert_recv_count = _builder_assign(
                "workspace_expert_recv_count",
                T.decl_buffer(
                    (num_processes, num_experts_per_rank),
                    "uint64",
                    data=workspace_expert_recv_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_expert_recv_count", _BUILDER_MISSING),
            )
            workspace_expert_recv_count_sum = _builder_assign(
                "workspace_expert_recv_count_sum",
                T.decl_buffer(
                    (num_experts_per_rank,),
                    "uint64",
                    data=workspace_expert_recv_count_sum_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_expert_recv_count_sum", _BUILDER_MISSING),
            )
            workspace_src_token_topk_idx = _builder_assign(
                "workspace_src_token_topk_idx",
                T.decl_buffer(
                    (
                        num_experts_per_rank,
                        num_processes,
                        workspace_layout.num_max_recv_tokens_per_expert,
                    ),
                    "uint32",
                    data=workspace_src_token_topk_idx_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_src_token_topk_idx", _BUILDER_MISSING),
            )
            workspace_token_src_metadata = _builder_assign(
                "workspace_token_src_metadata",
                T.decl_buffer(
                    (workspace_layout.num_max_pool_tokens, 3),
                    "uint32",
                    data=workspace_token_src_metadata_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_token_src_metadata", _BUILDER_MISSING),
            )
            workspace_l1_full_count = _builder_assign(
                "workspace_l1_full_count",
                T.decl_buffer(
                    (workspace_layout.num_ring_blocks,),
                    "uint32",
                    data=workspace_l1_full_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_l1_full_count", _BUILDER_MISSING),
            )
            workspace_l1_empty_count = _builder_assign(
                "workspace_l1_empty_count",
                T.decl_buffer(
                    (workspace_layout.num_ring_blocks,),
                    "uint32",
                    data=workspace_l1_empty_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_l1_empty_count", _BUILDER_MISSING),
            )
            workspace_l2_full_count = _builder_assign(
                "workspace_l2_full_count",
                T.decl_buffer(
                    (workspace_layout.num_ring_blocks,),
                    "uint32",
                    data=workspace_l2_full_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_l2_full_count", _BUILDER_MISSING),
            )
            workspace_l2_empty_count = _builder_assign(
                "workspace_l2_empty_count",
                T.decl_buffer(
                    (workspace_layout.num_ring_blocks,),
                    "uint32",
                    data=workspace_l2_empty_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_l2_empty_count", _BUILDER_MISSING),
            )
            workspace_l1_task_count = _builder_assign(
                "workspace_l1_task_count",
                T.decl_buffer(
                    (1,), "uint32", data=workspace_l1_task_count_data, scope="global", elem_offset=0
                ),
                locals().get("workspace_l1_task_count", _BUILDER_MISSING),
            )
            workspace_l2_task_count = _builder_assign(
                "workspace_l2_task_count",
                T.decl_buffer(
                    (1,), "uint32", data=workspace_l2_task_count_data, scope="global", elem_offset=0
                ),
                locals().get("workspace_l2_task_count", _BUILDER_MISSING),
            )
            workspace_shared_l1_task_count = _builder_assign(
                "workspace_shared_l1_task_count",
                T.decl_buffer(
                    (1,),
                    "uint32",
                    data=workspace_shared_l1_task_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_shared_l1_task_count", _BUILDER_MISSING),
            )
            workspace_shared_l2_task_count = _builder_assign(
                "workspace_shared_l2_task_count",
                T.decl_buffer(
                    (1,),
                    "uint32",
                    data=workspace_shared_l2_task_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_shared_l2_task_count", _BUILDER_MISSING),
            )
            workspace_shared_l2_full_count = _builder_assign(
                "workspace_shared_l2_full_count",
                T.decl_buffer(
                    (workspace_layout.num_shared_l2_pool_blocks,),
                    "uint32",
                    data=workspace_shared_l2_full_count_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("workspace_shared_l2_full_count", _BUILDER_MISSING),
            )
            l1_topk_weights = _builder_assign(
                "l1_topk_weights",
                T.decl_buffer(
                    (num_ring_tokens,),
                    "float32",
                    data=l1_topk_weights_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("l1_topk_weights", _BUILDER_MISSING),
            )
            l2_acts = _builder_assign(
                "l2_acts",
                T.decl_buffer(
                    (num_ring_tokens, intermediate_hidden),
                    "int8",
                    data=symm_buffer.data,
                    scope="global",
                    elem_offset=symm_buffer_layout.l2_token_offset,
                ),
                locals().get("l2_acts", _BUILDER_MISSING),
            )
            l2_sf_buffer = _builder_assign(
                "l2_sf_buffer",
                T.decl_buffer(
                    (intermediate_hidden // 128 * workspace_layout.num_sf_ring_tokens * 4,),
                    "int8",
                    data=symm_buffer.data,
                    scope="global",
                    elem_offset=symm_buffer_layout.l2_sf_offset,
                ),
                locals().get("l2_sf_buffer", _BUILDER_MISSING),
            )
            combine_tokens = _builder_assign(
                "combine_tokens",
                T.decl_buffer(
                    (num_topk, workspace_layout.num_max_tokens_per_rank, hidden),
                    "uint16",
                    data=combine_tokens_data,
                    scope="global",
                    elem_offset=0,
                ),
                locals().get("combine_tokens", _BUILDER_MISSING),
            )

            smem = _builder_assign(
                "smem",
                T.alloc_buffer([smem_total_bytes], "uint8", scope="shared.dyn"),
                locals().get("smem", _BUILDER_MISSING),
            )
            _builder_emit(T.attr({"tirx.dyn_smem_bytes": smem_total_bytes}))
            smem_expert_count_data = _builder_bind(
                "smem_expert_count_data",
                T.reinterpret(
                    PointerType(PrimType("int32")), smem.ptr_to([smem_expert_count_offset])
                ),
                None,
            )
            smem_send_buffer_data = _builder_bind(
                "smem_send_buffer_data",
                T.reinterpret(
                    PointerType(PrimType("int8")), smem.ptr_to([smem_send_buffer_offset])
                ),
                None,
            )
            smem_a_data = _builder_bind(
                "smem_a_data",
                T.reinterpret(PointerType(PrimType("int8")), smem.ptr_to([smem_a_offset])),
                None,
            )
            smem_b_data = _builder_bind(
                "smem_b_data",
                T.reinterpret(PointerType(PrimType("uint8")), smem.ptr_to([smem_b_offset])),
                None,
            )
            smem_sfa_data = _builder_bind(
                "smem_sfa_data",
                T.reinterpret(PointerType(PrimType("int32")), smem.ptr_to([smem_sfa_offset])),
                None,
            )
            smem_sfb_data = _builder_bind(
                "smem_sfb_data",
                T.reinterpret(PointerType(PrimType("int32")), smem.ptr_to([smem_sfb_offset])),
                None,
            )
            smem_amax_reduction_data = _builder_bind(
                "smem_amax_reduction_data",
                T.reinterpret(
                    PointerType(PrimType("float32")), smem.ptr_to([smem_amax_reduction_offset])
                ),
                None,
            )
            smem_task_info_data = _builder_bind(
                "smem_task_info_data",
                T.reinterpret(
                    PointerType(PrimType("uint32")), smem.ptr_to([smem_task_info_offset])
                ),
                None,
            )
            smem_cd_data = _builder_bind(
                "smem_cd_data",
                T.reinterpret(PointerType(PrimType("uint8")), smem.ptr_to([smem_cd_offset])),
                None,
            )
            smem_barrier_data = _builder_bind(
                "smem_barrier_data",
                T.reinterpret(PointerType(PrimType("uint64")), smem.ptr_to([smem_barrier_offset])),
                None,
            )
            smem_tmem_ptr_data = _builder_bind(
                "smem_tmem_ptr_data",
                T.reinterpret(PointerType(PrimType("uint32")), smem.ptr_to([smem_tmem_ptr_offset])),
                None,
            )
            smem_symm_rank_bases_data = _builder_bind(
                "smem_symm_rank_bases_data",
                T.reinterpret(
                    PointerType(PrimType("uint64")), smem.ptr_to([smem_symm_rank_bases_offset])
                ),
                None,
            )
            smem_expert_count = _builder_assign(
                "smem_expert_count",
                T.decl_buffer(
                    (num_experts,),
                    "int32",
                    data=smem_expert_count_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=shared_alignment,
                ),
                locals().get("smem_expert_count", _BUILDER_MISSING),
            )
            smem_send_buffers = _builder_assign(
                "smem_send_buffers",
                T.decl_buffer(
                    (kernel_config.num_dispatch_warps, kernel_config.num_bytes_per_pull),
                    "int8",
                    data=smem_send_buffer_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=shared_alignment,
                ),
                locals().get("smem_send_buffers", _BUILDER_MISSING),
            )
            smem_a = _builder_assign(
                "smem_a",
                T.decl_buffer(
                    (num_stages, kernel_config.load_block_m, kernel_config.block_k),
                    "int8",
                    data=smem_a_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=shared_alignment,
                    layout=smem_a_layout,
                ),
                locals().get("smem_a", _BUILDER_MISSING),
            )
            smem_a_fp8 = _builder_assign(
                "smem_a_fp8",
                T.decl_buffer(
                    (num_stages, kernel_config.load_block_m, kernel_config.block_k),
                    "float8_e4m3fn",
                    data=smem_a_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=shared_alignment,
                    layout=smem_a_layout,
                ),
                locals().get("smem_a_fp8", _BUILDER_MISSING),
            )
            smem_b = _builder_assign(
                "smem_b",
                T.decl_buffer(
                    (num_stages, kernel_config.load_block_n, kernel_config.block_k),
                    "uint8",
                    data=smem_b_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=shared_alignment,
                    layout=smem_b_layout,
                ),
                locals().get("smem_b", _BUILDER_MISSING),
            )
            smem_sfa_i32 = _builder_assign(
                "smem_sfa_i32",
                T.decl_buffer(
                    (num_stages, sf_block_m, sf_smem_outer_dim),
                    "int32",
                    data=smem_sfa_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_sfa_i32", _BUILDER_MISSING),
            )
            smem_sfa = _builder_assign(
                "smem_sfa",
                T.decl_buffer(
                    (num_stages, sf_block_m * sf_smem_outer_dim),
                    "uint32",
                    data=smem_sfa_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_sfa", _BUILDER_MISSING),
            )
            smem_sfb_i32 = _builder_assign(
                "smem_sfb_i32",
                T.decl_buffer(
                    (num_stages, kernel_config.block_n, sf_smem_outer_dim),
                    "int32",
                    data=smem_sfb_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_sfb_i32", _BUILDER_MISSING),
            )
            smem_sfb = _builder_assign(
                "smem_sfb",
                T.decl_buffer(
                    (num_stages, kernel_config.block_n * sf_smem_outer_dim),
                    "uint32",
                    data=smem_sfb_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_sfb", _BUILDER_MISSING),
            )
            smem_amax_reduction = _builder_assign(
                "smem_amax_reduction",
                T.decl_buffer(
                    (kernel_config.num_epilogue_warps * kernel_config.store_block_m,),
                    "float32",
                    data=smem_amax_reduction_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_amax_reduction", _BUILDER_MISSING),
            )
            smem_task_infos = _builder_assign(
                "smem_task_infos",
                T.decl_buffer(
                    (num_schedule_stages, 8),
                    "uint32",
                    data=smem_task_info_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_task_infos", _BUILDER_MISSING),
            )
            smem_cd_l1 = _builder_assign(
                "smem_cd_l1",
                T.decl_buffer(
                    (
                        num_tma_store_stages,
                        num_epilogue_wgs,
                        kernel_config.store_block_m,
                        l1_out_block_n,
                    ),
                    "int8",
                    data=smem_cd_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_cd_l1", _BUILDER_MISSING),
            )
            smem_cd_l2 = _builder_assign(
                "smem_cd_l2",
                T.decl_buffer(
                    (num_epilogue_wgs, kernel_config.store_block_m, kernel_config.block_n),
                    "uint16",
                    data=smem_cd_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_cd_l2", _BUILDER_MISSING),
            )
            smem_barriers = _builder_assign(
                "smem_barriers",
                T.decl_buffer(
                    (num_total_barriers,),
                    "uint64",
                    data=smem_barrier_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=8,
                ),
                locals().get("smem_barriers", _BUILDER_MISSING),
            )
            tmem_ptr_in_smem = _builder_assign(
                "tmem_ptr_in_smem",
                T.decl_buffer(
                    (1,),
                    "uint32",
                    data=smem_tmem_ptr_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=4,
                ),
                locals().get("tmem_ptr_in_smem", _BUILDER_MISSING),
            )
            smem_symm_rank_bases = _builder_assign(
                "smem_symm_rank_bases",
                T.decl_buffer(
                    (num_processes,),
                    "uint64",
                    data=smem_symm_rank_bases_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=8,
                ),
                locals().get("smem_symm_rank_bases", _BUILDER_MISSING),
            )
            combine_chunks = _builder_assign(
                "combine_chunks",
                T.decl_buffer(
                    (num_chunk_slots, kernel_config.num_epilogue_warps, num_chunk_uint4, 4),
                    "uint32",
                    data=smem.data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("combine_chunks", _BUILDER_MISSING),
            )
            tmem = _builder_assign(
                "tmem",
                T.decl_buffer(
                    (128, num_tmem_cols),
                    "float32",
                    scope="tmem",
                    allocated_addr=tmem_ptr_in_smem[0],
                    layout=TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)]),
                ),
                locals().get("tmem", _BUILDER_MISSING),
            )
            sfa_tmem = _builder_assign(
                "sfa_tmem",
                T.decl_buffer(
                    (128, sf_block_m // 32),
                    "float8_e8m0fnu",
                    scope="tmem",
                    allocated_addr=num_accum_tmem_cols,
                    layout=sf_tmem_layout(128, SF_K=sf_block_m // 32, sf_per_mma=sf_block_m // 32),
                ),
                locals().get("sfa_tmem", _BUILDER_MISSING),
            )
            sfb_tmem = _builder_assign(
                "sfb_tmem",
                T.decl_buffer(
                    (128, kernel_config.block_n // 32),
                    "float8_e8m0fnu",
                    scope="tmem",
                    allocated_addr=num_accum_tmem_cols + num_sfa_tmem_cols,
                    layout=sf_tmem_layout(
                        128,
                        SF_K=kernel_config.block_n // 32,
                        sf_per_mma=kernel_config.block_n // 32,
                    ),
                ),
                locals().get("sfb_tmem", _BUILDER_MISSING),
            )
            desc_a = _builder_alloc_scalar("desc_a", "uint64")
            desc_b = _builder_alloc_scalar("desc_b", "uint64")
            desc_sf = _builder_alloc_scalar("desc_sf", "uint64")
            desc_i = _builder_alloc_scalar("desc_i", "uint32")
            runtime_desc_i = _builder_alloc_scalar("runtime_desc_i", "uint32")
            a_desc_lo = _builder_alloc_scalar("a_desc_lo", "uint32")
            b_desc_lo = _builder_alloc_scalar("b_desc_lo", "uint32")
            a_desc_base_lo = _builder_alloc_scalar("a_desc_base_lo", "uint32")
            b_desc_base_lo = _builder_alloc_scalar("b_desc_base_lo", "uint32")
            dispatch_token_iter = _builder_alloc_scalar("dispatch_token_iter", "int32")
            dispatch_token_topk_idx = _builder_alloc_scalar("dispatch_token_topk_idx", "int32")
            dispatch_expert_idx = _builder_alloc_scalar("dispatch_expert_idx", "int32")
            IRBuilder.name("dispatch_expert_idx", dispatch_expert_idx.buffer)
            dispatch_dst_rank_idx = _builder_alloc_scalar("dispatch_dst_rank_idx", "int32")
            dispatch_dst_local_expert_idx = _builder_alloc_scalar(
                "dispatch_dst_local_expert_idx", "int32"
            )
            dispatch_dst_slot_idx = _builder_alloc_scalar("dispatch_dst_slot_idx", "int32")
            pull_local_expert_idx = _builder_alloc_scalar("pull_local_expert_idx", "int32")
            pull_num_tokens = _builder_alloc_scalar("pull_num_tokens", "int32")
            pull_pool_block_offset = _builder_alloc_scalar("pull_pool_block_offset", "int32")
            pull_src_token_topk_idx = _builder_alloc_scalar("pull_src_token_topk_idx", "int32")
            pull_src_token_idx = _builder_alloc_scalar("pull_src_token_idx", "int32")
            pull_src_topk_idx = _builder_alloc_scalar("pull_src_topk_idx", "int32")
            pull_pool_token_idx = _builder_alloc_scalar("pull_pool_token_idx", "int32")
            pull_pool_block_idx = _builder_alloc_scalar("pull_pool_block_idx", "int32")
            pull_ring_block_idx = _builder_alloc_scalar("pull_ring_block_idx", "int32")
            pull_ring_token_idx = _builder_alloc_scalar("pull_ring_token_idx", "int32")
            token_idx_in_block = _builder_alloc_scalar("token_idx_in_block", "int32")
            l1_empty_count_target = _builder_alloc_scalar("l1_empty_count_target", "int32")
            pull_mbarrier_phase = _builder_alloc_scalar("pull_mbarrier_phase", "int32")
            epilogue_value = _builder_alloc_scalar("epilogue_value", "float32")
            combine_accum = _builder_alloc_scalar("combine_accum", "float32")
            gate_accum = _builder_alloc_scalar("gate_accum", "float32")
            up_accum = _builder_alloc_scalar("up_accum", "float32")
            current_ring_count = _builder_alloc_scalar("current_ring_count", "uint32")
            barrier_status_printf = _builder_alloc_scalar("barrier_status_printf", "int32")
            # atom returns the pre-op value; these sites only want the side effect.
            # Kept as atom (not red) so the emitted PTX is unchanged.
            atom_prev_unused = _builder_alloc_scalar("atom_prev_unused", "uint32")
            expected_ring_count = _builder_alloc_scalar("expected_ring_count", "int32")
            scheduler_num_m_blocks = _builder_alloc_scalar("scheduler_num_m_blocks", "int32")
            scheduler_cached_status = _builder_alloc_scalar("scheduler_cached_status", "uint64")
            ordinary_global_u64 = _builder_alloc_scalar("ordinary_global_u64", "uint64")
            ordinary_global_u32 = _builder_alloc_scalar("ordinary_global_u32", "uint32")
            ordinary_global_s64 = _builder_alloc_scalar("ordinary_global_s64", "int64")
            symm_rank_base = _builder_alloc_scalar("symm_rank_base", "uint64")
            nvl_counter_value = _builder_alloc_scalar("nvl_counter_value", "uint32")
            smem_expert_count_value = _builder_alloc_scalar("smem_expert_count_value", "uint32")
            tmem_allocated = _builder_alloc_scalar("tmem_allocated", "uint32")
            dst_rank_idx_u32 = _builder_alloc_scalar("dst_rank_idx_u32", "uint32")
            dst_token_idx_u32 = _builder_alloc_scalar("dst_token_idx_u32", "uint32")
            dst_topk_idx_u32 = _builder_alloc_scalar("dst_topk_idx_u32", "uint32")
            sched_stage_idx = _builder_alloc_scalar("sched_stage_idx", "int32")
            sched_phase = _builder_alloc_scalar("sched_phase", "int32")
            sched_num_total_m_blocks = _builder_alloc_scalar("sched_num_total_m_blocks", "int32")
            sched_num_l1_waves = _builder_alloc_scalar("sched_num_l1_waves", "uint32")
            sched_task_idx = _builder_alloc_scalar("sched_task_idx", "uint32")
            sched_task_valid = _builder_alloc_scalar("sched_task_valid", "int32")
            sched_block_offset = _builder_alloc_scalar("sched_block_offset", "uint32")
            sched_expert_num_m_blocks = _builder_alloc_scalar("sched_expert_num_m_blocks", "uint32")
            sched_inclusive_sum = _builder_alloc_scalar("sched_inclusive_sum", "uint32")
            sched_lane_pool_block_offset = _builder_alloc_scalar(
                "sched_lane_pool_block_offset", "uint32"
            )
            sched_owner_mask = _builder_alloc_scalar("sched_owner_mask", "uint32")
            sched_owner_m_block_idx = _builder_alloc_scalar("sched_owner_m_block_idx", "uint32")
            sched_owner_valid_m = _builder_alloc_scalar("sched_owner_valid_m", "uint32")
            sched_required_l1_tasks = _builder_alloc_scalar("sched_required_l1_tasks", "uint32")
            current_expert_idx = _builder_alloc_scalar("current_expert_idx", "int32")
            old_expert_idx = _builder_alloc_scalar("old_expert_idx", "int32")
            expert_start_idx = _builder_alloc_scalar("expert_start_idx", "int32")
            expert_end_idx = _builder_alloc_scalar("expert_end_idx", "int32")
            token_idx_in_rank = _builder_alloc_scalar("token_idx_in_rank", "int32")
            token_idx_in_expert = _builder_alloc_scalar("token_idx_in_expert", "int32")
            current_rank_in_expert_idx = _builder_alloc_scalar(
                "current_rank_in_expert_idx", "int32"
            )
            rank_count_mask = _builder_alloc_scalar("rank_count_mask", "uint32")
            num_active_ranks = _builder_alloc_scalar("num_active_ranks", "int32")
            active_lane_count = _builder_alloc_scalar("active_lane_count", "int32")
            num_actives_in_lane = _builder_alloc_scalar("num_actives_in_lane", "int32")
            min_in_lane = _builder_alloc_scalar("min_in_lane", "uint32")
            min_active_count = _builder_alloc_scalar("min_active_count", "int32")
            round_token_count = _builder_alloc_scalar("round_token_count", "int32")
            slot_idx_in_round = _builder_alloc_scalar("slot_idx_in_round", "int32")
            round_offset = _builder_alloc_scalar("round_offset", "int32")
            barrier_status = _builder_alloc_scalar("barrier_status", "int32")
            barrier_signal_phase = _builder_alloc_scalar("barrier_signal_phase", "int32")
            barrier_signal_sign = _builder_alloc_scalar("barrier_signal_sign", "int32")
            barrier_target = _builder_alloc_scalar("barrier_target", "int32")
            epilogue_thread_idx = _builder_alloc_scalar("epilogue_thread_idx", "int32")
            sf_row_idx = _builder_alloc_scalar("sf_row_idx", "int32")
            epilogue_wg_idx = _builder_alloc_scalar("epilogue_wg_idx", "int32")
            grid_sync_old_value = _builder_alloc_scalar("grid_sync_old_value", "uint32")
            grid_sync_new_value = _builder_alloc_scalar("grid_sync_new_value", "uint32")
            nvl_barrier_start_clock = _builder_alloc_scalar("nvl_barrier_start_clock", "uint64")
            pipeline_stage_idx = _builder_alloc_scalar("pipeline_stage_idx", "int32")
            pipeline_phase = _builder_alloc_scalar("pipeline_phase", "int32")
            k_block_idx = _builder_alloc_scalar("k_block_idx", "int32")
            k_idx = _builder_alloc_scalar("k_idx", "int32")
            k_idx_packed = _builder_alloc_scalar("k_idx_packed", "int32")
            m_idx = _builder_alloc_scalar("m_idx", "int32")
            n_idx = _builder_alloc_scalar("n_idx", "int32")
            pool_block_idx = _builder_alloc_scalar("pool_block_idx", "int32")
            ring_block_idx = _builder_alloc_scalar("ring_block_idx", "int32")
            ring_m_idx = _builder_alloc_scalar("ring_m_idx", "int32")
            pool_m_idx = _builder_alloc_scalar("pool_m_idx", "int32")
            valid_m = _builder_alloc_scalar("valid_m", "int32")
            sfa_m_idx = _builder_alloc_scalar("sfa_m_idx", "int32")
            stored_num_tokens_per_expert = _builder_assign(
                "stored_num_tokens_per_expert",
                T.alloc_local((num_experts_per_lane,), "uint32"),
                locals().get("stored_num_tokens_per_expert", _BUILDER_MISSING),
            )
            selected_num_tokens = _builder_assign(
                "selected_num_tokens",
                T.alloc_local((1,), "int32"),
                locals().get("selected_num_tokens", _BUILDER_MISSING),
            )
            pool_block_offset_sum = _builder_assign(
                "pool_block_offset_sum",
                T.alloc_local((1,), "int32"),
                locals().get("pool_block_offset_sum", _BUILDER_MISSING),
            )
            task_info_regs = _builder_assign(
                "task_info_regs",
                T.alloc_local((8,), "uint32"),
                locals().get("task_info_regs", _BUILDER_MISSING),
            )
            sched_inclusive_vals = _builder_assign(
                "sched_inclusive_vals",
                T.alloc_local((1,), "uint32"),
                locals().get("sched_inclusive_vals", _BUILDER_MISSING),
            )
            stored_rank_counts = _builder_assign(
                "stored_rank_counts",
                T.alloc_local((num_ranks_per_lane,), "uint32"),
                locals().get("stored_rank_counts", _BUILDER_MISSING),
            )
            remaining_rank_counts = _builder_assign(
                "remaining_rank_counts",
                T.alloc_local((num_ranks_per_lane,), "uint32"),
                locals().get("remaining_rank_counts", _BUILDER_MISSING),
            )
            combine_phase_local = _builder_alloc_scalar("combine_phase_local", "int32")
            combine_load_stage_idx = _builder_alloc_scalar("combine_load_stage_idx", "int32")
            combine_total_mask = _builder_alloc_scalar("combine_total_mask", "uint32")
            combine_slot_mask = _builder_alloc_scalar("combine_slot_mask", "uint32")
            combine_slot_idx = _builder_alloc_scalar("combine_slot_idx", "int32")
            combine_chunk_offset_elems = _builder_alloc_scalar(
                "combine_chunk_offset_elems", "int32"
            )
            combine_do_reduce = _builder_alloc_scalar("combine_do_reduce", "int32")
            combine_next_do_reduce = _builder_alloc_scalar("combine_next_do_reduce", "int32")

            def workspace_grid_sync(
                counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
            ):
                _builder_emit(
                    T.ptx.barrier.sync(T.uint32(sync_barrier_idx), T.uint32(sync_num_threads))
                )
                with T.If(sync_thread_idx == 0):
                    with T.Then():
                        with T.If(sm_idx == 0):
                            with T.Then():
                                _builder_emit(
                                    atomic_add_rel_u32(
                                        grid_sync_old_value,
                                        workspace_grid_sync_count.ptr_to([counter_idx]),
                                        T.uint32(0x80000000 - (kernel_config.num_sms - 1)),
                                    )
                                )
                            with T.Else():
                                _builder_emit(
                                    atomic_add_rel_u32(
                                        grid_sync_old_value,
                                        workspace_grid_sync_count.ptr_to([counter_idx]),
                                        T.uint32(1),
                                    )
                                )
                        _builder_emit(
                            load_acq_u32(
                                grid_sync_new_value, workspace_grid_sync_count.ptr_to([counter_idx])
                            )
                        )
                        with T.While(
                            grid_sync_done_u32(grid_sync_new_value, grid_sync_old_value)
                            == T.uint32(0)
                        ):
                            _builder_emit(
                                load_acq_u32(
                                    grid_sync_new_value,
                                    workspace_grid_sync_count.ptr_to([counter_idx]),
                                )
                            )
                _builder_emit(
                    T.ptx.barrier.sync(T.uint32(sync_barrier_idx), T.uint32(sync_num_threads))
                )

            def nvlink_barrier(
                counter_idx,
                barrier_tag,
                sync_num_threads,
                sync_barrier_idx,
                sync_thread_idx,
                sync_prologue,
                sync_epilogue,
            ):
                nonlocal barrier_signal_phase, barrier_signal_sign, barrier_status
                nonlocal barrier_target, nvl_barrier_start_clock, symm_rank_base
                if num_processes == 1:
                    _builder_emit(
                        workspace_grid_sync(
                            counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                        )
                    )
                else:
                    if sync_prologue:
                        _builder_emit(
                            workspace_grid_sync(
                                counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                            )
                        )
                    with T.If(sm_idx == 0):
                        with T.Then():
                            _builder_emit(
                                load_global_u32(
                                    nvl_counter_value, workspace_nvl_barrier_counter.ptr_to([0])
                                )
                            )
                            barrier_status = _builder_assign(
                                "barrier_status",
                                T.cast(T.bitwise_and(nvl_counter_value, T.uint32(3)), "int32"),
                                locals().get("barrier_status", _BUILDER_MISSING),
                            )
                            barrier_signal_phase = _builder_assign(
                                "barrier_signal_phase",
                                T.bitwise_and(barrier_status, T.int32(1)),
                                locals().get("barrier_signal_phase", _BUILDER_MISSING),
                            )
                            barrier_signal_sign = _builder_assign(
                                "barrier_signal_sign",
                                T.shift_right(barrier_status, T.int32(1)),
                                locals().get("barrier_signal_sign", _BUILDER_MISSING),
                            )
                            with T.If(sync_thread_idx < T.int32(num_processes)):
                                with T.Then():
                                    barrier_target = _builder_assign(
                                        "barrier_target",
                                        T.int32(1),
                                        locals().get("barrier_target", _BUILDER_MISSING),
                                    )
                                    with T.If(barrier_signal_sign != 0):
                                        with T.Then():
                                            barrier_target = _builder_assign(
                                                "barrier_target",
                                                T.int32(-1),
                                                locals().get("barrier_target", _BUILDER_MISSING),
                                            )
                                    symm_rank_base = _builder_assign(
                                        "symm_rank_base",
                                        sym_buffer_base + T.cast(symm_rank_offsets[0], "uint64"),
                                        locals().get("symm_rank_base", _BUILDER_MISSING),
                                    )
                                    _builder_emit(
                                        load_symm_rank_base(
                                            symm_rank_base, smem_symm_rank_bases, sync_thread_idx
                                        )
                                    )
                                    _builder_emit(
                                        peer_red_add_rel_sys_s32(
                                            symm_rank_base,
                                            T.uint64(workspace_layout.barrier_offset + 20)
                                            + T.cast(barrier_signal_phase * 4, "uint64"),
                                            barrier_target,
                                        )
                                    )
                            _builder_emit(
                                T.ptx.bar.sync(
                                    T.uint32(sync_barrier_idx), T.uint32(sync_num_threads)
                                )
                            )
                            with T.If(sync_thread_idx == 0):
                                with T.Then():
                                    _builder_emit(
                                        red_add_gpu_u32(
                                            workspace_nvl_barrier_counter.ptr_to([0]), T.uint32(1)
                                        )
                                    )
                                    barrier_target = _builder_assign(
                                        "barrier_target",
                                        T.int32(num_processes),
                                        locals().get("barrier_target", _BUILDER_MISSING),
                                    )
                                    with T.If(barrier_signal_sign != 0):
                                        with T.Then():
                                            barrier_target = _builder_assign(
                                                "barrier_target",
                                                T.int32(0),
                                                locals().get("barrier_target", _BUILDER_MISSING),
                                            )
                                    _builder_emit(
                                        load_acq_sys_s32(
                                            barrier_status,
                                            workspace_nvl_barrier_signal.ptr_to(
                                                [barrier_signal_phase]
                                            ),
                                        )
                                    )
                                    nvl_barrier_start_clock = _builder_assign(
                                        "nvl_barrier_start_clock",
                                        cuda_clock64(),
                                        locals().get("nvl_barrier_start_clock", _BUILDER_MISSING),
                                    )
                                    with T.While(barrier_status != barrier_target):
                                        with T.If(
                                            cuda_clock64() - nvl_barrier_start_clock
                                            >= T.uint64(num_nvlink_barrier_timeout_cycles)
                                        ):
                                            with T.Then():
                                                if kernel_emit_nvl_barrier_timeout_printf:
                                                    _builder_emit(
                                                        load_acq_sys_s32(
                                                            barrier_status_printf,
                                                            workspace_nvl_barrier_signal.ptr_to(
                                                                [barrier_signal_phase]
                                                            ),
                                                        )
                                                    )
                                                    _builder_emit(
                                                        load_global_u32(
                                                            nvl_counter_value,
                                                            workspace_nvl_barrier_counter.ptr_to(
                                                                [0]
                                                            ),
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.cuda.printf(
                                                            f"DeepGEMM NVLink barrier timeout "
                                                            f"({nvlink_barrier_timeout_seconds}s): "
                                                            "rank=%d, counter=%d, signal=%d, target=%d, "
                                                            "phase=%d, sign=%d, tag=%d\n",
                                                            rank_idx,
                                                            T.cast(nvl_counter_value, "int32"),
                                                            barrier_status_printf,
                                                            barrier_target,
                                                            barrier_signal_phase,
                                                            barrier_signal_sign,
                                                            barrier_tag,
                                                        )
                                                    )
                                                _builder_emit(T.cuda.trap_when_assert_failed(False))
                                        _builder_emit(
                                            load_acq_sys_s32(
                                                barrier_status,
                                                workspace_nvl_barrier_signal.ptr_to(
                                                    [barrier_signal_phase]
                                                ),
                                            )
                                        )
                    if sync_epilogue:
                        _builder_emit(
                            workspace_grid_sync(
                                counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                            )
                        )

            def dispatch_nvlink_barrier_before_pull(thread_idx_in_scope):
                _builder_emit(
                    nvlink_barrier(
                        dispatch_grid_sync_index,
                        before_dispatch_pull_barrier_tag,
                        kernel_config.num_dispatch_threads,
                        dispatch_sync_barrier_idx,
                        thread_idx_in_scope,
                        0,
                        1,
                    )
                )

            def dispatch_nvlink_barrier_after_workspace_clean(thread_idx_in_scope):
                _builder_emit(
                    nvlink_barrier(
                        dispatch_grid_sync_index,
                        after_workspace_clean_barrier_tag,
                        kernel_config.num_dispatch_threads,
                        dispatch_sync_barrier_idx,
                        thread_idx_in_scope,
                        1,
                        0,
                    )
                )

            def epilogue_nvlink_barrier_before_combine_reduce(thread_idx_in_scope):
                _builder_emit(
                    nvlink_barrier(
                        epilogue_grid_sync_index,
                        before_combine_reduce_barrier_tag,
                        kernel_config.num_epilogue_threads,
                        epilogue_full_sync_barrier_idx,
                        thread_idx_in_scope,
                        1,
                        1,
                    )
                )

            def scheduler_fetch_expert_recv_count():
                nonlocal dispatch_expert_idx, sched_num_total_m_blocks
                nonlocal sched_num_l1_waves, scheduler_cached_status
                with T.serial(0, num_experts_per_lane) as expert_lane_idx:
                    IRBuilder.name("expert_lane_idx", expert_lane_idx)
                    dispatch_expert_idx = _builder_assign(
                        "dispatch_expert_idx",
                        expert_lane_idx * 32 + lane_idx,
                        locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                    )
                    scheduler_cached_status = _builder_assign(
                        "scheduler_cached_status",
                        T.uint64(0),
                        locals().get("scheduler_cached_status", _BUILDER_MISSING),
                    )
                    with T.If(dispatch_expert_idx < num_experts_per_rank):
                        with T.Then():
                            with T.While(
                                T.cast(T.shift_right(scheduler_cached_status, 32), "int32")
                                != kernel_config.num_sms * num_processes
                            ):
                                _builder_emit(
                                    load_volatile_u64(
                                        scheduler_cached_status,
                                        workspace_expert_recv_count_sum.ptr_to(
                                            [dispatch_expert_idx]
                                        ),
                                    )
                                )
                    T.buffer_store(
                        stored_num_tokens_per_expert,
                        T.cast(
                            T.bitwise_and(scheduler_cached_status, T.uint64(0xFFFFFFFF)), "uint32"
                        ),
                        [expert_lane_idx],
                    )
                _builder_emit(T.cuda.warp_sync())
                # `num_total_m_blocks = get_num_total_pool_blocks()` plus the L1 warmup
                # wave seed (scheduler/mega_moe.cuh `fetch_expert_recv_count`).
                _builder_emit(
                    scheduler_get_pool_block_offset(
                        T.int32(num_experts_per_rank),
                        lane_idx,
                        stored_num_tokens_per_expert,
                        pool_block_offset_sum,
                    )
                )
                sched_num_total_m_blocks = _builder_assign(
                    "sched_num_total_m_blocks",
                    pool_block_offset_sum[0],
                    locals().get("sched_num_total_m_blocks", _BUILDER_MISSING),
                )
                sched_num_total_m_blocks_u32 = _builder_assign(
                    "sched_num_total_m_blocks_u32",
                    T.cast(sched_num_total_m_blocks, "uint32"),
                    locals().get("sched_num_total_m_blocks_u32", _BUILDER_MISSING),
                )
                sched_num_total_l1_tasks = _builder_assign(
                    "sched_num_total_l1_tasks",
                    sched_num_total_m_blocks_u32 * T.uint32(num_l1_clusters),
                    locals().get("sched_num_total_l1_tasks", _BUILDER_MISSING),
                )
                sched_num_total_l1_waves = _builder_assign(
                    "sched_num_total_l1_waves",
                    (sched_num_total_l1_tasks + T.uint32(num_sched_clusters - 1))
                    // T.uint32(num_sched_clusters),
                    locals().get("sched_num_total_l1_waves", _BUILDER_MISSING),
                )
                sched_warmup_interleave = _builder_assign(
                    "sched_warmup_interleave",
                    (
                        T.uint32(num_l1_clusters)
                        + (sched_num_total_m_blocks_u32 - T.uint32(1))
                        * T.uint32(sched_interleave_cluster_diff)
                        + T.uint32(num_sched_clusters - 1)
                    )
                    // T.uint32(num_sched_clusters)
                    + T.uint32(1),
                    locals().get("sched_warmup_interleave", _BUILDER_MISSING),
                )
                sched_min_l1_warmup_waves = _builder_assign(
                    "sched_min_l1_warmup_waves",
                    T.max(T.uint32(sched_l1_warmup_first_l2_wave), sched_warmup_interleave),
                    locals().get("sched_min_l1_warmup_waves", _BUILDER_MISSING),
                )
                sched_num_l1_waves = _builder_assign(
                    "sched_num_l1_waves",
                    T.min(sched_min_l1_warmup_waves, sched_num_total_l1_waves),
                    locals().get("sched_num_l1_waves", _BUILDER_MISSING),
                )

            def sched_advance_pipeline():
                T.buffer_store(sched_stage_idx.buffer, sched_stage_idx ^ T.int32(1), [0])
                with T.If(sched_stage_idx == T.int32(0)):
                    with T.Then():
                        T.buffer_store(sched_phase.buffer, sched_phase ^ T.int32(1), [0])

            def consumer_get_next_task():
                # `get_next_task`: wait for the published TaskInfo, copy it into
                # registers (2x LDS.128 of the alignas(16) 32-byte struct), advance.
                _builder_emit(
                    barrier_wait(
                        smem_barriers.ptr_to([task_info_full_barrier_base + sched_stage_idx]),
                        sched_phase,
                    )
                )
                _builder_emit(
                    lds128(smem_task_infos.ptr_to([sched_stage_idx, 0]), task_info_regs, 0)
                )
                _builder_emit(
                    lds128(smem_task_infos.ptr_to([sched_stage_idx, 4]), task_info_regs, 4)
                )
                _builder_emit(sched_advance_pipeline())

            def consumer_bind_task_args():
                nonlocal block_phase, local_expert_idx, m_block_idx, n_cluster_idx
                nonlocal pool_block_idx, valid_m, shape_n, shape_k
                block_phase = _builder_assign(
                    "block_phase",
                    T.cast(task_info_regs[0], "int32"),
                    locals().get("block_phase", _BUILDER_MISSING),
                )
                local_expert_idx = _builder_assign(
                    "local_expert_idx",
                    T.cast(task_info_regs[1], "int32"),
                    locals().get("local_expert_idx", _BUILDER_MISSING),
                )
                m_block_idx = _builder_assign(
                    "m_block_idx",
                    T.cast(task_info_regs[2], "int32"),
                    locals().get("m_block_idx", _BUILDER_MISSING),
                )
                n_cluster_idx = _builder_assign(
                    "n_cluster_idx",
                    T.cast(task_info_regs[3], "int32"),
                    locals().get("n_cluster_idx", _BUILDER_MISSING),
                )
                pool_block_idx = _builder_assign(
                    "pool_block_idx",
                    T.cast(task_info_regs[4], "int32"),
                    locals().get("pool_block_idx", _BUILDER_MISSING),
                )
                valid_m = _builder_assign(
                    "valid_m",
                    T.cast(task_info_regs[5], "int32"),
                    locals().get("valid_m", _BUILDER_MISSING),
                )
                shape_n = _builder_assign(
                    "shape_n",
                    T.cast(task_info_regs[6], "int32"),
                    locals().get("shape_n", _BUILDER_MISSING),
                )
                shape_k = _builder_assign(
                    "shape_k",
                    T.cast(task_info_regs[7], "int32"),
                    locals().get("shape_k", _BUILDER_MISSING),
                )

            def scheduler_release_task_info():
                # `release_task_info`: all epilogue threads (both CTAs) arrive at the
                # leader CTA's empty barrier of the just-consumed stage.
                _rem1 = _builder_assign(
                    "_rem1", T.alloc_local([1], "uint64"), locals().get("_rem1", _BUILDER_MISSING)
                )
                _builder_emit(
                    T.ptx.mapa.shared__cluster.u64(
                        _rem1[0],
                        smem_barriers.ptr_to(
                            [task_info_empty_barrier_base + (sched_stage_idx ^ T.int32(1))]
                        ),
                        T.uint32(0),
                    )
                )
                _builder_emit(T.ptx.mbarrier.arrive.b64(_rem1[0], T.uint32(1), pred=T.bool(True)))

            def producer_create_task(
                task_block_phase, task_num_clusters, task_shape_n, task_shape_k
            ):
                nonlocal sched_block_offset, sched_expert_num_m_blocks
                nonlocal sched_inclusive_sum, sched_lane_pool_block_offset
                nonlocal sched_owner_m_block_idx, sched_owner_mask, sched_owner_valid_m
                # `create_task`: resolve the owning expert / m-block / valid_m of the
                # pool block via a per-lane token-count scan + warp ballot.
                T.buffer_store(task_info_regs, T.uint32(task_block_phase), [0])
                T.buffer_store(task_info_regs, T.uint32(0), [1])
                T.buffer_store(task_info_regs, T.uint32(0), [2])
                T.buffer_store(task_info_regs, sched_task_idx % T.uint32(task_num_clusters), [3])
                T.buffer_store(task_info_regs, sched_task_idx // T.uint32(task_num_clusters), [4])
                T.buffer_store(task_info_regs, T.uint32(0), [5])
                T.buffer_store(task_info_regs, T.uint32(task_shape_n), [6])
                T.buffer_store(task_info_regs, T.uint32(task_shape_k), [7])
                sched_block_offset = _builder_assign(
                    "sched_block_offset",
                    T.uint32(0),
                    locals().get("sched_block_offset", _BUILDER_MISSING),
                )
                with T.unroll(0, num_experts_per_lane) as expert_lane_idx:
                    IRBuilder.name("expert_lane_idx", expert_lane_idx)
                    sched_expert_num_m_blocks = _builder_assign(
                        "sched_expert_num_m_blocks",
                        (
                            stored_num_tokens_per_expert[expert_lane_idx]
                            + T.uint32(kernel_config.block_m - 1)
                        )
                        // T.uint32(kernel_config.block_m),
                        locals().get("sched_expert_num_m_blocks", _BUILDER_MISSING),
                    )
                    # `math::warp_inclusive_sum`
                    T.buffer_store(sched_inclusive_vals, sched_expert_num_m_blocks, [0])
                    with T.unroll(0, 5) as shuffle_offset:
                        IRBuilder.name("shuffle_offset", shuffle_offset)
                        sched_inclusive_sum = _builder_assign(
                            "sched_inclusive_sum",
                            T.tvm_warp_shuffle_up(
                                T.uint32(0xFFFFFFFF),
                                sched_inclusive_vals[0],
                                1 << shuffle_offset,
                                32,
                                32,
                            ),
                            locals().get("sched_inclusive_sum", _BUILDER_MISSING),
                        )
                        with T.If(lane_idx >= (1 << shuffle_offset)):
                            with T.Then():
                                T.buffer_store(
                                    sched_inclusive_vals,
                                    sched_inclusive_vals[0] + sched_inclusive_sum,
                                    [0],
                                )
                    sched_lane_pool_block_offset = _builder_assign(
                        "sched_lane_pool_block_offset",
                        (sched_block_offset + sched_inclusive_vals[0] - sched_expert_num_m_blocks),
                        locals().get("sched_lane_pool_block_offset", _BUILDER_MISSING),
                    )
                    sched_owner_mask = _builder_assign(
                        "sched_owner_mask",
                        ballot_sync(
                            T.uint32(0xFFFFFFFF),
                            (
                                T.cast(expert_lane_idx * 32 + lane_idx, "uint32")
                                < T.uint32(num_experts_per_rank)
                            )
                            & (task_info_regs[4] >= sched_lane_pool_block_offset)
                            & (
                                task_info_regs[4]
                                < sched_lane_pool_block_offset + sched_expert_num_m_blocks
                            ),
                        ),
                        locals().get("sched_owner_mask", _BUILDER_MISSING),
                    )
                    with T.If(sched_owner_mask != T.uint32(0)):
                        with T.Then():
                            sched_owner_lane_idx = _builder_assign(
                                "sched_owner_lane_idx",
                                ffs_u32(sched_owner_mask) - T.int32(1),
                                locals().get("sched_owner_lane_idx", _BUILDER_MISSING),
                            )
                            sched_owner_m_block_idx = _builder_assign(
                                "sched_owner_m_block_idx",
                                task_info_regs[4] - sched_lane_pool_block_offset,
                                locals().get("sched_owner_m_block_idx", _BUILDER_MISSING),
                            )
                            sched_owner_valid_m = _builder_assign(
                                "sched_owner_valid_m",
                                T.min(
                                    stored_num_tokens_per_expert[expert_lane_idx]
                                    - sched_owner_m_block_idx * T.uint32(kernel_config.block_m),
                                    T.uint32(kernel_config.block_m),
                                ),
                                locals().get("sched_owner_valid_m", _BUILDER_MISSING),
                            )
                            T.buffer_store(
                                task_info_regs,
                                T.tvm_warp_shuffle(
                                    T.uint32(0xFFFFFFFF),
                                    T.cast(expert_lane_idx * 32 + lane_idx, "uint32"),
                                    sched_owner_lane_idx,
                                    32,
                                    32,
                                ),
                                [1],
                            )
                            T.buffer_store(
                                task_info_regs,
                                T.tvm_warp_shuffle(
                                    T.uint32(0xFFFFFFFF),
                                    sched_owner_m_block_idx,
                                    sched_owner_lane_idx,
                                    32,
                                    32,
                                ),
                                [2],
                            )
                            T.buffer_store(
                                task_info_regs,
                                T.tvm_warp_shuffle(
                                    T.uint32(0xFFFFFFFF),
                                    sched_owner_valid_m,
                                    sched_owner_lane_idx,
                                    32,
                                    32,
                                ),
                                [5],
                            )
                    sched_block_offset = _builder_assign(
                        "sched_block_offset",
                        sched_block_offset
                        + T.tvm_warp_shuffle(
                            T.uint32(0xFFFFFFFF), sched_inclusive_vals[0], T.int32(31), 32, 32
                        ),
                        locals().get("sched_block_offset", _BUILDER_MISSING),
                    )

            def producer_get_next_task():
                nonlocal sched_num_l1_waves, sched_required_l1_tasks
                nonlocal sched_task_idx, sched_task_valid
                # Producer-side `get_next_task`: interleave L1/L2 task pulls from the
                # global atomic counters with the L1 warmup-wave ordering.
                T.buffer_store(task_info_regs, T.uint32(0), [0])
                sched_task_valid = _builder_assign(
                    "sched_task_valid",
                    T.int32(0),
                    locals().get("sched_task_valid", _BUILDER_MISSING),
                )
                with T.While(sched_task_valid == T.int32(0)):
                    with T.If(
                        T.And(
                            sched_num_l1_waves != T.uint32(sched_l1_waves_done),
                            sched_num_l1_waves != T.uint32(0),
                        )
                    ):
                        with T.Then():
                            sched_num_l1_waves = _builder_assign(
                                "sched_num_l1_waves",
                                sched_num_l1_waves - T.uint32(1),
                                locals().get("sched_num_l1_waves", _BUILDER_MISSING),
                            )
                            sched_task_idx = _builder_assign(
                                "sched_task_idx",
                                T.uint32(0),
                                locals().get("sched_task_idx", _BUILDER_MISSING),
                            )
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    _builder_emit(
                                        atomic_add_u32(
                                            sched_task_idx,
                                            workspace_l1_task_count.ptr_to([0]),
                                            T.uint32(1),
                                        )
                                    )
                            sched_task_idx = _builder_assign(
                                "sched_task_idx",
                                T.tvm_warp_shuffle(
                                    T.uint32(0xFFFFFFFF), sched_task_idx, T.int32(0), 32, 32
                                ),
                                locals().get("sched_task_idx", _BUILDER_MISSING),
                            )
                            with T.If(
                                sched_task_idx
                                >= T.cast(sched_num_total_m_blocks, "uint32")
                                * T.uint32(num_l1_clusters)
                            ):
                                with T.Then():
                                    sched_num_l1_waves = _builder_assign(
                                        "sched_num_l1_waves",
                                        T.uint32(sched_l1_waves_done),
                                        locals().get("sched_num_l1_waves", _BUILDER_MISSING),
                                    )
                                with T.Else():
                                    _builder_emit(
                                        producer_create_task(
                                            1, num_l1_clusters, intermediate_hidden * 2, hidden
                                        )
                                    )
                                    sched_task_valid = _builder_assign(
                                        "sched_task_valid",
                                        T.int32(1),
                                        locals().get("sched_task_valid", _BUILDER_MISSING),
                                    )
                        with T.Else():
                            sched_task_idx = _builder_assign(
                                "sched_task_idx",
                                T.uint32(0),
                                locals().get("sched_task_idx", _BUILDER_MISSING),
                            )
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    _builder_emit(
                                        atomic_add_u32(
                                            sched_task_idx,
                                            workspace_l2_task_count.ptr_to([0]),
                                            T.uint32(1),
                                        )
                                    )
                            sched_task_idx = _builder_assign(
                                "sched_task_idx",
                                T.tvm_warp_shuffle(
                                    T.uint32(0xFFFFFFFF), sched_task_idx, T.int32(0), 32, 32
                                ),
                                locals().get("sched_task_idx", _BUILDER_MISSING),
                            )
                            with T.If(
                                sched_task_idx
                                >= T.cast(sched_num_total_m_blocks, "uint32")
                                * T.uint32(num_l2_clusters)
                            ):
                                with T.Then():
                                    T.evaluate(T.break_loop())
                            with T.If(sched_num_l1_waves != T.uint32(sched_l1_waves_done)):
                                with T.Then():
                                    sched_num_l1_waves = _builder_assign(
                                        "sched_num_l1_waves",
                                        T.uint32(1),
                                        locals().get("sched_num_l1_waves", _BUILDER_MISSING),
                                    )
                            _builder_emit(
                                producer_create_task(
                                    2, num_l2_clusters, hidden, intermediate_hidden
                                )
                            )
                            # Wait until all required L1 tasks are fetched
                            sched_required_l1_tasks = _builder_assign(
                                "sched_required_l1_tasks",
                                (task_info_regs[4] + T.uint32(1)) * T.uint32(num_l1_clusters),
                                locals().get("sched_required_l1_tasks", _BUILDER_MISSING),
                            )
                            sched_l1_count = _builder_assign(
                                "sched_l1_count",
                                T.uint32(0),
                                locals().get("sched_l1_count", _BUILDER_MISSING),
                            )
                            _builder_emit(
                                load_volatile_u32(
                                    sched_l1_count, workspace_l1_task_count.ptr_to([0])
                                )
                            )
                            with T.While(sched_l1_count < sched_required_l1_tasks):
                                _builder_emit(
                                    load_volatile_u32(
                                        sched_l1_count, workspace_l1_task_count.ptr_to([0])
                                    )
                                )
                            sched_task_valid = _builder_assign(
                                "sched_task_valid",
                                T.int32(1),
                                locals().get("sched_task_valid", _BUILDER_MISSING),
                            )

            def producer_publish_task():
                # `publish_task`: lanes 0/1 arrive-and-expect-tx at each CTA's full
                # barrier, then st.async the 32-byte TaskInfo into that CTA's smem.
                with T.If(lane_idx < T.int32(2)):
                    with T.Then():
                        _rem_ti = _builder_assign(
                            "_rem_ti",
                            T.alloc_local([1], "uint64"),
                            locals().get("_rem_ti", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            T.ptx.mapa.shared__cluster.u64(
                                _rem_ti[0],
                                smem_barriers.ptr_to(
                                    [task_info_full_barrier_base + sched_stage_idx]
                                ),
                                T.uint32(lane_idx),
                            )
                        )
                        _builder_emit(
                            T.ptx.mbarrier.arrive.expect_tx.release.cluster.b64(
                                _rem_ti[0], T.uint32(task_info_bytes), pred=T.bool(True)
                            )
                        )
                        _builder_emit(
                            T.evaluate(
                                st_async_cluster_task_info(
                                    smem_task_infos.ptr_to([sched_stage_idx, 0]),
                                    smem_barriers.ptr_to(
                                        [task_info_full_barrier_base + sched_stage_idx]
                                    ),
                                    lane_idx,
                                    task_info_regs,
                                )
                            )
                        )
                _builder_emit(T.cuda.warp_sync())
                _builder_emit(sched_advance_pipeline())

            def update_get_valid_m_true():
                nonlocal get_valid_m_true, get_valid_m_true_half, get_valid_m_true_eighth
                valid_m_u32 = _builder_assign(
                    "valid_m_u32",
                    T.cast(valid_m, "uint32"),
                    locals().get("valid_m_u32", _BUILDER_MISSING),
                )
                get_valid_m_true_u32 = _builder_assign(
                    "get_valid_m_true_u32",
                    (valid_m_u32 + T.uint32(15)) // T.uint32(16) * T.uint32(16),
                    locals().get("get_valid_m_true_u32", _BUILDER_MISSING),
                )
                get_valid_m_true = _builder_assign(
                    "get_valid_m_true",
                    T.cast(get_valid_m_true_u32, "int32"),
                    locals().get("get_valid_m_true", _BUILDER_MISSING),
                )
                get_valid_m_true_half = _builder_assign(
                    "get_valid_m_true_half",
                    T.cast(get_valid_m_true_u32 // T.uint32(2), "int32"),
                    locals().get("get_valid_m_true_half", _BUILDER_MISSING),
                )
                get_valid_m_true_eighth = _builder_assign(
                    "get_valid_m_true_eighth",
                    T.cast(get_valid_m_true_u32 // T.uint32(8), "int32"),
                    locals().get("get_valid_m_true_eighth", _BUILDER_MISSING),
                )

            def advance_pipeline():
                T.buffer_store(
                    pipeline_stage_idx.buffer,
                    T.if_then_else(
                        pipeline_stage_idx == T.int32(num_stages - 1),
                        T.int32(0),
                        pipeline_stage_idx + T.int32(1),
                    ),
                    [0],
                )
                with T.If(pipeline_stage_idx == 0):
                    with T.Then():
                        T.buffer_store(pipeline_phase.buffer, pipeline_phase ^ T.int32(1), [0])

            def barrier_wait(barrier_ptr, phase):
                _builder_emit(T.cuda.mbarrier_wait(barrier_ptr, phase))

            def tmem_empty_barrier_arrive_cta0(tmem_empty_barrier_ptr):
                _rem2 = _builder_assign(
                    "_rem2", T.alloc_local([1], "uint64"), locals().get("_rem2", _BUILDER_MISSING)
                )
                _builder_emit(
                    T.ptx.mapa.shared__cluster.u64(_rem2[0], tmem_empty_barrier_ptr, T.uint32(0))
                )
                _builder_emit(T.ptx.mbarrier.arrive.b64(_rem2[0], T.uint32(1), pred=T.bool(True)))

            def umma_arrive_multicast_2x1sm(barrier_ptr):
                with T.If(T.cuda.elect_sync()):
                    with T.Then():
                        _builder_emit(
                            T.ptx[
                                f"tcgen05.commit.cta_group::{kernel_config.num_ctas_per_cluster}"
                                ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                            ](barrier_ptr, T.uint16(tcgen05_cta_mask))
                        )

            def umma_arrive(barrier_ptr):
                _builder_emit(umma_arrive_multicast_2x1sm(barrier_ptr))

            def empty_barrier_arrive(do_tmem_full_arrive, empty_barrier_ptr, tmem_full_barrier_ptr):
                _builder_emit(umma_arrive(empty_barrier_ptr))
                with T.If(do_tmem_full_arrive):
                    with T.Then():
                        _builder_emit(umma_arrive(tmem_full_barrier_ptr))
                _builder_emit(T.cuda.warp_sync())

            def empty_barrier_arrive_current(do_tmem_full_arrive):
                _builder_emit(
                    empty_barrier_arrive(
                        do_tmem_full_arrive,
                        smem_barriers.ptr_to([empty_barrier_base + pipeline_stage_idx]),
                        smem_barriers.ptr_to([tmem_full_barrier_base + accum_stage_idx]),
                    )
                )

            def fence_view_async_tmem_load():
                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())

            def warpgroup_reg_dealloc(num_registers):
                _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(num_registers))

            def warpgroup_reg_alloc(num_registers):
                _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(num_registers))


            def tma_copy_2d_multicast_select(
                dst_ptr,
                barrier_ptr,
                tensor_map_l1_ptr,
                tensor_map_l2_ptr,
                block_phase_value,
                coord0,
                coord1,
            ):
                _builder_emit(
                    sm100_tma_2sm_load_2d_select(
                        dst_ptr,
                        barrier_ptr,
                        tensor_map_l1_ptr,
                        tensor_map_l2_ptr,
                        block_phase_value,
                        coord0,
                        coord1,
                    )
                )

            def full_barrier_arrive_and_expect_tx(full_barrier_ptr, transaction_bytes):
                _builder_emit(
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        full_barrier_ptr, T.uint32(transaction_bytes)
                    )
                )

            def full_barrier_arrive_cta0(full_barrier_ptr):
                _rem3 = _builder_assign(
                    "_rem3", T.alloc_local([1], "uint64"), locals().get("_rem3", _BUILDER_MISSING)
                )
                _builder_emit(
                    T.ptx.mapa.shared__cluster.u64(_rem3[0], full_barrier_ptr, T.uint32(0))
                )
                _builder_emit(T.ptx.mbarrier.arrive.b64(_rem3[0], T.uint32(1), pred=T.bool(True)))

            def make_instr_desc_block_scaled():
                _builder_emit(
                    T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                        T.address_of(desc_i),
                        d_dtype="float32",
                        a_dtype="float4_e2m1fn",
                        b_dtype="float8_e4m3fn",
                        sfa_dtype="float8_e8m0fnu",
                        sfb_dtype="float8_e8m0fnu",
                        sfa_tmem_addr=0,
                        sfb_tmem_addr=0,
                        M=umma_m,
                        N=umma_n,
                        K=umma_k,
                        trans_a=False,
                        trans_b=False,
                        n_cta_groups=kernel_config.num_ctas_per_cluster,
                    )
                )

            def make_sf_desc():
                _builder_emit(
                    T.cuda.tcgen05.encode_matrix_descriptor(
                        T.address_of(desc_sf),
                        smem_sfa.ptr_to([0, 0]),
                        ldo=0,
                        sdo=sf_desc_sdo,
                        swizzle=0,
                    )
                )

            def make_umma_desc_a():
                _builder_emit(
                    T.cuda.tcgen05.encode_matrix_descriptor(
                        T.address_of(desc_a),
                        smem_a_fp8.ptr_to([0, 0, 0]),
                        ldo=0,
                        sdo=a_desc_sdo,
                        swizzle=3,
                    )
                )

            def make_umma_desc_b():
                _builder_emit(
                    T.cuda.tcgen05.encode_matrix_descriptor(
                        T.address_of(desc_b),
                        smem_b.ptr_to([0, 0, 0]),
                        ldo=0,
                        sdo=b_desc_sdo,
                        swizzle=3,
                    )
                )

            def utccp_copy(tmem_addr, sf_desc):
                _builder_emit(
                    T.ptx[
                        f"tcgen05.cp.cta_group::{kernel_config.num_ctas_per_cluster}.32x128b.warpx4"
                    ](T.cast(tmem_addr, "uint32"), sf_desc)
                )

            def sm100_u8x4_stsm_t_copy(fp8x4_word, smem_ptr):
                _builder_emit(stmatrix_fp8x4_trans(smem_ptr, fp8x4_word))

            def sm90_u32x4_stsm_t_copy(packed_values_buf, smem_ptr):
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem_ptr,
                        packed_values_buf[0],
                        packed_values_buf[1],
                        packed_values_buf[2],
                        packed_values_buf[3],
                    )
                )

            def sm90_tma_store_2d_copy(src_ptr, tensor_map, coord0, coord1):
                _builder_emit(T.evaluate(tma_store_2d(src_ptr, tensor_map, coord0, coord1)))


            def store_token_src_metadata(pool_token_idx, src_rank_idx, src_token_idx, src_topk_idx):
                _builder_emit(
                    store_global_u32(
                        workspace_token_src_metadata.ptr_to([pool_token_idx, 0]),
                        T.cast(src_rank_idx, "uint32"),
                    )
                )
                _builder_emit(
                    store_global_u32(
                        workspace_token_src_metadata.ptr_to([pool_token_idx, 1]),
                        T.cast(src_token_idx, "uint32"),
                    )
                )
                _builder_emit(
                    store_global_u32(
                        workspace_token_src_metadata.ptr_to([pool_token_idx, 2]),
                        T.cast(src_topk_idx, "uint32"),
                    )
                )

            # Relaxed arrive — no prior memory effect needs to be released to peers
            # before TMEM alloc + mbarrier init below. Wait still .acquire (default).
            _builder_emit(T.ptx.barrier.cluster.arrive.relaxed.aligned())
            _builder_emit(T.ptx.barrier.cluster.wait.acquire.aligned())
            full_barrier_init_count = _builder_scalar("full_barrier_init_count", 2 * 2, "int32")
            tmem_empty_barrier_init_count = _builder_scalar(
                "tmem_empty_barrier_init_count", 2 * kernel_config.num_epilogue_threads, "int32"
            )
            is_reserved_non_epilogue_warp = _builder_assign(
                "is_reserved_non_epilogue_warp",
                (flat_warp_idx == kernel_config.reserved_non_epilogue_warp_idx),
                locals().get("is_reserved_non_epilogue_warp", _BUILDER_MISSING),
            )
            with T.If(flat_warp_idx == 0):
                with T.Then():
                    if num_processes > 1:
                        with T.If(lane_idx < num_processes):
                            with T.Then():
                                _builder_emit(
                                    st_shared_u64(
                                        smem_symm_rank_bases.ptr_to([lane_idx]),
                                        sym_buffer_base
                                        + T.cast(
                                            symm_rank_offset_arg_expr(symm_rank_offsets, lane_idx),
                                            "uint64",
                                        ),
                                    )
                                )
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    st_shared_bulk(
                                        smem_expert_count.ptr_to([0]), T.uint32(num_experts * 4)
                                    )
                                )
                            )
                with T.Else():
                    with T.If(flat_warp_idx == 1):
                        with T.Then():
                            dispatch_expert_idx = _builder_assign(
                                "dispatch_expert_idx",
                                lane_idx,
                                locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                            )
                            with T.While(dispatch_expert_idx < kernel_config.num_dispatch_warps):
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        smem_barriers.ptr_to(
                                            [dispatch_barrier_base + dispatch_expert_idx]
                                        ),
                                        T.uint32(1),
                                    )
                                )
                                dispatch_expert_idx = _builder_assign(
                                    "dispatch_expert_idx",
                                    dispatch_expert_idx + 32,
                                    locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                                )
                            _builder_emit(T.evaluate(fence_barrier_init()))
                        with T.Else():
                            with T.If(flat_warp_idx == 2):
                                with T.Then():
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            dispatch_expert_idx = _builder_assign(
                                                "dispatch_expert_idx",
                                                T.int32(0),
                                                locals().get(
                                                    "dispatch_expert_idx", _BUILDER_MISSING
                                                ),
                                            )
                                            with T.While(dispatch_expert_idx < num_stages):
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                full_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(full_barrier_init_count),
                                                    )
                                                )
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                empty_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(1),
                                                    )
                                                )
                                                dispatch_expert_idx = _builder_assign(
                                                    "dispatch_expert_idx",
                                                    dispatch_expert_idx + 1,
                                                    locals().get(
                                                        "dispatch_expert_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                            dispatch_expert_idx = _builder_assign(
                                                "dispatch_expert_idx",
                                                T.int32(0),
                                                locals().get(
                                                    "dispatch_expert_idx", _BUILDER_MISSING
                                                ),
                                            )
                                            with T.While(dispatch_expert_idx < num_epilogue_stages):
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                tmem_full_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(1),
                                                    )
                                                )
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                tmem_empty_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(tmem_empty_barrier_init_count),
                                                    )
                                                )
                                                dispatch_expert_idx = _builder_assign(
                                                    "dispatch_expert_idx",
                                                    dispatch_expert_idx + 1,
                                                    locals().get(
                                                        "dispatch_expert_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                            dispatch_expert_idx = _builder_assign(
                                                "dispatch_expert_idx",
                                                T.int32(0),
                                                locals().get(
                                                    "dispatch_expert_idx", _BUILDER_MISSING
                                                ),
                                            )
                                            with T.While(
                                                dispatch_expert_idx
                                                < kernel_config.num_epilogue_warps * 2
                                            ):
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                combine_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(1),
                                                    )
                                                )
                                                dispatch_expert_idx = _builder_assign(
                                                    "dispatch_expert_idx",
                                                    dispatch_expert_idx + 1,
                                                    locals().get(
                                                        "dispatch_expert_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                            dispatch_expert_idx = _builder_assign(
                                                "dispatch_expert_idx",
                                                T.int32(0),
                                                locals().get(
                                                    "dispatch_expert_idx", _BUILDER_MISSING
                                                ),
                                            )
                                            with T.While(dispatch_expert_idx < num_schedule_stages):
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                task_info_full_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(1),
                                                    )
                                                )
                                                _builder_emit(
                                                    T.ptx.mbarrier.init.shared.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                task_info_empty_barrier_base
                                                                + dispatch_expert_idx
                                                            ]
                                                        ),
                                                        T.uint32(num_schedule_consumer_threads),
                                                    )
                                                )
                                                dispatch_expert_idx = _builder_assign(
                                                    "dispatch_expert_idx",
                                                    dispatch_expert_idx + 1,
                                                    locals().get(
                                                        "dispatch_expert_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                    _builder_emit(T.evaluate(fence_barrier_init()))
                                with T.Else():
                                    with T.If(
                                        flat_warp_idx == kernel_config.num_dispatch_warps - 1
                                    ):
                                        with T.Then():
                                            _builder_emit(
                                                T.ptx[
                                                    f"tcgen05.alloc.cta_group::{kernel_config.num_ctas_per_cluster}.sync.aligned.shared::cta.b32"
                                                ](
                                                    T.address_of(tmem_ptr_in_smem[0]),
                                                    T.uint32(num_tmem_cols),
                                                )
                                            )
            # `fence_barrier_init` publishes the earlier mbarrier initialization,
            # but the tcgen05.alloc above performs a later weak shared-memory store
            # of the TMEM base.  Publish that store before epilogue warps read it.
            _builder_emit(T.ptx.barrier.cluster.arrive.release.aligned())
            _builder_emit(T.ptx.barrier.cluster.wait.acquire.aligned())
            with T.If(flat_warp_idx < kernel_config.num_dispatch_warps):
                with T.Then():
                    _builder_emit(warpgroup_reg_dealloc(num_dispatch_registers))
                    dispatch_token_iter = _builder_assign(
                        "dispatch_token_iter",
                        (sm_idx * kernel_config.num_dispatch_warps + flat_warp_idx)
                        * kernel_config.num_tokens_per_warp,
                        locals().get("dispatch_token_iter", _BUILDER_MISSING),
                    )
                    with T.While(
                        T.cast(dispatch_token_iter, "uint32") < T.cast(num_tokens, "uint32")
                    ):
                        with T.If(lane_idx < kernel_config.num_activate_lanes):
                            with T.Then():
                                lane_idx_u32 = _builder_assign(
                                    "lane_idx_u32",
                                    T.cast(lane_idx, "uint32"),
                                    locals().get("lane_idx_u32", _BUILDER_MISSING),
                                )
                                token_idx = _builder_assign(
                                    "token_idx",
                                    dispatch_token_iter
                                    + T.cast(lane_idx_u32 // T.uint32(num_topk), "int32"),
                                    locals().get("token_idx", _BUILDER_MISSING),
                                )
                                with T.If(
                                    T.cast(token_idx, "uint32") < T.cast(num_tokens, "uint32")
                                ):
                                    with T.Then():
                                        topk_idx = _builder_assign(
                                            "topk_idx",
                                            T.cast(lane_idx_u32 % T.uint32(num_topk), "int32"),
                                            locals().get("topk_idx", _BUILDER_MISSING),
                                        )
                                        _builder_emit(
                                            load_global_s64(
                                                ordinary_global_s64,
                                                input_topk_idx.ptr_to([token_idx, topk_idx]),
                                            )
                                        )
                                        dispatch_expert_idx = _builder_assign(
                                            "dispatch_expert_idx",
                                            T.cast(ordinary_global_s64, "int32"),
                                            locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                                        )
                                        with T.If(dispatch_expert_idx >= 0):
                                            with T.Then():
                                                _builder_emit(
                                                    T.evaluate(
                                                        T.cuda.atomic_add(
                                                            smem_expert_count.ptr_to(
                                                                [dispatch_expert_idx]
                                                            ),
                                                            1,
                                                        )
                                                    )
                                                )
                        _builder_emit(T.cuda.warp_sync())
                        dispatch_token_iter = _builder_assign(
                            "dispatch_token_iter",
                            (
                                dispatch_token_iter
                                + kernel_config.num_sms
                                * kernel_config.num_dispatch_warps
                                * kernel_config.num_tokens_per_warp
                            ),
                            locals().get("dispatch_token_iter", _BUILDER_MISSING),
                        )

                    _builder_emit(
                        T.ptx.bar.sync(
                            T.uint32(dispatch_sync_barrier_idx),
                            T.uint32(kernel_config.num_dispatch_threads),
                        )
                    )
                    dispatch_expert_idx = _builder_assign(
                        "dispatch_expert_idx",
                        flat_warp_idx * 32 + lane_idx,
                        locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                    )
                    with T.While(T.cast(dispatch_expert_idx, "uint32") < T.uint32(num_experts)):
                        _builder_emit(
                            load_shared_u32(
                                smem_expert_count_value,
                                smem_expert_count.ptr_to([dispatch_expert_idx]),
                            )
                        )
                        send_value = _builder_assign(
                            "send_value",
                            T.bitwise_or(
                                T.uint64(1 << 32), T.cast(smem_expert_count_value, "uint64")
                            ),
                            locals().get("send_value", _BUILDER_MISSING),
                        )
                        prev_send_count = _builder_alloc_scalar("prev_send_count", "uint64")
                        _builder_emit(
                            T.ptx.atom.global_.add.u64(
                                prev_send_count,
                                workspace_expert_send_count.ptr_to([dispatch_expert_idx]),
                                send_value,
                            )
                        )
                        _builder_emit(
                            st_shared_u32(
                                smem_expert_count.ptr_to([dispatch_expert_idx]),
                                T.cast(prev_send_count, "uint32"),
                            )
                        )
                        dispatch_expert_idx = _builder_assign(
                            "dispatch_expert_idx",
                            dispatch_expert_idx + kernel_config.num_dispatch_threads,
                            locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                        )
                    _builder_emit(
                        T.ptx.bar.sync(
                            T.uint32(dispatch_sync_barrier_idx),
                            T.uint32(kernel_config.num_dispatch_threads),
                        )
                    )

            with T.If(flat_warp_idx < kernel_config.num_dispatch_warps):
                with T.Then():
                    dispatch_token_iter = _builder_assign(
                        "dispatch_token_iter",
                        (sm_idx * kernel_config.num_dispatch_warps + flat_warp_idx)
                        * kernel_config.num_tokens_per_warp,
                        locals().get("dispatch_token_iter", _BUILDER_MISSING),
                    )
                    with T.While(
                        T.cast(dispatch_token_iter, "uint32") < T.cast(num_tokens, "uint32")
                    ):
                        with T.If(lane_idx < kernel_config.num_activate_lanes):
                            with T.Then():
                                lane_idx_u32 = _builder_assign(
                                    "lane_idx_u32",
                                    T.cast(lane_idx, "uint32"),
                                    locals().get("lane_idx_u32", _BUILDER_MISSING),
                                )
                                token_idx = _builder_assign(
                                    "token_idx",
                                    dispatch_token_iter
                                    + T.cast(lane_idx_u32 // T.uint32(num_topk), "int32"),
                                    locals().get("token_idx", _BUILDER_MISSING),
                                )
                                with T.If(
                                    T.cast(token_idx, "uint32") < T.cast(num_tokens, "uint32")
                                ):
                                    with T.Then():
                                        topk_idx = _builder_assign(
                                            "topk_idx",
                                            T.cast(lane_idx_u32 % T.uint32(num_topk), "int32"),
                                            locals().get("topk_idx", _BUILDER_MISSING),
                                        )
                                        dispatch_token_topk_idx = _builder_assign(
                                            "dispatch_token_topk_idx",
                                            token_idx * num_topk + topk_idx,
                                            locals().get(
                                                "dispatch_token_topk_idx", _BUILDER_MISSING
                                            ),
                                        )
                                        _builder_emit(
                                            load_global_s64(
                                                ordinary_global_s64,
                                                input_topk_idx.ptr_to([token_idx, topk_idx]),
                                            )
                                        )
                                        dispatch_expert_idx = _builder_assign(
                                            "dispatch_expert_idx",
                                            T.cast(ordinary_global_s64, "int32"),
                                            locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                                        )
                                        with T.If(dispatch_expert_idx >= 0):
                                            with T.Then():
                                                dispatch_expert_idx_u32 = _builder_assign(
                                                    "dispatch_expert_idx_u32",
                                                    T.cast(dispatch_expert_idx, "uint32"),
                                                    locals().get(
                                                        "dispatch_expert_idx_u32", _BUILDER_MISSING
                                                    ),
                                                )
                                                dispatch_dst_rank_idx = _builder_assign(
                                                    "dispatch_dst_rank_idx",
                                                    T.cast(
                                                        dispatch_expert_idx_u32
                                                        // T.uint32(num_experts_per_rank),
                                                        "int32",
                                                    ),
                                                    locals().get(
                                                        "dispatch_dst_rank_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                dispatch_dst_local_expert_idx = _builder_assign(
                                                    "dispatch_dst_local_expert_idx",
                                                    T.cast(
                                                        dispatch_expert_idx_u32
                                                        % T.uint32(num_experts_per_rank),
                                                        "int32",
                                                    ),
                                                    locals().get(
                                                        "dispatch_dst_local_expert_idx",
                                                        _BUILDER_MISSING,
                                                    ),
                                                )
                                                dispatch_dst_slot_idx = _builder_assign(
                                                    "dispatch_dst_slot_idx",
                                                    T.cuda.atomic_add(
                                                        smem_expert_count.ptr_to(
                                                            [dispatch_expert_idx]
                                                        ),
                                                        1,
                                                    ),
                                                    locals().get(
                                                        "dispatch_dst_slot_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                symm_rank_base = _builder_assign(
                                                    "symm_rank_base",
                                                    sym_buffer_base
                                                    + T.cast(symm_rank_offsets[0], "uint64"),
                                                    locals().get(
                                                        "symm_rank_base", _BUILDER_MISSING
                                                    ),
                                                )
                                                _builder_emit(
                                                    load_symm_rank_base(
                                                        symm_rank_base,
                                                        smem_symm_rank_bases,
                                                        dispatch_dst_rank_idx,
                                                    )
                                                )
                                                _builder_emit(
                                                    peer_store_u32(
                                                        symm_rank_base,
                                                        T.uint64(
                                                            workspace_layout.src_token_topk_idx_offset
                                                            + (
                                                                dispatch_dst_local_expert_idx
                                                                * num_processes
                                                                * workspace_layout.num_max_recv_tokens_per_expert
                                                                + rank_idx
                                                                * workspace_layout.num_max_recv_tokens_per_expert
                                                                + dispatch_dst_slot_idx
                                                            )
                                                            * 4
                                                        ),
                                                        T.cast(dispatch_token_topk_idx, "uint32"),
                                                    )
                                                )
                        _builder_emit(T.cuda.warp_sync())
                        dispatch_token_iter = _builder_assign(
                            "dispatch_token_iter",
                            (
                                dispatch_token_iter
                                + kernel_config.num_sms
                                * kernel_config.num_dispatch_warps
                                * kernel_config.num_tokens_per_warp
                            ),
                            locals().get("dispatch_token_iter", _BUILDER_MISSING),
                        )
            with T.If(flat_warp_idx < kernel_config.num_dispatch_warps):
                with T.Then():
                    _builder_emit(
                        workspace_grid_sync(
                            0,
                            kernel_config.num_dispatch_threads,
                            dispatch_sync_barrier_idx,
                            flat_warp_idx * 32 + lane_idx,
                        )
                    )
            with T.If(T.And(sm_idx == 0, flat_warp_idx < kernel_config.num_dispatch_warps)):
                with T.Then():
                    dispatch_expert_idx = _builder_assign(
                        "dispatch_expert_idx",
                        flat_warp_idx * 32 + lane_idx,
                        locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                    )
                    with T.While(T.cast(dispatch_expert_idx, "uint32") < T.uint32(num_experts)):
                        dispatch_expert_idx_u32 = _builder_assign(
                            "dispatch_expert_idx_u32",
                            T.cast(dispatch_expert_idx, "uint32"),
                            locals().get("dispatch_expert_idx_u32", _BUILDER_MISSING),
                        )
                        dispatch_dst_rank_idx = _builder_assign(
                            "dispatch_dst_rank_idx",
                            T.cast(
                                dispatch_expert_idx_u32 // T.uint32(num_experts_per_rank), "int32"
                            ),
                            locals().get("dispatch_dst_rank_idx", _BUILDER_MISSING),
                        )
                        dispatch_dst_local_expert_idx = _builder_assign(
                            "dispatch_dst_local_expert_idx",
                            T.cast(
                                dispatch_expert_idx_u32 % T.uint32(num_experts_per_rank), "int32"
                            ),
                            locals().get("dispatch_dst_local_expert_idx", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            load_global_u64(
                                scheduler_cached_status,
                                workspace_expert_send_count.ptr_to([dispatch_expert_idx]),
                            )
                        )
                        symm_rank_base = _builder_assign(
                            "symm_rank_base",
                            sym_buffer_base + T.cast(symm_rank_offsets[0], "uint64"),
                            locals().get("symm_rank_base", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            load_symm_rank_base(
                                symm_rank_base, smem_symm_rank_bases, dispatch_dst_rank_idx
                            )
                        )
                        _builder_emit(
                            peer_store_u64(
                                symm_rank_base,
                                T.uint64(
                                    workspace_layout.expert_recv_count_offset
                                    + (
                                        rank_idx * num_experts_per_rank
                                        + dispatch_dst_local_expert_idx
                                    )
                                    * 8
                                ),
                                T.bitwise_and(scheduler_cached_status, T.uint64(0xFFFFFFFF)),
                            )
                        )
                        peer_recv_count_sum_prev = _builder_alloc_scalar(
                            "peer_recv_count_sum_prev", "uint64"
                        )  # atom returns the old value; unused here
                        symm_rank_base = _builder_assign(
                            "symm_rank_base",
                            sym_buffer_base + T.cast(symm_rank_offsets[0], "uint64"),
                            locals().get("symm_rank_base", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            load_symm_rank_base(
                                symm_rank_base, smem_symm_rank_bases, dispatch_dst_rank_idx
                            )
                        )
                        _builder_emit(
                            peer_atomic_add_u64(
                                peer_recv_count_sum_prev,
                                symm_rank_base,
                                T.uint64(
                                    workspace_layout.expert_recv_count_sum_offset
                                    + dispatch_dst_local_expert_idx * 8
                                ),
                                scheduler_cached_status,
                            )
                        )
                        dispatch_expert_idx = _builder_assign(
                            "dispatch_expert_idx",
                            dispatch_expert_idx + kernel_config.num_dispatch_threads,
                            locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                        )
            with T.If(flat_warp_idx < kernel_config.num_dispatch_warps):
                with T.Then():
                    _builder_emit(
                        T.ptx.bar.sync(
                            T.uint32(dispatch_sync_barrier_idx),
                            T.uint32(kernel_config.num_dispatch_threads),
                        )
                    )
            with T.If(flat_warp_idx < kernel_config.num_dispatch_warps):
                with T.Then():
                    _builder_emit(
                        dispatch_nvlink_barrier_before_pull(flat_warp_idx * 32 + lane_idx)
                    )
                    _builder_emit(
                        T.evaluate(
                            sync_unaligned(
                                dispatch_with_epilogue_sync_barrier_idx,
                                kernel_config.num_dispatch_threads
                                + kernel_config.num_epilogue_threads,
                            )
                        )
                    )
            with T.If(flat_warp_idx < kernel_config.num_dispatch_warps):
                with T.Then():
                    pull_mbarrier_phase = _builder_assign(
                        "pull_mbarrier_phase",
                        T.int32(0),
                        locals().get("pull_mbarrier_phase", _BUILDER_MISSING),
                    )
                    current_expert_idx = _builder_assign(
                        "current_expert_idx",
                        T.int32(-1),
                        locals().get("current_expert_idx", _BUILDER_MISSING),
                    )
                    old_expert_idx = _builder_assign(
                        "old_expert_idx",
                        T.int32(-1),
                        locals().get("old_expert_idx", _BUILDER_MISSING),
                    )
                    expert_start_idx = _builder_assign(
                        "expert_start_idx",
                        T.int32(0),
                        locals().get("expert_start_idx", _BUILDER_MISSING),
                    )
                    expert_end_idx = _builder_assign(
                        "expert_end_idx",
                        T.int32(0),
                        locals().get("expert_end_idx", _BUILDER_MISSING),
                    )
                    pull_pool_block_offset = _builder_assign(
                        "pull_pool_block_offset",
                        T.int32(0),
                        locals().get("pull_pool_block_offset", _BUILDER_MISSING),
                    )
                    # Wait token data arrival
                    _builder_emit(scheduler_fetch_expert_recv_count())
                    dispatch_token_iter = _builder_assign(
                        "dispatch_token_iter",
                        sm_idx * kernel_config.num_dispatch_warps + flat_warp_idx,
                        locals().get("dispatch_token_iter", _BUILDER_MISSING),
                    )
                    with T.While(True):
                        old_expert_idx = _builder_assign(
                            "old_expert_idx",
                            current_expert_idx,
                            locals().get("old_expert_idx", _BUILDER_MISSING),
                        )
                        with T.While(
                            T.cast(dispatch_token_iter, "uint32")
                            >= T.cast(expert_end_idx, "uint32")
                        ):
                            current_expert_idx = _builder_assign(
                                "current_expert_idx",
                                current_expert_idx + T.int32(1),
                                locals().get("current_expert_idx", _BUILDER_MISSING),
                            )
                            with T.If(
                                T.cast(current_expert_idx, "uint32")
                                >= T.uint32(num_experts_per_rank)
                            ):
                                with T.Then():
                                    T.evaluate(T.break_loop())
                            expert_token_count_u32 = _builder_assign(
                                "expert_token_count_u32",
                                T.cast(expert_end_idx - expert_start_idx, "uint32"),
                                locals().get("expert_token_count_u32", _BUILDER_MISSING),
                            )
                            expert_num_blocks_u32 = _builder_assign(
                                "expert_num_blocks_u32",
                                (expert_token_count_u32 + T.uint32(kernel_config.block_m - 1))
                                // T.uint32(kernel_config.block_m),
                                locals().get("expert_num_blocks_u32", _BUILDER_MISSING),
                            )
                            pull_pool_block_offset = _builder_assign(
                                "pull_pool_block_offset",
                                pull_pool_block_offset + T.cast(expert_num_blocks_u32, "int32"),
                                locals().get("pull_pool_block_offset", _BUILDER_MISSING),
                            )
                            expert_start_idx = _builder_assign(
                                "expert_start_idx",
                                expert_end_idx,
                                locals().get("expert_start_idx", _BUILDER_MISSING),
                            )
                            _builder_emit(
                                scheduler_get_num_tokens(
                                    current_expert_idx,
                                    lane_idx,
                                    stored_num_tokens_per_expert,
                                    selected_num_tokens,
                                )
                            )
                            expert_end_idx = _builder_assign(
                                "expert_end_idx",
                                expert_end_idx + selected_num_tokens[0],
                                locals().get("expert_end_idx", _BUILDER_MISSING),
                            )
                        with T.If(
                            T.cast(current_expert_idx, "uint32") >= T.uint32(num_experts_per_rank)
                        ):
                            with T.Then():
                                T.evaluate(T.break_loop())
                        with T.If(old_expert_idx != current_expert_idx):
                            with T.Then():
                                old_expert_idx = _builder_assign(
                                    "old_expert_idx",
                                    current_expert_idx,
                                    locals().get("old_expert_idx", _BUILDER_MISSING),
                                )
                                with T.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                                    IRBuilder.name("rank_lane_idx", rank_lane_idx)
                                    dispatch_dst_rank_idx = _builder_assign(
                                        "dispatch_dst_rank_idx",
                                        rank_lane_idx * 32 + lane_idx,
                                        locals().get("dispatch_dst_rank_idx", _BUILDER_MISSING),
                                    )
                                    T.buffer_store(stored_rank_counts, T.uint32(0), [rank_lane_idx])
                                    with T.If(
                                        T.cast(dispatch_dst_rank_idx, "uint32")
                                        < T.uint32(num_processes)
                                    ):
                                        with T.Then():
                                            _builder_emit(
                                                load_global_u64(
                                                    ordinary_global_u64,
                                                    workspace_expert_recv_count.ptr_to(
                                                        [dispatch_dst_rank_idx, current_expert_idx]
                                                    ),
                                                )
                                            )
                                            T.buffer_store(
                                                stored_rank_counts,
                                                T.cast(ordinary_global_u64, "uint32"),
                                                [rank_lane_idx],
                                            )
                        token_idx_in_expert = _builder_assign(
                            "token_idx_in_expert",
                            dispatch_token_iter - expert_start_idx,
                            locals().get("token_idx_in_expert", _BUILDER_MISSING),
                        )
                        dispatch_dst_slot_idx = _builder_assign(
                            "dispatch_dst_slot_idx",
                            token_idx_in_expert,
                            locals().get("dispatch_dst_slot_idx", _BUILDER_MISSING),
                        )
                        round_offset = _builder_assign(
                            "round_offset",
                            T.int32(0),
                            locals().get("round_offset", _BUILDER_MISSING),
                        )
                        with T.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                            IRBuilder.name("rank_lane_idx", rank_lane_idx)
                            T.buffer_store(
                                remaining_rank_counts,
                                stored_rank_counts[rank_lane_idx],
                                [rank_lane_idx],
                            )
                        with T.While(True):
                            min_in_lane = _builder_assign(
                                "min_in_lane",
                                T.uint32(0xFFFFFFFF),
                                locals().get("min_in_lane", _BUILDER_MISSING),
                            )
                            num_actives_in_lane = _builder_assign(
                                "num_actives_in_lane",
                                T.int32(0),
                                locals().get("num_actives_in_lane", _BUILDER_MISSING),
                            )
                            with T.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                                IRBuilder.name("rank_lane_idx", rank_lane_idx)
                                with T.If(remaining_rank_counts[rank_lane_idx] > T.uint32(0)):
                                    with T.Then():
                                        num_actives_in_lane = _builder_assign(
                                            "num_actives_in_lane",
                                            num_actives_in_lane + T.int32(1),
                                            locals().get("num_actives_in_lane", _BUILDER_MISSING),
                                        )
                                        min_in_lane = _builder_assign(
                                            "min_in_lane",
                                            T.min(
                                                min_in_lane, remaining_rank_counts[rank_lane_idx]
                                            ),
                                            locals().get("min_in_lane", _BUILDER_MISSING),
                                        )
                            num_active_ranks = _builder_assign(
                                "num_active_ranks",
                                T.cast(
                                    reduce_add_sync_u32(
                                        T.uint32(0xFFFFFFFF), T.cast(num_actives_in_lane, "uint32")
                                    ),
                                    "int32",
                                ),
                                locals().get("num_active_ranks", _BUILDER_MISSING),
                            )
                            min_active_count = _builder_assign(
                                "min_active_count",
                                T.cast(
                                    reduce_min_sync_u32(T.uint32(0xFFFFFFFF), min_in_lane), "int32"
                                ),
                                locals().get("min_active_count", _BUILDER_MISSING),
                            )
                            round_token_count = _builder_assign(
                                "round_token_count",
                                min_active_count * num_active_ranks,
                                locals().get("round_token_count", _BUILDER_MISSING),
                            )
                            with T.If(
                                T.cast(dispatch_dst_slot_idx, "uint32")
                                < T.cast(round_token_count, "uint32")
                            ):
                                with T.Then():
                                    dispatch_dst_slot_idx_u32 = _builder_assign(
                                        "dispatch_dst_slot_idx_u32",
                                        T.cast(dispatch_dst_slot_idx, "uint32"),
                                        locals().get("dispatch_dst_slot_idx_u32", _BUILDER_MISSING),
                                    )
                                    num_active_ranks_u32 = _builder_assign(
                                        "num_active_ranks_u32",
                                        T.cast(num_active_ranks, "uint32"),
                                        locals().get("num_active_ranks_u32", _BUILDER_MISSING),
                                    )
                                    slot_idx_in_round = _builder_assign(
                                        "slot_idx_in_round",
                                        T.cast(
                                            dispatch_dst_slot_idx_u32 % num_active_ranks_u32,
                                            "int32",
                                        ),
                                        locals().get("slot_idx_in_round", _BUILDER_MISSING),
                                    )
                                    num_seen_ranks = _builder_assign(
                                        "num_seen_ranks",
                                        T.int32(0),
                                        locals().get("num_seen_ranks", _BUILDER_MISSING),
                                    )
                                    current_rank_in_expert_idx = _builder_assign(
                                        "current_rank_in_expert_idx",
                                        T.int32(0),
                                        locals().get(
                                            "current_rank_in_expert_idx", _BUILDER_MISSING
                                        ),
                                    )
                                    with T.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                                        IRBuilder.name("rank_lane_idx", rank_lane_idx)
                                        rank_count_mask = _builder_assign(
                                            "rank_count_mask",
                                            ballot_sync(
                                                T.uint32(0xFFFFFFFF),
                                                remaining_rank_counts[rank_lane_idx] > T.uint32(0),
                                            ),
                                            locals().get("rank_count_mask", _BUILDER_MISSING),
                                        )
                                        active_lane_count = _builder_assign(
                                            "active_lane_count",
                                            T.cast(T.popcount(rank_count_mask), "int32"),
                                            locals().get("active_lane_count", _BUILDER_MISSING),
                                        )
                                        with T.If(
                                            T.And(
                                                T.cast(slot_idx_in_round, "uint32")
                                                >= T.cast(num_seen_ranks, "uint32"),
                                                T.cast(slot_idx_in_round, "uint32")
                                                < T.cast(
                                                    num_seen_ranks + active_lane_count, "uint32"
                                                ),
                                            )
                                        ):
                                            with T.Then():
                                                rank_slot_bit = _builder_alloc_scalar(
                                                    "rank_slot_bit", "uint32"
                                                )
                                                _builder_emit(
                                                    fns_b32(
                                                        rank_slot_bit,
                                                        rank_count_mask,
                                                        T.uint32(0),
                                                        slot_idx_in_round
                                                        - num_seen_ranks
                                                        + T.int32(1),
                                                    )
                                                )
                                                current_rank_in_expert_idx = _builder_assign(
                                                    "current_rank_in_expert_idx",
                                                    rank_lane_idx * 32
                                                    + T.cast(rank_slot_bit, "int32"),
                                                    locals().get(
                                                        "current_rank_in_expert_idx",
                                                        _BUILDER_MISSING,
                                                    ),
                                                )
                                        num_seen_ranks = _builder_assign(
                                            "num_seen_ranks",
                                            num_seen_ranks + active_lane_count,
                                            locals().get("num_seen_ranks", _BUILDER_MISSING),
                                        )
                                    token_idx_in_rank = _builder_assign(
                                        "token_idx_in_rank",
                                        round_offset
                                        + T.cast(
                                            dispatch_dst_slot_idx_u32 // num_active_ranks_u32,
                                            "int32",
                                        ),
                                        locals().get("token_idx_in_rank", _BUILDER_MISSING),
                                    )
                                    T.evaluate(T.break_loop())
                            dispatch_dst_slot_idx = _builder_assign(
                                "dispatch_dst_slot_idx",
                                dispatch_dst_slot_idx - round_token_count,
                                locals().get("dispatch_dst_slot_idx", _BUILDER_MISSING),
                            )
                            round_offset = _builder_assign(
                                "round_offset",
                                round_offset + min_active_count,
                                locals().get("round_offset", _BUILDER_MISSING),
                            )
                            with T.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                                IRBuilder.name("rank_lane_idx", rank_lane_idx)
                                T.buffer_store(
                                    remaining_rank_counts,
                                    remaining_rank_counts[rank_lane_idx]
                                    - T.min(
                                        remaining_rank_counts[rank_lane_idx],
                                        T.cast(min_active_count, "uint32"),
                                    ),
                                    [rank_lane_idx],
                                )
                        _builder_emit(
                            load_global_u32(
                                ordinary_global_u32,
                                workspace_src_token_topk_idx.ptr_to(
                                    [
                                        current_expert_idx,
                                        current_rank_in_expert_idx,
                                        token_idx_in_rank,
                                    ]
                                ),
                            )
                        )
                        pull_src_token_topk_idx = _builder_assign(
                            "pull_src_token_topk_idx",
                            T.cast(ordinary_global_u32, "int32"),
                            locals().get("pull_src_token_topk_idx", _BUILDER_MISSING),
                        )
                        pull_src_token_topk_idx_u32 = _builder_assign(
                            "pull_src_token_topk_idx_u32",
                            T.cast(pull_src_token_topk_idx, "uint32"),
                            locals().get("pull_src_token_topk_idx_u32", _BUILDER_MISSING),
                        )
                        pull_src_token_idx = _builder_assign(
                            "pull_src_token_idx",
                            T.cast(pull_src_token_topk_idx_u32 // T.uint32(num_topk), "int32"),
                            locals().get("pull_src_token_idx", _BUILDER_MISSING),
                        )
                        pull_src_topk_idx = _builder_assign(
                            "pull_src_topk_idx",
                            T.cast(pull_src_token_topk_idx_u32 % T.uint32(num_topk), "int32"),
                            locals().get("pull_src_topk_idx", _BUILDER_MISSING),
                        )
                        pull_pool_token_idx = _builder_assign(
                            "pull_pool_token_idx",
                            (pull_pool_block_offset * kernel_config.block_m + token_idx_in_expert),
                            locals().get("pull_pool_token_idx", _BUILDER_MISSING),
                        )
                        pull_pool_block_idx = _builder_assign(
                            "pull_pool_block_idx",
                            pull_pool_token_idx // kernel_config.block_m,
                            locals().get("pull_pool_block_idx", _BUILDER_MISSING),
                        )
                        pull_ring_block_idx = _builder_assign(
                            "pull_ring_block_idx",
                            pull_pool_block_idx % num_ring_blocks,
                            locals().get("pull_ring_block_idx", _BUILDER_MISSING),
                        )
                        pull_ring_token_idx = _builder_assign(
                            "pull_ring_token_idx",
                            pull_pool_token_idx % num_ring_tokens,
                            locals().get("pull_ring_token_idx", _BUILDER_MISSING),
                        )
                        l1_empty_count_target = _builder_assign(
                            "l1_empty_count_target",
                            pull_pool_block_idx // num_ring_blocks * num_l1_block_ns,
                            locals().get("l1_empty_count_target", _BUILDER_MISSING),
                        )
                        with T.If(l1_empty_count_target > 0):
                            with T.Then():
                                _builder_emit(
                                    load_acq_u32(
                                        current_ring_count,
                                        workspace_l1_empty_count.ptr_to([pull_ring_block_idx]),
                                    )
                                )
                                with T.While(
                                    current_ring_count < T.cast(l1_empty_count_target, "uint32")
                                ):
                                    _builder_emit(
                                        load_acq_u32(
                                            current_ring_count,
                                            workspace_l1_empty_count.ptr_to([pull_ring_block_idx]),
                                        )
                                    )
                        with T.If(T.cuda.elect_sync()):
                            with T.Then():
                                with T.unroll(0, num_pull_chunks) as pull_chunk_idx:
                                    IRBuilder.name("pull_chunk_idx", pull_chunk_idx)
                                    symm_rank_base = _builder_assign(
                                        "symm_rank_base",
                                        sym_buffer_base + T.cast(symm_rank_offsets[0], "uint64"),
                                        locals().get("symm_rank_base", _BUILDER_MISSING),
                                    )
                                    _builder_emit(
                                        load_symm_rank_base(
                                            symm_rank_base,
                                            smem_symm_rank_bases,
                                            current_rank_in_expert_idx,
                                        )
                                    )
                                    _builder_emit(
                                        tma_load_1d_symm(
                                            smem_send_buffers.ptr_to([flat_warp_idx, 0]),
                                            symm_rank_base,
                                            T.uint64(
                                                symm_buffer_layout.input_token_offset
                                                + pull_src_token_idx * hidden
                                                + pull_chunk_idx * kernel_config.num_bytes_per_pull
                                            ),
                                            smem_barriers.ptr_to(
                                                [dispatch_barrier_base + flat_warp_idx]
                                            ),
                                            kernel_config.num_bytes_per_pull,
                                        )
                                    )
                                    _builder_emit(
                                        mbarrier_arrive_and_set_tx(
                                            smem_barriers.ptr_to(
                                                [dispatch_barrier_base + flat_warp_idx]
                                            ),
                                            kernel_config.num_bytes_per_pull,
                                        )
                                    )
                                    with T.If(pull_chunk_idx != num_pull_chunks - 1):
                                        with T.Then():
                                            _builder_emit(
                                                mbarrier_wait_phase(
                                                    smem_barriers.ptr_to(
                                                        [dispatch_barrier_base + flat_warp_idx]
                                                    ),
                                                    pull_mbarrier_phase,
                                                )
                                            )
                                            pull_mbarrier_phase = _builder_assign(
                                                "pull_mbarrier_phase",
                                                pull_mbarrier_phase ^ T.int32(1),
                                                locals().get(
                                                    "pull_mbarrier_phase", _BUILDER_MISSING
                                                ),
                                            )
                                            _builder_emit(
                                                tma_store_1d(
                                                    T.address_of(
                                                        l1_acts[
                                                            pull_ring_token_idx,
                                                            pull_chunk_idx
                                                            * kernel_config.num_bytes_per_pull,
                                                        ]
                                                    ),
                                                    smem_send_buffers.ptr_to([flat_warp_idx, 0]),
                                                    kernel_config.num_bytes_per_pull,
                                                )
                                            )
                                            _builder_emit(T.evaluate(tma_store_arrive()))
                                            _builder_emit(T.evaluate(tma_store_wait(0)))
                        _builder_emit(T.cuda.warp_sync())
                        token_idx_in_block = _builder_assign(
                            "token_idx_in_block",
                            token_idx_in_expert % kernel_config.block_m,
                            locals().get("token_idx_in_block", _BUILDER_MISSING),
                        )
                        sf_row_idx = _builder_assign(
                            "sf_row_idx",
                            pull_ring_block_idx * sf_block_m
                            + transform_sf_token_idx(token_idx_in_block),
                            locals().get("sf_row_idx", _BUILDER_MISSING),
                        )
                        dispatch_dst_rank_idx = _builder_assign(
                            "dispatch_dst_rank_idx",
                            lane_idx,
                            locals().get("dispatch_dst_rank_idx", _BUILDER_MISSING),
                        )
                        with T.While(dispatch_dst_rank_idx < hidden // 128):
                            pulled_sf = _builder_alloc_scalar("pulled_sf", "uint32")
                            symm_rank_base = _builder_assign(
                                "symm_rank_base",
                                sym_buffer_base + T.cast(symm_rank_offsets[0], "uint64"),
                                locals().get("symm_rank_base", _BUILDER_MISSING),
                            )
                            _builder_emit(
                                load_symm_rank_base(
                                    symm_rank_base, smem_symm_rank_bases, current_rank_in_expert_idx
                                )
                            )
                            _builder_emit(
                                peer_load_u32(
                                    pulled_sf,
                                    symm_rank_base,
                                    T.uint64(
                                        symm_buffer_layout.input_sf_offset
                                        + (
                                            pull_src_token_idx * (hidden // 128)
                                            + dispatch_dst_rank_idx
                                        )
                                        * 4
                                    ),
                                )
                            )
                            _builder_emit(
                                store_global_u32(
                                    l1_acts_sf.ptr_to([dispatch_dst_rank_idx, sf_row_idx]),
                                    pulled_sf,
                                )
                            )
                            dispatch_dst_rank_idx = _builder_assign(
                                "dispatch_dst_rank_idx",
                                dispatch_dst_rank_idx + 32,
                                locals().get("dispatch_dst_rank_idx", _BUILDER_MISSING),
                            )
                        _builder_emit(T.cuda.warp_sync())
                        with T.If(T.cuda.elect_sync()):
                            with T.Then():
                                pulled_weight = _builder_alloc_scalar("pulled_weight", "float32")
                                symm_rank_base = _builder_assign(
                                    "symm_rank_base",
                                    sym_buffer_base + T.cast(symm_rank_offsets[0], "uint64"),
                                    locals().get("symm_rank_base", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    load_symm_rank_base(
                                        symm_rank_base,
                                        smem_symm_rank_bases,
                                        current_rank_in_expert_idx,
                                    )
                                )
                                _builder_emit(
                                    peer_load_f32(
                                        pulled_weight,
                                        symm_rank_base,
                                        T.uint64(
                                            symm_buffer_layout.input_topk_weights_offset
                                            + pull_src_token_topk_idx * 4
                                        ),
                                    )
                                )
                                _builder_emit(
                                    store_global_f32(
                                        l1_topk_weights.ptr_to([pull_ring_token_idx]), pulled_weight
                                    )
                                )
                                _builder_emit(
                                    store_token_src_metadata(
                                        pull_pool_token_idx,
                                        current_rank_in_expert_idx,
                                        pull_src_token_idx,
                                        pull_src_topk_idx,
                                    )
                                )
                                _builder_emit(
                                    mbarrier_wait_phase(
                                        smem_barriers.ptr_to(
                                            [dispatch_barrier_base + flat_warp_idx]
                                        ),
                                        pull_mbarrier_phase,
                                    )
                                )
                                pull_mbarrier_phase = _builder_assign(
                                    "pull_mbarrier_phase",
                                    pull_mbarrier_phase ^ T.int32(1),
                                    locals().get("pull_mbarrier_phase", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    tma_store_1d(
                                        T.address_of(
                                            l1_acts[
                                                pull_ring_token_idx,
                                                (num_pull_chunks - 1)
                                                * kernel_config.num_bytes_per_pull,
                                            ]
                                        ),
                                        smem_send_buffers.ptr_to([flat_warp_idx, 0]),
                                        kernel_config.num_bytes_per_pull,
                                    )
                                )
                                _builder_emit(T.evaluate(tma_store_arrive()))
                                _builder_emit(T.evaluate(tma_store_wait(0)))
                                _builder_emit(
                                    atomic_add_rel_u32(
                                        atom_prev_unused,
                                        workspace_l1_full_count.ptr_to([pull_ring_block_idx]),
                                        T.Select(
                                            dispatch_token_iter == expert_end_idx - T.int32(1),
                                            T.uint32(kernel_config.block_m)
                                            - T.cast(token_idx_in_block, "uint32"),
                                            T.uint32(1),
                                        ),
                                    )
                                )
                        _builder_emit(T.cuda.warp_sync())
                        dispatch_token_iter = _builder_assign(
                            "dispatch_token_iter",
                            (
                                dispatch_token_iter
                                + kernel_config.num_sms * kernel_config.num_dispatch_warps
                            ),
                            locals().get("dispatch_token_iter", _BUILDER_MISSING),
                        )
                    _builder_emit(
                        T.evaluate(
                            sync_unaligned(
                                dispatch_with_epilogue_sync_barrier_idx,
                                kernel_config.num_dispatch_threads
                                + kernel_config.num_epilogue_threads,
                            )
                        )
                    )
                    # Dispatch and load-A read reusable global workspace after the
                    # preceding grid rendezvous. The epilogue grid rendezvous cannot
                    # publish those reads because it precedes this CTA-local join.
                    _builder_emit(
                        workspace_grid_sync(
                            dispatch_grid_sync_index,
                            kernel_config.num_dispatch_threads + 32,
                            dispatch_with_load_a_sync_barrier_idx,
                            flat_warp_idx * 32 + lane_idx,
                        )
                    )
                    with T.If(sm_idx == 0):
                        with T.Then():
                            # SM 0: clear expert send count and schedule task counters
                            dispatch_expert_idx = _builder_assign(
                                "dispatch_expert_idx",
                                thread_idx,
                                locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                            )
                            with T.While(dispatch_expert_idx < num_experts):
                                _builder_emit(
                                    store_global_u64(
                                        workspace_expert_send_count.ptr_to([dispatch_expert_idx]),
                                        T.uint64(0),
                                    )
                                )
                                dispatch_expert_idx = _builder_assign(
                                    "dispatch_expert_idx",
                                    dispatch_expert_idx + kernel_config.num_dispatch_threads,
                                    locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                                )
                            with T.If((flat_warp_idx == 0) & T.cuda.elect_sync() != 0):
                                with T.Then():
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_l1_task_count.ptr_to([0]), T.uint32(0)
                                        )
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_l2_task_count.ptr_to([0]), T.uint32(0)
                                        )
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_shared_l1_task_count.ptr_to([0]), T.uint32(0)
                                        )
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_shared_l2_task_count.ptr_to([0]), T.uint32(0)
                                        )
                                    )
                            _builder_emit(T.cuda.warp_sync())
                            dispatch_expert_idx = _builder_assign(
                                "dispatch_expert_idx",
                                thread_idx,
                                locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                            )
                            with T.While(
                                dispatch_expert_idx < workspace_layout.num_shared_l2_pool_blocks
                            ):
                                _builder_emit(
                                    store_global_u32(
                                        workspace_shared_l2_full_count.ptr_to(
                                            [dispatch_expert_idx]
                                        ),
                                        T.uint32(0),
                                    )
                                )
                                dispatch_expert_idx = _builder_assign(
                                    "dispatch_expert_idx",
                                    dispatch_expert_idx + kernel_config.num_dispatch_threads,
                                    locals().get("dispatch_expert_idx", _BUILDER_MISSING),
                                )
                            _builder_emit(T.cuda.warp_sync())
                        with T.Else():
                            pull_local_expert_idx = _builder_assign(
                                "pull_local_expert_idx",
                                sm_idx - 1,
                                locals().get("pull_local_expert_idx", _BUILDER_MISSING),
                            )
                            with T.While(pull_local_expert_idx < num_experts_per_rank):
                                _builder_emit(
                                    scheduler_get_num_tokens(
                                        pull_local_expert_idx,
                                        lane_idx,
                                        stored_num_tokens_per_expert,
                                        selected_num_tokens,
                                    )
                                )
                                pull_num_tokens = _builder_assign(
                                    "pull_num_tokens",
                                    selected_num_tokens[0],
                                    locals().get("pull_num_tokens", _BUILDER_MISSING),
                                )
                                pull_num_tokens_u32 = _builder_assign(
                                    "pull_num_tokens_u32",
                                    T.cast(pull_num_tokens, "uint32"),
                                    locals().get("pull_num_tokens_u32", _BUILDER_MISSING),
                                )
                                scheduler_num_m_blocks = _builder_assign(
                                    "scheduler_num_m_blocks",
                                    T.cast(
                                        (pull_num_tokens_u32 + T.uint32(kernel_config.block_m - 1))
                                        // T.uint32(kernel_config.block_m),
                                        "int32",
                                    ),
                                    locals().get("scheduler_num_m_blocks", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    scheduler_get_pool_block_offset(
                                        pull_local_expert_idx,
                                        lane_idx,
                                        stored_num_tokens_per_expert,
                                        pool_block_offset_sum,
                                    )
                                )
                                pull_pool_block_offset = _builder_assign(
                                    "pull_pool_block_offset",
                                    pool_block_offset_sum[0],
                                    locals().get("pull_pool_block_offset", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    T.ptx.bar.sync(
                                        T.uint32(dispatch_sync_barrier_idx),
                                        T.uint32(kernel_config.num_dispatch_threads),
                                    )
                                )
                                with T.If(thread_idx == 0):
                                    with T.Then():
                                        _builder_emit(
                                            store_global_u64(
                                                workspace_expert_recv_count_sum.ptr_to(
                                                    [pull_local_expert_idx]
                                                ),
                                                T.uint64(0),
                                            )
                                        )
                                if kernel_collect_stats:
                                    with T.If(T.And(flat_warp_idx == 1, lane_idx == 0)):
                                        with T.Then():
                                            _builder_emit(
                                                T.evaluate(
                                                    red_add_gpu_s32(
                                                        cumulative_local_expert_recv_stats.ptr_to(
                                                            [pull_local_expert_idx]
                                                        ),
                                                        pull_num_tokens,
                                                    )
                                                )
                                            )
                                dispatch_dst_rank_idx = _builder_assign(
                                    "dispatch_dst_rank_idx",
                                    thread_idx,
                                    locals().get("dispatch_dst_rank_idx", _BUILDER_MISSING),
                                )
                                with T.While(dispatch_dst_rank_idx < T.int32(num_processes)):
                                    _builder_emit(
                                        store_global_u64(
                                            workspace_expert_recv_count.ptr_to(
                                                [dispatch_dst_rank_idx, pull_local_expert_idx]
                                            ),
                                            T.uint64(0),
                                        )
                                    )
                                    dispatch_dst_rank_idx = _builder_assign(
                                        "dispatch_dst_rank_idx",
                                        (
                                            dispatch_dst_rank_idx
                                            + kernel_config.num_dispatch_threads
                                        ),
                                        locals().get("dispatch_dst_rank_idx", _BUILDER_MISSING),
                                    )
                                dispatch_dst_slot_idx = _builder_assign(
                                    "dispatch_dst_slot_idx",
                                    thread_idx,
                                    locals().get("dispatch_dst_slot_idx", _BUILDER_MISSING),
                                )
                                with T.While(dispatch_dst_slot_idx < scheduler_num_m_blocks):
                                    pull_ring_block_idx = _builder_assign(
                                        "pull_ring_block_idx",
                                        (pull_pool_block_offset + dispatch_dst_slot_idx)
                                        % num_ring_blocks,
                                        locals().get("pull_ring_block_idx", _BUILDER_MISSING),
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_l1_full_count.ptr_to([pull_ring_block_idx]),
                                            T.uint32(0),
                                        )
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_l1_empty_count.ptr_to([pull_ring_block_idx]),
                                            T.uint32(0),
                                        )
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_l2_full_count.ptr_to([pull_ring_block_idx]),
                                            T.uint32(0),
                                        )
                                    )
                                    _builder_emit(
                                        store_global_u32(
                                            workspace_l2_empty_count.ptr_to([pull_ring_block_idx]),
                                            T.uint32(0),
                                        )
                                    )
                                    dispatch_dst_slot_idx = _builder_assign(
                                        "dispatch_dst_slot_idx",
                                        (
                                            dispatch_dst_slot_idx
                                            + kernel_config.num_dispatch_threads
                                        ),
                                        locals().get("dispatch_dst_slot_idx", _BUILDER_MISSING),
                                    )
                                pull_local_expert_idx = _builder_assign(
                                    "pull_local_expert_idx",
                                    pull_local_expert_idx + (kernel_config.num_sms - 1),
                                    locals().get("pull_local_expert_idx", _BUILDER_MISSING),
                                )
                    _builder_emit(
                        dispatch_nvlink_barrier_after_workspace_clean(flat_warp_idx * 32 + lane_idx)
                    )
            current_iter_idx = _builder_alloc_scalar("current_iter_idx", "int32")
            accum_stage_idx = _builder_alloc_scalar("accum_stage_idx", "int32")
            accum_phase = _builder_alloc_scalar("accum_phase", "int32")
            block_phase = _builder_alloc_scalar("block_phase", "int32")
            local_expert_idx = _builder_alloc_scalar("local_expert_idx", "int32")
            num_k_blocks = _builder_alloc_scalar("num_k_blocks", "int32")
            m_block_idx = _builder_alloc_scalar("m_block_idx", "int32")
            n_block_idx = _builder_alloc_scalar("n_block_idx", "int32")
            n_cluster_idx = _builder_alloc_scalar("n_cluster_idx", "int32")
            get_valid_m_true = _builder_alloc_scalar("get_valid_m_true", "int32")
            get_valid_m_true_half = _builder_alloc_scalar("get_valid_m_true_half", "int32")
            get_valid_m_true_eighth = _builder_alloc_scalar("get_valid_m_true_eighth", "int32")
            shape_k = _builder_alloc_scalar("shape_k", "int32")
            shape_n = _builder_alloc_scalar("shape_n", "int32")
            shape_sfa_k = _builder_alloc_scalar("shape_sfa_k", "int32")
            shape_sfb_k = _builder_alloc_scalar("shape_sfb_k", "int32")
            current_iter_idx = _builder_assign(
                "current_iter_idx", 0, locals().get("current_iter_idx", _BUILDER_MISSING)
            )
            accum_stage_idx = _builder_assign(
                "accum_stage_idx", 0, locals().get("accum_stage_idx", _BUILDER_MISSING)
            )
            accum_phase = _builder_assign(
                "accum_phase", 0, locals().get("accum_phase", _BUILDER_MISSING)
            )
            block_phase = _builder_assign(
                "block_phase", 0, locals().get("block_phase", _BUILDER_MISSING)
            )
            local_expert_idx = _builder_assign(
                "local_expert_idx", 0, locals().get("local_expert_idx", _BUILDER_MISSING)
            )
            num_k_blocks = _builder_assign(
                "num_k_blocks", 0, locals().get("num_k_blocks", _BUILDER_MISSING)
            )
            m_block_idx = _builder_assign(
                "m_block_idx", 0, locals().get("m_block_idx", _BUILDER_MISSING)
            )
            n_block_idx = _builder_assign(
                "n_block_idx", 0, locals().get("n_block_idx", _BUILDER_MISSING)
            )
            n_cluster_idx = _builder_assign(
                "n_cluster_idx", 0, locals().get("n_cluster_idx", _BUILDER_MISSING)
            )
            get_valid_m_true = _builder_assign(
                "get_valid_m_true", 0, locals().get("get_valid_m_true", _BUILDER_MISSING)
            )
            get_valid_m_true_half = _builder_assign(
                "get_valid_m_true_half", 0, locals().get("get_valid_m_true_half", _BUILDER_MISSING)
            )
            get_valid_m_true_eighth = _builder_assign(
                "get_valid_m_true_eighth",
                0,
                locals().get("get_valid_m_true_eighth", _BUILDER_MISSING),
            )
            shape_k = _builder_assign("shape_k", 0, locals().get("shape_k", _BUILDER_MISSING))
            shape_n = _builder_assign("shape_n", 0, locals().get("shape_n", _BUILDER_MISSING))
            shape_sfa_k = _builder_assign(
                "shape_sfa_k", 0, locals().get("shape_sfa_k", _BUILDER_MISSING)
            )
            shape_sfb_k = _builder_assign(
                "shape_sfb_k", 0, locals().get("shape_sfb_k", _BUILDER_MISSING)
            )

            with T.If(flat_warp_idx == kernel_config.load_a_warp_idx):
                with T.Then():
                    _builder_emit(warpgroup_reg_dealloc(num_non_epilogue_registers))
                    sched_stage_idx = _builder_assign(
                        "sched_stage_idx",
                        T.int32(0),
                        locals().get("sched_stage_idx", _BUILDER_MISSING),
                    )
                    sched_phase = _builder_assign(
                        "sched_phase", T.int32(0), locals().get("sched_phase", _BUILDER_MISSING)
                    )
                    pipeline_stage_idx = _builder_assign(
                        "pipeline_stage_idx",
                        T.int32(0),
                        locals().get("pipeline_stage_idx", _BUILDER_MISSING),
                    )
                    pipeline_phase = _builder_assign(
                        "pipeline_phase",
                        T.int32(0),
                        locals().get("pipeline_phase", _BUILDER_MISSING),
                    )
                    with T.While(True):
                        _builder_emit(consumer_get_next_task())
                        _builder_emit(consumer_bind_task_args())
                        with T.If(block_phase == T.int32(0)):
                            with T.Then():
                                T.evaluate(T.break_loop())
                        shape_k_u32 = _builder_assign(
                            "shape_k_u32",
                            T.cast(shape_k, "uint32"),
                            locals().get("shape_k_u32", _BUILDER_MISSING),
                        )
                        num_k_blocks = _builder_assign(
                            "num_k_blocks",
                            T.cast(
                                (shape_k_u32 + T.uint32(kernel_config.block_k - 1))
                                // T.uint32(kernel_config.block_k),
                                "int32",
                            ),
                            locals().get("num_k_blocks", _BUILDER_MISSING),
                        )
                        ring_block_idx = _builder_assign(
                            "ring_block_idx",
                            pool_block_idx % num_ring_blocks,
                            locals().get("ring_block_idx", _BUILDER_MISSING),
                        )
                        with T.If(block_phase == T.int32(1)):
                            with T.Then():
                                expected_ring_count = _builder_assign(
                                    "expected_ring_count",
                                    kernel_config.block_m * (pool_block_idx // num_ring_blocks + 1),
                                    locals().get("expected_ring_count", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    load_acq_u32(
                                        current_ring_count,
                                        workspace_l1_full_count.ptr_to([ring_block_idx]),
                                    )
                                )
                                with T.While(
                                    current_ring_count != T.cast(expected_ring_count, "uint32")
                                ):
                                    _builder_emit(
                                        load_acq_u32(
                                            current_ring_count,
                                            workspace_l1_full_count.ptr_to([ring_block_idx]),
                                        )
                                    )
                            with T.Else():
                                expected_ring_count = _builder_assign(
                                    "expected_ring_count",
                                    (
                                        intermediate_hidden
                                        // kernel_config.block_n
                                        * 2
                                        * (pool_block_idx // num_ring_blocks + 1)
                                    ),
                                    locals().get("expected_ring_count", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    load_acq_u32(
                                        current_ring_count,
                                        workspace_l2_full_count.ptr_to([ring_block_idx]),
                                    )
                                )
                                with T.While(
                                    current_ring_count != T.cast(expected_ring_count, "uint32")
                                ):
                                    _builder_emit(
                                        load_acq_u32(
                                            current_ring_count,
                                            workspace_l2_full_count.ptr_to([ring_block_idx]),
                                        )
                                    )
                        # The release/acquire counters publish generic global stores
                        # from dispatch (L1) or the preceding epilogue's scale-factor
                        # writes (L2).  TMA reads through the async proxy, so bridge the
                        # acquired frontier before loading either ring.
                        _builder_emit(T.ptx.fence.proxy.async_.global_())
                        with T.serial(0, num_k_blocks) as k_block_idx:
                            IRBuilder.name("k_block_idx", k_block_idx)
                            _builder_emit(
                                barrier_wait(
                                    smem_barriers.ptr_to([empty_barrier_base + pipeline_stage_idx]),
                                    pipeline_phase ^ T.int32(1),
                                )
                            )
                            m_idx = _builder_assign(
                                "m_idx",
                                ring_block_idx * kernel_config.block_m,
                                locals().get("m_idx", _BUILDER_MISSING),
                            )
                            k_idx = _builder_assign(
                                "k_idx",
                                k_block_idx * kernel_config.block_k,
                                locals().get("k_idx", _BUILDER_MISSING),
                            )
                            sfa_m_idx = _builder_assign(
                                "sfa_m_idx",
                                ring_block_idx * sf_block_m,
                                locals().get("sfa_m_idx", _BUILDER_MISSING),
                            )
                            sfa_k_idx = _builder_scalar(
                                "sfa_k_idx", k_block_idx * sf_smem_outer_dim, "int32"
                            )
                            with T.If(cta_idx_in_cluster != 0):
                                with T.Then():
                                    _builder_emit(update_get_valid_m_true())
                                    m_idx = _builder_assign(
                                        "m_idx",
                                        m_idx + get_valid_m_true_half,
                                        locals().get("m_idx", _BUILDER_MISSING),
                                    )
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    full_barrier_ptr = _builder_assign(
                                        "full_barrier_ptr",
                                        smem_barriers.ptr_to(
                                            [full_barrier_base + pipeline_stage_idx]
                                        ),
                                        locals().get("full_barrier_ptr", _BUILDER_MISSING),
                                    )
                                    with T.unroll(
                                        0, kernel_config.block_k // umma_block_k
                                    ) as tma_k_atom_idx:
                                        IRBuilder.name("tma_k_atom_idx", tma_k_atom_idx)
                                        _builder_emit(
                                            tma_copy_2d_multicast_select(
                                                smem_a.ptr_to(
                                                    [
                                                        pipeline_stage_idx,
                                                        0,
                                                        tma_k_atom_idx * umma_block_k,
                                                    ]
                                                ),
                                                full_barrier_ptr,
                                                tensor_map_l1_acts,
                                                tensor_map_l2_acts,
                                                block_phase,
                                                k_idx + tma_k_atom_idx * umma_block_k,
                                                m_idx,
                                            )
                                        )
                                    _builder_emit(
                                        tma_copy_2d_multicast_select(
                                            smem_sfa_i32.ptr_to([pipeline_stage_idx, 0, 0]),
                                            full_barrier_ptr,
                                            tensor_map_l1_acts_sf,
                                            tensor_map_l2_acts_sf,
                                            block_phase,
                                            sfa_m_idx,
                                            sfa_k_idx,
                                        )
                                    )
                                    with T.If(cta_idx_in_cluster == 0):
                                        with T.Then():
                                            _builder_emit(
                                                full_barrier_arrive_and_expect_tx(
                                                    full_barrier_ptr, full_a_expect_tx_leader_bytes
                                                )
                                            )
                                        with T.Else():
                                            _builder_emit(
                                                full_barrier_arrive_cta0(full_barrier_ptr)
                                            )
                            _builder_emit(T.cuda.warp_sync())
                            _builder_emit(advance_pipeline())
                    _builder_emit(
                        workspace_grid_sync(
                            dispatch_grid_sync_index,
                            kernel_config.num_dispatch_threads + 32,
                            dispatch_with_load_a_sync_barrier_idx,
                            kernel_config.num_dispatch_threads + lane_idx,
                        )
                    )
                with T.Else():
                    with T.If(flat_warp_idx == kernel_config.load_b_warp_idx):
                        with T.Then():
                            _builder_emit(warpgroup_reg_dealloc(num_non_epilogue_registers))
                            sched_stage_idx = _builder_assign(
                                "sched_stage_idx",
                                T.int32(0),
                                locals().get("sched_stage_idx", _BUILDER_MISSING),
                            )
                            sched_phase = _builder_assign(
                                "sched_phase",
                                T.int32(0),
                                locals().get("sched_phase", _BUILDER_MISSING),
                            )
                            pipeline_stage_idx = _builder_assign(
                                "pipeline_stage_idx",
                                T.int32(0),
                                locals().get("pipeline_stage_idx", _BUILDER_MISSING),
                            )
                            pipeline_phase = _builder_assign(
                                "pipeline_phase",
                                T.int32(0),
                                locals().get("pipeline_phase", _BUILDER_MISSING),
                            )
                            with T.While(True):
                                _builder_emit(consumer_get_next_task())
                                _builder_emit(consumer_bind_task_args())
                                with T.If(block_phase == T.int32(0)):
                                    with T.Then():
                                        T.evaluate(T.break_loop())
                                shape_k_u32 = _builder_assign(
                                    "shape_k_u32",
                                    T.cast(shape_k, "uint32"),
                                    locals().get("shape_k_u32", _BUILDER_MISSING),
                                )
                                shape_sfb_k = _builder_assign(
                                    "shape_sfb_k",
                                    T.cast((shape_k_u32 + T.uint32(127)) // T.uint32(128), "int32"),
                                    locals().get("shape_sfb_k", _BUILDER_MISSING),
                                )
                                n_block_idx = _builder_assign(
                                    "n_block_idx",
                                    n_cluster_idx * 2
                                    + T.Select(cta_idx_in_cluster == 0, T.int32(0), T.int32(1)),
                                    locals().get("n_block_idx", _BUILDER_MISSING),
                                )
                                num_k_blocks = _builder_assign(
                                    "num_k_blocks",
                                    T.cast(
                                        (shape_k_u32 + T.uint32(kernel_config.block_k - 1))
                                        // T.uint32(kernel_config.block_k),
                                        "int32",
                                    ),
                                    locals().get("num_k_blocks", _BUILDER_MISSING),
                                )
                                with T.serial(0, num_k_blocks) as k_block_idx:
                                    IRBuilder.name("k_block_idx", k_block_idx)
                                    _builder_emit(
                                        barrier_wait(
                                            smem_barriers.ptr_to(
                                                [empty_barrier_base + pipeline_stage_idx]
                                            ),
                                            pipeline_phase ^ T.int32(1),
                                        )
                                    )
                                    n_idx = _builder_assign(
                                        "n_idx",
                                        local_expert_idx * shape_n
                                        + n_block_idx * kernel_config.block_n,
                                        locals().get("n_idx", _BUILDER_MISSING),
                                    )
                                    k_idx = _builder_assign(
                                        "k_idx",
                                        k_block_idx * kernel_config.block_k,
                                        locals().get("k_idx", _BUILDER_MISSING),
                                    )
                                    sfb_n_idx = _builder_assign(
                                        "sfb_n_idx",
                                        n_block_idx * kernel_config.block_n,
                                        locals().get("sfb_n_idx", _BUILDER_MISSING),
                                    )
                                    sfb_k_idx = _builder_assign(
                                        "sfb_k_idx",
                                        local_expert_idx * shape_sfb_k
                                        + k_block_idx * sf_smem_outer_dim,
                                        locals().get("sfb_k_idx", _BUILDER_MISSING),
                                    )
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            full_barrier_ptr = _builder_assign(
                                                "full_barrier_ptr",
                                                smem_barriers.ptr_to(
                                                    [full_barrier_base + pipeline_stage_idx]
                                                ),
                                                locals().get("full_barrier_ptr", _BUILDER_MISSING),
                                            )
                                            with T.unroll(
                                                0, kernel_config.block_k // umma_block_k
                                            ) as tma_k_atom_idx:
                                                IRBuilder.name("tma_k_atom_idx", tma_k_atom_idx)
                                                _builder_emit(
                                                    tma_copy_2d_multicast_select(
                                                        smem_b.ptr_to(
                                                            [
                                                                pipeline_stage_idx,
                                                                0,
                                                                tma_k_atom_idx * umma_block_k,
                                                            ]
                                                        ),
                                                        full_barrier_ptr,
                                                        tensor_map_l1_weights,
                                                        tensor_map_l2_weights,
                                                        block_phase,
                                                        k_idx + tma_k_atom_idx * umma_block_k,
                                                        n_idx,
                                                    )
                                                )
                                            _builder_emit(
                                                tma_copy_2d_multicast_select(
                                                    smem_sfb_i32.ptr_to([pipeline_stage_idx, 0, 0]),
                                                    full_barrier_ptr,
                                                    tensor_map_l1_weights_sf,
                                                    tensor_map_l2_weights_sf,
                                                    block_phase,
                                                    sfb_n_idx,
                                                    sfb_k_idx,
                                                )
                                            )
                                            with T.If(cta_idx_in_cluster == 0):
                                                with T.Then():
                                                    _builder_emit(
                                                        full_barrier_arrive_and_expect_tx(
                                                            full_barrier_ptr,
                                                            full_b_expect_tx_leader_bytes,
                                                        )
                                                    )
                                                with T.Else():
                                                    _builder_emit(
                                                        full_barrier_arrive_cta0(full_barrier_ptr)
                                                    )
                                    _builder_emit(T.cuda.warp_sync())
                                    _builder_emit(advance_pipeline())
                        with T.Else():
                            with T.If(flat_warp_idx == kernel_config.mma_issue_warp_idx):
                                with T.Then():
                                    _builder_emit(warpgroup_reg_dealloc(num_non_epilogue_registers))
                                    with T.If(cta_idx_in_cluster == 0):
                                        with T.Then():
                                            _builder_emit(make_instr_desc_block_scaled())
                                            _builder_emit(make_sf_desc())
                                            _builder_emit(make_umma_desc_a())
                                            _builder_emit(make_umma_desc_b())
                                            a_desc_lo = _builder_assign(
                                                "a_desc_lo",
                                                T.Select(
                                                    lane_idx < T.int32(num_stages),
                                                    T.cast(
                                                        T.bitwise_and(desc_a, T.uint64(0xFFFFFFFF)),
                                                        "uint32",
                                                    )
                                                    + T.cast(
                                                        lane_idx
                                                        * (smem_a_size_per_stage // f128_bytes),
                                                        "uint32",
                                                    ),
                                                    T.uint32(0),
                                                ),
                                                locals().get("a_desc_lo", _BUILDER_MISSING),
                                            )
                                            b_desc_lo = _builder_assign(
                                                "b_desc_lo",
                                                T.Select(
                                                    lane_idx < T.int32(num_stages),
                                                    T.cast(
                                                        T.bitwise_and(desc_b, T.uint64(0xFFFFFFFF)),
                                                        "uint32",
                                                    )
                                                    + T.cast(
                                                        lane_idx
                                                        * (smem_b_size_per_stage // f128_bytes),
                                                        "uint32",
                                                    ),
                                                    T.uint32(0),
                                                ),
                                                locals().get("b_desc_lo", _BUILDER_MISSING),
                                            )
                                            sched_stage_idx = _builder_assign(
                                                "sched_stage_idx",
                                                T.int32(0),
                                                locals().get("sched_stage_idx", _BUILDER_MISSING),
                                            )
                                            sched_phase = _builder_assign(
                                                "sched_phase",
                                                T.int32(0),
                                                locals().get("sched_phase", _BUILDER_MISSING),
                                            )
                                            current_iter_idx = _builder_assign(
                                                "current_iter_idx",
                                                0,
                                                locals().get("current_iter_idx", _BUILDER_MISSING),
                                            )
                                            pipeline_stage_idx = _builder_assign(
                                                "pipeline_stage_idx",
                                                T.int32(0),
                                                locals().get(
                                                    "pipeline_stage_idx", _BUILDER_MISSING
                                                ),
                                            )
                                            pipeline_phase = _builder_assign(
                                                "pipeline_phase",
                                                T.int32(0),
                                                locals().get("pipeline_phase", _BUILDER_MISSING),
                                            )
                                            with T.While(True):
                                                _builder_emit(consumer_get_next_task())
                                                _builder_emit(consumer_bind_task_args())
                                                with T.If(block_phase == T.int32(0)):
                                                    with T.Then():
                                                        T.evaluate(T.break_loop())
                                                num_k_blocks = _builder_assign(
                                                    "num_k_blocks",
                                                    T.cast(
                                                        T.cast(shape_k, "uint32")
                                                        // T.uint32(kernel_config.block_k),
                                                        "int32",
                                                    ),
                                                    locals().get("num_k_blocks", _BUILDER_MISSING),
                                                )
                                                current_iter_idx_u32 = _builder_assign(
                                                    "current_iter_idx_u32",
                                                    T.cast(current_iter_idx, "uint32"),
                                                    locals().get(
                                                        "current_iter_idx_u32", _BUILDER_MISSING
                                                    ),
                                                )
                                                accum_stage_idx = _builder_assign(
                                                    "accum_stage_idx",
                                                    T.cast(
                                                        current_iter_idx_u32
                                                        % T.uint32(num_epilogue_stages),
                                                        "int32",
                                                    ),
                                                    locals().get(
                                                        "accum_stage_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                accum_phase = _builder_assign(
                                                    "accum_phase",
                                                    T.cast(
                                                        T.bitwise_and(
                                                            current_iter_idx_u32
                                                            // T.uint32(num_epilogue_stages),
                                                            T.uint32(1),
                                                        ),
                                                        "int32",
                                                    ),
                                                    locals().get("accum_phase", _BUILDER_MISSING),
                                                )
                                                current_iter_idx = _builder_assign(
                                                    "current_iter_idx",
                                                    current_iter_idx + T.int32(1),
                                                    locals().get(
                                                        "current_iter_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                _builder_emit(update_get_valid_m_true())
                                                desc_i = _builder_assign(
                                                    "desc_i",
                                                    T.bitwise_or(
                                                        T.bitwise_and(desc_i, T.uint32(0xFF81FFFF)),
                                                        T.shift_left(
                                                            T.cast(
                                                                get_valid_m_true_eighth, "uint32"
                                                            ),
                                                            T.uint32(17),
                                                        ),
                                                    ),
                                                    locals().get("desc_i", _BUILDER_MISSING),
                                                )
                                                _builder_emit(
                                                    barrier_wait(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                tmem_empty_barrier_base
                                                                + accum_stage_idx
                                                            ]
                                                        ),
                                                        accum_phase ^ T.int32(1),
                                                    )
                                                )
                                                _builder_emit(
                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                )
                                                with T.serial(0, num_k_blocks) as k_block_idx:
                                                    IRBuilder.name("k_block_idx", k_block_idx)
                                                    full_wait_phase = _builder_scalar(
                                                        "full_wait_phase", pipeline_phase, "int32"
                                                    )
                                                    _builder_emit(
                                                        mbarrier_wait_phase(
                                                            smem_barriers.ptr_to(
                                                                [
                                                                    full_barrier_base
                                                                    + pipeline_stage_idx
                                                                ]
                                                            ),
                                                            full_wait_phase,
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.ptx.tcgen05.fence__after_thread_sync()
                                                    )
                                                    a_desc_base_lo = _builder_assign(
                                                        "a_desc_base_lo",
                                                        T.tvm_warp_shuffle(
                                                            T.uint32(0xFFFFFFFF),
                                                            a_desc_lo,
                                                            pipeline_stage_idx,
                                                            32,
                                                            32,
                                                        ),
                                                        locals().get(
                                                            "a_desc_base_lo", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    b_desc_base_lo = _builder_assign(
                                                        "b_desc_base_lo",
                                                        T.tvm_warp_shuffle(
                                                            T.uint32(0xFFFFFFFF),
                                                            b_desc_lo,
                                                            pipeline_stage_idx,
                                                            32,
                                                            32,
                                                        ),
                                                        locals().get(
                                                            "b_desc_base_lo", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    with T.If(T.cuda.elect_sync()):
                                                        with T.Then():
                                                            with T.unroll(
                                                                0,
                                                                kernel_config.block_k
                                                                // umma_block_k,
                                                            ) as umma_k_block_idx:
                                                                IRBuilder.name(
                                                                    "umma_k_block_idx",
                                                                    umma_k_block_idx,
                                                                )
                                                                with T.unroll(
                                                                    0, num_sfa_utccp_chunks
                                                                ) as sfa_chunk_idx:
                                                                    IRBuilder.name(
                                                                        "sfa_chunk_idx",
                                                                        sfa_chunk_idx,
                                                                    )
                                                                    desc_sf = _builder_assign(
                                                                        "desc_sf",
                                                                        replace_smem_desc_addr(
                                                                            desc_sf,
                                                                            smem_sfa.ptr_to(
                                                                                [
                                                                                    pipeline_stage_idx,
                                                                                    umma_k_block_idx
                                                                                    * sf_block_m
                                                                                    + sfa_chunk_idx
                                                                                    * 128,
                                                                                ]
                                                                            ),
                                                                        ),
                                                                        locals().get(
                                                                            "desc_sf",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        utccp_copy(
                                                                            sfa_tmem.allocated_addr[
                                                                                0
                                                                            ]
                                                                            + sfa_chunk_idx * 4,
                                                                            desc_sf,
                                                                        )
                                                                    )
                                                                with T.unroll(
                                                                    0, num_sfb_utccp_chunks
                                                                ) as sfb_chunk_idx:
                                                                    IRBuilder.name(
                                                                        "sfb_chunk_idx",
                                                                        sfb_chunk_idx,
                                                                    )
                                                                    desc_sf = _builder_assign(
                                                                        "desc_sf",
                                                                        replace_smem_desc_addr(
                                                                            desc_sf,
                                                                            smem_sfb.ptr_to(
                                                                                [
                                                                                    pipeline_stage_idx,
                                                                                    umma_k_block_idx
                                                                                    * kernel_config.block_n
                                                                                    + sfb_chunk_idx
                                                                                    * 128,
                                                                                ]
                                                                            ),
                                                                        ),
                                                                        locals().get(
                                                                            "desc_sf",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        utccp_copy(
                                                                            sfb_tmem.allocated_addr[
                                                                                0
                                                                            ]
                                                                            + sfb_chunk_idx * 4,
                                                                            desc_sf,
                                                                        )
                                                                    )
                                                                with T.unroll(
                                                                    0, umma_block_k // umma_k
                                                                ) as umma_k_idx:
                                                                    IRBuilder.name(
                                                                        "k_idx", umma_k_idx
                                                                    )
                                                                    runtime_desc_i = _builder_assign(
                                                                        "runtime_desc_i",
                                                                        make_runtime_instr_desc_with_sf_id(
                                                                            desc_i,
                                                                            umma_k_idx,
                                                                            umma_k_idx,
                                                                        ),
                                                                        locals().get(
                                                                            "runtime_desc_i",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    desc_a = _builder_assign(
                                                                        "desc_a",
                                                                        advance_umma_desc_lo(
                                                                            desc_a,
                                                                            a_desc_base_lo,
                                                                            umma_k_block_idx
                                                                            * umma_block_k
                                                                            * kernel_config.load_block_m,
                                                                            umma_k_idx * umma_k,
                                                                        ),
                                                                        locals().get(
                                                                            "desc_a",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    desc_b = _builder_assign(
                                                                        "desc_b",
                                                                        advance_umma_desc_lo(
                                                                            desc_b,
                                                                            b_desc_base_lo,
                                                                            umma_k_block_idx
                                                                            * umma_block_k
                                                                            * kernel_config.load_block_n,
                                                                            umma_k_idx * umma_k,
                                                                        ),
                                                                        locals().get(
                                                                            "desc_b",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    # kind::mxf8f6f4/scale_vec::1X from the
                                                                    # (f32, e2m1, e4m3, e8m0, e8m0) dtypes; the
                                                                    # instruction A slot carries desc_b (B^T*A).
                                                                    _builder_emit(
                                                                        T.ptx[
                                                                            f"tcgen05.mma.cta_group::{kernel_config.num_ctas_per_cluster}.kind::mxf8f6f4.block_scale.scale_vec::1X"
                                                                        ](
                                                                            T.cast(
                                                                                accum_stage_idx
                                                                                * umma_n,
                                                                                "uint32",
                                                                            ),
                                                                            desc_b,
                                                                            desc_a,
                                                                            runtime_desc_i,
                                                                            T.cast(
                                                                                sfb_tmem.allocated_addr[
                                                                                    0
                                                                                ],
                                                                                "uint32",
                                                                            ),
                                                                            T.cast(
                                                                                sfa_tmem.allocated_addr[
                                                                                    0
                                                                                ],
                                                                                "uint32",
                                                                            ),
                                                                            T.Or(
                                                                                k_block_idx
                                                                                > T.int32(0),
                                                                                T.Or(
                                                                                    umma_k_block_idx
                                                                                    > 0,
                                                                                    umma_k_idx > 0,
                                                                                ),
                                                                            ),
                                                                        )
                                                                    )
                                                                # The next UMMA sub-block overwrites these fixed
                                                                # scale-factor columns. MMA -> CP is not an
                                                                # implicit TCGEN pipeline, so order that reuse
                                                                # without draining the outer K-block pipeline.
                                                                with T.If(
                                                                    umma_k_block_idx + 1
                                                                    < (
                                                                        kernel_config.block_k
                                                                        // umma_block_k
                                                                    )
                                                                ):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            T.ptx.tcgen05.fence__before_thread_sync()
                                                                        )
                                                    _builder_emit(T.cuda.warp_sync())
                                                    _builder_emit(
                                                        empty_barrier_arrive_current(
                                                            k_block_idx == num_k_blocks - T.int32(1)
                                                        )
                                                    )
                                                    _builder_emit(advance_pipeline())
                                            with T.If(current_iter_idx > 0):
                                                with T.Then():
                                                    previous_iter_idx_u32 = _builder_assign(
                                                        "previous_iter_idx_u32",
                                                        T.cast(
                                                            current_iter_idx - T.int32(1), "uint32"
                                                        ),
                                                        locals().get(
                                                            "previous_iter_idx_u32",
                                                            _BUILDER_MISSING,
                                                        ),
                                                    )
                                                    accum_phase = _builder_assign(
                                                        "accum_phase",
                                                        T.cast(
                                                            T.bitwise_and(
                                                                previous_iter_idx_u32
                                                                // T.uint32(num_epilogue_stages),
                                                                T.uint32(1),
                                                            ),
                                                            "int32",
                                                        ),
                                                        locals().get(
                                                            "accum_phase", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    _builder_emit(
                                                        barrier_wait(
                                                            smem_barriers.ptr_to(
                                                                [
                                                                    tmem_empty_barrier_base
                                                                    + T.cast(
                                                                        previous_iter_idx_u32
                                                                        % T.uint32(
                                                                            num_epilogue_stages
                                                                        ),
                                                                        "int32",
                                                                    )
                                                                ]
                                                            ),
                                                            accum_phase,
                                                        )
                                                    )
                                with T.Else():
                                    with T.If(is_reserved_non_epilogue_warp):
                                        with T.Then():
                                            _builder_emit(
                                                warpgroup_reg_dealloc(num_non_epilogue_registers)
                                            )
                                            # Task scheduler mainloop, run by the leader CTA only
                                            with T.If(cta_idx_in_cluster == 0):
                                                with T.Then():
                                                    sched_stage_idx = _builder_assign(
                                                        "sched_stage_idx",
                                                        T.int32(0),
                                                        locals().get(
                                                            "sched_stage_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    sched_phase = _builder_assign(
                                                        "sched_phase",
                                                        T.int32(0),
                                                        locals().get(
                                                            "sched_phase", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    # Wait dispatch's results
                                                    _builder_emit(
                                                        scheduler_fetch_expert_recv_count()
                                                    )
                                                    # Generate routed tasks. Keep the original wait -> claim -> publish
                                                    # ordering: `get_next_task()` advances global task counters and must
                                                    # not run before the schedule slot is released by consumers.
                                                    sched_task_valid = _builder_assign(
                                                        "sched_task_valid",
                                                        T.int32(1),
                                                        locals().get(
                                                            "sched_task_valid", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    with T.While(sched_task_valid != T.int32(0)):
                                                        _builder_emit(
                                                            barrier_wait(
                                                                smem_barriers.ptr_to(
                                                                    [
                                                                        task_info_empty_barrier_base
                                                                        + sched_stage_idx
                                                                    ]
                                                                ),
                                                                sched_phase ^ T.int32(1),
                                                            )
                                                        )
                                                        _builder_emit(producer_get_next_task())
                                                        with T.If(sched_task_valid != T.int32(0)):
                                                            with T.Then():
                                                                _builder_emit(
                                                                    producer_publish_task()
                                                                )
                                                    # Sentinel
                                                    _builder_emit(
                                                        barrier_wait(
                                                            smem_barriers.ptr_to(
                                                                [
                                                                    task_info_empty_barrier_base
                                                                    + sched_stage_idx
                                                                ]
                                                            ),
                                                            sched_phase ^ T.int32(1),
                                                        )
                                                    )
                                                    with T.unroll(0, 8) as task_reg_idx:
                                                        IRBuilder.name("task_reg_idx", task_reg_idx)
                                                        T.buffer_store(
                                                            task_info_regs,
                                                            T.uint32(0),
                                                            [task_reg_idx],
                                                        )
                                                    _builder_emit(producer_publish_task())
                                        with T.Else():
                                            with T.If(
                                                T.cast(flat_warp_idx, "uint32")
                                                >= T.uint32(kernel_config.epilogue_warp_start_idx)
                                            ):
                                                with T.Then():
                                                    _builder_emit(
                                                        warpgroup_reg_alloc(num_epilogue_registers)
                                                    )
                                                    activation_values = _builder_assign(
                                                        "activation_values",
                                                        T.alloc_local(
                                                            (num_atoms_per_store, 2, 2), "float32"
                                                        ),
                                                        locals().get(
                                                            "activation_values", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    amax_values = _builder_assign(
                                                        "amax_values",
                                                        T.alloc_local(
                                                            (num_atoms_per_store, 2), "float32"
                                                        ),
                                                        locals().get(
                                                            "amax_values", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    thread_local_amax = _builder_assign(
                                                        "thread_local_amax",
                                                        T.alloc_local((2,), "float32"),
                                                        locals().get(
                                                            "thread_local_amax", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    values = _builder_assign(
                                                        "values",
                                                        T.alloc_local((8,), "uint32"),
                                                        locals().get("values", _BUILDER_MISSING),
                                                    )
                                                    epilogue_fp8_packed = _builder_assign(
                                                        "epilogue_fp8_packed",
                                                        T.alloc_local((1,), "uint32"),
                                                        locals().get(
                                                            "epilogue_fp8_packed", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    weights = _builder_assign(
                                                        "weights",
                                                        T.alloc_local((2,), "float32"),
                                                        locals().get("weights", _BUILDER_MISSING),
                                                    )
                                                    wp_amax = _builder_assign(
                                                        "wp_amax",
                                                        T.alloc_local((2,), "float32"),
                                                        locals().get("wp_amax", _BUILDER_MISSING),
                                                    )
                                                    sf = _builder_assign(
                                                        "sf",
                                                        T.alloc_local((2,), "float32"),
                                                        locals().get("sf", _BUILDER_MISSING),
                                                    )
                                                    sf_inv = _builder_assign(
                                                        "sf_inv",
                                                        T.alloc_local((2,), "float32"),
                                                        locals().get("sf_inv", _BUILDER_MISSING),
                                                    )
                                                    scaled_pair = _builder_assign(
                                                        "scaled_pair",
                                                        T.alloc_local((1,), "uint64"),
                                                        locals().get(
                                                            "scaled_pair", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    scaled_values = _builder_assign(
                                                        "scaled_values",
                                                        T.alloc_local((2,), "float32"),
                                                        locals().get(
                                                            "scaled_values", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    scaled_bits = _builder_assign(
                                                        "scaled_bits",
                                                        T.alloc_local((2,), "uint32"),
                                                        locals().get(
                                                            "scaled_bits", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    scale_exponents = _builder_assign(
                                                        "scale_exponents",
                                                        T.alloc_local((2,), "int32"),
                                                        locals().get(
                                                            "scale_exponents", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    scaled_upper = _builder_assign(
                                                        "scaled_upper",
                                                        T.alloc_local((1,), "uint64"),
                                                        locals().get(
                                                            "scaled_upper", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    scaled_lower = _builder_assign(
                                                        "scaled_lower",
                                                        T.alloc_local((1,), "uint64"),
                                                        locals().get(
                                                            "scaled_lower", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    epilogue_bf16_packed = _builder_assign(
                                                        "epilogue_bf16_packed",
                                                        T.alloc_local((4,), "uint32"),
                                                        locals().get(
                                                            "epilogue_bf16_packed", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    tmem_addr = _builder_assign(
                                                        "tmem_addr",
                                                        T.alloc_local((1,), "uint32"),
                                                        locals().get("tmem_addr", _BUILDER_MISSING),
                                                    )
                                                    reduced = _builder_assign(
                                                        "reduced",
                                                        T.alloc_local(
                                                            (
                                                                num_uint4_per_lane
                                                                * num_elems_per_uint4,
                                                                2,
                                                            ),
                                                            "float32",
                                                        ),
                                                        locals().get("reduced", _BUILDER_MISSING),
                                                    )
                                                    _builder_emit(
                                                        load_shared_u32(
                                                            tmem_allocated,
                                                            tmem_ptr_in_smem.ptr_to([0]),
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.cuda.trap_when_assert_failed(
                                                            tmem_allocated == T.uint32(0)
                                                        )
                                                    )
                                                    epilogue_warp_idx = _builder_assign(
                                                        "epilogue_warp_idx",
                                                        flat_warp_idx
                                                        - kernel_config.epilogue_warp_start_idx,
                                                        locals().get(
                                                            "epilogue_warp_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    epilogue_warp_idx_u32 = _builder_assign(
                                                        "epilogue_warp_idx_u32",
                                                        T.cast(epilogue_warp_idx, "uint32"),
                                                        locals().get(
                                                            "epilogue_warp_idx_u32",
                                                            _BUILDER_MISSING,
                                                        ),
                                                    )
                                                    epilogue_wg_idx = _builder_assign(
                                                        "epilogue_wg_idx",
                                                        T.cast(
                                                            epilogue_warp_idx_u32 // T.uint32(4),
                                                            "int32",
                                                        ),
                                                        locals().get(
                                                            "epilogue_wg_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    warp_idx_in_wg = _builder_assign(
                                                        "warp_idx_in_wg",
                                                        T.cast(
                                                            epilogue_warp_idx_u32 % T.uint32(4),
                                                            "int32",
                                                        ),
                                                        locals().get(
                                                            "warp_idx_in_wg", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    _builder_emit(
                                                        T.evaluate(
                                                            sync_unaligned(
                                                                dispatch_with_epilogue_sync_barrier_idx,
                                                                kernel_config.num_dispatch_threads
                                                                + kernel_config.num_epilogue_threads,
                                                            )
                                                        )
                                                    )
                                                    sched_stage_idx = _builder_assign(
                                                        "sched_stage_idx",
                                                        T.int32(0),
                                                        locals().get(
                                                            "sched_stage_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    sched_phase = _builder_assign(
                                                        "sched_phase",
                                                        T.int32(0),
                                                        locals().get(
                                                            "sched_phase", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    current_iter_idx = _builder_assign(
                                                        "current_iter_idx",
                                                        0,
                                                        locals().get(
                                                            "current_iter_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    with T.While(True):
                                                        _builder_emit(consumer_get_next_task())
                                                        _builder_emit(consumer_bind_task_args())
                                                        with T.If(block_phase == T.int32(0)):
                                                            with T.Then():
                                                                T.evaluate(T.break_loop())
                                                        current_iter_idx_u32 = _builder_assign(
                                                            "current_iter_idx_u32",
                                                            T.cast(current_iter_idx, "uint32"),
                                                            locals().get(
                                                                "current_iter_idx_u32",
                                                                _BUILDER_MISSING,
                                                            ),
                                                        )
                                                        accum_stage_idx = _builder_assign(
                                                            "accum_stage_idx",
                                                            T.cast(
                                                                current_iter_idx_u32
                                                                % T.uint32(num_epilogue_stages),
                                                                "int32",
                                                            ),
                                                            locals().get(
                                                                "accum_stage_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        accum_phase = _builder_assign(
                                                            "accum_phase",
                                                            T.cast(
                                                                T.bitwise_and(
                                                                    current_iter_idx_u32
                                                                    // T.uint32(
                                                                        num_epilogue_stages
                                                                    ),
                                                                    T.uint32(1),
                                                                ),
                                                                "int32",
                                                            ),
                                                            locals().get(
                                                                "accum_phase", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        current_iter_idx = _builder_assign(
                                                            "current_iter_idx",
                                                            current_iter_idx + T.int32(1),
                                                            locals().get(
                                                                "current_iter_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        _builder_emit(
                                                            barrier_wait(
                                                                smem_barriers.ptr_to(
                                                                    [
                                                                        tmem_full_barrier_base
                                                                        + accum_stage_idx
                                                                    ]
                                                                ),
                                                                accum_phase,
                                                            )
                                                        )
                                                        _builder_emit(
                                                            T.ptx.tcgen05.fence__after_thread_sync()
                                                        )
                                                        # Now we can release the task
                                                        _builder_emit(scheduler_release_task_info())
                                                        # Match DeepGEMM's `ptx::exchange(..., 0)`: all lanes have the same
                                                        # scheduler result, but broadcasting it lets the CUDA compiler treat
                                                        # the valid-row early exit as warp-uniform instead of divergent.
                                                        valid_m = _builder_assign(
                                                            "valid_m",
                                                            T.tvm_warp_shuffle(
                                                                T.uint32(0xFFFFFFFF),
                                                                valid_m,
                                                                T.int32(0),
                                                                32,
                                                                32,
                                                            ),
                                                            locals().get(
                                                                "valid_m", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ring_block_idx = _builder_assign(
                                                            "ring_block_idx",
                                                            pool_block_idx % num_ring_blocks,
                                                            locals().get(
                                                                "ring_block_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        ring_m_idx = _builder_assign(
                                                            "ring_m_idx",
                                                            ring_block_idx * kernel_config.block_m,
                                                            locals().get(
                                                                "ring_m_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        pool_m_idx = _builder_assign(
                                                            "pool_m_idx",
                                                            pool_block_idx * kernel_config.block_m,
                                                            locals().get(
                                                                "pool_m_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        n_block_idx = _builder_assign(
                                                            "n_block_idx",
                                                            n_cluster_idx * 2
                                                            + T.Select(
                                                                cta_idx_in_cluster == 0,
                                                                T.int32(0),
                                                                T.int32(1),
                                                            ),
                                                            locals().get(
                                                                "n_block_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        valid_rows_in_wg = _builder_assign(
                                                            "valid_rows_in_wg",
                                                            T.max(
                                                                T.min(
                                                                    valid_m
                                                                    - epilogue_wg_idx * wg_block_m,
                                                                    wg_block_m,
                                                                ),
                                                                T.int32(0),
                                                            ),
                                                            locals().get(
                                                                "valid_rows_in_wg", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(block_phase == T.int32(1)):
                                                            with T.Then():
                                                                expected_ring_count = (
                                                                    _builder_assign(
                                                                        "expected_ring_count",
                                                                        (
                                                                            hidden
                                                                            // kernel_config.block_n
                                                                            * (
                                                                                pool_block_idx
                                                                                // num_ring_blocks
                                                                            )
                                                                        ),
                                                                        locals().get(
                                                                            "expected_ring_count",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    load_acq_u32(
                                                                        current_ring_count,
                                                                        workspace_l2_empty_count.ptr_to(
                                                                            [ring_block_idx]
                                                                        ),
                                                                    )
                                                                )
                                                                with T.While(
                                                                    current_ring_count
                                                                    != T.cast(
                                                                        expected_ring_count,
                                                                        "uint32",
                                                                    )
                                                                ):
                                                                    _builder_emit(
                                                                        load_acq_u32(
                                                                            current_ring_count,
                                                                            workspace_l2_empty_count.ptr_to(
                                                                                [ring_block_idx]
                                                                            ),
                                                                        )
                                                                    )
                                                                n_idx = _builder_assign(
                                                                    "n_idx",
                                                                    n_block_idx * l1_out_block_n,
                                                                    locals().get(
                                                                        "n_idx", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                # Declared outside the `for s` loop so the per-32-rows weight cache persists
                                                                # across store iters. When wg_block_m is not a multiple of 32 (e.g., 48 for
                                                                # block_m=96) the load gate `(j*atom_m) % 32 == 0` fires at a different
                                                                # rhythm than the s loop, so resetting per-s would leave the cache zero on
                                                                # iterations between loads. Upstream 559d79f defaults the weight
                                                                # to 1.0f (weightless shared-expert path multiplies by 1).
                                                                stored_cached_weight = (
                                                                    _builder_assign(
                                                                        "stored_cached_weight",
                                                                        T.float32(1.0),
                                                                        locals().get(
                                                                            "stored_cached_weight",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                )
                                                                with T.serial(
                                                                    0,
                                                                    wg_block_m
                                                                    // kernel_config.store_block_m,
                                                                    unroll=True,
                                                                ) as s:
                                                                    IRBuilder.name("s", s)
                                                                    with T.If(
                                                                        s
                                                                        * kernel_config.store_block_m
                                                                        >= valid_rows_in_wg
                                                                    ):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                tmem_empty_barrier_arrive_cta0(
                                                                                    smem_barriers.ptr_to(
                                                                                        [
                                                                                            tmem_empty_barrier_base
                                                                                            + accum_stage_idx
                                                                                        ]
                                                                                    )
                                                                                )
                                                                            )
                                                                            T.evaluate(
                                                                                T.break_loop()
                                                                            )
                                                                    with T.unroll(
                                                                        0, num_atoms_per_store
                                                                    ) as i:
                                                                        IRBuilder.name("i", i)
                                                                        j = _builder_assign(
                                                                            "j",
                                                                            s * num_atoms_per_store
                                                                            + i,
                                                                            locals().get(
                                                                                "j",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(
                                                                            (j * atom_m) % 32 == 0
                                                                        ):
                                                                            with T.Then():
                                                                                # Lanes whose row falls past wg_block_m must skip the load — the
                                                                                # warp-shuffle source lanes below are always < wg_block_m, so leaving
                                                                                # OOB lanes' stored_cached_weight stale is fine. (Matches upstream's
                                                                                # runtime guard for non-32-aligned wg_block_m.)
                                                                                if (
                                                                                    wg_block_m % 32
                                                                                    == 0
                                                                                ):
                                                                                    l1_topk_weight_ptr = _builder_assign(
                                                                                        "l1_topk_weight_ptr",
                                                                                        l1_topk_weights.ptr_to(
                                                                                            [
                                                                                                ring_m_idx
                                                                                                + epilogue_wg_idx
                                                                                                * wg_block_m
                                                                                                + j
                                                                                                * atom_m
                                                                                                + lane_idx
                                                                                            ]
                                                                                        ),
                                                                                        locals().get(
                                                                                            "l1_topk_weight_ptr",
                                                                                            _BUILDER_MISSING,
                                                                                        ),
                                                                                    )
                                                                                    _builder_emit(
                                                                                        load_f32(
                                                                                            stored_cached_weight,
                                                                                            l1_topk_weight_ptr,
                                                                                        )
                                                                                    )
                                                                                else:
                                                                                    with T.If(
                                                                                        T.cast(
                                                                                            j
                                                                                            * atom_m
                                                                                            + lane_idx,
                                                                                            "uint32",
                                                                                        )
                                                                                        < T.uint32(
                                                                                            wg_block_m
                                                                                        )
                                                                                    ):
                                                                                        with (
                                                                                            T.Then()
                                                                                        ):
                                                                                            l1_topk_weight_ptr = _builder_assign(
                                                                                                "l1_topk_weight_ptr",
                                                                                                l1_topk_weights.ptr_to(
                                                                                                    [
                                                                                                        ring_m_idx
                                                                                                        + epilogue_wg_idx
                                                                                                        * wg_block_m
                                                                                                        + j
                                                                                                        * atom_m
                                                                                                        + lane_idx
                                                                                                    ]
                                                                                                ),
                                                                                                locals().get(
                                                                                                    "l1_topk_weight_ptr",
                                                                                                    _BUILDER_MISSING,
                                                                                                ),
                                                                                            )
                                                                                            _builder_emit(
                                                                                                load_f32(
                                                                                                    stored_cached_weight,
                                                                                                    l1_topk_weight_ptr,
                                                                                                )
                                                                                            )
                                                                        T.buffer_store(
                                                                            weights,
                                                                            T.tvm_warp_shuffle(
                                                                                T.uint32(
                                                                                    0xFFFFFFFF
                                                                                ),
                                                                                stored_cached_weight,
                                                                                (j * atom_m) % 32
                                                                                + (lane_idx % 4)
                                                                                * 2,
                                                                                32,
                                                                                32,
                                                                            ),
                                                                            [0],
                                                                        )
                                                                        T.buffer_store(
                                                                            weights,
                                                                            T.tvm_warp_shuffle(
                                                                                T.uint32(
                                                                                    0xFFFFFFFF
                                                                                ),
                                                                                stored_cached_weight,
                                                                                (j * atom_m) % 32
                                                                                + (lane_idx % 4) * 2
                                                                                + 1,
                                                                                32,
                                                                                32,
                                                                            ),
                                                                            [1],
                                                                        )
                                                                        T.buffer_store(
                                                                            tmem_addr,
                                                                            T.cast(
                                                                                accum_stage_idx
                                                                                * umma_n
                                                                                + epilogue_wg_idx
                                                                                * wg_block_m
                                                                                + j * atom_m,
                                                                                "uint32",
                                                                            ),
                                                                            [0],
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx[
                                                                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                                                            ](
                                                                                values[0],
                                                                                values[1],
                                                                                values[2],
                                                                                values[3],
                                                                                T.uint32(
                                                                                    tmem_addr[0]
                                                                                ),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx[
                                                                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                                                            ](
                                                                                values[4],
                                                                                values[5],
                                                                                values[6],
                                                                                values[7],
                                                                                T.uint32(
                                                                                    T.bitwise_or(
                                                                                        tmem_addr[
                                                                                            0
                                                                                        ],
                                                                                        T.uint32(
                                                                                            0x00100000
                                                                                        ),
                                                                                    )
                                                                                ),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            fence_view_async_tmem_load()
                                                                        )
                                                                        with T.If(
                                                                            j
                                                                            == wg_block_m // atom_m
                                                                            - 1
                                                                        ):
                                                                            with T.Then():
                                                                                _builder_emit(
                                                                                    tmem_empty_barrier_arrive_cta0(
                                                                                        smem_barriers.ptr_to(
                                                                                            [
                                                                                                tmem_empty_barrier_base
                                                                                                + accum_stage_idx
                                                                                            ]
                                                                                        )
                                                                                    )
                                                                                )
                                                                        with T.unroll(0, 2) as k:
                                                                            IRBuilder.name("k", k)
                                                                            _builder_emit(
                                                                                activation_pair_store(
                                                                                    activation_values,
                                                                                    i,
                                                                                    k,
                                                                                    uint32_bits_to_float(
                                                                                        values[
                                                                                            k * 4
                                                                                        ]
                                                                                    ),
                                                                                    uint32_bits_to_float(
                                                                                        values[
                                                                                            k * 4
                                                                                            + 1
                                                                                        ]
                                                                                    ),
                                                                                    uint32_bits_to_float(
                                                                                        values[
                                                                                            k * 4
                                                                                            + 2
                                                                                        ]
                                                                                    ),
                                                                                    uint32_bits_to_float(
                                                                                        values[
                                                                                            k * 4
                                                                                            + 3
                                                                                        ]
                                                                                    ),
                                                                                    weights[0],
                                                                                    weights[1],
                                                                                )
                                                                            )
                                                                        T.buffer_store(
                                                                            thread_local_amax,
                                                                            T.float32(0.0),
                                                                            [0],
                                                                        )
                                                                        T.buffer_store(
                                                                            thread_local_amax,
                                                                            T.float32(0.0),
                                                                            [1],
                                                                        )
                                                                        with T.unroll(0, 2) as k:
                                                                            IRBuilder.name("k", k)
                                                                            T.buffer_store(
                                                                                thread_local_amax,
                                                                                T.max(
                                                                                    thread_local_amax[
                                                                                        0
                                                                                    ],
                                                                                    T.fabs(
                                                                                        activation_values[
                                                                                            i, k, 0
                                                                                        ]
                                                                                    ),
                                                                                ),
                                                                                [0],
                                                                            )
                                                                            T.buffer_store(
                                                                                thread_local_amax,
                                                                                T.max(
                                                                                    thread_local_amax[
                                                                                        1
                                                                                    ],
                                                                                    T.fabs(
                                                                                        activation_values[
                                                                                            i, k, 1
                                                                                        ]
                                                                                    ),
                                                                                ),
                                                                                [1],
                                                                            )
                                                                        T.buffer_store(
                                                                            amax_values,
                                                                            thread_local_amax[0],
                                                                            [i, 0],
                                                                        )
                                                                        T.buffer_store(
                                                                            amax_values,
                                                                            thread_local_amax[1],
                                                                            [i, 1],
                                                                        )
                                                                        _builder_emit(
                                                                            warp_reduce_max_4(
                                                                                amax_values, i, 0
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            warp_reduce_max_4(
                                                                                amax_values, i, 1
                                                                            )
                                                                        )
                                                                        with T.If(lane_idx < 4):
                                                                            with T.Then():
                                                                                amax_reduction_idx = _builder_assign(
                                                                                    "amax_reduction_idx",
                                                                                    (
                                                                                        epilogue_warp_idx
                                                                                        * (
                                                                                            kernel_config.store_block_m
                                                                                            // 2
                                                                                        )
                                                                                        + i
                                                                                        * (
                                                                                            atom_m
                                                                                            // 2
                                                                                        )
                                                                                        + lane_idx
                                                                                    )
                                                                                    * 2,
                                                                                    locals().get(
                                                                                        "amax_reduction_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    T.ptx.st.shared.v2.f32(
                                                                                        smem_amax_reduction.ptr_to(
                                                                                            [
                                                                                                amax_reduction_idx
                                                                                            ]
                                                                                        ),
                                                                                        amax_values[
                                                                                            i, 0
                                                                                        ],
                                                                                        amax_values[
                                                                                            i, 1
                                                                                        ],
                                                                                    )
                                                                                )
                                                                        _builder_emit(
                                                                            T.cuda.warp_sync()
                                                                        )
                                                                    tma_stage_idx = _builder_assign(
                                                                        "tma_stage_idx",
                                                                        s % num_tma_store_stages,
                                                                        locals().get(
                                                                            "tma_stage_idx",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            tma_store_wait(1)
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                epilogue_wg_sync_barrier_start_idx
                                                                                + epilogue_wg_idx
                                                                            ),
                                                                            128,
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        0, num_atoms_per_store
                                                                    ) as i:
                                                                        IRBuilder.name("i", i)
                                                                        j = _builder_assign(
                                                                            "j",
                                                                            s * num_atoms_per_store
                                                                            + i,
                                                                            locals().get(
                                                                                "j",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        amax_reduction_idx = _builder_assign(
                                                                            "amax_reduction_idx",
                                                                            (
                                                                                (
                                                                                    epilogue_warp_idx
                                                                                    ^ 1
                                                                                )
                                                                                * (
                                                                                    kernel_config.store_block_m
                                                                                    // 2
                                                                                )
                                                                                + i * (atom_m // 2)
                                                                                + (lane_idx % 4)
                                                                            )
                                                                            * 2,
                                                                            locals().get(
                                                                                "amax_reduction_idx",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx.ld.shared.v2.f32(
                                                                                wp_amax[0],
                                                                                wp_amax[1],
                                                                                smem_amax_reduction.ptr_to(
                                                                                    [
                                                                                        amax_reduction_idx
                                                                                    ]
                                                                                ),
                                                                            )
                                                                        )
                                                                        T.buffer_store(
                                                                            amax_values,
                                                                            T.max(
                                                                                amax_values[i, 0],
                                                                                wp_amax[0],
                                                                            ),
                                                                            [i, 0],
                                                                        )
                                                                        T.buffer_store(
                                                                            amax_values,
                                                                            T.max(
                                                                                amax_values[i, 1],
                                                                                wp_amax[1],
                                                                            ),
                                                                            [i, 1],
                                                                        )
                                                                        _builder_emit(
                                                                            get_e4m3_sf_and_sf_inv(
                                                                                sf,
                                                                                sf_inv,
                                                                                scaled_pair,
                                                                                scaled_values,
                                                                                scaled_bits,
                                                                                scale_exponents,
                                                                                amax_values[i, 0],
                                                                                amax_values[i, 1],
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            scale_pack_fp8x4_e4m3(
                                                                                epilogue_fp8_packed,
                                                                                scaled_upper,
                                                                                scaled_lower,
                                                                                activation_values[
                                                                                    i, 0, 0
                                                                                ],
                                                                                activation_values[
                                                                                    i, 0, 1
                                                                                ],
                                                                                activation_values[
                                                                                    i, 1, 0
                                                                                ],
                                                                                activation_values[
                                                                                    i, 1, 1
                                                                                ],
                                                                                sf_inv[0],
                                                                                sf_inv[1],
                                                                            )
                                                                        )
                                                                        row = _builder_scalar(
                                                                            "row", lane_idx, "int32"
                                                                        )
                                                                        col = _builder_assign(
                                                                            "col",
                                                                            warp_idx_in_wg,
                                                                            locals().get(
                                                                                "col",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        smem_ptr = _builder_assign(
                                                                            "smem_ptr",
                                                                            smem_cd_l1.ptr_to(
                                                                                [
                                                                                    tma_stage_idx,
                                                                                    epilogue_wg_idx,
                                                                                    i * atom_m
                                                                                    + row,
                                                                                    (
                                                                                        col
                                                                                        ^ (row // 2)
                                                                                    )
                                                                                    * num_bank_group_bytes,
                                                                                ]
                                                                            ),
                                                                            locals().get(
                                                                                "smem_ptr",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            sm100_u8x4_stsm_t_copy(
                                                                                epilogue_fp8_packed[
                                                                                    0
                                                                                ],
                                                                                smem_ptr,
                                                                            )
                                                                        )
                                                                        with T.If(
                                                                            T.And(
                                                                                warp_idx_in_wg % 2
                                                                                == 0,
                                                                                lane_idx < 4,
                                                                            )
                                                                        ):
                                                                            with T.Then():
                                                                                # Factored form of upstream 891d57b: token_base_idx is < BLOCK_M so
                                                                                # `m_block_idx * BLOCK_M` factors out as `m_block_idx * SF_BLOCK_M`
                                                                                # past `transform_sf_token_idx` (which is bitwise-independent in
                                                                                # that range). `lane_idx * 2` only touches bits 0..2 of the input
                                                                                # (token_base_idx is a multiple of atom_m=8), so its contribution
                                                                                # collapses to a constant `lane_idx * 8` (= `(lane_idx*2) << 2`).
                                                                                # Eliminates one mul + the residual modulo work in the original
                                                                                # composed form.
                                                                                token_base_idx = _builder_assign(
                                                                                    "token_base_idx",
                                                                                    (
                                                                                        epilogue_wg_idx
                                                                                        * wg_block_m
                                                                                        + s
                                                                                        * kernel_config.store_block_m
                                                                                        + i * atom_m
                                                                                    ),
                                                                                    locals().get(
                                                                                        "token_base_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                sf_pool_token_idx = _builder_assign(
                                                                                    "sf_pool_token_idx",
                                                                                    (
                                                                                        T.cast(
                                                                                            ring_block_idx,
                                                                                            "uint64",
                                                                                        )
                                                                                        * T.uint64(
                                                                                            sf_block_m
                                                                                        )
                                                                                        + T.cast(
                                                                                            transform_sf_token_idx(
                                                                                                token_base_idx
                                                                                            ),
                                                                                            "uint64",
                                                                                        )
                                                                                        + T.cast(
                                                                                            lane_idx,
                                                                                            "uint64",
                                                                                        )
                                                                                        * T.uint64(
                                                                                            8
                                                                                        )
                                                                                    ),
                                                                                    locals().get(
                                                                                        "sf_pool_token_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                mn_stride = _builder_assign(
                                                                                    "mn_stride",
                                                                                    T.uint64(
                                                                                        workspace_layout.num_sf_ring_tokens
                                                                                        * 4
                                                                                    ),
                                                                                    locals().get(
                                                                                        "mn_stride",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                k_idx = _builder_assign(
                                                                                    "k_idx",
                                                                                    n_block_idx * 2
                                                                                    + warp_idx_in_wg
                                                                                    // 2,
                                                                                    locals().get(
                                                                                        "k_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                k_uint_idx = _builder_assign(
                                                                                    "k_uint_idx",
                                                                                    k_idx // 4,
                                                                                    locals().get(
                                                                                        "k_uint_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                byte_idx = _builder_assign(
                                                                                    "byte_idx",
                                                                                    k_idx % 4,
                                                                                    locals().get(
                                                                                        "byte_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                sf_addr = _builder_assign(
                                                                                    "sf_addr",
                                                                                    (
                                                                                        T.cast(
                                                                                            k_uint_idx,
                                                                                            "uint64",
                                                                                        )
                                                                                        * mn_stride
                                                                                        + sf_pool_token_idx
                                                                                        * T.uint64(
                                                                                            4
                                                                                        )
                                                                                        + T.cast(
                                                                                            byte_idx,
                                                                                            "uint64",
                                                                                        )
                                                                                    ),
                                                                                    locals().get(
                                                                                        "sf_addr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                sf_bits = _builder_assign(
                                                                                    "sf_bits",
                                                                                    float_bits(
                                                                                        sf[0]
                                                                                    ),
                                                                                    locals().get(
                                                                                        "sf_bits",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                sf_bits_hi = _builder_assign(
                                                                                    "sf_bits_hi",
                                                                                    float_bits(
                                                                                        sf[1]
                                                                                    ),
                                                                                    locals().get(
                                                                                        "sf_bits_hi",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    store_global_u8(
                                                                                        l2_sf_buffer.ptr_to(
                                                                                            [
                                                                                                sf_addr
                                                                                            ]
                                                                                        ),
                                                                                        T.cast(
                                                                                            T.shift_right(
                                                                                                sf_bits,
                                                                                                T.uint32(
                                                                                                    23
                                                                                                ),
                                                                                            ),
                                                                                            "uint8",
                                                                                        ),
                                                                                    )
                                                                                )
                                                                                _builder_emit(
                                                                                    store_global_u8(
                                                                                        l2_sf_buffer.ptr_to(
                                                                                            [
                                                                                                sf_addr
                                                                                                + T.uint64(
                                                                                                    16
                                                                                                )
                                                                                            ]
                                                                                        ),
                                                                                        T.cast(
                                                                                            T.shift_right(
                                                                                                sf_bits_hi,
                                                                                                T.uint32(
                                                                                                    23
                                                                                                ),
                                                                                            ),
                                                                                            "uint8",
                                                                                        ),
                                                                                    )
                                                                                )
                                                                    _builder_emit(
                                                                        T.cuda.warp_sync()
                                                                    )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                epilogue_wg_sync_barrier_start_idx
                                                                                + epilogue_wg_idx
                                                                            ),
                                                                            128,
                                                                        )
                                                                    )
                                                                    with T.If(
                                                                        (warp_idx_in_wg == 0)
                                                                        & T.cuda.elect_sync()
                                                                        != 0
                                                                    ):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    tma_store_fence()
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                sm90_tma_store_2d_copy(
                                                                                    smem_cd_l1.ptr_to(
                                                                                        [
                                                                                            tma_stage_idx,
                                                                                            epilogue_wg_idx,
                                                                                            0,
                                                                                            0,
                                                                                        ]
                                                                                    ),
                                                                                    tensor_map_l1_output,
                                                                                    n_idx,
                                                                                    ring_m_idx
                                                                                    + epilogue_wg_idx
                                                                                    * wg_block_m
                                                                                    + s
                                                                                    * kernel_config.store_block_m,
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                T.evaluate(
                                                                                    tma_store_arrive()
                                                                                )
                                                                            )
                                                                    _builder_emit(
                                                                        T.cuda.warp_sync()
                                                                    )
                                                                _builder_emit(
                                                                    T.evaluate(tma_store_wait(0))
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.bar.sync(
                                                                        T.uint32(
                                                                            epilogue_full_sync_barrier_idx
                                                                        ),
                                                                        T.uint32(
                                                                            kernel_config.num_epilogue_threads
                                                                        ),
                                                                    )
                                                                )
                                                                with T.If(
                                                                    (epilogue_warp_idx == 0)
                                                                    & T.cuda.elect_sync()
                                                                    != 0
                                                                ):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            atomic_add_rel_u32(
                                                                                atom_prev_unused,
                                                                                workspace_l2_full_count.ptr_to(
                                                                                    [ring_block_idx]
                                                                                ),
                                                                                T.uint32(1),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.evaluate(
                                                                                red_add_gpu_u32(
                                                                                    workspace_l1_empty_count.ptr_to(
                                                                                        [
                                                                                            ring_block_idx
                                                                                        ]
                                                                                    ),
                                                                                    T.uint32(1),
                                                                                )
                                                                            )
                                                                        )
                                                                _builder_emit(T.cuda.warp_sync())
                                                            with T.Else():
                                                                with T.If(
                                                                    (epilogue_warp_idx == 0)
                                                                    & T.cuda.elect_sync()
                                                                    != 0
                                                                ):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            T.evaluate(
                                                                                red_add_gpu_u32(
                                                                                    workspace_l2_empty_count.ptr_to(
                                                                                        [
                                                                                            ring_block_idx
                                                                                        ]
                                                                                    ),
                                                                                    T.uint32(1),
                                                                                )
                                                                            )
                                                                        )
                                                                _builder_emit(T.cuda.warp_sync())
                                                                n_idx = _builder_assign(
                                                                    "n_idx",
                                                                    n_block_idx
                                                                    * kernel_config.block_n,
                                                                    locals().get(
                                                                        "n_idx", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                                with T.serial(
                                                                    0,
                                                                    wg_block_m
                                                                    // kernel_config.store_block_m,
                                                                    unroll=True,
                                                                ) as s:
                                                                    IRBuilder.name("s", s)
                                                                    with T.If(
                                                                        s
                                                                        * kernel_config.store_block_m
                                                                        >= valid_rows_in_wg
                                                                    ):
                                                                        with T.Then():
                                                                            _builder_emit(
                                                                                tmem_empty_barrier_arrive_cta0(
                                                                                    smem_barriers.ptr_to(
                                                                                        [
                                                                                            tmem_empty_barrier_base
                                                                                            + accum_stage_idx
                                                                                        ]
                                                                                    )
                                                                                )
                                                                            )
                                                                            T.evaluate(
                                                                                T.break_loop()
                                                                            )
                                                                    with T.unroll(
                                                                        0, num_atoms_per_store
                                                                    ) as i:
                                                                        IRBuilder.name("i", i)
                                                                        j = _builder_assign(
                                                                            "j",
                                                                            s * num_atoms_per_store
                                                                            + i,
                                                                            locals().get(
                                                                                "j",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        T.buffer_store(
                                                                            tmem_addr,
                                                                            T.cast(
                                                                                accum_stage_idx
                                                                                * umma_n
                                                                                + epilogue_wg_idx
                                                                                * wg_block_m
                                                                                + j * atom_m,
                                                                                "uint32",
                                                                            ),
                                                                            [0],
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx[
                                                                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                                                            ](
                                                                                values[0],
                                                                                values[1],
                                                                                values[2],
                                                                                values[3],
                                                                                T.uint32(
                                                                                    tmem_addr[0]
                                                                                ),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx[
                                                                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                                                            ](
                                                                                values[4],
                                                                                values[5],
                                                                                values[6],
                                                                                values[7],
                                                                                T.uint32(
                                                                                    T.bitwise_or(
                                                                                        tmem_addr[
                                                                                            0
                                                                                        ],
                                                                                        T.uint32(
                                                                                            0x00100000
                                                                                        ),
                                                                                    )
                                                                                ),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            fence_view_async_tmem_load()
                                                                        )
                                                                        with T.If(
                                                                            T.And(i == 0, s > 0)
                                                                        ):
                                                                            with T.Then():
                                                                                _builder_emit(
                                                                                    T.ptx.bar.sync(
                                                                                        T.uint32(
                                                                                            epilogue_wg_sync_barrier_start_idx
                                                                                            + epilogue_wg_idx
                                                                                        ),
                                                                                        128,
                                                                                    )
                                                                                )
                                                                        with T.If(
                                                                            T.And(
                                                                                s
                                                                                == wg_block_m
                                                                                // kernel_config.store_block_m
                                                                                - 1,
                                                                                i
                                                                                == kernel_config.store_block_m
                                                                                // atom_m
                                                                                - 1,
                                                                            )
                                                                        ):
                                                                            with T.Then():
                                                                                _builder_emit(
                                                                                    tmem_empty_barrier_arrive_cta0(
                                                                                        smem_barriers.ptr_to(
                                                                                            [
                                                                                                tmem_empty_barrier_base
                                                                                                + accum_stage_idx
                                                                                            ]
                                                                                        )
                                                                                    )
                                                                                )
                                                                        T.buffer_store(
                                                                            epilogue_bf16_packed,
                                                                            cast_into_bf16_and_pack(
                                                                                uint32_bits_to_float(
                                                                                    values[0]
                                                                                ),
                                                                                uint32_bits_to_float(
                                                                                    values[1]
                                                                                ),
                                                                            ),
                                                                            [0],
                                                                        )
                                                                        T.buffer_store(
                                                                            epilogue_bf16_packed,
                                                                            cast_into_bf16_and_pack(
                                                                                uint32_bits_to_float(
                                                                                    values[2]
                                                                                ),
                                                                                uint32_bits_to_float(
                                                                                    values[3]
                                                                                ),
                                                                            ),
                                                                            [1],
                                                                        )
                                                                        T.buffer_store(
                                                                            epilogue_bf16_packed,
                                                                            cast_into_bf16_and_pack(
                                                                                uint32_bits_to_float(
                                                                                    values[4]
                                                                                ),
                                                                                uint32_bits_to_float(
                                                                                    values[5]
                                                                                ),
                                                                            ),
                                                                            [2],
                                                                        )
                                                                        T.buffer_store(
                                                                            epilogue_bf16_packed,
                                                                            cast_into_bf16_and_pack(
                                                                                uint32_bits_to_float(
                                                                                    values[6]
                                                                                ),
                                                                                uint32_bits_to_float(
                                                                                    values[7]
                                                                                ),
                                                                            ),
                                                                            [3],
                                                                        )
                                                                        row = _builder_assign(
                                                                            "row",
                                                                            lane_idx % 8,
                                                                            locals().get(
                                                                                "row",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        col = _builder_assign(
                                                                            "col",
                                                                            (epilogue_warp_idx % 2)
                                                                            * 4
                                                                            + lane_idx // 8,
                                                                            locals().get(
                                                                                "col",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        smem_ptr = _builder_assign(
                                                                            "smem_ptr",
                                                                            smem.ptr_to(
                                                                                [
                                                                                    smem_cd_offset
                                                                                    + epilogue_wg_idx
                                                                                    * kernel_config.store_block_m
                                                                                    * kernel_config.block_n
                                                                                    * 2
                                                                                    + (
                                                                                        warp_idx_in_wg
                                                                                        // 2
                                                                                    )
                                                                                    * kernel_config.store_block_m
                                                                                    * swizzle_cd_mode
                                                                                    + i
                                                                                    * atom_m
                                                                                    * swizzle_cd_mode
                                                                                    + row
                                                                                    * (
                                                                                        num_bank_group_bytes
                                                                                        * 8
                                                                                    )
                                                                                    + (col ^ row)
                                                                                    * num_bank_group_bytes
                                                                                ]
                                                                            ),
                                                                            locals().get(
                                                                                "smem_ptr",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            sm90_u32x4_stsm_t_copy(
                                                                                epilogue_bf16_packed,
                                                                                smem_ptr,
                                                                            )
                                                                        )
                                                                    _builder_emit(
                                                                        T.ptx.bar.sync(
                                                                            T.uint32(
                                                                                epilogue_wg_sync_barrier_start_idx
                                                                                + epilogue_wg_idx
                                                                            ),
                                                                            128,
                                                                        )
                                                                    )
                                                                    row_in_atom = _builder_assign(
                                                                        "row_in_atom",
                                                                        (
                                                                            warp_idx_in_wg * 2
                                                                            + lane_idx // 16
                                                                        )
                                                                        % atom_m,
                                                                        locals().get(
                                                                            "row_in_atom",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    bank_group_idx = (
                                                                        _builder_assign(
                                                                            "bank_group_idx",
                                                                            lane_idx % 8,
                                                                            locals().get(
                                                                                "bank_group_idx",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                    )
                                                                    lane_col_offset = (
                                                                        _builder_assign(
                                                                            "lane_col_offset",
                                                                            (lane_idx % 16) * 8,
                                                                            locals().get(
                                                                                "lane_col_offset",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        0, num_rows_per_warp
                                                                    ) as j:
                                                                        IRBuilder.name("j", j)
                                                                        row_in_store = _builder_assign(
                                                                            "row_in_store",
                                                                            j * 8
                                                                            + warp_idx_in_wg * 2
                                                                            + lane_idx // 16,
                                                                            locals().get(
                                                                                "row_in_store",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        m_idx_in_block = _builder_assign(
                                                                            "m_idx_in_block",
                                                                            (
                                                                                epilogue_wg_idx
                                                                                * wg_block_m
                                                                                + s
                                                                                * kernel_config.store_block_m
                                                                                + row_in_store
                                                                            ),
                                                                            locals().get(
                                                                                "m_idx_in_block",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(
                                                                            T.cast(
                                                                                m_idx_in_block,
                                                                                "uint32",
                                                                            )
                                                                            < T.cast(
                                                                                valid_m, "uint32"
                                                                            )
                                                                        ):
                                                                            with T.Then():
                                                                                src_metadata_idx = _builder_assign(
                                                                                    "src_metadata_idx",
                                                                                    pool_m_idx
                                                                                    + m_idx_in_block,
                                                                                    locals().get(
                                                                                        "src_metadata_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    load_global_u32(
                                                                                        dst_rank_idx_u32,
                                                                                        workspace_token_src_metadata.ptr_to(
                                                                                            [
                                                                                                src_metadata_idx,
                                                                                                0,
                                                                                            ]
                                                                                        ),
                                                                                    )
                                                                                )
                                                                                _builder_emit(
                                                                                    load_global_u32(
                                                                                        dst_token_idx_u32,
                                                                                        workspace_token_src_metadata.ptr_to(
                                                                                            [
                                                                                                src_metadata_idx,
                                                                                                1,
                                                                                            ]
                                                                                        ),
                                                                                    )
                                                                                )
                                                                                _builder_emit(
                                                                                    load_global_u32(
                                                                                        dst_topk_idx_u32,
                                                                                        workspace_token_src_metadata.ptr_to(
                                                                                            [
                                                                                                src_metadata_idx,
                                                                                                2,
                                                                                            ]
                                                                                        ),
                                                                                    )
                                                                                )
                                                                                dst_rank_idx = _builder_assign(
                                                                                    "dst_rank_idx",
                                                                                    T.cast(
                                                                                        dst_rank_idx_u32,
                                                                                        "int32",
                                                                                    ),
                                                                                    locals().get(
                                                                                        "dst_rank_idx",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                dst_token_base_offset = _builder_assign(
                                                                                    "dst_token_base_offset",
                                                                                    T.cast(
                                                                                        dst_topk_idx_u32,
                                                                                        "uint64",
                                                                                    )
                                                                                    * T.uint64(
                                                                                        workspace_layout.num_max_tokens_per_rank
                                                                                        * hidden
                                                                                        * 2
                                                                                    )
                                                                                    + T.cast(
                                                                                        dst_token_idx_u32,
                                                                                        "uint64",
                                                                                    )
                                                                                    * T.uint64(
                                                                                        hidden * 2
                                                                                    ),
                                                                                    locals().get(
                                                                                        "dst_token_base_offset",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                dst_col_byte_offset = _builder_assign(
                                                                                    "dst_col_byte_offset",
                                                                                    T.cast(
                                                                                        n_idx * 2,
                                                                                        "uint64",
                                                                                    ),
                                                                                    locals().get(
                                                                                        "dst_col_byte_offset",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                lane_byte_offset = _builder_assign(
                                                                                    "lane_byte_offset",
                                                                                    T.cast(
                                                                                        (
                                                                                            lane_idx
                                                                                            % 16
                                                                                        )
                                                                                        * 16,
                                                                                        "uint64",
                                                                                    ),
                                                                                    locals().get(
                                                                                        "lane_byte_offset",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                dst_ptr = _builder_assign(
                                                                                    "dst_ptr",
                                                                                    (
                                                                                        dst_token_base_offset
                                                                                        + dst_col_byte_offset
                                                                                        + lane_byte_offset
                                                                                    ),
                                                                                    locals().get(
                                                                                        "dst_ptr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                smem_ptr = _builder_assign(
                                                                                    "smem_ptr",
                                                                                    smem.ptr_to(
                                                                                        [
                                                                                            smem_cd_offset
                                                                                            + epilogue_wg_idx
                                                                                            * kernel_config.store_block_m
                                                                                            * kernel_config.block_n
                                                                                            * 2
                                                                                            + (
                                                                                                (
                                                                                                    lane_idx
                                                                                                    % 16
                                                                                                )
                                                                                                // 8
                                                                                            )
                                                                                            * kernel_config.store_block_m
                                                                                            * swizzle_cd_mode
                                                                                            + row_in_store
                                                                                            * swizzle_cd_mode
                                                                                            + (
                                                                                                bank_group_idx
                                                                                                ^ row_in_atom
                                                                                            )
                                                                                            * num_bank_group_bytes
                                                                                        ]
                                                                                    ),
                                                                                    locals().get(
                                                                                        "smem_ptr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    lds128(
                                                                                        smem_ptr,
                                                                                        epilogue_bf16_packed,
                                                                                    )
                                                                                )
                                                                                symm_rank_base = _builder_assign(
                                                                                    "symm_rank_base",
                                                                                    sym_buffer_base
                                                                                    + T.cast(
                                                                                        symm_rank_offsets[
                                                                                            0
                                                                                        ],
                                                                                        "uint64",
                                                                                    ),
                                                                                    locals().get(
                                                                                        "symm_rank_base",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    load_symm_rank_base(
                                                                                        symm_rank_base,
                                                                                        smem_symm_rank_bases,
                                                                                        dst_rank_idx,
                                                                                    )
                                                                                )
                                                                                dst_ptr = _builder_assign(
                                                                                    "dst_ptr",
                                                                                    (
                                                                                        T.cast(
                                                                                            symm_buffer_layout.combine_token_offset,
                                                                                            "uint64",
                                                                                        )
                                                                                        + dst_ptr
                                                                                    ),
                                                                                    locals().get(
                                                                                        "dst_ptr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    stg128_symm(
                                                                                        symm_rank_base,
                                                                                        dst_ptr,
                                                                                        epilogue_bf16_packed[
                                                                                            0
                                                                                        ],
                                                                                        epilogue_bf16_packed[
                                                                                            1
                                                                                        ],
                                                                                        epilogue_bf16_packed[
                                                                                            2
                                                                                        ],
                                                                                        epilogue_bf16_packed[
                                                                                            3
                                                                                        ],
                                                                                    )
                                                                                )
                                                                _builder_emit(
                                                                    T.ptx.bar.sync(
                                                                        T.uint32(
                                                                            epilogue_full_sync_barrier_idx
                                                                        ),
                                                                        T.uint32(
                                                                            kernel_config.num_epilogue_threads
                                                                        ),
                                                                    )
                                                                )
                                                    epilogue_thread_idx = _builder_assign(
                                                        "epilogue_thread_idx",
                                                        epilogue_warp_idx * 32 + lane_idx,
                                                        locals().get(
                                                            "epilogue_thread_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    _builder_emit(
                                                        epilogue_nvlink_barrier_before_combine_reduce(
                                                            epilogue_thread_idx
                                                        )
                                                    )
                                                    # The grid barrier above includes every epilogue thread in both CTAs,
                                                    # so no peer can still be accessing TMEM when warp 0 deallocates it.
                                                    with T.If(epilogue_warp_idx == 0):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.ptx[
                                                                    f"tcgen05.dealloc.cta_group::{kernel_config.num_ctas_per_cluster}.sync.aligned.b32"
                                                                ](
                                                                    T.uint32(0),
                                                                    T.uint32(num_tmem_cols),
                                                                )
                                                            )
                                                    _builder_emit(
                                                        T.evaluate(
                                                            sync_unaligned(
                                                                dispatch_with_epilogue_sync_barrier_idx,
                                                                kernel_config.num_dispatch_threads
                                                                + kernel_config.num_epilogue_threads,
                                                            )
                                                        )
                                                    )
                                                    # The preceding grid barrier publishes every epilogue's generic
                                                    # stores into the symmetric combine-token buffer.  Combine reads
                                                    # those stores through TMA's async global proxy.
                                                    _builder_emit(
                                                        T.ptx.fence.proxy.async_.global_()
                                                    )
                                                    token_idx = _builder_assign(
                                                        "token_idx",
                                                        sm_idx * kernel_config.num_epilogue_warps
                                                        + epilogue_warp_idx,
                                                        locals().get("token_idx", _BUILDER_MISSING),
                                                    )
                                                    combine_phase = _builder_assign(
                                                        "combine_phase",
                                                        T.int32(0),
                                                        locals().get(
                                                            "combine_phase", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    load_stage_idx = _builder_assign(
                                                        "load_stage_idx",
                                                        T.int32(0),
                                                        locals().get(
                                                            "load_stage_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    with T.While(
                                                        T.cast(token_idx, "uint32")
                                                        < T.cast(num_tokens, "uint32")
                                                    ):
                                                        stored_topk_slot_idx = _builder_assign(
                                                            "stored_topk_slot_idx",
                                                            T.int32(-1),
                                                            locals().get(
                                                                "stored_topk_slot_idx",
                                                                _BUILDER_MISSING,
                                                            ),
                                                        )
                                                        with T.If(lane_idx < num_topk):
                                                            with T.Then():
                                                                _builder_emit(
                                                                    load_global_s64(
                                                                        ordinary_global_s64,
                                                                        input_topk_idx.ptr_to(
                                                                            [token_idx, lane_idx]
                                                                        ),
                                                                    )
                                                                )
                                                                stored_topk_slot_idx = (
                                                                    _builder_assign(
                                                                        "stored_topk_slot_idx",
                                                                        T.cast(
                                                                            ordinary_global_s64,
                                                                            "int32",
                                                                        ),
                                                                        locals().get(
                                                                            "stored_topk_slot_idx",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                )
                                                        total_mask = _builder_assign(
                                                            "total_mask",
                                                            ballot_sync(
                                                                T.uint32(0xFFFFFFFF),
                                                                stored_topk_slot_idx >= T.int32(0),
                                                            ),
                                                            locals().get(
                                                                "total_mask", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.unroll(0, num_chunks) as chunk:
                                                            IRBuilder.name("chunk", chunk)
                                                            mask = _builder_assign(
                                                                "mask",
                                                                total_mask,
                                                                locals().get(
                                                                    "mask", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            chunk_byte_offset = _builder_assign(
                                                                "chunk_byte_offset",
                                                                chunk * num_chunk_bytes,
                                                                locals().get(
                                                                    "chunk_byte_offset",
                                                                    _BUILDER_MISSING,
                                                                ),
                                                            )
                                                            chunk_offset_elems = _builder_assign(
                                                                "chunk_offset_elems",
                                                                chunk_byte_offset // 2,
                                                                locals().get(
                                                                    "chunk_offset_elems",
                                                                    _BUILDER_MISSING,
                                                                ),
                                                            )
                                                            with T.unroll(
                                                                0,
                                                                num_uint4_per_lane
                                                                * num_elems_per_uint4,
                                                            ) as reduced_idx:
                                                                IRBuilder.name(
                                                                    "reduced_idx", reduced_idx
                                                                )
                                                                T.buffer_store(
                                                                    reduced,
                                                                    T.float32(0.0),
                                                                    [reduced_idx, 0],
                                                                )
                                                                T.buffer_store(
                                                                    reduced,
                                                                    T.float32(0.0),
                                                                    [reduced_idx, 1],
                                                                )
                                                            do_reduce = _builder_assign(
                                                                "do_reduce",
                                                                T.int32(0),
                                                                locals().get(
                                                                    "do_reduce", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            with T.If(mask != T.uint32(0)):
                                                                with T.Then():
                                                                    slot_idx = _builder_assign(
                                                                        "slot_idx",
                                                                        ffs_u32(mask) - T.int32(1),
                                                                        locals().get(
                                                                            "slot_idx",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    mask = _builder_assign(
                                                                        "mask",
                                                                        T.bitwise_xor(
                                                                            mask,
                                                                            T.shift_left(
                                                                                T.uint32(1),
                                                                                T.cast(
                                                                                    slot_idx,
                                                                                    "uint32",
                                                                                ),
                                                                            ),
                                                                        ),
                                                                        locals().get(
                                                                            "mask", _BUILDER_MISSING
                                                                        ),
                                                                    )
                                                                    with T.If(T.cuda.elect_sync()):
                                                                        with T.Then():
                                                                            src_ptr = _builder_assign(
                                                                                "src_ptr",
                                                                                combine_tokens.ptr_to(
                                                                                    [
                                                                                        slot_idx,
                                                                                        token_idx,
                                                                                        chunk_offset_elems,
                                                                                    ]
                                                                                ),
                                                                                locals().get(
                                                                                    "src_ptr",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            load_barrier_ptr = _builder_assign(
                                                                                "load_barrier_ptr",
                                                                                smem_barriers.ptr_to(
                                                                                    [
                                                                                        combine_barrier_base
                                                                                        + epilogue_warp_idx
                                                                                        * 2
                                                                                        + load_stage_idx
                                                                                    ]
                                                                                ),
                                                                                locals().get(
                                                                                    "load_barrier_ptr",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            load_buffer_ptr = _builder_assign(
                                                                                "load_buffer_ptr",
                                                                                combine_chunks.ptr_to(
                                                                                    [
                                                                                        load_stage_idx,
                                                                                        epilogue_warp_idx,
                                                                                        0,
                                                                                        0,
                                                                                    ]
                                                                                ),
                                                                                locals().get(
                                                                                    "load_buffer_ptr",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            _builder_emit(
                                                                                tma_load_1d(
                                                                                    load_buffer_ptr,
                                                                                    src_ptr,
                                                                                    load_barrier_ptr,
                                                                                    num_chunk_bytes,
                                                                                )
                                                                            )
                                                                            _builder_emit(
                                                                                mbarrier_arrive_and_set_tx(
                                                                                    load_barrier_ptr,
                                                                                    num_chunk_bytes,
                                                                                )
                                                                            )
                                                                    do_reduce = _builder_assign(
                                                                        "do_reduce",
                                                                        T.int32(1),
                                                                        locals().get(
                                                                            "do_reduce",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                            _builder_emit(T.cuda.warp_sync())
                                                            with T.While(do_reduce != T.int32(0)):
                                                                next_do_reduce = _builder_assign(
                                                                    "next_do_reduce",
                                                                    T.int32(0),
                                                                    locals().get(
                                                                        "next_do_reduce",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                with T.If(mask != T.uint32(0)):
                                                                    with T.Then():
                                                                        slot_idx = _builder_assign(
                                                                            "slot_idx",
                                                                            ffs_u32(mask)
                                                                            - T.int32(1),
                                                                            locals().get(
                                                                                "slot_idx",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        mask = _builder_assign(
                                                                            "mask",
                                                                            T.bitwise_xor(
                                                                                mask,
                                                                                T.shift_left(
                                                                                    T.uint32(1),
                                                                                    T.cast(
                                                                                        slot_idx,
                                                                                        "uint32",
                                                                                    ),
                                                                                ),
                                                                            ),
                                                                            locals().get(
                                                                                "mask",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        with T.If(
                                                                            T.cuda.elect_sync()
                                                                        ):
                                                                            with T.Then():
                                                                                src_ptr = _builder_assign(
                                                                                    "src_ptr",
                                                                                    combine_tokens.ptr_to(
                                                                                        [
                                                                                            slot_idx,
                                                                                            token_idx,
                                                                                            chunk_offset_elems,
                                                                                        ]
                                                                                    ),
                                                                                    locals().get(
                                                                                        "src_ptr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                load_barrier_ptr = _builder_assign(
                                                                                    "load_barrier_ptr",
                                                                                    smem_barriers.ptr_to(
                                                                                        [
                                                                                            combine_barrier_base
                                                                                            + epilogue_warp_idx
                                                                                            * 2
                                                                                            + (
                                                                                                load_stage_idx
                                                                                                ^ T.int32(
                                                                                                    1
                                                                                                )
                                                                                            )
                                                                                        ]
                                                                                    ),
                                                                                    locals().get(
                                                                                        "load_barrier_ptr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                prefetch_buffer_ptr = _builder_assign(
                                                                                    "prefetch_buffer_ptr",
                                                                                    combine_chunks.ptr_to(
                                                                                        [
                                                                                            load_stage_idx
                                                                                            ^ T.int32(
                                                                                                1
                                                                                            ),
                                                                                            epilogue_warp_idx,
                                                                                            0,
                                                                                            0,
                                                                                        ]
                                                                                    ),
                                                                                    locals().get(
                                                                                        "prefetch_buffer_ptr",
                                                                                        _BUILDER_MISSING,
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    tma_load_1d(
                                                                                        prefetch_buffer_ptr,
                                                                                        src_ptr,
                                                                                        load_barrier_ptr,
                                                                                        num_chunk_bytes,
                                                                                    )
                                                                                )
                                                                                _builder_emit(
                                                                                    mbarrier_arrive_and_set_tx(
                                                                                        load_barrier_ptr,
                                                                                        num_chunk_bytes,
                                                                                    )
                                                                                )
                                                                        next_do_reduce = _builder_assign(
                                                                            "next_do_reduce",
                                                                            T.int32(1),
                                                                            locals().get(
                                                                                "next_do_reduce",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                _builder_emit(T.cuda.warp_sync())
                                                                _builder_emit(
                                                                    mbarrier_wait_phase(
                                                                        smem_barriers.ptr_to(
                                                                            [
                                                                                combine_barrier_base
                                                                                + epilogue_warp_idx
                                                                                * 2
                                                                                + load_stage_idx
                                                                            ]
                                                                        ),
                                                                        combine_phase,
                                                                    )
                                                                )
                                                                with T.unroll(
                                                                    0, num_uint4_per_lane
                                                                ) as j:
                                                                    IRBuilder.name("j", j)
                                                                    load_ptr = _builder_assign(
                                                                        "load_ptr",
                                                                        combine_chunks.ptr_to(
                                                                            [
                                                                                load_stage_idx,
                                                                                epilogue_warp_idx,
                                                                                j * 32 + lane_idx,
                                                                                0,
                                                                            ]
                                                                        ),
                                                                        locals().get(
                                                                            "load_ptr",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        lds128(
                                                                            load_ptr,
                                                                            epilogue_bf16_packed,
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        0, num_elems_per_uint4
                                                                    ) as elem_idx:
                                                                        IRBuilder.name(
                                                                            "elem_idx", elem_idx
                                                                        )
                                                                        # d = convert(a) + c, accumulating in place.
                                                                        _builder_emit(
                                                                            T.ptx.add.rn.f32.bf16(
                                                                                reduced[
                                                                                    j
                                                                                    * num_elems_per_uint4
                                                                                    + elem_idx,
                                                                                    0,
                                                                                ],
                                                                                bf16x2_lo(
                                                                                    epilogue_bf16_packed[
                                                                                        elem_idx
                                                                                    ]
                                                                                ),
                                                                                reduced[
                                                                                    j
                                                                                    * num_elems_per_uint4
                                                                                    + elem_idx,
                                                                                    0,
                                                                                ],
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx.add.rn.f32.bf16(
                                                                                reduced[
                                                                                    j
                                                                                    * num_elems_per_uint4
                                                                                    + elem_idx,
                                                                                    1,
                                                                                ],
                                                                                bf16x2_hi(
                                                                                    epilogue_bf16_packed[
                                                                                        elem_idx
                                                                                    ]
                                                                                ),
                                                                                reduced[
                                                                                    j
                                                                                    * num_elems_per_uint4
                                                                                    + elem_idx,
                                                                                    1,
                                                                                ],
                                                                            )
                                                                        )
                                                                combine_phase = _builder_assign(
                                                                    "combine_phase",
                                                                    combine_phase ^ load_stage_idx,
                                                                    locals().get(
                                                                        "combine_phase",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                load_stage_idx = _builder_assign(
                                                                    "load_stage_idx",
                                                                    load_stage_idx ^ T.int32(1),
                                                                    locals().get(
                                                                        "load_stage_idx",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                do_reduce = _builder_assign(
                                                                    "do_reduce",
                                                                    next_do_reduce,
                                                                    locals().get(
                                                                        "do_reduce",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                            with T.unroll(
                                                                0, num_uint4_per_lane
                                                            ) as j:
                                                                IRBuilder.name("j", j)
                                                                with T.unroll(
                                                                    0, num_elems_per_uint4
                                                                ) as elem_idx:
                                                                    IRBuilder.name(
                                                                        "elem_idx", elem_idx
                                                                    )
                                                                    T.buffer_store(
                                                                        epilogue_bf16_packed,
                                                                        cast_into_bf16_and_pack(
                                                                            reduced[
                                                                                j
                                                                                * num_elems_per_uint4
                                                                                + elem_idx,
                                                                                0,
                                                                            ],
                                                                            reduced[
                                                                                j
                                                                                * num_elems_per_uint4
                                                                                + elem_idx,
                                                                                1,
                                                                            ],
                                                                        ),
                                                                        [elem_idx],
                                                                    )
                                                                with T.If(j == 0):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            T.evaluate(
                                                                                tma_store_wait(0)
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.cuda.warp_sync()
                                                                        )
                                                                combine_store_ptr = _builder_assign(
                                                                    "combine_store_ptr",
                                                                    combine_chunks.ptr_to(
                                                                        [
                                                                            2,
                                                                            epilogue_warp_idx,
                                                                            j * 32 + lane_idx,
                                                                            0,
                                                                        ]
                                                                    ),
                                                                    locals().get(
                                                                        "combine_store_ptr",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                _builder_emit(
                                                                    sts128(
                                                                        combine_store_ptr,
                                                                        epilogue_bf16_packed[0],
                                                                        epilogue_bf16_packed[1],
                                                                        epilogue_bf16_packed[2],
                                                                        epilogue_bf16_packed[3],
                                                                    )
                                                                )
                                                            _builder_emit(T.cuda.warp_sync())
                                                            with T.If(T.cuda.elect_sync()):
                                                                with T.Then():
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            tma_store_fence()
                                                                        )
                                                                    )
                                                                    dst_ptr = _builder_assign(
                                                                        "dst_ptr",
                                                                        T.address_of(
                                                                            y[
                                                                                token_idx,
                                                                                chunk_offset_elems,
                                                                            ]
                                                                        ),
                                                                        _BUILDER_MISSING,
                                                                    )
                                                                    combine_store_ptr = _builder_assign(
                                                                        "combine_store_ptr",
                                                                        combine_chunks.ptr_to(
                                                                            [
                                                                                2,
                                                                                epilogue_warp_idx,
                                                                                0,
                                                                                0,
                                                                            ]
                                                                        ),
                                                                        locals().get(
                                                                            "combine_store_ptr",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    _builder_emit(
                                                                        tma_store_1d(
                                                                            dst_ptr,
                                                                            combine_store_ptr,
                                                                            num_chunk_bytes,
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        T.evaluate(
                                                                            tma_store_arrive()
                                                                        )
                                                                    )
                                                        _builder_emit(T.cuda.warp_sync())
                                                        token_idx = _builder_assign(
                                                            "token_idx",
                                                            token_idx
                                                            + kernel_config.num_sms
                                                            * kernel_config.num_epilogue_warps,
                                                            locals().get(
                                                                "token_idx", _BUILDER_MISSING
                                                            ),
                                                        )

    return builder.get().with_attr("tirx.kernel_launch_params", get_tirx_launch_param_tags())
