# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""The MegaMoE dispatch, GEMM, and combine megakernel.

Configuration and layout derivation live in :mod:`.spec`; nothing here reaches
back into :mod:`.data`. K owns the warp partition, register heads, shared and
tensor storage, and reusable pipeline cursors. Kernel-specific cross-rank,
task-publication, instruction-order, and synchronization semantics remain
explicit in low-level TIRx.

GUARD: the ``@K.prim_func def mega_moe`` name and the ``"main"`` global var are
part of the emitted CUDA symbol -- never rename them.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

import math
import os

import tirx_kernels.kern as K
from tvm.ir.type import PointerType, PrimType

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
    return K.ptx.ld.global_.s64(dst, address)


def load_global_u64(dst, address):
    return K.ptx.ld.global_.u64(dst, address)


def load_global_u32(dst, address):
    return K.ptx.ld.global_.u32(dst, address)


def store_global_u64(address, value):
    return K.ptx.st.global_.u64(address, value)


def store_global_u32(address, value):
    return K.ptx.st.global_.u32(address, value)


def store_global_u8(address, value):
    return K.ptx.st.global_.u8(address, value)


# Destination-first, mirroring the PTX these wrap: the caller declares the
# register and passes it in.
def load_acq_sys_s32(dst, address):
    return K.ptx.ld.acquire.sys.global_.s32(dst, address)


def atomic_add_rel_u32(dst, address, value):
    return K.ptx.atom.release.gpu.global_.add.u32(dst, address, value)


def load_acq_u32(dst, address):
    return K.ptx.ld.acquire.gpu.global_.b32(dst, address)


def grid_sync_done_u32(new_value, old_value):
    return K.cast(
        K.bitwise_and(K.bitwise_xor(new_value, old_value), K.uint32(0x80000000)) != K.uint32(0),
        "uint32",
    )


def load_f32(dst, address):
    # ptx destinations are declared registers, so the helper writes into
    # one the caller owns rather than returning a value.
    return K.ptx.ld.global_.f32(dst, address)


def load_shared_u32(dst, address):
    return K.ptx.ld.shared.u32(dst, address)


def uint32_bits_to_float(bits):
    return K.cuda.uint_as_float(bits)


def float_bits(x):
    return K.cuda.float_as_uint(x)


def sync_unaligned(barrier_idx, num_threads):
    return K.ptx.barrier.sync(K.uint32(barrier_idx), K.uint32(num_threads))


def prefetch_tensormap(tensor_map):
    return K.ptx.prefetch.tensormap(K.address_of(tensor_map))


def lds128(src_ptr, dst, base=0):
    return K.ptx.ld.shared.v4.u32(dst[base], dst[base + 1], dst[base + 2], dst[base + 3], src_ptr)


def mbarrier_arrive_and_set_tx(barrier_ptr, num_bytes):
    return K.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier_ptr, K.uint32(num_bytes))


def mbarrier_wait_phase(barrier_ptr, phase):
    return K.cuda.mbarrier_wait(barrier_ptr, phase)


def replace_smem_desc_addr(desc, smem_ptr):
    start_addr = K.cast(
        K.bitwise_and(
            K.shift_right(K.cuda.cvta_generic_to_shared(smem_ptr), K.uint32(4)), K.uint32(0x3FFF)
        ),
        "uint64",
    )
    return K.bitwise_or(K.bitwise_and(desc, K.bitwise_not(K.uint64(0x3FFF))), start_addr)


#: Bulk global -> shared copy, completion signalled on an mbarrier.
_bulk_g2s_chain = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"

#: Bulk shared -> global copy, retired through the bulk commit group.
_bulk_s2g_chain = "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"

#: `cute::TMA::CacheHintSm90::EVICT_FIRST`. Ring-buffer traffic is streamed
#: through L2 once, so it should be the first thing evicted.
_evict_first_policy = K.uint64(0x12F0000000000000)


def tma_load_1d(dst_ptr, src_ptr, barrier_ptr, num_bytes):
    return K.ptx[_bulk_g2s_chain](
        dst_ptr, src_ptr, K.cast(num_bytes, "uint32"), barrier_ptr, _evict_first_policy
    )


def tma_store_1d(dst_ptr, src_ptr, num_bytes):
    return K.ptx[_bulk_s2g_chain](
        dst_ptr, src_ptr, K.cast(num_bytes, "uint32"), _evict_normal_policy
    )


def tma_store_fence():
    return K.ptx.fence.proxy.async_.shared__cta()


def fence_barrier_init():
    return K.ptx.fence.mbarrier_init.release.cluster()


def tma_store_arrive():
    return K.ptx.cp.async_.bulk.commit_group()


def tma_store_wait(num_prior_groups):
    if num_prior_groups == 0:
        return K.ptx.cp.async_.bulk.wait_group(0)
    if num_prior_groups == 1:
        return K.ptx.cp.async_.bulk.wait_group(1)
    raise ValueError("Unsupported TMA store wait distance")


def tma_store_2d_addr(src, tensormap_addr, coord0, coord1):
    """2D TMA store taking an already-computed descriptor address, so the
    caller can select between two descriptors without materializing a local."""
    return K.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
        tensormap_addr, coord0, coord1, src
    )


#: Unicast (cta_mask=1) at cta_group::2 scope with the evict-normal L2 policy;
#: the mbarrier arrives as a precomputed leader-CTA shared address.
_sm100_2sm_load_chain = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)

#: `cute::TMA::CacheHintSm100::EVICT_NORMAL`. Weights and activations are
#: re-read across blocks, so they keep the default L2 residency.
_evict_normal_policy = K.uint64(1152921504606846976)


def sm100_tma_2sm_load_2d_addr(dst, mbar, tensormap_addr, coord0, coord1):
    mbar_addr = K.cuda.sm100_2sm_leader_smem_addr(mbar)
    K.ptx[_sm100_2sm_load_chain](
        dst, tensormap_addr, coord0, coord1, mbar_addr, _evict_normal_policy
    )


def stg128_symm(peer_base, byte_offset, r0, r1, r2, r3):
    return K.ptx.st.global_.v4.b32(peer_ptr(peer_base, byte_offset), r0, r1, r2, r3)


def ptr_to_u64(ptr):
    return K.reinterpret("uint64", ptr)


def peer_ptr(peer_base, byte_offset):
    return K.reinterpret("handle", peer_base + byte_offset)


def peer_store_u32(peer_base, byte_offset, value):
    return K.ptx.st.global_.u32(peer_ptr(peer_base, byte_offset), value)


def peer_store_u64(peer_base, byte_offset, value):
    return K.ptx.st.global_.u64(peer_ptr(peer_base, byte_offset), value)


def st_shared_bulk(ptr, num_bytes):
    return K.ptx.st_bulk.weak.shared__cta(ptr, K.cast(num_bytes, "uint64"))


def peer_atomic_add_u64(dst, peer_base, byte_offset, value):
    return K.ptx.atom.sys.global_.add.u64(dst, peer_ptr(peer_base, byte_offset), value)


def peer_red_add_rel_sys_s32(peer_base, byte_offset, value):
    return K.ptx.red.release.sys.global_.add.s32(peer_ptr(peer_base, byte_offset), value)


def peer_load_u32(dst, peer_base, byte_offset):
    return K.ptx.ld.global_.u32(dst, peer_ptr(peer_base, byte_offset))


def peer_load_f32(dst, peer_base, byte_offset):
    return K.ptx.ld.global_.f32(dst, peer_ptr(peer_base, byte_offset))


def tma_load_1d_symm(dst_ptr, peer_base, byte_offset, barrier_ptr, num_bytes):
    return K.ptx[_bulk_g2s_chain](
        dst_ptr,
        peer_ptr(peer_base, byte_offset),
        K.cast(num_bytes, "uint32"),
        barrier_ptr,
        _evict_first_policy,
    )


def ballot_sync(mask, pred):
    return K.cuda.ballot_sync(mask, pred)


def ffs_u32(value):
    return K.cuda.ffs_u32(value)


def reduce_add_sync_u32(mask, value):
    return K.cuda.reduce_add_sync_u32(mask, value)


def red_add_gpu_u32(address, value):
    return K.ptx.red.gpu.global_.add.u32(address, value)


def cuda_clock64():
    return K.cuda.clock64()


def bf16x2_lo(packed):
    return K.cast(K.bitwise_and(packed, K.uint32(0xFFFF)), "uint16")


def bf16x2_hi(packed):
    return K.cast(K.bitwise_and(K.shift_right(packed, K.uint32(16)), K.uint32(0xFFFF)), "uint16")


def cast_into_bf16_and_pack(v0, v1):
    return K.cuda.float22bfloat162_rn(v0, v1)


def make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
    runtime_desc = K.bitwise_and(desc, K.uint32(0x9FFFFFCF))
    runtime_desc = K.bitwise_or(runtime_desc, K.shift_left(K.cast(sfa_id, "uint32"), K.uint32(29)))
    runtime_desc = K.bitwise_or(runtime_desc, K.shift_left(K.cast(sfb_id, "uint32"), K.uint32(4)))
    return runtime_desc


def st_async_cluster_task_info(dst_ptr, bar_ptr, dst_cta_idx, task_info_regs):
    mapped_bar = K.local_scalar("uint32")
    mapped_dst = K.local_scalar("uint32")
    mapped_dst_hi = K.local_scalar("uint32")
    cta = K.cast(dst_cta_idx, "uint32")
    K.ptx.mapa.shared__cluster.u32(mapped_bar, K.cuda.cvta_generic_to_shared(bar_ptr), cta)
    K.ptx.mapa.shared__cluster.u32(mapped_dst, K.cuda.cvta_generic_to_shared(dst_ptr), cta)
    K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
        mapped_dst,
        task_info_regs[0],
        task_info_regs[1],
        task_info_regs[2],
        task_info_regs[3],
        mapped_bar,
    )
    K.ptx.add.u32(mapped_dst_hi, mapped_dst, K.uint32(16))
    return K.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
        mapped_dst_hi,
        task_info_regs[4],
        task_info_regs[5],
        task_info_regs[6],
        task_info_regs[7],
        mapped_bar,
    )


def atomic_add_u32(dst, address, value):
    return K.ptx.atom.global_.add.u32(dst, address, value)


def load_volatile_u32(dst, address):
    return K.ptx.ld.volatile.global_.u32(dst, address)


def get_kernel(
    *,
    num_processes: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    num_shared_experts: int = 0,
    activation_clamp: float = 10.0,
    fast_math: int = 1,
    collect_stats: bool = False,
    emit_nvl_barrier_timeout_printf: bool = True,
):
    # ---- compile-time constants (all Python ints; nothing below is emitted) ----
    runtime_config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_shared_experts=num_shared_experts,
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
    # Shared experts (`kNumSharedExperts`): a width multiplier only. One fused
    # rank-local FFN of intermediate width `S * I`, unweighted, whose output
    # lands in an extra combine slot. `has_shared` gates every region below;
    # at S == 0 nothing shared is emitted and the generated CUDA is unchanged
    # apart from the nine always-present tensor-map parameters.
    has_shared = num_shared_experts > 0
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts
    # Scheduler shapes (scheduler/mega_moe.cuh:154-157).
    shared_l1_shape_n = shared_intermediate_hidden * 2
    shared_l1_shape_k = hidden
    shared_l2_shape_n = hidden
    shared_l2_shape_k = shared_intermediate_hidden
    num_shared_l1_block_ns = shared_l1_shape_n // kernel_config.block_n
    num_shared_l2_block_ns = shared_l2_shape_n // kernel_config.block_n
    num_shared_l1_clusters = num_shared_l1_block_ns // 2
    num_shared_l2_clusters = num_shared_l2_block_ns // 2
    # `BlockPhase`: None=0, Linear1=1, Linear2=2, SharedLinear1=3, SharedLinear2=4.
    # `TaskInfo::is_shared()` is `block_phase > Linear2`.
    block_phase_shared_l1 = 3
    block_phase_shared_l2 = 4
    # Shared experts add one combine slot per token, written rank-locally.
    num_combine_slots = num_topk + (1 if has_shared else 0)
    num_max_shared_sf_tokens = symm_buffer_layout.num_max_shared_sf_tokens
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
        sf_inv_pair = K.cuda.make_float2(sf_inv_x, sf_inv_y)
        K.assign(upper[0], K.cuda.fmul2_rn(K.cuda.make_float2(v0, v1), sf_inv_pair))
        K.assign(lower[0], K.cuda.fmul2_rn(K.cuda.make_float2(v2, v3), sf_inv_pair))
        K.assign(
            out[0],
            K.cuda.fp8x4_e4m3_from_float4(
                K.cuda.float2_x(upper[0]),
                K.cuda.float2_y(upper[0]),
                K.cuda.float2_x(lower[0]),
                K.cuda.float2_y(lower[0]),
            ),
        )

    def get_e4m3_sf_and_sf_inv(
        sf, sf_inv, scaled_pair, scaled_values, scaled_bits, scale_exponents, amax_x, amax_y
    ):
        K.assign(
            scaled_pair[0],
            K.cuda.fmul2_rn(
                K.cuda.make_float2(amax_x, amax_y),
                K.cuda.make_float2(K.float32(1.0 / 448.0), K.float32(1.0 / 448.0)),
            ),
        )
        K.ptx.mov.b32(scaled_values[0], K.cuda.float2_x(scaled_pair[0]))
        K.ptx.mov.b32(scaled_values[1], K.cuda.float2_y(scaled_pair[0]))
        with K.unroll(0, 2) as dim:
            K.ptx.mov.b32(scaled_bits[dim], float_bits(scaled_values[dim]))
            K.ptx.mov.b32(
                scale_exponents[dim],
                (
                    K.cast(K.shift_right(scaled_bits[dim], K.uint32(23)), "int32")
                    - K.int32(127)
                    + K.Select(
                        K.bitwise_and(scaled_bits[dim], K.uint32((1 << 23) - 1)) != K.uint32(0),
                        K.int32(1),
                        K.int32(0),
                    )
                ),
            )
            K.ptx.mov.b32(
                sf[dim],
                uint32_bits_to_float(
                    K.shift_left(
                        K.cast(scale_exponents[dim] + K.int32(127), "uint32"), K.uint32(23)
                    )
                ),
            )
            K.ptx.mov.b32(
                sf_inv[dim],
                uint32_bits_to_float(
                    K.shift_left(
                        K.cast(K.int32(127) - scale_exponents[dim], "uint32"), K.uint32(23)
                    )
                ),
            )

    kernel_activation_clamp = float(activation_clamp)
    kernel_fast_math = bool(fast_math)
    kernel_collect_stats = bool(collect_stats)
    kernel_emit_nvl_barrier_timeout_printf = bool(emit_nvl_barrier_timeout_printf)
    use_activation_clamp = math.isfinite(kernel_activation_clamp)

    def warp_reduce_max_4(values, atom_idx, dim):
        K.ptx.mov.b32(
            values[atom_idx, dim],
            K.max(
                values[atom_idx, dim],
                K.tvm_warp_shuffle_xor(0xFFFFFFFF, values[atom_idx, dim], 4, 32, 32),
            ),
        )
        K.ptx.mov.b32(
            values[atom_idx, dim],
            K.max(
                values[atom_idx, dim],
                K.tvm_warp_shuffle_xor(0xFFFFFFFF, values[atom_idx, dim], 8, 32, 32),
            ),
        )
        K.ptx.mov.b32(
            values[atom_idx, dim],
            K.max(
                values[atom_idx, dim],
                K.tvm_warp_shuffle_xor(0xFFFFFFFF, values[atom_idx, dim], 16, 32, 32),
            ),
        )

    def transform_sf_token_idx(token_idx_in_expert):
        token_idx_u32 = K.cast(token_idx_in_expert, "uint32")
        idx = token_idx_u32 % K.uint32(kernel_config.block_m)
        return K.cast(
            token_idx_u32 // K.uint32(kernel_config.block_m) * K.uint32(sf_block_m)
            + K.bitwise_and(idx, K.uint32(0xFFFFFF80))
            + K.shift_left(K.bitwise_and(idx, K.uint32(31)), K.uint32(2))
            + K.bitwise_and(K.shift_right(idx, K.uint32(5)), K.uint32(3)),
            "int32",
        )

    def sm100_tma_2sm_load_2d_select(
        dst,
        mbar,
        tensor_map_l1,
        tensor_map_l2,
        block_phase_value,
        coord0,
        coord1,
        tensor_map_shared_l1=None,
        tensor_map_shared_l2=None,
    ):
        # Chained descriptor select on `block_phase`, mirroring the source's
        # ternary chain (impls .cuh:676-683). At S == 0 only Linear1/Linear2
        # exist, so this collapses back to the original two-way select. The
        # expression is built in Python and passed inline: binding it to a name
        # would materialize a local buffer instead of an inline ternary.
        sm100_tma_2sm_load_2d_addr(
            dst,
            mbar,
            K.if_then_else(
                has_shared,
                K.Select(
                    block_phase_value == 1,
                    K.address_of(tensor_map_l1),
                    K.Select(
                        block_phase_value == 2,
                        K.address_of(tensor_map_l2),
                        K.Select(
                            block_phase_value == 3,
                            K.address_of(tensor_map_shared_l1),
                            K.address_of(tensor_map_shared_l2),
                        ),
                    ),
                ),
                K.Select(
                    block_phase_value == 1, K.address_of(tensor_map_l1), K.address_of(tensor_map_l2)
                ),
            ),
            coord0,
            coord1,
        )

    def activation_pair_store(out, atom_idx, pair_idx, gate0, gate1, up0, up1, weight0, weight1):
        bf16_gate = cast_into_bf16_and_pack(gate0, gate1)
        bf16_up = cast_into_bf16_and_pack(up0, up1)

        with K.If(use_activation_clamp), K.Then():
            activation_clamp_value = K.float32(kernel_activation_clamp)
            clamp_pos = cast_into_bf16_and_pack(activation_clamp_value, activation_clamp_value)
            clamp_neg = cast_into_bf16_and_pack(-activation_clamp_value, -activation_clamp_value)
            bf16_gate = K.cuda.hmin2(bf16_gate, clamp_pos)
            bf16_up = K.cuda.hmax2(bf16_up, clamp_neg)
            bf16_up = K.cuda.hmin2(bf16_up, clamp_pos)

        gate = K.cuda.bfloat1622float2(bf16_gate)
        gate_x = K.cuda.float2_x(gate)
        gate_y = K.cuda.float2_y(gate)
        neg_gate_exp = K.cuda.make_float2(K.exp(-gate_x), K.exp(-gate_y))
        denom = K.cuda.fadd2_rn(K.cuda.make_float2(K.float32(1.0), K.float32(1.0)), neg_gate_exp)
        with K.If(kernel_fast_math):
            with K.Then():
                rcp_x = K.local_scalar("float32")
                rcp_y = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(rcp_x, K.cuda.float2_x(denom))
                K.ptx.rcp.approx.ftz.f32(rcp_y, K.cuda.float2_y(denom))
                gate = K.cuda.fmul2_rn(gate, K.cuda.make_float2(rcp_x, rcp_y))
            with K.Else():
                gate = K.cuda.make_float2(
                    gate_x / K.cuda.float2_x(denom), gate_y / K.cuda.float2_y(denom)
                )

        up = K.cuda.bfloat1622float2(bf16_up)
        weights = K.cuda.make_float2(weight0, weight1)
        result = K.cuda.fmul2_rn(K.cuda.fmul2_rn(gate, up), weights)
        K.ptx.mov.b32(out[atom_idx, pair_idx, 0], K.cuda.float2_x(result))
        K.ptx.mov.b32(out[atom_idx, pair_idx, 1], K.cuda.float2_y(result))

    def advance_umma_desc_lo(desc, base_lo, mn_offset, k_offset):
        return K.bitwise_or(
            K.bitwise_and(desc, K.shift_left(K.uint64(0xFFFFFFFF), K.uint64(32))),
            K.cast(base_lo + K.cast((mn_offset + k_offset) // f128_bytes, "uint32"), "uint64"),
        )

    def l2_cd_swizzled_elem_offset(half_idx, row_idx, bank_group_idx, swizzle_row):
        """Physical uint16 offset within one epilogue WG's L2 CD plane."""
        return (half_idx * kernel_config.store_block_m + row_idx) * (swizzle_cd_mode // 2) + (
            bank_group_idx ^ swizzle_row
        ) * (num_bank_group_bytes // 2)

    def scheduler_get_num_tokens(
        expert_idx, lane_idx, stored_num_tokens_per_expert, selected_num_tokens
    ):
        K.assign(selected_num_tokens[0], K.int32(0))
        for expert_lane_idx in range(num_experts_per_lane):
            K.assign(
                selected_num_tokens[0],
                K.Select(
                    expert_idx == expert_lane_idx * 32 + lane_idx,
                    K.cast(stored_num_tokens_per_expert[expert_lane_idx], "int32"),
                    selected_num_tokens[0],
                ),
            )
        expert_lane_idx_u32 = K.cast(expert_idx, "uint32") % K.uint32(32)
        K.assign(
            selected_num_tokens[0],
            K.tvm_warp_shuffle(
                K.uint32(0xFFFFFFFF),
                selected_num_tokens[0],
                K.cast(expert_lane_idx_u32, "int32"),
                32,
                32,
            ),
        )

    def scheduler_get_pool_block_offset(
        expert_idx, lane_idx, stored_num_tokens_per_expert, pool_block_offset_sum
    ):
        K.assign(pool_block_offset_sum[0], K.int32(0))
        for expert_lane_idx in range(num_experts_per_lane):
            expert_num_blocks_u32 = (
                stored_num_tokens_per_expert[expert_lane_idx] + K.uint32(kernel_config.block_m - 1)
            ) // K.uint32(kernel_config.block_m)
            K.assign(
                pool_block_offset_sum[0],
                pool_block_offset_sum[0]
                + K.Select(
                    expert_lane_idx * 32 + lane_idx < expert_idx,
                    K.cast(expert_num_blocks_u32, "int32"),
                    K.int32(0),
                ),
            )
        K.assign(
            pool_block_offset_sum[0],
            K.cast(
                reduce_add_sync_u32(
                    K.uint32(0xFFFFFFFF), K.cast(pool_block_offset_sum[0], "uint32")
                ),
                "int32",
            ),
        )

    def symm_rank_offset_arg_expr(symm_rank_offsets, mapped_rank_idx):
        if num_processes == 1:
            return symm_rank_offsets[0]
        mapped_rank_idx_u32 = K.cast(mapped_rank_idx, "uint32")
        rank_offset = symm_rank_offsets[0]
        for rank in range(1, num_processes):
            rank_offset = K.Select(
                mapped_rank_idx_u32 == K.uint32(rank), symm_rank_offsets[rank], rank_offset
            )
        return rank_offset

    def load_symm_rank_base(dst, smem_symm_rank_bases, mapped_rank_idx):
        if num_processes > 1:
            return K.ptx.ld.shared.u64(
                dst, smem_symm_rank_bases.ptr_to([K.cast(mapped_rank_idx, "int32")])
            )

    # ---- shared-memory budget, barrier numbering, and epilogue chunking ----
    sm100_smem_capacity = 232448
    shared_alignment = 1024
    f32_bytes = 4
    f128_bytes = 16
    num_epilogue_wgs = kernel_config.num_epilogue_wgs
    wg_block_m = kernel_config.wg_block_m
    atom_m = kernel_config.atom_m
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
    full_shared_b_expect_tx_leader_bytes: int = (smem_b_size_per_stage * 2) + (
        smem_sfb_size_per_stage * 2
    )
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
    # Combine reuses the prefix of SharedStorage up to the first barrier.
    smem_barrier_offset = (
        _align_up(
            smem_dispatch_size
            + smem_cd_size
            + num_stages
            * (
                smem_a_size_per_stage
                + smem_b_size_per_stage
                + smem_sfa_size_per_stage
                + smem_sfb_size_per_stage
            )
            + smem_amax_reduction_size,
            16,
        )
        + smem_task_info_size
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
    if num_chunk_uint4 % 32 != 0:
        raise ValueError("Combine chunk must be a multiple of 32 16-byte elements (one per lane)")
    if num_topk > 32:
        raise ValueError("Top-k must fit in a single warp")

    # ---- the kernel body ----
    @K.kernel(
        warps=kernel_config.num_total_warps,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=kernel_config.num_sms,
    )
    def mega_moe(
        y: K.gptr[K.bf16],
        cumulative_local_expert_recv_stats: K.gptr[K.i32],
        symm_buffer: K.gptr[K.i8],
        symm_rank_offset_0: K.i64,
        symm_rank_offset_1: K.i64,
        symm_rank_offset_2: K.i64,
        symm_rank_offset_3: K.i64,
        symm_rank_offset_4: K.i64,
        symm_rank_offset_5: K.i64,
        symm_rank_offset_6: K.i64,
        symm_rank_offset_7: K.i64,
        symm_rank_offset_8: K.i64,
        symm_rank_offset_9: K.i64,
        symm_rank_offset_10: K.i64,
        symm_rank_offset_11: K.i64,
        symm_rank_offset_12: K.i64,
        symm_rank_offset_13: K.i64,
        symm_rank_offset_14: K.i64,
        symm_rank_offset_15: K.i64,
        symm_rank_offset_16: K.i64,
        symm_rank_offset_17: K.i64,
        symm_rank_offset_18: K.i64,
        symm_rank_offset_19: K.i64,
        symm_rank_offset_20: K.i64,
        symm_rank_offset_21: K.i64,
        symm_rank_offset_22: K.i64,
        symm_rank_offset_23: K.i64,
        symm_rank_offset_24: K.i64,
        symm_rank_offset_25: K.i64,
        symm_rank_offset_26: K.i64,
        symm_rank_offset_27: K.i64,
        symm_rank_offset_28: K.i64,
        symm_rank_offset_29: K.i64,
        symm_rank_offset_30: K.i64,
        symm_rank_offset_31: K.i64,
        symm_rank_offset_32: K.i64,
        symm_rank_offset_33: K.i64,
        symm_rank_offset_34: K.i64,
        symm_rank_offset_35: K.i64,
        symm_rank_offset_36: K.i64,
        symm_rank_offset_37: K.i64,
        symm_rank_offset_38: K.i64,
        symm_rank_offset_39: K.i64,
        symm_rank_offset_40: K.i64,
        symm_rank_offset_41: K.i64,
        symm_rank_offset_42: K.i64,
        symm_rank_offset_43: K.i64,
        symm_rank_offset_44: K.i64,
        symm_rank_offset_45: K.i64,
        symm_rank_offset_46: K.i64,
        symm_rank_offset_47: K.i64,
        symm_rank_offset_48: K.i64,
        symm_rank_offset_49: K.i64,
        symm_rank_offset_50: K.i64,
        symm_rank_offset_51: K.i64,
        symm_rank_offset_52: K.i64,
        symm_rank_offset_53: K.i64,
        symm_rank_offset_54: K.i64,
        symm_rank_offset_55: K.i64,
        symm_rank_offset_56: K.i64,
        symm_rank_offset_57: K.i64,
        symm_rank_offset_58: K.i64,
        symm_rank_offset_59: K.i64,
        symm_rank_offset_60: K.i64,
        symm_rank_offset_61: K.i64,
        symm_rank_offset_62: K.i64,
        symm_rank_offset_63: K.i64,
        symm_rank_offset_64: K.i64,
        symm_rank_offset_65: K.i64,
        symm_rank_offset_66: K.i64,
        symm_rank_offset_67: K.i64,
        symm_rank_offset_68: K.i64,
        symm_rank_offset_69: K.i64,
        symm_rank_offset_70: K.i64,
        symm_rank_offset_71: K.i64,
        tensor_map_l1_acts: K.TensorMap,
        tensor_map_l1_acts_sf: K.TensorMap,
        tensor_map_l1_weights: K.TensorMap,
        tensor_map_l1_weights_sf: K.TensorMap,
        tensor_map_l1_output: K.TensorMap,
        tensor_map_l2_acts: K.TensorMap,
        tensor_map_l2_acts_sf: K.TensorMap,
        tensor_map_l2_weights: K.TensorMap,
        tensor_map_l2_weights_sf: K.TensorMap,
        # The nine shared-expert descriptors are always in the signature. At
        # `num_shared_experts == 0` the host binds the matching routed descriptor
        # into each slot as a dummy, mirroring the upstream launcher, so the ABI
        # is independent of S.
        tensor_map_shared_l1_acts: K.TensorMap,
        tensor_map_shared_l1_acts_sf: K.TensorMap,
        tensor_map_shared_l1_weights: K.TensorMap,
        tensor_map_shared_l1_weights_sf: K.TensorMap,
        tensor_map_shared_l1_output: K.TensorMap,
        tensor_map_shared_l2_acts: K.TensorMap,
        tensor_map_shared_l2_acts_sf: K.TensorMap,
        tensor_map_shared_l2_weights: K.TensorMap,
        tensor_map_shared_l2_weights_sf: K.TensorMap,
        num_tokens: K.i32,
        rank_idx: K.i32,
    ):
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

        def prefetch_all_tensormaps(prefetch_warp_idx):
            with K.If(prefetch_warp_idx == 0), K.Then():
                prefetch_tensormap(tensor_map_l1_acts)
                prefetch_tensormap(tensor_map_l1_acts_sf)
                prefetch_tensormap(tensor_map_l1_weights)
                prefetch_tensormap(tensor_map_l1_weights_sf)
                prefetch_tensormap(tensor_map_l1_output)
                prefetch_tensormap(tensor_map_l2_acts)
                prefetch_tensormap(tensor_map_l2_acts_sf)
                prefetch_tensormap(tensor_map_l2_weights)
                prefetch_tensormap(tensor_map_l2_weights_sf)
                # Unconditional in the source for every specialization: at S == 0
                # these alias the routed descriptors above (impls .cuh:106-114).
                prefetch_tensormap(tensor_map_shared_l1_acts)
                prefetch_tensormap(tensor_map_shared_l1_acts_sf)
                prefetch_tensormap(tensor_map_shared_l1_weights)
                prefetch_tensormap(tensor_map_shared_l1_weights_sf)
                prefetch_tensormap(tensor_map_shared_l1_output)
                prefetch_tensormap(tensor_map_shared_l2_acts)
                prefetch_tensormap(tensor_map_shared_l2_acts_sf)
                prefetch_tensormap(tensor_map_shared_l2_weights)
                prefetch_tensormap(tensor_map_shared_l2_weights_sf)

        input_topk_idx = K.decl_buffer(
            (workspace_layout.num_max_tokens_per_rank, num_topk),
            "int64",
            data=K.reinterpret(
                PointerType(PrimType("int64")),
                symm_buffer.ptr_to([symm_buffer_layout.input_topk_idx_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        l1_acts = K.decl_buffer(
            (num_ring_tokens, hidden),
            "int8",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l1_token_offset,
        )
        l1_acts_sf = K.decl_buffer(
            (hidden // 128, workspace_layout.num_sf_ring_tokens),
            "int32",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l1_sf_offset // 4,
        )
        workspace_expert_send_count = K.decl_buffer(
            (num_experts,),
            "uint64",
            data=K.reinterpret(
                PointerType(PrimType("uint64")),
                symm_buffer.ptr_to([workspace_layout.expert_send_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_grid_sync_count = K.decl_buffer(
            (4,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.barrier_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_nvl_barrier_counter = K.decl_buffer(
            (1,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.barrier_offset + 16]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_nvl_barrier_signal = K.decl_buffer(
            (2,),
            "int32",
            data=K.reinterpret(
                PointerType(PrimType("int32")),
                symm_buffer.ptr_to([workspace_layout.barrier_offset + 20]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_expert_recv_count = K.decl_buffer(
            (num_processes, num_experts_per_rank),
            "uint64",
            data=K.reinterpret(
                PointerType(PrimType("uint64")),
                symm_buffer.ptr_to([workspace_layout.expert_recv_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_expert_recv_count_sum = K.decl_buffer(
            (num_experts_per_rank,),
            "uint64",
            data=K.reinterpret(
                PointerType(PrimType("uint64")),
                symm_buffer.ptr_to([workspace_layout.expert_recv_count_sum_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_src_token_topk_idx = K.decl_buffer(
            (num_experts_per_rank, num_processes, workspace_layout.num_max_recv_tokens_per_expert),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.src_token_topk_idx_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_token_src_metadata = K.decl_buffer(
            (workspace_layout.num_max_pool_tokens, 3),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.token_src_metadata_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_l1_full_count = K.decl_buffer(
            (workspace_layout.num_ring_blocks,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.l1_full_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_l1_empty_count = K.decl_buffer(
            (workspace_layout.num_ring_blocks,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.l1_empty_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_l2_full_count = K.decl_buffer(
            (workspace_layout.num_ring_blocks,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.l2_full_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_l2_empty_count = K.decl_buffer(
            (workspace_layout.num_ring_blocks,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.l2_empty_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_l1_task_count = K.decl_buffer(
            (1,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.l1_task_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_l2_task_count = K.decl_buffer(
            (1,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.l2_task_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_shared_l1_task_count = K.decl_buffer(
            (1,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.shared_l1_task_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_shared_l2_task_count = K.decl_buffer(
            (1,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.shared_l2_task_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        workspace_shared_l2_full_count = K.decl_buffer(
            (workspace_layout.num_shared_l2_pool_blocks,),
            "uint32",
            data=K.reinterpret(
                PointerType(PrimType("uint32")),
                symm_buffer.ptr_to([workspace_layout.shared_l2_full_count_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        l1_topk_weights = K.decl_buffer(
            (num_ring_tokens,),
            "float32",
            data=K.reinterpret(
                PointerType(PrimType("float32")),
                symm_buffer.ptr_to([symm_buffer_layout.l1_topk_weights_offset]),
            ),
            scope="global",
            elem_offset=0,
        )
        l2_acts = K.decl_buffer(
            (num_ring_tokens, intermediate_hidden),
            "int8",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l2_token_offset,
        )
        l2_sf_buffer = K.decl_buffer(
            (intermediate_hidden // 128 * workspace_layout.num_sf_ring_tokens * 4,),
            "int8",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l2_sf_offset,
        )
        combine_tokens = K.decl_buffer(
            (num_combine_slots, workspace_layout.num_max_tokens_per_rank, hidden),
            "uint16",
            data=K.reinterpret(
                PointerType(PrimType("uint16")),
                symm_buffer.ptr_to([symm_buffer_layout.combine_token_offset]),
            ),
            scope="global",
            elem_offset=0,
        )

        smem = K.smem_pool()
        smem_expert_count = smem.alloc((num_experts,), K.i32, align=shared_alignment)
        smem_send_buffers = smem.alloc(
            (kernel_config.num_dispatch_warps, kernel_config.num_bytes_per_pull),
            K.i8,
            align=shared_alignment,
        )
        smem_cd_raw = smem.alloc((smem_cd_size,), K.u8, align=shared_alignment)
        smem_a_tile = smem.alloc(
            (num_stages, kernel_config.load_block_m, kernel_config.block_k),
            K.i8,
            align=shared_alignment,
            swizzle=K.SW128B,
        )
        smem_a = smem_a_tile.buf
        smem_a_fp8 = smem_a.view("float8_e4m3fn")
        smem_b_tile = smem.alloc(
            (num_stages, kernel_config.load_block_n, kernel_config.block_k),
            K.u8,
            align=shared_alignment,
            swizzle=K.SW128B,
        )
        smem_b = smem_b_tile.buf
        smem_sfa_i32 = smem.alloc((num_stages, sf_block_m, sf_smem_outer_dim), K.i32, align=16)
        smem_sfa_data = K.reinterpret(
            PointerType(PrimType("uint32")), smem_sfa_i32.ptr_to([0, 0, 0])
        )
        smem_sfa = K.decl_buffer(
            (num_stages, sf_block_m * sf_smem_outer_dim),
            "uint32",
            data=smem_sfa_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sfb_i32 = smem.alloc(
            (num_stages, kernel_config.block_n, sf_smem_outer_dim), K.i32, align=16
        )
        smem_sfb_data = K.reinterpret(
            PointerType(PrimType("uint32")), smem_sfb_i32.ptr_to([0, 0, 0])
        )
        smem_sfb = K.decl_buffer(
            (num_stages, kernel_config.block_n * sf_smem_outer_dim),
            "uint32",
            data=smem_sfb_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_amax_reduction = smem.alloc(
            (kernel_config.num_epilogue_warps * kernel_config.store_block_m,), K.f32, align=16
        )
        smem_task_infos = smem.alloc((num_schedule_stages, 8), K.u32, align=16)
        dispatch_barriers = K.TMABar(smem, kernel_config.num_dispatch_warps)
        full_barriers = K.TMABar(smem, num_stages)
        empty_barriers = K.TCGen05Bar(smem, num_stages)
        tmem_full_barriers = K.TCGen05Bar(smem, num_epilogue_stages)
        tmem_empty_barriers = K.MBarrier(smem, num_epilogue_stages)
        combine_barriers = K.TMABar(smem, kernel_config.num_epilogue_warps * 2)
        task_info_full_barriers = K.MBarrier(smem, num_schedule_stages)
        task_info_empty_barriers = K.MBarrier(smem, num_schedule_stages)
        tmem_ptr_in_smem = smem.alloc((1,), K.u32, align=4)
        if num_processes > 1:
            smem_symm_rank_bases = smem.alloc((num_processes,), K.u64, align=8)
        else:
            # The single-rank specialization never reads this view. Keep the
            # original zero-byte ABI instead of reserving an unreachable slot.
            smem_symm_rank_bases_data = K.reinterpret(
                PointerType(PrimType("uint64")), tmem_ptr_in_smem.ptr_to([0])
            )
            smem_symm_rank_bases = K.decl_buffer(
                (1,),
                "uint64",
                data=smem_symm_rank_bases_data,
                scope="shared.dyn",
                elem_offset=0,
                align=8,
            )
        if smem.bytes > sm100_smem_capacity:
            raise ValueError(
                f"MegaMoE K SMEM layout uses {smem.bytes} bytes, capacity is {sm100_smem_capacity}"
            )
        smem.commit()

        # Aliases must retain the pool allocation offset; sharing only `data` rebases to SMEM 0.
        smem_cd_l1 = K.decl_buffer(
            (num_tma_store_stages, num_epilogue_wgs, kernel_config.store_block_m, l1_out_block_n),
            "int8",
            data=smem_cd_raw.data,
            scope="shared.dyn",
            elem_offset=smem_cd_raw.elem_offset,
            align=16,
        )
        smem_cd_l2 = K.decl_buffer(
            (num_epilogue_wgs, kernel_config.store_block_m * kernel_config.block_n),
            "uint16",
            data=smem_cd_raw.data,
            scope="shared.dyn",
            # elem_offset is measured in the alias dtype, so retain the same byte base.
            elem_offset=smem_cd_raw.elem_offset // 2,
            align=16,
        )
        combine_chunks = K.decl_buffer(
            (num_chunk_slots, kernel_config.num_epilogue_warps, num_chunk_uint4, 4),
            "uint32",
            data=smem_expert_count.data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        tmem = K.decl_buffer(
            (128, num_tmem_cols), "float32", scope="tmem", allocated_addr=tmem_ptr_in_smem[0]
        )
        sfa_tmem = K.decl_buffer(
            (128, sf_block_m // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=num_accum_tmem_cols,
        )
        sfb_tmem = K.decl_buffer(
            (128, kernel_config.block_n // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=num_accum_tmem_cols + num_sfa_tmem_cols,
        )

        roles = K.specialize(chain_dispatch=True)
        dispatch_role = roles.role(
            "dispatch", warps=range(kernel_config.num_dispatch_warps), regs=num_dispatch_registers
        )
        load_a_role = roles.role(
            "load_a", warps=[kernel_config.load_a_warp_idx], regs=num_non_epilogue_registers
        )
        load_b_role = roles.role(
            "load_b", warps=[kernel_config.load_b_warp_idx], regs=num_non_epilogue_registers
        )
        mma_role = roles.role(
            "mma", warps=[kernel_config.mma_issue_warp_idx], regs=num_non_epilogue_registers
        )
        reserved_role = roles.role(
            "reserved",
            warps=[kernel_config.reserved_non_epilogue_warp_idx],
            regs=num_non_epilogue_registers,
        )
        epilogue_role = roles.role(
            "epilogue",
            warps=range(kernel_config.epilogue_warp_start_idx, kernel_config.num_total_warps),
            regs=num_epilogue_registers,
        )

        # `symm_buffer` is a kernel parameter, so its base address is a pure
        # reinterpret of an argument rather than a cvta of an allocation, and
        # `K.warp_id()` is the cached warp-uniform scope id, so splitting and
        # recombining it folds straight back to that id.  Re-emitting either
        # per use costs nothing.
        sym_buffer_base = ptr_to_u64(symm_buffer.ptr_to([0]))
        cta_idx_in_cluster = K.cta_id_in_cluster([kernel_config.num_ctas_per_cluster])
        sm_idx = K.cta_id()
        wg_id = K.warp_id() // 4
        warp_id = K.warp_id() % 4
        lane_idx = K.lane_id()
        flat_warp_idx = wg_id * 4 + warp_id
        prefetch_all_tensormaps(flat_warp_idx)

        desc_a = K.local_scalar("uint64")
        desc_b = K.local_scalar("uint64")
        desc_sf = K.local_scalar("uint64")
        desc_i = K.local_scalar("uint32")
        desc_i_shared = K.local_scalar("uint32")
        desc_i_active = K.local_scalar("uint32")
        runtime_desc_i = K.local_scalar("uint32")
        a_desc_lo = K.local_scalar("uint32")
        b_desc_lo = K.local_scalar("uint32")
        a_desc_base_lo = K.local_scalar("uint32")
        b_desc_base_lo = K.local_scalar("uint32")
        dispatch_token_iter = K.local_scalar("int32")
        dispatch_token_topk_idx = K.local_scalar("int32")
        dispatch_expert_idx = K.local_scalar("int32")
        dispatch_dst_rank_idx = K.local_scalar("int32")
        dispatch_dst_local_expert_idx = K.local_scalar("int32")
        dispatch_dst_slot_idx = K.local_scalar("int32")
        pull_local_expert_idx = K.local_scalar("int32")
        pull_num_tokens = K.local_scalar("int32")
        pull_pool_block_offset = K.local_scalar("int32")
        pull_src_token_topk_idx = K.local_scalar("int32")
        pull_src_token_idx = K.local_scalar("int32")
        pull_src_topk_idx = K.local_scalar("int32")
        pull_pool_token_idx = K.local_scalar("int32")
        pull_pool_block_idx = K.local_scalar("int32")
        pull_ring_block_idx = K.local_scalar("int32")
        pull_ring_token_idx = K.local_scalar("int32")
        token_idx_in_block = K.local_scalar("int32")
        l1_empty_count_target = K.local_scalar("int32")
        epilogue_value = K.local_scalar("float32")
        combine_accum = K.local_scalar("float32")
        gate_accum = K.local_scalar("float32")
        up_accum = K.local_scalar("float32")
        current_ring_count = K.local_scalar("uint32")
        barrier_status_printf = K.local_scalar("int32")
        atom_prev_unused = K.local_scalar("uint32")
        expected_ring_count = K.local_scalar("int32")
        scheduler_num_m_blocks = K.local_scalar("int32")
        scheduler_cached_status = K.local_scalar("uint64")
        ordinary_global_u64 = K.local_scalar("uint64")
        ordinary_global_u32 = K.local_scalar("uint32")
        ordinary_global_s64 = K.local_scalar("int64")
        symm_rank_base = K.local_scalar("uint64")
        nvl_counter_value = K.local_scalar("uint32")
        smem_expert_count_value = K.local_scalar("uint32")
        tmem_allocated = K.local_scalar("uint32")
        dst_rank_idx_u32 = K.local_scalar("uint32")
        dst_token_idx_u32 = K.local_scalar("uint32")
        dst_topk_idx_u32 = K.local_scalar("uint32")
        sched_num_total_m_blocks = K.local_scalar("int32")
        sched_num_l1_waves = K.local_scalar("uint32")
        sched_task_idx = K.local_scalar("uint32")
        sched_task_valid = K.local_scalar("int32")
        sched_block_offset = K.local_scalar("uint32")
        sched_expert_num_m_blocks = K.local_scalar("uint32")
        sched_inclusive_sum = K.local_scalar("uint32")
        sched_lane_pool_block_offset = K.local_scalar("uint32")
        sched_owner_mask = K.local_scalar("uint32")
        sched_owner_m_block_idx = K.local_scalar("uint32")
        sched_owner_valid_m = K.local_scalar("uint32")
        sched_required_l1_tasks = K.local_scalar("uint32")
        sched_shared_num_tasks = K.local_scalar("uint32")
        sched_shared_m_block_idx = K.local_scalar("uint32")
        sched_shared_valid_m = K.local_scalar("uint32")
        sched_shared_running = K.local_scalar("int32")
        current_expert_idx = K.local_scalar("int32")
        old_expert_idx = K.local_scalar("int32")
        expert_start_idx = K.local_scalar("int32")
        expert_end_idx = K.local_scalar("int32")
        token_idx_in_rank = K.local_scalar("int32")
        token_idx_in_expert = K.local_scalar("int32")
        current_rank_in_expert_idx = K.local_scalar("int32")
        rank_count_mask = K.local_scalar("uint32")
        num_active_ranks = K.local_scalar("int32")
        active_lane_count = K.local_scalar("int32")
        num_actives_in_lane = K.local_scalar("int32")
        min_in_lane = K.local_scalar("uint32")
        min_active_count = K.local_scalar("int32")
        round_token_count = K.local_scalar("int32")
        slot_idx_in_round = K.local_scalar("int32")
        round_offset = K.local_scalar("int32")
        barrier_status = K.local_scalar("int32")
        barrier_signal_phase = K.local_scalar("int32")
        barrier_signal_sign = K.local_scalar("int32")
        barrier_target = K.local_scalar("int32")
        epilogue_thread_idx = K.local_scalar("int32")
        sf_row_idx = K.local_scalar("int32")
        epilogue_wg_idx = K.local_scalar("int32")
        grid_sync_old_value = K.local_scalar("uint32")
        grid_sync_new_value = K.local_scalar("uint32")
        nvl_barrier_start_clock = K.local_scalar("uint64")
        k_idx_packed = K.local_scalar("int32")
        m_idx = K.local_scalar("int32")
        n_idx = K.local_scalar("int32")
        pool_block_idx = K.local_scalar("int32")
        ring_block_idx = K.local_scalar("int32")
        block_idx = K.local_scalar("int32")
        ring_m_idx = K.local_scalar("int32")
        pool_m_idx = K.local_scalar("int32")
        valid_m = K.local_scalar("int32")
        sfa_m_idx = K.local_scalar("int32")
        stored_num_tokens_per_expert = K.alloc_local((num_experts_per_lane,), "uint32")
        selected_num_tokens = K.alloc_local((1,), "int32")
        pool_block_offset_sum = K.alloc_local((1,), "int32")
        task_info_regs = K.alloc_local((8,), "uint32")
        sched_inclusive_vals = K.local_scalar("uint32")
        stored_rank_counts = K.alloc_local((num_ranks_per_lane,), "uint32")
        remaining_rank_counts = K.alloc_local((num_ranks_per_lane,), "uint32")
        combine_stored_topk_slot_idx = K.local_scalar("int32")
        combine_total_mask = K.local_scalar("uint32")
        combine_slot_mask = K.local_scalar("uint32")
        combine_slot_idx = K.local_scalar("int32")
        combine_token_idx = K.local_scalar("int32")
        combine_chunk_offset_elems = K.local_scalar("int32")
        combine_do_reduce = K.local_scalar("int32")
        combine_next_do_reduce = K.local_scalar("int32")
        has_accum_task = K.local_scalar("int32")
        accum_stage_idx = K.local_scalar("int32")
        accum_phase = K.local_scalar("int32")
        block_phase = K.local_scalar("int32")
        local_expert_idx = K.local_scalar("int32")
        num_k_blocks = K.local_scalar("int32")
        m_block_idx = K.local_scalar("int32")
        n_block_idx = K.local_scalar("int32")
        n_cluster_idx = K.local_scalar("int32")
        get_valid_m_true = K.local_scalar("int32")
        get_valid_m_true_half = K.local_scalar("int32")
        get_valid_m_true_eighth = K.local_scalar("int32")
        shape_k = K.local_scalar("int32")
        shape_n = K.local_scalar("int32")
        shape_sfa_k = K.local_scalar("int32")
        shape_sfb_k = K.local_scalar("int32")
        pull_state = K.PipelineState(1)
        sched_state = K.PipelineState(num_schedule_stages)
        pipeline_state = K.PipelineState(num_stages)
        accum_state = K.PipelineState(num_epilogue_stages)
        combine_state = K.PipelineState(2)

        def workspace_grid_sync(counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx):
            K.ptx.barrier.sync(K.uint32(sync_barrier_idx), K.uint32(sync_num_threads))
            with K.If(sync_thread_idx == 0), K.Then():
                with K.If(sm_idx == 0):
                    with K.Then():
                        atomic_add_rel_u32(
                            grid_sync_old_value,
                            workspace_grid_sync_count.ptr_to([counter_idx]),
                            K.uint32(0x80000000 - (kernel_config.num_sms - 1)),
                        )
                    with K.Else():
                        atomic_add_rel_u32(
                            grid_sync_old_value,
                            workspace_grid_sync_count.ptr_to([counter_idx]),
                            K.uint32(1),
                        )
                load_acq_u32(grid_sync_new_value, workspace_grid_sync_count.ptr_to([counter_idx]))
                with K.While(
                    grid_sync_done_u32(grid_sync_new_value, grid_sync_old_value) == K.uint32(0)
                ):
                    load_acq_u32(
                        grid_sync_new_value, workspace_grid_sync_count.ptr_to([counter_idx])
                    )
            K.ptx.barrier.sync(K.uint32(sync_barrier_idx), K.uint32(sync_num_threads))

        def nvlink_barrier(
            counter_idx,
            barrier_tag,
            sync_num_threads,
            sync_barrier_idx,
            sync_thread_idx,
            sync_prologue,
            sync_epilogue,
        ):
            with K.If(num_processes == 1):
                with K.Then():
                    workspace_grid_sync(
                        counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                    )
                with K.Else():
                    with K.If(sync_prologue != 0), K.Then():
                        workspace_grid_sync(
                            counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                        )
                    with K.If(sm_idx == 0), K.Then():
                        load_global_u32(
                            nvl_counter_value, workspace_nvl_barrier_counter.ptr_to([0])
                        )
                        K.assign(
                            barrier_status,
                            K.cast(K.bitwise_and(nvl_counter_value, K.uint32(3)), "int32"),
                        )
                        K.assign(barrier_signal_phase, K.bitwise_and(barrier_status, K.int32(1)))
                        K.assign(barrier_signal_sign, K.shift_right(barrier_status, K.int32(1)))
                        with K.If(sync_thread_idx < K.int32(num_processes)), K.Then():
                            K.assign(barrier_target, K.int32(1))
                            with K.If(barrier_signal_sign != 0), K.Then():
                                K.assign(barrier_target, K.int32(-1))
                            K.assign(
                                symm_rank_base,
                                sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64"),
                            )
                            load_symm_rank_base(
                                symm_rank_base, smem_symm_rank_bases, sync_thread_idx
                            )
                            peer_red_add_rel_sys_s32(
                                symm_rank_base,
                                K.uint64(workspace_layout.barrier_offset + 20)
                                + K.cast(barrier_signal_phase * 4, "uint64"),
                                barrier_target,
                            )
                        K.ptx.bar.sync(K.uint32(sync_barrier_idx), K.uint32(sync_num_threads))
                        with K.If(sync_thread_idx == 0), K.Then():
                            red_add_gpu_u32(workspace_nvl_barrier_counter.ptr_to([0]), K.uint32(1))
                            K.assign(barrier_target, K.int32(num_processes))
                            with K.If(barrier_signal_sign != 0), K.Then():
                                K.assign(barrier_target, K.int32(0))
                            load_acq_sys_s32(
                                barrier_status,
                                workspace_nvl_barrier_signal.ptr_to([barrier_signal_phase]),
                            )
                            K.assign(nvl_barrier_start_clock, cuda_clock64())
                            with K.While(barrier_status != barrier_target):
                                with (
                                    K.If(
                                        cuda_clock64() - nvl_barrier_start_clock
                                        >= K.uint64(num_nvlink_barrier_timeout_cycles)
                                    ),
                                    K.Then(),
                                ):
                                    with K.If(kernel_emit_nvl_barrier_timeout_printf), K.Then():
                                        load_acq_sys_s32(
                                            barrier_status_printf,
                                            workspace_nvl_barrier_signal.ptr_to(
                                                [barrier_signal_phase]
                                            ),
                                        )
                                        load_global_u32(
                                            nvl_counter_value,
                                            workspace_nvl_barrier_counter.ptr_to([0]),
                                        )
                                        K.cuda.printf(
                                            f"DeepGEMM NVLink barrier timeout "
                                            f"({nvlink_barrier_timeout_seconds}s): "
                                            "rank=%d, counter=%d, signal=%d, target=%d, "
                                            "phase=%d, sign=%d, tag=%d\n",
                                            rank_idx,
                                            K.cast(nvl_counter_value, "int32"),
                                            barrier_status_printf,
                                            barrier_target,
                                            barrier_signal_phase,
                                            barrier_signal_sign,
                                            barrier_tag,
                                        )
                                    K.cuda.trap_when_assert_failed(False)
                                load_acq_sys_s32(
                                    barrier_status,
                                    workspace_nvl_barrier_signal.ptr_to([barrier_signal_phase]),
                                )
                    with K.If(sync_epilogue != 0), K.Then():
                        workspace_grid_sync(
                            counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                        )

        def dispatch_nvlink_barrier_before_pull(thread_idx_in_scope):
            nvlink_barrier(
                dispatch_grid_sync_index,
                before_dispatch_pull_barrier_tag,
                kernel_config.num_dispatch_threads,
                dispatch_sync_barrier_idx,
                thread_idx_in_scope,
                0,
                1,
            )

        def dispatch_nvlink_barrier_after_workspace_clean(thread_idx_in_scope):
            nvlink_barrier(
                dispatch_grid_sync_index,
                after_workspace_clean_barrier_tag,
                kernel_config.num_dispatch_threads,
                dispatch_sync_barrier_idx,
                thread_idx_in_scope,
                1,
                0,
            )

        def epilogue_nvlink_barrier_before_combine_reduce(thread_idx_in_scope):
            nvlink_barrier(
                epilogue_grid_sync_index,
                before_combine_reduce_barrier_tag,
                kernel_config.num_epilogue_threads,
                epilogue_full_sync_barrier_idx,
                thread_idx_in_scope,
                1,
                1,
            )

        def scheduler_fetch_expert_recv_count():
            with K.serial(0, num_experts_per_lane) as expert_lane_idx:
                K.assign(dispatch_expert_idx, expert_lane_idx * 32 + lane_idx)
                K.assign(scheduler_cached_status, K.uint64(0))
                with K.If(dispatch_expert_idx < num_experts_per_rank), K.Then():
                    with K.While(
                        K.cast(K.shift_right(scheduler_cached_status, 32), "int32")
                        != kernel_config.num_sms * num_processes
                    ):
                        K.ptx.ld.volatile.global_.u64(
                            scheduler_cached_status,
                            workspace_expert_recv_count_sum.ptr_to([dispatch_expert_idx]),
                        )
                K.ptx.mov.b32(
                    stored_num_tokens_per_expert[expert_lane_idx],
                    K.cast(K.bitwise_and(scheduler_cached_status, K.uint64(0xFFFFFFFF)), "uint32"),
                )
            K.cuda.warp_sync()
            # `num_total_m_blocks = get_num_total_pool_blocks()` plus the L1 warmup
            # wave seed (scheduler/mega_moe.cuh `fetch_expert_recv_count`).
            scheduler_get_pool_block_offset(
                K.int32(num_experts_per_rank),
                lane_idx,
                stored_num_tokens_per_expert,
                pool_block_offset_sum,
            )
            K.assign(sched_num_total_m_blocks, pool_block_offset_sum[0])
            sched_num_total_m_blocks_u32 = K.cast(sched_num_total_m_blocks, "uint32")
            sched_num_total_l1_tasks = sched_num_total_m_blocks_u32 * K.uint32(num_l1_clusters)
            sched_num_total_l1_waves = (
                sched_num_total_l1_tasks + K.uint32(num_sched_clusters - 1)
            ) // K.uint32(num_sched_clusters)
            sched_warmup_interleave = (
                K.uint32(num_l1_clusters)
                + (sched_num_total_m_blocks_u32 - K.uint32(1))
                * K.uint32(sched_interleave_cluster_diff)
                + K.uint32(num_sched_clusters - 1)
            ) // K.uint32(num_sched_clusters) + K.uint32(1)
            sched_min_l1_warmup_waves = K.max(
                K.uint32(sched_l1_warmup_first_l2_wave), sched_warmup_interleave
            )
            K.assign(sched_num_l1_waves, K.min(sched_min_l1_warmup_waves, sched_num_total_l1_waves))

        def sched_advance_pipeline():
            sched_state.advance()

        def consumer_get_next_task():
            # `get_next_task`: wait for the published TaskInfo, copy it into
            # registers (2x LDS.128 of the alignas(16) 32-byte struct), advance.
            barrier_wait(task_info_full_barriers.ptr_to([sched_state.stage]), sched_state.phase)
            lds128(smem_task_infos.ptr_to([sched_state.stage, 0]), task_info_regs, 0)
            lds128(smem_task_infos.ptr_to([sched_state.stage, 4]), task_info_regs, 4)
            sched_advance_pipeline()

        def consumer_bind_task_args():
            K.assign(block_phase, K.cast(task_info_regs[0], "int32"))
            K.assign(local_expert_idx, K.cast(task_info_regs[1], "int32"))
            K.assign(m_block_idx, K.cast(task_info_regs[2], "int32"))
            K.assign(n_cluster_idx, K.cast(task_info_regs[3], "int32"))
            K.assign(pool_block_idx, K.cast(task_info_regs[4], "int32"))
            K.assign(valid_m, K.cast(task_info_regs[5], "int32"))
            K.assign(shape_n, K.cast(task_info_regs[6], "int32"))
            K.assign(shape_k, K.cast(task_info_regs[7], "int32"))

        def scheduler_release_task_info():
            # `release_task_info`: all epilogue threads (both CTAs) arrive at the
            # leader CTA's empty barrier of the just-consumed stage.
            _rem1 = K.local_scalar("uint64")
            K.ptx.mapa.shared__cluster.u64(
                _rem1,
                task_info_empty_barriers.ptr_to([sched_state.stage ^ K.int32(1)]),
                K.uint32(0),
            )
            K.ptx.mbarrier.arrive.b64(_rem1, K.uint32(1), pred=K.bool(True))

        def producer_create_task(task_block_phase, task_num_clusters, task_shape_n, task_shape_k):
            # `create_task`: resolve the owning expert / m-block / valid_m of the
            # pool block via a per-lane token-count scan + warp ballot.
            K.ptx.mov.b32(task_info_regs[0], K.uint32(task_block_phase))
            K.ptx.mov.b32(task_info_regs[1], K.uint32(0))
            K.ptx.mov.b32(task_info_regs[2], K.uint32(0))
            K.ptx.mov.b32(task_info_regs[3], sched_task_idx % K.uint32(task_num_clusters))
            K.ptx.mov.b32(task_info_regs[4], sched_task_idx // K.uint32(task_num_clusters))
            K.ptx.mov.b32(task_info_regs[5], K.uint32(0))
            K.ptx.mov.b32(task_info_regs[6], K.uint32(task_shape_n))
            K.ptx.mov.b32(task_info_regs[7], K.uint32(task_shape_k))
            K.assign(sched_block_offset, K.uint32(0))
            with K.unroll(0, num_experts_per_lane) as expert_lane_idx:
                K.assign(
                    sched_expert_num_m_blocks,
                    (
                        stored_num_tokens_per_expert[expert_lane_idx]
                        + K.uint32(kernel_config.block_m - 1)
                    )
                    // K.uint32(kernel_config.block_m),
                )
                # `math::warp_inclusive_sum`
                K.assign(sched_inclusive_vals, sched_expert_num_m_blocks)
                with K.unroll(0, 5) as shuffle_offset:
                    K.assign(
                        sched_inclusive_sum,
                        K.tvm_warp_shuffle_up(
                            K.uint32(0xFFFFFFFF), sched_inclusive_vals, 1 << shuffle_offset, 32, 32
                        ),
                    )
                    with K.If(lane_idx >= (1 << shuffle_offset)), K.Then():
                        K.assign(sched_inclusive_vals, (sched_inclusive_vals + sched_inclusive_sum))
                K.assign(
                    sched_lane_pool_block_offset,
                    (sched_block_offset + sched_inclusive_vals - sched_expert_num_m_blocks),
                )
                K.assign(
                    sched_owner_mask,
                    ballot_sync(
                        K.uint32(0xFFFFFFFF),
                        (
                            K.cast(expert_lane_idx * 32 + lane_idx, "uint32")
                            < K.uint32(num_experts_per_rank)
                        )
                        & (task_info_regs[4] >= sched_lane_pool_block_offset)
                        & (
                            task_info_regs[4]
                            < sched_lane_pool_block_offset + sched_expert_num_m_blocks
                        ),
                    ),
                )
                with K.If(sched_owner_mask != K.uint32(0)), K.Then():
                    sched_owner_lane_idx = ffs_u32(sched_owner_mask) - K.int32(1)
                    K.assign(
                        sched_owner_m_block_idx, (task_info_regs[4] - sched_lane_pool_block_offset)
                    )
                    K.assign(
                        sched_owner_valid_m,
                        K.min(
                            stored_num_tokens_per_expert[expert_lane_idx]
                            - sched_owner_m_block_idx * K.uint32(kernel_config.block_m),
                            K.uint32(kernel_config.block_m),
                        ),
                    )
                    K.ptx.mov.b32(
                        task_info_regs[1],
                        K.tvm_warp_shuffle(
                            K.uint32(0xFFFFFFFF),
                            K.cast(expert_lane_idx * 32 + lane_idx, "uint32"),
                            sched_owner_lane_idx,
                            32,
                            32,
                        ),
                    )
                    K.ptx.mov.b32(
                        task_info_regs[2],
                        K.tvm_warp_shuffle(
                            K.uint32(0xFFFFFFFF),
                            sched_owner_m_block_idx,
                            sched_owner_lane_idx,
                            32,
                            32,
                        ),
                    )
                    K.ptx.mov.b32(
                        task_info_regs[5],
                        K.tvm_warp_shuffle(
                            K.uint32(0xFFFFFFFF), sched_owner_valid_m, sched_owner_lane_idx, 32, 32
                        ),
                    )
                K.assign(
                    sched_block_offset,
                    sched_block_offset
                    + K.tvm_warp_shuffle(
                        K.uint32(0xFFFFFFFF), sched_inclusive_vals, K.int32(31), 32, 32
                    ),
                )

        def producer_get_next_task():
            # Producer-side `get_next_task`: interleave L1/L2 task pulls from the
            # global atomic counters with the L1 warmup-wave ordering.
            K.ptx.mov.b32(task_info_regs[0], K.uint32(0))
            K.assign(sched_task_valid, K.int32(0))
            with K.While(sched_task_valid == K.int32(0)):
                with K.If(
                    K.And(
                        sched_num_l1_waves != K.uint32(sched_l1_waves_done),
                        sched_num_l1_waves != K.uint32(0),
                    )
                ):
                    with K.Then():
                        K.assign(sched_num_l1_waves, sched_num_l1_waves - K.uint32(1))
                        K.assign(sched_task_idx, K.uint32(0))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            atomic_add_u32(
                                sched_task_idx, workspace_l1_task_count.ptr_to([0]), K.uint32(1)
                            )
                        K.assign(
                            sched_task_idx,
                            K.tvm_warp_shuffle(
                                K.uint32(0xFFFFFFFF), sched_task_idx, K.int32(0), 32, 32
                            ),
                        )
                        with K.If(
                            sched_task_idx
                            >= K.cast(sched_num_total_m_blocks, "uint32")
                            * K.uint32(num_l1_clusters)
                        ):
                            with K.Then():
                                K.assign(sched_num_l1_waves, K.uint32(sched_l1_waves_done))
                            with K.Else():
                                producer_create_task(
                                    1, num_l1_clusters, intermediate_hidden * 2, hidden
                                )
                                K.assign(sched_task_valid, K.int32(1))
                    with K.Else():
                        K.assign(sched_task_idx, K.uint32(0))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            atomic_add_u32(
                                sched_task_idx, workspace_l2_task_count.ptr_to([0]), K.uint32(1)
                            )
                        K.assign(
                            sched_task_idx,
                            K.tvm_warp_shuffle(
                                K.uint32(0xFFFFFFFF), sched_task_idx, K.int32(0), 32, 32
                            ),
                        )
                        with (
                            K.If(
                                sched_task_idx
                                >= K.cast(sched_num_total_m_blocks, "uint32")
                                * K.uint32(num_l2_clusters)
                            ),
                            K.Then(),
                        ):
                            K.Break()
                        with K.If(sched_num_l1_waves != K.uint32(sched_l1_waves_done)), K.Then():
                            K.assign(sched_num_l1_waves, K.uint32(1))
                        producer_create_task(2, num_l2_clusters, hidden, intermediate_hidden)
                        # Wait until all required L1 tasks are fetched
                        K.assign(
                            sched_required_l1_tasks,
                            (task_info_regs[4] + K.uint32(1)) * K.uint32(num_l1_clusters),
                        )
                        sched_l1_count = K.local_scalar("uint32")
                        K.assign(sched_l1_count, K.uint32(0))
                        load_volatile_u32(sched_l1_count, workspace_l1_task_count.ptr_to([0]))
                        with K.While(sched_l1_count < sched_required_l1_tasks):
                            load_volatile_u32(sched_l1_count, workspace_l1_task_count.ptr_to([0]))
                        K.assign(sched_task_valid, K.int32(1))

        def producer_publish_task():
            # `publish_task`: lanes 0/1 arrive-and-expect-tx at each CTA's full
            # barrier, then st.async the 32-byte TaskInfo into that CTA's smem.
            with K.If(lane_idx < K.int32(2)), K.Then():
                _rem_ti = K.local_scalar("uint64")
                K.ptx.mapa.shared__cluster.u64(
                    _rem_ti, task_info_full_barriers.ptr_to([sched_state.stage]), K.uint32(lane_idx)
                )
                K.ptx.mbarrier.arrive.expect_tx.release.cluster.b64(
                    _rem_ti, K.uint32(task_info_bytes), pred=K.bool(True)
                )
                st_async_cluster_task_info(
                    smem_task_infos.ptr_to([sched_state.stage, 0]),
                    task_info_full_barriers.ptr_to([sched_state.stage]),
                    lane_idx,
                    task_info_regs,
                )
            K.cuda.warp_sync()
            sched_advance_pipeline()

        def producer_shared_mainloop(
            task_phase, task_count_ptr, task_num_clusters, task_shape_n, task_shape_k
        ):
            # `shared_mainloop` (scheduler/mega_moe.cuh:364-381). Dynamic
            # scheduling over `ceil_div(num_tokens, BLOCK_M) * kNumNClusters`
            # tasks: the shared FFN is rank-local, so the m-block index is the
            # token block directly and doubles as the (non-ring) pool block.
            K.assign(
                sched_shared_num_tasks,
                (
                    (K.cast(num_tokens, "uint32") + K.uint32(kernel_config.block_m - 1))
                    // K.uint32(kernel_config.block_m)
                )
                * K.uint32(task_num_clusters),
            )
            K.assign(sched_shared_running, K.int32(1))
            with K.While(sched_shared_running != K.int32(0)):
                barrier_wait(
                    task_info_empty_barriers.ptr_to([sched_state.stage]),
                    sched_state.phase ^ K.int32(1),
                )
                K.assign(sched_task_idx, K.uint32(0))
                with K.If(K.cuda.elect_sync()), K.Then():
                    atomic_add_u32(sched_task_idx, task_count_ptr, K.uint32(1))
                K.assign(
                    sched_task_idx,
                    K.tvm_warp_shuffle(K.uint32(0xFFFFFFFF), sched_task_idx, K.int32(0), 32, 32),
                )
                with K.If(sched_task_idx >= sched_shared_num_tasks):
                    with K.Then():
                        K.assign(sched_shared_running, K.int32(0))
                    with K.Else():
                        K.assign(
                            sched_shared_m_block_idx, sched_task_idx // K.uint32(task_num_clusters)
                        )
                        K.assign(
                            sched_shared_valid_m,
                            K.min(
                                K.cast(num_tokens, "uint32")
                                - sched_shared_m_block_idx * K.uint32(kernel_config.block_m),
                                K.uint32(kernel_config.block_m),
                            ),
                        )
                        K.ptx.mov.b32(task_info_regs[0], K.uint32(task_phase))
                        K.ptx.mov.b32(task_info_regs[1], K.uint32(0))
                        K.ptx.mov.b32(task_info_regs[2], sched_shared_m_block_idx)
                        K.ptx.mov.b32(
                            task_info_regs[3], sched_task_idx % K.uint32(task_num_clusters)
                        )
                        K.ptx.mov.b32(task_info_regs[4], sched_shared_m_block_idx)
                        K.ptx.mov.b32(task_info_regs[5], sched_shared_valid_m)
                        K.ptx.mov.b32(task_info_regs[6], K.uint32(task_shape_n))
                        K.ptx.mov.b32(task_info_regs[7], K.uint32(task_shape_k))
                        producer_publish_task()

        def update_get_valid_m_true():
            valid_m_u32 = K.cast(valid_m, "uint32")
            get_valid_m_true_u32 = (valid_m_u32 + K.uint32(15)) // K.uint32(16) * K.uint32(16)
            K.assign(get_valid_m_true, K.cast(get_valid_m_true_u32, "int32"))
            K.assign(get_valid_m_true_half, K.cast(get_valid_m_true_u32 // K.uint32(2), "int32"))
            K.assign(get_valid_m_true_eighth, K.cast(get_valid_m_true_u32 // K.uint32(8), "int32"))

        def advance_pipeline():
            pipeline_state.advance()

        def barrier_wait(barrier_ptr, phase):
            K.cuda.mbarrier_wait(barrier_ptr, phase)

        def tmem_empty_barrier_arrive_cta0(tmem_empty_barrier_ptr):
            _rem2 = K.local_scalar("uint64")
            K.ptx.mapa.shared__cluster.u64(_rem2, tmem_empty_barrier_ptr, K.uint32(0))
            K.ptx.mbarrier.arrive.b64(_rem2, K.uint32(1), pred=K.bool(True))

        def umma_arrive_multicast_2x1sm(barrier_ptr):
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx[
                    f"tcgen05.commit.cta_group::{kernel_config.num_ctas_per_cluster}"
                    ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                ](barrier_ptr, K.uint16(tcgen05_cta_mask))

        def umma_arrive(barrier_ptr):
            umma_arrive_multicast_2x1sm(barrier_ptr)

        def empty_barrier_arrive(do_tmem_full_arrive, empty_barrier_ptr, tmem_full_barrier_ptr):
            umma_arrive(empty_barrier_ptr)
            with K.If(do_tmem_full_arrive), K.Then():
                umma_arrive(tmem_full_barrier_ptr)
            K.cuda.warp_sync()

        def empty_barrier_arrive_current(do_tmem_full_arrive):
            empty_barrier_arrive(
                do_tmem_full_arrive,
                empty_barriers.ptr_to([pipeline_state.stage]),
                tmem_full_barriers.ptr_to([accum_stage_idx]),
            )

        def fence_view_async_tmem_load():
            K.ptx.tcgen05.wait__ld.sync.aligned()

        def tma_copy_2d_multicast_select(
            dst_ptr,
            barrier_ptr,
            tensor_map_l1_ptr,
            tensor_map_l2_ptr,
            block_phase_value,
            coord0,
            coord1,
            tensor_map_shared_l1_ptr=None,
            tensor_map_shared_l2_ptr=None,
        ):
            sm100_tma_2sm_load_2d_select(
                dst_ptr,
                barrier_ptr,
                tensor_map_l1_ptr,
                tensor_map_l2_ptr,
                block_phase_value,
                coord0,
                coord1,
                tensor_map_shared_l1_ptr,
                tensor_map_shared_l2_ptr,
            )

        def epilogue_signal_routed_l1_done():
            atomic_add_rel_u32(
                atom_prev_unused, workspace_l2_full_count.ptr_to([ring_block_idx]), K.uint32(1)
            )
            red_add_gpu_u32(workspace_l1_empty_count.ptr_to([ring_block_idx]), K.uint32(1))

        def epilogue_wait_l2_empty():
            K.assign(
                expected_ring_count,
                (hidden // kernel_config.block_n * (pool_block_idx // num_ring_blocks)),
            )
            load_acq_u32(current_ring_count, workspace_l2_empty_count.ptr_to([ring_block_idx]))
            with K.While(current_ring_count != K.cast(expected_ring_count, "uint32")):
                load_acq_u32(current_ring_count, workspace_l2_empty_count.ptr_to([ring_block_idx]))

        def load_a_wait_l1_full():
            K.assign(
                expected_ring_count, kernel_config.block_m * (pool_block_idx // num_ring_blocks + 1)
            )
            load_acq_u32(current_ring_count, workspace_l1_full_count.ptr_to([block_idx]))
            with K.While(current_ring_count != K.cast(expected_ring_count, "uint32")):
                load_acq_u32(current_ring_count, workspace_l1_full_count.ptr_to([block_idx]))

        def load_a_wait_l2_full():
            K.assign(
                expected_ring_count,
                (
                    intermediate_hidden
                    // kernel_config.block_n
                    * 2
                    * (pool_block_idx // num_ring_blocks + 1)
                ),
            )
            load_acq_u32(current_ring_count, workspace_l2_full_count.ptr_to([block_idx]))
            with K.While(current_ring_count != K.cast(expected_ring_count, "uint32")):
                load_acq_u32(current_ring_count, workspace_l2_full_count.ptr_to([block_idx]))

        def sm90_tma_store_2d_copy_select(
            src_ptr, tensor_map, tensor_map_shared, block_phase_value, coord0, coord1
        ):
            # `task_info.is_shared() ? &tensor_map_shared_l1_output
            #                        : &tensor_map_l1_output` (impls .cuh:1162)
            tma_store_2d_addr(
                src_ptr,
                K.if_then_else(
                    has_shared,
                    K.Select(
                        block_phase_value > K.int32(2),
                        K.address_of(tensor_map_shared),
                        K.address_of(tensor_map),
                    ),
                    K.address_of(tensor_map),
                ),
                coord0,
                coord1,
            )

        def full_barrier_arrive_and_expect_tx(full_barrier_ptr, transaction_bytes):
            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                full_barrier_ptr, K.uint32(transaction_bytes)
            )

        def full_barrier_arrive_cta0(full_barrier_ptr):
            _rem3 = K.local_scalar("uint64")
            K.ptx.mapa.shared__cluster.u64(_rem3, full_barrier_ptr, K.uint32(0))
            K.ptx.mbarrier.arrive.b64(_rem3, K.uint32(1), pred=K.bool(True))

        def make_instr_desc_block_scaled():
            K.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                K.address_of(desc_i),
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

        def make_instr_desc_block_scaled_shared():
            K.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                K.address_of(desc_i_shared),
                d_dtype="float32",
                a_dtype="float8_e4m3fn",
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

        def make_sf_desc():
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc_sf), smem_sfa.ptr_to([0, 0]), ldo=0, sdo=sf_desc_sdo, swizzle=0
            )

        def make_umma_desc_a():
            # Intentional raw boundary: descriptor LBO is correctness-sensitive, and
            # typed generic construction has a documented physical-LBO mismatch. This
            # kernel also forwards per-stage low halves across lanes to preserve the
            # exact encoder/update instruction sequence.
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc_a), smem_a_fp8.ptr_to([0, 0, 0]), ldo=0, sdo=a_desc_sdo, swizzle=3
            )

        def make_umma_desc_b():
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc_b), smem_b.ptr_to([0, 0, 0]), ldo=0, sdo=b_desc_sdo, swizzle=3
            )

        def utccp_copy(tmem_addr, sf_desc):
            K.ptx[f"tcgen05.cp.cta_group::{kernel_config.num_ctas_per_cluster}.32x128b.warpx4"](
                K.cast(tmem_addr, "uint32"), sf_desc
            )

        def sm100_u8x4_stsm_t_copy(fp8x4_word, smem_ptr):
            K.ptx.stmatrix.sync.aligned.m16n8.x1.trans.shared.b8(smem_ptr, fp8x4_word)

        def sm90_u32x4_stsm_t_copy(packed_values_buf, smem_ptr):
            K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                smem_ptr,
                packed_values_buf[0],
                packed_values_buf[1],
                packed_values_buf[2],
                packed_values_buf[3],
            )

        def store_token_src_metadata(pool_token_idx, src_rank_idx, src_token_idx, src_topk_idx):
            store_global_u32(
                workspace_token_src_metadata.ptr_to([pool_token_idx, 0]),
                K.cast(src_rank_idx, "uint32"),
            )
            store_global_u32(
                workspace_token_src_metadata.ptr_to([pool_token_idx, 1]),
                K.cast(src_token_idx, "uint32"),
            )
            store_global_u32(
                workspace_token_src_metadata.ptr_to([pool_token_idx, 2]),
                K.cast(src_topk_idx, "uint32"),
            )

        def init_gemm_context():
            K.assign(has_accum_task, 0)
            K.assign(accum_stage_idx, 0)
            K.assign(accum_phase, 0)
            K.assign(block_phase, 0)
            K.assign(local_expert_idx, 0)
            K.assign(num_k_blocks, 0)
            K.assign(m_block_idx, 0)
            K.assign(n_block_idx, 0)
            K.assign(n_cluster_idx, 0)
            K.assign(get_valid_m_true, 0)
            K.assign(get_valid_m_true_half, 0)
            K.assign(get_valid_m_true_eighth, 0)
            K.assign(shape_k, 0)
            K.assign(shape_n, 0)
            K.assign(shape_sfa_k, 0)
            K.assign(shape_sfb_k, 0)

        def prologue():
            # Relaxed arrive — no prior memory effect needs to be released to peers
            # before TMEM alloc + mbarrier init below. Wait still .acquire (default).
            K.ptx.barrier.cluster.arrive.relaxed.aligned()
            K.ptx.barrier.cluster.wait.acquire.aligned()
            full_barrier_init_count = 2 * 2
            tmem_empty_barrier_init_count = 2 * kernel_config.num_epilogue_threads
            with K.If(flat_warp_idx == 0):
                with K.Then():
                    with K.If(num_processes > 1), K.Then():
                        with K.If(lane_idx < num_processes), K.Then():
                            K.ptx.st.shared.u64(
                                smem_symm_rank_bases.ptr_to([lane_idx]),
                                sym_buffer_base
                                + K.cast(
                                    symm_rank_offset_arg_expr(symm_rank_offsets, lane_idx), "uint64"
                                ),
                            )
                    with K.If(K.cuda.elect_sync()), K.Then():
                        st_shared_bulk(smem_expert_count.ptr_to([0]), K.uint32(num_experts * 4))
                with K.Else():
                    with K.If(flat_warp_idx == 1):
                        with K.Then():
                            K.assign(dispatch_expert_idx, lane_idx)
                            with K.While(dispatch_expert_idx < kernel_config.num_dispatch_warps):
                                K.ptx.mbarrier.init.shared.b64(
                                    dispatch_barriers.ptr_to([dispatch_expert_idx]), K.uint32(1)
                                )
                                K.assign(dispatch_expert_idx, dispatch_expert_idx + 32)
                            fence_barrier_init()
                        with K.Else():
                            with K.If(flat_warp_idx == 2):
                                with K.Then():
                                    with K.If(K.cuda.elect_sync()), K.Then():
                                        K.assign(dispatch_expert_idx, K.int32(0))
                                        with K.While(dispatch_expert_idx < num_stages):
                                            K.ptx.mbarrier.init.shared.b64(
                                                full_barriers.ptr_to([dispatch_expert_idx]),
                                                K.uint32(full_barrier_init_count),
                                            )
                                            K.ptx.mbarrier.init.shared.b64(
                                                empty_barriers.ptr_to([dispatch_expert_idx]),
                                                K.uint32(1),
                                            )
                                            K.assign(dispatch_expert_idx, (dispatch_expert_idx + 1))
                                        K.assign(dispatch_expert_idx, K.int32(0))
                                        with K.While(dispatch_expert_idx < num_epilogue_stages):
                                            K.ptx.mbarrier.init.shared.b64(
                                                tmem_full_barriers.ptr_to([dispatch_expert_idx]),
                                                K.uint32(1),
                                            )
                                            K.ptx.mbarrier.init.shared.b64(
                                                tmem_empty_barriers.ptr_to([dispatch_expert_idx]),
                                                K.uint32(tmem_empty_barrier_init_count),
                                            )
                                            K.assign(dispatch_expert_idx, (dispatch_expert_idx + 1))
                                        K.assign(dispatch_expert_idx, K.int32(0))
                                        with K.While(
                                            dispatch_expert_idx
                                            < kernel_config.num_epilogue_warps * 2
                                        ):
                                            K.ptx.mbarrier.init.shared.b64(
                                                combine_barriers.ptr_to([dispatch_expert_idx]),
                                                K.uint32(1),
                                            )
                                            K.assign(dispatch_expert_idx, (dispatch_expert_idx + 1))
                                        K.assign(dispatch_expert_idx, K.int32(0))
                                        with K.While(dispatch_expert_idx < num_schedule_stages):
                                            K.ptx.mbarrier.init.shared.b64(
                                                task_info_full_barriers.ptr_to(
                                                    [dispatch_expert_idx]
                                                ),
                                                K.uint32(1),
                                            )
                                            K.ptx.mbarrier.init.shared.b64(
                                                task_info_empty_barriers.ptr_to(
                                                    [dispatch_expert_idx]
                                                ),
                                                K.uint32(num_schedule_consumer_threads),
                                            )
                                            K.assign(dispatch_expert_idx, (dispatch_expert_idx + 1))
                                    fence_barrier_init()
                                with K.Else():
                                    with (
                                        K.If(flat_warp_idx == kernel_config.num_dispatch_warps - 1),
                                        K.Then(),
                                    ):
                                        K.ptx[
                                            f"tcgen05.alloc.cta_group::{kernel_config.num_ctas_per_cluster}.sync.aligned.shared::cta.b32"
                                        ](
                                            K.address_of(tmem_ptr_in_smem[0]),
                                            K.uint32(num_tmem_cols),
                                        )
            # `fence_barrier_init` publishes the earlier mbarrier initialization,
            # but the tcgen05.alloc above performs a later weak shared-memory store
            # of the TMEM base.  Publish that store before epilogue warps read it.
            K.ptx.barrier.cluster.arrive.release.aligned()
            K.ptx.barrier.cluster.wait.acquire.aligned()

        def dispatch(role_warp_idx, role_thread_idx):
            K.assign(
                dispatch_token_iter,
                (sm_idx * kernel_config.num_dispatch_warps + role_warp_idx)
                * kernel_config.num_tokens_per_warp,
            )
            with K.While(K.cast(dispatch_token_iter, "uint32") < K.cast(num_tokens, "uint32")):
                with K.If(lane_idx < kernel_config.num_activate_lanes), K.Then():
                    lane_idx_u32 = K.cast(lane_idx, "uint32")
                    token_idx = dispatch_token_iter + K.cast(
                        lane_idx_u32 // K.uint32(num_topk), "int32"
                    )
                    with K.If(K.cast(token_idx, "uint32") < K.cast(num_tokens, "uint32")), K.Then():
                        topk_idx = K.cast(lane_idx_u32 % K.uint32(num_topk), "int32")
                        load_global_s64(
                            ordinary_global_s64, input_topk_idx.ptr_to([token_idx, topk_idx])
                        )
                        K.assign(dispatch_expert_idx, K.cast(ordinary_global_s64, "int32"))
                        with K.If(dispatch_expert_idx >= 0), K.Then():
                            K.evaluate(
                                K.cuda.atomic_add(
                                    smem_expert_count.ptr_to([dispatch_expert_idx]), 1
                                )
                            )
                K.cuda.warp_sync()
                K.assign(
                    dispatch_token_iter,
                    (
                        dispatch_token_iter
                        + kernel_config.num_sms
                        * kernel_config.num_dispatch_warps
                        * kernel_config.num_tokens_per_warp
                    ),
                )

            K.ptx.bar.sync(
                K.uint32(dispatch_sync_barrier_idx), K.uint32(kernel_config.num_dispatch_threads)
            )
            K.assign(dispatch_expert_idx, role_thread_idx)
            with K.While(K.cast(dispatch_expert_idx, "uint32") < K.uint32(num_experts)):
                load_shared_u32(
                    smem_expert_count_value, smem_expert_count.ptr_to([dispatch_expert_idx])
                )
                send_value = K.bitwise_or(
                    K.uint64(1 << 32), K.cast(smem_expert_count_value, "uint64")
                )
                prev_send_count = K.local_scalar("uint64")
                K.ptx.atom.global_.add.u64(
                    prev_send_count,
                    workspace_expert_send_count.ptr_to([dispatch_expert_idx]),
                    send_value,
                )
                K.ptx.st.shared.u32(
                    smem_expert_count.ptr_to([dispatch_expert_idx]),
                    K.cast(prev_send_count, "uint32"),
                )
                K.assign(
                    dispatch_expert_idx, (dispatch_expert_idx + kernel_config.num_dispatch_threads)
                )
            K.ptx.bar.sync(
                K.uint32(dispatch_sync_barrier_idx), K.uint32(kernel_config.num_dispatch_threads)
            )
            K.assign(
                dispatch_token_iter,
                (sm_idx * kernel_config.num_dispatch_warps + role_warp_idx)
                * kernel_config.num_tokens_per_warp,
            )
            with K.While(K.cast(dispatch_token_iter, "uint32") < K.cast(num_tokens, "uint32")):
                with K.If(lane_idx < kernel_config.num_activate_lanes), K.Then():
                    lane_idx_u32 = K.cast(lane_idx, "uint32")
                    token_idx = dispatch_token_iter + K.cast(
                        lane_idx_u32 // K.uint32(num_topk), "int32"
                    )
                    with K.If(K.cast(token_idx, "uint32") < K.cast(num_tokens, "uint32")), K.Then():
                        topk_idx = K.cast(lane_idx_u32 % K.uint32(num_topk), "int32")
                        K.assign(dispatch_token_topk_idx, token_idx * num_topk + topk_idx)
                        load_global_s64(
                            ordinary_global_s64, input_topk_idx.ptr_to([token_idx, topk_idx])
                        )
                        K.assign(dispatch_expert_idx, K.cast(ordinary_global_s64, "int32"))
                        with K.If(dispatch_expert_idx >= 0), K.Then():
                            dispatch_expert_idx_u32 = K.cast(dispatch_expert_idx, "uint32")
                            K.assign(
                                dispatch_dst_rank_idx,
                                K.cast(
                                    dispatch_expert_idx_u32 // K.uint32(num_experts_per_rank),
                                    "int32",
                                ),
                            )
                            K.assign(
                                dispatch_dst_local_expert_idx,
                                K.cast(
                                    dispatch_expert_idx_u32 % K.uint32(num_experts_per_rank),
                                    "int32",
                                ),
                            )
                            K.assign(
                                dispatch_dst_slot_idx,
                                K.cuda.atomic_add(
                                    smem_expert_count.ptr_to([dispatch_expert_idx]), 1
                                ),
                            )
                            K.assign(
                                symm_rank_base,
                                sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64"),
                            )
                            load_symm_rank_base(
                                symm_rank_base, smem_symm_rank_bases, dispatch_dst_rank_idx
                            )
                            peer_store_u32(
                                symm_rank_base,
                                K.uint64(
                                    workspace_layout.src_token_topk_idx_offset
                                    + (
                                        dispatch_dst_local_expert_idx
                                        * num_processes
                                        * workspace_layout.num_max_recv_tokens_per_expert
                                        + rank_idx * workspace_layout.num_max_recv_tokens_per_expert
                                        + dispatch_dst_slot_idx
                                    )
                                    * 4
                                ),
                                K.cast(dispatch_token_topk_idx, "uint32"),
                            )
                K.cuda.warp_sync()
                K.assign(
                    dispatch_token_iter,
                    (
                        dispatch_token_iter
                        + kernel_config.num_sms
                        * kernel_config.num_dispatch_warps
                        * kernel_config.num_tokens_per_warp
                    ),
                )
            workspace_grid_sync(
                0, kernel_config.num_dispatch_threads, dispatch_sync_barrier_idx, role_thread_idx
            )
            with K.If(sm_idx == 0), K.Then():
                K.assign(dispatch_expert_idx, role_thread_idx)
                with K.While(K.cast(dispatch_expert_idx, "uint32") < K.uint32(num_experts)):
                    dispatch_expert_idx_u32 = K.cast(dispatch_expert_idx, "uint32")
                    K.assign(
                        dispatch_dst_rank_idx,
                        K.cast(dispatch_expert_idx_u32 // K.uint32(num_experts_per_rank), "int32"),
                    )
                    K.assign(
                        dispatch_dst_local_expert_idx,
                        K.cast(dispatch_expert_idx_u32 % K.uint32(num_experts_per_rank), "int32"),
                    )
                    load_global_u64(
                        scheduler_cached_status,
                        workspace_expert_send_count.ptr_to([dispatch_expert_idx]),
                    )
                    K.assign(
                        symm_rank_base, sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64")
                    )
                    load_symm_rank_base(symm_rank_base, smem_symm_rank_bases, dispatch_dst_rank_idx)
                    peer_store_u64(
                        symm_rank_base,
                        K.uint64(
                            workspace_layout.expert_recv_count_offset
                            + (rank_idx * num_experts_per_rank + dispatch_dst_local_expert_idx) * 8
                        ),
                        K.bitwise_and(scheduler_cached_status, K.uint64(0xFFFFFFFF)),
                    )
                    peer_recv_count_sum_prev = K.local_scalar(
                        "uint64"
                    )  # atom returns the old value; unused here
                    K.assign(
                        symm_rank_base, sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64")
                    )
                    load_symm_rank_base(symm_rank_base, smem_symm_rank_bases, dispatch_dst_rank_idx)
                    peer_atomic_add_u64(
                        peer_recv_count_sum_prev,
                        symm_rank_base,
                        K.uint64(
                            workspace_layout.expert_recv_count_sum_offset
                            + dispatch_dst_local_expert_idx * 8
                        ),
                        scheduler_cached_status,
                    )
                    K.assign(
                        dispatch_expert_idx,
                        (dispatch_expert_idx + kernel_config.num_dispatch_threads),
                    )
            K.ptx.bar.sync(
                K.uint32(dispatch_sync_barrier_idx), K.uint32(kernel_config.num_dispatch_threads)
            )
            dispatch_nvlink_barrier_before_pull(role_thread_idx)
            sync_unaligned(
                dispatch_with_epilogue_sync_barrier_idx,
                kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
            )
            pull_state.init(0)
            K.assign(current_expert_idx, K.int32(-1))
            K.assign(old_expert_idx, K.int32(-1))
            K.assign(expert_start_idx, K.int32(0))
            K.assign(expert_end_idx, K.int32(0))
            K.assign(pull_pool_block_offset, K.int32(0))
            # Wait token data arrival
            scheduler_fetch_expert_recv_count()
            K.assign(dispatch_token_iter, sm_idx * kernel_config.num_dispatch_warps + role_warp_idx)
            with K.While(True):
                K.assign(old_expert_idx, current_expert_idx)
                with K.While(
                    K.cast(dispatch_token_iter, "uint32") >= K.cast(expert_end_idx, "uint32")
                ):
                    K.assign(current_expert_idx, current_expert_idx + K.int32(1))
                    with (
                        K.If(
                            K.cast(current_expert_idx, "uint32") >= K.uint32(num_experts_per_rank)
                        ),
                        K.Then(),
                    ):
                        K.Break()
                    expert_token_count_u32 = K.cast(expert_end_idx - expert_start_idx, "uint32")
                    expert_num_blocks_u32 = (
                        expert_token_count_u32 + K.uint32(kernel_config.block_m - 1)
                    ) // K.uint32(kernel_config.block_m)
                    K.assign(
                        pull_pool_block_offset,
                        pull_pool_block_offset + K.cast(expert_num_blocks_u32, "int32"),
                    )
                    K.assign(expert_start_idx, expert_end_idx)
                    scheduler_get_num_tokens(
                        current_expert_idx,
                        lane_idx,
                        stored_num_tokens_per_expert,
                        selected_num_tokens,
                    )
                    K.assign(expert_end_idx, expert_end_idx + selected_num_tokens[0])
                with (
                    K.If(K.cast(current_expert_idx, "uint32") >= K.uint32(num_experts_per_rank)),
                    K.Then(),
                ):
                    K.Break()
                with K.If(old_expert_idx != current_expert_idx), K.Then():
                    K.assign(old_expert_idx, current_expert_idx)
                    with K.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                        K.assign(dispatch_dst_rank_idx, rank_lane_idx * 32 + lane_idx)
                        K.ptx.mov.b32(stored_rank_counts[rank_lane_idx], K.uint32(0))
                        with (
                            K.If(K.cast(dispatch_dst_rank_idx, "uint32") < K.uint32(num_processes)),
                            K.Then(),
                        ):
                            load_global_u64(
                                ordinary_global_u64,
                                workspace_expert_recv_count.ptr_to(
                                    [dispatch_dst_rank_idx, current_expert_idx]
                                ),
                            )
                            K.ptx.mov.b32(
                                stored_rank_counts[rank_lane_idx],
                                K.cast(ordinary_global_u64, "uint32"),
                            )
                K.assign(token_idx_in_expert, dispatch_token_iter - expert_start_idx)
                K.assign(dispatch_dst_slot_idx, token_idx_in_expert)
                K.assign(round_offset, K.int32(0))
                with K.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                    K.ptx.mov.b32(
                        remaining_rank_counts[rank_lane_idx], stored_rank_counts[rank_lane_idx]
                    )
                with K.While(True):
                    K.assign(min_in_lane, K.uint32(0xFFFFFFFF))
                    K.assign(num_actives_in_lane, K.int32(0))
                    with K.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                        with K.If(remaining_rank_counts[rank_lane_idx] > K.uint32(0)), K.Then():
                            K.assign(num_actives_in_lane, num_actives_in_lane + K.int32(1))
                            K.assign(
                                min_in_lane,
                                K.min(min_in_lane, remaining_rank_counts[rank_lane_idx]),
                            )
                    K.assign(
                        num_active_ranks,
                        K.cast(
                            reduce_add_sync_u32(
                                K.uint32(0xFFFFFFFF), K.cast(num_actives_in_lane, "uint32")
                            ),
                            "int32",
                        ),
                    )
                    K.assign(
                        min_active_count,
                        K.cast(
                            K.cuda.reduce_min_sync_u32(K.uint32(0xFFFFFFFF), min_in_lane), "int32"
                        ),
                    )
                    K.assign(round_token_count, min_active_count * num_active_ranks)
                    with (
                        K.If(
                            K.cast(dispatch_dst_slot_idx, "uint32")
                            < K.cast(round_token_count, "uint32")
                        ),
                        K.Then(),
                    ):
                        dispatch_dst_slot_idx_u32 = K.cast(dispatch_dst_slot_idx, "uint32")
                        num_active_ranks_u32 = K.cast(num_active_ranks, "uint32")
                        K.assign(
                            slot_idx_in_round,
                            K.cast(dispatch_dst_slot_idx_u32 % num_active_ranks_u32, "int32"),
                        )
                        num_seen_ranks = K.int32(0)
                        K.assign(current_rank_in_expert_idx, K.int32(0))
                        with K.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                            K.assign(
                                rank_count_mask,
                                ballot_sync(
                                    K.uint32(0xFFFFFFFF),
                                    remaining_rank_counts[rank_lane_idx] > K.uint32(0),
                                ),
                            )
                            K.assign(
                                active_lane_count, K.cast(K.popcount(rank_count_mask), "int32")
                            )
                            with (
                                K.If(
                                    K.And(
                                        K.cast(slot_idx_in_round, "uint32")
                                        >= K.cast(num_seen_ranks, "uint32"),
                                        K.cast(slot_idx_in_round, "uint32")
                                        < K.cast(num_seen_ranks + active_lane_count, "uint32"),
                                    )
                                ),
                                K.Then(),
                            ):
                                rank_slot_bit = K.local_scalar("uint32")
                                K.ptx.fns.b32(
                                    rank_slot_bit,
                                    rank_count_mask,
                                    K.uint32(0),
                                    slot_idx_in_round - num_seen_ranks + K.int32(1),
                                )
                                K.assign(
                                    current_rank_in_expert_idx,
                                    (rank_lane_idx * 32 + K.cast(rank_slot_bit, "int32")),
                                )
                            num_seen_ranks = num_seen_ranks + active_lane_count
                        K.assign(
                            token_idx_in_rank,
                            round_offset
                            + K.cast(dispatch_dst_slot_idx_u32 // num_active_ranks_u32, "int32"),
                        )
                        K.Break()
                    K.assign(dispatch_dst_slot_idx, (dispatch_dst_slot_idx - round_token_count))
                    K.assign(round_offset, round_offset + min_active_count)
                    with K.unroll(0, num_ranks_per_lane) as rank_lane_idx:
                        K.ptx.mov.b32(
                            remaining_rank_counts[rank_lane_idx],
                            remaining_rank_counts[rank_lane_idx]
                            - K.min(
                                remaining_rank_counts[rank_lane_idx],
                                K.cast(min_active_count, "uint32"),
                            ),
                        )
                load_global_u32(
                    ordinary_global_u32,
                    workspace_src_token_topk_idx.ptr_to(
                        [current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank]
                    ),
                )
                K.assign(pull_src_token_topk_idx, K.cast(ordinary_global_u32, "int32"))
                pull_src_token_topk_idx_u32 = K.cast(pull_src_token_topk_idx, "uint32")
                K.assign(
                    pull_src_token_idx,
                    K.cast(pull_src_token_topk_idx_u32 // K.uint32(num_topk), "int32"),
                )
                K.assign(
                    pull_src_topk_idx,
                    K.cast(pull_src_token_topk_idx_u32 % K.uint32(num_topk), "int32"),
                )
                K.assign(
                    pull_pool_token_idx,
                    (pull_pool_block_offset * kernel_config.block_m + token_idx_in_expert),
                )
                K.assign(pull_pool_block_idx, pull_pool_token_idx // kernel_config.block_m)
                K.assign(pull_ring_block_idx, pull_pool_block_idx % num_ring_blocks)
                K.assign(pull_ring_token_idx, pull_pool_token_idx % num_ring_tokens)
                K.assign(
                    l1_empty_count_target,
                    (pull_pool_block_idx // num_ring_blocks * num_l1_block_ns),
                )
                with K.If(l1_empty_count_target > 0), K.Then():
                    load_acq_u32(
                        current_ring_count, workspace_l1_empty_count.ptr_to([pull_ring_block_idx])
                    )
                    with K.While(current_ring_count < K.cast(l1_empty_count_target, "uint32")):
                        load_acq_u32(
                            current_ring_count,
                            workspace_l1_empty_count.ptr_to([pull_ring_block_idx]),
                        )
                with K.If(K.cuda.elect_sync()), K.Then():
                    with K.unroll(0, num_pull_chunks) as pull_chunk_idx:
                        K.assign(
                            symm_rank_base, sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64")
                        )
                        load_symm_rank_base(
                            symm_rank_base, smem_symm_rank_bases, current_rank_in_expert_idx
                        )
                        tma_load_1d_symm(
                            smem_send_buffers.ptr_to([role_warp_idx, 0]),
                            symm_rank_base,
                            K.uint64(
                                symm_buffer_layout.input_token_offset
                                + pull_src_token_idx * hidden
                                + pull_chunk_idx * kernel_config.num_bytes_per_pull
                            ),
                            dispatch_barriers.ptr_to([role_warp_idx]),
                            kernel_config.num_bytes_per_pull,
                        )
                        mbarrier_arrive_and_set_tx(
                            dispatch_barriers.ptr_to([role_warp_idx]),
                            kernel_config.num_bytes_per_pull,
                        )
                        with K.If(pull_chunk_idx != num_pull_chunks - 1), K.Then():
                            mbarrier_wait_phase(
                                dispatch_barriers.ptr_to([role_warp_idx]), pull_state.phase
                            )
                            pull_state.advance()
                            tma_store_1d(
                                K.address_of(
                                    l1_acts[
                                        pull_ring_token_idx,
                                        pull_chunk_idx * kernel_config.num_bytes_per_pull,
                                    ]
                                ),
                                smem_send_buffers.ptr_to([role_warp_idx, 0]),
                                kernel_config.num_bytes_per_pull,
                            )
                            tma_store_arrive()
                            tma_store_wait(0)
                K.cuda.warp_sync()
                K.assign(token_idx_in_block, token_idx_in_expert % kernel_config.block_m)
                K.assign(
                    sf_row_idx,
                    (pull_ring_block_idx * sf_block_m + transform_sf_token_idx(token_idx_in_block)),
                )
                K.assign(dispatch_dst_rank_idx, lane_idx)
                with K.While(dispatch_dst_rank_idx < hidden // 128):
                    pulled_sf = K.local_scalar("uint32")
                    K.assign(
                        symm_rank_base, sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64")
                    )
                    load_symm_rank_base(
                        symm_rank_base, smem_symm_rank_bases, current_rank_in_expert_idx
                    )
                    peer_load_u32(
                        pulled_sf,
                        symm_rank_base,
                        K.uint64(
                            symm_buffer_layout.input_sf_offset
                            + (pull_src_token_idx * (hidden // 128) + dispatch_dst_rank_idx) * 4
                        ),
                    )
                    store_global_u32(
                        l1_acts_sf.ptr_to([dispatch_dst_rank_idx, sf_row_idx]), pulled_sf
                    )
                    K.assign(dispatch_dst_rank_idx, dispatch_dst_rank_idx + 32)
                K.cuda.warp_sync()
                with K.If(K.cuda.elect_sync()), K.Then():
                    pulled_weight = K.local_scalar("float32")
                    K.assign(
                        symm_rank_base, sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64")
                    )
                    load_symm_rank_base(
                        symm_rank_base, smem_symm_rank_bases, current_rank_in_expert_idx
                    )
                    peer_load_f32(
                        pulled_weight,
                        symm_rank_base,
                        K.uint64(
                            symm_buffer_layout.input_topk_weights_offset
                            + pull_src_token_topk_idx * 4
                        ),
                    )
                    K.ptx.st.global_.f32(
                        l1_topk_weights.ptr_to([pull_ring_token_idx]), pulled_weight
                    )
                    store_token_src_metadata(
                        pull_pool_token_idx,
                        current_rank_in_expert_idx,
                        pull_src_token_idx,
                        pull_src_topk_idx,
                    )
                    mbarrier_wait_phase(dispatch_barriers.ptr_to([role_warp_idx]), pull_state.phase)
                    pull_state.advance()
                    tma_store_1d(
                        K.address_of(
                            l1_acts[
                                pull_ring_token_idx,
                                (num_pull_chunks - 1) * kernel_config.num_bytes_per_pull,
                            ]
                        ),
                        smem_send_buffers.ptr_to([role_warp_idx, 0]),
                        kernel_config.num_bytes_per_pull,
                    )
                    tma_store_arrive()
                    tma_store_wait(0)
                    atomic_add_rel_u32(
                        atom_prev_unused,
                        workspace_l1_full_count.ptr_to([pull_ring_block_idx]),
                        K.Select(
                            dispatch_token_iter == expert_end_idx - K.int32(1),
                            K.uint32(kernel_config.block_m) - K.cast(token_idx_in_block, "uint32"),
                            K.uint32(1),
                        ),
                    )
                K.cuda.warp_sync()
                K.assign(
                    dispatch_token_iter,
                    (
                        dispatch_token_iter
                        + kernel_config.num_sms * kernel_config.num_dispatch_warps
                    ),
                )
            sync_unaligned(
                dispatch_with_epilogue_sync_barrier_idx,
                kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
            )
            # Dispatch and load-A read reusable global workspace after the
            # preceding grid rendezvous. The epilogue grid rendezvous cannot
            # publish those reads because it precedes this CTA-local join.
            workspace_grid_sync(
                dispatch_grid_sync_index,
                kernel_config.num_dispatch_threads + 32,
                dispatch_with_load_a_sync_barrier_idx,
                role_thread_idx,
            )
            with K.If(sm_idx == 0):
                with K.Then():
                    # SM 0: clear expert send count and schedule task counters
                    K.assign(dispatch_expert_idx, role_thread_idx)
                    with K.While(dispatch_expert_idx < num_experts):
                        store_global_u64(
                            workspace_expert_send_count.ptr_to([dispatch_expert_idx]), K.uint64(0)
                        )
                        K.assign(
                            dispatch_expert_idx,
                            (dispatch_expert_idx + kernel_config.num_dispatch_threads),
                        )
                    with K.If((role_warp_idx == 0) & K.cuda.elect_sync() != 0), K.Then():
                        store_global_u32(workspace_l1_task_count.ptr_to([0]), K.uint32(0))
                        store_global_u32(workspace_l2_task_count.ptr_to([0]), K.uint32(0))
                        store_global_u32(workspace_shared_l1_task_count.ptr_to([0]), K.uint32(0))
                        store_global_u32(workspace_shared_l2_task_count.ptr_to([0]), K.uint32(0))
                    K.cuda.warp_sync()
                    K.assign(dispatch_expert_idx, role_thread_idx)
                    with K.While(dispatch_expert_idx < workspace_layout.num_shared_l2_pool_blocks):
                        store_global_u32(
                            workspace_shared_l2_full_count.ptr_to([dispatch_expert_idx]),
                            K.uint32(0),
                        )
                        K.assign(
                            dispatch_expert_idx,
                            (dispatch_expert_idx + kernel_config.num_dispatch_threads),
                        )
                    K.cuda.warp_sync()
                with K.Else():
                    K.assign(pull_local_expert_idx, sm_idx - 1)
                    with K.While(pull_local_expert_idx < num_experts_per_rank):
                        scheduler_get_num_tokens(
                            pull_local_expert_idx,
                            lane_idx,
                            stored_num_tokens_per_expert,
                            selected_num_tokens,
                        )
                        K.assign(pull_num_tokens, selected_num_tokens[0])
                        pull_num_tokens_u32 = K.cast(pull_num_tokens, "uint32")
                        K.assign(
                            scheduler_num_m_blocks,
                            K.cast(
                                (pull_num_tokens_u32 + K.uint32(kernel_config.block_m - 1))
                                // K.uint32(kernel_config.block_m),
                                "int32",
                            ),
                        )
                        scheduler_get_pool_block_offset(
                            pull_local_expert_idx,
                            lane_idx,
                            stored_num_tokens_per_expert,
                            pool_block_offset_sum,
                        )
                        K.assign(pull_pool_block_offset, pool_block_offset_sum[0])
                        K.ptx.bar.sync(
                            K.uint32(dispatch_sync_barrier_idx),
                            K.uint32(kernel_config.num_dispatch_threads),
                        )
                        with K.If(role_thread_idx == 0), K.Then():
                            store_global_u64(
                                workspace_expert_recv_count_sum.ptr_to([pull_local_expert_idx]),
                                K.uint64(0),
                            )
                        with (
                            K.If(
                                K.And(
                                    K.And(kernel_collect_stats, role_warp_idx == 1), lane_idx == 0
                                )
                            ),
                            K.Then(),
                        ):
                            K.ptx.red.gpu.global_.add.s32(
                                cumulative_local_expert_recv_stats.ptr_to([pull_local_expert_idx]),
                                pull_num_tokens,
                            )
                        K.assign(dispatch_dst_rank_idx, role_thread_idx)
                        with K.While(dispatch_dst_rank_idx < K.int32(num_processes)):
                            store_global_u64(
                                workspace_expert_recv_count.ptr_to(
                                    [dispatch_dst_rank_idx, pull_local_expert_idx]
                                ),
                                K.uint64(0),
                            )
                            K.assign(
                                dispatch_dst_rank_idx,
                                (dispatch_dst_rank_idx + kernel_config.num_dispatch_threads),
                            )
                        K.assign(dispatch_dst_slot_idx, role_thread_idx)
                        with K.While(dispatch_dst_slot_idx < scheduler_num_m_blocks):
                            K.assign(
                                pull_ring_block_idx,
                                (pull_pool_block_offset + dispatch_dst_slot_idx) % num_ring_blocks,
                            )
                            store_global_u32(
                                workspace_l1_full_count.ptr_to([pull_ring_block_idx]), K.uint32(0)
                            )
                            store_global_u32(
                                workspace_l1_empty_count.ptr_to([pull_ring_block_idx]), K.uint32(0)
                            )
                            store_global_u32(
                                workspace_l2_full_count.ptr_to([pull_ring_block_idx]), K.uint32(0)
                            )
                            store_global_u32(
                                workspace_l2_empty_count.ptr_to([pull_ring_block_idx]), K.uint32(0)
                            )
                            K.assign(
                                dispatch_dst_slot_idx,
                                (dispatch_dst_slot_idx + kernel_config.num_dispatch_threads),
                            )
                        K.assign(
                            pull_local_expert_idx,
                            pull_local_expert_idx + (kernel_config.num_sms - 1),
                        )
            dispatch_nvlink_barrier_after_workspace_clean(role_thread_idx)

        def load_a():
            init_gemm_context()
            sched_state.init(0)
            pipeline_state.init(0)
            with K.While(True):
                consumer_get_next_task()
                consumer_bind_task_args()
                with K.If(block_phase == K.int32(0)), K.Then():
                    K.Break()
                shape_k_u32 = K.cast(shape_k, "uint32")
                K.assign(
                    num_k_blocks,
                    K.cast(
                        (shape_k_u32 + K.uint32(kernel_config.block_k - 1))
                        // K.uint32(kernel_config.block_k),
                        "int32",
                    ),
                )
                K.assign(ring_block_idx, pool_block_idx % num_ring_blocks)
                if has_shared:
                    K.assign(
                        block_idx,
                        K.Select(block_phase > K.int32(2), pool_block_idx, ring_block_idx),
                    )
                else:
                    K.assign(block_idx, ring_block_idx)
                if has_shared:
                    with K.If(block_phase == K.int32(1)):
                        with K.Then():
                            load_a_wait_l1_full()
                        with K.Else():
                            with K.If(block_phase == K.int32(2)):
                                with K.Then():
                                    load_a_wait_l2_full()
                                with K.Else():
                                    with (
                                        K.If(block_phase == K.int32(block_phase_shared_l2)),
                                        K.Then(),
                                    ):
                                        # SharedLinear2 waits for its full-size (non-ring) block,
                                        # so there is no wave multiplier. SharedLinear1 reads the
                                        # host-written input tokens and waits for nothing.
                                        K.assign(
                                            expected_ring_count,
                                            (shared_l2_shape_k // kernel_config.block_n * 2),
                                        )
                                        load_acq_u32(
                                            current_ring_count,
                                            workspace_shared_l2_full_count.ptr_to([block_idx]),
                                        )
                                        with K.While(
                                            current_ring_count
                                            != K.cast(expected_ring_count, "uint32")
                                        ):
                                            load_acq_u32(
                                                current_ring_count,
                                                workspace_shared_l2_full_count.ptr_to([block_idx]),
                                            )
                else:
                    with K.If(block_phase == K.int32(1)):
                        with K.Then():
                            load_a_wait_l1_full()
                        with K.Else():
                            load_a_wait_l2_full()
                # The release/acquire counters publish generic global stores
                # from dispatch (L1) or the preceding epilogue's scale-factor
                # writes (L2).  TMA reads through the async proxy, so bridge the
                # acquired frontier before loading either ring.
                K.ptx.fence.proxy.async_.global_()
                with K.serial(0, num_k_blocks) as k_block_idx:
                    barrier_wait(
                        empty_barriers.ptr_to([pipeline_state.stage]),
                        pipeline_state.phase ^ K.int32(1),
                    )
                    K.assign(m_idx, block_idx * kernel_config.block_m)
                    k_idx = k_block_idx * kernel_config.block_k
                    K.assign(sfa_m_idx, block_idx * sf_block_m)
                    sfa_k_idx = k_block_idx * sf_smem_outer_dim
                    with K.If(cta_idx_in_cluster != 0), K.Then():
                        update_get_valid_m_true()
                        K.assign(m_idx, m_idx + get_valid_m_true_half)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        full_barrier_ptr = full_barriers.ptr_to([pipeline_state.stage])
                        with K.unroll(0, kernel_config.block_k // umma_block_k) as tma_k_atom_idx:
                            tma_copy_2d_multicast_select(
                                smem_a.ptr_to(
                                    [pipeline_state.stage, 0, tma_k_atom_idx * umma_block_k]
                                ),
                                full_barrier_ptr,
                                tensor_map_l1_acts,
                                tensor_map_l2_acts,
                                block_phase,
                                k_idx + tma_k_atom_idx * umma_block_k,
                                m_idx,
                                tensor_map_shared_l1_acts,
                                tensor_map_shared_l2_acts,
                            )
                        tma_copy_2d_multicast_select(
                            smem_sfa_i32.ptr_to([pipeline_state.stage, 0, 0]),
                            full_barrier_ptr,
                            tensor_map_l1_acts_sf,
                            tensor_map_l2_acts_sf,
                            block_phase,
                            sfa_m_idx,
                            sfa_k_idx,
                            tensor_map_shared_l1_acts_sf,
                            tensor_map_shared_l2_acts_sf,
                        )
                        with K.If(cta_idx_in_cluster == 0):
                            with K.Then():
                                full_barrier_arrive_and_expect_tx(
                                    full_barrier_ptr, full_a_expect_tx_leader_bytes
                                )
                            with K.Else():
                                full_barrier_arrive_cta0(full_barrier_ptr)
                    K.cuda.warp_sync()
                    advance_pipeline()
            workspace_grid_sync(
                dispatch_grid_sync_index,
                kernel_config.num_dispatch_threads + 32,
                dispatch_with_load_a_sync_barrier_idx,
                kernel_config.num_dispatch_threads + lane_idx,
            )

        def load_b():
            init_gemm_context()
            sched_state.init(0)
            pipeline_state.init(0)
            with K.While(True):
                consumer_get_next_task()
                consumer_bind_task_args()
                with K.If(block_phase == K.int32(0)), K.Then():
                    K.Break()
                shape_k_u32 = K.cast(shape_k, "uint32")
                K.assign(
                    shape_sfb_k, K.cast((shape_k_u32 + K.uint32(127)) // K.uint32(128), "int32")
                )
                K.assign(
                    n_block_idx,
                    n_cluster_idx * 2 + K.Select(cta_idx_in_cluster == 0, K.int32(0), K.int32(1)),
                )
                K.assign(
                    num_k_blocks,
                    K.cast(
                        (shape_k_u32 + K.uint32(kernel_config.block_k - 1))
                        // K.uint32(kernel_config.block_k),
                        "int32",
                    ),
                )
                with K.serial(0, num_k_blocks) as k_block_idx:
                    barrier_wait(
                        empty_barriers.ptr_to([pipeline_state.stage]),
                        pipeline_state.phase ^ K.int32(1),
                    )
                    # Shared tasks have no expert-group dimension, so they drop
                    # the `local_expert_idx` group offset (impls .cuh:762, :765).
                    if has_shared:
                        K.assign(
                            n_idx,
                            K.Select(
                                block_phase > K.int32(2),
                                n_block_idx * kernel_config.block_n,
                                local_expert_idx * shape_n + n_block_idx * kernel_config.block_n,
                            ),
                        )
                    else:
                        K.assign(
                            n_idx,
                            (local_expert_idx * shape_n + n_block_idx * kernel_config.block_n),
                        )
                    k_idx = k_block_idx * kernel_config.block_k
                    sfb_n_idx = n_block_idx * kernel_config.block_n
                    if has_shared:
                        sfb_k_idx = K.Select(
                            block_phase > K.int32(2),
                            k_block_idx * sf_smem_outer_dim,
                            local_expert_idx * shape_sfb_k + k_block_idx * sf_smem_outer_dim,
                        )
                    else:
                        sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx * sf_smem_outer_dim
                    with K.If(K.cuda.elect_sync()), K.Then():
                        full_barrier_ptr = full_barriers.ptr_to([pipeline_state.stage])
                        with K.unroll(0, kernel_config.block_k // umma_block_k) as tma_k_atom_idx:
                            tma_copy_2d_multicast_select(
                                smem_b.ptr_to(
                                    [pipeline_state.stage, 0, tma_k_atom_idx * umma_block_k]
                                ),
                                full_barrier_ptr,
                                tensor_map_l1_weights,
                                tensor_map_l2_weights,
                                block_phase,
                                k_idx + tma_k_atom_idx * umma_block_k,
                                n_idx,
                                tensor_map_shared_l1_weights,
                                tensor_map_shared_l2_weights,
                            )
                        tma_copy_2d_multicast_select(
                            smem_sfb_i32.ptr_to([pipeline_state.stage, 0, 0]),
                            full_barrier_ptr,
                            tensor_map_l1_weights_sf,
                            tensor_map_l2_weights_sf,
                            block_phase,
                            sfb_n_idx,
                            sfb_k_idx,
                            tensor_map_shared_l1_weights_sf,
                            tensor_map_shared_l2_weights_sf,
                        )
                        with K.If(cta_idx_in_cluster == 0):
                            with K.Then():
                                # Routed weights are FP4-packed in gmem, so a 2-CTA
                                # multicast moves exactly `smem_b` bytes; the shared
                                # weights are plain e4m3 and move twice that
                                # (impls .cuh:775 vs :785).
                                if has_shared:
                                    full_barrier_arrive_and_expect_tx(
                                        full_barrier_ptr,
                                        K.Select(
                                            block_phase > K.int32(2),
                                            K.uint32(full_shared_b_expect_tx_leader_bytes),
                                            K.uint32(full_b_expect_tx_leader_bytes),
                                        ),
                                    )
                                else:
                                    full_barrier_arrive_and_expect_tx(
                                        full_barrier_ptr, full_b_expect_tx_leader_bytes
                                    )
                            with K.Else():
                                full_barrier_arrive_cta0(full_barrier_ptr)
                    K.cuda.warp_sync()
                    advance_pipeline()

        def mma():
            init_gemm_context()
            with K.If(cta_idx_in_cluster == 0), K.Then():
                make_instr_desc_block_scaled()
                if has_shared:
                    make_instr_desc_block_scaled_shared()
                make_sf_desc()
                make_umma_desc_a()
                make_umma_desc_b()
                K.assign(
                    a_desc_lo,
                    K.Select(
                        lane_idx < K.int32(num_stages),
                        K.cast(K.bitwise_and(desc_a, K.uint64(0xFFFFFFFF)), "uint32")
                        + K.cast(lane_idx * (smem_a_size_per_stage // f128_bytes), "uint32"),
                        K.uint32(0),
                    ),
                )
                K.assign(
                    b_desc_lo,
                    K.Select(
                        lane_idx < K.int32(num_stages),
                        K.cast(K.bitwise_and(desc_b, K.uint64(0xFFFFFFFF)), "uint32")
                        + K.cast(lane_idx * (smem_b_size_per_stage // f128_bytes), "uint32"),
                        K.uint32(0),
                    ),
                )
                sched_state.init(0)
                pipeline_state.init(0)
                accum_state.init(0)
                with K.While(True):
                    consumer_get_next_task()
                    consumer_bind_task_args()
                    with K.If(block_phase == K.int32(0)), K.Then():
                        K.Break()
                    K.assign(
                        num_k_blocks,
                        K.cast(
                            K.cast(shape_k, "uint32") // K.uint32(kernel_config.block_k), "int32"
                        ),
                    )
                    K.assign(accum_stage_idx, accum_state.stage)
                    K.assign(accum_phase, accum_state.phase)
                    accum_state.advance()
                    K.assign(has_accum_task, K.int32(1))
                    update_get_valid_m_true()
                    # Dynamic UMMA-N update on the descriptor for this task kind
                    # (impls .cuh:835-836).
                    if has_shared:
                        K.assign(
                            desc_i_active,
                            K.bitwise_or(
                                K.bitwise_and(
                                    K.Select(block_phase > K.int32(2), desc_i_shared, desc_i),
                                    K.uint32(0xFF81FFFF),
                                ),
                                K.shift_left(
                                    K.cast(get_valid_m_true_eighth, "uint32"), K.uint32(17)
                                ),
                            ),
                        )
                    else:
                        K.assign(
                            desc_i,
                            K.bitwise_or(
                                K.bitwise_and(desc_i, K.uint32(0xFF81FFFF)),
                                K.shift_left(
                                    K.cast(get_valid_m_true_eighth, "uint32"), K.uint32(17)
                                ),
                            ),
                        )
                    barrier_wait(
                        tmem_empty_barriers.ptr_to([accum_stage_idx]), accum_phase ^ K.int32(1)
                    )
                    K.ptx.tcgen05.fence__after_thread_sync()
                    with K.serial(0, num_k_blocks) as k_block_idx:
                        full_wait_phase = pipeline_state.phase
                        mbarrier_wait_phase(
                            full_barriers.ptr_to([pipeline_state.stage]), full_wait_phase
                        )
                        K.ptx.tcgen05.fence__after_thread_sync()
                        K.assign(
                            a_desc_base_lo,
                            K.tvm_warp_shuffle(
                                K.uint32(0xFFFFFFFF), a_desc_lo, pipeline_state.stage, 32, 32
                            ),
                        )
                        K.assign(
                            b_desc_base_lo,
                            K.tvm_warp_shuffle(
                                K.uint32(0xFFFFFFFF), b_desc_lo, pipeline_state.stage, 32, 32
                            ),
                        )
                        with K.If(K.cuda.elect_sync()), K.Then():
                            with K.unroll(
                                0, kernel_config.block_k // umma_block_k
                            ) as umma_k_block_idx:
                                with K.unroll(0, num_sfa_utccp_chunks) as sfa_chunk_idx:
                                    K.assign(
                                        desc_sf,
                                        replace_smem_desc_addr(
                                            desc_sf,
                                            smem_sfa.ptr_to(
                                                [
                                                    pipeline_state.stage,
                                                    umma_k_block_idx * sf_block_m
                                                    + sfa_chunk_idx * 128,
                                                ]
                                            ),
                                        ),
                                    )
                                    utccp_copy(
                                        sfa_tmem.allocated_addr[0] + sfa_chunk_idx * 4, desc_sf
                                    )
                                with K.unroll(0, num_sfb_utccp_chunks) as sfb_chunk_idx:
                                    K.assign(
                                        desc_sf,
                                        replace_smem_desc_addr(
                                            desc_sf,
                                            smem_sfb.ptr_to(
                                                [
                                                    pipeline_state.stage,
                                                    umma_k_block_idx * kernel_config.block_n
                                                    + sfb_chunk_idx * 128,
                                                ]
                                            ),
                                        ),
                                    )
                                    utccp_copy(
                                        sfb_tmem.allocated_addr[0] + sfb_chunk_idx * 4, desc_sf
                                    )
                                with K.unroll(0, umma_block_k // umma_k) as k_idx:
                                    K.assign(
                                        runtime_desc_i,
                                        (
                                            make_runtime_instr_desc_with_sf_id(
                                                K.if_then_else(has_shared, desc_i_active, desc_i),
                                                k_idx,
                                                k_idx,
                                            )
                                        ),
                                    )
                                    K.assign(
                                        desc_a,
                                        advance_umma_desc_lo(
                                            desc_a,
                                            a_desc_base_lo,
                                            umma_k_block_idx
                                            * umma_block_k
                                            * kernel_config.load_block_m,
                                            k_idx * umma_k,
                                        ),
                                    )
                                    K.assign(
                                        desc_b,
                                        advance_umma_desc_lo(
                                            desc_b,
                                            b_desc_base_lo,
                                            umma_k_block_idx
                                            * umma_block_k
                                            * kernel_config.load_block_n,
                                            k_idx * umma_k,
                                        ),
                                    )
                                    # kind::mxf8f6f4/scale_vec::1X from the
                                    # (f32, e2m1, e4m3, e8m0, e8m0) dtypes; the
                                    # instruction A slot carries desc_b (B^T*A).
                                    K.ptx[
                                        f"tcgen05.mma.cta_group::{kernel_config.num_ctas_per_cluster}.kind::mxf8f6f4.block_scale.scale_vec::1X"
                                    ](
                                        K.cast(accum_stage_idx * umma_n, "uint32"),
                                        desc_b,
                                        desc_a,
                                        runtime_desc_i,
                                        K.cast(sfb_tmem.allocated_addr[0], "uint32"),
                                        K.cast(sfa_tmem.allocated_addr[0], "uint32"),
                                        K.Or(
                                            k_block_idx > K.int32(0),
                                            K.Or(umma_k_block_idx > 0, k_idx > 0),
                                        ),
                                    )
                                # The next UMMA sub-block overwrites these fixed
                                # scale-factor columns. MMA -> CP is not an
                                # implicit TCGEN pipeline, so order that reuse
                                # without draining the outer K-block pipeline.
                                with (
                                    K.If(
                                        umma_k_block_idx + 1
                                        < (kernel_config.block_k // umma_block_k)
                                    ),
                                    K.Then(),
                                ):
                                    K.ptx.tcgen05.fence__before_thread_sync()
                        K.cuda.warp_sync()
                        empty_barrier_arrive_current(k_block_idx == num_k_blocks - K.int32(1))
                        advance_pipeline()
                with K.If(has_accum_task != K.int32(0)), K.Then():
                    barrier_wait(tmem_empty_barriers.ptr_to([accum_stage_idx]), accum_phase)

        def reserved():
            init_gemm_context()
            with K.If(cta_idx_in_cluster == 0), K.Then():
                sched_state.init(0)
                if has_shared:
                    # Shared expert L1 tasks do not depend on dispatch, so they
                    # are enumerated first and fill the EP-dispatch bubble.
                    producer_shared_mainloop(
                        block_phase_shared_l1,
                        workspace_shared_l1_task_count.ptr_to([0]),
                        num_shared_l1_clusters,
                        shared_l1_shape_n,
                        shared_l1_shape_k,
                    )
                # Wait dispatch's results
                scheduler_fetch_expert_recv_count()
                # Generate routed tasks. Keep the original wait -> claim -> publish
                # ordering: `get_next_task()` advances global task counters and must
                # not run before the schedule slot is released by consumers.
                K.assign(sched_task_valid, K.int32(1))
                with K.While(sched_task_valid != K.int32(0)):
                    barrier_wait(
                        task_info_empty_barriers.ptr_to([sched_state.stage]),
                        sched_state.phase ^ K.int32(1),
                    )
                    producer_get_next_task()
                    with K.If(sched_task_valid != K.int32(0)), K.Then():
                        producer_publish_task()
                if has_shared:
                    # Shared expert L2 tasks depend on SharedLinear1 completion.
                    producer_shared_mainloop(
                        block_phase_shared_l2,
                        workspace_shared_l2_task_count.ptr_to([0]),
                        num_shared_l2_clusters,
                        shared_l2_shape_n,
                        shared_l2_shape_k,
                    )
                # Sentinel
                barrier_wait(
                    task_info_empty_barriers.ptr_to([sched_state.stage]),
                    sched_state.phase ^ K.int32(1),
                )
                with K.unroll(0, 8) as task_reg_idx:
                    K.ptx.mov.b32(task_info_regs[task_reg_idx], K.uint32(0))
                producer_publish_task()

        def epilogue(role_warp_idx, role_thread_idx):
            init_gemm_context()
            activation_values = K.alloc_local((num_atoms_per_store, 2, 2), "float32")
            amax_values = K.alloc_local((num_atoms_per_store, 2), "float32")
            thread_local_amax = K.alloc_local((2,), "float32")
            values = K.alloc_local((8,), "uint32")
            epilogue_fp8_packed = K.alloc_local((1,), "uint32")
            weights = K.alloc_local((2,), "float32")
            wp_amax = K.alloc_local((2,), "float32")
            sf = K.alloc_local((2,), "float32")
            sf_inv = K.alloc_local((2,), "float32")
            scaled_pair = K.alloc_local((1,), "uint64")
            scaled_values = K.alloc_local((2,), "float32")
            scaled_bits = K.alloc_local((2,), "uint32")
            scale_exponents = K.alloc_local((2,), "int32")
            scaled_upper = K.alloc_local((1,), "uint64")
            scaled_lower = K.alloc_local((1,), "uint64")
            epilogue_bf16_packed = K.alloc_local((4,), "uint32")
            tmem_addr = K.local_scalar("uint32")
            reduced = K.alloc_local((num_uint4_per_lane * num_elems_per_uint4, 2), "float32")
            load_shared_u32(tmem_allocated, tmem_ptr_in_smem.ptr_to([0]))
            K.cuda.trap_when_assert_failed(tmem_allocated == K.uint32(0))
            epilogue_warp_idx = role_warp_idx
            epilogue_warp_idx_u32 = K.cast(epilogue_warp_idx, "uint32")
            K.assign(epilogue_wg_idx, K.cast(epilogue_warp_idx_u32 // K.uint32(4), "int32"))
            warp_idx_in_wg = K.cast(epilogue_warp_idx_u32 % K.uint32(4), "int32")
            sync_unaligned(
                dispatch_with_epilogue_sync_barrier_idx,
                kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
            )
            sched_state.init(0)
            accum_state.init(0)
            with K.While(True):
                consumer_get_next_task()
                consumer_bind_task_args()
                with K.If(block_phase == K.int32(0)), K.Then():
                    K.Break()
                K.assign(accum_stage_idx, accum_state.stage)
                K.assign(accum_phase, accum_state.phase)
                accum_state.advance()
                barrier_wait(tmem_full_barriers.ptr_to([accum_stage_idx]), accum_phase)
                K.ptx.tcgen05.fence__after_thread_sync()
                # Now we can release the task
                scheduler_release_task_info()
                # Match DeepGEMM's `ptx::exchange(..., 0)`: all lanes have the same
                # scheduler result, but broadcasting it lets the CUDA compiler treat
                # the valid-row early exit as warp-uniform instead of divergent.
                K.assign(
                    valid_m, K.tvm_warp_shuffle(K.uint32(0xFFFFFFFF), valid_m, K.int32(0), 32, 32)
                )
                K.assign(ring_block_idx, pool_block_idx % num_ring_blocks)
                if has_shared:
                    K.assign(
                        block_idx,
                        K.Select(block_phase > K.int32(2), pool_block_idx, ring_block_idx),
                    )
                else:
                    K.assign(block_idx, ring_block_idx)
                # `ring_m_idx` addresses the reusable ring buffers; the L1 store
                # destination row uses `block_idx`, which for shared tasks is the
                # absolute (non-ring) pool block (impls .cuh:981-982).
                K.assign(ring_m_idx, ring_block_idx * kernel_config.block_m)
                K.assign(pool_m_idx, pool_block_idx * kernel_config.block_m)
                K.assign(
                    n_block_idx,
                    n_cluster_idx * 2 + K.Select(cta_idx_in_cluster == 0, K.int32(0), K.int32(1)),
                )
                valid_rows_in_wg = K.max(
                    K.min(valid_m - epilogue_wg_idx * wg_block_m, wg_block_m), K.int32(0)
                )
                with K.If(
                    K.if_then_else(
                        has_shared,
                        K.Or(
                            (block_phase == K.int32(1)),
                            (block_phase == K.int32(block_phase_shared_l1)),
                        ),
                        (block_phase == K.int32(1)),
                    )
                ):
                    with K.Then():
                        # Shared L1 writes a full-size (non-ring) buffer, so it has
                        # no ring handshake to wait on (impls .cuh:988-993).
                        if has_shared:
                            with K.If(block_phase <= K.int32(2)), K.Then():
                                epilogue_wait_l2_empty()
                        else:
                            epilogue_wait_l2_empty()
                        K.assign(n_idx, n_block_idx * l1_out_block_n)
                        # Declared outside the `for s` loop so the per-32-rows weight cache persists
                        # across store iters. When wg_block_m is not a multiple of 32 (e.g., 48 for
                        # block_m=96) the load gate `(j*atom_m) % 32 == 0` fires at a different
                        # rhythm than the s loop, so resetting per-s would leave the cache zero on
                        # iterations between loads. Upstream 559d79f defaults the weight
                        # to 1.0f (weightless shared-expert path multiplies by 1).
                        stored_cached_weight = K.local_scalar("float32")
                        K.assign(stored_cached_weight, K.float32(1.0))
                        with K.serial(
                            0, wg_block_m // kernel_config.store_block_m, unroll=True
                        ) as s:
                            with (
                                K.If(s * kernel_config.store_block_m >= valid_rows_in_wg),
                                K.Then(),
                            ):
                                tmem_empty_barrier_arrive_cta0(
                                    tmem_empty_barriers.ptr_to([accum_stage_idx])
                                )
                                K.Break()
                            with K.unroll(0, num_atoms_per_store) as i:
                                j = s * num_atoms_per_store + i
                                # Shared experts are unweighted: the gather is skipped
                                # and `stored_cached_weight` stays 1.0 (impls .cuh:1017).
                                with (
                                    K.If(
                                        K.And(
                                            (j * atom_m) % 32 == 0,
                                            K.if_then_else(
                                                has_shared, block_phase <= K.int32(2), True
                                            ),
                                        )
                                    ),
                                    K.Then(),
                                ):
                                    # Lanes whose row falls past wg_block_m must skip the load — the
                                    # warp-shuffle source lanes below are always < wg_block_m, so leaving
                                    # OOB lanes' stored_cached_weight stale is fine. (Matches upstream's
                                    # runtime guard for non-32-aligned wg_block_m.)
                                    with K.If(wg_block_m % 32 == 0):
                                        with K.Then():
                                            l1_topk_weight_ptr = l1_topk_weights.ptr_to(
                                                [
                                                    ring_m_idx
                                                    + epilogue_wg_idx * wg_block_m
                                                    + j * atom_m
                                                    + lane_idx
                                                ]
                                            )
                                            load_f32(stored_cached_weight, l1_topk_weight_ptr)
                                        with K.Else():
                                            with (
                                                K.If(
                                                    K.cast(j * atom_m + lane_idx, "uint32")
                                                    < K.uint32(wg_block_m)
                                                ),
                                                K.Then(),
                                            ):
                                                l1_topk_weight_ptr = l1_topk_weights.ptr_to(
                                                    [
                                                        ring_m_idx
                                                        + epilogue_wg_idx * wg_block_m
                                                        + j * atom_m
                                                        + lane_idx
                                                    ]
                                                )
                                                load_f32(stored_cached_weight, l1_topk_weight_ptr)
                                K.ptx.mov.b32(
                                    weights[0],
                                    K.tvm_warp_shuffle(
                                        K.uint32(0xFFFFFFFF),
                                        stored_cached_weight,
                                        (j * atom_m) % 32 + (lane_idx % 4) * 2,
                                        32,
                                        32,
                                    ),
                                )
                                K.ptx.mov.b32(
                                    weights[1],
                                    K.tvm_warp_shuffle(
                                        K.uint32(0xFFFFFFFF),
                                        stored_cached_weight,
                                        (j * atom_m) % 32 + (lane_idx % 4) * 2 + 1,
                                        32,
                                        32,
                                    ),
                                )
                                K.assign(
                                    tmem_addr,
                                    K.cast(
                                        accum_stage_idx * umma_n
                                        + epilogue_wg_idx * wg_block_m
                                        + j * atom_m,
                                        "uint32",
                                    ),
                                )
                                K.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                    values[0], values[1], values[2], values[3], K.uint32(tmem_addr)
                                )
                                K.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                    values[4],
                                    values[5],
                                    values[6],
                                    values[7],
                                    K.uint32(K.bitwise_or(tmem_addr, K.uint32(0x00100000))),
                                )
                                fence_view_async_tmem_load()
                                with K.If(j == wg_block_m // atom_m - 1), K.Then():
                                    tmem_empty_barrier_arrive_cta0(
                                        tmem_empty_barriers.ptr_to([accum_stage_idx])
                                    )
                                with K.unroll(0, 2) as k:
                                    activation_pair_store(
                                        activation_values,
                                        i,
                                        k,
                                        uint32_bits_to_float(values[k * 4]),
                                        uint32_bits_to_float(values[k * 4 + 1]),
                                        uint32_bits_to_float(values[k * 4 + 2]),
                                        uint32_bits_to_float(values[k * 4 + 3]),
                                        weights[0],
                                        weights[1],
                                    )
                                K.ptx.mov.b32(thread_local_amax[0], K.float32(0.0))
                                K.ptx.mov.b32(thread_local_amax[1], K.float32(0.0))
                                with K.unroll(0, 2) as k:
                                    K.ptx.mov.b32(
                                        thread_local_amax[0],
                                        K.max(
                                            thread_local_amax[0], K.fabs(activation_values[i, k, 0])
                                        ),
                                    )
                                    K.ptx.mov.b32(
                                        thread_local_amax[1],
                                        K.max(
                                            thread_local_amax[1], K.fabs(activation_values[i, k, 1])
                                        ),
                                    )
                                K.ptx.mov.b32(amax_values[i, 0], thread_local_amax[0])
                                K.ptx.mov.b32(amax_values[i, 1], thread_local_amax[1])
                                warp_reduce_max_4(amax_values, i, 0)
                                warp_reduce_max_4(amax_values, i, 1)
                                with K.If(lane_idx < 4), K.Then():
                                    amax_reduction_idx = (
                                        epilogue_warp_idx * (kernel_config.store_block_m // 2)
                                        + i * (atom_m // 2)
                                        + lane_idx
                                    ) * 2
                                    K.ptx.st.shared.v2.f32(
                                        smem_amax_reduction.ptr_to([amax_reduction_idx]),
                                        amax_values[i, 0],
                                        amax_values[i, 1],
                                    )
                                K.cuda.warp_sync()
                            tma_stage_idx = s % num_tma_store_stages
                            tma_store_wait(1)
                            K.ptx.bar.sync(
                                K.uint32(epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx), 128
                            )
                            with K.unroll(0, num_atoms_per_store) as i:
                                j = s * num_atoms_per_store + i
                                amax_reduction_idx = (
                                    (epilogue_warp_idx ^ 1) * (kernel_config.store_block_m // 2)
                                    + i * (atom_m // 2)
                                    + (lane_idx % 4)
                                ) * 2
                                K.ptx.ld.shared.v2.f32(
                                    wp_amax[0],
                                    wp_amax[1],
                                    smem_amax_reduction.ptr_to([amax_reduction_idx]),
                                )
                                K.ptx.mov.b32(
                                    amax_values[i, 0], K.max(amax_values[i, 0], wp_amax[0])
                                )
                                K.ptx.mov.b32(
                                    amax_values[i, 1], K.max(amax_values[i, 1], wp_amax[1])
                                )
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
                                scale_pack_fp8x4_e4m3(
                                    epilogue_fp8_packed,
                                    scaled_upper,
                                    scaled_lower,
                                    activation_values[i, 0, 0],
                                    activation_values[i, 0, 1],
                                    activation_values[i, 1, 0],
                                    activation_values[i, 1, 1],
                                    sf_inv[0],
                                    sf_inv[1],
                                )
                                row = lane_idx
                                col = warp_idx_in_wg
                                smem_ptr = smem_cd_l1.ptr_to(
                                    [
                                        tma_stage_idx,
                                        epilogue_wg_idx,
                                        i * atom_m + row,
                                        (col ^ (row // 2)) * num_bank_group_bytes,
                                    ]
                                )
                                sm100_u8x4_stsm_t_copy(epilogue_fp8_packed[0], smem_ptr)
                                with K.If(K.And(warp_idx_in_wg % 2 == 0, lane_idx < 4)), K.Then():
                                    # Factored form of upstream 891d57b: token_base_idx is < BLOCK_M so
                                    # `m_block_idx * BLOCK_M` factors out as `m_block_idx * SF_BLOCK_M`
                                    # past `transform_sf_token_idx` (which is bitwise-independent in
                                    # that range). `lane_idx * 2` only touches bits 0..2 of the input
                                    # (token_base_idx is a multiple of atom_m=8), so its contribution
                                    # collapses to a constant `lane_idx * 8` (= `(lane_idx*2) << 2`).
                                    # Eliminates one mul + the residual modulo work in the original
                                    # composed form.
                                    token_base_idx = (
                                        epilogue_wg_idx * wg_block_m
                                        + s * kernel_config.store_block_m
                                        + i * atom_m
                                    )
                                    sf_pool_token_idx = (
                                        K.cast(block_idx, "uint64") * K.uint64(sf_block_m)
                                        + K.cast(transform_sf_token_idx(token_base_idx), "uint64")
                                        + K.cast(lane_idx, "uint64") * K.uint64(8)
                                    )
                                    # Shared tasks write the full-size shared L2 SF
                                    # plane, which has its own MN stride and base
                                    # (impls .cuh:1134-1136). Selecting the base as a
                                    # byte delta keeps this a single address select
                                    # feeding one pair of stores, as in the source.
                                    if has_shared:
                                        mn_stride = K.Select(
                                            block_phase > K.int32(2),
                                            K.uint64(num_max_shared_sf_tokens * 4),
                                            K.uint64(workspace_layout.num_sf_ring_tokens * 4),
                                        )
                                    else:
                                        mn_stride = K.uint64(
                                            workspace_layout.num_sf_ring_tokens * 4
                                        )
                                    k_idx = n_block_idx * 2 + warp_idx_in_wg // 2
                                    k_uint_idx = k_idx // 4
                                    byte_idx = k_idx % 4
                                    sf_addr = (
                                        K.cast(k_uint_idx, "uint64") * mn_stride
                                        + sf_pool_token_idx * K.uint64(4)
                                        + K.cast(byte_idx, "uint64")
                                    )
                                    if has_shared:
                                        # Signed byte delta to the shared L2 SF plane,
                                        # which sits *before* the routed ring in the
                                        # symm buffer. Keeps the destination a single
                                        # address select feeding one pair of stores.
                                        sf_addr = K.cast(sf_addr, "int64") + K.Select(
                                            block_phase > K.int32(2),
                                            K.int64(
                                                symm_buffer_layout.shared_l2_sf_offset
                                                - symm_buffer_layout.l2_sf_offset
                                            ),
                                            K.int64(0),
                                        )
                                    sf_bits = float_bits(sf[0])
                                    sf_bits_hi = float_bits(sf[1])
                                    store_global_u8(
                                        l2_sf_buffer.ptr_to([sf_addr]),
                                        K.cast(K.shift_right(sf_bits, K.uint32(23)), "uint8"),
                                    )
                                    store_global_u8(
                                        l2_sf_buffer.ptr_to(
                                            [
                                                sf_addr
                                                + K.if_then_else(
                                                    has_shared, K.int64(16), K.uint64(16)
                                                )
                                            ]
                                        ),
                                        K.cast(K.shift_right(sf_bits_hi, K.uint32(23)), "uint8"),
                                    )
                            K.cuda.warp_sync()
                            K.ptx.bar.sync(
                                K.uint32(epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx), 128
                            )
                            with K.If((warp_idx_in_wg == 0) & K.cuda.elect_sync() != 0), K.Then():
                                tma_store_fence()
                                sm90_tma_store_2d_copy_select(
                                    smem_cd_l1.ptr_to([tma_stage_idx, epilogue_wg_idx, 0, 0]),
                                    tensor_map_l1_output,
                                    tensor_map_shared_l1_output,
                                    block_phase,
                                    n_idx,
                                    K.if_then_else(
                                        has_shared, block_idx * kernel_config.block_m, ring_m_idx
                                    )
                                    + epilogue_wg_idx * wg_block_m
                                    + s * kernel_config.store_block_m,
                                )
                                tma_store_arrive()
                            K.cuda.warp_sync()
                        tma_store_wait(0)
                        K.ptx.bar.sync(
                            K.uint32(epilogue_full_sync_barrier_idx),
                            K.uint32(kernel_config.num_epilogue_threads),
                        )
                        with K.If((epilogue_warp_idx == 0) & K.cuda.elect_sync() != 0), K.Then():
                            # Shared L1 only signals its own full-size consumer; it has
                            # no L1 ring slot to release (impls .cuh:1179-1189).
                            if has_shared:
                                with K.If(block_phase > K.int32(2)):
                                    with K.Then():
                                        atomic_add_rel_u32(
                                            atom_prev_unused,
                                            workspace_shared_l2_full_count.ptr_to([pool_block_idx]),
                                            K.uint32(1),
                                        )
                                    with K.Else():
                                        epilogue_signal_routed_l1_done()
                            else:
                                epilogue_signal_routed_l1_done()
                        K.cuda.warp_sync()
                    with K.Else():
                        # Shared L2 has no ring slot to release (impls .cuh:1194).
                        if has_shared:
                            with K.If(block_phase <= K.int32(2)), K.Then():
                                with (
                                    K.If((epilogue_warp_idx == 0) & K.cuda.elect_sync() != 0),
                                    K.Then(),
                                ):
                                    red_add_gpu_u32(
                                        workspace_l2_empty_count.ptr_to([ring_block_idx]),
                                        K.uint32(1),
                                    )
                                K.cuda.warp_sync()
                        else:
                            with (
                                K.If((epilogue_warp_idx == 0) & K.cuda.elect_sync() != 0),
                                K.Then(),
                            ):
                                red_add_gpu_u32(
                                    workspace_l2_empty_count.ptr_to([ring_block_idx]), K.uint32(1)
                                )
                            K.cuda.warp_sync()
                        K.assign(n_idx, n_block_idx * kernel_config.block_n)
                        with K.serial(
                            0, wg_block_m // kernel_config.store_block_m, unroll=True
                        ) as s:
                            with (
                                K.If(s * kernel_config.store_block_m >= valid_rows_in_wg),
                                K.Then(),
                            ):
                                tmem_empty_barrier_arrive_cta0(
                                    tmem_empty_barriers.ptr_to([accum_stage_idx])
                                )
                                K.Break()
                            with K.unroll(0, num_atoms_per_store) as i:
                                j = s * num_atoms_per_store + i
                                K.assign(
                                    tmem_addr,
                                    K.cast(
                                        accum_stage_idx * umma_n
                                        + epilogue_wg_idx * wg_block_m
                                        + j * atom_m,
                                        "uint32",
                                    ),
                                )
                                K.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                    values[0], values[1], values[2], values[3], K.uint32(tmem_addr)
                                )
                                K.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
                                    values[4],
                                    values[5],
                                    values[6],
                                    values[7],
                                    K.uint32(K.bitwise_or(tmem_addr, K.uint32(0x00100000))),
                                )
                                fence_view_async_tmem_load()
                                with K.If(K.And(i == 0, s > 0)), K.Then():
                                    K.ptx.bar.sync(
                                        K.uint32(
                                            epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx
                                        ),
                                        128,
                                    )
                                with (
                                    K.If(
                                        K.And(
                                            s == wg_block_m // kernel_config.store_block_m - 1,
                                            i == kernel_config.store_block_m // atom_m - 1,
                                        )
                                    ),
                                    K.Then(),
                                ):
                                    tmem_empty_barrier_arrive_cta0(
                                        tmem_empty_barriers.ptr_to([accum_stage_idx])
                                    )
                                K.ptx.mov.b32(
                                    epilogue_bf16_packed[0],
                                    cast_into_bf16_and_pack(
                                        uint32_bits_to_float(values[0]),
                                        uint32_bits_to_float(values[1]),
                                    ),
                                )
                                K.ptx.mov.b32(
                                    epilogue_bf16_packed[1],
                                    cast_into_bf16_and_pack(
                                        uint32_bits_to_float(values[2]),
                                        uint32_bits_to_float(values[3]),
                                    ),
                                )
                                K.ptx.mov.b32(
                                    epilogue_bf16_packed[2],
                                    cast_into_bf16_and_pack(
                                        uint32_bits_to_float(values[4]),
                                        uint32_bits_to_float(values[5]),
                                    ),
                                )
                                K.ptx.mov.b32(
                                    epilogue_bf16_packed[3],
                                    cast_into_bf16_and_pack(
                                        uint32_bits_to_float(values[6]),
                                        uint32_bits_to_float(values[7]),
                                    ),
                                )
                                row = lane_idx % 8
                                col = (epilogue_warp_idx % 2) * 4 + lane_idx // 8
                                l2_cd_elem_offset = l2_cd_swizzled_elem_offset(
                                    warp_idx_in_wg // 2, i * atom_m + row, col, row
                                )
                                smem_ptr = smem_cd_l2.ptr_to([epilogue_wg_idx, l2_cd_elem_offset])
                                sm90_u32x4_stsm_t_copy(epilogue_bf16_packed, smem_ptr)
                            K.ptx.bar.sync(
                                K.uint32(epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx), 128
                            )
                            row_in_atom = (warp_idx_in_wg * 2 + lane_idx // 16) % atom_m
                            bank_group_idx = lane_idx % 8
                            with K.unroll(0, num_rows_per_warp) as j:
                                row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx // 16
                                m_idx_in_block = (
                                    epilogue_wg_idx * wg_block_m
                                    + s * kernel_config.store_block_m
                                    + row_in_store
                                )
                                with (
                                    K.If(
                                        K.cast(m_idx_in_block, "uint32") < K.cast(valid_m, "uint32")
                                    ),
                                    K.Then(),
                                ):
                                    src_metadata_idx = pool_m_idx + m_idx_in_block
                                    # Shared output is rank-local: it goes straight to
                                    # this rank's slot `kNumTopk` at the local token
                                    # index, with no dispatch metadata lookup and no
                                    # NVLink hop (impls .cuh:1275-1284).
                                    with K.If(K.And(has_shared, block_phase > K.int32(2))):
                                        with K.Then():
                                            K.assign(dst_rank_idx_u32, K.cast(rank_idx, "uint32"))
                                            K.assign(
                                                dst_token_idx_u32,
                                                K.cast(src_metadata_idx, "uint32"),
                                            )
                                            K.assign(dst_topk_idx_u32, K.uint32(num_topk))
                                        with K.Else():
                                            load_global_u32(
                                                dst_rank_idx_u32,
                                                workspace_token_src_metadata.ptr_to(
                                                    [src_metadata_idx, 0]
                                                ),
                                            )
                                            load_global_u32(
                                                dst_token_idx_u32,
                                                workspace_token_src_metadata.ptr_to(
                                                    [src_metadata_idx, 1]
                                                ),
                                            )
                                            load_global_u32(
                                                dst_topk_idx_u32,
                                                workspace_token_src_metadata.ptr_to(
                                                    [src_metadata_idx, 2]
                                                ),
                                            )
                                    dst_rank_idx = K.cast(dst_rank_idx_u32, "int32")
                                    dst_token_base_offset = K.cast(
                                        dst_topk_idx_u32, "uint64"
                                    ) * K.uint64(
                                        workspace_layout.num_max_tokens_per_rank * hidden * 2
                                    ) + K.cast(dst_token_idx_u32, "uint64") * K.uint64(hidden * 2)
                                    dst_col_byte_offset = K.cast(n_idx * 2, "uint64")
                                    lane_byte_offset = K.cast((lane_idx % 16) * 16, "uint64")
                                    dst_ptr = (
                                        dst_token_base_offset
                                        + dst_col_byte_offset
                                        + lane_byte_offset
                                    )
                                    l2_cd_elem_offset = l2_cd_swizzled_elem_offset(
                                        (lane_idx % 16) // 8,
                                        row_in_store,
                                        bank_group_idx,
                                        row_in_atom,
                                    )
                                    smem_ptr = smem_cd_l2.ptr_to(
                                        [epilogue_wg_idx, l2_cd_elem_offset]
                                    )
                                    lds128(smem_ptr, epilogue_bf16_packed)
                                    K.assign(
                                        symm_rank_base,
                                        sym_buffer_base + K.cast(symm_rank_offsets[0], "uint64"),
                                    )
                                    load_symm_rank_base(
                                        symm_rank_base, smem_symm_rank_bases, dst_rank_idx
                                    )
                                    dst_ptr = (
                                        K.cast(symm_buffer_layout.combine_token_offset, "uint64")
                                        + dst_ptr
                                    )
                                    stg128_symm(
                                        symm_rank_base,
                                        dst_ptr,
                                        epilogue_bf16_packed[0],
                                        epilogue_bf16_packed[1],
                                        epilogue_bf16_packed[2],
                                        epilogue_bf16_packed[3],
                                    )
                        K.ptx.bar.sync(
                            K.uint32(epilogue_full_sync_barrier_idx),
                            K.uint32(kernel_config.num_epilogue_threads),
                        )
            K.assign(epilogue_thread_idx, role_thread_idx)
            epilogue_nvlink_barrier_before_combine_reduce(epilogue_thread_idx)
            # The grid barrier above includes every epilogue thread in both CTAs,
            # so no peer can still be accessing TMEM when warp 0 deallocates it.
            with K.If(epilogue_warp_idx == 0), K.Then():
                K.ptx[
                    f"tcgen05.dealloc.cta_group::{kernel_config.num_ctas_per_cluster}.sync.aligned.b32"
                ](K.uint32(0), K.uint32(num_tmem_cols))
            sync_unaligned(
                dispatch_with_epilogue_sync_barrier_idx,
                kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
            )
            # The preceding grid barrier publishes every epilogue's generic
            # stores into the symmetric combine-token buffer.  Combine reads
            # those stores through TMA's async global proxy.
            K.ptx.fence.proxy.async_.global_()
            K.assign(
                combine_token_idx, sm_idx * kernel_config.num_epilogue_warps + epilogue_warp_idx
            )
            combine_state.init(0)
            with K.While(K.cast(combine_token_idx, "uint32") < K.cast(num_tokens, "uint32")):
                K.assign(combine_stored_topk_slot_idx, K.int32(-1))
                with K.If(lane_idx < num_topk), K.Then():
                    load_global_s64(
                        ordinary_global_s64, input_topk_idx.ptr_to([combine_token_idx, lane_idx])
                    )
                    K.assign(combine_stored_topk_slot_idx, K.cast(ordinary_global_s64, "int32"))
                if has_shared:
                    # Lane `kNumTopk` synthesizes the shared expert's slot, so the
                    # shared output is folded into every token's sum without any
                    # routing metadata (impls .cuh:1370).
                    with K.If(lane_idx == num_topk), K.Then():
                        K.assign(combine_stored_topk_slot_idx, K.int32(num_topk))
                K.assign(
                    combine_total_mask,
                    ballot_sync(K.uint32(0xFFFFFFFF), combine_stored_topk_slot_idx >= K.int32(0)),
                )
                with K.unroll(0, num_chunks) as chunk:
                    K.assign(combine_slot_mask, combine_total_mask)
                    chunk_byte_offset = chunk * num_chunk_bytes
                    chunk_offset_elems = chunk_byte_offset // 2
                    with K.unroll(0, num_uint4_per_lane * num_elems_per_uint4) as reduced_idx:
                        K.ptx.mov.b32(reduced[reduced_idx, 0], K.float32(0.0))
                        K.ptx.mov.b32(reduced[reduced_idx, 1], K.float32(0.0))
                    K.assign(combine_do_reduce, K.int32(0))
                    with K.If(combine_slot_mask != K.uint32(0)), K.Then():
                        K.assign(combine_slot_idx, ffs_u32(combine_slot_mask) - K.int32(1))
                        K.assign(
                            combine_slot_mask,
                            K.bitwise_xor(
                                combine_slot_mask,
                                K.shift_left(K.uint32(1), K.cast(combine_slot_idx, "uint32")),
                            ),
                        )
                        with K.If(K.cuda.elect_sync()), K.Then():
                            src_ptr = combine_tokens.ptr_to(
                                [combine_slot_idx, combine_token_idx, chunk_offset_elems]
                            )
                            load_barrier_ptr = combine_barriers.ptr_to(
                                [epilogue_warp_idx * 2 + combine_state.stage]
                            )
                            load_buffer_ptr = combine_chunks.ptr_to(
                                [combine_state.stage, epilogue_warp_idx, 0, 0]
                            )
                            tma_load_1d(load_buffer_ptr, src_ptr, load_barrier_ptr, num_chunk_bytes)
                            mbarrier_arrive_and_set_tx(load_barrier_ptr, num_chunk_bytes)
                        K.assign(combine_do_reduce, K.int32(1))
                    K.cuda.warp_sync()
                    with K.While(combine_do_reduce != K.int32(0)):
                        K.assign(combine_next_do_reduce, K.int32(0))
                        with K.If(combine_slot_mask != K.uint32(0)), K.Then():
                            K.assign(combine_slot_idx, ffs_u32(combine_slot_mask) - K.int32(1))
                            K.assign(
                                combine_slot_mask,
                                K.bitwise_xor(
                                    combine_slot_mask,
                                    K.shift_left(K.uint32(1), K.cast(combine_slot_idx, "uint32")),
                                ),
                            )
                            with K.If(K.cuda.elect_sync()), K.Then():
                                src_ptr = combine_tokens.ptr_to(
                                    [combine_slot_idx, combine_token_idx, chunk_offset_elems]
                                )
                                load_barrier_ptr = combine_barriers.ptr_to(
                                    [epilogue_warp_idx * 2 + (combine_state.stage ^ K.int32(1))]
                                )
                                prefetch_buffer_ptr = combine_chunks.ptr_to(
                                    [combine_state.stage ^ K.int32(1), epilogue_warp_idx, 0, 0]
                                )
                                tma_load_1d(
                                    prefetch_buffer_ptr, src_ptr, load_barrier_ptr, num_chunk_bytes
                                )
                                mbarrier_arrive_and_set_tx(load_barrier_ptr, num_chunk_bytes)
                            K.assign(combine_next_do_reduce, K.int32(1))
                        K.cuda.warp_sync()
                        mbarrier_wait_phase(
                            combine_barriers.ptr_to([epilogue_warp_idx * 2 + combine_state.stage]),
                            combine_state.phase,
                        )
                        with K.unroll(0, num_uint4_per_lane) as j:
                            load_ptr = combine_chunks.ptr_to(
                                [combine_state.stage, epilogue_warp_idx, j * 32 + lane_idx, 0]
                            )
                            lds128(load_ptr, epilogue_bf16_packed)
                            with K.unroll(0, num_elems_per_uint4) as elem_idx:
                                # d = convert(a) + c, accumulating in place.
                                K.ptx.add.rn.f32.bf16(
                                    reduced[j * num_elems_per_uint4 + elem_idx, 0],
                                    bf16x2_lo(epilogue_bf16_packed[elem_idx]),
                                    reduced[j * num_elems_per_uint4 + elem_idx, 0],
                                )
                                K.ptx.add.rn.f32.bf16(
                                    reduced[j * num_elems_per_uint4 + elem_idx, 1],
                                    bf16x2_hi(epilogue_bf16_packed[elem_idx]),
                                    reduced[j * num_elems_per_uint4 + elem_idx, 1],
                                )
                        # A later selection reuses this load stage through TMA's
                        # async proxy. Publish every lane's completed generic
                        # loads before the elected lane can issue that overwrite.
                        with K.If(combine_slot_mask != K.uint32(0)), K.Then():
                            tma_store_fence()
                            K.cuda.warp_sync()
                        combine_state.advance()
                        K.assign(combine_do_reduce, combine_next_do_reduce)
                    with K.unroll(0, num_uint4_per_lane) as j:
                        with K.unroll(0, num_elems_per_uint4) as elem_idx:
                            K.ptx.mov.b32(
                                epilogue_bf16_packed[elem_idx],
                                cast_into_bf16_and_pack(
                                    reduced[j * num_elems_per_uint4 + elem_idx, 0],
                                    reduced[j * num_elems_per_uint4 + elem_idx, 1],
                                ),
                            )
                        with K.If(j == 0), K.Then():
                            tma_store_wait(0)
                            K.cuda.warp_sync()
                        combine_store_ptr = combine_chunks.ptr_to(
                            [2, epilogue_warp_idx, j * 32 + lane_idx, 0]
                        )
                        K.ptx.st.shared.v4.b32(
                            combine_store_ptr,
                            epilogue_bf16_packed[0],
                            epilogue_bf16_packed[1],
                            epilogue_bf16_packed[2],
                            epilogue_bf16_packed[3],
                        )
                    K.cuda.warp_sync()
                    with K.If(K.cuda.elect_sync()), K.Then():
                        tma_store_fence()
                        dst_ptr = y.ptr_to([combine_token_idx * hidden + chunk_offset_elems])
                        combine_store_ptr = combine_chunks.ptr_to([2, epilogue_warp_idx, 0, 0])
                        tma_store_1d(dst_ptr, combine_store_ptr, num_chunk_bytes)
                        tma_store_arrive()
                K.cuda.warp_sync()
                K.assign(
                    combine_token_idx,
                    combine_token_idx + kernel_config.num_sms * kernel_config.num_epilogue_warps,
                )

        prologue()
        with dispatch_role:
            dispatch(K.warp_id_in_role(), K.tid_in_role())
        with load_a_role:
            load_a()
        with load_b_role:
            load_b()
        with mma_role:
            mma()
        with reserved_role:
            reserved()
        with epilogue_role:
            epilogue(K.warp_id_in_role(), K.tid_in_role())

    return mega_moe.func.with_attr("tirx.kernel_launch_params", get_tirx_launch_param_tags())
