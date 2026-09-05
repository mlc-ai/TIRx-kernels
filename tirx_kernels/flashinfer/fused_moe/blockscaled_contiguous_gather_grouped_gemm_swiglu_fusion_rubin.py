# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE.
#
# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ 012cfdb97f217e0d48bc9352c17a74068c9e495b)
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM107 gathered grouped block-scaled GEMM with fused SwiGLU and FP4 output.

Upstream source:
- flashinfer/fused_moe/cute_dsl/rubin/
  blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py
"""

import hashlib
import importlib
import sys
from functools import cache
from pathlib import Path

import tirx_kernels.kern as K

SOURCE_SHA256 = "6393684e152bdf6d5cd3666666054641eb0b952737f0db2f20018260af1ac97c"
SOURCE_DEPENDENCY_SHA256 = {
    "custom_pipeline.py": "97210fb00b05db803cbad6418e2e9d1b7059b13d5cb1473f12f72a696f04083b",
    "inline_ptx.py": "b35fca3bf8173fbbe71472e3626293be2738c9669d07d90e5827e58a4628c944",
    "utils.py": "87ef7c7199abb652a6b7660e047ac9876c6941c8574f90a8c4fd912ac4832e98",
    "blockscaled_contiguous_gather_grouped_gemm_act_fusion.py": (
        "eeaadde0c52e636159e5b34a20c37cfa7585a2d90d9178252c8232e5322cae0a"
    ),
    "tuner.py": "e91e57be076514e9fe24ef5d5c87746487dcc296f5b01398821beb80557397cf",
}

KERNEL_META = {
    "name": "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "012cfdb97f217e0d48bc9352c17a74068c9e495b",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

_SOURCE_ROOT = Path("/root-vol/aarch64-ws/kernel-libs/vr200/flashinfer")
_TRY_WAIT_TICKS = 10_000_000
_TILE_M = 128
_TILE_N = 128
_TILE_K = 256
_EPI_N = 64
_AB_STAGES = 8
_C_STAGES = 9
_ACC_STAGES = 2
_TILE_STAGES = 2
_SFA_TMEM_STAGES = 4
_KERNEL_THREADS = 640
_SMEM_BYTES = 332_800
_TMEM_COLUMNS = 576

_INFO_OFFSET = 0
_A_FULL_OFFSET = 40
_B_FULL_OFFSET = 168
_SFA_FULL_OFFSET = 296
_SFA_T_FULL_OFFSET = 424
_ACC_FULL_OFFSET = 488
_TILE_FULL_OFFSET = 520
_TMEM_DEALLOC_OFFSET = 552
_TMEM_PTR_OFFSET = 560
_C_OFFSET = 1024
_SFA_OFFSET = 37_888
_A_OFFSET = 54_272
_B_OFFSET = 185_344
_SFB_OFFSET = 316_416


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _routing_counts(num_experts, seq_len, routing="balanced"):
    total = seq_len * 8
    if routing == "empty_edges" and num_experts > 2:
        live = num_experts - 2
        base, extra = divmod(total, live)
        return (0, *(base + (index < extra) for index in range(live)), 0)
    base, extra = divmod(total, num_experts)
    return tuple(base + (index < extra) for index in range(num_experts))


def _problem(num_experts, seq_len, N, K_dim, routing="balanced"):
    if min(num_experts, seq_len, N, K_dim) <= 0:
        raise ValueError("num_experts, seq_len, N, and K must be positive")
    if N % 128 or K_dim % 256:
        raise ValueError("the Rubin production tactic requires N%128 == 0 and K%256 == 0")
    counts = _routing_counts(num_experts, seq_len, routing)
    aligned = tuple(_ceil_div(count, 128) * 128 for count in counts)
    return counts, sum(aligned), sum(value // 128 for value in aligned)


def _case(label, num_experts, seq_len, N, K_dim, routing="balanced"):
    counts = _routing_counts(num_experts, seq_len, routing)
    return {
        "label": label,
        "num_experts": num_experts,
        "seq_len": seq_len,
        "expected_m_per_expert": min(counts) if counts else 0,
        "N": N,
        "K": K_dim,
        "routing": routing,
        "a_dtype": "float4_e2m1fn",
        "b_dtype": "float4_e2m1fn",
        "sf_dtype": "float8_e4m3fn",
        "sf_vec_size": 16,
        "c_dtype": "float4_e2m1fn",
        "topk": 8,
        "tactic": 0,
        "a_path": "cpasync",
        "vectorized_f32": True,
        "raster_along_m": False,
        "use_pdl": True,
        "alpha": 1.0,
        "ugpu_half_gemm": False,
    }


CONFIGS = [
    _case("tiny_single", 1, 16, 128, 256),
    _case("k_multistage", 3, 33, 256, 512),
    _case("n_multitile", 6, 64, 384, 768),
    _case("routing_tail", 8, 129, 512, 1024),
    _case("empty_edge_experts", 6, 17, 256, 512, "empty_edges"),
    _case("many_experts", 72, 64, 128, 256),
    _case("production_shape_smoke", 1, 128, 4096, 7168),
]


def _production_profiles():
    profiles = [(6, 768, 1024), (6, 384, 512), (1, 128, 1024), (2, 128, 512), (4, 128, 256)]
    for ranks in (4, 8, 16, 32, 36, 48, 72):
        for tokens in (64, 128, 256, 384, 512, 768, 1024):
            groups = 288 // ranks
            profiles.append((groups, tokens, tokens * 8 // groups))
    answer = []
    for groups, tokens, expected in profiles:
        if not any(g == groups and e == expected for g, _, e in answer):
            answer.append((groups, tokens, expected))
    return tuple(answer)


BENCH_CONFIGS = [
    {
        **_case(f"bench_g{groups}_m{expected}_n4096_k7168", groups, tokens, 4096, 7168),
        "expected_m_per_expert": expected,
    }
    for groups, tokens, expected in _production_profiles()
]
assert len(BENCH_CONFIGS) == 51


def _descriptor_base(ldo, sdo, swizzle):
    arrangement = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement & 7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _descriptor_with_address(base, shared_address):
    field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x7FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), field)


def _instruction_descriptor():
    value = (1 << 3) | (1 << 7) | (1 << 10)
    value |= ((128 >> 3) & 0x3F) << 17
    value |= (0 & 3) << 23
    value |= ((128 >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF


def _advance(state):
    state.advance()


def _elected():
    lane = K.local_scalar("uint32")
    pred = K.local_scalar("uint32")
    K.ptx.elect_sync(lane, pred, K.uint32(0xFFFFFFFF))
    return pred == K.uint32(1)


def _try_wait(dst, barrier, phase):
    K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, K.cast(phase, "uint32")
    )


def _wait(barrier, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _validate_config(config):
    required = {
        "a_dtype": "float4_e2m1fn",
        "b_dtype": "float4_e2m1fn",
        "sf_dtype": "float8_e4m3fn",
        "sf_vec_size": 16,
        "c_dtype": "float4_e2m1fn",
        "topk": 8,
        "tactic": 0,
        "a_path": "cpasync",
        "vectorized_f32": True,
        "raster_along_m": False,
        "ugpu_half_gemm": False,
    }
    for key, expected in required.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"production Rubin specialization requires {key}={expected!r}")


@cache
def _make_kernel(num_experts, seq_len, N, K_dim, routing, num_sms, use_pdl):
    counts, permuted_m, permuted_tiles = _problem(num_experts, seq_len, N, K_dim, routing)
    del counts
    n_tiles = N // _TILE_N
    k_tiles = K_dim // _TILE_K
    total_work = permuted_tiles * n_tiles
    num_clusters = min(total_work, num_sms)
    if num_clusters <= 0:
        raise ValueError("routing must contain at least one active tile")
    balance_persistent_grid = routing == "balanced" and (
        (num_experts == 9 and permuted_tiles in (36, 72))
        or (num_experts == 18 and permuted_tiles == 72)
    )
    materialize_gather_a = routing == "balanced" and num_experts == 9 and permuted_tiles == 72
    if balance_persistent_grid:
        persistent_rounds = _ceil_div(total_work, num_clusters)
        num_clusters = _ceil_div(total_work, persistent_rounds)
    a_desc_base = _descriptor_base(1, 64, 3)
    b_desc_base = _descriptor_base(1, 64, 3)
    sf_desc_base = _descriptor_base(1, 8, 0)
    instr_desc = _instruction_descriptor()

    def host_prelude(params):
        b = params["b"]
        sfb = params["sfb"]
        c = params["c"]
        b_map = K.stack_alloca("tensormap", 1)
        sfb_map = K.stack_alloca("tensormap", 1)
        c_map = K.stack_alloca("tensormap", 1)

        def encode(desc, dtype, rank, data, *fields):
            K.call_packed("runtime.cuTensorMapEncodeTiled", desc, dtype, rank, data, *fields)

        encode(
            b_map,
            "float4_e2m1fn",
            3,
            b.data,
            K_dim,
            N,
            num_experts,
            K_dim // 2,
            N * K_dim // 2,
            256,
            128,
            1,
            1,
            1,
            1,
            0,
            3,
            2,
            0,
            13,
        )
        sf_k_groups = K_dim // 64
        sf_n_groups = N // 128
        encode(
            sfb_map,
            "uint16",
            4,
            sfb.data,
            256,
            sf_k_groups,
            sf_n_groups,
            num_experts,
            512,
            sf_k_groups * 512,
            sf_n_groups * sf_k_groups * 512,
            256,
            4,
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
        encode(
            c_map,
            "float4_e2m1fn",
            3,
            c.data,
            N // 2,
            permuted_m,
            1,
            N // 4,
            permuted_m * N // 4,
            64,
            128,
            1,
            1,
            1,
            1,
            0,
            1,
            2,
            0,
            13,
        )
        return b_map, sfb_map, c_map

    def kernel(
        a,
        b,
        sfa,
        sfb,
        c,
        sfc,
        alpha,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        global_scale,
        *,
        host,
    ):
        del b, sfb, c
        required_block_size = K.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        b_map, sfb_map, c_map = host
        _bx, _by, work_id = K.cta_id()
        cluster_x, cluster_y = K.cta_id_in_cluster([1, 1], preferred=[1, 1])
        del _bx, _by, cluster_x, cluster_y
        warp = K.warp_id()
        lane = K.lane_id()

        roles = K.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3], regs=168)
        gather_a_role = roles.role("gather_a", warps=[4, 5, 6, 7], regs=80)
        # These four warps retain the entry allocation.  The source calls
        # setmaxnreg only for the epilogue, A/SFA gather, and SFA transform
        # warps; assigning an explicit target here would both invent an
        # instruction and over-subscribe the rounded 20-warp CTA pool.
        mma_role = roles.role("mma", warps=[8])
        tma_role = roles.role("tma", warps=[9])
        scheduler_role = roles.role("scheduler", warps=[10])
        idle_role = roles.role("idle", warps=[11])
        gather_sfa_role = roles.role("gather_sfa", warps=[12, 13, 14, 15], regs=80)
        transform_role = roles.role("transform_sfa", warps=[16, 17, 18, 19], regs=48)

        smem = K.alloc_buffer((_SMEM_BYTES,), K.u8, scope="shared.dyn", align=1024)
        pool = K.smem_pool(base=smem)
        info = pool.alloc((10,), K.i32, align=4)
        a_pipe = K.Pipeline(pool, 8, full="mbar", empty="tcgen05", leader=K.bool(False))
        b_pipe = K.Pipeline(pool, 8, full="tma", empty="tcgen05", leader=K.bool(False))
        sfa_pipe = K.Pipeline(pool, 8, full="mbar", empty="mbar", leader=K.bool(False))
        sfa_t_pipe = K.Pipeline(pool, 4, full="mbar", empty="tcgen05", leader=K.bool(False))
        acc_pipe = K.Pipeline(pool, 2, full="tcgen05", empty="mbar", leader=K.bool(False))
        tile_pipe = K.Pipeline(pool, 2, full="mbar", empty="mbar", leader=K.bool(False))
        if pool.bytes != _TMEM_DEALLOC_OFFSET:
            raise AssertionError(f"protocol layout changed: {pool.bytes}")
        _tmem_dealloc = pool.alloc((1,), K.u64, align=8)
        tmem_slot = pool.alloc((1,), K.u32, align=4)
        del _tmem_dealloc
        if pool.bytes != _TMEM_PTR_OFFSET + 4:
            raise AssertionError(f"protocol header changed: {pool.bytes}")

        with tma_role:
            K.ptx.prefetch.tensormap(K.address_of(b_map))
            K.ptx.prefetch.tensormap(K.address_of(sfb_map))
            K.ptx.prefetch.tensormap(K.address_of(c_map))

        def init_pipe(pipe, stages, full_arrivals, empty_arrivals):
            with K.If(_elected()):
                with K.Then():
                    with K.unroll(0, stages) as stage:
                        K.ptx.mbarrier.init.shared.b64(
                            pipe.full.ptr_to([stage]), K.uint32(full_arrivals)
                        )
            with K.If(_elected()):
                with K.Then():
                    with K.unroll(0, stages) as stage:
                        K.ptx.mbarrier.init.shared.b64(
                            pipe.empty.ptr_to([stage]), K.uint32(empty_arrivals)
                        )

        with K.If(warp == 0):
            with K.Then():
                init_pipe(sfa_pipe, 8, 128, 128)
                init_pipe(sfa_t_pipe, 4, 128, 1)
                init_pipe(a_pipe, 8, 128, 1)
                init_pipe(b_pipe, 8, 1, 1)
                init_pipe(acc_pipe, 2, 1, 4)
                init_pipe(tile_pipe, 2, 32, 576)
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.sync(K.uint32(0), K.uint32(_KERNEL_THREADS))
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.sync(K.uint32(0), K.uint32(_KERNEL_THREADS))
        if use_pdl:
            K.ptx.griddepcontrol.wait()

        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + _A_OFFSET)
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + _B_OFFSET)
        )

        def wait_and_read_info(state, tile_m, tile_n, expert, valid, mn_limit=None):
            _wait(tile_pipe.full.ptr_to([state.stage]), state.phase)
            base = state.stage * 5
            K.ptx.ld.shared.b32(tile_m, info.ptr_to([base]))
            K.ptx.ld.shared.b32(tile_n, info.ptr_to([base + 1]))
            K.ptx.ld.shared.b32(expert, info.ptr_to([base + 2]))
            K.ptx.ld.shared.b32(valid, info.ptr_to([base + 3]))
            if mn_limit is not None:
                K.ptx.ld.shared.b32(mn_limit, info.ptr_to([base + 4]))
            K.ptx.fence.proxy.async_.shared__cta()
            K.ptx.mbarrier.arrive.shared.b64(tile_pipe.empty.ptr_to([state.stage]), K.uint32(1))
            _advance(state)

        def drain_empty(pipe, state, stages):
            with K.unroll(0, stages) as _stage:
                _wait(pipe.empty.ptr_to([state.stage]), state.phase)
                _advance(state)
            del _stage

        with scheduler_role:
            state = K.PipelineState(2, phase=1)
            work = K.local_scalar("int32", init=work_id)
            active_tiles = K.local_scalar("int32")
            K.ptx.ld.global_.s32(active_tiles, num_non_exiting_tiles.ptr_to([0]))
            with K.While(work < active_tiles * n_tiles):
                tile_m = work // n_tiles
                tile_n = work % n_tiles
                _wait(tile_pipe.empty.ptr_to([state.stage]), state.phase)
                expert = K.local_scalar("int32")
                limit = K.local_scalar("int32")
                K.ptx.ld.global_.s32(expert, tile_idx_to_expert_idx.ptr_to([tile_m]))
                K.ptx.ld.global_.s32(limit, tile_idx_to_mn_limit.ptr_to([tile_m]))
                with K.If(_elected()):
                    with K.Then():
                        base = state.stage * 5
                        K.ptx.st.shared.b32(info.ptr_to([base]), tile_m)
                        K.ptx.st.shared.b32(info.ptr_to([base + 1]), tile_n)
                        K.ptx.st.shared.b32(info.ptr_to([base + 2]), expert)
                        K.ptx.st.shared.b32(info.ptr_to([base + 3]), K.int32(1))
                        K.ptx.st.shared.b32(info.ptr_to([base + 4]), limit)
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(4), K.uint32(32))
                K.ptx.mbarrier.arrive.shared.b64(tile_pipe.full.ptr_to([state.stage]), K.uint32(1))
                _advance(state)
                K.assign(work, work + num_clusters)
            _wait(tile_pipe.empty.ptr_to([state.stage]), state.phase)
            with K.If(_elected()):
                with K.Then():
                    base = state.stage * 5
                    K.ptx.st.shared.b32(info.ptr_to([base]), K.int32(0))
                    K.ptx.st.shared.b32(info.ptr_to([base + 1]), K.int32(0))
                    K.ptx.st.shared.b32(info.ptr_to([base + 2]), K.int32(-1))
                    K.ptx.st.shared.b32(info.ptr_to([base + 3]), K.int32(0))
                    K.ptx.st.shared.b32(info.ptr_to([base + 4]), K.int32(-1))
            K.ptx.fence.proxy.async_.shared__cta()
            K.ptx.bar.sync(K.uint32(4), K.uint32(32))
            K.ptx.mbarrier.arrive.shared.b64(tile_pipe.full.ptr_to([state.stage]), K.uint32(1))
            _advance(state)
            drain_empty(tile_pipe, state, 2)

        with gather_a_role:
            thread = (warp - 4) * 32 + lane
            tile_state = K.PipelineState(2, phase=0)
            producer = K.PipelineState(8, phase=1)
            if materialize_gather_a:
                source_bases = K.alloc_local((8,), "uint64")
                row_base = K.local_scalar("int32", init=thread // 8)
                byte_base = K.local_scalar("int32", init=(thread % 8) * 16)
                unswizzled_base = K.local_scalar(
                    "uint32",
                    init=smem_base
                    + K.uint32(_A_OFFSET)
                    + K.cast(row_base * 128 + byte_base, "uint32"),
                )
                swizzled_base = K.local_scalar(
                    "int32",
                    init=K.cast(
                        K.bitwise_xor(
                            unswizzled_base,
                            K.bitwise_and(
                                K.shift_right(unswizzled_base, K.uint32(3)), K.uint32(112)
                            ),
                        )
                        - smem_base,
                        "int32",
                    ),
                )
            else:
                token_rows = K.alloc_local((8,), "int32")
            predicates = K.alloc_local((8,), "uint32")
            tile_m = K.local_scalar("int32")
            tile_n = K.local_scalar("int32")
            expert = K.local_scalar("int32")
            valid = K.local_scalar("int32")
            mn_limit = K.local_scalar("int32")
            wait_and_read_info(tile_state, tile_m, tile_n, expert, valid, mn_limit)
            with K.While(valid != 0):
                for item in range(8):
                    row = thread // 8 + item * 16
                    global_row = tile_m * 128 + row
                    token = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(token, token_id_mapping.ptr_to([global_row]))
                    K.assign(predicates[item], K.cast(global_row < mn_limit, "uint32"))
                    token_row = K.local_scalar(
                        "int32", init=K.if_then_else(global_row < mn_limit, token // 8, 0)
                    )
                    if materialize_gather_a:
                        K.assign(
                            source_bases[item],
                            K.reinterpret(
                                "uint64",
                                a.ptr_to(
                                    [
                                        K.cast(token_row, "int64") * K_dim // 2
                                        + K.cast(byte_base, "int64")
                                    ]
                                ),
                            ),
                        )
                    else:
                        K.assign(token_rows[item], token_row)
                count = K.local_scalar("int32", init=0)
                with K.While(count < k_tiles):
                    _wait(a_pipe.empty.ptr_to([producer.stage]), producer.phase)
                    if materialize_gather_a:
                        count_uniform = K.uniform(count)
                        k_offset = K.local_scalar("uint64")
                        K.ptx.shl.b64(k_offset, K.cast(count_uniform, "uint64"), K.uint32(7))
                        stage_offset = K.local_scalar("int32", init=producer.stage * 16_384)
                        for item in range(8):
                            destination = swizzled_base + stage_offset + item * 2_048
                            K.ptx["cp.async.cg.shared.global.L2::128B"](
                                smem.ptr_to([destination]),
                                K.reinterpret("handle", source_bases[item] + k_offset),
                                16,
                                K.cast(K.if_then_else(predicates[item] != 0, 16, 0), "uint32"),
                            )
                    else:
                        for item in range(8):
                            row = thread // 8 + item * 16
                            byte = (thread % 8) * 16
                            unswizzled = (
                                smem_base + _A_OFFSET + producer.stage * 16_384 + row * 128 + byte
                            )
                            swizzled = K.bitwise_xor(
                                unswizzled,
                                K.bitwise_and(
                                    K.shift_right(unswizzled, K.uint32(3)), K.uint32(112)
                                ),
                            )
                            source = (
                                K.cast(token_rows[item], "int64") * K_dim // 2
                                + K.cast(count, "int64") * 128
                                + byte
                            )
                            K.ptx["cp.async.cg.shared.global.L2::128B"](
                                smem.ptr_to([K.cast(swizzled - smem_base, "int32")]),
                                a.ptr_to([source]),
                                16,
                                K.cast(K.if_then_else(predicates[item] != 0, 16, 0), "uint32"),
                            )
                    K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                        a_pipe.full.ptr_to([producer.stage])
                    )
                    _advance(producer)
                    K.assign(count, count + 1)
                wait_and_read_info(tile_state, tile_m, tile_n, expert, valid, mn_limit)
            drain_empty(a_pipe, producer, 8)

        with gather_sfa_role:
            thread = (warp - 12) * 32 + lane
            tile_state = K.PipelineState(2, phase=0)
            producer = K.PipelineState(8, phase=1)
            tile_m = K.local_scalar("int32")
            tile_n = K.local_scalar("int32")
            expert = K.local_scalar("int32")
            valid = K.local_scalar("int32")
            mn_limit = K.local_scalar("int32")
            wait_and_read_info(tile_state, tile_m, tile_n, expert, valid, mn_limit)
            with K.While(valid != 0):
                global_row = tile_m * 128 + thread
                token = K.local_scalar("int32")
                K.ptx.ld.global_.s32(token, token_id_mapping.ptr_to([global_row]))
                source_row = K.local_scalar(
                    "int32", init=K.if_then_else(global_row < mn_limit, token // 8, 0)
                )
                predicate = K.local_scalar("uint32", init=K.cast(global_row < mn_limit, "uint32"))
                count = K.local_scalar("int32", init=0)
                with K.While(count < k_tiles):
                    _wait(sfa_pipe.empty.ptr_to([producer.stage]), producer.phase)
                    source = (
                        K.cast(source_row, "int64") * (K_dim // 16) + K.cast(count, "int64") * 16
                    )
                    K.ptx["cp.async.cg.shared.global.L2::128B"](
                        smem.ptr_to([_SFA_OFFSET + producer.stage * 2048 + thread * 16]),
                        sfa.ptr_to([source]),
                        16,
                        K.cast(K.if_then_else(predicate != 0, 16, 0), "uint32"),
                    )
                    K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                        sfa_pipe.full.ptr_to([producer.stage])
                    )
                    _advance(producer)
                    K.assign(count, count + 1)
                wait_and_read_info(tile_state, tile_m, tile_n, expert, valid, mn_limit)
            drain_empty(sfa_pipe, producer, 8)

        with transform_role:
            K.ptx.bar.sync(K.uint32(3), K.uint32(288))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            tile_state = K.PipelineState(2, phase=0)
            consumer = K.PipelineState(8, phase=0)
            producer = K.PipelineState(4, phase=1)
            words = K.alloc_local((16,), "uint32")
            tile_m = K.local_scalar("int32")
            tile_n = K.local_scalar("int32")
            expert = K.local_scalar("int32")
            valid = K.local_scalar("int32")
            mn_limit = K.local_scalar("int32")
            wait_and_read_info(tile_state, tile_m, tile_n, expert, valid, mn_limit)
            with K.While(valid != 0):
                count = K.local_scalar("int32", init=0)
                with K.While(count < k_tiles):
                    _wait(sfa_pipe.full.ptr_to([consumer.stage]), consumer.phase)
                    for row_group in range(4):
                        address = _SFA_OFFSET + consumer.stage * 2048 + (row_group * 32 + lane) * 16
                        K.ptx.ld.shared.v4.b32(
                            words[row_group * 4],
                            words[row_group * 4 + 1],
                            words[row_group * 4 + 2],
                            words[row_group * 4 + 3],
                            smem.ptr_to([address]),
                        )
                    _wait(sfa_t_pipe.empty.ptr_to([producer.stage]), producer.phase)
                    for group in range(4):
                        K.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
                            K.cast(tmem_base + 256 + producer.stage * 16 + group * 4, "uint32"),
                            words[group],
                            words[4 + group],
                            words[8 + group],
                            words[12 + group],
                        )
                    K.ptx["tcgen05.wait::st.sync.aligned"]()
                    K.ptx.mbarrier.arrive.shared.b64(
                        sfa_t_pipe.full.ptr_to([producer.stage]), K.uint32(1)
                    )
                    K.ptx.mbarrier.arrive.shared.b64(
                        sfa_pipe.empty.ptr_to([consumer.stage]), K.uint32(1)
                    )
                    _advance(producer)
                    _advance(consumer)
                    K.assign(count, count + 1)
                wait_and_read_info(tile_state, tile_m, tile_n, expert, valid, mn_limit)
            drain_empty(sfa_t_pipe, producer, 4)

        with idle_role:
            pass

        with tma_role:
            tile_state = K.PipelineState(2, phase=0)
            producer = K.PipelineState(8, phase=1)
            tile_m = K.local_scalar("int32")
            tile_n = K.local_scalar("int32")
            expert = K.local_scalar("int32")
            valid = K.local_scalar("int32")
            wait_and_read_info(tile_state, tile_m, tile_n, expert, valid)
            with K.While(valid != 0):
                count = K.local_scalar("int32", init=0)
                with K.While(count < k_tiles):
                    speculative = K.local_scalar("uint32")
                    _try_wait(speculative, b_pipe.empty.ptr_to([producer.stage]), producer.phase)
                    with K.If(speculative == K.uint32(0)):
                        with K.Then():
                            _wait(b_pipe.empty.ptr_to([producer.stage]), producer.phase)
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                b_pipe.full.ptr_to([producer.stage]), K.uint32(18_432)
                            )
                            K.ptx[
                                "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint"
                            ](
                                smem.ptr_to([_B_OFFSET + producer.stage * 16_384]),
                                K.address_of(b_map),
                                K.cast(count * 256, "int32"),
                                K.cast(tile_n * 128, "int32"),
                                K.cast(expert, "int32"),
                                b_pipe.full.ptr_to([producer.stage]),
                                K.uint64(0),
                            )
                            K.ptx[
                                "cp.async.bulk.tensor.4d.shared::cta.global.tile"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint"
                            ](
                                smem.ptr_to([_SFB_OFFSET + producer.stage * 2048]),
                                K.address_of(sfb_map),
                                K.int32(0),
                                K.cast(count * 4, "int32"),
                                K.cast(tile_n, "int32"),
                                K.cast(expert, "int32"),
                                b_pipe.full.ptr_to([producer.stage]),
                                K.uint64(0),
                            )
                    _advance(producer)
                    K.assign(count, count + 1)
                wait_and_read_info(tile_state, tile_m, tile_n, expert, valid)
            drain_empty(b_pipe, producer, 8)

        with mma_role:
            K.ptx.bar.sync(K.uint32(3), K.uint32(288))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            sfb_descriptor = K.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + _SFB_OFFSET)
            )

            def runtime_descriptor(sfa_addr, sfb_addr):
                desc = K.bitwise_and(K.uint32(instr_desc), K.uint32(0x9FFFFFCF))
                desc = K.bitwise_or(
                    desc, K.bitwise_and(K.shift_right(sfa_addr, K.uint32(1)), K.uint32(0x60000000))
                )
                return K.bitwise_or(
                    desc, K.bitwise_and(K.shift_right(sfb_addr, K.uint32(26)), K.uint32(0x30))
                )

            tile_state = K.PipelineState(2, phase=0)
            a_consumer = K.PipelineState(8, phase=0)
            b_consumer = K.PipelineState(8, phase=0)
            sfa_t_consumer = K.PipelineState(4, phase=0)
            acc_producer = K.PipelineState(2, phase=1)
            tile_m = K.local_scalar("int32")
            tile_n = K.local_scalar("int32")
            expert = K.local_scalar("int32")
            valid = K.local_scalar("int32")
            wait_and_read_info(tile_state, tile_m, tile_n, expert, valid)
            with K.While(valid != 0):
                _wait(acc_pipe.empty.ptr_to([acc_producer.stage]), acc_producer.phase)
                count = K.local_scalar("int32", init=0)
                accumulate = K.local_scalar("uint32", init=K.uint32(0))
                with K.While(count < k_tiles):
                    _wait(a_pipe.full.ptr_to([a_consumer.stage]), a_consumer.phase)
                    _wait(b_pipe.full.ptr_to([b_consumer.stage]), b_consumer.phase)
                    _wait(sfa_t_pipe.full.ptr_to([sfa_t_consumer.stage]), sfa_t_consumer.phase)
                    for chunk in range(4):
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
                                    K.cast(tmem_base + 320 + chunk * 4, "uint32"),
                                    sfb_descriptor
                                    + K.cast(b_consumer.stage * 128 + chunk * 32, "uint64"),
                                )
                    for kblock in range(2):
                        sfa_addr = K.cast(
                            tmem_base + 256 + sfa_t_consumer.stage * 16 + kblock * 8, "uint32"
                        )
                        sfb_addr = K.cast(tmem_base + 320 + kblock * 8, "uint32")
                        mma_desc = runtime_descriptor(sfa_addr, sfb_addr)
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx[
                                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4"
                                    ".block_scale.block16.collector::a::discard"
                                ](
                                    K.cast(tmem_base + acc_producer.stage * 128, "uint32"),
                                    a_descriptor
                                    + K.cast(a_consumer.stage * 1024 + kblock * 4, "uint64"),
                                    b_descriptor
                                    + K.cast(b_consumer.stage * 1024 + kblock * 4, "uint64"),
                                    mma_desc,
                                    sfa_addr,
                                    sfb_addr,
                                    K.ptx.pred(
                                        K.cast(
                                            K.if_then_else(kblock == 0, accumulate, K.uint32(1)),
                                            "bool",
                                        )
                                    ),
                                )
                    K.assign(accumulate, K.uint32(1))
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                ".shared::cluster.b64"
                            ](a_pipe.empty.ptr_to([a_consumer.stage]))
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                ".shared::cluster.b64"
                            ](b_pipe.empty.ptr_to([b_consumer.stage]))
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                                ".shared::cluster.b64"
                            ](sfa_t_pipe.empty.ptr_to([sfa_t_consumer.stage]))
                    _advance(a_consumer)
                    _advance(b_consumer)
                    _advance(sfa_t_consumer)
                    K.assign(count, count + 1)
                with K.If(_elected()):
                    with K.Then():
                        K.ptx[
                            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
                        ](acc_pipe.full.ptr_to([acc_producer.stage]))
                _advance(acc_producer)
                wait_and_read_info(tile_state, tile_m, tile_n, expert, valid)
            drain_empty(acc_pipe, acc_producer, 2)

        with epilogue_role:
            with K.If(warp == 0):
                with K.Then():
                    K.ptx["tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32"](
                        tmem_slot.ptr_to([0]), K.uint32(_TMEM_COLUMNS)
                    )
            K.ptx.bar.sync(K.uint32(3), K.uint32(288))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            tile_state = K.PipelineState(2, phase=0)
            acc_consumer = K.PipelineState(2, phase=0)
            up = K.alloc_local((64,), "float32")
            gate = K.alloc_local((64,), "float32")
            raw_scales = K.alloc_local((4,), "float32")
            decoded_scales = K.alloc_local((4,), "float32")
            scale_pairs = K.alloc_local((2,), "uint16")
            decoded_pairs = K.alloc_local((2,), "uint32")
            quant_scales = K.alloc_local((4,), "float32")
            fp4_pairs = K.alloc_local((32,), "uint8")
            packed = K.alloc_local((8,), "uint32")
            norm = K.local_scalar("float32")
            K.ptx.ld.global_.b32(norm, global_scale.ptr_to([0]))
            executed = K.local_scalar("int32", init=0)
            tile_m = K.local_scalar("int32")
            tile_n = K.local_scalar("int32")
            expert = K.local_scalar("int32")
            valid = K.local_scalar("int32")

            def multiply_pair(dst0, dst1, left0, left1, right0, right1):
                both = K.local_scalar("uint64")
                K.ptx.mul.rn.f32x2(
                    both, K.cuda.make_float2(left0, left1), K.cuda.make_float2(right0, right1)
                )
                K.ptx.mov.b64(dst0, dst1, both)

            def add_pair(dst0, dst1, left0, left1, right0, right1):
                both = K.local_scalar("uint64")
                K.ptx.add.rn.f32x2(
                    both, K.cuda.make_float2(left0, left1), K.cuda.make_float2(right0, right1)
                )
                K.ptx.mov.b64(dst0, dst1, both)

            wait_and_read_info(tile_state, tile_m, tile_n, expert, valid)
            with K.While(valid != 0):
                _wait(acc_pipe.full.ptr_to([acc_consumer.stage]), acc_consumer.phase)
                alpha_value = K.local_scalar("float32")
                K.ptx.ld.global_.b32(alpha_value, alpha.ptr_to([expert]))
                load_base = K.local_scalar(
                    "uint32", init=tmem_base + (warp << 21) + acc_consumer.stage * 128
                )
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"](
                    *[up[index] for index in range(64)], load_base
                )
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"](
                    *[gate[index] for index in range(64)], load_base + K.uint32(64)
                )
                for index in range(0, 64, 2):
                    scaled_up0 = K.local_scalar("float32")
                    scaled_up1 = K.local_scalar("float32")
                    scaled_gate0 = K.local_scalar("float32")
                    scaled_gate1 = K.local_scalar("float32")
                    reciprocal0 = K.local_scalar("float32")
                    reciprocal1 = K.local_scalar("float32")
                    multiply_pair(
                        scaled_up0, scaled_up1, up[index], up[index + 1], alpha_value, alpha_value
                    )
                    multiply_pair(
                        scaled_gate0,
                        scaled_gate1,
                        gate[index],
                        gate[index + 1],
                        alpha_value,
                        alpha_value,
                    )
                    multiply_pair(
                        reciprocal0,
                        reciprocal1,
                        scaled_gate0,
                        scaled_gate1,
                        K.float32(-1.4426950408889634),
                        K.float32(-1.4426950408889634),
                    )
                    K.ptx.ex2.approx.ftz.f32(reciprocal0, reciprocal0)
                    K.ptx.ex2.approx.ftz.f32(reciprocal1, reciprocal1)
                    add_pair(
                        reciprocal0,
                        reciprocal1,
                        reciprocal0,
                        reciprocal1,
                        K.float32(1.0),
                        K.float32(1.0),
                    )
                    K.ptx.rcp.approx.ftz.f32(reciprocal0, reciprocal0)
                    K.ptx.rcp.approx.ftz.f32(reciprocal1, reciprocal1)
                    multiply_pair(
                        reciprocal0,
                        reciprocal1,
                        reciprocal0,
                        reciprocal1,
                        scaled_gate0,
                        scaled_gate1,
                    )
                    multiply_pair(
                        up[index], up[index + 1], reciprocal0, reciprocal1, scaled_up0, scaled_up1
                    )

                for group in range(4):
                    group_max = K.local_scalar("float32", init=K.float32(0.0))
                    for inner in range(16):
                        absolute = K.local_scalar("float32")
                        K.ptx.abs.f32(absolute, up[group * 16 + inner])
                        K.ptx.max.NaN.f32(group_max, group_max, absolute)
                    K.assign(raw_scales[group], group_max)
                multiply_pair(
                    raw_scales[0],
                    raw_scales[1],
                    raw_scales[0],
                    raw_scales[1],
                    K.float32(1.0 / 6.0),
                    K.float32(1.0 / 6.0),
                )
                multiply_pair(
                    raw_scales[2],
                    raw_scales[3],
                    raw_scales[2],
                    raw_scales[3],
                    K.float32(1.0 / 6.0),
                    K.float32(1.0 / 6.0),
                )
                multiply_pair(
                    raw_scales[0], raw_scales[1], raw_scales[0], raw_scales[1], norm, norm
                )
                multiply_pair(
                    raw_scales[2], raw_scales[3], raw_scales[2], raw_scales[3], norm, norm
                )
                K.ptx.cvt.rn.satfinite.e4m3x2.f32(scale_pairs[0], raw_scales[1], raw_scales[0])
                K.ptx.cvt.rn.satfinite.e4m3x2.f32(scale_pairs[1], raw_scales[3], raw_scales[2])
                global_row = tile_m * 128 + warp * 32 + lane
                rest_k = N // 128
                sfc_offset = ((tile_m * rest_k + tile_n) * 32 + global_row % 32) * 16 + (
                    (global_row // 32) % 4
                ) * 4
                K.ptx.st.global_.v2.b16(sfc.ptr_to([sfc_offset]), scale_pairs[0], scale_pairs[1])
                for pair in range(2):
                    K.ptx.cvt.rn.f16x2.e4m3x2(decoded_pairs[pair], scale_pairs[pair])
                    K.ptx.cvt.f32.f16(
                        decoded_scales[pair * 2],
                        K.cast(K.bitwise_and(decoded_pairs[pair], K.uint32(0xFFFF)), "uint16"),
                    )
                    K.ptx.cvt.f32.f16(
                        decoded_scales[pair * 2 + 1],
                        K.cast(K.shift_right(decoded_pairs[pair], K.uint32(16)), "uint16"),
                    )
                for group in range(4):
                    K.ptx.rcp.approx.ftz.f32(quant_scales[group], decoded_scales[group])
                multiply_pair(
                    quant_scales[0], quant_scales[1], quant_scales[0], quant_scales[1], norm, norm
                )
                multiply_pair(
                    quant_scales[2], quant_scales[3], quant_scales[2], quant_scales[3], norm, norm
                )
                for group in range(4):
                    K.ptx["min.NaN.f32"](
                        quant_scales[group], quant_scales[group], K.float32(3.4028234663852886e38)
                    )
                    for inner in range(0, 16, 2):
                        index = group * 16 + inner
                        multiply_pair(
                            up[index],
                            up[index + 1],
                            up[index],
                            up[index + 1],
                            quant_scales[group],
                            quant_scales[group],
                        )
                for index in range(32):
                    K.ptx.cvt.rn.satfinite.e2m1x2.f32(
                        fp4_pairs[index], up[index * 2 + 1], up[index * 2]
                    )
                for word in range(8):
                    K.assign(
                        packed[word],
                        K.bitwise_or(
                            K.bitwise_or(
                                K.cast(fp4_pairs[word * 4], "uint32"),
                                K.shift_left(
                                    K.cast(fp4_pairs[word * 4 + 1], "uint32"), K.uint32(8)
                                ),
                            ),
                            K.bitwise_or(
                                K.shift_left(
                                    K.cast(fp4_pairs[word * 4 + 2], "uint32"), K.uint32(16)
                                ),
                                K.shift_left(
                                    K.cast(fp4_pairs[word * 4 + 3], "uint32"), K.uint32(24)
                                ),
                            ),
                        ),
                    )
                c_stage = (executed + 1) % 9
                for vector in range(2):
                    unswizzled = (
                        smem_base
                        + _C_OFFSET
                        + c_stage * 4096
                        + warp * 1024
                        + lane * 32
                        + vector * 16
                    )
                    swizzled = K.bitwise_xor(
                        unswizzled,
                        K.bitwise_and(K.shift_right(unswizzled, K.uint32(3)), K.uint32(16)),
                    )
                    K.ptx.st.shared.v4.b32(
                        smem.ptr_to([K.cast(swizzled - smem_base, "int32")]),
                        packed[vector * 4],
                        packed[vector * 4 + 1],
                        packed[vector * 4 + 2],
                        packed[vector * 4 + 3],
                    )
                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                with K.If(warp == 0):
                    with K.Then():
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.global.shared::cta.tile"
                                    ".bulk_group.L2::cache_hint"
                                ](
                                    K.address_of(c_map),
                                    K.cast(tile_n * 64, "int32"),
                                    K.cast(tile_m * 128, "int32"),
                                    K.int32(0),
                                    smem.ptr_to([_C_OFFSET + c_stage * 4096]),
                                    K.uint64(0),
                                )
                                K.ptx.cp.async_.bulk.commit_group()
                                K.ptx.cp.async_.bulk.wait_group.read(8)
                K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                with K.If(_elected()):
                    with K.Then():
                        K.ptx.mbarrier.arrive.shared.b64(
                            acc_pipe.empty.ptr_to([acc_consumer.stage]), K.uint32(1)
                        )
                _advance(acc_consumer)
                K.assign(executed, executed + 1)
                wait_and_read_info(tile_state, tile_m, tile_n, expert, valid)
            with K.If(warp == 0):
                with K.Then():
                    K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(128))
            with K.If(warp == 0):
                with K.Then():
                    K.ptx["tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32"](
                        tmem_base, K.uint32(_TMEM_COLUMNS)
                    )
            K.ptx.cp.async_.bulk.wait_group.read(0)

        if use_pdl:
            K.ptx.griddepcontrol.launch_dependents()

        required_block_size.__exit__(None, None, None)

    kernel.__annotations__ = {
        "a": K.gptr[K.u8, (seq_len * K_dim // 2,)],
        "b": K.gptr[K.u8, (num_experts * N * K_dim // 2,)],
        "sfa": K.gptr[K.u8, (seq_len * K_dim // 16,)],
        "sfb": K.gptr[K.u8, (num_experts * N * K_dim // 16,)],
        "c": K.gptr[K.u8, (permuted_m * N // 4,)],
        "sfc": K.gptr[K.u8, (permuted_m * N // 32,)],
        "alpha": K.gptr[K.f32, (num_experts,)],
        "tile_idx_to_expert_idx": K.gptr[K.i32, (permuted_tiles,)],
        "tile_idx_to_mn_limit": K.gptr[K.i32, (permuted_tiles,)],
        "token_id_mapping": K.gptr[K.i32, (permuted_m,)],
        "num_non_exiting_tiles": K.gptr[K.i32, (1,)],
        "global_scale": K.gptr[K.f32, (1,)],
    }
    return K.kernel(
        warps=20,
        arch="sm_107a",
        min_blocks_per_sm=1,
        grid=[1, 1, num_clusters],
        host_prelude=host_prelude,
    )(kernel)


def _config_dict(**config):
    answer = {
        "num_experts": int(config["num_experts"]),
        "seq_len": int(config["seq_len"]),
        "expected_m_per_expert": int(config.get("expected_m_per_expert", 0)),
        "N": int(config["N"]),
        "K": int(config["K"]),
        "routing": config.get("routing", "balanced"),
        "a_dtype": config.get("a_dtype", "float4_e2m1fn"),
        "b_dtype": config.get("b_dtype", "float4_e2m1fn"),
        "sf_dtype": config.get("sf_dtype", "float8_e4m3fn"),
        "sf_vec_size": int(config.get("sf_vec_size", 16)),
        "c_dtype": config.get("c_dtype", "float4_e2m1fn"),
        "topk": int(config.get("topk", 8)),
        "tactic": int(config.get("tactic", 0)),
        "a_path": config.get("a_path", "cpasync"),
        "vectorized_f32": bool(config.get("vectorized_f32", True)),
        "raster_along_m": bool(config.get("raster_along_m", False)),
        "use_pdl": bool(config.get("use_pdl", True)),
        "alpha": float(config.get("alpha", 1.0)),
        "ugpu_half_gemm": bool(config.get("ugpu_half_gemm", False)),
        "label": config.get("label", ""),
    }
    _validate_config(answer)
    if answer["alpha"] != 1.0:
        raise ValueError("production Rubin specialization requires alpha=1.0")
    _problem(answer["num_experts"], answer["seq_len"], answer["N"], answer["K"], answer["routing"])
    return answer


def get_kernel(**raw_config):
    """Return the fixed production Rubin specialization for one concrete shape."""
    from tirx_kernels.runner import hardware_num_sms

    config = _config_dict(**raw_config)
    return _make_kernel(
        config["num_experts"],
        config["seq_len"],
        config["N"],
        config["K"],
        config["routing"],
        hardware_num_sms(216),
        config["use_pdl"],
    ).func


_SOURCE_FILES = {
    _SOURCE_ROOT / "flashinfer/fused_moe/cute_dsl/rubin/"
    "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py": SOURCE_SHA256,
    **{
        _SOURCE_ROOT / "flashinfer/fused_moe/cute_dsl/rubin" / name: digest
        for name, digest in SOURCE_DEPENDENCY_SHA256.items()
        if name in {"custom_pipeline.py", "inline_ptx.py", "utils.py"}
    },
    **{
        _SOURCE_ROOT / "flashinfer/fused_moe/cute_dsl" / name: digest
        for name, digest in SOURCE_DEPENDENCY_SHA256.items()
        if name in {"blockscaled_contiguous_gather_grouped_gemm_act_fusion.py", "tuner.py"}
    },
}


@cache
def _source_module():
    for path, expected in _SOURCE_FILES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen FlashInfer source hash mismatch: {path} sha256={actual}")
    source_root = str(_SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion"
    )


def _physical_u8(torch, tensor):
    return torch.as_strided(tensor, (tensor.numel(),), (1,)).view(torch.uint8)


def prepare_data(**raw_config):
    """Create deterministic raw FP4/E4M3 inputs accepted by both implementations."""
    import torch

    config = _config_dict(**raw_config)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        raise RuntimeError("this Rubin kernel requires an SM107 CUDA device")
    counts, permuted_m, _ = _problem(
        config["num_experts"], config["seq_len"], config["N"], config["K"], config["routing"]
    )
    seed = (
        config["num_experts"] * 1_000_003
        + config["seq_len"] * 10_009
        + config["N"] * 101
        + config["K"]
    ) & 0x7FFFFFFF
    torch.manual_seed(seed)
    device = torch.device("cuda")
    a = torch.empty(
        (config["seq_len"], config["K"] // 2), device=device, dtype=torch.uint8
    ).random_(0, 256)
    b = torch.empty(
        (config["num_experts"], config["N"], config["K"] // 2), device=device, dtype=torch.uint8
    ).random_(0, 256)
    # 0x38 is exactly +1.0 in E4M3.  Constant scales keep data creation cheap
    # for the 1 GiB production weight profiles while the random FP4 payloads
    # exercise every finite E2M1 code and sign bit.
    sfa = torch.full((config["seq_len"], config["K"] // 16), 0x38, device=device, dtype=torch.uint8)
    sfb = torch.full(
        (config["num_experts"] * config["N"] * config["K"] // 16,),
        0x38,
        device=device,
        dtype=torch.uint8,
    )
    source = _source_module()
    (
        token_id_mapping,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
        source_permuted_m,
        _aligned_counts,
    ) = source.create_gather_gemm_tensors(config["seq_len"], config["topk"], list(counts), _TILE_M)
    if source_permuted_m != permuted_m:
        raise AssertionError(
            f"routing layout mismatch: TIRx={permuted_m}, source={source_permuted_m}"
        )
    alpha = torch.ones(config["num_experts"], device=device, dtype=torch.float32)
    global_scale = torch.ones(1, device=device, dtype=torch.float32)
    tirx_c = torch.full((permuted_m, config["N"] // 4), 0xA5, device=device, dtype=torch.uint8)
    source_c = torch.full_like(tirx_c, 0xA5)
    sfc_shape = (32, 4, permuted_m // 128, 4, config["N"] // 128, 1)
    tirx_sfc = torch.full(sfc_shape, 0xA5, device=device, dtype=torch.uint8)
    source_sfc = torch.full_like(tirx_sfc, 0xA5)
    return {
        "config": config,
        "source": source,
        "a": a,
        "b": b,
        "sfa": sfa,
        "sfb": sfb,
        "tirx_c": tirx_c,
        "source_c": source_c,
        "tirx_sfc": tirx_sfc,
        "source_sfc": source_sfc,
        "alpha": alpha,
        "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "token_id_mapping": token_id_mapping,
        "num_non_exiting_tiles": num_non_exiting_tiles,
        "global_scale": global_scale,
    }


@cache
def _compile_executable(num_experts, seq_len, N, K_dim, routing, use_pdl):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(
            num_experts=num_experts, seq_len=seq_len, N=N, K=K_dim, routing=routing, use_pdl=use_pdl
        )
    )


def _tirx_launch(executable, data):
    def launch():
        executable(
            data["a"].flatten(),
            data["b"].flatten(),
            data["sfa"].flatten(),
            data["sfb"],
            data["tirx_c"].flatten(),
            data["tirx_sfc"].flatten(),
            data["alpha"],
            data["tile_idx_to_expert_idx"],
            data["tile_idx_to_mn_limit"],
            data["token_id_mapping"],
            data["num_non_exiting_tiles"],
            data["global_scale"],
        )

    launch._keep_alive = data
    return launch


def _source_launch(data):
    config = data["config"]
    source = data["source"]

    def launch():
        source.blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
            data["a"],
            data["b"],
            data["sfa"],
            data["sfb"],
            data["alpha"],
            data["tile_idx_to_expert_idx"],
            data["tile_idx_to_mn_limit"],
            data["token_id_mapping"],
            data["num_non_exiting_tiles"],
            out=data["source_c"],
            out_scale=data["source_sfc"],
            global_scale=data["global_scale"],
            topk=config["topk"],
            ab_dtype=config["a_dtype"],
            sf_dtype=config["sf_dtype"],
            c_dtype=config["c_dtype"],
            sf_vec_size=config["sf_vec_size"],
            mma_tiler=(_TILE_M, _TILE_N, _TILE_K),
            mma_inst_shape=(128, 128, 128),
            cluster_shape_mn=(1, 1),
            vectorized_f32=config["vectorized_f32"],
            raster_along_m=config["raster_along_m"],
            enable_pdl=config["use_pdl"],
        )

    launch._keep_alive = data
    return launch


def _check_outputs(data, with_source):
    import torch

    if not with_source:
        if bool((data["tirx_c"] == 0xA5).any().item()):
            raise AssertionError("TIRx left poison bytes in the full FP4 output allocation")
        if bool((data["tirx_sfc"] == 0xA5).any().item()):
            raise AssertionError("TIRx left poison bytes in the full SFC allocation")
        return {"bitwise": None}
    mismatches = []
    for name in ("c", "sfc"):
        actual = data[f"tirx_{name}"]
        expected = data[f"source_{name}"]
        if not torch.equal(actual, expected):
            differing = int((actual != expected).sum().item())
            mismatches.append(f"{name}: differing_bytes={differing}/{actual.numel()}")
    if mismatches:
        raise AssertionError(
            "Rubin fused gather GEMM full-allocation bitwise mismatch: " + "; ".join(mismatches)
        )
    return {"bitwise": True, "differing_bytes": 0}


def _executable_for(config):
    return _compile_executable(
        config["num_experts"],
        config["seq_len"],
        config["N"],
        config["K"],
        config["routing"],
        config["use_pdl"],
    )


def run_test(**raw_config):
    import torch

    config = _config_dict(**raw_config)
    data = prepare_data(**config)
    tirx = _tirx_launch(_executable_for(config), data)
    source = _source_launch(data)
    tirx()
    source()
    torch.cuda.synchronize()
    result = _check_outputs(data, True)
    c_snapshot = data["tirx_c"].clone()
    sfc_snapshot = data["tirx_sfc"].clone()
    data["tirx_c"].fill_(0xA5)
    data["tirx_sfc"].fill_(0xA5)
    tirx()
    torch.cuda.synchronize()
    if not torch.equal(data["tirx_c"], c_snapshot):
        raise AssertionError("TIRx FP4 output is not bitwise deterministic on repeat launch")
    if not torch.equal(data["tirx_sfc"], sfc_snapshot):
        raise AssertionError("TIRx SFC output is not bitwise deterministic on repeat launch")
    return result


def prepare_bench(**raw_config):
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = _config_dict(**raw_config)
    return prepared_gpu_benchmark(
        run_gpu, {"config": config, "executable": _executable_for(config)}
    )


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **_):
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    data = prepare_data(**prepared["config"])
    tirx = _tirx_launch(prepared["executable"], data)
    tirx()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source = _source_launch(data)
        source()
        torch.cuda.synchronize()
        references = {"flashinfer": lambda: source}
    _check_outputs(data, with_source)
    torch.cuda.empty_cache()
    return bench(
        {"tirx": tirx},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
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
