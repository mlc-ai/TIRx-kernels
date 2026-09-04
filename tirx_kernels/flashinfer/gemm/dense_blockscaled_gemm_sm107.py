# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ 012cfdb97f217e0d48bc9352c17a74068c9e495b)
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM107 dense block-scaled FP4 GEMM transcribed from FlashInfer CuTeDSL.

Upstream sources:
- flashinfer/gemm/kernels/dense_blockscaled_gemm_sm107.py
- flashinfer/gemm/kernels/epilogue_utils.py
- flashinfer/gemm/gemm_base.py
- flashinfer/gemm/kernels/utils.py
"""

import hashlib
import importlib
import importlib.util
import sys
import types
from functools import cache
from pathlib import Path

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "dense_blockscaled_gemm_sm107",
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

SOURCE_COMMIT = "012cfdb97f217e0d48bc9352c17a74068c9e495b"
SOURCE_SHA256 = "c9def937d2bf76b363bb321aa4053ffd39055febdcef3c890572bd2c4f2946ad"
SOURCE_DEPENDENCY_SHA256 = {
    "epilogue_utils.py": "ae48ecfca2220975cd3e0e1d60803f8634aa72827857aaff3b2af24205c01c92",
    "gemm_base.py": "ef5cb58d7d85e1391a4987327bdcd6f9e20ffcd1a849dc0b0b9a13ffce3d0d95",
    "kernels/utils.py": "705a10946cb6ee68c132784fc48de7ce3bad07714ae5244c4058c970e1f02ed2",
}
CUTLASS_PARENT_COMMIT = "cdcf8d86daa9b417840fd99875a1b1af685d389d"
CUTLASS_PARENT_SHA256 = "1517d4cde6b7988d5f44eca5fc2de4516b6f582ae60c37a5d1455a09c647453b"

_SOURCE_ROOT = Path("/root-vol/aarch64-ws/kernel-libs/vr200/flashinfer")
_CUTLASS_ROOT = Path(__file__).resolve().parents[3] / ".reference-deps" / "cutlass-v4.8.0dev"
_CUTLASS_PARENT_RELATIVE = Path(
    "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/"
    "dense_blockscaled_gemm_persistent.py"
)
_CUTLASS_PARENT_MODULE = (
    "nvidia_cutlass_dsl.examples.CuTeDSL.cute.blackwell.kernel.blockscaled_gemm."
    "dense_blockscaled_gemm_persistent"
)

_SMEM_CAPACITY = 334_848
_TMEM_COLUMNS = 576
_TRY_WAIT_TICKS = 10_000_000
_MAX_ACTIVE_CLUSTERS = {1: 200, 2: 100, 4: 40}
_SF_MODES = {"nvfp4": ("float8_e4m3fn", 16), "mxfp4": ("float8_e8m0fnu", 32)}
_OUT_DTYPES = ("float16", "bfloat16")

# (MMA tiler MN, MMA instruction MN, cluster MN, swap AB, prefetch distance)
TACTICS = (
    ((256, 128), (128, 128), (2, 1), False, 0),
    ((256, 128), (256, 128), (2, 1), True, 2),
    ((128, 192), (128, 192), (1, 2), False, None),
    ((128, 64), (128, 64), (1, 1), True, 2),
    ((256, 256), (256, 256), (2, 1), False, 0),
)


def _case(prefix, tactic, shape, sf_mode, out_dtype, alpha):
    M, N, K_dim = shape
    return {
        "label": (
            f"{prefix}_t{tactic}_{sf_mode}_{'fp16' if out_dtype == 'float16' else 'bf16'}"
            f"_m{M}_n{N}_k{K_dim}"
        ),
        "M": M,
        "N": N,
        "K": K_dim,
        "sf_mode": sf_mode,
        "out_dtype": out_dtype,
        "alpha": alpha,
        "tactic": tactic,
    }


CONFIGS = [
    _case("guard", 0, (520, 520, 512), "nvfp4", "float16", 0.75),
    _case("guard", 0, (520, 520, 512), "mxfp4", "bfloat16", 0.3333),
    _case("guard", 1, (520, 272, 512), "mxfp4", "bfloat16", 0.3333),
    _case("guard", 2, (272, 392, 768), "nvfp4", "float16", 0.75),
    _case("guard", 2, (272, 392, 768), "mxfp4", "bfloat16", 0.3333),
    _case("guard", 3, (136, 272, 512), "mxfp4", "float16", 0.75),
    _case("guard", 4, (272, 520, 512), "nvfp4", "bfloat16", 0.3333),
]

BENCH_CONFIGS = [
    _case("bench", 0, (4096, 2048, 7168), "nvfp4", "float16", 1.0),
    _case("bench", 1, (4096, 1024, 3072), "mxfp4", "bfloat16", 1.0),
    _case("bench", 2, (1024, 4096, 3072), "nvfp4", "float16", 1.0),
    _case("bench", 3, (256, 10304, 2688), "mxfp4", "float16", 1.0),
    _case("bench", 4, (4096, 2048, 7168), "nvfp4", "bfloat16", 1.0),
]


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align_up(value, alignment):
    return _ceil_div(value, alignment) * alignment


def _descriptor_base(ldo, sdo, swizzle):
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _descriptor_with_address(base, shared_address):
    address_field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x7FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), address_field)


def _instruction_descriptor(inst_m, inst_n, sf_dtype):
    sf_format = {"float8_e4m3fn": 0, "float8_e8m0fnu": 1}[sf_dtype]
    value = (1 << 3) | (1 << 7) | (1 << 10)
    value |= ((inst_n >> 3) & 0x3F) << 17
    value |= (sf_format & 1) << 23
    value |= ((inst_m >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF


def _advance(state):
    state.advance()


def _try_wait_acquire(dst, barrier, phase):
    K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, K.cast(phase, "uint32")
    )


def _wait_plain(barrier, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _wait_plain_if_needed(barrier, phase, speculative_ready):
    with K.If(speculative_ready == K.uint32(0)):
        with K.Then():
            _wait_plain(barrier, phase)


def _elected():
    elected_lane = K.local_scalar("uint32")
    elected_pred = K.local_scalar("uint32")
    K.ptx.elect_sync(elected_lane, elected_pred, K.uint32(0xFFFFFFFF))
    return elected_pred == K.uint32(1)


def _validate_problem(M, N, K_dim, sf_mode, out_dtype, alpha, tactic):
    if not 0 <= tactic < len(TACTICS):
        raise ValueError(f"tactic must be in [0, {len(TACTICS) - 1}], got {tactic}")
    if sf_mode not in _SF_MODES:
        raise ValueError(f"unsupported scale mode {sf_mode!r}")
    if out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported output dtype {out_dtype!r}")
    if min(M, N, K_dim) <= 0 or M % 8 or N % 8 or K_dim % 32:
        raise ValueError("source requires positive M/N multiples of 8 and K multiple of 32")
    if not isinstance(alpha, float | int):
        raise TypeError("alpha must be a host scalar")
    (tile_m, tile_n), (inst_m, inst_n), (cluster_m, cluster_n), swap, _ = TACTICS[tactic]
    if tile_m not in (128, 256) or tile_n not in (64, 128, 192, 256):
        raise ValueError("invalid source MMA tile")
    if inst_m not in (128, 256) or inst_n != tile_n or tile_m not in (inst_m, 2 * inst_m):
        raise ValueError("invalid source instruction/tile relation")
    cta_group = inst_m // 128
    if cluster_m % cta_group:
        raise ValueError("cluster M must be divisible by the CTA group")
    kernel_m, kernel_n = (N, M) if swap else (M, N)
    cta_m = tile_m // cta_group
    if kernel_m < cta_m * cluster_m or kernel_n < tile_n * cluster_n:
        raise ValueError("problem does not fill the selected source cluster")


@cache
def _make_kernel(M, N, K_dim, sf_mode, out_dtype, alpha, tactic):
    _validate_problem(M, N, K_dim, sf_mode, out_dtype, alpha, tactic)
    (tile_m, n_tile), (inst_m, inst_n), (cluster_m, cluster_n), swap, prefetch = TACTICS[tactic]
    sf_dtype, sf_vec_size = _SF_MODES[sf_mode]
    cta_group = inst_m // 128
    b_reuse = tile_m == 2 * inst_m
    cta_m = tile_m // cta_group
    b_rows = n_tile // cta_group
    cluster_size = cluster_m * cluster_n
    cluster_m_groups = cluster_m // cta_group
    kernel_m, kernel_n = (N, M) if swap else (M, N)
    m_tiles = _ceil_div(kernel_m, cta_m)
    n_tiles = _ceil_div(kernel_n, n_tile)
    cluster_m_tiles = _ceil_div(m_tiles, cluster_m)
    cluster_n_tiles = _ceil_div(n_tiles, cluster_n)
    cluster_work = cluster_m_tiles * cluster_n_tiles
    num_clusters = min(cluster_work, _MAX_ACTIVE_CLUSTERS[cluster_size])
    k_tile = 256
    k_tiles = _ceil_div(K_dim, k_tile)

    acc_stages = 1 if b_reuse and n_tile in (192, 256) else 2
    a_stage_bytes = cta_m * k_tile // 2
    b_stage_bytes = b_rows * k_tile // 2
    sfa_stage_bytes = cta_m * k_tile // sf_vec_size
    sfb_stage_bytes = _align_up(n_tile, 128) * k_tile // sf_vec_size
    ab_stage_bytes = a_stage_bytes + b_stage_bytes + sfa_stage_bytes + sfb_stage_bytes
    epi_n = 32
    c_stage_bytes = 128 * epi_n * 2
    ab_stages = (_SMEM_CAPACITY - (1024 + 2 * c_stage_bytes)) // ab_stage_bytes
    c_stages = (
        2
        + (_SMEM_CAPACITY - ab_stages * ab_stage_bytes - (1024 + 2 * c_stage_bytes))
        // c_stage_bytes
    )
    prefetch_distance = 0 if tactic in (2, 3) else (ab_stages if prefetch is None else prefetch)
    alpha_is_one = float(alpha) == 1.0

    ab_full_offset = 0
    ab_empty_offset = ab_stages * 8
    acc_full_offset = 2 * ab_stages * 8
    acc_empty_offset = acc_full_offset + acc_stages * 8
    tmem_dealloc_offset = acc_empty_offset + acc_stages * 8
    tmem_ptr_offset = tmem_dealloc_offset + 8
    c_offset = 1024
    a_offset = c_offset + c_stages * c_stage_bytes
    b_offset = a_offset + ab_stages * a_stage_bytes
    sfa_offset = b_offset + ab_stages * b_stage_bytes
    sfb_offset = _align_up(sfa_offset + ab_stages * sfa_stage_bytes, 1024)
    shared_bytes = _align_up(sfb_offset + ab_stages * sfb_stage_bytes, 1024)
    if shared_bytes > _SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {_SMEM_CAPACITY}")

    ab_empty_arrivals = cluster_n + cluster_m_groups - 1
    acc_empty_arrivals = 4 * cta_group
    num_tma_load_bytes = ab_stage_bytes * cta_group
    acc_columns = n_tile * acc_stages * (2 if b_reuse else 1)
    sfa_chunks = sfa_stage_bytes // 512
    sfb_chunks = sfb_stage_bytes // 512
    sfa_tmem_column = acc_columns
    sfb_tmem_column = sfa_tmem_column + sfa_chunks * 4
    tmem_columns = _TMEM_COLUMNS
    if sfb_tmem_column + sfb_chunks * 4 + (2 if n_tile in (64, 192) else 0) > tmem_columns:
        raise ValueError("source TMEM intervals exceed the SM107 allocation")

    a_desc_base = _descriptor_base(1, 64, 3)
    b_desc_base = _descriptor_base(1, 64, 3)
    sf_desc_base = _descriptor_base(1, 8, 0)
    instr_desc = _instruction_descriptor(inst_m, inst_n, sf_dtype)
    sf_k_box = k_tile // (4 * sf_vec_size)
    sfb_n_box = _ceil_div(n_tile, 128)
    a_cluster_piece = cta_m // cluster_n
    b_cluster_piece = b_rows // cluster_m_groups
    a_piece_bytes = a_stage_bytes // cluster_n
    b_piece_bytes = b_stage_bytes // cluster_m_groups
    sfa_piece_values = 256 * sf_k_box // cluster_n
    sfa_m_box = cta_m // 128
    sfb_piece_values = 256 * sf_k_box * sfb_n_box // cluster_m
    epilogue_subtiles = (cta_m // 128) * (n_tile // epi_n)
    tma_cache_hint = 0

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        c = params["c"]
        sfa = params["sfa"]
        sfb = params["sfb"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        sfa_map = K.stack_alloca("tensormap", 1)
        sfb_map = K.stack_alloca("tensormap", 1)
        c_map = K.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, rank, data, *fields):
            K.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

        encode(
            a_map,
            "float4_e2m1fn",
            2,
            a.data,
            K_dim,
            kernel_m,
            K_dim // 2,
            k_tile,
            a_cluster_piece,
            1,
            1,
            0,
            3,
            2,
            0,
            13,
        )
        encode(
            b_map,
            "float4_e2m1fn",
            2,
            b.data,
            K_dim,
            kernel_n,
            K_dim // 2,
            k_tile,
            b_cluster_piece,
            1,
            1,
            0,
            3,
            2,
            0,
            13,
        )
        sf_k_groups = _ceil_div(K_dim, 4 * sf_vec_size)
        sf_m_groups = _ceil_div(kernel_m, 128)
        sf_n_groups = _ceil_div(kernel_n, 128)
        sfa_box_0 = min(256, sfa_piece_values)
        sfa_box_1 = sfa_piece_values // sfa_box_0
        encode(
            sfa_map,
            "uint16",
            3,
            sfa.data,
            256,
            sf_k_groups,
            sf_m_groups,
            512,
            sf_k_groups * 512,
            sfa_box_0,
            sfa_box_1,
            sfa_m_box,
            1,
            1,
            1,
            0,
            0,
            2,
            0,
        )
        sfb_box_0 = min(256, sfb_piece_values)
        sfb_remaining = sfb_piece_values // sfb_box_0
        sfb_box_1 = min(sf_k_box, sfb_remaining)
        sfb_box_2 = sfb_remaining // sfb_box_1
        encode(
            sfb_map,
            "uint16",
            3,
            sfb.data,
            256,
            sf_k_groups,
            sf_n_groups,
            512,
            sf_k_groups * 512,
            sfb_box_0,
            sfb_box_1,
            sfb_box_2,
            1,
            1,
            1,
            0,
            0,
            2,
            0,
        )
        if swap:
            encode(
                c_map,
                out_dtype,
                2,
                c.data,
                kernel_m,
                kernel_n,
                kernel_m * 2,
                64,
                epi_n,
                1,
                1,
                0,
                3,
                2,
                0,
            )
        else:
            encode(
                c_map,
                out_dtype,
                2,
                c.data,
                kernel_n,
                kernel_m,
                kernel_n * 2,
                epi_n,
                128,
                1,
                1,
                0,
                2,
                2,
                0,
            )
        return a_map, b_map, sfa_map, sfb_map, c_map

    def kernel(a, b, sfa, sfb, c, alpha_ptr, *, host):
        del a, b, sfa, sfb, c
        required_block_size = K.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        a_map, b_map, sfa_map, sfb_map, c_map = host

        if alpha_is_one:
            alpha_value = K.local_scalar("float32", init=K.float32(1.0))
        else:
            alpha_raw = K.local_scalar("float32")
            K.ptx.ld.global_.f32(alpha_raw, alpha_ptr.ptr_to([0]))
            alpha_bits = K.local_scalar("uint16")
            alpha_value = K.local_scalar("float32")
            if out_dtype == "float16":
                K.ptx.cvt.rn.f16.f32(alpha_bits, alpha_raw)
                K.ptx.cvt.f32.f16(alpha_value, alpha_bits)
            else:
                K.ptx.cvt.rn.bf16.f32(alpha_bits, alpha_raw)
                K.ptx.cvt.f32.bf16(alpha_value, alpha_bits)

        _block_x, _block_y, cluster_work_id = K.cta_id()
        cluster_x_scope, cluster_y_scope = K.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        del _block_x, _block_y, cluster_x_scope, cluster_y_scope
        cluster_rank = K.local_scalar("int32", init=K.cuda.mov_sreg(32, "cluster_ctarank"))
        if cluster_m == 1:
            cluster_x = K.local_scalar("int32", init=0)
        else:
            cluster_x = K.local_scalar("int32", init=cluster_rank & (cluster_m - 1))
        if cluster_n == 1:
            cluster_y = K.local_scalar("int32", init=0)
        else:
            cluster_y = K.local_scalar("int32", init=cluster_rank >> (cluster_m.bit_length() - 1))
        if cta_group == 2:
            cta_v = cluster_x & 1
            leader_cta = cta_v == 0
            cluster_m_group = cluster_x >> 1
            pair_leader_x = cluster_m_group << 1
            leader_rank = pair_leader_x + cluster_m * cluster_y
        else:
            cta_v = K.local_scalar("int32", init=0)
            leader_cta = K.bool(True)
            cluster_m_group = cluster_x
            pair_leader_x = cluster_x
            leader_rank = cluster_rank
        warp = K.warp_id()
        lane = K.lane_id()

        roles = K.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])

        smem = K.alloc_buffer((shared_bytes,), K.u8, scope="shared.dyn", align=1024)
        protocol_pool = K.smem_pool(base=smem)
        ab_pipe = K.Pipeline(
            protocol_pool,
            ab_stages,
            full="tma",
            empty="tcgen05",
            init_empty=ab_empty_arrivals,
            leader=K.bool(False),
        )
        acc_pipe = K.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=acc_empty_arrivals,
            leader=K.bool(False),
        )
        if protocol_pool.bytes != tmem_dealloc_offset:
            raise AssertionError("protocol storage offsets changed")
        tmem_dealloc = protocol_pool.alloc((1,), K.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), K.u32, align=4)
        if protocol_pool.bytes != tmem_ptr_offset + 4:
            raise AssertionError("protocol storage header changed")

        with tma_role:
            K.ptx.prefetch.tensormap(K.address_of(a_map))
            K.ptx.prefetch.tensormap(K.address_of(b_map))
            K.ptx.prefetch.tensormap(K.address_of(sfa_map))
            K.ptx.prefetch.tensormap(K.address_of(sfb_map))
            K.ptx.prefetch.tensormap(K.address_of(c_map))

        with K.If(warp == 0):
            with K.Then():
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, ab_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, ab_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), K.uint32(ab_empty_arrivals)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, acc_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, acc_stages) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), K.uint32(acc_empty_arrivals)
                            )
        if cta_group == 2:
            with K.If(warp == 0):
                with K.Then():
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), K.uint32(32))
            K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.fence.mbarrier_init.release.cluster()
        if cluster_size > 1:
            K.ptx.barrier.cluster.arrive.relaxed()

        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        tmem_slot_addr = K.uniform(smem_base + K.uint32(tmem_ptr_offset))
        cluster_smem_u64 = K.local_scalar("uint64")
        K.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = K.local_scalar("uint32", init=K.cast(cluster_smem_u64, "uint32"))
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + a_offset)
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + b_offset)
        )

        a_mcast_mask = K.local_scalar("uint32", init=K.uint32(0))
        for peer_n in range(cluster_n):
            K.assign(
                a_mcast_mask,
                K.bitwise_or(
                    a_mcast_mask, K.uint32(1) << K.cast(cluster_x + cluster_m * peer_n, "uint32")
                ),
            )
        b_mcast_mask = K.local_scalar("uint32", init=K.uint32(0))
        for peer_group in range(cluster_m_groups):
            peer_x = cta_v + cta_group * peer_group
            K.assign(
                b_mcast_mask,
                K.bitwise_or(
                    b_mcast_mask, K.uint32(1) << K.cast(peer_x + cluster_m * cluster_y, "uint32")
                ),
            )
        sfb_mcast_mask = K.local_scalar("uint32", init=K.uint32(0))
        for peer_x in range(cluster_m):
            K.assign(
                sfb_mcast_mask,
                K.bitwise_or(
                    sfb_mcast_mask, K.uint32(1) << K.cast(peer_x + cluster_m * cluster_y, "uint32")
                ),
            )
        ab_consumer_mask = K.local_scalar("uint32", init=K.uint32(0))
        for pair_v in range(cta_group):
            for peer_n in range(cluster_n):
                K.assign(
                    ab_consumer_mask,
                    K.bitwise_or(
                        ab_consumer_mask,
                        K.uint32(1)
                        << K.cast(pair_leader_x + pair_v + cluster_m * peer_n, "uint32"),
                    ),
                )
            for peer_group in range(cluster_m_groups):
                peer_x = pair_v + cta_group * peer_group
                K.assign(
                    ab_consumer_mask,
                    K.bitwise_or(
                        ab_consumer_mask,
                        K.uint32(1) << K.cast(peer_x + cluster_m * cluster_y, "uint32"),
                    ),
                )
        if cta_group == 2:
            acc_producer_mask = K.local_scalar(
                "uint32", init=K.uint32(3) << K.cast(leader_rank, "uint32")
            )
        else:
            acc_producer_mask = K.local_scalar(
                "uint32", init=K.uint32(1) << K.cast(cluster_rank, "uint32")
            )
        ab_full_leader = ab_pipe.full.remote_view(leader_rank)
        acc_empty_leader = acc_pipe.empty.remote_view(leader_rank)
        if cluster_size > 1:
            K.ptx.barrier.cluster.wait()
        else:
            K.ptx.bar.sync(K.uint32(0), K.uint32(192))

        def scheduler_coords(work):
            cluster_m_idx = work % cluster_m_tiles
            cluster_n_idx = work // cluster_m_tiles
            tile_m_idx = cluster_m_idx * cluster_m + cluster_x
            tile_n_idx = cluster_n_idx * cluster_n + cluster_y
            return tile_m_idx, tile_n_idx

        def advance_work(work):
            K.assign(work, work + num_clusters)

        def cta2_tma_barrier(stage):
            # Bit 24 is the CTA-in-pair selector; CTA2 completions target rank 0.
            return K.bitwise_and(
                K.cuda.cvta_generic_to_shared(ab_pipe.full.ptr_to([stage])), K.uint32(0xFEFFFFF8)
            )

        def prefetch_inputs(tile_m_idx, tile_n_idx, future_k):
            a_coord_m = tile_m_idx * cta_m + cluster_y * a_cluster_piece
            a_coord_k = future_k * k_tile
            b_coord_n = tile_n_idx * n_tile + cta_v * b_rows + cluster_m_group * b_cluster_piece
            b_coord_k = future_k * k_tile
            sfa_linear = cluster_y * sfa_piece_values
            sfa_coord_0 = sfa_linear % 256
            sfa_coord_1 = future_k * sf_k_box + sfa_linear // 256
            sfb_linear = cluster_x * sfb_piece_values
            sfb_coord_0 = sfb_linear % 256
            sfb_quotient = sfb_linear // 256
            sfb_coord_1 = future_k * sf_k_box + sfb_quotient % sf_k_box
            sfb_tile_group = tile_n_idx * n_tile // 128
            sfb_coord_2 = sfb_tile_group + sfb_quotient // sf_k_box
            K.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"](
                K.address_of(a_map),
                K.cast(a_coord_k, "int32"),
                K.cast(a_coord_m, "int32"),
                K.uint64(tma_cache_hint),
            )
            K.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"](
                K.address_of(b_map),
                K.cast(b_coord_k, "int32"),
                K.cast(b_coord_n, "int32"),
                K.uint64(tma_cache_hint),
            )
            K.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"](
                K.address_of(sfa_map),
                K.cast(sfa_coord_0, "int32"),
                K.cast(sfa_coord_1, "int32"),
                K.cast(tile_m_idx * sfa_m_box, "int32"),
                K.uint64(tma_cache_hint),
            )
            K.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"](
                K.address_of(sfb_map),
                K.cast(sfb_coord_0, "int32"),
                K.cast(sfb_coord_1, "int32"),
                K.cast(sfb_coord_2, "int32"),
                K.uint64(tma_cache_hint),
            )

        with tma_role:
            tma_state = K.PipelineState(ab_stages, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            count = K.local_scalar("int32")
            speculative = K.local_scalar("uint32")
            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx = scheduler_coords(work)
                if prefetch_distance > 0:
                    prefetch_k = K.local_scalar("int32", init=0)
                    with K.While((prefetch_k < prefetch_distance) & (prefetch_k < k_tiles)):
                        prefetch_inputs(tile_m_idx, tile_n_idx, prefetch_k)
                        K.assign(prefetch_k, prefetch_k + 1)
                K.assign(count, 0)
                K.assign(speculative, K.uint32(1))
                with K.If(count < k_tiles):
                    with K.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase
                        )
                with K.While(count < k_tiles):
                    _wait_plain_if_needed(
                        ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase, speculative
                    )
                    with K.If(leader_cta):
                        with K.Then():
                            with K.If(_elected()):
                                with K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        K.uint32(num_tma_load_bytes),
                                    )
                    with K.If(_elected()):
                        with K.Then():
                            a_coord_m = tile_m_idx * cta_m + cluster_y * a_cluster_piece
                            a_coord_k = count * k_tile
                            a_smem_offset = (
                                a_offset
                                + tma_state.stage * a_stage_bytes
                                + cluster_y * a_piece_bytes
                            )
                            if cta_group == 2:
                                a_barrier = cta2_tma_barrier(tma_state.stage)
                                if cluster_n == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_k, "int32"),
                                        K.cast(a_coord_m, "int32"),
                                        a_barrier,
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    K.ptx[
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + a_smem_offset,
                                        K.address_of(a_map),
                                        K.cast(a_coord_k, "int32"),
                                        K.cast(a_coord_m, "int32"),
                                        a_barrier,
                                        K.cast(a_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )
                            elif cluster_n == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([a_smem_offset]),
                                    K.address_of(a_map),
                                    K.cast(a_coord_k, "int32"),
                                    K.cast(a_coord_m, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem + a_smem_offset,
                                    K.address_of(a_map),
                                    K.cast(a_coord_k, "int32"),
                                    K.cast(a_coord_m, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(a_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                    with K.If(_elected()):
                        with K.Then():
                            b_coord_n = (
                                tile_n_idx * n_tile
                                + cta_v * b_rows
                                + cluster_m_group * b_cluster_piece
                            )
                            b_coord_k = count * k_tile
                            b_smem_offset = (
                                b_offset
                                + tma_state.stage * b_stage_bytes
                                + cluster_m_group * b_piece_bytes
                            )
                            if cta_group == 2:
                                b_barrier = cta2_tma_barrier(tma_state.stage)
                                if cluster_m_groups == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + b_smem_offset,
                                        K.address_of(b_map),
                                        K.cast(b_coord_k, "int32"),
                                        K.cast(b_coord_n, "int32"),
                                        b_barrier,
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    K.ptx[
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + b_smem_offset,
                                        K.address_of(b_map),
                                        K.cast(b_coord_k, "int32"),
                                        K.cast(b_coord_n, "int32"),
                                        b_barrier,
                                        K.cast(b_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )
                            elif cluster_m_groups == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.2d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([b_smem_offset]),
                                    K.address_of(b_map),
                                    K.cast(b_coord_k, "int32"),
                                    K.cast(b_coord_n, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint"
                                ](
                                    cluster_smem + b_smem_offset,
                                    K.address_of(b_map),
                                    K.cast(b_coord_k, "int32"),
                                    K.cast(b_coord_n, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(b_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )

                    with K.If(_elected()):
                        with K.Then():
                            sfa_linear = cluster_y * sfa_piece_values
                            sfa_coord_0 = sfa_linear % 256
                            sfa_coord_1 = count * sf_k_box + sfa_linear // 256
                            sfa_smem_offset = (
                                sfa_offset
                                + tma_state.stage * sfa_stage_bytes
                                + cluster_y * (sfa_stage_bytes // cluster_n)
                            )
                            if cta_group == 2:
                                if cluster_n == 1:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + sfa_smem_offset,
                                        K.address_of(sfa_map),
                                        K.cast(sfa_coord_0, "int32"),
                                        K.cast(sfa_coord_1, "int32"),
                                        K.cast(tile_m_idx * sfa_m_box, "int32"),
                                        cta2_tma_barrier(tma_state.stage),
                                        K.uint64(tma_cache_hint),
                                    )
                                else:
                                    K.ptx[
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                        ".mbarrier::complete_tx::bytes.multicast::cluster"
                                        ".L2::cache_hint.cta_group::2"
                                    ](
                                        cluster_smem + sfa_smem_offset,
                                        K.address_of(sfa_map),
                                        K.cast(sfa_coord_0, "int32"),
                                        K.cast(sfa_coord_1, "int32"),
                                        K.cast(tile_m_idx * sfa_m_box, "int32"),
                                        cta2_tma_barrier(tma_state.stage),
                                        K.cast(a_mcast_mask, "uint16"),
                                        K.uint64(tma_cache_hint),
                                    )
                            elif cluster_n == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([sfa_smem_offset]),
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx * sfa_m_box, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                ](
                                    cluster_smem + sfa_smem_offset,
                                    K.address_of(sfa_map),
                                    K.cast(sfa_coord_0, "int32"),
                                    K.cast(sfa_coord_1, "int32"),
                                    K.cast(tile_m_idx * sfa_m_box, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(a_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )

                    with K.If(_elected()):
                        with K.Then():
                            sfb_linear = cluster_x * sfb_piece_values
                            sfb_coord_0 = sfb_linear % 256
                            sfb_quotient = sfb_linear // 256
                            sfb_coord_1 = count * sf_k_box + sfb_quotient % sf_k_box
                            sfb_tile_group = tile_n_idx * n_tile // 128
                            sfb_coord_2 = sfb_tile_group + sfb_quotient // sf_k_box
                            sfb_smem_offset = (
                                sfb_offset
                                + tma_state.stage * sfb_stage_bytes
                                + cluster_x * (sfb_stage_bytes // cluster_m)
                            )
                            if cta_group == 2:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2"
                                ](
                                    cluster_smem + sfb_smem_offset,
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    cta2_tma_barrier(tma_state.stage),
                                    K.cast(sfb_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                            elif cluster_m_groups == 1:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cta.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint"
                                ](
                                    smem.ptr_to([sfb_smem_offset]),
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.uint64(tma_cache_hint),
                                )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                ](
                                    cluster_smem + sfb_smem_offset,
                                    K.address_of(sfb_map),
                                    K.cast(sfb_coord_0, "int32"),
                                    K.cast(sfb_coord_1, "int32"),
                                    K.cast(sfb_coord_2, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    K.cast(b_mcast_mask, "uint16"),
                                    K.uint64(tma_cache_hint),
                                )
                    if prefetch_distance > 0:
                        with K.If(count < k_tiles - prefetch_distance):
                            with K.Then():
                                prefetch_inputs(tile_m_idx, tile_n_idx, count + prefetch_distance)
                    _advance(tma_state)
                    K.assign(count, count + 1)
                    K.assign(speculative, K.uint32(1))
                    with K.If(count < k_tiles):
                        with K.Then():
                            _try_wait_acquire(
                                speculative,
                                ab_pipe.empty.ptr_to([tma_state.stage]),
                                tma_state.phase,
                            )
                advance_work(work)
            with K.unroll(0, ab_stages) as unused_stage:
                _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                _advance(tma_state)
            del unused_stage

        with mma_role:
            K.ptx.bar.sync(K.uint32(2), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            sfa_descriptor = K.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + sfa_offset)
            )
            sfb_descriptor = K.local_scalar(
                "uint64", init=_descriptor_with_address(sf_desc_base, smem_base + sfb_offset)
            )
            mma_state = K.PipelineState(ab_stages, phase=0)
            acc_state = K.PipelineState(acc_stages, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            count = K.local_scalar("int32")
            speculative = K.local_scalar("uint32")
            accumulate = K.local_scalar("uint32")

            def runtime_descriptor(sfa_addr, sfb_addr):
                desc = K.bitwise_and(K.uint32(instr_desc), K.uint32(0x9FFFFFCF))
                desc = K.bitwise_or(
                    desc, K.bitwise_and(K.shift_right(sfa_addr, K.uint32(1)), K.uint32(0x60000000))
                )
                return K.bitwise_or(
                    desc, K.bitwise_and(K.shift_right(sfb_addr, K.uint32(26)), K.uint32(0x30))
                )

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx = scheduler_coords(work)
                K.assign(count, 0)
                K.assign(speculative, K.uint32(1))
                with K.If((count < k_tiles) & leader_cta):
                    with K.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                        )
                with K.If(leader_cta):
                    with K.Then():
                        _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                K.assign(accumulate, K.uint32(0))
                with K.While(count < k_tiles):
                    with K.If(leader_cta):
                        with K.Then():
                            _wait_plain_if_needed(
                                ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase, speculative
                            )
                            sfb_odd_shift = K.local_scalar(
                                "uint32",
                                init=K.Select(
                                    (tile_n_idx & 1) != 0,
                                    K.uint32(2 if n_tile in (64, 192) else 0),
                                    K.uint32(0),
                                ),
                            )
                            if b_reuse:
                                sf_layout_scale = sf_vec_size // 16
                                sfa_segment_chunks = 2
                                sfb_kblock_chunks = sfb_chunks // 2
                                for kblock in range(2):
                                    for chunk in range(sfa_segment_chunks):
                                        with K.If(_elected()):
                                            with K.Then():
                                                K.ptx[
                                                    f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                                ](
                                                    K.cast(
                                                        tmem_base
                                                        + sfa_tmem_column
                                                        + kblock * (8 // sf_layout_scale)
                                                        + chunk * (16 // sf_layout_scale),
                                                        "uint32",
                                                    ),
                                                    sfa_descriptor
                                                    + K.cast(
                                                        mma_state.stage * (sfa_stage_bytes // 16)
                                                        + kblock * (64 // sf_layout_scale)
                                                        + chunk * (128 // sf_layout_scale),
                                                        "uint64",
                                                    ),
                                                )
                                    for chunk in range(sfb_kblock_chunks):
                                        with K.If(_elected()):
                                            with K.Then():
                                                K.ptx[
                                                    f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                                ](
                                                    K.cast(
                                                        tmem_base
                                                        + sfb_tmem_column
                                                        + kblock * sfb_kblock_chunks * 4
                                                        + chunk * 4,
                                                        "uint32",
                                                    ),
                                                    sfb_descriptor
                                                    + K.cast(
                                                        mma_state.stage * (sfb_stage_bytes // 16)
                                                        + kblock * (64 // sf_layout_scale)
                                                        + chunk * 32,
                                                        "uint64",
                                                    ),
                                                )
                                    for chunk in range(sfa_segment_chunks):
                                        with K.If(_elected()):
                                            with K.Then():
                                                K.ptx[
                                                    f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                                ](
                                                    K.cast(
                                                        tmem_base
                                                        + sfa_tmem_column
                                                        + (4 if sf_vec_size == 16 else 0)
                                                        + kblock * (8 // sf_layout_scale)
                                                        + chunk * (16 // sf_layout_scale),
                                                        "uint32",
                                                    ),
                                                    sfa_descriptor
                                                    + K.cast(
                                                        mma_state.stage * (sfa_stage_bytes // 16)
                                                        + kblock * (64 // sf_layout_scale)
                                                        + (32 if sf_vec_size == 16 else 0)
                                                        + chunk * (128 // sf_layout_scale),
                                                        "uint64",
                                                    ),
                                                )
                                    sfb_addr = K.cast(
                                        tmem_base
                                        + sfb_tmem_column
                                        + kblock * sfb_kblock_chunks * 4,
                                        "uint32",
                                    )
                                    sfa_keep = K.cast(
                                        tmem_base
                                        + sfa_tmem_column
                                        + kblock * (8 // sf_layout_scale),
                                        "uint32",
                                    )
                                    sfa_reuse = K.cast(sfa_keep + 16 // sf_layout_scale, "uint32")
                                    keep_desc = runtime_descriptor(sfa_keep, sfb_addr)
                                    reuse_desc = runtime_descriptor(sfa_reuse, sfb_addr)
                                    with K.If(_elected()):
                                        with K.Then():
                                            K.ptx[
                                                f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4"
                                                f".block_scale.block{sf_vec_size}"
                                                ".collector::a::discard.collector::b::fill"
                                            ](
                                                K.cast(
                                                    tmem_base + acc_state.stage * n_tile * 2,
                                                    "uint32",
                                                ),
                                                a_descriptor
                                                + K.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + K.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                keep_desc,
                                                sfa_keep,
                                                sfb_addr,
                                                K.ptx.pred(K.cast(accumulate, "bool")),
                                            )
                                    with K.If(_elected()):
                                        with K.Then():
                                            K.ptx[
                                                f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4"
                                                f".block_scale.block{sf_vec_size}"
                                                ".collector::a::discard.collector::b::lastuse"
                                            ](
                                                K.cast(
                                                    tmem_base
                                                    + acc_state.stage * n_tile * 2
                                                    + n_tile,
                                                    "uint32",
                                                ),
                                                a_descriptor
                                                + K.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + 1024
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + K.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                reuse_desc,
                                                sfa_reuse,
                                                sfb_addr,
                                                K.ptx.pred(K.cast(accumulate, "bool")),
                                            )
                                    K.assign(accumulate, K.uint32(1))
                            else:
                                mma_name = (
                                    f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4"
                                    f".block_scale.block{sf_vec_size}.collector::a::discard"
                                )
                                for sf_chunk in range(sfa_chunks):
                                    with K.If(_elected()):
                                        with K.Then():
                                            K.ptx[
                                                f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                            ](
                                                K.cast(
                                                    tmem_base + sfa_tmem_column + sf_chunk * 4,
                                                    "uint32",
                                                ),
                                                sfa_descriptor
                                                + K.cast(
                                                    mma_state.stage * (sfa_stage_bytes // 16)
                                                    + sf_chunk * 32,
                                                    "uint64",
                                                ),
                                            )
                                for sf_chunk in range(sfb_chunks):
                                    sfb_shared_chunk = (
                                        sf_chunk % sfb_n_box
                                    ) * sf_k_box + sf_chunk // sfb_n_box
                                    with K.If(_elected()):
                                        with K.Then():
                                            K.ptx[
                                                f"tcgen05.cp.cta_group::{cta_group}.32x128b.warpx4"
                                            ](
                                                K.cast(
                                                    tmem_base + sfb_tmem_column + sf_chunk * 4,
                                                    "uint32",
                                                ),
                                                sfb_descriptor
                                                + K.cast(
                                                    mma_state.stage * (sfb_stage_bytes // 16)
                                                    + sfb_shared_chunk * 32,
                                                    "uint64",
                                                ),
                                            )
                                for kblock in range(2):
                                    sfa_addr = K.cast(
                                        tmem_base
                                        + sfa_tmem_column
                                        + kblock * (sfa_chunks // 2) * 4,
                                        "uint32",
                                    )
                                    sfb_addr = K.cast(
                                        tmem_base
                                        + sfb_tmem_column
                                        + sfb_odd_shift
                                        + kblock * (sfb_chunks // 2) * 4,
                                        "uint32",
                                    )
                                    mma_desc = runtime_descriptor(sfa_addr, sfb_addr)
                                    with K.If(_elected()):
                                        with K.Then():
                                            K.ptx[mma_name](
                                                K.cast(
                                                    tmem_base + acc_state.stage * n_tile, "uint32"
                                                ),
                                                a_descriptor
                                                + K.cast(
                                                    mma_state.stage * (a_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                b_descriptor
                                                + K.cast(
                                                    mma_state.stage * (b_stage_bytes // 16)
                                                    + kblock * 4,
                                                    "uint64",
                                                ),
                                                mma_desc,
                                                sfa_addr,
                                                sfb_addr,
                                                K.ptx.pred(K.cast(accumulate, "bool")),
                                            )
                                    K.assign(accumulate, K.uint32(1))
                            with K.If(_elected()):
                                with K.Then():
                                    if cluster_size == 1:
                                        K.ptx[
                                            f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                            "arrive::one.shared::cluster.b64"
                                        ](ab_pipe.empty.ptr_to([mma_state.stage]))
                                    else:
                                        K.ptx[
                                            f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                            "arrive::one.shared::cluster.multicast::cluster.b64"
                                        ](
                                            ab_pipe.empty.ptr_to([mma_state.stage]),
                                            K.cast(ab_consumer_mask, "uint16"),
                                        )
                    _advance(mma_state)
                    K.assign(count, count + 1)
                    K.assign(speculative, K.uint32(1))
                    with K.If((count < k_tiles) & leader_cta):
                        with K.Then():
                            _try_wait_acquire(
                                speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                            )
                with K.If(leader_cta):
                    with K.Then():
                        with K.If(_elected()):
                            with K.Then():
                                if cluster_size == 1:
                                    K.ptx[
                                        f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                        "arrive::one.shared::cluster.b64"
                                    ](acc_pipe.full.ptr_to([acc_state.stage]))
                                else:
                                    K.ptx[
                                        f"tcgen05.commit.cta_group::{cta_group}.mbarrier::"
                                        "arrive::one.shared::cluster.multicast::cluster.b64"
                                    ](
                                        acc_pipe.full.ptr_to([acc_state.stage]),
                                        K.cast(acc_producer_mask, "uint16"),
                                    )
                _advance(acc_state)
                advance_work(work)
            with K.If(leader_cta):
                with K.Then():
                    for _ in range(acc_stages - 1):
                        _advance(acc_state)
                    _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            with K.If(warp == 0):
                with K.Then():
                    K.ptx[
                        f"tcgen05.alloc.exclusive.cta_group::{cta_group}.sync.aligned."
                        "shared::cta.b32"
                    ](tmem_slot_addr, K.uint32(tmem_columns))
            K.ptx.bar.sync(K.uint32(2), K.uint32(160))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot_addr)
            acc_state = K.PipelineState(acc_stages, phase=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            executed_subtiles = K.local_scalar("int32", init=0)
            values = K.alloc_local((32,), "float32")
            words = K.alloc_local((16,), "uint32")
            offsets = K.alloc_local((4,), "int32")

            def scale_and_pack():
                if not alpha_is_one:
                    for index in range(0, 32, 2):
                        packed = K.local_scalar("uint64")
                        scale_pair = K.local_scalar("uint64")
                        K.ptx.mov.b64(packed, values[index], values[index + 1])
                        K.ptx.mov.b64(scale_pair, alpha_value, alpha_value)
                        K.ptx.mul.f32x2(packed, scale_pair, packed)
                        K.ptx.mov.b64(values[index], values[index + 1], packed)
                for index in range(16):
                    if out_dtype == "float16":
                        K.ptx.cvt.rn.f16x2.f32(
                            words[index], values[index * 2 + 1], values[index * 2]
                        )
                    else:
                        K.ptx.cvt.rn.bf16x2.f32(
                            words[index], values[index * 2 + 1], values[index * 2]
                        )

            def row_major_output_offset(stage, vector):
                row_bytes = epi_n * 2
                unswizzled = (
                    smem_base
                    + c_offset
                    + stage * c_stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes
                    + vector * 16
                )
                swizzled = K.bitwise_xor(
                    unswizzled, K.bitwise_and(K.shift_right(unswizzled, K.uint32(3)), K.uint32(48))
                )
                return K.cast(swizzled - smem_base, "int32")

            with K.While(work < cluster_work):
                tile_m_idx, tile_n_idx = scheduler_coords(work)
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                subtile = K.local_scalar("int32", init=0)
                with K.While(subtile < epilogue_subtiles):
                    m_subtile = subtile % (cta_m // 128)
                    n_subtile = subtile // (cta_m // 128)
                    tmem_address = K.local_scalar(
                        "uint32",
                        init=(
                            tmem_base
                            + (warp << 21)
                            + acc_state.stage * n_tile * (2 if b_reuse else 1)
                            + m_subtile * n_tile
                            + n_subtile * epi_n
                        ),
                    )
                    if swap:
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[values[index] for index in range(16)], tmem_address
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[values[index] for index in range(16, 32)],
                            tmem_address + K.uint32(1 << 20),
                        )
                    else:
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                            *[values[index] for index in range(32)], tmem_address
                        )
                    scale_and_pack()
                    output_stage = executed_subtiles % c_stages
                    if not swap:
                        for vector in range(4):
                            K.assign(offsets[vector], row_major_output_offset(output_stage, vector))
                        for vector in range(4):
                            K.ptx.st.shared.v4.b32(
                                smem.ptr_to([offsets[vector]]),
                                words[vector * 4],
                                words[vector * 4 + 1],
                                words[vector * 4 + 2],
                                words[vector * 4 + 3],
                            )
                    else:
                        thread = warp * 32 + lane
                        temporary = K.bitwise_or(
                            K.bitwise_and(thread << 5, K.int32(6144)),
                            K.bitwise_and(thread, K.int32(40)),
                        )
                        raw_address = K.bitwise_or(
                            K.bitwise_or(K.bitwise_and(thread << 7, K.int32(896)), temporary << 1),
                            K.bitwise_and(thread << 6, K.int32(1024)),
                        )
                        first_unswizzled = (
                            smem_base + c_offset + output_stage * c_stage_bytes + raw_address
                        )
                        first_swizzled = K.bitwise_xor(
                            first_unswizzled,
                            K.bitwise_and(
                                K.shift_right(first_unswizzled, K.uint32(3)), K.uint32(112)
                            ),
                        )
                        second_unswizzled = first_unswizzled + 32
                        second_swizzled = K.bitwise_xor(
                            second_unswizzled,
                            K.bitwise_and(
                                K.shift_right(second_unswizzled, K.uint32(3)), K.uint32(112)
                            ),
                        )
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([K.cast(first_swizzled - smem_base, "int32")]),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([K.cast(first_swizzled - smem_base + 2048, "int32")]),
                            words[4],
                            words[5],
                            words[6],
                            words[7],
                        )
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([K.cast(second_swizzled - smem_base, "int32")]),
                            words[8],
                            words[9],
                            words[10],
                            words[11],
                        )
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem.ptr_to([K.cast(second_swizzled - smem_base + 2048, "int32")]),
                            words[12],
                            words[13],
                            words[14],
                            words[15],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    with K.If(warp == 0):
                        with K.Then():
                            if swap:
                                for row_copy in range(2):
                                    K.ptx[
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile"
                                        ".bulk_group.L2::cache_hint"
                                    ](
                                        K.address_of(c_map),
                                        K.cast(
                                            tile_m_idx * cta_m + m_subtile * 128 + row_copy * 64,
                                            "int32",
                                        ),
                                        K.cast(tile_n_idx * n_tile + n_subtile * epi_n, "int32"),
                                        smem.ptr_to(
                                            [
                                                c_offset
                                                + output_stage * c_stage_bytes
                                                + row_copy * 4096
                                            ]
                                        ),
                                        K.uint64(tma_cache_hint),
                                    )
                            else:
                                K.ptx[
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile"
                                    ".bulk_group.L2::cache_hint"
                                ](
                                    K.address_of(c_map),
                                    K.cast(tile_n_idx * n_tile + n_subtile * epi_n, "int32"),
                                    K.cast(tile_m_idx * cta_m + m_subtile * 128, "int32"),
                                    smem.ptr_to([c_offset + output_stage * c_stage_bytes]),
                                    K.uint64(tma_cache_hint),
                                )
                            K.ptx.cp.async_.bulk.commit_group()
                            K.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                    K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                    K.assign(executed_subtiles, executed_subtiles + 1)
                    K.assign(subtile, subtile + 1)
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                with K.If(_elected()):
                    with K.Then():
                        if cta_group == 2:
                            K.ptx.mbarrier.arrive.shared__cluster.b64(
                                acc_empty_leader.ptr_to([acc_state.stage]), K.uint32(1)
                            )
                        else:
                            K.ptx.mbarrier.arrive.shared.b64(
                                acc_pipe.empty.ptr_to([acc_state.stage]), K.uint32(1)
                            )
                _advance(acc_state)
                advance_work(work)

            K.ptx.cp.async_.bulk.wait_group.read(0)
            with K.If(warp == 0):
                with K.Then():
                    K.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
            with K.If(warp == 0):
                with K.Then():
                    if cta_group == 2:
                        remote_dealloc = K.local_scalar("uint32")
                        K.ptx.mapa.shared__cluster.u32(
                            remote_dealloc,
                            K.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])),
                            K.cast(cluster_rank ^ 1, "uint32"),
                        )
                        K.ptx.mbarrier.arrive.shared__cluster.b64(remote_dealloc, K.uint32(1))
                        _wait_plain(tmem_dealloc.ptr_to([0]), K.uint32(0))
                    K.ptx[f"tcgen05.dealloc.exclusive.cta_group::{cta_group}.sync.aligned.b32"](
                        tmem_base, K.uint32(tmem_columns)
                    )

        required_block_size.__exit__(None, None, None)

    kernel.__annotations__ = {
        "a": K.gptr[K.u8, (kernel_m * K_dim // 2,)],
        "b": K.gptr[K.u8, (kernel_n * K_dim // 2,)],
        "sfa": K.gptr[K.u8, (_ceil_div(kernel_m, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "sfb": K.gptr[K.u8, (_ceil_div(kernel_n, 128) * _ceil_div(K_dim, 4 * sf_vec_size) * 512,)],
        "c": K.gptr[K.u8, (kernel_m * kernel_n * 2,)],
        "alpha_ptr": K.gptr[K.f32, (1,)],
    }
    return K.kernel(
        warps=6,
        arch="sm_107a",
        min_blocks_per_sm=1,
        grid=[cluster_m, cluster_n, num_clusters],
        host_prelude=host_prelude,
    )(kernel)


def get_kernel(M, N, K, sf_mode, out_dtype, alpha, tactic):
    """Return the concrete batchless SM107 kernel specialization."""
    return _make_kernel(M, N, K, sf_mode, out_dtype, alpha, tactic).func


_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_PAYLOAD_VALUES = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
_SF_VALUES = (1.0, 2.0)
_OUTPUT_GUARD = 256


def _torch_dtype(torch, dtype):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e8m0fnu": torch.float8_e8m0fnu,
    }[dtype]


def _generator(torch, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _packed_operand(torch, rows, K_dim, seed):
    table = torch.tensor(_PAYLOAD_VALUES, dtype=torch.float32, device="cuda")
    indices = torch.randint(
        0,
        len(_PAYLOAD_VALUES),
        (rows, K_dim),
        device="cuda",
        generator=_generator(torch, seed),
        dtype=torch.int64,
    )
    values = table[indices]
    magnitudes = torch.tensor(_FP4_VALUES[:8], dtype=torch.float32, device="cuda")
    matches = values.abs().unsqueeze(-1) == magnitudes
    codes = matches.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    codes = codes | ((values < 0).to(torch.uint8) << 3)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return {"raw": packed.reshape(-1), "matrix": packed, "values": values}


def _scale_factors(torch, rows, K_dim, sf_dtype, sf_vec_size, seed):
    rest_m = _ceil_div(rows, 128)
    sf_k = _ceil_div(K_dim, sf_vec_size)
    rest_k = _ceil_div(sf_k, 4)
    values = torch.tensor(_SF_VALUES, dtype=torch.float32, device="cuda")
    indices = torch.randint(
        0,
        len(_SF_VALUES),
        (1, rest_m, rest_k, 32, 4, 4),
        device="cuda",
        generator=_generator(torch, seed),
        dtype=torch.int64,
    )
    storage = values[indices].to(_torch_dtype(torch, sf_dtype)).contiguous()
    row = torch.arange(rows, device="cuda", dtype=torch.int64)
    group = torch.arange(sf_k, device="cuda", dtype=torch.int64)
    logical = storage[
        0,
        (row // 128)[:, None],
        (group // 4)[None, :],
        (row % 32)[:, None],
        ((row // 32) % 4)[:, None],
        (group % 4)[None, :],
    ].float()
    return {"storage": storage, "raw": storage.view(torch.uint8).reshape(-1), "values": logical}


def _guarded_output(torch, M, N, dtype, fill_byte):
    payload_bytes = M * N * 2
    allocation = torch.full(
        (payload_bytes + 2 * _OUTPUT_GUARD,), fill_byte, dtype=torch.uint8, device="cuda"
    )
    raw = allocation[_OUTPUT_GUARD : _OUTPUT_GUARD + payload_bytes]
    output = raw.view(dtype).reshape(M, N)
    output.fill_(float("nan"))
    return {"allocation": allocation, "raw": raw, "output": output, "fill": fill_byte}


def prepare_data(M, N, K, sf_mode, out_dtype, alpha, tactic):
    """Allocate deterministic exact-valued inputs shared by TIRx, source, and oracle."""
    _validate_problem(M, N, K, sf_mode, out_dtype, alpha, tactic)
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        raise RuntimeError("dense_blockscaled_gemm_sm107 requires an SM107 CUDA device")
    sf_dtype, sf_vec_size = _SF_MODES[sf_mode]
    a = _packed_operand(torch, M, K, 101)
    b = _packed_operand(torch, N, K, 211)
    sfa = _scale_factors(torch, M, K, sf_dtype, sf_vec_size, 307)
    sfb = _scale_factors(torch, N, K, sf_dtype, sf_vec_size, 401)
    output_dtype = _torch_dtype(torch, out_dtype)
    tirx = _guarded_output(torch, M, N, output_dtype, 0xA5)
    source = _guarded_output(torch, M, N, output_dtype, 0x5A)
    alpha_tensor = torch.tensor([float(alpha)], dtype=torch.float32, device="cuda")
    return {
        "a": a,
        "b": b,
        "sfa": sfa,
        "sfb": sfb,
        "tirx": tirx,
        "source": source,
        "alpha": alpha_tensor,
        "config": {
            "M": M,
            "N": N,
            "K": K,
            "sf_mode": sf_mode,
            "out_dtype": out_dtype,
            "alpha": alpha,
            "tactic": tactic,
        },
    }


@cache
def _compile_executable(M, N, K_dim, sf_mode, out_dtype, alpha, tactic):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(get_kernel(M, N, K_dim, sf_mode, out_dtype, alpha, tactic))


def _tirx_launch(executable, data):
    swap = TACTICS[data["config"]["tactic"]][3]
    a_raw = data["b"]["raw"] if swap else data["a"]["raw"]
    b_raw = data["a"]["raw"] if swap else data["b"]["raw"]
    sfa_raw = data["sfb"]["raw"] if swap else data["sfa"]["raw"]
    sfb_raw = data["sfa"]["raw"] if swap else data["sfb"]["raw"]

    def launch():
        executable(a_raw, b_raw, sfa_raw, sfb_raw, data["tirx"]["raw"], data["alpha"])

    launch._keep_alive = data
    return launch


def _register_package(name, path):
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module


@cache
def _install_cutlass_parent():
    parent_path = _CUTLASS_ROOT / _CUTLASS_PARENT_RELATIVE
    if not parent_path.is_file():
        raise RuntimeError(f"missing pinned CUTLASS parent: {parent_path}")
    actual = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if actual != CUTLASS_PARENT_SHA256:
        raise RuntimeError(f"CUTLASS parent hash mismatch: sha256={actual}")
    package_paths = (
        ("nvidia_cutlass_dsl.examples", _CUTLASS_ROOT / "examples/python"),
        ("nvidia_cutlass_dsl.examples.CuTeDSL", _CUTLASS_ROOT / "examples/python/CuTeDSL"),
        (
            "nvidia_cutlass_dsl.examples.CuTeDSL.cute",
            _CUTLASS_ROOT / "examples/python/CuTeDSL/cute",
        ),
        (
            "nvidia_cutlass_dsl.examples.CuTeDSL.cute.blackwell",
            _CUTLASS_ROOT / "examples/python/CuTeDSL/cute/blackwell",
        ),
        (
            "nvidia_cutlass_dsl.examples.CuTeDSL.cute.blackwell.kernel",
            _CUTLASS_ROOT / "examples/python/CuTeDSL/cute/blackwell/kernel",
        ),
        (
            "nvidia_cutlass_dsl.examples.CuTeDSL.cute.blackwell.kernel.blockscaled_gemm",
            _CUTLASS_ROOT / "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm",
        ),
    )
    for name, path in package_paths:
        _register_package(name, path)
    spec = importlib.util.spec_from_file_location(_CUTLASS_PARENT_MODULE, parent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned CUTLASS parent: {parent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CUTLASS_PARENT_MODULE] = module
    spec.loader.exec_module(module)


@cache
def _source_runner(out_dtype, sf_mode):
    source_files = {
        "flashinfer/gemm/kernels/dense_blockscaled_gemm_sm107.py": SOURCE_SHA256,
        "flashinfer/gemm/kernels/epilogue_utils.py": SOURCE_DEPENDENCY_SHA256["epilogue_utils.py"],
        "flashinfer/gemm/gemm_base.py": SOURCE_DEPENDENCY_SHA256["gemm_base.py"],
        "flashinfer/gemm/kernels/utils.py": SOURCE_DEPENDENCY_SHA256["kernels/utils.py"],
    }
    for relative, expected in source_files.items():
        source = _SOURCE_ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"frozen FlashInfer source is unavailable: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen source hash mismatch: {source} sha256={actual}")
    _install_cutlass_parent()
    loaded = sys.modules.get("flashinfer")
    loaded_file = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
    if loaded_file is not None and _SOURCE_ROOT not in loaded_file.parents:
        for name in tuple(sys.modules):
            if name == "flashinfer" or name.startswith("flashinfer."):
                del sys.modules[name]
    source_root = str(_SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import torch

    gemm_base = importlib.import_module("flashinfer.gemm.gemm_base")
    return gemm_base._cute_dsl_gemm_fp4_runner(
        10, 7, False, _torch_dtype(torch, out_dtype), sf_mode == "nvfp4"
    )


def _source_launch(data):
    config = data["config"]
    runner = _source_runner(config["out_dtype"], config["sf_mode"])
    (tile_m, tile_n), (inst_m, inst_n), cluster, swap, prefetch = TACTICS[config["tactic"]]
    tactic_tuple = (
        (tile_m, tile_n),
        cluster,
        swap,
        False,
        "sm107",
        (inst_m, inst_n, 128, 256, prefetch),
    )
    inputs = [
        data["a"]["matrix"],
        data["b"]["matrix"].T,
        data["sfa"]["raw"],
        data["sfb"]["raw"],
        data["alpha"],
        None,
        data["source"]["output"],
        None,
        None,
        None,
    ]

    def launch():
        runner.forward(inputs, tactic=tactic_tuple)

    launch._keep_alive = (data, inputs, runner)
    return launch


def _assert_guards(data):
    for name in ("tirx", "source"):
        guarded = data[name]
        allocation = guarded["allocation"]
        expected = guarded["fill"]
        prefix = allocation[:_OUTPUT_GUARD]
        suffix = allocation[-_OUTPUT_GUARD:]
        if not bool(((prefix == expected).all() & (suffix == expected).all()).item()):
            raise AssertionError(f"{name} output guard was modified")


def _oracle(data):
    import torch

    config = data["config"]
    sf_vec_size = _SF_MODES[config["sf_mode"]][1]
    a_scales = data["sfa"]["values"].repeat_interleave(sf_vec_size, dim=1)[:, : config["K"]]
    b_scales = data["sfb"]["values"].repeat_interleave(sf_vec_size, dim=1)[:, : config["K"]]
    effective_a = data["a"]["values"].double() * a_scales.double()
    effective_b = data["b"]["values"].double() * b_scales.double()
    rounded_alpha = (
        torch.tensor(
            float(config["alpha"]), dtype=_torch_dtype(torch, config["out_dtype"]), device="cuda"
        )
        .float()
        .double()
    )
    return (effective_a @ effective_b.T * rounded_alpha).to(
        _torch_dtype(torch, config["out_dtype"])
    )


def _validate_outputs(data, *, with_source, with_oracle):
    import torch

    actual = data["tirx"]["output"]
    if not bool(torch.isfinite(actual.float()).all().item()):
        raise AssertionError("TIRx output contains a non-finite value")
    _assert_guards(data)
    result = {"bitwise_source": None, "bitwise_oracle": None, "max_abs_diff": 0.0}
    if with_source:
        expected = data["source"]["output"]
        if not bool(torch.isfinite(expected.float()).all().item()):
            raise AssertionError("source output contains a non-finite value")
        if not torch.equal(actual, expected):
            delta = (actual.float() - expected.float()).abs()
            worst = int(torch.argmax(delta).item())
            raise AssertionError(
                "bitwise mismatch against frozen FlashInfer source: "
                f"differing={int((actual != expected).sum().item())}, "
                f"max_abs_diff={float(delta.max().item())}, flat_index={worst}"
            )
        result["bitwise_source"] = True
    if with_oracle:
        oracle = _oracle(data)
        if not torch.equal(actual, oracle):
            delta = (actual.float() - oracle.float()).abs()
            worst = int(torch.argmax(delta).item())
            raise AssertionError(
                "zero-tolerance mismatch against independent FP64 oracle: "
                f"differing={int((actual != oracle).sum().item())}, "
                f"max_abs_diff={float(delta.max().item())}, flat_index={worst}"
            )
        result["bitwise_oracle"] = True
    return result


def run_test(M, N, K, sf_mode, out_dtype, alpha, tactic):
    import torch

    data = prepare_data(M, N, K, sf_mode, out_dtype, alpha, tactic)
    executable = _compile_executable(M, N, K, sf_mode, out_dtype, alpha, tactic)
    tirx_launch = _tirx_launch(executable, data)
    source_launch = _source_launch(data)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    first = data["tirx"]["output"].clone()
    data["tirx"]["output"].fill_(float("nan"))
    tirx_launch()
    torch.cuda.synchronize()
    if not torch.equal(first, data["tirx"]["output"]):
        raise AssertionError("TIRx output is not deterministic across identical launches")
    return _validate_outputs(data, with_source=True, with_oracle=True)


def prepare_bench(M, N, K, sf_mode, out_dtype, alpha, tactic):
    from tirx_kernels.runner import prepared_gpu_benchmark

    _validate_problem(M, N, K, sf_mode, out_dtype, alpha, tactic)
    state = {
        "config": {
            "M": M,
            "N": N,
            "K": K,
            "sf_mode": sf_mode,
            "out_dtype": out_dtype,
            "alpha": alpha,
            "tactic": tactic,
        },
        "executable": _compile_executable(M, N, K, sf_mode, out_dtype, alpha, tactic),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    data = prepare_data(**config)
    tirx_launch = _tirx_launch(prepared["executable"], data)
    tirx_launch()
    torch.cuda.synchronize()
    references = None
    with_source = external_references_enabled()
    if with_source:
        source_launch = _source_launch(data)
        source_launch()
        torch.cuda.synchronize()
        references = {"flashinfer": lambda: source_launch}
    _validate_outputs(data, with_source=with_source, with_oracle=False)
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    M,
    N,
    K,
    sf_mode,
    out_dtype,
    alpha,
    tactic,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
):
    return prepare_bench(M, N, K, sf_mode, out_dtype, alpha, tactic).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "TACTICS",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
