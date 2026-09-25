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

"""SM107 FP8 batched GEMM transcribed from FlashInfer's CuTeDSL kernel.

Upstream sources:
- flashinfer/gemm/kernels/bmm_fp8_rubin.py
- flashinfer/gemm/kernels/bmm_fp8_blackwell.py
- flashinfer/gemm/kernels/bmm_fp8_wrapper.py
- flashinfer/gemm/kernels/epilogue_utils.py
"""

# txl.kernel traces concrete annotation objects; postponed annotations would turn them into strings.
import importlib
import sys
from functools import cache
from pathlib import Path
from typing import Any

import tirx_kernels.tirx_lite as txl
from tirx_kernels.flashinfer._shared.source_checkout import flashinfer_source_root

KERNEL_META = {
    "name": "bmm_fp8_rubin",
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

_AB_DTYPES = ("float8_e4m3fn", "float8_e5m2")
_C_DTYPES = ("bfloat16", "float16", "float32")
_DTYPE_LABEL = {
    "float8_e4m3fn": "e4m3",
    "float8_e5m2": "e5m2",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}

_TRY_WAIT_TICKS = 10_000_000
_SMEM_CAPACITY = 335_872
# Stand-in for the source's ``HardwareInfo().get_max_active_clusters(cluster_size)``
# query on the 200-SM sm_107a part, so the persistent grid stays a static
# specialization fact: 100 two-CTA clusters and 40 four-CTA clusters.
_MAX_ACTIVE_CLUSTERS = {2: 100, 4: 40}

# (mma_tiler, mma_instruction, cluster_mn, raster)
TACTICS = (
    ((256, 256, 128), (256, 256, 64), (2, 1), "m"),
    ((256, 128, 128), (256, 128, 64), (2, 1), "m"),
    ((256, 256, 64), (256, 256, 32), (2, 1), "m"),
    ((256, 256, 64), (256, 256, 64), (2, 1), "m"),
    ((256, 256, 128), (256, 256, 64), (2, 2), "m"),
    ((256, 128, 128), (256, 128, 64), (4, 1), "m"),
    ((256, 256, 128), (256, 256, 64), (2, 1), "n"),
    ((256, 128, 128), (256, 128, 64), (2, 1), "n"),
)

_CORRECTNESS_SHAPES = (
    (2, 272, 1040, 1008),
    (2, 272, 1040, 1008),
    (2, 272, 1040, 1008),
    (2, 272, 1040, 1008),
    (2, 272, 528, 1008),
    (2, 528, 1040, 1008),
    (2, 272, 1040, 1008),
    (2, 272, 1040, 1008),
)

_BENCH_SHAPES = (
    (1, 256, 10304, 2688),
    (4, 256, 4096, 2688),
    (1, 512, 4096, 2720),
    (2, 512, 4096, 2688),
    (1, 1024, 4096, 3072),
    (1, 2048, 4096, 3072),
    (1, 4096, 1024, 3072),
    (2, 4096, 1024, 3072),
)


def _config(
    prefix: str, tactic: int, shape: tuple[int, int, int, int], ab_dtype: str, c_dtype: str
):
    B, M, N, K_dim = shape
    return {
        "label": (
            f"{prefix}_t{tactic}_{_DTYPE_LABEL[ab_dtype]}_{_DTYPE_LABEL[c_dtype]}_"
            f"b{B}_m{M}_n{N}_k{K_dim}"
        ),
        "B": B,
        "M": M,
        "N": N,
        "K": K_dim,
        "ab_dtype": ab_dtype,
        "c_dtype": c_dtype,
        "tactic": tactic,
    }


CONFIGS = [
    _config("guard", tactic, _CORRECTNESS_SHAPES[tactic], ab_dtype, c_dtype)
    for tactic in range(len(TACTICS))
    for ab_dtype in _AB_DTYPES
    for c_dtype in _C_DTYPES
]
CONFIGS.append(_config("source_anchor", 0, _BENCH_SHAPES[0], "float8_e4m3fn", "bfloat16"))

BENCH_CONFIGS = [
    _config("bench", tactic, _BENCH_SHAPES[tactic], ab_dtype, c_dtype)
    for tactic in range(len(TACTICS))
    for ab_dtype in _AB_DTYPES
    for c_dtype in _C_DTYPES
]


def _validate_problem(
    B: int, M: int, N: int, K: int, ab_dtype: str, c_dtype: str, tactic: int
) -> None:
    if not 0 <= tactic < len(TACTICS):
        raise ValueError(f"tactic must be in [0, {len(TACTICS) - 1}], got {tactic}")
    if ab_dtype not in _AB_DTYPES:
        raise ValueError(f"unsupported FP8 input dtype {ab_dtype!r}")
    if c_dtype not in _C_DTYPES:
        raise ValueError(f"unsupported output dtype {c_dtype!r}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if M % 16 or N % 16 or K % 16:
        raise ValueError(f"M, N, and K must be multiples of 16, got {(M, N, K)}")
    mma_tiler, _mma_instruction, cluster, _raster = TACTICS[tactic]
    cta_m = mma_tiler[0] // 2
    if M < cta_m * cluster[0] or N < mma_tiler[1] * cluster[1]:
        raise ValueError(
            f"shape {(M, N)} cannot fill tactic {tactic} cluster {cluster} "
            f"with CTA tile {(cta_m, mma_tiler[1])}"
        )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _descriptor_base(ldo: int, sdo: int, swizzle: int) -> int:
    arrangement_type = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = 0
    value |= (ldo & 0x3FFF) << 16
    value |= (sdo & 0x3FFF) << 32
    value |= 1 << 46
    value |= (arrangement_type & 0x7) << 61
    return value & 0xFFFFFFFFFFFFFFFF


def _descriptor_with_address(base: int, shared_address):
    base_value = txl.bitwise_or(
        txl.shift_left(txl.uint64(base >> 32), txl.uint64(32)), txl.uint64(base & 0xFFFFFFFF)
    )
    address_field = txl.cast(
        txl.bitwise_and(txl.shift_right(shared_address, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64"
    )
    return txl.bitwise_or(base_value, address_field)


def _advance(state) -> None:
    state.advance()


def _try_wait_acquire(dst, barrier, phase) -> None:
    txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
        dst, barrier, txl.cast(phase, "uint32")
    )


def _wait_plain(barrier, phase) -> None:
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, txl.cast(phase, "uint32"), txl.uint32(_TRY_WAIT_TICKS)
        )


def _wait_plain_if_needed(barrier, phase, speculative_ready) -> None:
    with txl.If(speculative_ready == txl.uint32(0)):
        with txl.Then():
            _wait_plain(barrier, phase)


def _elected():
    elected_lane = txl.local_scalar("uint32")
    elected_pred = txl.local_scalar("uint32")
    txl.ptx.elect_sync(elected_lane, elected_pred, txl.uint32(0xFFFFFFFF))
    return elected_pred == txl.uint32(1)


def _instruction_descriptor(n_tile: int, instruction_k: int, ab_dtype: str) -> int:
    descriptors = {
        (256, 64, "float8_e4m3fn"): 0x30410010,
        (256, 64, "float8_e5m2"): 0x30410490,
        (128, 64, "float8_e4m3fn"): 0x30210010,
        (128, 64, "float8_e5m2"): 0x30210490,
        (256, 32, "float8_e4m3fn"): 0x10410010,
        (256, 32, "float8_e5m2"): 0x10410490,
    }
    return descriptors[(n_tile, instruction_k, ab_dtype)]


@cache
def _make_bmm_kernel(B: int, M: int, N: int, K_dim: int, ab_dtype: str, c_dtype: str, tactic: int):
    _validate_problem(B, M, N, K_dim, ab_dtype, c_dtype, tactic)
    mma_tiler, mma_instruction, (cluster_m, cluster_n), raster = TACTICS[tactic]
    tile_m, n_tile, k_tile = mma_tiler
    instruction_m, instruction_n, instruction_k = mma_instruction
    if tile_m != 256 or instruction_m != 256 or instruction_n != n_tile:
        raise AssertionError("the frozen source surface contains only 256-row two-CTA MMA")

    cta_group = 2
    cta_m = 128
    b_rows = n_tile // cta_group
    cluster_size = cluster_m * cluster_n
    cluster_m_groups = cluster_m // cta_group
    k_phases = k_tile // instruction_k
    k_tiles = _ceil_div(K_dim, k_tile)
    m_tiles = _ceil_div(M, cta_m)
    n_tiles = _ceil_div(N, n_tile)
    cluster_m_tiles = _ceil_div(m_tiles, cluster_m)
    cluster_n_tiles = _ceil_div(n_tiles, cluster_n)
    cluster_work = cluster_m_tiles * cluster_n_tiles * B
    num_clusters = min(cluster_work, _MAX_ACTIVE_CLUSTERS[cluster_size])

    c_bits = {"bfloat16": 16, "float16": 16, "float32": 32}[c_dtype]
    acc_stages = 2
    if tactic in (2, 3):
        c_stages = 2
        ab_stages = 18 if c_dtype == "float32" else 19
    else:
        c_stages = 2 if c_dtype == "float32" else 4
        ab_stages = 9 if n_tile == 256 else 12

    a_stage_bytes = cta_m * k_tile
    b_stage_bytes = b_rows * k_tile
    ab_stage_bytes = a_stage_bytes + b_stage_bytes
    c_stage_bytes = cta_m * 32 * c_bits // 8

    ab_full_offset = 0
    ab_empty_offset = ab_full_offset + ab_stages * 8
    acc_full_offset = ab_empty_offset + ab_stages * 8
    acc_empty_offset = acc_full_offset + acc_stages * 8
    tmem_dealloc_offset = acc_empty_offset + acc_stages * 8
    tmem_ptr_offset = tmem_dealloc_offset + 8
    a_offset = 384 if ab_stages > 12 else 256
    b_offset = a_offset + ab_stages * a_stage_bytes
    c_offset = b_offset + ab_stages * b_stage_bytes
    shared_bytes = c_offset + c_stages * c_stage_bytes
    if shared_bytes not in (327_936, 328_064):
        raise AssertionError(f"source shared-memory layout changed: {shared_bytes}")
    if shared_bytes > _SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {shared_bytes} exceeds {_SMEM_CAPACITY}")

    ab_empty_arrivals = cluster_n + cluster_m_groups - 1
    acc_empty_arrivals = 8
    tmem_columns = acc_stages * n_tile
    if k_tile == 128:
        a_swizzle, a_ldo, a_sdo = 3, 1, 64
    else:
        a_swizzle, a_ldo, a_sdo = 2, 1, 32
    if b_rows == 128:
        b_swizzle, b_chunk, b_ldo, b_sdo = 3, 128, 0, 64
    else:
        b_swizzle, b_chunk, b_ldo, b_sdo = 2, 64, 0, 32
    a_cluster_piece = cta_m // cluster_n
    b_cluster_piece = k_tile // cluster_m_groups
    a_piece_bytes = a_stage_bytes // cluster_n
    b_piece_bytes = b_stage_bytes // cluster_m_groups
    num_tma_load_bytes = ab_stage_bytes * cta_group
    a_desc_base = _descriptor_base(ldo=a_ldo, sdo=a_sdo, swizzle=a_swizzle)
    b_desc_base = _descriptor_base(ldo=b_ldo, sdo=b_sdo, swizzle=b_swizzle)
    instruction_descriptor = _instruction_descriptor(n_tile, instruction_k, ab_dtype)
    epilogue_subtiles = n_tile // 32
    tma_cache_hint = 0

    def host_prelude(params):
        a = params["a"]
        b = params["b"]
        c = params["c"]
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)
        c_map = txl.stack_alloca("tensormap", 1)

        def encode(descriptor, dtype, rank, data, *fields):
            txl.call_packed("runtime.cuTensorMapEncodeTiled", descriptor, dtype, rank, data, *fields)

        a_fields = (K_dim, M, B, K_dim, M * K_dim, k_tile, a_cluster_piece, 1)
        b_fields = (N, K_dim, B, N, N * K_dim, b_chunk, b_cluster_piece, 1)
        tensor_tail = (1, 1, 1, 0)
        encode(a_map, ab_dtype, 3, a.data, *a_fields, *tensor_tail, a_swizzle, 2, 0)
        encode(b_map, ab_dtype, 3, b.data, *b_fields, *tensor_tail, b_swizzle, 2, 0)

        element_bytes = c_bits // 8
        row_bytes = 32 * element_bytes
        c_swizzle = 3 if row_bytes == 128 else 2
        c_fields = (N, M, B, N * element_bytes, M * N * element_bytes, 32, cta_m, 1)
        encode(c_map, c_dtype, 3, c.data, *c_fields, 1, 1, 1, 0, c_swizzle, 2, 0)
        return a_map, b_map, c_map

    def kernel(a, b, c, output_scale, *, host):
        del a, b, c
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        a_map, b_map, c_map = host
        scale = txl.local_scalar("float32")
        txl.ptx.ld.global_.f32(scale, output_scale.ptr_to([0]))

        _block_x, _block_y, cluster_work_id = txl.cta_id()
        cluster_x_scope, cluster_y_scope = txl.cta_id_in_cluster(
            [cluster_m, cluster_n], preferred=[cluster_m, cluster_n]
        )
        del _block_x, _block_y, cluster_x_scope, cluster_y_scope
        cluster_rank = txl.local_scalar("int32", init=txl.cuda.mov_sreg(32, "cluster_ctarank"))
        cluster_x = txl.local_scalar("int32", init=cluster_rank & (cluster_m - 1))
        cluster_y = txl.local_scalar("int32", init=cluster_rank >> (cluster_m.bit_length() - 1))
        cta_v = cluster_x & 1
        leader_cta = cta_v == 0
        cluster_m_group = cluster_x >> 1
        pair_leader_x = cluster_m_group << 1
        leader_rank = pair_leader_x + cluster_m * cluster_y
        warp = txl.warp_id()
        lane = txl.lane_id()

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4])
        tma_role = roles.role("tma", warps=[5])

        smem = txl.alloc_buffer((shared_bytes,), txl.u8, scope="shared.dyn", align=1024)
        protocol_pool = txl.smem_pool(base=smem)
        ab_pipe = txl.Pipeline(
            protocol_pool,
            ab_stages,
            full="tma",
            empty="tcgen05",
            init_empty=ab_empty_arrivals,
            leader=txl.bool(False),
        )
        acc_pipe = txl.Pipeline(
            protocol_pool,
            acc_stages,
            full="tcgen05",
            empty="mbar",
            init_empty=acc_empty_arrivals,
            leader=txl.bool(False),
        )
        if protocol_pool.bytes != tmem_dealloc_offset:
            raise AssertionError("protocol storage offsets changed")
        tmem_dealloc = protocol_pool.alloc((1,), txl.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), txl.u32, align=4)
        if protocol_pool.bytes != tmem_ptr_offset + 4:
            raise AssertionError("protocol storage header changed")

        with tma_role:
            txl.ptx.prefetch.tensormap(txl.address_of(a_map))
            txl.ptx.prefetch.tensormap(txl.address_of(b_map))
            txl.ptx.prefetch.tensormap(txl.address_of(c_map))

        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, ab_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, ab_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), txl.uint32(ab_empty_arrivals)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, acc_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, acc_stages) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), txl.uint32(acc_empty_arrivals)
                            )
        with tma_role:
            with txl.If(_elected()):
                with txl.Then():
                    txl.ptx.mbarrier.init.shared.b64(tmem_dealloc.ptr_to([0]), txl.uint32(32))
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.barrier.cluster.arrive.relaxed()

        smem_base = txl.local_scalar("uint32")
        txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_u64 = txl.local_scalar("uint64")
        txl.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = txl.local_scalar("uint32", init=txl.cast(cluster_smem_u64, "uint32"))
        a_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(a_desc_base, smem_base + a_offset)
        )
        b_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + b_offset)
        )

        a_mcast_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for peer_n in range(cluster_n):
            txl.assign(
                a_mcast_mask,
                txl.bitwise_or(
                    a_mcast_mask, txl.uint32(1) << txl.cast(cluster_x + cluster_m * peer_n, "uint32")
                ),
            )
        b_mcast_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for peer_group in range(cluster_m_groups):
            peer_x = cta_v + 2 * peer_group
            txl.assign(
                b_mcast_mask,
                txl.bitwise_or(
                    b_mcast_mask, txl.uint32(1) << txl.cast(peer_x + cluster_m * cluster_y, "uint32")
                ),
            )
        ab_consumer_mask = txl.local_scalar("uint32", init=txl.uint32(0))
        for pair_v in range(2):
            for peer_n in range(cluster_n):
                txl.assign(
                    ab_consumer_mask,
                    txl.bitwise_or(
                        ab_consumer_mask,
                        txl.uint32(1)
                        << txl.cast(pair_leader_x + pair_v + cluster_m * peer_n, "uint32"),
                    ),
                )
            for peer_group in range(cluster_m_groups):
                peer_x = pair_v + 2 * peer_group
                txl.assign(
                    ab_consumer_mask,
                    txl.bitwise_or(
                        ab_consumer_mask,
                        txl.uint32(1) << txl.cast(peer_x + cluster_m * cluster_y, "uint32"),
                    ),
                )
        acc_producer_mask = txl.local_scalar(
            "uint32", init=txl.uint32(3) << txl.cast(leader_rank, "uint32")
        )
        ab_full_leader = ab_pipe.full.remote_view(leader_rank)
        acc_empty_leader = acc_pipe.empty.remote_view(leader_rank)
        txl.ptx.barrier.cluster.wait()

        def scheduler_coords(work):
            if raster == "m":
                cluster_m_idx = work % cluster_m_tiles
                quotient = work // cluster_m_tiles
                cluster_n_idx = quotient % cluster_n_tiles
                batch_idx = quotient // cluster_n_tiles
            else:
                cluster_n_idx = work % cluster_n_tiles
                quotient = work // cluster_n_tiles
                cluster_m_idx = quotient % cluster_m_tiles
                batch_idx = quotient // cluster_m_tiles
            tile_m_idx = cluster_m_idx * cluster_m + cluster_x
            tile_n_idx = cluster_n_idx * cluster_n + cluster_y
            return tile_m_idx, tile_n_idx, batch_idx

        def advance_work(work) -> None:
            txl.assign(work, work + num_clusters)

        with tma_role:
            tma_state = txl.PipelineState(ab_stages, phase=1)
            work = txl.local_scalar("int32", init=cluster_work_id)
            count = txl.local_scalar("int32")
            speculative = txl.local_scalar("uint32")
            with txl.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                txl.assign(count, 0)
                txl.assign(speculative, txl.uint32(1))
                with txl.If(count < k_tiles):
                    with txl.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase
                        )
                with txl.While(count < k_tiles):
                    _wait_plain_if_needed(
                        ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase, speculative
                    )
                    with txl.If(leader_cta):
                        with txl.Then():
                            with txl.If(_elected()):
                                with txl.Then():
                                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        txl.uint32(num_tma_load_bytes),
                                    )

                    with txl.If(_elected()):
                        with txl.Then():
                            a_coord_m = tile_m_idx * cta_m + cluster_y * a_cluster_piece
                            a_coord_k = count * k_tile
                            a_smem_offset = (
                                a_offset
                                + tma_state.stage * a_stage_bytes
                                + cluster_y * a_piece_bytes
                            )
                            if cluster_n == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                ](
                                    cluster_smem + a_smem_offset,
                                    txl.address_of(a_map),
                                    txl.cast(a_coord_k, "int32"),
                                    txl.cast(a_coord_m, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    txl.uint64(tma_cache_hint),
                                )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2"
                                ](
                                    cluster_smem + a_smem_offset,
                                    txl.address_of(a_map),
                                    txl.cast(a_coord_k, "int32"),
                                    txl.cast(a_coord_m, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    txl.cast(a_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )

                    with txl.If(_elected()):
                        with txl.Then():
                            b_coord_n = tile_n_idx * n_tile + cta_v * b_rows
                            b_coord_k = count * k_tile + cluster_m_group * b_cluster_piece
                            b_smem_offset = (
                                b_offset
                                + tma_state.stage * b_stage_bytes
                                + cluster_m_group * b_piece_bytes
                            )
                            if cluster_m_groups == 1:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
                                ](
                                    cluster_smem + b_smem_offset,
                                    txl.address_of(b_map),
                                    txl.cast(b_coord_n, "int32"),
                                    txl.cast(b_coord_k, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    txl.uint64(tma_cache_hint),
                                )
                            else:
                                txl.ptx[
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster"
                                    ".L2::cache_hint.cta_group::2"
                                ](
                                    cluster_smem + b_smem_offset,
                                    txl.address_of(b_map),
                                    txl.cast(b_coord_n, "int32"),
                                    txl.cast(b_coord_k, "int32"),
                                    txl.cast(batch_idx, "int32"),
                                    ab_full_leader.ptr_to([tma_state.stage]),
                                    txl.cast(b_mcast_mask, "uint16"),
                                    txl.uint64(tma_cache_hint),
                                )
                    _advance(tma_state)
                    txl.assign(count, count + 1)
                    txl.assign(speculative, txl.uint32(1))
                    with txl.If(count < k_tiles):
                        with txl.Then():
                            _try_wait_acquire(
                                speculative,
                                ab_pipe.empty.ptr_to([tma_state.stage]),
                                tma_state.phase,
                            )
                advance_work(work)
            with txl.unroll(0, ab_stages) as unused_stage:
                _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                _advance(tma_state)
            del unused_stage

        with mma_role:
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            mma_state = txl.PipelineState(ab_stages, phase=0)
            acc_state = txl.PipelineState(acc_stages, phase=1)
            work = txl.local_scalar("int32", init=cluster_work_id)
            count = txl.local_scalar("int32")
            speculative = txl.local_scalar("uint32")
            accumulate = txl.local_scalar("uint32")
            with txl.While(work < cluster_work):
                txl.assign(count, 0)
                txl.assign(speculative, txl.uint32(1))
                with txl.If((count < k_tiles) & leader_cta):
                    with txl.Then():
                        _try_wait_acquire(
                            speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                        )
                with txl.If(leader_cta):
                    with txl.Then():
                        _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                txl.assign(accumulate, txl.uint32(0))
                with txl.While(count < k_tiles):
                    with txl.If(leader_cta):
                        with txl.Then():
                            _wait_plain_if_needed(
                                ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase, speculative
                            )
                            for kphase in range(k_phases):
                                a_kphase = kphase * (instruction_k // 16)
                                b_kphase = kphase * (b_chunk * instruction_k // 16)
                                with txl.If(_elected()):
                                    with txl.Then():
                                        txl.ptx[
                                            "tcgen05.mma.cta_group::2.kind::f8f6f4"
                                            ".collector::a::discard"
                                        ](
                                            txl.cast(tmem_base + acc_state.stage * n_tile, "uint32"),
                                            a_descriptor
                                            + txl.cast(
                                                mma_state.stage * (a_stage_bytes // 16) + a_kphase,
                                                "uint64",
                                            ),
                                            b_descriptor
                                            + txl.cast(
                                                mma_state.stage * (b_stage_bytes // 16) + b_kphase,
                                                "uint64",
                                            ),
                                            txl.uint32(instruction_descriptor),
                                            *[txl.uint32(0) for _ in range(8)],
                                            txl.ptx.pred(txl.cast(accumulate, "bool")),
                                        )
                                txl.assign(accumulate, txl.uint32(1))
                            with txl.If(_elected()):
                                with txl.Then():
                                    txl.ptx[
                                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                                        ".shared::cluster.multicast::cluster.b64"
                                    ](
                                        ab_pipe.empty.ptr_to([mma_state.stage]),
                                        txl.cast(ab_consumer_mask, "uint16"),
                                    )
                    _advance(mma_state)
                    txl.assign(count, count + 1)
                    txl.assign(speculative, txl.uint32(1))
                    with txl.If((count < k_tiles) & leader_cta):
                        with txl.Then():
                            _try_wait_acquire(
                                speculative, ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase
                            )
                with txl.If(leader_cta):
                    with txl.Then():
                        with txl.If(_elected()):
                            with txl.Then():
                                txl.ptx[
                                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                                    ".shared::cluster.multicast::cluster.b64"
                                ](
                                    acc_pipe.full.ptr_to([acc_state.stage]),
                                    txl.cast(acc_producer_mask, "uint16"),
                                )
                _advance(acc_state)
                advance_work(work)
            with txl.If(leader_cta):
                with txl.Then():
                    _advance(acc_state)
                    _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            with txl.If(warp == 0):
                with txl.Then():
                    txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                        tmem_slot.ptr_to([0]), txl.uint32(tmem_columns)
                    )
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            acc_state = txl.PipelineState(acc_stages, phase=0)
            work = txl.local_scalar("int32", init=cluster_work_id)
            executed_tiles = txl.local_scalar("int32", init=0)
            values = txl.alloc_local((32,), "float32")
            if c_bits == 32:
                words = txl.alloc_local((32,), "uint32")
                offsets = txl.alloc_local((8,), "int32")
            else:
                words = txl.alloc_local((16,), "uint32")
                offsets = txl.alloc_local((4,), "int32")

            def scale_values() -> None:
                for index in range(0, 32, 2):
                    packed = txl.local_scalar("uint64")
                    scale_pair = txl.local_scalar("uint64")
                    txl.ptx.mov.b64(packed, values[index], values[index + 1])
                    txl.ptx.mov.b64(scale_pair, scale, scale)
                    txl.ptx.mul.f32x2(packed, scale_pair, packed)
                    txl.ptx.mov.b64(values[index], values[index + 1], packed)

            def pack_values() -> None:
                if c_dtype == "float32":
                    for index in range(32):
                        txl.assign(words[index], txl.reinterpret("uint32", values[index]))
                elif c_dtype == "float16":
                    for index in range(16):
                        txl.ptx.cvt.rn.f16x2.f32(
                            words[index], values[index * 2 + 1], values[index * 2]
                        )
                else:
                    for index in range(16):
                        txl.ptx.cvt.rn.bf16x2.f32(
                            words[index], values[index * 2 + 1], values[index * 2]
                        )

            def output_offset(stage, vector):
                row_bytes = 32 * c_bits // 8
                unswizzled = (
                    smem_base
                    + c_offset
                    + stage * c_stage_bytes
                    + warp * (32 * row_bytes)
                    + lane * row_bytes
                    + vector * 16
                )
                swizzle_mask = 112 if row_bytes == 128 else 48
                swizzled = txl.bitwise_xor(
                    unswizzled,
                    txl.bitwise_and(txl.shift_right(unswizzled, txl.uint32(3)), txl.uint32(swizzle_mask)),
                )
                return txl.cast(swizzled - smem_base, "int32")

            with txl.While(work < cluster_work):
                tile_m_idx, tile_n_idx, batch_idx = scheduler_coords(work)
                _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                subtile = txl.local_scalar("int32", init=0)
                with txl.While(subtile < epilogue_subtiles):
                    tmem_address = txl.local_scalar(
                        "uint32",
                        init=(tmem_base + (warp << 21) + acc_state.stage * n_tile + subtile * 32),
                    )
                    txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[values[index] for index in range(32)], tmem_address
                    )
                    scale_values()
                    pack_values()
                    output_stage = (executed_tiles * epilogue_subtiles + subtile) % c_stages
                    for vector in range((32 * c_bits // 32) // 4):
                        txl.assign(offsets[vector], output_offset(output_stage, vector))
                    for vector in range((32 * c_bits // 32) // 4):
                        txl.ptx.st.shared.v4.b32(
                            smem.ptr_to([offsets[vector]]),
                            words[vector * 4],
                            words[vector * 4 + 1],
                            words[vector * 4 + 2],
                            words[vector * 4 + 3],
                        )
                    txl.ptx.fence.proxy.async_.shared__cta()
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                    with txl.If(txl.And(warp == 0, lane == 0)):
                        with txl.Then():
                            txl.ptx[
                                "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
                                ".L2::cache_hint"
                            ](
                                txl.address_of(c_map),
                                txl.cast(tile_n_idx * n_tile + subtile * 32, "int32"),
                                txl.cast(tile_m_idx * cta_m, "int32"),
                                txl.cast(batch_idx, "int32"),
                                smem.ptr_to([c_offset + output_stage * c_stage_bytes]),
                                txl.uint64(tma_cache_hint),
                            )
                            txl.ptx.cp.async_.bulk.commit_group()
                            txl.ptx.cp.async_.bulk.wait_group.read(c_stages - 1)
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                    txl.assign(subtile, subtile + 1)
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(128))
                with txl.If(_elected()):
                    with txl.Then():
                        txl.ptx.mbarrier.arrive.shared__cluster.b64(
                            acc_empty_leader.ptr_to([acc_state.stage]), txl.uint32(1)
                        )
                _advance(acc_state)
                txl.assign(executed_tiles, executed_tiles + 1)
                advance_work(work)

            txl.ptx.cp.async_.bulk.wait_group.read(0)
            with txl.If(warp == 0):
                with txl.Then():
                    txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
                    remote_dealloc = txl.local_scalar("uint32")
                    txl.ptx.mapa.shared__cluster.u32(
                        remote_dealloc,
                        txl.cuda.cvta_generic_to_shared(tmem_dealloc.ptr_to([0])),
                        txl.cast(cluster_rank ^ 1, "uint32"),
                    )
                    txl.ptx.mbarrier.arrive.shared__cluster.b64(remote_dealloc, txl.uint32(1))
                    _wait_plain(tmem_dealloc.ptr_to([0]), txl.uint32(0))
                    txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](
                        tmem_base, txl.uint32(tmem_columns)
                    )

        required_block_size.__exit__(None, None, None)

    kernel.__annotations__ = {
        "a": txl.gptr[txl.u8, (B * M * K_dim,)],
        "b": txl.gptr[txl.u8, (B * K_dim * N,)],
        "c": txl.gptr[txl.u8, (B * M * N * c_bits // 8,)],
        "output_scale": txl.gptr[txl.f32, (1,)],
    }
    return txl.kernel(
        warps=6,
        arch="sm_107a",
        min_blocks_per_sm=1,
        grid=[cluster_m, cluster_n, num_clusters],
        host_prelude=host_prelude,
    )(kernel)


def get_kernel(B: int, M: int, N: int, K: int, ab_dtype: str, c_dtype: str, tactic: int):
    _validate_problem(B, M, N, K, ab_dtype, c_dtype, tactic)
    return _make_bmm_kernel(B, M, N, K, ab_dtype, c_dtype, tactic).func


def _torch_dtype(torch, dtype: str):
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _pattern(torch, shape: tuple[int, ...], phase: int, device):
    count = 1
    for extent in shape:
        count *= extent
    values = (torch.arange(count, device=device, dtype=torch.int64) + phase) % 5 - 2
    return values.reshape(shape).to(torch.float32)


def prepare_data(
    B: int, M: int, N: int, K: int, ab_dtype: str, c_dtype: str, tactic: int
) -> dict[str, Any]:
    _validate_problem(B, M, N, K, ab_dtype, c_dtype, tactic)
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        raise RuntimeError("bmm_fp8_rubin requires an SM107 CUDA device")
    device = torch.device("cuda")
    a = _pattern(torch, (B, M, K), 0, device).to(_torch_dtype(torch, ab_dtype)).contiguous()
    b = _pattern(torch, (B, K, N), 2, device).to(_torch_dtype(torch, ab_dtype)).contiguous()
    a_scale = torch.tensor(0.75, dtype=torch.float32, device=device)
    b_scale = torch.tensor(1.25, dtype=torch.float32, device=device)
    output_scale = (a_scale * b_scale).reshape(1)
    output_dtype = _torch_dtype(torch, c_dtype)
    tirx_c = torch.full((B, M, N), float("nan"), dtype=output_dtype, device=device)
    source_c = torch.full((B, M, N), float("nan"), dtype=output_dtype, device=device)
    return {
        "a": a,
        "b": b,
        "a_raw": a.view(torch.uint8).reshape(-1),
        "b_raw": b.view(torch.uint8).reshape(-1),
        "tirx_c": tirx_c,
        "tirx_c_raw": tirx_c.view(torch.uint8).reshape(-1),
        "source_c": source_c,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "output_scale": output_scale,
    }


@cache
def _compile_executable(
    B: int, M: int, N: int, K_dim: int, ab_dtype: str, c_dtype: str, tactic: int
):
    from tirx_kernels.bench.runner import compile_kernel

    return compile_kernel(get_kernel(B, M, N, K_dim, ab_dtype, c_dtype, tactic))


def _tirx_launch(executable, data):
    def launch():
        executable(data["a_raw"], data["b_raw"], data["tirx_c_raw"], data["output_scale"])

    launch._keep_alive = data
    return launch


@cache
def _source_bmm_op():
    root = flashinfer_source_root()
    source = root / "flashinfer/gemm/kernels/bmm_fp8_rubin.py"
    if not source.is_file():
        raise RuntimeError(f"FlashInfer source is unavailable: {source}")
    loaded = sys.modules.get("flashinfer")
    loaded_file = Path(getattr(loaded, "__file__", "")).resolve() if loaded is not None else None
    if loaded_file is not None and root not in loaded_file.parents:
        for name in tuple(sys.modules):
            if name == "flashinfer" or name.startswith("flashinfer."):
                del sys.modules[name]
    source_root = str(root)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    module = importlib.import_module("flashinfer.gemm.kernels.bmm_fp8_wrapper")
    return module.bmm_fp8_cute_dsl


def _source_launch(data, c_dtype: str, tactic: int):
    bmm_fp8_cute_dsl = _source_bmm_op()
    output_dtype = _torch_dtype(__import__("torch"), c_dtype)

    def launch():
        bmm_fp8_cute_dsl(
            data["a"],
            data["b"],
            data["a_scale"],
            data["b_scale"],
            output_dtype,
            out=data["source_c"],
            config_index=tactic,
            arch="sm107",
        )

    launch._keep_alive = data
    return launch


def _validate_outputs(data, *, with_source: bool) -> dict[str, Any]:
    import torch

    actual = data["tirx_c"]
    if not bool(torch.isfinite(actual.float()).all().item()):
        raise AssertionError("TIRx output contains a non-finite value")
    if not with_source:
        return {"bitwise": None, "max_abs_diff": None}
    expected = data["source_c"]
    if not bool(torch.isfinite(expected.float()).all().item()):
        raise AssertionError("source output contains a non-finite value")
    if torch.equal(actual, expected):
        return {"bitwise": True, "max_abs_diff": 0.0}
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = (actual_f32 - expected_f32).abs()
    worst = int(torch.argmax(difference).item())
    differing = int((actual != expected).sum().item())
    raise AssertionError(
        "bmm_fp8_rubin bitwise mismatch against frozen FlashInfer source: "
        f"differing={differing}, max_abs_diff={float(difference.max().item())}, "
        f"actual={float(actual_f32.reshape(-1)[worst].item())}, "
        f"expected={float(expected_f32.reshape(-1)[worst].item())}, flat_index={worst}"
    )


def run_test(
    B: int, M: int, N: int, K: int, ab_dtype: str, c_dtype: str, tactic: int
) -> dict[str, Any]:
    import torch

    data = prepare_data(B, M, N, K, ab_dtype, c_dtype, tactic)
    executable = _compile_executable(B, M, N, K, ab_dtype, c_dtype, tactic)
    tirx_launch = _tirx_launch(executable, data)
    source_launch = _source_launch(data, c_dtype, tactic)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    return _validate_outputs(data, with_source=True)


def prepare_bench(B: int, M: int, N: int, K: int, ab_dtype: str, c_dtype: str, tactic: int):
    from tirx_kernels.bench.runner import prepared_gpu_benchmark

    _validate_problem(B, M, N, K, ab_dtype, c_dtype, tactic)
    state = {
        "config": {
            "B": B,
            "M": M,
            "N": N,
            "K": K,
            "ab_dtype": ab_dtype,
            "c_dtype": c_dtype,
            "tactic": tactic,
        },
        "executable": _compile_executable(B, M, N, K, ab_dtype, c_dtype, tactic),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    import torch

    from tirx_kernels.bench.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    data = prepare_data(**config)
    tirx_launch = _tirx_launch(prepared["executable"], data)
    tirx_launch()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source_launch = _source_launch(data, config["c_dtype"], config["tactic"])
        source_launch()
        torch.cuda.synchronize()
        references = {"flashinfer": lambda: source_launch}
    _validate_outputs(data, with_source=with_source)
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
    B: int,
    M: int,
    N: int,
    K: int,
    ab_dtype: str,
    c_dtype: str,
    tactic: int,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
):
    return prepare_bench(B, M, N, K, ab_dtype, c_dtype, tactic).run_gpu(
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
