# This file is a TIRx port of code from DeepEP
# (https://github.com/deepseek-ai/DeepEP @ 01dc3aa), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""DeepEP V2 elastic combine (single-domain NVLink path) ported to TIRx.

Source: /home/bohanhou/DeepEP
  - deep_ep/include/deep_ep/impls/combine.cuh (`combine_impl`, direct path)
  - deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh
    (`combine_reduce_epilogue_impl`)
  - deep_ep/include/deep_ep/impls/combine_utils.cuh

Frozen sketch (source of the implementation plan):
  `.agents/sketch/deepep/combine.md`

Fixed specialization: bf16 tokens, non-expanded layout,
allow_multiple_reduction=True, no bias, num_scaleout_ranks==1,
is_scaleup_nvlink=True.
"""

from __future__ import annotations

from typing import Any

from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from .utils._buffer import get_theoretical_num_sms

KERNEL_META = {"name": "deepep_combine", "category": "deepep", "compute_capability": 10}

# Correctness matrix. Every config runs the same source specialization
# (bf16, non-expanded, allow_multiple_reduction, no bias) on `world_size` ranks.
CONFIGS = [
    {
        "label": "t128_h7168_e256_k6",
        "world_size": 8,
        "num_tokens": 128,
        "hidden": 7168,
        "num_experts": 256,
        "num_topk": 6,
        "expert_alignment": 1,
        "masked_ratio": 0.0,
    },
    {
        "label": "t4096_h7168_e256_k6",
        "world_size": 8,
        "num_tokens": 4096,
        "hidden": 7168,
        "num_experts": 256,
        "num_topk": 6,
        "expert_alignment": 1,
        "masked_ratio": 0.0,
    },
    {
        "label": "t1024_h7168_e256_k6_masked",
        "world_size": 8,
        "num_tokens": 1024,
        "hidden": 7168,
        "num_experts": 256,
        "num_topk": 6,
        "expert_alignment": 1,
        "masked_ratio": 0.3,
    },
    {
        "label": "t1024_h7168_e256_k6_align128",
        "world_size": 8,
        "num_tokens": 1024,
        "hidden": 7168,
        "num_experts": 256,
        "num_topk": 6,
        "expert_alignment": 128,
        "masked_ratio": 0.0,
    },
]

# Benchmark-only matrix; the bench suite selects from these labels.
BENCH_CONFIGS = [
    {
        "label": "t4096_h7168_e256_k6",
        "world_size": 8,
        "num_tokens": 4096,
        "hidden": 7168,
        "num_experts": 256,
        "num_topk": 6,
        "expert_alignment": 1,
        "masked_ratio": 0.0,
    }
]

# ---------------------------------------------------------------------------
# Specialization constants (frozen sketch: "Static specialization boundary")
# ---------------------------------------------------------------------------

NUM_RANKS = 8
NUM_EXPERTS = 256
NUM_TOPK = 6
HIDDEN_BYTES = 7168 * 2  # bf16
NUM_WARPS = 16
NUM_THREADS = NUM_WARPS * 32

SMEM_TOTAL = 232448  # max opt-in dynamic smem per block on SM100
COMBINE_TOKEN_BYTES = 14400  # align(14336,32) + align(0,32) + align(6*4+6*4,32)
COMBINE_TOKEN_SMEM = 14432  # + align(8,32) mbarrier area (kernel 1 only)
OUTPUT_TOKEN_BYTES = 14336  # flat hidden (kernel 2 staging / combined_x)
NUM_TOKENS_IN_LAYOUT = NUM_TOPK  # kUseRankLayout=false (8 ranks > 6 topk)

# layout::WorkspaceLayout byte offsets (common/layout.cuh) + port scratch
WS_BARRIER_COUNTER = 0
WS_BARRIER_SIGNAL = 8
WS_PORT_SCRATCH = 12_820_528  # software grid-barrier slots (dispatch substitution 2)

TIMEOUT_CYCLES = 200_000_000_000  # 100 s at ~2 GHz (comm.cuh: kNumOneSecCycles)

_BULK_G2S_CHAIN = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
_BULK_S2G_CHAIN = "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"
_GEZ_VEC_LOAD = "ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32"
_EVICT_FIRST = 0x12F0000000000000
_EVICT_NORMAL = 0x1000000000000000


def _gptr(base_u64, byte_off):
    return T.reinterpret("handle", base_u64 + T.cast(byte_off, "uint64"))


def _peer_u64(table, dst):
    return T.cast(table[dst], "uint64")


def _ld_acquire_gpu_u64(dst, addr):
    return T.ptx.ld.acquire.gpu.global_.u64(dst, addr)


def _shfl_idx(dst, src, src_lane):
    return T.ptx.shfl_sync.idx.b32(
        dst, src, T.cast(src_lane, "uint32"), T.uint32(31), T.uint32(0xFFFFFFFF)
    )


def _launch_tags(cluster: int, pdl: bool) -> list[str]:
    tags = ["blockIdx.x"]
    if cluster > 1:
        tags.append("clusterCtaIdx.x")
    tags.append("threadIdx.x")
    if pdl:
        tags.append("tirx.use_programtic_dependent_launch")
    tags.append("tirx.use_dyn_shared_memory")
    return tags


# ---------------------------------------------------------------------------
# Kernel 1: combine_impl
# ---------------------------------------------------------------------------


def _build_combine_kernel(num_sms: int, num_max_tokens_per_rank: int, num_ranks: int) -> Any:
    """`combine_impl` (frozen sketch kernel 1)."""

    NUM_RANKS_ = num_ranks
    cluster = 2 - num_sms % 2

    @T.prim_func
    def deepep_combine(
        x_ptr: T.handle,
        topk_weights_ptr: T.handle,
        src_metadata_ptr: T.handle,
        psum_rank_ptr: T.handle,
        peer_ws_ptrs: T.handle,
        peer_buf_ptrs: T.handle,
        workspace_addr: T.int64,
        buffer_addr: T.int64,
        num_reduced_tokens: T.int32,
        rank_idx: T.int32,
    ):
        x = T.match_buffer(x_ptr, (NUM_RANKS_ * num_max_tokens_per_rank * HIDDEN_BYTES,), "uint8")
        topk_weights = T.match_buffer(
            topk_weights_ptr, (NUM_RANKS_ * num_max_tokens_per_rank * NUM_TOPK,), "float32"
        )
        src_metadata = T.match_buffer(
            src_metadata_ptr, (NUM_RANKS_ * num_max_tokens_per_rank * (2 + NUM_TOPK),), "int32"
        )
        psum_rank = T.match_buffer(psum_rank_ptr, (NUM_RANKS_,), "int32")
        peer_ws = T.match_buffer(peer_ws_ptrs, (NUM_RANKS_,), "int64")
        peer_buf = T.match_buffer(peer_buf_ptrs, (NUM_RANKS_,), "int64")

        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        smem = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": SMEM_TOTAL})

        sm_idx = T.cta_id([num_sms])
        if cluster > 1:
            cta_in_cluster = T.cta_id_in_cluster([cluster])
        thread_idx = T.thread_id([NUM_THREADS])
        lane = T.lane_id([32])

        ws_u64 = T.cast(workspace_addr, "uint64")
        buf_u64 = T.cast(buffer_addr, "uint64")

        # Rotated warp index (combine.cuh:39)
        warp_u32 = T.alloc_local([1], "uint32")
        T.evaluate(
            T.ptx.shfl_sync.idx.b32(
                warp_u32[0], thread_idx // 32, T.uint32(0), T.uint32(31), T.uint32(0xFFFFFFFF)
            )
        )
        warp = (T.cast(warp_u32[0], "int32") + rank_idx) % NUM_WARPS
        global_warp_idx = warp * num_sms + sm_idx

        # --- NVLink barrier (comm.cuh:88-129), SM 0 only --------------------
        @T.inline
        def nvlink_barrier(tag):
            if sm_idx == 0:
                counter_ptr = _gptr(ws_u64, WS_BARRIER_COUNTER)
                # Plain low-word read of the counter (comm.cuh:98; reviewer r2)
                status32 = T.alloc_local([1], "uint32")
                T.evaluate(T.ptx["ld.global.b32"](status32[0], counter_ptr))
                status = T.cast(T.bitwise_and(status32[0], T.uint32(3)), "int32")
                bphase = T.bitwise_and(status, 1)
                bsign = status // 2
                if thread_idx < NUM_RANKS_:
                    delta = T.Select(bsign == 0, T.int32(1), T.int32(-1))
                    T.evaluate(
                        T.ptx.red.release.sys.global_.add.s32(
                            _gptr(_peer_u64(peer_ws, thread_idx), WS_BARRIER_SIGNAL + bphase * 4),
                            delta,
                        )
                    )
                # comm.cuh:107 __syncthreads (SM 0's CTA only)
                T.ptx.bar.sync(T.uint32(0), T.uint32(NUM_THREADS))
                if thread_idx == 0:
                    old = T.alloc_local([1], "uint64")
                    T.evaluate(T.ptx.atom.global_.add.u64(old[0], counter_ptr, T.uint64(1)))
                    target = T.Select(bsign == 0, T.int32(NUM_RANKS_), T.int32(0))
                    sig = T.alloc_local([1], "int32")
                    sig_ptr = _gptr(ws_u64, WS_BARRIER_SIGNAL + bphase * 4)
                    T.evaluate(T.ptx["ld.acquire.sys.L1::no_allocate.global.s32"](sig[0], sig_ptr))
                    start_clock = T.cuda.clock64()
                    while sig[0] != target:
                        if T.cuda.clock64() - start_clock >= T.uint64(TIMEOUT_CYCLES):
                            T.cuda.printf(
                                "DeepEP NVLink barrier timeout, tag: %d, nvl: %d, "
                                "signal: %d, phase: %d, target: %d\n",
                                T.int32(tag),
                                rank_idx,
                                sig[0],
                                bphase,
                                target,
                            )
                            T.cuda.trap_when_assert_failed(False)
                        T.evaluate(
                            T.ptx["ld.acquire.sys.L1::no_allocate.global.s32"](sig[0], sig_ptr)
                        )

        # --- Software grid barrier (dispatch substitution 2) -----------------
        @T.inline
        def grid_barrier(site):
            counter_ptr = _gptr(ws_u64, WS_PORT_SCRATCH + site * 8)
            if thread_idx == 0:
                c0 = T.alloc_local([1], "uint64")
                T.evaluate(_ld_acquire_gpu_u64(c0[0], counter_ptr))
                target = (c0[0] // T.uint64(num_sms) + T.uint64(1)) * T.uint64(num_sms)
                T.evaluate(T.ptx.red.release.gpu.global_.add.u64(counter_ptr, T.uint64(1)))
                now = T.alloc_local([1], "uint64")
                T.evaluate(_ld_acquire_gpu_u64(now[0], counter_ptr))
                while now[0] < target:
                    T.evaluate(_ld_acquire_gpu_u64(now[0], counter_ptr))
            T.ptx.bar.sync(T.uint32(0), T.uint32(NUM_THREADS))

        # Real received-token count from the GPU prefix (combine.cuh:45-46);
        # kernel 1 has no PDL edge, so `__ldg` is legal here
        num_red_reg = T.alloc_local([1], "int32")
        T.evaluate(T.ptx["ld.global.nc.s32"](num_red_reg[0], psum_rank.ptr_to([NUM_RANKS_ - 1])))
        num_reduced = T.Select(
            num_reduced_tokens == NUM_RANKS_ * num_max_tokens_per_rank,
            num_red_reg[0],
            num_reduced_tokens,
        )

        tok_off = warp * COMBINE_TOKEN_SMEM
        tma_mbar = T.decl_buffer(
            (1,),
            "uint64",
            data=T.reinterpret(
                PointerType(PrimType("uint64")), smem.ptr_to([tok_off + COMBINE_TOKEN_BYTES])
            ),
            scope="shared.dyn",
        )

        phase = T.alloc_local([1], "uint32")
        phase[0] = T.uint32(0)
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1)))
            T.evaluate(T.ptx.fence.mbarrier_init.release.cluster())
        T.cuda.warp_sync()

        # Entry barrier (combine.cuh:77-80): gpu_barrier<tag0, false, false, true>
        nvlink_barrier(4)
        grid_barrier(0)

        # Contiguous token chunks per warp (combine.cuh:83-85: NOT a strided loop)
        num_tokens_per_warp = (num_reduced + num_sms * NUM_WARPS - 1) // (num_sms * NUM_WARPS)
        token_start = num_tokens_per_warp * global_warp_idx
        token_end = T.min(token_start + num_tokens_per_warp, num_reduced)
        chunk_trips = T.max(T.int32(0), token_end - token_start)
        for it in T.serial(0, chunk_trips):
            i = token_start + it
            # Source routing indices (combine.cuh:88-92)
            meta0 = T.alloc_local([1], "int32")
            T.evaluate(T.ptx["ld.global.nc.s32"](meta0[0], src_metadata.ptr_to([i * 8 + 0])))
            src_token_idx = meta0[0] % num_max_tokens_per_rank
            meta1 = T.alloc_local([1], "int32")
            T.evaluate(T.ptx["ld.global.nc.s32"](meta1[0], src_metadata.ptr_to([i * 8 + 1])))
            src_rank_idx = meta1[0] // NUM_TOPK
            src_topk_idx = meta1[0] % NUM_TOPK

            # Master slot on the SOURCE rank's combine recv region (combine.cuh:96-106)
            master_u64 = _peer_u64(peer_buf, src_rank_idx) + T.cast(
                (src_topk_idx * num_max_tokens_per_rank + src_token_idx) * COMBINE_TOKEN_BYTES,
                "uint64",
            )

            # no_local_reduce is always true in this specialization (combine.cuh:125-127)
            if T.cuda.elect_sync():
                # Drain the previous token's TMA store before reusing the SMEM slot
                T.ptx.cp.async_.bulk.wait_group(0)
                T.evaluate(
                    T.ptx[_BULK_G2S_CHAIN](
                        smem.ptr_to([tok_off]),
                        x.ptr_to([i * HIDDEN_BYTES]),
                        T.uint32(HIDDEN_BYTES),
                        tma_mbar.ptr_to([0]),
                        T.uint64(_EVICT_FIRST),
                    )
                )
                T.evaluate(
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        tma_mbar.ptr_to([0]), T.uint32(HIDDEN_BYTES)
                    )
                )
                T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), phase[0])
                phase[0] = phase[0] ^ T.uint32(1)
                T.evaluate(
                    T.ptx[_BULK_S2G_CHAIN](
                        T.reinterpret("handle", master_u64),
                        smem.ptr_to([tok_off]),
                        T.uint32(HIDDEN_BYTES),
                        T.uint64(_EVICT_NORMAL),
                    )
                )
                T.evaluate(T.ptx.cp.async_.bulk.commit_group())
            T.cuda.warp_sync()

            # Write this token's top-k weights into the same remote slot
            # (combine.cuh:216-226)
            if lane < NUM_TOPK:
                w = T.alloc_local([1], "float32")
                T.evaluate(
                    T.ptx["ld.global.nc.f32"](w[0], topk_weights.ptr_to([i * NUM_TOPK + lane]))
                )
                T.evaluate(
                    T.ptx.st.global_.b32(
                        _gptr(master_u64, 14360 + lane * 4), T.cuda.float_as_uint(w[0])
                    )
                )
            T.cuda.warp_sync()

        # Exit barrier (combine.cuh:239-242): gpu_barrier<tag1, true, true, false>
        T.evaluate(T.ptx.cp.async_.bulk.commit_group())
        T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.warp_sync()
        grid_barrier(1)
        nvlink_barrier(5)
        # No PDL trigger in kernel 1 (combine.cuh has none): kernel 2's
        # griddepcontrol.wait releases at kernel-1 completion.

    return deepep_combine.with_attr("tirx.kernel_launch_params", _launch_tags(cluster, False))


# ---------------------------------------------------------------------------
# Kernel 2: combine_reduce_epilogue_impl
# ---------------------------------------------------------------------------


def _build_reduce_epilogue_kernel(
    num_sms: int, num_max_tokens_per_rank: int, num_ranks: int
) -> Any:
    """`combine_reduce_epilogue_impl` (frozen sketch kernel 2)."""

    NUM_RANKS_ = num_ranks
    EXPERTS_PER_RANK = NUM_EXPERTS // num_ranks

    @T.prim_func
    def deepep_combine_reduce_epilogue(
        combined_x_ptr: T.handle,
        combined_topk_weights_ptr: T.handle,
        combined_topk_idx_ptr: T.handle,
        buffer_addr: T.int64,
        num_combined_tokens: T.int32,
    ):
        combined_x = T.match_buffer(
            combined_x_ptr, (num_max_tokens_per_rank * HIDDEN_BYTES,), "uint8"
        )
        combined_topk_weights = T.match_buffer(
            combined_topk_weights_ptr, (num_max_tokens_per_rank * NUM_TOPK,), "float32"
        )
        # The i64 top-k indices are passed as an int32 view; the source reads
        # only the little-endian low word of each entry (reviewer r2 finding 3)
        combined_topk_idx = T.match_buffer(
            combined_topk_idx_ptr, (num_max_tokens_per_rank * NUM_TOPK * 2,), "int32"
        )

        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        smem = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": SMEM_TOTAL})

        sm_idx = T.cta_id([num_sms])
        thread_idx = T.thread_id([NUM_THREADS])
        lane = T.lane_id([32])

        buf_u64 = T.cast(buffer_addr, "uint64")

        warp_u32 = T.alloc_local([1], "uint32")
        T.evaluate(
            T.ptx.shfl_sync.idx.b32(
                warp_u32[0], thread_idx // 32, T.uint32(0), T.uint32(31), T.uint32(0xFFFFFFFF)
            )
        )
        warp = T.cast(warp_u32[0], "int32")
        global_warp_idx = warp * num_sms + sm_idx

        tok_off = warp * OUTPUT_TOKEN_BYTES

        # Block until kernel 1 finished and all peer data are visible (epilogue.cuh:59)
        T.evaluate(T.ptx.griddepcontrol.wait())

        epi_stride = NUM_WARPS * num_sms
        epi_trips = T.max(
            T.int32(0), (num_combined_tokens - global_warp_idx + epi_stride - 1) // epi_stride
        )
        for epi_it in T.serial(0, epi_trips):
            token_idx = global_warp_idx + epi_it * epi_stride

            # Dst expert/rank per top-k lane (epilogue.cuh:66-71; plain ld, NOT
            # __ldg: "PDL is used, please do not use __ldg")
            dst_expert = T.alloc_local([1], "int32")
            dst_expert[0] = -1
            if lane < NUM_TOPK:
                T.evaluate(
                    T.ptx["ld.global.b32"](
                        dst_expert[0],
                        combined_topk_idx.ptr_to([token_idx * NUM_TOPK * 2 + lane * 2]),
                    )
                )
            T.cuda.warp_sync()
            dst_rank = T.Select(dst_expert[0] >= 0, dst_expert[0] // EXPERTS_PER_RANK, -1)

            # Dedup on dst rank (epilogue.cuh:82-84 else-branch, scaleout == 1)
            match_mask = T.alloc_local([1], "uint32")
            T.evaluate(T.ptx.match.any.sync.b32(match_mask[0], dst_rank, T.uint32(0xFFFFFFFF)))
            master = T.alloc_local([1], "uint32")
            T.evaluate(T.ptx.bfind.u32(master[0], match_mask[0]))
            is_master = T.cast(master[0], "int32") == lane
            ballot = T.alloc_local([1], "uint32")
            T.evaluate(
                T.ptx.vote_sync.ballot.b32(
                    ballot[0], is_master and (dst_rank >= 0), T.uint32(0xFFFFFFFF)
                )
            )

            # Sort valid top-k slots to front (combine_utils.cuh:43-53; fetch is
            # identity because kUseRankLayout=false)
            topk_slot = T.alloc_local([NUM_TOKENS_IN_LAYOUT], "int32")
            mask = T.alloc_local([1], "uint32")
            mask[0] = ballot[0]
            for k in T.unroll(NUM_TOKENS_IN_LAYOUT):
                lowest = T.cast(T.cuda.ffs_u32(mask[0]), "int32") - 1
                topk_slot[k] = T.Select(lowest >= 0, lowest, -1)
                mask[0] = T.bitwise_and(mask[0], mask[0] - T.uint32(1))

            # combine_reduce<896, 4, 6, 6>: no bias in this specialization, so the
            # path selector is exactly (topk_slot[2] < 0) (utils.cuh:68-70)
            if topk_slot[2] < 0:
                # --- bf16 hadd path (utils.cuh:73-110) ---
                for c in T.serial(0, 7):
                    values_0 = T.alloc_local([16], "int32", align=16)
                    values_1 = T.alloc_local([16], "int32", align=16)
                    for j in T.unroll(16):
                        values_0[j] = T.uint32(0)
                        values_1[j] = T.uint32(0)
                    if topk_slot[0] >= 0:
                        base0 = buf_u64 + T.cast(
                            (topk_slot[0] * num_max_tokens_per_rank + token_idx)
                            * COMBINE_TOKEN_BYTES,
                            "uint64",
                        )
                        for j in T.unroll(4):
                            T.evaluate(
                                T.ptx[_GEZ_VEC_LOAD](
                                    values_0[j * 4],
                                    values_0[j * 4 + 1],
                                    values_0[j * 4 + 2],
                                    values_0[j * 4 + 3],
                                    T.reinterpret(
                                        "handle",
                                        base0 + T.cast((c * 128 + j * 32 + lane) * 16, "uint64"),
                                    ),
                                    T.uint64(_EVICT_FIRST),
                                )
                            )
                    if topk_slot[1] >= 0:
                        base1 = buf_u64 + T.cast(
                            (topk_slot[1] * num_max_tokens_per_rank + token_idx)
                            * COMBINE_TOKEN_BYTES,
                            "uint64",
                        )
                        for j in T.unroll(4):
                            T.evaluate(
                                T.ptx[_GEZ_VEC_LOAD](
                                    values_1[j * 4],
                                    values_1[j * 4 + 1],
                                    values_1[j * 4 + 2],
                                    values_1[j * 4 + 3],
                                    T.reinterpret(
                                        "handle",
                                        base1 + T.cast((c * 128 + j * 32 + lane) * 16, "uint64"),
                                    ),
                                    T.uint64(_EVICT_FIRST),
                                )
                            )
                    if c == 0:
                        # Drain the previous token's TMA store before reusing smem
                        T.ptx.cp.async_.bulk.wait_group(0)
                        T.cuda.warp_sync()
                    for j in T.unroll(4):
                        sums = T.alloc_local([4], "uint32")
                        for w in T.unroll(4):
                            T.evaluate(
                                T.ptx["add.bf16x2"](
                                    sums[w],
                                    T.cast(values_0[j * 4 + w], "uint32"),
                                    T.cast(values_1[j * 4 + w], "uint32"),
                                )
                            )
                        T.evaluate(
                            T.ptx.st.shared_.v4.b32(
                                smem.ptr_to([tok_off + (c * 128 + j * 32 + lane) * 16]),
                                sums[0],
                                sums[1],
                                sums[2],
                                sums[3],
                            )
                        )
            else:
                # --- fp32 accumulate path (utils.cuh:111-168) ---
                for c in T.serial(0, 7):
                    reduced = T.alloc_local([32], "float32")
                    for j in T.unroll(32):
                        reduced[j] = T.float32(0.0)
                    for k in T.unroll(NUM_TOKENS_IN_LAYOUT):
                        values = T.alloc_local([16], "int32", align=16)
                        for j in T.unroll(16):
                            values[j] = T.uint32(0)
                        if topk_slot[k] >= 0:
                            base_k = buf_u64 + T.cast(
                                (topk_slot[k] * num_max_tokens_per_rank + token_idx)
                                * COMBINE_TOKEN_BYTES,
                                "uint64",
                            )
                            for j in T.unroll(4):
                                T.evaluate(
                                    T.ptx[_GEZ_VEC_LOAD](
                                        values[j * 4],
                                        values[j * 4 + 1],
                                        values[j * 4 + 2],
                                        values[j * 4 + 3],
                                        T.reinterpret(
                                            "handle",
                                            base_k
                                            + T.cast((c * 128 + j * 32 + lane) * 16, "uint64"),
                                        ),
                                        T.uint64(_EVICT_FIRST),
                                    )
                                )
                        for j in T.unroll(4):
                            for w in T.unroll(4):
                                word_u32 = T.cast(values[j * 4 + w], "uint32")
                                lo = T.cast(T.bitwise_and(word_u32, T.uint32(0xFFFF)), "uint16")
                                hi = T.cast(word_u32 >> T.uint32(16), "uint16")
                                e = j * 8 + w * 2
                                T.evaluate(T.ptx.add.rn.f32.bf16(reduced[e], lo, reduced[e]))
                                T.evaluate(
                                    T.ptx.add.rn.f32.bf16(reduced[e + 1], hi, reduced[e + 1])
                                )
                    if c == 0:
                        T.ptx.cp.async_.bulk.wait_group(0)
                        T.cuda.warp_sync()
                    for j in T.unroll(4):
                        casted = T.alloc_local([4], "uint32")
                        for p in T.unroll(4):
                            casted[p] = T.cuda.float22bfloat162_rn(
                                reduced[j * 8 + p * 2], reduced[j * 8 + p * 2 + 1]
                            )
                        T.evaluate(
                            T.ptx.st.shared_.v4.b32(
                                smem.ptr_to([tok_off + (c * 128 + j * 32 + lane) * 16]),
                                casted[0],
                                casted[1],
                                casted[2],
                                casted[3],
                            )
                        )

            # Async-proxy fence so the TMA engine sees the smem reduce output
            T.evaluate(T.ptx["fence.proxy.async.shared::cta"]())
            T.cuda.warp_sync()

            # TMA-store the reduced token (epilogue.cuh:120-123)
            if T.cuda.elect_sync():
                T.evaluate(
                    T.ptx[_BULK_S2G_CHAIN](
                        combined_x.ptr_to([token_idx * OUTPUT_TOKEN_BYTES]),
                        smem.ptr_to([tok_off]),
                        T.uint32(OUTPUT_TOKEN_BYTES),
                        T.uint64(_EVICT_NORMAL),
                    )
                )
                T.evaluate(T.ptx.cp.async_.bulk.commit_group())
            T.cuda.warp_sync()

            # Write combined top-k weights from the master lane's rank buffer
            # (epilogue.cuh:128-141)
            if lane < NUM_TOPK:
                w32 = T.alloc_local([1], "uint32")
                w32[0] = T.uint32(0)
                if dst_rank >= 0:
                    T.evaluate(
                        T.ptx["ld.global.b32"](
                            w32[0],
                            _gptr(
                                buf_u64,
                                (T.cast(master[0], "int32") * num_max_tokens_per_rank + token_idx)
                                * COMBINE_TOKEN_BYTES
                                + 14360
                                + lane * 4,
                            ),
                        )
                    )
                combined_topk_weights[token_idx * NUM_TOPK + lane] = T.cuda.uint_as_float(w32[0])
            T.cuda.warp_sync()

    return deepep_combine_reduce_epilogue.with_attr(
        "tirx.kernel_launch_params", _launch_tags(1, True)
    )


# ---------------------------------------------------------------------------
# Module entries (tirx-kernels conventions)
# ---------------------------------------------------------------------------


def _device_num_sms() -> int:
    """Device SM count without forcing CUDA init in the bench-suite CPU stage.

    Reads the suite-provided `TIRX_PREPARE_NUM_SMS` first; falls back to a
    torch device query in unguarded contexts.
    """

    import os

    from tirx_kernels.runner import PREPARE_NUM_SMS_ENV

    value = os.environ.get(PREPARE_NUM_SMS_ENV)
    if value:
        return int(value)
    import torch

    return torch.cuda.get_device_properties(0).multi_processor_count


def get_kernel(
    world_size: int = NUM_RANKS,
    num_tokens: int = 4096,
    hidden: int = 7168,
    num_experts: int = NUM_EXPERTS,
    num_topk: int = NUM_TOPK,
    expert_alignment: int = 1,
    num_sms: int = 0,
    **_: Any,
) -> list[Any]:
    """Return the combine kernel pair (main + reduce epilogue), closure-specialized."""

    if num_experts != NUM_EXPERTS or num_topk != NUM_TOPK:
        raise ValueError("config is outside the ported specialization boundary")
    if hidden * 2 != HIDDEN_BYTES:
        raise ValueError("config is outside the ported specialization boundary")
    if num_experts % world_size != 0:
        raise ValueError(f"num_experts={num_experts} not divisible by world_size={world_size}")
    if num_sms == 0:
        num_sms = get_theoretical_num_sms(
            world_size, num_experts, num_topk, prefer_overlap_with_compute=False
        )
    # Source parity: the reduce epilogue always launches with full device SMs
    # (buffer.hpp launch_combine_reduce_epilogue: device_runtime->get_num_sms()),
    # not the combine kernel's num_sms.
    epilogue_num_sms = _device_num_sms()
    return [
        _build_combine_kernel(num_sms, num_tokens, world_size),
        _build_reduce_epilogue_kernel(epilogue_num_sms, num_tokens, world_size),
    ]


def prepare_data(
    world_size: int,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    expert_alignment: int = 1,
    masked_ratio: float = 0.0,
    seed: int = 42,
    rank: int = 0,
    **_: Any,
) -> dict[str, Any]:
    """Rank-local logical inputs; identical generation to the dispatch module."""

    from . import dispatch as _dispatch

    return _dispatch.prepare_data(
        world_size=world_size,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        expert_alignment=expert_alignment,
        masked_ratio=masked_ratio,
        seed=seed,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Distributed worker
# ---------------------------------------------------------------------------


def _closed_form_x(alloc: int, hidden: int, device: Any) -> Any:
    """Deterministic combine input, identical on every rank for the same row.

    x[i][h] = sin(i * 0.373 + h * 0.001) in bf16. The combine kernels move and
    sum arbitrary x rows; an exact closed form lets the small-rank torch golden
    reproduce any row without communication.
    """

    import torch

    rows = torch.arange(alloc, device=device, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(hidden, device=device, dtype=torch.float32).unsqueeze(0)
    return torch.sin(rows * 0.373 + cols * 0.001).to(torch.bfloat16)


def _simulate_combine_torch(
    topk_idx: Any,
    topk_weights: Any,
    x_input: Any,
    world_size: int,
    num_tokens_max: int,
    num_experts: int,
    rank: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Small-rank oracle: simulate dispatch routing and the combine golden.

    The reference ElasticBuffer segfaults below 8 ranks on this host, so 1/2/4-
    rank bring-up validates against pure-torch semantics instead:
    metadata/psum/topk_weights inputs are simulated per the dispatch contract
    (one received token per (token, rank) group, master lane = group identity,
    rows sorted by (src rank, src token)), and the golden combine output is
    computed from the same rows: per local token, sum each contributing
    group's x row in ascending master-lane order in fp32, then cast to bf16.
    """

    import numpy as np
    import torch
    import torch.distributed as dist

    num_topk = topk_idx.shape[1]
    hidden = x_input.shape[1]
    experts_per_rank = num_experts // world_size
    alloc = world_size * num_tokens_max

    idx_all = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(idx_all, topk_idx)
    w_all = [torch.empty_like(topk_weights) for _ in range(world_size)]
    dist.all_gather(w_all, topk_weights)
    idx_all_np = [t.cpu().numpy() for t in idx_all]
    w_all_np = [t.cpu().numpy() for t in w_all]

    # Per-receiver received rows: recv_rows[g] = [(src_global, src * topk + master)]
    recv_rows: list[list[tuple[int, int]]] = []
    row_of: dict[tuple[int, int], int] = {}
    for g in range(world_size):
        rows: list[tuple[int, int]] = []
        for s in range(world_size):
            for t in range(idx_all_np[s].shape[0]):
                lanes = idx_all_np[s][t]
                groups: dict[int, int] = {}
                for lane in range(num_topk):
                    expert = int(lanes[lane])
                    if expert >= 0:
                        groups.setdefault(expert // experts_per_rank, lane)
                if g in groups:
                    rows.append((s * num_tokens_max + t, s * num_topk + groups[g]))
        rows.sort()
        for row_idx, (src_global, meta1) in enumerate(rows):
            row_of[(g, src_global)] = row_idx
        recv_rows.append(rows)

    # This rank's kernel inputs
    my_rows = recv_rows[rank]
    num_recv = len(my_rows)
    metadata = torch.zeros((alloc, 2 + num_topk), dtype=torch.int32, device=device)
    if num_recv:
        metadata[:num_recv, 0] = torch.tensor(
            [r[0] for r in my_rows], dtype=torch.int32, device=device
        )
        metadata[:num_recv, 1] = torch.tensor(
            [r[1] for r in my_rows], dtype=torch.int32, device=device
        )
    counts = [0] * world_size
    for src_global, _ in my_rows:
        counts[src_global // num_tokens_max] += 1
    psum_rank = torch.tensor(np.cumsum(counts), dtype=torch.int32, device=device)
    tw_input = torch.zeros((alloc, num_topk), dtype=torch.float32, device=device)
    for i, (src_global, _) in enumerate(my_rows):
        s, t = src_global // num_tokens_max, src_global % num_tokens_max
        tw_input[i] = torch.tensor(w_all_np[s][t], dtype=torch.float32, device=device)

    # Golden: per local token, sum groups' x rows in ascending master-lane order
    local_num_tokens = topk_idx.shape[0]
    my_idx_np = idx_all_np[rank]
    my_w_np = w_all_np[rank]
    golden_x = torch.zeros((local_num_tokens, hidden), dtype=torch.bfloat16, device=device)
    cols = torch.arange(hidden, device=device, dtype=torch.float32).unsqueeze(0)
    for t in range(local_num_tokens):
        groups = {}
        for lane in range(num_topk):
            expert = int(my_idx_np[t][lane])
            if expert >= 0:
                groups.setdefault(expert // experts_per_rank, lane)
        acc = torch.zeros((hidden,), dtype=torch.float32, device=device)
        for g in sorted(groups, key=groups.get):
            row = row_of[(g, rank * num_tokens_max + t)]
            acc += torch.sin(row * 0.373 + cols).view(-1)
        golden_x[t] = acc.to(torch.bfloat16)
    golden_w = torch.where(topk_idx >= 0, topk_weights, torch.zeros_like(topk_weights))
    return metadata, psum_rank, tw_input, golden_x, golden_w


def _run_worker(
    runtime: Any, modules: dict[str, Any], mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    import os

    import torch
    import torch.distributed as dist

    from .utils._buffer import SymmetricWindow

    rank = runtime.rank
    world_size = kwargs["world_size"]
    num_tokens_max = kwargs["num_tokens"]
    hidden = kwargs["hidden"]
    num_experts = kwargs["num_experts"]
    num_topk = kwargs["num_topk"]
    expert_alignment = kwargs["expert_alignment"]
    num_sms = kwargs["num_sms"]

    data = prepare_data(**{k: v for k, v in kwargs.items() if k != "num_sms"}, rank=rank)
    x, topk_idx, topk_weights = data["x"], data["topk_idx"], data["topk_weights"]
    local_num_tokens = data["num_tokens"]

    device = torch.device("cuda", rank)
    alloc = world_size * num_tokens_max
    x_input = _closed_form_x(alloc, hidden, device)

    combine_fn = modules["combine"].get_function("main")
    epilogue_fn = modules["reduce_epilogue"].get_function("main")

    # TIRx-side symmetric window + output tensors
    window = SymmetricWindow(
        dist.group.WORLD, NUM_TOKENS_IN_LAYOUT * num_tokens_max * COMBINE_TOKEN_BYTES
    )
    # Buffers are declared with num_tokens_max rows (the kernel's static bound);
    # only the first local_num_tokens rows are written and compared.
    combined_x = torch.empty((num_tokens_max, hidden), dtype=torch.bfloat16, device=device)
    combined_w = torch.empty((num_tokens_max, num_topk), dtype=torch.float32, device=device)
    topk_idx_padded = torch.zeros((num_tokens_max, num_topk), dtype=torch.int64, device=device)
    topk_idx_padded[:local_num_tokens] = topk_idx

    ref_buffer = None
    handle = None
    golden_x = golden_w = None
    if world_size == 8:
        # The reference runtime needs GIN disabled on this host (verified in the
        # dispatch scaffolding: single-node NVLink LSA path works with EP_DISABLE_GIN=1).
        os.environ.setdefault("EP_DISABLE_GIN", "1")
        import deep_ep

        ref_buffer = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=num_tokens_max,
            hidden=hidden,
            num_topk=num_topk,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
        )
        _, _, recv_topk_weights, handle, _ = ref_buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=num_tokens_max,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_sms=num_sms,
            do_cpu_sync=False,
        )
        metadata = handle.recv_src_metadata
        psum_rank = handle.psum_num_recv_tokens_per_scaleup_rank
        tw_input = recv_topk_weights
    else:
        metadata, psum_rank, tw_input, golden_x, golden_w = _simulate_combine_torch(
            topk_idx, topk_weights, x_input, world_size, num_tokens_max, num_experts, rank, device
        )

    def tirx_launch() -> None:
        combine_fn(
            x_input.view(torch.uint8).view(-1),
            tw_input.view(-1),
            metadata.view(-1),
            psum_rank,
            window.peer_ws_ptrs,
            window.peer_buf_ptrs,
            window.base_ptr,
            window.buffer_ptr,
            alloc,
            rank,
        )
        epilogue_fn(
            combined_x.view(torch.uint8).view(-1),
            combined_w.view(-1),
            topk_idx_padded.view(torch.int32).view(-1),
            window.buffer_ptr,
            local_num_tokens,
        )

    def reference_launch():
        return ref_buffer.combine(x_input, handle, topk_weights=tw_input)

    try:
        if mode == "test":

            def _launch_and_check() -> None:
                with torch.cuda.stream(runtime.timing_stream):
                    if world_size == 8:
                        ref_cx, ref_cw, _ = reference_launch()
                    tirx_launch()
                runtime.device.sync(runtime.compute_stream)

                our_cx = combined_x[:local_num_tokens]
                our_cw = combined_w[:local_num_tokens]
                if world_size == 8:
                    assert torch.equal(ref_cx, our_cx), (
                        f"rank {rank}: combined_x mismatch "
                        f"(max abs diff {(ref_cx.float() - our_cx.float()).abs().max().item()})"
                    )
                    assert torch.equal(ref_cw, our_cw), (
                        f"rank {rank}: combined_topk_weights mismatch"
                    )
                else:
                    assert torch.equal(golden_x, our_cx), (
                        f"rank {rank}: combined_x mismatch vs torch golden "
                        f"(max abs diff {(golden_x.float() - our_cx.float()).abs().max().item()})"
                    )
                    assert torch.equal(golden_w, our_cw), (
                        f"rank {rank}: combined_topk_weights mismatch vs torch golden"
                    )

            check_error = ""
            try:
                _launch_and_check()
            except Exception as error:  # reported uniformly below
                check_error = f"{type(error).__name__}: {error}"
                print(f"[rank {rank}] CHECK FAILED: {check_error}", flush=True)
            # Never diverge: a failed rank must not strand the others at the
            # next collective.
            status = torch.tensor([not check_error], dtype=torch.int32, device=device)
            dist.all_reduce(status, op=dist.ReduceOp.MIN)
            if not status.item():
                raise AssertionError(f"deepep_combine check failed on some rank: {check_error}")
            return {"status": "OK"}

        # mode == "bench"
        from tvm.tirx.bench import bench

        def build_reference():
            def launch() -> None:
                reference_launch()

            return launch

        with torch.cuda.stream(runtime.timing_stream):
            result = bench(
                {"tirx": tirx_launch},
                references={"deepep": build_reference},
                timer="kineto",
                rounds=kwargs.get("rounds", 1),
                cooldown_s=kwargs.get("cooldown_s", 1.0),
                distributed=runtime.bench_context(),
            )
        return {"status": "OK", **result}
    finally:
        window.destroy()
        if ref_buffer is not None:
            ref_buffer.destroy()


def _resolve_num_sms(config: dict[str, Any]) -> int:
    # prefer_overlap_with_compute=False mirrors the source perf test default
    # (tests/elastic/test_ep.py), e.g. 64 SMs for e256/k6.
    return get_theoretical_num_sms(
        config["world_size"],
        config["num_experts"],
        config["num_topk"],
        prefer_overlap_with_compute=False,
    )


def run_test(**config: Any) -> None:
    """Correctness entry point used by the runner."""

    from .utils._runtime import run_distributed

    num_sms = _resolve_num_sms(config)
    combine_kernel, epilogue_kernel = get_kernel(**config, num_sms=num_sms)
    run_distributed(
        {"combine": combine_kernel, "reduce_epilogue": epilogue_kernel},
        world_size=config["world_size"],
        worker=_run_worker,
        mode="test",
        worker_kwargs={**config, "num_sms": num_sms},
    )


def _resolve_num_sms_cpu(config: dict[str, Any]) -> int:
    """CUDA-free mirror of `ElasticBuffer.get_theoretical_num_sms` for the
    single-domain path, for the bench-suite CPU-prepare stage.

    The source model's only CUDA-initializing call is its closing
    `torch.cuda.get_device_properties`; the bench suite forbids CUDA in CPU
    prepare, so the arithmetic is mirrored with the suite-provided device SM
    count (TIRX_PREPARE_NUM_SMS via `_device_num_sms`) and the
    subprocess-based `get_nvlink_gbs` (no CUDA init). For the bench config
    (world=8, e=256, k=6) the result is 64 over a wide bandwidth range, equal
    to the source model's output.
    """

    import math

    from deep_ep.utils.envs import get_nvlink_gbs

    world = config["world_size"]
    experts = config["num_experts"]
    topk = config["num_topk"]
    expected_topk = world * (
        1 - math.comb(experts - experts // world, topk) / math.comb(experts, topk)
    )
    gbs = get_nvlink_gbs()
    nvlink_traffic = 1 - 1 / world if world > 1 else 0.0
    device_sms = _device_num_sms()
    num_sms = float(device_sms)
    if nvlink_traffic > 0:
        num_sms = max(
            gbs / nvlink_traffic * (1 / expected_topk) / 200, gbs / nvlink_traffic * 1 / 50
        )
    # align(max(4, ceil(x * 1.25)), 2); prefer_overlap_with_compute=False -> max(.., 64)
    num_sms = max(4, math.ceil(num_sms * 1.25))
    num_sms += num_sms % 2
    num_sms = max(num_sms, 64)
    return min(num_sms, device_sms)


def _run_bench_gpu(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """GPU stage of the two-stage benchmark contract (ranks start CUDA here).

    All TIRx specialization and compilation completed in prepare_bench; this
    stage only spawns ranks against the prebuilt libraries.
    """

    from .utils._runtime import run_distributed

    config = state["config"]
    return run_distributed(
        {},
        world_size=config["world_size"],
        worker=_run_worker,
        mode="bench",
        worker_kwargs={
            **config,
            "num_sms": state["num_sms"],
            "rounds": kwargs.get("rounds", 1),
            "cooldown_s": kwargs.get("cooldown_s", 1.0),
        },
        prepared_libraries=state["library_paths"],
    )


def prepare_bench(**config: Any):
    """CPU-side prepare for the two-stage benchmark contract.

    Completes every CUDA-free step the suite requires before READY: SM-count
    resolution (no CUDA-initializing calls), TIRx specialization, and
    tvm.compile + export. Ranks start CUDA later in `_run_bench_gpu`.
    """

    import tempfile

    from tirx_kernels.runner import prepared_gpu_benchmark

    from .utils._runtime import compile_kernels

    if config.get("timer") not in {None, "kineto"}:
        raise ValueError(
            f"deepep_combine is distributed and supports only kineto, got {config['timer']}"
        )
    num_sms = _resolve_num_sms_cpu(config)
    combine_kernel, epilogue_kernel = get_kernel(**config, num_sms=num_sms)
    tmpdir = tempfile.TemporaryDirectory(prefix="tirx-deepep-prepare-")
    library_paths = compile_kernels(
        {"combine": combine_kernel, "reduce_epilogue": epilogue_kernel}, tmpdir.name
    )
    state = {
        "config": dict(config),
        "num_sms": num_sms,
        "library_paths": library_paths,
        "tmpdir": tmpdir,
    }
    return prepared_gpu_benchmark(
        _run_bench_gpu, state, required_num_gpus=config["world_size"], close=state["tmpdir"].cleanup
    )


def run_bench(
    *args: Any,
    warmup: Any = None,
    repeat: Any = None,
    timer: Any = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark entry point used by the runner (kineto only, distributed)."""

    if timer is not None and timer != "kineto":
        raise ValueError(f"deepep_combine is distributed and supports only kineto, got {timer}")
    if warmup is not None or repeat is not None:
        raise ValueError("kineto uses fixed iteration counts and rejects warmup/repeat overrides")
    config = dict(kwargs)
    if args:
        raise TypeError(f"unexpected positional arguments: {args}")
    return prepare_bench(**config).run_gpu(rounds=rounds, cooldown_s=cooldown_s)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
