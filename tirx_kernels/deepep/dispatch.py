# This file is a TIRx port of code from DeepEP
# (https://github.com/deepseek-ai/DeepEP @ 01dc3aa), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""DeepEP V2 elastic dispatch (single-domain NVLink path) ported to TIRx.

Source: /home/bohanhou/kernel-libs/deepep
  - deep_ep/include/deep_ep/impls/dispatch.cuh (`dispatch_impl`, direct path)
  - deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh
    (`dispatch_copy_epilogue_impl`)

Frozen sketch (source of the implementation plan):
  `.agents/sketch/deepep/dispatch.md`

Fixed specialization: bf16 tokens, num_sf_packs=0, non-cached, non-expand,
do_cpu_sync=False, deterministic=False, num_scaleout_ranks==1,
is_scaleup_nvlink=True.
"""

from __future__ import annotations

from typing import Any

from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from .utils._buffer import get_theoretical_num_sms

KERNEL_META = {"name": "deepep_dispatch", "category": "deepep", "compute_capability": 10}

# Correctness matrix. Every config runs the same source specialization
# (bf16, non-cached, non-expand, do_cpu_sync=False) on `world_size` ranks.
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
EXPERTS_PER_RANK = NUM_EXPERTS // NUM_RANKS
HIDDEN_BYTES = 7168 * 2  # bf16
NUM_NOTIFY_WARPS = 4
NUM_NOTIFY_THREADS = NUM_NOTIFY_WARPS * 32

SMEM_TOTAL = 232448  # max opt-in dynamic smem per block on SM100
TOKEN_META_BYTES = 76
TOKEN_BYTES_GMEM = 14432  # align(14336,32) + align(0,32) + align(76,32)
TOKEN_BYTES_SMEM = 14464  # + align(8,32) mbarrier
NOTIFY_SMEM_BYTES = 1536  # align(8 + 256, 128) * 4

# layout::WorkspaceLayout byte offsets (common/layout.cuh)
WS_BARRIER_COUNTER = 0
WS_BARRIER_SIGNAL = 8
WS_NOTIFY_REDUCTION = 16
WS_COUNT_SEND = 24592
WS_COUNT_RECV = 49168
WS_SENDER_COUNTER = 73744
WS_PORT_SCRATCH = 12_820_528  # software grid-barrier slots (sketch substitution 2)

NUM_COUNT_SLOTS = NUM_RANKS + NUM_EXPERTS  # 264

TIMEOUT_CYCLES = 200_000_000_000  # 100 s at ~2 GHz (comm.cuh: kNumOneSecCycles)

_BULK_G2S_CHAIN = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
_BULK_S2G_CHAIN = "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"
_EVICT_FIRST = 0x12F0000000000000
_EVICT_NORMAL = 0x1000000000000000


def _gptr(base_u64, byte_off):
    return T.reinterpret("handle", base_u64 + T.cast(byte_off, "uint64"))


def _peer_u64(table, dst):
    return T.cast(table[dst], "uint64")


def _ld_volatile_u64(dst, addr):
    return T.ptx.ld.volatile.global_.u64(dst, addr)


def _ld_volatile_s64(dst, addr):
    return T.ptx.ld.volatile.global_.s64(dst, addr)


def _ld_acquire_sys_s32(dst, addr):
    return T.ptx.ld.acquire.sys.global_.s32(dst, addr)


def _ld_acquire_gpu_s32(dst, addr):
    return T.ptx.ld.acquire.gpu.global_.s32(dst, addr)


def _ld_acquire_gpu_u64(dst, addr):
    return T.ptx.ld.acquire.gpu.global_.u64(dst, addr)


def _shfl_idx(dst, src, src_lane):
    return T.ptx.shfl_sync.idx.b32(
        dst, src, T.cast(src_lane, "uint32"), T.uint32(31), T.uint32(0xFFFFFFFF)
    )


def _warp_inclusive_sum(value, lane):
    # 5-step Hillis-Steele over the full warp (ptx.cuh:423-431)
    result = value
    for offset in (1, 2, 4, 8, 16):
        tmp = T.alloc_local([1], "int32")
        T.evaluate(
            T.ptx.shfl_sync.up.b32(
                tmp[0], result, T.uint32(offset), T.uint32(0), T.uint32(0xFFFFFFFF)
            )
        )
        result = T.Select(lane >= offset, result + tmp[0], result)
    return result


def _launch_tags(cluster: int) -> list[str]:
    tags = ["blockIdx.x"]
    if cluster > 1:
        tags.append("clusterCtaIdx.x")
    tags += ["threadIdx.x", "tirx.use_programtic_dependent_launch", "tirx.use_dyn_shared_memory"]
    return tags


def _build_dispatch_kernel(
    num_sms: int, num_max_tokens_per_rank: int, expert_alignment: int, num_ranks: int
) -> Any:
    """`dispatch_impl` for the direct single-domain path (frozen sketch kernel 1)."""

    # Rank-count-dependent constants, closure-specialized per world size. The
    # workspace byte offsets stay sized for 8 ranks, which is the worst case.
    NUM_RANKS = num_ranks
    EXPERTS_PER_RANK = NUM_EXPERTS // num_ranks
    NUM_COUNT_SLOTS = NUM_RANKS + NUM_EXPERTS

    num_dispatch_warps = min((SMEM_TOTAL - NOTIFY_SMEM_BYTES) // TOKEN_BYTES_SMEM, 28)
    num_threads = (NUM_NOTIFY_WARPS + num_dispatch_warps) * 32
    cluster = 2 - num_sms % 2
    recv_region_bytes_per_rank = num_max_tokens_per_rank * TOKEN_BYTES_GMEM

    @T.prim_func
    def deepep_dispatch(
        x_ptr: T.handle,
        topk_idx_ptr: T.handle,
        topk_weights_ptr: T.handle,
        copied_topk_idx_ptr: T.handle,
        psum_rank_ptr: T.handle,
        psum_expert_ptr: T.handle,
        num_unaligned_ptr: T.handle,
        dst_slot_idx_ptr: T.handle,
        peer_ws_ptrs_ptr: T.handle,
        peer_buf_ptrs_ptr: T.handle,
        workspace_addr: T.int64,
        buffer_addr: T.int64,
        num_tokens: T.int32,
        rank_idx: T.int32,
    ):
        x = T.match_buffer(x_ptr, (num_tokens * HIDDEN_BYTES,), "uint8")
        topk_idx = T.match_buffer(topk_idx_ptr, (num_tokens * NUM_TOPK,), "int64")
        topk_weights = T.match_buffer(topk_weights_ptr, (num_tokens * NUM_TOPK,), "float32")
        copied_topk_idx = T.match_buffer(copied_topk_idx_ptr, (num_tokens * NUM_TOPK,), "int64")
        psum_rank = T.match_buffer(psum_rank_ptr, (NUM_RANKS,), "int32")
        psum_expert = T.match_buffer(psum_expert_ptr, (EXPERTS_PER_RANK + 1,), "int32")
        num_unaligned = T.match_buffer(num_unaligned_ptr, (EXPERTS_PER_RANK,), "int32")
        dst_slot_idx = T.match_buffer(dst_slot_idx_ptr, (num_tokens * NUM_TOPK,), "int32")
        peer_ws_ptrs = T.match_buffer(peer_ws_ptrs_ptr, (NUM_RANKS,), "int64")
        peer_buf_ptrs = T.match_buffer(peer_buf_ptrs_ptr, (NUM_RANKS,), "int64")

        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        smem = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": SMEM_TOTAL})

        sm_idx = T.cta_id([num_sms])
        if cluster > 1:
            cta_in_cluster = T.cta_id_in_cluster([cluster])
        thread_idx = T.thread_id([num_threads])
        lane = T.lane_id([32])

        # --- scalar helpers (module level: _gptr/_peer_u64/_ld_*/_shfl_idx) -
        ws_u64 = T.cast(workspace_addr, "uint64")

        # warp-uniform warp index (ptx.cuh: get_warp_idx)
        warp_u32 = T.alloc_local([1], "uint32")
        T.evaluate(_shfl_idx(warp_u32[0], thread_idx // 32, 0))
        warp = T.cast(warp_u32[0], "int32")

        # --- NVLink barrier (comm.cuh:88-129), SM 0 only --------------------
        @T.inline
        def nvlink_barrier(tag):
            if sm_idx == 0:
                counter_ptr = _gptr(ws_u64, WS_BARRIER_COUNTER)
                cnt = T.alloc_local([1], "uint64")
                T.evaluate(_ld_volatile_u64(cnt[0], counter_ptr))
                status = T.cast(T.bitwise_and(cnt[0], T.uint64(3)), "int32")
                phase = T.bitwise_and(status, 1)
                sign = status // 2
                if thread_idx < NUM_RANKS:
                    delta = T.Select(sign == 0, T.int32(1), T.int32(-1))
                    T.evaluate(
                        T.ptx.red.release.sys.global_.add.s32(
                            _gptr(
                                _peer_u64(peer_ws_ptrs, thread_idx), WS_BARRIER_SIGNAL + phase * 4
                            ),
                            delta,
                        )
                    )
                # comm.cuh:107 __syncthreads (SM 0's CTA only)
                T.ptx.bar.sync(T.uint32(0), T.uint32(num_threads))
                if thread_idx == 0:
                    old = T.alloc_local([1], "uint64")
                    T.evaluate(T.ptx.atom.global_.add.u64(old[0], counter_ptr, T.uint64(1)))
                    target = T.Select(sign == 0, T.int32(NUM_RANKS), T.int32(0))
                    sig = T.alloc_local([1], "int32")
                    sig_ptr = _gptr(ws_u64, WS_BARRIER_SIGNAL + phase * 4)
                    T.evaluate(_ld_acquire_sys_s32(sig[0], sig_ptr))
                    start_clock = T.cuda.clock64()
                    while sig[0] != target:
                        if T.cuda.clock64() - start_clock >= T.uint64(TIMEOUT_CYCLES):
                            T.cuda.printf(
                                "DeepEP NVLink barrier timeout, tag: %d, nvl: %d, "
                                "signal: %d, phase: %d, target: %d\n",
                                T.int32(tag),
                                rank_idx,
                                sig[0],
                                phase,
                                target,
                            )
                            T.cuda.trap_when_assert_failed(False)
                        T.evaluate(_ld_acquire_sys_s32(sig[0], sig_ptr))

        # --- Software grid barrier (sketch substitution 2) ------------------
        # Monotonic per-site u64 counter: every CTA's thread 0 adds 1, then
        # spins until the counter crosses the next multiple of num_sms. No
        # reset and no atomic-return ticket -> nothing for the compiler to
        # mangle, and no reset race with straggler CTAs.
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
            T.ptx.bar.sync(T.uint32(0), T.uint32(num_threads))

        # -------------------------------------------------------------------
        # Entry NVLink barrier (tag0) + end grid sync (dispatch.cuh:73-76)
        # -------------------------------------------------------------------
        # Reset the atomic sender counters up front (source resets them at
        # kernel end; moved here after the mid-kernel reset race traced to
        # the end-of-kernel cleanup). Every CTA allocates only after
        # grid_barrier(0), which is gated behind this store on SM 0.
        if sm_idx == 0 and thread_idx < NUM_RANKS:
            T.evaluate(
                T.ptx.st.global_.s32(_gptr(ws_u64, WS_SENDER_COUNTER + thread_idx * 4), T.int32(0))
            )
        nvlink_barrier(2)
        grid_barrier(0)

        if warp < NUM_NOTIFY_WARPS:
            # =================================================================
            # NOTIFY ROLE: warps 0..3 (dispatch.cuh:79-258)
            # =================================================================
            rank_expert_count = T.decl_buffer(
                (384,),
                "int32",
                data=T.reinterpret(PointerType(PrimType("int32")), smem.ptr_to([0])),
                scope="shared.dyn",
            )

            # Clean initial counts (dispatch.cuh:87-89)
            for i in T.serial(0, 3):
                rank_expert_count[i * NUM_NOTIFY_THREADS + thread_idx] = 0
            T.ptx.bar.sync(T.uint32(1), T.uint32(NUM_NOTIFY_THREADS))

            # Per-token counting (dispatch.cuh:94-107)
            atom_dst = T.alloc_local([1], "int32")
            global_warp_idx = warp * num_sms + sm_idx
            notify_stride = NUM_NOTIFY_WARPS * num_sms
            notify_trips = T.max(
                T.int32(0), (num_tokens - global_warp_idx + notify_stride - 1) // notify_stride
            )
            for notify_it in T.serial(0, notify_trips):
                i = global_warp_idx + notify_it * notify_stride
                e64 = T.alloc_local([1], "int64")
                e64[0] = T.int64(-1)
                if lane < NUM_TOPK:
                    T.evaluate(
                        T.ptx["ld.global.nc.s64"](e64[0], topk_idx.ptr_to([i * NUM_TOPK + lane]))
                    )
                dst_expert = T.cast(e64[0], "int32")
                if dst_expert >= 0:
                    T.evaluate(
                        T.ptx.atom.shared.add.s32(
                            atom_dst[0],
                            rank_expert_count.ptr_to([NUM_RANKS + dst_expert]),
                            T.int32(1),
                        )
                    )
                dst_rank = T.Select(dst_expert >= 0, dst_expert // EXPERTS_PER_RANK, -1)
                match_mask = T.alloc_local([1], "uint32")
                T.evaluate(T.ptx.match.any.sync.b32(match_mask[0], dst_rank, T.uint32(0xFFFFFFFF)))
                master = T.alloc_local([1], "uint32")
                T.evaluate(T.ptx.bfind.u32(master[0], match_mask[0]))
                if T.cast(master[0], "int32") == lane and dst_rank >= 0:
                    T.evaluate(
                        T.ptx.atom.shared.add.s32(
                            atom_dst[0], rank_expert_count.ptr_to([dst_rank]), T.int32(1)
                        )
                    )
            T.ptx.bar.sync(T.uint32(1), T.uint32(NUM_NOTIFY_THREADS))

            # Full-grid reduction into workspace (dispatch.cuh:111-115)
            for _it in T.serial(
                0, (NUM_COUNT_SLOTS - thread_idx + NUM_NOTIFY_THREADS - 1) // NUM_NOTIFY_THREADS
            ):
                i = thread_idx + _it * NUM_NOTIFY_THREADS
                T.evaluate(
                    T.ptx.red.gpu.global_.add.u64(
                        _gptr(ws_u64, WS_NOTIFY_REDUCTION + i * 8),
                        (T.uint64(1) << T.uint64(32))
                        | T.cast(T.cast(rank_expert_count[i], "uint32"), "uint64"),
                    )
                )

            if sm_idx == 0:
                # Wait all SMs, decode, clean (dispatch.cuh:121-147)
                for _it in T.serial(
                    0, (NUM_COUNT_SLOTS - thread_idx + NUM_NOTIFY_THREADS - 1) // NUM_NOTIFY_THREADS
                ):
                    i = thread_idx + _it * NUM_NOTIFY_THREADS
                    status = T.alloc_local([1], "uint64")
                    T.evaluate(
                        _ld_volatile_u64(status[0], _gptr(ws_u64, WS_NOTIFY_REDUCTION + i * 8))
                    )
                    start_clock = T.cuda.clock64()
                    while T.cast(status[0] >> T.uint64(32), "int64") != T.int64(num_sms):
                        if T.cuda.clock64() - start_clock >= T.uint64(TIMEOUT_CYCLES):
                            T.cuda.printf(
                                "DeepEP notify (GPU reduction) timeout, rank: %d, "
                                "thread: %d, status: %d\n",
                                rank_idx,
                                thread_idx,
                                T.cast(status[0], "int32"),
                            )
                            T.cuda.trap_when_assert_failed(False)
                        T.evaluate(
                            _ld_volatile_u64(status[0], _gptr(ws_u64, WS_NOTIFY_REDUCTION + i * 8))
                        )
                    total = T.cast(T.bitwise_and(status[0], T.uint64(0xFFFFFFFF)), "int64")
                    encoded = T.cast(-total - 1, "int32")
                    rank_expert_count[i] = encoded
                    T.evaluate(
                        T.ptx.st.global_.u64(
                            _gptr(ws_u64, WS_NOTIFY_REDUCTION + i * 8), T.uint64(0)
                        )
                    )
                T.ptx.bar.sync(T.uint32(1), T.uint32(NUM_NOTIFY_THREADS))

                # Publish rank counters to every peer (dispatch.cuh:152-158)
                for _it in T.serial(
                    0, (NUM_RANKS - thread_idx + NUM_NOTIFY_THREADS - 1) // NUM_NOTIFY_THREADS
                ):
                    i = thread_idx + _it * NUM_NOTIFY_THREADS
                    T.evaluate(
                        T.ptx.st.relaxed.sys.global_.u64(
                            _gptr(_peer_u64(peer_ws_ptrs, i), WS_COUNT_RECV + rank_idx * 8),
                            T.cast(rank_expert_count[i], "uint64"),
                        )
                    )
                T.cuda.warp_sync()

                # Publish per-expert counters (dispatch.cuh:162-170)
                for _it in T.serial(
                    0, (NUM_EXPERTS - thread_idx + NUM_NOTIFY_THREADS - 1) // NUM_NOTIFY_THREADS
                ):
                    i = thread_idx + _it * NUM_NOTIFY_THREADS
                    idx = EXPERTS_PER_RANK * rank_idx + i % EXPERTS_PER_RANK
                    T.evaluate(
                        T.ptx.st.relaxed.sys.global_.u64(
                            _gptr(
                                _peer_u64(peer_ws_ptrs, i // EXPERTS_PER_RANK),
                                WS_COUNT_RECV + NUM_RANKS * 8 + idx * 8,
                            ),
                            T.cast(rank_expert_count[NUM_RANKS + i], "uint64"),
                        )
                    )
                T.ptx.bar.sync(T.uint32(1), T.uint32(NUM_NOTIFY_THREADS))

                # Wait for every peer's counts; consume and clean (dispatch.cuh:184-201)
                start_clock = T.cuda.clock64()
                for _it in T.serial(
                    0, (NUM_COUNT_SLOTS - thread_idx + NUM_NOTIFY_THREADS - 1) // NUM_NOTIFY_THREADS
                ):
                    i = thread_idx + _it * NUM_NOTIFY_THREADS
                    count = T.alloc_local([1], "int64")
                    T.evaluate(_ld_volatile_s64(count[0], _gptr(ws_u64, WS_COUNT_RECV + i * 8)))
                    decoded = -count[0] - 1
                    while decoded < 0:
                        if T.cuda.clock64() - start_clock >= T.uint64(TIMEOUT_CYCLES):
                            T.cuda.printf(
                                "DeepEP notify timeout, rank: %d, thread: %d, count: %d\n",
                                rank_idx,
                                i,
                                T.cast(count[0], "int32"),
                            )
                            T.cuda.trap_when_assert_failed(False)
                        T.evaluate(
                            T.ptx.ld.volatile.global_.s64(
                                count[0], _gptr(ws_u64, WS_COUNT_RECV + i * 8)
                            )
                        )
                        decoded = -count[0] - 1
                    T.evaluate(
                        T.ptx.st.global_.u64(_gptr(ws_u64, WS_COUNT_RECV + i * 8), T.uint64(0))
                    )
                    rank_expert_count[i] = T.cast(decoded, "int32")
                T.ptx.bar.sync(T.uint32(1), T.uint32(NUM_NOTIFY_THREADS))

                # Per-expert reduce across source ranks + align (dispatch.cuh:205-220)
                for _it in T.serial(
                    0,
                    (EXPERTS_PER_RANK - thread_idx + NUM_NOTIFY_THREADS - 1) // NUM_NOTIFY_THREADS,
                ):
                    i = thread_idx + _it * NUM_NOTIFY_THREADS
                    total = T.alloc_local([1], "int32")
                    total[0] = 0
                    for j in T.serial(0, NUM_RANKS):
                        total[0] = (
                            total[0] + rank_expert_count[NUM_RANKS + j * EXPERTS_PER_RANK + i]
                        )
                    num_unaligned[i] = total[0]
                    rank_expert_count[NUM_RANKS + i] = (
                        (total[0] + expert_alignment - 1) // expert_alignment
                    ) * expert_alignment
                T.ptx.bar.sync(T.uint32(1), T.uint32(NUM_NOTIFY_THREADS))

                # (kDoCPUSync=false: host-workspace write compiled out)

                # Prefix sums, one warp each (dispatch.cuh:234-257)
                if warp == 0:
                    # Inclusive prefix over 8 rank counts -> psum_rank[0:8)
                    value = T.Select(lane < NUM_RANKS, rank_expert_count[lane], 0)
                    scan = _warp_inclusive_sum(value, lane)
                    if lane < NUM_RANKS:
                        psum_rank[lane] = scan
                if warp == 1:
                    # Exclusive prefix over the expert counts -> psum_expert[0:EPR+1)
                    psum = T.alloc_local([1], "int32")
                    psum[0] = 0
                    for it in T.serial(0, (EXPERTS_PER_RANK + 1 + 31) // 32):
                        idx = it * 32 + lane
                        value = T.Select(
                            idx >= 1 and idx - 1 < EXPERTS_PER_RANK,
                            rank_expert_count[NUM_RANKS + idx - 1],
                            0,
                        )
                        # Out-of-range lanes contribute 0; guard the smem read.
                        if idx == 0 or idx > EXPERTS_PER_RANK:
                            value = 0
                        scan = psum[0] + _warp_inclusive_sum(value, lane)
                        if idx < EXPERTS_PER_RANK + 1:
                            psum_expert[idx] = scan
                        carry = T.alloc_local([1], "uint32")
                        T.evaluate(_shfl_idx(carry[0], scan, 31))
                        psum[0] = T.cast(carry[0], "int32")

        else:
            # =================================================================
            # DISPATCH ROLE: one channel per warp (dispatch.cuh:259-394)
            # =================================================================
            dispatch_warp_idx = warp - NUM_NOTIFY_WARPS
            tok_off = NOTIFY_SMEM_BYTES + dispatch_warp_idx * TOKEN_BYTES_SMEM
            tma_topk_idx = T.decl_buffer(
                (NUM_TOPK,),
                "int32",
                data=T.reinterpret(PointerType(PrimType("int32")), smem.ptr_to([tok_off + 14336])),
                scope="shared.dyn",
            )
            tma_topk_w = T.decl_buffer(
                (NUM_TOPK,),
                "float32",
                data=T.reinterpret(
                    PointerType(PrimType("float32")), smem.ptr_to([tok_off + 14360])
                ),
                scope="shared.dyn",
            )
            tma_src_idx = T.decl_buffer(
                (1,),
                "int32",
                data=T.reinterpret(PointerType(PrimType("int32")), smem.ptr_to([tok_off + 14384])),
                scope="shared.dyn",
            )
            tma_mbar = T.decl_buffer(
                (1,),
                "uint64",
                data=T.reinterpret(PointerType(PrimType("uint64")), smem.ptr_to([tok_off + 14432])),
                scope="shared.dyn",
            )

            phase = T.alloc_local([1], "uint32")
            phase[0] = T.uint32(0)
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1)))
                T.evaluate(T.ptx.fence.mbarrier_init.release.cluster())
            T.cuda.warp_sync()

            token_start = dispatch_warp_idx * num_sms + sm_idx
            token_stride = num_dispatch_warps * num_sms
            token_trips = T.max(
                T.int32(0), (num_tokens - token_start + token_stride - 1) // token_stride
            )
            for token_it in T.serial(0, token_trips):
                token_idx = token_start + token_it * token_stride
                # Drain prior TMA stores' SMEM reads before reusing the slot.
                # Deliberate relaxation vs dispatch.cuh:284 (full wait_group 0):
                # `.read` only waits for the TMA engine to finish reading the
                # SMEM source, letting the previous NVLink store overlap the
                # next token's load. Peer visibility is unaffected — the exit
                # path still does a full commit+wait_group(0) before the
                # tag1 grid/NVLink barriers.
                T.ptx.cp.async_.bulk.wait_group.read(0)
                T.cuda.warp_sync()

                # TMA-load the token's hidden bytes into SMEM (dispatch.cuh:288-291)
                if T.cuda.elect_sync():
                    T.evaluate(
                        T.ptx[_BULK_G2S_CHAIN](
                            smem.ptr_to([tok_off]),
                            x.ptr_to([token_idx * HIDDEN_BYTES]),
                            T.uint32(HIDDEN_BYTES),
                            tma_mbar.ptr_to([0]),
                            T.uint64(_EVICT_FIRST),
                        )
                    )
                T.cuda.warp_sync()

                # Load top-k into registers and SMEM metadata (dispatch.cuh:317-326)
                stored_dst_rank = T.alloc_local([1], "int32")
                stored_dst_rank[0] = -1
                if lane < NUM_TOPK:
                    raw = T.alloc_local([1], "int64")
                    T.evaluate(
                        T.ptx["ld.global.nc.s64"](
                            raw[0], topk_idx.ptr_to([token_idx * NUM_TOPK + lane])
                        )
                    )
                    dst_expert = T.cast(raw[0], "int32")
                    stored_dst_rank[0] = T.Select(
                        dst_expert >= 0, dst_expert // EXPERTS_PER_RANK, -1
                    )
                    tma_topk_idx[lane] = dst_expert
                    w = T.alloc_local([1], "float32")
                    T.evaluate(
                        T.ptx["ld.global.nc.f32"](
                            w[0], topk_weights.ptr_to([token_idx * NUM_TOPK + lane])
                        )
                    )
                    tma_topk_w[lane] = w[0]
                    copied_topk_idx[token_idx * NUM_TOPK + lane] = raw[0]
                T.cuda.warp_sync()

                # Source metadata; last SMEM write before the fence (dispatch.cuh:331-333)
                if T.cuda.elect_sync():
                    tma_src_idx[0] = rank_idx * num_max_tokens_per_rank + token_idx
                T.evaluate(T.ptx.fence.proxy.async_.shared__cta())
                T.cuda.warp_sync()

                # Deduplicate destination ranks and allocate slots (dispatch.cuh:337-351)
                stored_slot = T.alloc_local([1], "int32")
                stored_slot[0] = -1
                match_mask = T.alloc_local([1], "uint32")
                T.evaluate(
                    T.ptx.match.any.sync.b32(
                        match_mask[0], stored_dst_rank[0], T.uint32(0xFFFFFFFF)
                    )
                )
                master = T.alloc_local([1], "uint32")
                T.evaluate(T.ptx.bfind.u32(master[0], match_mask[0]))
                if T.cast(master[0], "int32") == lane and stored_dst_rank[0] >= 0:
                    T.evaluate(
                        T.ptx.atom.global_.add.s32(
                            stored_slot[0],
                            _gptr(ws_u64, WS_SENDER_COUNTER + stored_dst_rank[0] * 4),
                            T.int32(1),
                        )
                    )
                if lane < NUM_TOPK:
                    dst_slot_idx[token_idx * NUM_TOPK + lane] = T.Select(
                        stored_slot[0] >= 0, rank_idx * num_max_tokens_per_rank + stored_slot[0], -1
                    )
                T.cuda.warp_sync()

                # Publish expected bytes and wait TMA load arrival (dispatch.cuh:356-359)
                if T.cuda.elect_sync():
                    T.evaluate(
                        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            tma_mbar.ptr_to([0]), T.uint32(HIDDEN_BYTES)
                        )
                    )
                    T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), phase[0])
                    phase[0] = phase[0] ^ T.uint32(1)
                T.cuda.warp_sync()

                # TMA-store the whole token slot to the destination rank (dispatch.cuh:372-379)
                if stored_slot[0] >= 0:
                    T.evaluate(
                        T.ptx[_BULK_S2G_CHAIN](
                            _gptr(
                                _peer_u64(peer_buf_ptrs, stored_dst_rank[0]),
                                rank_idx * recv_region_bytes_per_rank
                                + stored_slot[0] * TOKEN_BYTES_GMEM,
                            ),
                            smem.ptr_to([tok_off]),
                            T.uint32(TOKEN_BYTES_GMEM),
                            T.uint64(_EVICT_NORMAL),
                        )
                    )
                T.evaluate(T.ptx.cp.async_.bulk.commit_group())
                T.cuda.warp_sync()

        # -------------------------------------------------------------------
        # Exit barrier (tag1): TMA flush + start grid sync (dispatch.cuh:398-400)
        # -------------------------------------------------------------------
        T.evaluate(T.ptx.cp.async_.bulk.commit_group())
        T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.warp_sync()
        grid_barrier(1)
        nvlink_barrier(3)

        # Chain the copy epilogue (dispatch.cuh:403)
        T.evaluate(T.ptx.griddepcontrol.launch_dependents())

    return deepep_dispatch.with_attr("tirx.kernel_launch_params", _launch_tags(cluster))


def _build_epilogue_kernel(
    num_sms: int, num_max_tokens_per_rank: int, expert_alignment: int, num_ranks: int
) -> Any:
    """`dispatch_copy_epilogue_impl` (frozen sketch kernel 2)."""

    NUM_RANKS = num_ranks
    EXPERTS_PER_RANK = NUM_EXPERTS // num_ranks

    num_warps = min(SMEM_TOTAL // TOKEN_BYTES_SMEM, 32)
    num_threads = num_warps * 32
    recv_region_bytes_per_rank = num_max_tokens_per_rank * TOKEN_BYTES_GMEM
    num_recv_alloc = NUM_RANKS * num_max_tokens_per_rank

    @T.prim_func
    def deepep_dispatch_copy_epilogue(
        buffer_addr: T.int64,
        psum_rank_ptr: T.handle,
        psum_expert_ptr: T.handle,
        recv_x_ptr: T.handle,
        recv_topk_idx_ptr: T.handle,
        recv_topk_weights_ptr: T.handle,
        recv_src_metadata_ptr: T.handle,
        num_unaligned_ptr: T.handle,
        num_recv_tokens: T.int32,
        rank_idx: T.int32,
    ):
        psum_rank = T.match_buffer(psum_rank_ptr, (NUM_RANKS,), "int32")
        psum_expert = T.match_buffer(psum_expert_ptr, (EXPERTS_PER_RANK,), "int32")
        recv_x = T.match_buffer(recv_x_ptr, (num_recv_alloc * HIDDEN_BYTES,), "uint8")
        recv_topk_idx = T.match_buffer(recv_topk_idx_ptr, (num_recv_alloc * NUM_TOPK,), "int64")
        recv_topk_weights = T.match_buffer(
            recv_topk_weights_ptr, (num_recv_alloc * NUM_TOPK,), "float32"
        )
        recv_src_metadata = T.match_buffer(
            recv_src_metadata_ptr, (num_recv_alloc * (2 + NUM_TOPK),), "int32"
        )
        num_unaligned = T.match_buffer(num_unaligned_ptr, (EXPERTS_PER_RANK,), "int32")

        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        smem = T.alloc_buffer([SMEM_TOTAL], "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": SMEM_TOTAL})

        sm_idx = T.cta_id([num_sms])
        thread_idx = T.thread_id([num_threads])
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

        tok_off = warp * TOKEN_BYTES_SMEM
        tma_topk_w = T.decl_buffer(
            (NUM_TOPK,),
            "float32",
            data=T.reinterpret(PointerType(PrimType("float32")), smem.ptr_to([tok_off + 14360])),
            scope="shared.dyn",
        )
        tma_src_idx = T.decl_buffer(
            (1,),
            "int32",
            data=T.reinterpret(PointerType(PrimType("int32")), smem.ptr_to([tok_off + 14384])),
            scope="shared.dyn",
        )
        tma_mbar = T.decl_buffer(
            (1,),
            "uint64",
            data=T.reinterpret(PointerType(PrimType("uint64")), smem.ptr_to([tok_off + 14432])),
            scope="shared.dyn",
        )

        phase = T.alloc_local([1], "uint32")
        phase[0] = T.uint32(0)
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1)))
            T.evaluate(T.ptx.fence.mbarrier_init.release.cluster())
        T.cuda.warp_sync()

        # Block until kernel 1 finished and all data is visible (epilogue.cuh:60)
        T.evaluate(T.ptx.griddepcontrol.wait())

        # Worst-case host count -> read the real count from the GPU prefix (epilogue.cuh:63-64)
        # Plain ld.global (no .nc): PDL visibility rule (epilogue.cuh:59).
        num_recv_reg = T.alloc_local([1], "int32")
        T.evaluate(T.ptx.ld.global_.s32(num_recv_reg[0], psum_rank.ptr_to([NUM_RANKS - 1])))
        num_recv = T.Select(
            num_recv_tokens == NUM_RANKS * num_max_tokens_per_rank, num_recv_reg[0], num_recv_tokens
        )

        # Per-warp strided loop over received tokens (epilogue.cuh:67-208)
        current_rank = T.alloc_local([1], "int32")
        rank_start = T.alloc_local([1], "int32")
        rank_end = T.alloc_local([1], "int32")
        stored_psum = T.alloc_local([1], "int32")
        current_rank[0] = -1
        rank_start[0] = 0
        rank_end[0] = 0
        stored_psum[0] = 0
        epi_stride = num_warps * num_sms
        epi_trips = T.max(T.int32(0), (num_recv - global_warp_idx + epi_stride - 1) // epi_stride)
        for epi_it in T.serial(0, epi_trips):
            i = global_warp_idx + epi_it * epi_stride
            # Locate the source rank of received token i via the inclusive prefix
            while i >= rank_end[0]:
                current_rank[0] = current_rank[0] + 1
                T.cuda.trap_when_assert_failed(current_rank[0] < NUM_RANKS)
                stored_lane = current_rank[0] % 32
                if stored_lane == 0 and current_rank[0] + lane < NUM_RANKS:
                    # Plain ld.global (no .nc): PDL visibility rule (epilogue.cuh:59).
                    T.evaluate(
                        T.ptx.ld.global_.s32(
                            stored_psum[0], psum_rank.ptr_to([current_rank[0] + lane])
                        )
                    )
                rank_start[0] = rank_end[0]
                shuffled = T.alloc_local([1], "uint32")
                T.evaluate(
                    T.ptx.shfl_sync.idx.b32(
                        shuffled[0],
                        stored_psum[0],
                        T.cast(stored_lane, "uint32"),
                        T.uint32(31),
                        T.uint32(0xFFFFFFFF),
                    )
                )
                rank_end[0] = T.cast(shuffled[0], "int32")

            token_off = (
                current_rank[0] * recv_region_bytes_per_rank
                + (i - rank_start[0]) * TOKEN_BYTES_GMEM
            )

            # Drain prior TMA stores' SMEM reads before reusing the SMEM slot.
            # Deliberate relaxation vs epilogue.cuh:84 (full wait_group 0):
            # `.read` only waits for the TMA engine to finish reading the SMEM
            # source, so the previous store's HBM write overlaps the next
            # token's TMA load. End-of-kernel store drain semantics are
            # unchanged (same as the source: no trailing full wait).
            T.ptx.cp.async_.bulk.wait_group.read(0)
            T.cuda.warp_sync()

            # TMA-load the full token slot (hidden + metadata) (epilogue.cuh:89-93)
            if T.cuda.elect_sync():
                T.evaluate(
                    T.ptx[_BULK_G2S_CHAIN](
                        smem.ptr_to([tok_off]),
                        _gptr(buf_u64, token_off),
                        T.uint32(TOKEN_BYTES_GMEM),
                        tma_mbar.ptr_to([0]),
                        T.uint64(_EVICT_FIRST),
                    )
                )
                T.evaluate(
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        tma_mbar.ptr_to([0]), T.uint32(TOKEN_BYTES_GMEM)
                    )
                )
            T.cuda.warp_sync()

            # Read target expert indices early, DIRECTLY FROM THE GMEM token slot,
            # to tolerate TMA latency (epilogue.cuh:96-100; plain ld, no .nc)
            dst_expert = T.alloc_local([1], "int32")
            dst_expert[0] = -1
            if lane < NUM_TOPK:
                T.evaluate(
                    T.ptx.ld.global_.s32(
                        dst_expert[0], _gptr(buf_u64, token_off + 14336 + lane * 4)
                    )
                )
            T.cuda.warp_sync()

            # Validate, localize, and check per-token rank uniqueness (epilogue.cuh:104-109)
            expert_start = EXPERTS_PER_RANK * rank_idx
            in_range = (
                dst_expert[0] >= expert_start and dst_expert[0] < expert_start + EXPERTS_PER_RANK
            )
            ballot = T.alloc_local([1], "uint32")
            T.evaluate(T.ptx.vote_sync.ballot.b32(ballot[0], in_range, T.uint32(0xFFFFFFFF)))
            master_lane = T.alloc_local([1], "uint32")
            T.evaluate(T.ptx.bfind.u32(master_lane[0], ballot[0]))
            dst_expert[0] = T.Select(in_range, dst_expert[0] - expert_start, -1)
            dedup_mask = T.alloc_local([1], "uint32")
            T.evaluate(T.ptx.match.any.sync.b32(dedup_mask[0], dst_expert[0], T.uint32(0xFFFFFFFF)))
            dedup_master = T.alloc_local([1], "uint32")
            T.evaluate(T.ptx.bfind.u32(dedup_master[0], dedup_mask[0]))
            T.cuda.trap_when_assert_failed(
                T.cast(dedup_master[0], "int32") == lane or dst_expert[0] == -1
            )
            if lane < NUM_TOPK:
                recv_topk_idx[i * NUM_TOPK + lane] = T.cast(dst_expert[0], "int64")
            T.cuda.warp_sync()

            # Wait TMA arrival (epilogue.cuh:126-127)
            if T.cuda.elect_sync():
                T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), phase[0])
                phase[0] = phase[0] ^ T.uint32(1)
            T.cuda.warp_sync()

            # TMA-store hidden to the output tensor (epilogue.cuh:138-142)
            if T.cuda.elect_sync():
                T.evaluate(
                    T.ptx[_BULK_S2G_CHAIN](
                        recv_x.ptr_to([i * HIDDEN_BYTES]),
                        smem.ptr_to([tok_off]),
                        T.uint32(HIDDEN_BYTES),
                        T.uint64(_EVICT_NORMAL),
                    )
                )
                T.evaluate(T.ptx.cp.async_.bulk.commit_group())
            T.cuda.warp_sync()

            # Store top-k weights (epilogue.cuh:182-184)
            if lane < NUM_TOPK:
                recv_topk_weights[i * NUM_TOPK + lane] = tma_topk_w[lane]
            T.cuda.warp_sync()

            # Write source metadata, non-cached mode (epilogue.cuh:192-201)
            if T.cuda.elect_sync():
                recv_src_metadata[i * (2 + NUM_TOPK) + 0] = tma_src_idx[0]
                recv_src_metadata[i * (2 + NUM_TOPK) + 1] = current_rank[0] * NUM_TOPK + T.cast(
                    master_lane[0], "int32"
                )
            T.cuda.warp_sync()

    return deepep_dispatch_copy_epilogue.with_attr("tirx.kernel_launch_params", _launch_tags(1))


# ---------------------------------------------------------------------------
# Module entries (tirx-kernels conventions)
# ---------------------------------------------------------------------------


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
    """Return the dispatch kernel pair (main + copy epilogue), closure-specialized."""

    if num_experts != NUM_EXPERTS or num_topk != NUM_TOPK:
        raise ValueError("config is outside the ported specialization boundary")
    if hidden * 2 != HIDDEN_BYTES:
        raise ValueError("config is outside the ported specialization boundary")
    if num_experts % world_size != 0:
        raise ValueError(f"num_experts={num_experts} not divisible by world_size={world_size}")
    if num_sms == 0:
        num_sms = get_theoretical_num_sms(world_size, num_experts, num_topk)
    # Source parity: the copy epilogue always launches with full device SMs
    # (buffer.hpp "Launch copy kernels with full SMs": device_runtime->get_num_sms()),
    # not the dispatch kernel's num_sms.
    import torch

    epilogue_num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return [
        _build_dispatch_kernel(num_sms, num_tokens, expert_alignment, world_size),
        _build_epilogue_kernel(epilogue_num_sms, num_tokens, expert_alignment, world_size),
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
    """Rank-local logical inputs, mirroring tests/elastic/test_ep.py data generation."""

    import torch

    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(seed + rank)

    # Variable per-rank token counts, same as the source test.
    local_num_tokens = max(1, num_tokens - rank)

    x = torch.randn(
        (local_num_tokens, hidden), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (local_num_tokens, num_experts), dtype=torch.float32, device=device, generator=generator
    )
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if masked_ratio > 0.0:
        mask = (
            torch.rand((local_num_tokens, num_topk), device=device, generator=generator)
            < masked_ratio
        )
        topk_idx = topk_idx.masked_fill(mask, -1)

    return {
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_tokens": local_num_tokens,
    }


def _run_worker(
    runtime: Any, modules: dict[str, Any], mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Rank-local test/bench worker (see basic/allgather_gemm.py protocol)."""

    import torch
    import torch.distributed as dist

    from .utils._buffer import SymmetricWindow

    rank = runtime.rank
    group = dist.group.WORLD
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

    # The reference runtime needs GIN disabled on this host (verified in
    # scaffolding: single-node NVLink LSA path works with EP_DISABLE_GIN=1).
    import os

    os.environ.setdefault("EP_DISABLE_GIN", "1")
    import deep_ep

    ref_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_tokens_max,
        hidden=hidden,
        num_topk=num_topk,
        # Match the source perf test (tests/elastic/test_ep.py defaults).
        prefer_overlap_with_compute=False,
        explicitly_destroy=True,
    )

    # TIRx-side symmetric window + metadata tensors.
    recv_region_bytes = world_size * num_tokens_max * TOKEN_BYTES_GMEM
    window = SymmetricWindow(group, recv_region_bytes)
    num_recv_alloc = world_size * num_tokens_max
    device = torch.device("cuda", rank)
    copied_topk_idx = torch.empty_like(topk_idx)
    psum_rank = torch.empty(world_size, dtype=torch.int32, device=device)
    psum_expert = torch.empty(num_experts // world_size + 1, dtype=torch.int32, device=device)
    num_unaligned = torch.empty(num_experts // world_size, dtype=torch.int32, device=device)
    dst_slot_idx = torch.empty((local_num_tokens, num_topk), dtype=torch.int32, device=device)
    recv_x = torch.empty((num_recv_alloc, hidden), dtype=torch.bfloat16, device=device)
    recv_topk_idx = torch.empty((num_recv_alloc, num_topk), dtype=torch.int64, device=device)
    recv_topk_weights = torch.empty((num_recv_alloc, num_topk), dtype=torch.float32, device=device)
    recv_src_metadata = torch.empty(
        (num_recv_alloc, 2 + num_topk), dtype=torch.int32, device=device
    )

    dispatch_fn = modules["dispatch"].get_function("main")
    epilogue_fn = modules["epilogue"].get_function("main")

    # Flattened views matching the kernels' 1-D buffer declarations.
    x_flat = x.view(torch.uint8).view(-1)
    topk_idx_flat = topk_idx.view(-1)
    topk_weights_flat = topk_weights.view(-1)
    copied_topk_idx_flat = copied_topk_idx.view(-1)
    dst_slot_idx_flat = dst_slot_idx.view(-1)
    recv_x_flat = recv_x.view(torch.uint8).view(-1)
    recv_topk_idx_flat = recv_topk_idx.view(-1)
    recv_topk_weights_flat = recv_topk_weights.view(-1)
    recv_src_metadata_flat = recv_src_metadata.view(-1)

    def tirx_launch() -> None:
        dispatch_fn(
            x_flat,
            topk_idx_flat,
            topk_weights_flat,
            copied_topk_idx_flat,
            psum_rank,
            psum_expert,
            num_unaligned,
            dst_slot_idx_flat,
            window.peer_ws_ptrs,
            window.peer_buf_ptrs,
            window.base_ptr,
            window.buffer_ptr,
            local_num_tokens,
            rank,
        )
        epilogue_fn(
            window.buffer_ptr,
            psum_rank,
            psum_expert[1:],
            recv_x_flat,
            recv_topk_idx_flat,
            recv_topk_weights_flat,
            recv_src_metadata_flat,
            num_unaligned,
            num_recv_alloc,
            rank,
        )

    def reference_launch():
        return ref_buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=num_tokens_max,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_sms=num_sms,
            do_cpu_sync=False,
        )

    try:
        if mode == "test":

            def _launch_and_check() -> None:
                with torch.cuda.stream(runtime.timing_stream):
                    ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, ref_handle, _ = (
                        reference_launch()
                    )
                    tirx_launch()
                runtime.device.sync(runtime.compute_stream)

                num_recv_ref = int(ref_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
                num_recv_ours = int(psum_rank[-1].item())
                assert num_recv_ref == num_recv_ours, (
                    f"rank {rank}: num_recv_tokens {num_recv_ours} != reference {num_recv_ref}"
                )
                assert torch.equal(ref_handle.psum_num_recv_tokens_per_scaleup_rank, psum_rank), (
                    f"rank {rank}: psum_num_recv_tokens_per_scaleup_rank mismatch"
                )
                assert torch.equal(ref_handle.psum_num_recv_tokens_per_expert, psum_expert[1:]), (
                    f"rank {rank}: psum_num_recv_tokens_per_expert mismatch"
                )
                assert torch.equal(
                    ref_handle.num_unaligned_recv_tokens_per_expert, num_unaligned
                ), f"rank {rank}: num_unaligned_recv_tokens_per_expert mismatch"
                assert torch.equal(copied_topk_idx, topk_idx), (
                    f"rank {rank}: copied_topk_idx mismatch"
                )

                # The receive order inside each source-rank segment is atomic-order
                # dependent in both implementations; compare up to permutation by
                # sorting on the unique src_token_global_idx.
                ref_meta = ref_handle.recv_src_metadata[:num_recv_ref]
                ours_meta = recv_src_metadata[:num_recv_ours]
                ref_order = torch.argsort(ref_meta[:, 0])
                ours_order = torch.argsort(ours_meta[:, 0])
                assert torch.equal(ref_meta[ref_order, 0], ours_meta[ours_order, 0]), (
                    f"rank {rank}: src_token_global_idx set mismatch"
                )
                assert torch.equal(ref_meta[ref_order, 1], ours_meta[ours_order, 1]), (
                    f"rank {rank}: recv_src_metadata[:, 1] mismatch"
                )
                assert torch.equal(ref_recv_x[:num_recv_ref][ref_order], recv_x[ours_order]), (
                    f"rank {rank}: recv_x mismatch"
                )
                assert torch.equal(
                    ref_recv_topk_idx[:num_recv_ref][ref_order], recv_topk_idx[ours_order]
                ), f"rank {rank}: recv_topk_idx mismatch"
                assert torch.equal(
                    ref_recv_topk_weights[:num_recv_ref][ref_order], recv_topk_weights[ours_order]
                ), f"rank {rank}: recv_topk_weights mismatch"

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
                raise AssertionError(f"deepep_dispatch check failed on some rank: {check_error}")
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
    dispatch_kernel, epilogue_kernel = get_kernel(**config, num_sms=num_sms)
    run_distributed(
        {"dispatch": dispatch_kernel, "epilogue": epilogue_kernel},
        world_size=config["world_size"],
        worker=_run_worker,
        mode="test",
        worker_kwargs={**config, "num_sms": num_sms},
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

    from .utils._runtime import run_distributed

    if timer is not None and timer != "kineto":
        raise ValueError(f"deepep_dispatch is distributed and supports only kineto, got {timer}")
    if warmup is not None or repeat is not None:
        raise ValueError("kineto uses fixed iteration counts and rejects warmup/repeat overrides")
    config = dict(kwargs)
    if args:
        raise TypeError(f"unexpected positional arguments: {args}")
    num_sms = _resolve_num_sms(config)
    dispatch_kernel, epilogue_kernel = get_kernel(**config, num_sms=num_sms)
    return run_distributed(
        {"dispatch": dispatch_kernel, "epilogue": epilogue_kernel},
        world_size=config["world_size"],
        worker=_run_worker,
        mode="bench",
        worker_kwargs={**config, "num_sms": num_sms, "rounds": rounds, "cooldown_s": cooldown_s},
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
