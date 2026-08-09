from __future__ import annotations

import math
import os

import torch
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState
from tvm.tirx.lang.smem_desc import SmemDescriptor
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

_TMA_G2S_2D = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_S2G_2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
_TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
_TMA_EVICT_NORMAL = 0x1000000000000000
_TCGEN05_CP_2SM = "tcgen05.cp.cta_group::2.32x128b.warpx4"
_MMA_FP8_2SM = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
_TMEM_LD_32X32_X8 = "tcgen05.ld.sync.aligned.32x32b.x8.b32"
_TMEM_LD_16X256_X1 = "tcgen05.ld.sync.aligned.16x256b.x1.b32"


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _replace_smem_desc_addr(desc, smem_ptr):
    start_addr = T.cast(
        T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)
        ),
        "uint64",
    )
    return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)


def _swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable swizzle mode")


def _deepgemm_num_stages(
    *, swap_ab: bool, block_m: int, block_n: int, load_block_m: int, load_block_n: int
) -> int:
    """Match DeepGEMM SM100 shared-memory stage count for FP8 normal GEMM."""

    block_k = 128
    swizzle_cd = _swizzle_mode(block_n, 2)
    if swap_ab:
        smem_cd = 16 * block_n * 2 * 2
    else:
        smem_cd = min(block_m, 128) * swizzle_cd * 2
    smem_barriers = 32 * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4
    smem_per_stage = (
        load_block_m * block_k
        + load_block_n * block_k
        + _align(block_m, 128) * 4
        + _align(block_n, 128) * 4
    )
    smem_capacity = 232448
    num_stages = (smem_capacity - smem_cd - smem_barriers - smem_tmem_ptr) // smem_per_stage
    return min(num_stages, 32)


def _choose_deepgemm_config(M: int, N: int, K: int) -> tuple[bool, int, int, int, int, int]:
    """Match DeepGEMM's SM100 FP8 normal-GEMM layout heuristic."""

    sm_count = 148
    candidates: list[tuple[int, int, int, int, int, int, bool, int, int, int, int, int]] = []
    for swap_ab in (False, True):
        if swap_ab:
            block_m_candidates = range(16, 257, 16)
            block_n_candidates = [128]
            cluster_candidates = [(1, 2)]
        else:
            block_m_candidates = [32] if M <= 32 else [64] if M <= 64 else [128]
            max_block_n = 128 if K <= 256 else 256
            block_n_candidates = [16, *range(32, max_block_n + 1, 32)]
            cluster_candidates = [(2, 1)]

        for cluster_m, cluster_n in cluster_candidates:
            if sm_count % (cluster_m * cluster_n) != 0:
                continue
            for block_m in block_m_candidates:
                load_block_m = block_m // cluster_n
                if load_block_m % 8 != 0:
                    continue
                if math.ceil(M / block_m) % cluster_m != 0:
                    continue
                for block_n in block_n_candidates:
                    load_block_n = block_n // cluster_m
                    if load_block_n % 8 != 0:
                        continue
                    if math.ceil(N / block_n) % cluster_n != 0:
                        continue
                    sf_block_m = _align(block_m, 128)
                    sf_block_n = _align(block_n, 128)
                    umma_n = block_m if swap_ab else block_n
                    if 2 * umma_n + sf_block_m // 32 + sf_block_n // 32 > 512:
                        continue
                    num_blocks = math.ceil(M / block_m) * math.ceil(N / block_n)
                    waves = math.ceil(num_blocks / sm_count)
                    last_wave_util = num_blocks % sm_count or sm_count
                    stages = _deepgemm_num_stages(
                        swap_ab=swap_ab,
                        block_m=block_m,
                        block_n=block_n,
                        load_block_m=load_block_m,
                        load_block_n=load_block_n,
                    )
                    candidates.append(
                        (
                            0 if waves == 1 else 1,
                            -cluster_m * cluster_n,
                            waves,
                            -last_wave_util,
                            block_m + block_n,
                            block_m * block_n,
                            swap_ab,
                            block_m,
                            block_n,
                            stages,
                            cluster_m,
                            cluster_n,
                        )
                    )

    if not candidates:
        raise RuntimeError(f"no DeepGEMM config candidate for M={M}, N={N}, K={K}")
    _, _, _, _, _, _, swap_ab, block_m, block_n, stages, cluster_m, cluster_n = min(candidates)
    return swap_ab, block_m, block_n, stages, cluster_m, cluster_n


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _dg_scale_view(tirx_scale_pack: torch.Tensor, mn: int) -> torch.Tensor:
    packed_k, physical_mn = tirx_scale_pack.shape
    if physical_mn != mn:
        raise ValueError(
            f"packed scale shape mismatch: expected physical MN {mn}, got {physical_mn}"
        )
    scale_i32 = tirx_scale_pack.view(torch.int32)
    return torch.as_strided(scale_i32, size=(mn, packed_k), stride=(1, physical_mn))


def prepare_data(M: int, N: int, K: int):
    A_origin = torch.randn((M, K), dtype=torch.float32)
    B_origin = torch.randn((N, K), dtype=torch.float32)
    A_fp8, sfa = per_token_cast_to_fp8(A_origin, use_ue8m0=True)
    B_fp8, sfb = per_block_cast_to_fp8(B_origin, use_ue8m0=True)
    sfa_uint8 = (sfa.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    sfb_uint8 = (sfb.view(torch.int32) >> 23).to(torch.uint8).contiguous().repeat(128, 1)[:N, :]
    sfa_pack = sfa_uint8.view(torch.uint32).T.contiguous()
    sfb_pack = sfb_uint8.view(torch.uint32).T.contiguous()
    A_fp8_de = A_fp8.to(torch.float32)
    B_fp8_de = B_fp8.to(torch.float32)
    A_de = (
        A_fp8_de.reshape(M, K // 128, 128) * 2.0 ** (sfa_uint8[:, :, None].to(torch.float32) - 127)
    ).reshape(M, K)
    B_de = (
        B_fp8_de.reshape(N, K // 128, 128) * 2.0 ** (sfb_uint8[:, :, None].to(torch.float32) - 127)
    ).reshape(N, K)
    C_ref = torch.matmul(A_de, B_de.T).to(torch.bfloat16)
    return (
        A_fp8.to("cuda"),
        B_fp8.to("cuda"),
        sfa.to("cuda"),
        sfb.to("cuda"),
        sfa_pack.to("cuda"),
        sfb_pack.to("cuda"),
        C_ref.to("cuda"),
        A_origin.to("cuda"),
        B_origin.to("cuda"),
    )


@T.jit
def _kernel(
    A: T.Buffer((M, K), "float8_e4m3fn"),
    B: T.Buffer((N, K), "float8_e4m3fn"),
    SFA: T.Buffer((math.ceil(K / 128) // 4, M), "uint32"),
    SFB: T.Buffer((math.ceil(K / 128) // 4, N), "uint32"),
    D: T.Buffer((M, N), "bfloat16"),
    *,
    # problem size
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    # block + cluster layout
    SWAP_AB: T.constexpr,
    DG_BLOCK_M: T.constexpr,
    DG_BLOCK_N: T.constexpr,
    LOGICAL_M_CLUSTER: T.constexpr,
    LOGICAL_N_CLUSTER: T.constexpr,
    # tile / MMA sizes
    BLK_K: T.constexpr = 128,
    MMA_K: T.constexpr = 32,
    EPI_TILE: T.constexpr = 32,
    TMEM_LD_SIZE: T.constexpr = 8,
    # pipeline depths
    SMEM_DEPTH: T.constexpr,
    TMEM_DEPTH: T.constexpr = 2,
    # warp / SM / scheduler
    WG_NUMBER: T.constexpr = 2,
    SM_NUMBER: T.constexpr = 148,
    TILE_GROUPS_ROW_SIZE: T.constexpr = 16,
):
    CTA_GROUP = T.meta_var(LOGICAL_M_CLUSTER * LOGICAL_N_CLUSTER)
    M_CLUSTER = T.meta_var(CTA_GROUP)
    N_CLUSTER = T.meta_var(1)
    MMA_N = T.meta_var(DG_BLOCK_M if SWAP_AB else DG_BLOCK_N)
    BLK_M = T.meta_var(DG_BLOCK_M // LOGICAL_N_CLUSTER if SWAP_AB else DG_BLOCK_M)
    BLK_N = T.meta_var(DG_BLOCK_N if SWAP_AB else DG_BLOCK_N // LOGICAL_M_CLUSTER)
    BLK_SFA = T.meta_var(_align(DG_BLOCK_M, 128))
    BLK_SFB = T.meta_var(_align(DG_BLOCK_N, 128))
    K_TILES = T.meta_var(K // BLK_K)
    SFA_post_layout = T.meta_var(
        T.TileLayout(T.S[(SMEM_DEPTH, BLK_SFA // 128, 4, 32) : (BLK_SFA, 128, 1, 4)])
    )
    SFB_post_layout = T.meta_var(
        T.TileLayout(T.S[(SMEM_DEPTH, BLK_SFB // 128, 4, 32) : (BLK_SFB, 128, 1, 4)])
    )
    K_ITERS = T.meta_var(BLK_K // MMA_K)
    SFA_smem_fp8_layout = T.meta_var(SFA_post_layout.unpack(4).broadcast(K_ITERS))
    SFB_smem_fp8_layout = T.meta_var(SFB_post_layout.unpack(4).broadcast(K_ITERS))
    AB_bytes = T.meta_var(BLK_M * BLK_K + BLK_N * BLK_K)  # fp8 A+B operands: 1 byte/elem
    SFAB_bytes = T.meta_var((DG_BLOCK_M + DG_BLOCK_N) * 4)  # SF packed as uint32: 4 B
    SCHED_M_NUM = T.meta_var(math.ceil(N / DG_BLOCK_N) if SWAP_AB else math.ceil(M / DG_BLOCK_M))
    SCHED_N_NUM = T.meta_var(math.ceil(M / DG_BLOCK_M) if SWAP_AB else math.ceil(N / DG_BLOCK_N))
    D_SMEM_M = T.meta_var(16 if SWAP_AB else BLK_M)
    D_SMEM_N = T.meta_var(DG_BLOCK_N if SWAP_AB else EPI_TILE)
    D_SWIZZLE = T.meta_var(
        SwizzleMode.SWIZZLE_128B_ATOM if SWAP_AB else SwizzleMode.SWIZZLE_64B_ATOM
    )
    D_TMA_TILE_M = T.meta_var(16 if SWAP_AB else DG_BLOCK_M)
    D_TMA_TILE_N = T.meta_var(DG_BLOCK_N if SWAP_AB else EPI_TILE)
    D_TMA_SWIZZLE = T.meta_var(3 if SWAP_AB else 2)

    A_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    B_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    SFA_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    SFB_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    D_tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        A_tensor_map,
        "float8_e4m3fn",
        2,
        A.data,
        K,
        M,
        K,
        BLK_K,
        BLK_M,
        1,
        1,
        0,
        3,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        B_tensor_map,
        "float8_e4m3fn",
        2,
        B.data,
        K,
        N,
        K,
        BLK_K,
        BLK_N,
        1,
        1,
        0,
        3,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        SFA_tensor_map,
        "uint32",
        2,
        SFA.data,
        M,
        K // 512,
        M * 4,
        DG_BLOCK_M,
        1,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        SFB_tensor_map,
        "uint32",
        2,
        SFB.data,
        N,
        K // 512,
        N * 4,
        DG_BLOCK_N,
        1,
        1,
        1,
        0,
        0,
        2,
        0,
    )
    if SWAP_AB:
        # Split the contiguous N axis into 64-element atoms.  The resulting 3-D
        # descriptor is the legal 128-byte/swizzle-128 encoding of a 16x128 BF16
        # store tile used by the reference dispatcher.
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            D_tensor_map,
            "bfloat16",
            3,
            D.data,
            64,
            M,
            N // 64,
            N * 2,
            128,
            64,
            16,
            2,
            1,
            1,
            1,
            0,
            3,
            2,
            0,
        )
    else:
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            D_tensor_map,
            "bfloat16",
            2,
            D.data,
            N,
            M,
            N * 2,
            D_TMA_TILE_N,
            D_TMA_TILE_M,
            1,
            1,
            0,
            D_TMA_SWIZZLE,
            2,
            0,
        )
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    cbx, cby = T.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
    cluster_rank = T.cuda.mov_sreg(32, "cluster_ctarank")
    bx = T.cta_id([SM_NUMBER])
    wg_id = T.warpgroup_id([WG_NUMBER])
    warp_id = T.warp_id_in_wg([4])
    tid_in_wg = T.thread_id_in_wg([128])
    lane_id = T.lane_id([32])
    if (wg_id == 0) & (warp_id == 0):
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(A_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(B_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(SFA_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(SFB_tensor_map)))
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(D_tensor_map)))
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    barrier_leader = (wg_id == 0) & (warp_id == 1) & (T.cuda.elect_sync() != T.uint32(0))
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=512,
        cta_group=CTA_GROUP,
        tmem_addr=tmem_addr,
        alloc_warp=2,
        dealloc_warp=0,
        sync_after_alloc=False,
    )
    smem_pipe = Pipeline(pool, SMEM_DEPTH, full="tma", empty="tcgen05", leader=barrier_leader)
    trans_done = MBarrier(pool, SMEM_DEPTH, leader=barrier_leader)
    trans_done.init(CTA_GROUP * 32)
    tmem_pipe = Pipeline(
        pool,
        TMEM_DEPTH,
        full="tcgen05",
        empty="mbar",
        init_empty=CTA_GROUP * 128,
        empty_phase_offset=1,
        leader=barrier_leader,
    )
    acc_buf = tmem_pool.alloc_tcgen05_mma_D(
        (128, TMEM_DEPTH * MMA_N), "float32", M=128 * CTA_GROUP, cta_group=CTA_GROUP
    )
    SFA_tmem = tmem_pool.alloc_sf(
        (TMEM_DEPTH, BLK_SFA, 4 * K_ITERS), "float8_e8m0fnu", sf_per_mma=1, sf_reuse=K_ITERS
    )
    SFB_tmem = tmem_pool.alloc_sf(
        (TMEM_DEPTH, BLK_SFB, 4 * K_ITERS), "float8_e8m0fnu", sf_per_mma=1, sf_reuse=K_ITERS
    )
    pool.move_base_to(1024)
    A_smem = pool.alloc_tcgen05_mma_AB((SMEM_DEPTH, BLK_M, BLK_K), "float8_e4m3fn")
    B_smem = pool.alloc_tcgen05_mma_AB((SMEM_DEPTH, BLK_N, BLK_K), "float8_e4m3fn")
    D_smem = pool.alloc_tcgen05_mma_AB(
        (TMEM_DEPTH, D_SMEM_M, D_SMEM_N), "bfloat16", swizzle_mode=D_SWIZZLE
    )
    SFA_smem = pool.alloc((SMEM_DEPTH, BLK_SFA), "uint32")
    SFB_smem = pool.alloc((SMEM_DEPTH, BLK_SFB), "uint32")
    pool.commit()
    if barrier_leader:
        T.ptx.fence.mbarrier_init.release.cluster()
    stage: T.int32
    tile_scheduler = ClusterPersistentScheduler2D(
        "tile_scheduler",
        num_m_tiles=SCHED_M_NUM,
        num_n_tiles=SCHED_N_NUM,
        l2_group_size=TILE_GROUPS_ROW_SIZE,
        num_clusters=SM_NUMBER,
    )
    tile_scheduler.init(bx)
    tmem_pool.commit()
    T.cuda.cluster_sync()
    T.evaluate(T.ptx.griddepcontrol.wait())
    tmem_allocated: T.uint32
    T.ptx.ld.shared.u32(tmem_allocated, tmem_addr.ptr_to([0]))
    T.cuda.trap_when_assert_failed(tmem_allocated == T.uint32(0))

    m_idx = T.meta_var(tile_scheduler.n_idx if SWAP_AB else tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.m_idx if SWAP_AB else tile_scheduler.n_idx)

    if wg_id == 0:
        if warp_id == 0:
            tma_cur = PipelineState(SMEM_DEPTH, 1)
            a_m = T.meta_var(
                m_idx * DG_BLOCK_M + cluster_rank * BLK_M if SWAP_AB else m_idx * DG_BLOCK_M
            )
            sf_m = T.meta_var(m_idx * DG_BLOCK_M)
            b_n = T.meta_var(
                n_idx * DG_BLOCK_N if SWAP_AB else n_idx * DG_BLOCK_N + cluster_rank * BLK_N
            )
            sf_n = T.meta_var(n_idx * DG_BLOCK_N)

            @T.inline
            def tma_load(k_tile):
                smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                stage = tma_cur.stage
                k = T.meta_var(k_tile * BLK_K)
                T.evaluate(
                    T.ptx[_TMA_G2S_2D](
                        A_smem.ptr_to([stage, 0, 0]),
                        T.address_of(A_tensor_map),
                        T.cast(k, "int32"),
                        T.cast(a_m, "int32"),
                        smem_pipe.full.ptr_to([stage]),
                        T.uint64(_TMA_EVICT_NORMAL),
                    )
                )
                T.evaluate(
                    T.ptx[_TMA_G2S_2D](
                        B_smem.ptr_to([stage, 0, 0]),
                        T.address_of(B_tensor_map),
                        T.cast(k, "int32"),
                        T.cast(b_n, "int32"),
                        smem_pipe.full.ptr_to([stage]),
                        T.uint64(_TMA_EVICT_NORMAL),
                    )
                )
                if k_tile % 4 == 0:
                    T.evaluate(
                        T.ptx[_TMA_G2S_2D](
                            SFA_smem.ptr_to([stage, 0]),
                            T.address_of(SFA_tensor_map),
                            T.cast(sf_m, "int32"),
                            T.cast(k_tile // 4, "int32"),
                            smem_pipe.full.ptr_to([stage]),
                            T.uint64(_TMA_EVICT_NORMAL),
                        )
                    )
                    T.evaluate(
                        T.ptx[_TMA_G2S_2D](
                            SFB_smem.ptr_to([stage, 0]),
                            T.address_of(SFB_tensor_map),
                            T.cast(sf_n, "int32"),
                            T.cast(k_tile // 4, "int32"),
                            smem_pipe.full.ptr_to([stage]),
                            T.uint64(_TMA_EVICT_NORMAL),
                        )
                    )

                smem_pipe.full.arrive(
                    tma_cur.stage,
                    tx_count=T.if_then_else(k_tile % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes),
                )

            @T.inline
            def tma_iter():
                for k_tile in T.serial(K_TILES):
                    tma_load(k_tile)
                    tma_cur.advance()

            if T.cuda.elect_sync():
                while tile_scheduler.valid():
                    tma_iter()
                    tile_scheduler.next_tile()
        elif warp_id == 2:
            trans_state = PipelineState(SMEM_DEPTH, 0)

            @T.inline
            def transpose(ks, k_tile):
                smem_pipe.full.wait(ks, trans_state.phase)
                if k_tile % 4 == 0:
                    sfa_regs = T.alloc_local((BLK_SFA // 32,), "uint32")
                    for r in T.unroll(BLK_SFA // 32):
                        quad = T.meta_var(
                            T.bitwise_xor(r, T.bitwise_and(T.shift_right(lane_id, 3), 3))
                        )
                        src = T.meta_var(
                            T.ptr_byte_offset(
                                SFA_smem.ptr_to([ks, 0]), (quad * 32 + lane_id) * 4, "uint32"
                            )
                        )
                        T.ptx.ld.shared.b32(sfa_regs[r], src)
                    T.cuda.warp_sync()
                    for r in T.unroll(BLK_SFA // 32):
                        quad = T.meta_var(
                            T.bitwise_xor(r, T.bitwise_and(T.shift_right(lane_id, 3), 3))
                        )
                        post = T.meta_var(quad // 4 * 128 + quad % 4 + lane_id * 4)
                        dst = T.meta_var(
                            T.ptr_byte_offset(SFA_smem.ptr_to([ks, 0]), post * 4, "uint32")
                        )
                        T.ptx.st.shared.b32(dst, sfa_regs[r])
                    T.cuda.warp_sync()
                    sfb_regs = T.alloc_local((BLK_SFB // 32,), "uint32")
                    for r in T.unroll(BLK_SFB // 32):
                        quad = T.meta_var(
                            T.bitwise_xor(r, T.bitwise_and(T.shift_right(lane_id, 3), 3))
                        )
                        src = T.meta_var(
                            T.ptr_byte_offset(
                                SFB_smem.ptr_to([ks, 0]), (quad * 32 + lane_id) * 4, "uint32"
                            )
                        )
                        T.ptx.ld.shared.b32(sfb_regs[r], src)
                    T.cuda.warp_sync()
                    for r in T.unroll(BLK_SFB // 32):
                        quad = T.meta_var(
                            T.bitwise_xor(r, T.bitwise_and(T.shift_right(lane_id, 3), 3))
                        )
                        post = T.meta_var(quad // 4 * 128 + quad % 4 + lane_id * 4)
                        dst = T.meta_var(
                            T.ptr_byte_offset(SFB_smem.ptr_to([ks, 0]), post * 4, "uint32")
                        )
                        T.ptx.st.shared.b32(dst, sfb_regs[r])
                    T.cuda.warp_sync()
                    T.ptx.fence.proxy.async_.shared__cta()
                trans_done.arrive(ks, remote=0)

            @T.inline
            def trans_iter():
                if K_TILES <= 16:
                    # Short-K kernels fully materialize the pipeline body in the reference
                    # codegen, avoiding loop-control overhead altogether.
                    for k_tile in T.unroll(K_TILES):
                        transpose(trans_state.stage, k_tile)
                        trans_state.advance()
                else:
                    # Long-K kernels use a four-way body.  Keeping this grouping explicit
                    # makes the wait/arrive cadence independent of nvcc's heuristic.
                    for k_group in T.serial(K_TILES // 4, unroll=False):
                        for k_inner in T.unroll(4):
                            k_tile = T.meta_var(k_group * 4 + k_inner)
                            transpose(trans_state.stage, k_tile)
                            trans_state.advance()

            while tile_scheduler.valid():
                trans_iter()
                tile_scheduler.next_tile()
        elif warp_id == 1 and cluster_rank == 0:
            SFA_smem_fp8 = SFA_smem.view("float8_e8m0fnu").view(
                SMEM_DEPTH, BLK_SFA, 4 * K_ITERS, layout=SFA_smem_fp8_layout
            )
            SFB_smem_fp8 = SFB_smem.view("float8_e8m0fnu").view(
                SMEM_DEPTH, BLK_SFB, 4 * K_ITERS, layout=SFB_smem_fp8_layout
            )
            desc_a = T.meta_var(SmemDescriptor())
            desc_b = T.meta_var(SmemDescriptor())
            desc_sf = T.meta_var(SmemDescriptor())
            if SWAP_AB:
                desc_a.init(B_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
                desc_b.init(A_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
            else:
                desc_a.init(A_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
                desc_b.init(B_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3)
            desc_sf.init(T.reinterpret("handle", T.uint64(0)), ldo=0, sdo=8, swizzle=0)
            MMA_A_ROWS = T.meta_var(BLK_N if SWAP_AB else BLK_M)
            MMA_B_ROWS = T.meta_var(BLK_M if SWAP_AB else BLK_N)
            MMA_SFA_BASE = T.meta_var(
                SFB_tmem.allocated_addr[0] if SWAP_AB else SFA_tmem.allocated_addr[0]
            )
            MMA_SFB_BASE = T.meta_var(
                SFA_tmem.allocated_addr[0] if SWAP_AB else SFB_tmem.allocated_addr[0]
            )
            MMA_SFA_STAGE_COLS = T.meta_var(BLK_SFB // 32 if SWAP_AB else BLK_SFA // 32)
            MMA_SFB_STAGE_COLS = T.meta_var(BLK_SFA // 32 if SWAP_AB else BLK_SFB // 32)
            tmem_idx: T.int32
            tmem_phase: T.int32
            mma_state = PipelineState(SMEM_DEPTH, 0)
            accum: T.int32

            @T.inline
            def mma(ks, k_tile):
                trans_done.wait(ks, mma_state.phase)
                if k_tile % 4 == 0:
                    for sf_chunk in T.unroll(BLK_SFA // 128):
                        sfa_desc = T.meta_var(
                            _replace_smem_desc_addr(
                                desc_sf.desc, SFA_smem_fp8.ptr_to([ks, sf_chunk * 128, 0])
                            )
                        )
                        T.ptx[_TCGEN05_CP_2SM](
                            T.cast(
                                SFA_tmem.allocated_addr[0]
                                + tmem_idx * (BLK_SFA // 32)
                                + sf_chunk * 4,
                                "uint32",
                            ),
                            sfa_desc,
                        )
                    for sf_chunk in T.unroll(BLK_SFB // 128):
                        sfb_desc = T.meta_var(
                            _replace_smem_desc_addr(
                                desc_sf.desc, SFB_smem_fp8.ptr_to([ks, sf_chunk * 128, 0])
                            )
                        )
                        T.ptx[_TCGEN05_CP_2SM](
                            T.cast(
                                SFB_tmem.allocated_addr[0]
                                + tmem_idx * (BLK_SFB // 32)
                                + sf_chunk * 4,
                                "uint32",
                            ),
                            sfb_desc,
                        )
                desc_i: T.uint32
                T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                    T.address_of(desc_i),
                    d_dtype="float32",
                    a_dtype="float8_e4m3fn",
                    b_dtype="float8_e4m3fn",
                    sfa_dtype="float8_e8m0fnu",
                    sfb_dtype="float8_e8m0fnu",
                    sfa_tmem_addr=MMA_SFA_BASE + tmem_idx * MMA_SFA_STAGE_COLS,
                    sfb_tmem_addr=MMA_SFB_BASE + tmem_idx * MMA_SFB_STAGE_COLS,
                    M=128 * CTA_GROUP,
                    N=MMA_N,
                    K=MMA_K,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=CTA_GROUP,
                )
                for ki in T.unroll(K_ITERS):
                    T.cuda.runtime_instr_desc(T.address_of(desc_i), k_tile % 4)
                    desc_a_ki = T.meta_var(
                        desc_a.add_16B_offset((ks * MMA_A_ROWS * BLK_K + ki * MMA_K) // 16)
                    )
                    desc_b_ki = T.meta_var(
                        desc_b.add_16B_offset((ks * MMA_B_ROWS * BLK_K + ki * MMA_K) // 16)
                    )
                    T.ptx[_MMA_FP8_2SM](
                        T.cast(acc_buf.allocated_addr[0] + tmem_idx * MMA_N, "uint32"),
                        desc_a_ki,
                        desc_b_ki,
                        desc_i,
                        T.cast(MMA_SFA_BASE + tmem_idx * MMA_SFA_STAGE_COLS, "uint32"),
                        T.cast(MMA_SFB_BASE + tmem_idx * MMA_SFB_STAGE_COLS, "uint32"),
                        T.Or(ki != 0, T.cast(accum, "bool")),
                    )
                accum = 1
                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

            @T.inline
            def mma_iter():
                if T.cuda.elect_sync():
                    tmem_idx = tile_scheduler.tile_idx % TMEM_DEPTH
                    tmem_phase = tile_scheduler.tile_idx // TMEM_DEPTH & 1
                    tmem_pipe.empty.wait(tmem_idx, tmem_phase)
                    accum = 0
                    for k_tile in T.serial(K_TILES):
                        mma(mma_state.stage, k_tile)
                        mma_state.advance()
                    tmem_pipe.full.arrive(tmem_idx, cta_group=CTA_GROUP, cta_mask=3)

            while tile_scheduler.valid():
                mma_iter()
                tile_scheduler.next_tile()
    elif wg_id == 1:
        tmem_idx: T.int32
        tmem_phase: T.int32

        # Stream acc -> D_smem -> TMA in EPI-wide slices. SWAP_AB only changes the
        # acc -> D_smem step (stmatrix transpose vs straight copy) and the tiling.
        EPI = T.meta_var(16 if SWAP_AB else EPI_TILE)
        STORE_TILES = T.meta_var(MMA_N // EPI)

        @T.inline
        def epilogue():
            swap_frag = T.alloc_local((8,), "float32")
            swap_bf16 = T.alloc_local((4,), "uint32", align=16)
            for ot in T.unroll(STORE_TILES):
                store_iter: T.let = tile_scheduler.tile_idx * STORE_TILES + ot
                stage = store_iter % TMEM_DEPTH
                if store_iter >= TMEM_DEPTH:
                    if warp_id == 0:
                        T.ptx.cp.async_.bulk.wait_group(TMEM_DEPTH - 1)
                    T.cuda.warpgroup_sync(10)
                if SWAP_AB:
                    for atom_m in T.unroll(2):
                        col_st: T.let = ot * 16 + atom_m * 8
                        for slab in T.unroll(2):
                            reg_base = T.meta_var(slab * 4)
                            T.ptx[_TMEM_LD_16X256_X1](
                                swap_frag[reg_base],
                                swap_frag[reg_base + 1],
                                swap_frag[reg_base + 2],
                                swap_frag[reg_base + 3],
                                T.cuda.get_tmem_addr(
                                    acc_buf.allocated_addr[0], slab * 16, tmem_idx * MMA_N + col_st
                                ),
                            )
                        T.ptx.tcgen05.wait__ld.sync.aligned()
                        for pair in T.unroll(4):
                            T.ptx.cvt.rn.bf16x2.f32(
                                swap_bf16[pair], swap_frag[pair * 2 + 1], swap_frag[pair * 2]
                            )
                        row = T.meta_var(lane_id % 8)
                        col = T.meta_var(warp_id % 2 * 4 + lane_id // 8)
                        smem_off = T.meta_var(
                            stage * D_SMEM_M * D_SMEM_N
                            + warp_id // 2 * 16 * 64
                            + atom_m * 8 * 64
                            + row * 64
                            + T.bitwise_xor(col, row) * 8
                        )
                        smem_ptr = T.meta_var(
                            T.ptr_byte_offset(D_smem.ptr_to([0, 0, 0]), smem_off * 2, "bfloat16")
                        )
                        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem_ptr, swap_bf16[0], swap_bf16[1], swap_bf16[2], swap_bf16[3]
                        )
                else:
                    for ki in T.unroll(EPI_TILE // TMEM_LD_SIZE):
                        Dreg = T.alloc_local((TMEM_LD_SIZE,), "float32")
                        acc_n = T.meta_var(ot * EPI_TILE + ki * TMEM_LD_SIZE)
                        T.ptx[_TMEM_LD_32X32_X8](
                            Dreg[0],
                            Dreg[1],
                            Dreg[2],
                            Dreg[3],
                            Dreg[4],
                            Dreg[5],
                            Dreg[6],
                            Dreg[7],
                            T.cast(acc_buf.allocated_addr[0] + tmem_idx * MMA_N + acc_n, "uint32"),
                        )
                        T.ptx.tcgen05.wait__ld.sync.aligned()
                        Dreg_bf16 = T.alloc_local((TMEM_LD_SIZE // 2,), "uint32", align=16)
                        for pair in T.unroll(TMEM_LD_SIZE // 2):
                            T.ptx.cvt.rn.bf16x2.f32(
                                Dreg_bf16[pair], Dreg[pair * 2 + 1], Dreg[pair * 2]
                            )
                        T.ptx.st.shared.v4.u32(
                            D_smem.ptr_to([stage, tid_in_wg, ki * TMEM_LD_SIZE]),
                            Dreg_bf16[0],
                            Dreg_bf16[1],
                            Dreg_bf16[2],
                            Dreg_bf16[3],
                        )
                if ot == STORE_TILES - 1:
                    tmem_pipe.empty.arrive(tmem_idx, remote=0)
                T.ptx.fence.proxy.async_.shared__cta()
                T.cuda.warpgroup_sync(10)
                d_m: T.let = m_idx * DG_BLOCK_M + (ot * 16 if SWAP_AB else 0)
                d_n: T.let = n_idx * DG_BLOCK_N + (0 if SWAP_AB else ot * EPI_TILE)
                if warp_id == 0:
                    if T.cuda.elect_sync():
                        if SWAP_AB:
                            T.evaluate(
                                T.ptx[_TMA_S2G_3D](
                                    T.address_of(D_tensor_map),
                                    T.int32(0),
                                    T.cast(d_m, "int32"),
                                    T.cast(d_n // 64, "int32"),
                                    D_smem.ptr_to([stage, 0, 0]),
                                )
                            )
                        else:
                            T.evaluate(
                                T.ptx[_TMA_S2G_2D](
                                    T.address_of(D_tensor_map),
                                    T.cast(d_n, "int32"),
                                    T.cast(d_m, "int32"),
                                    D_smem.ptr_to([stage, 0, 0]),
                                )
                            )
                        T.ptx.cp.async_.bulk.commit_group()

        epilogue_tmem_allocated: T.uint32
        T.ptx.ld.shared.u32(epilogue_tmem_allocated, tmem_addr.ptr_to([0]))
        T.cuda.trap_when_assert_failed(epilogue_tmem_allocated == T.uint32(0))
        while tile_scheduler.valid():
            tmem_idx = tile_scheduler.tile_idx % TMEM_DEPTH
            tmem_phase = tile_scheduler.tile_idx // TMEM_DEPTH & 1
            tmem_pipe.full.wait(tmem_idx, tmem_phase)
            epilogue()
            tile_scheduler.next_tile()
        if tid_in_wg == 0:
            T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    # The epilogue warpgroup and peer CTA must finish all TMEM reads first.
    T.cuda.cluster_sync()
    if (wg_id == 0) & (warp_id == 0):
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
        tmem_dealloc_addr: T.uint32
        T.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_addr.ptr_to([0]))
        T.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](tmem_dealloc_addr, T.uint32(512))


def tir_kernel(M: int, N: int, K: int):
    if K % 512 != 0:
        raise ValueError("K must be divisible by 512 for packed block scales")
    swap_ab, dg_block_m, dg_block_n, smem_pipe_depth, log_m, log_n = _choose_deepgemm_config(
        M, N, K
    )
    return _kernel.specialize(
        M=M,
        N=N,
        K=K,
        SWAP_AB=swap_ab,
        DG_BLOCK_M=dg_block_m,
        DG_BLOCK_N=dg_block_n,
        SMEM_DEPTH=smem_pipe_depth,
        LOGICAL_M_CLUSTER=log_m,
        LOGICAL_N_CLUSTER=log_n,
    )


KERNEL_META = {"name": "fp8_blockwise_gemm", "category": "gemm", "compute_capability": 10}
CONFIGS = [
    {"M": 4096, "N": 2112, "K": 7168, "label": "deepgemm_m4096_n2112_k7168"},
    {"M": 4096, "N": 576, "K": 7168, "label": "deepgemm_m4096_n576_k7168"},
    {"M": 4096, "N": 24576, "K": 1536, "label": "deepgemm_m4096_n24576_k1536"},
    {"M": 4096, "N": 32768, "K": 512, "label": "deepgemm_m4096_n32768_k512"},
    {"M": 4096, "N": 7168, "K": 16384, "label": "deepgemm_m4096_n7168_k16384"},
    {"M": 4096, "N": 4096, "K": 7168, "label": "deepgemm_m4096_n4096_k7168"},
    {"M": 4096, "N": 7168, "K": 2048, "label": "deepgemm_m4096_n7168_k2048"},
]


def get_kernel(M, N, K):
    return tir_kernel(M, N, K)


def _prepare_scale_slice_test_data(M: int, N: int, K: int):
    if K % 512 != 0:
        raise ValueError("scale-slice validation requires K to be divisible by 512")

    operand_values = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    a_rows = torch.arange(M)[:, None]
    b_rows = torch.arange(N)[:, None]
    k_values = torch.arange(K)[None, :]
    A_fp8 = operand_values[(3 * a_rows + 5 * k_values + 1) % operand_values.numel()].to(
        torch.float8_e4m3fn
    )
    B_fp8 = operand_values[(7 * b_rows + 2 * k_values + 4) % operand_values.numel()].to(
        torch.float8_e4m3fn
    )

    scale_blocks = K // 128
    scale_block = torch.arange(scale_blocks)[None, :]
    sfa_exponents = ((a_rows + 2 * scale_block) % 3 - 1).to(torch.int16)
    sfb_exponents = ((2 * b_rows + scale_block) % 3 - 1).to(torch.int16)
    sfa_pack = (sfa_exponents + 127).to(torch.uint8).view(torch.uint32).T.contiguous()
    sfb_pack = (sfb_exponents + 127).to(torch.uint8).view(torch.uint32).T.contiguous()

    A_fp8 = A_fp8.to("cuda")
    B_fp8 = B_fp8.to("cuda")
    sfa_pack = sfa_pack.to("cuda")
    sfb_pack = sfb_pack.to("cuda")
    A_dequant = (
        A_fp8.float().reshape(M, scale_blocks, 128)
        * torch.exp2(sfa_exponents.float().to("cuda"))[:, :, None]
    ).reshape(M, K)
    B_dequant = (
        B_fp8.float().reshape(N, scale_blocks, 128)
        * torch.exp2(sfb_exponents.float().to("cuda"))[:, :, None]
    ).reshape(N, K)
    C_ref = (A_dequant @ B_dequant.T).to(torch.bfloat16)
    return A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref


def run_test(M=1024, N=1024, K=1024, verify_scale_slices=False):
    """Compile, run, and verify kernel."""
    import torch
    import torch.nn.functional as F

    from tirx_kernels.runner import compile_kernel

    kernel = tir_kernel(M, N, K)
    if verify_scale_slices:
        A_fp8, B_fp8, sfa_pack, sfb_pack, C_ref = _prepare_scale_slice_test_data(M, N, K)
    else:
        A_fp8, B_fp8, _, _, sfa_pack, sfb_pack, C_ref, _, _ = prepare_data(M, N, K)
    C_tvm = torch.zeros_like(C_ref).to(torch.bfloat16).to("cuda")
    ex = compile_kernel(kernel)
    ex(A_fp8, B_fp8, sfa_pack, sfb_pack, C_tvm)
    if verify_scale_slices:
        torch.testing.assert_close(C_tvm, C_ref, rtol=0, atol=0)
        return
    cosine_sim = F.cosine_similarity(C_tvm.reshape(-1).float(), C_ref.reshape(-1).float(), dim=0)
    assert cosine_sim > 0.97, f"fp8_blockwise_gemm cosine_sim {cosine_sim:.6f} <= 0.97"


def run_bench(
    M=1024,
    N=1024,
    K=1024,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    kernel_fair: bool | None = None,
    **kwargs,
):
    """Benchmark DeepGEMM main kernel against the TIRx kernel."""
    import torch

    from tirx_kernels.runner import compile_kernel

    if kernel_fair is None:
        kernel_fair = _env_flag("TIRX_FP8_BLOCKWISE_GEMM_KERNEL_FAIR", default=True)

    kernel = tir_kernel(M, N, K)
    ex = compile_kernel(kernel)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    A_fp8, B_fp8, sfa, sfb, sfa_pack, sfb_pack, C_ref, _, _ = prepare_data(M, N, K)
    C_tvm = torch.zeros_like(C_ref).to(torch.bfloat16).to("cuda")

    funcs = {"tir": lambda: ex(A_fp8, B_fp8, sfa_pack, sfb_pack, C_tvm)}

    def _deepgemm():
        import deep_gemm

        C_dg = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        if kernel_fair:
            sfa_dg = _dg_scale_view(sfa_pack, M)
            sfb_dg = _dg_scale_view(sfb_pack, N)
            return lambda: deep_gemm.fp8_gemm_nt(
                (A_fp8, sfa_dg), (B_fp8, sfb_dg), C_dg, disable_ue8m0_cast=False, recipe=(1, 1, 128)
            )
        return lambda: deep_gemm.fp8_gemm_nt(
            (A_fp8, sfa), (B_fp8, sfb), C_dg, disable_ue8m0_cast=False, recipe=None
        )

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"deepgemm": _deepgemm},
        **kwargs,
    )
    result["kernel_fair"] = kernel_fair
    return result
