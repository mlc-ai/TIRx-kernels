# This file is a TIRx port of code from fast.cu
# (https://github.com/pranjalssh/fast.cu @ 2dfe5e26aecfd9e5f27bf9d5837deea01acda24b), Copyright (c) 2024 Pranjal Shankhdhar
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""GB300 NVFP4 GEMM port of fast.cu ``gb300/nvfp4/gemm9.cuh``."""

import ctypes
import functools
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import tirx_kernels.kern as K
import tvm

KERNEL_META = {
    "name": "fastcu_nvfp4_gemm_gb300",
    "category": "fastcu",
    "runtime_cuda_archs": ["sm_103a"],
}

SOURCE_COMMIT = "2dfe5e26aecfd9e5f27bf9d5837deea01acda24b"
SOURCE_SHA256 = "b9ec1b3f07ed934728499421b7a90bb7f31cbc179960fd2707ecefbf2ec8606c"
_SOURCE_RELATIVE = Path("gb300/nvfp4/gemm9.cuh")
_REFERENCE_ADAPTER = Path(__file__).with_name("nvfp4_gemm_gb300_reference.cu")

CONFIGS = [
    {"M": 128, "N": 256, "K": 16, "label": "minimum_k64_single"},
    {"M": 256, "N": 256, "K": 32, "label": "minimum_k64_pair"},
    {"M": 257, "N": 264, "K": 48, "label": "mn_boundary_k96"},
    {"M": 768, "N": 512, "K": 1136, "label": "ragged_k16"},
    {"M": 768, "N": 512, "K": 1280, "label": "k64_pair"},
    {"M": 768, "N": 512, "K": 1024, "label": "k64_single"},
    {"M": 768, "N": 512, "K": 1120, "label": "k64_inactive"},
    {"M": 768, "N": 512, "K": 1152, "label": "aligned"},
    *[
        {"M": size, "N": size, "K": size, "label": f"square_{size}"}
        for size in (1024, 2048, 4096, 8192, 16384)
    ],
]

BENCH_CONFIGS = [
    {"M": size, "N": size, "K": size, "label": f"square_{size}"}
    for size in (1024, 2048, 4096, 8192, 16384)
]

_NUM_CLUSTERS = 76
_SMEM_BYTES = 230400
_AB_READY = 0
_AB_FREE = 48
_SF_READY = 96
_SF_FREE = 152
_ACC_READY = 208
_ACC_FREE = 216
_DEALLOC = 224
_TMEM_ADDR = 232
_A_BASE = 1024
_B_BASE = 99328
_SFA_BASE = 197632
_SFB_BASE = 208896
_AB_STRIDE = 16384
_SFA_STRIDE = 1536
_SFB_STRIDE = 3072
_TRY_WAIT_TICKS = 10000000

_TMA_3D_MCAST = (
    "cp.async.bulk.tensor.3d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
)
_TMA_3D = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
_TMA_4D_MCAST = (
    "cp.async.bulk.tensor.4d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
)
_TCGEN05_CP = "tcgen05.cp.cta_group::2.32x128b.warpx4"
_TCGEN05_MMA = "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
_TCGEN05_COMMIT = (
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
)
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"


def _ring_next(slot, phase, size):
    wrap = K.local_scalar("uint32", init=K.cast(slot == size - 1, "uint32"))
    next_slot = K.local_scalar("int32", init=K.Select(wrap != 0, K.int32(0), slot + 1))
    next_phase = K.local_scalar("uint32", init=phase ^ wrap)
    return next_slot, next_phase


def _ring_prev(slot, phase, size):
    at_zero = K.local_scalar("uint32", init=K.cast(slot == 0, "uint32"))
    prev_slot = K.local_scalar("int32", init=K.Select(at_zero != 0, K.int32(size - 1), slot - 1))
    prev_phase = K.local_scalar("uint32", init=K.Select(at_zero != 0, phase, phase ^ K.uint32(1)))
    return prev_slot, prev_phase


def _wait_plain(barrier, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _wait_acquire_cta(barrier, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, barrier, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _bit_or64(*values):
    value = K.uint64(0)
    for item in values:
        value = K.bitwise_or(value, K.cast(item, "uint64"))
    return value


def _ab_desc(smbase, region, slot, atom):
    base = smbase + K.uint32(region) + K.cast(slot * _AB_STRIDE, "uint32")
    addr = K.cast(
        K.bitwise_and(K.shift_right(base, K.uint32(4)), K.uint32(0x7FC0)) + K.uint32(atom), "uint64"
    )
    lbo = K.cast(K.bitwise_and(K.shift_left(base, K.uint32(12)), K.uint32(0x7FC00000)), "uint64")
    return _bit_or64(K.uint64(0x4010404000000000), addr, lbo)


def _ab_desc_straddle(smbase, region, addr_slot, lbo_slot, atom):
    d_addr = _ab_desc(smbase, region, addr_slot, atom)
    d_lbo = _ab_desc(smbase, region, lbo_slot, 0)
    return _bit_or64(
        K.bitwise_and(d_addr, K.bitwise_not(K.uint64(0x00000000FFFF0000))),
        K.bitwise_and(d_lbo, K.uint64(0x00000000FFFF0000)),
    )


def _ab_desc_k64(smbase, region, slot, atom):
    return K.bitwise_and(
        _ab_desc(smbase, region, slot, atom), K.bitwise_not(K.uint64(0x00100000FFFF0000))
    )


def _ab_desc_low(smbase, region, slot, atom):
    """Low word of the source K96 shared-memory operand descriptor."""
    base = smbase + K.uint32(region) + K.cast(slot * _AB_STRIDE, "uint32")
    return K.bitwise_or(
        K.bitwise_and(K.shift_right(base, K.uint32(4)), K.uint32(0x7FC0)) + K.uint32(atom),
        K.bitwise_and(K.shift_left(base, K.uint32(12)), K.uint32(0x7FC00000)),
    )


def _ab_desc_low_straddle(smbase, region, addr_slot, lbo_slot, atom):
    addr = _ab_desc_low(smbase, region, addr_slot, atom)
    lbo = _ab_desc_low(smbase, region, lbo_slot, 0)
    return K.bitwise_or(
        K.bitwise_and(addr, K.uint32(0x0000FFFF)), K.bitwise_and(lbo, K.uint32(0xFFFF0000))
    )


def _join_desc(high, low):
    return K.bitwise_or(K.shift_left(K.cast(high, "uint64"), K.uint32(32)), K.cast(low, "uint64"))


def _sf_cp_desc(smbase, region, slot, slot_units, tile):
    base = smbase + K.uint32(region)
    addr = K.cast(K.bitwise_and(K.shift_right(base, K.uint32(4)), K.uint32(0x3FC0)), "uint64")
    desc = _bit_or64(K.uint64(0x400800010000), addr)
    return desc + K.cast(slot * slot_units + tile * 32, "uint64")


def _sf_id_bits(sfa, sfb):
    return K.bitwise_or(
        K.bitwise_and(K.shift_right(sfa, K.uint32(1)), K.uint32(0x60000000)),
        K.bitwise_and(K.shift_right(sfb, K.uint32(26)), K.uint32(48)),
    )


def _half_from_word(word, half):
    shifted = K.shift_right(word, K.cast(half * 16, "uint32"))
    return K.cast(K.bitwise_and(shifted, K.uint32(0xFFFF)), "uint16")


def _udiv(x, divisor):
    """Truncating division for values known nonnegative in the source contract."""
    return K.cast(K.cast(x, "uint32") // K.cast(divisor, "uint32"), "int32")


def _umod(x, divisor):
    """Remainder paired with :func:`_udiv`, without signed floor fixup."""
    return K.cast(K.cast(x, "uint32") % K.cast(divisor, "uint32"), "int32")


def _uceil(x, divisor):
    """Ceiling division for a nonnegative value and positive divisor."""
    du = K.cast(divisor, "uint32")
    return K.cast((K.cast(x, "uint32") + du - K.uint32(1)) // du, "int32")


@functools.lru_cache(maxsize=1)
def make_kernel():
    """Build the fixed-topology r9 kernel with runtime M/N/K."""

    @K.kernel(warps=7, arch="sm_103a", min_blocks_per_sm=1, grid=(2, 1, _NUM_CLUSTERS))
    def fastcu_nvfp4_gemm_gb300_kernel(
        A_tmap: K.TensorMap,
        B_tmap: K.TensorMap,
        SFA_tmap: K.TensorMap,
        SFB_tmap: K.TensorMap,
        C: K.gptr[K.f16],
        M: K.i32,
        N: K.i32,
        K_dim: K.i32,
        route_table: K.gptr[K.i32],
        sm_side: K.gptr[K.i32],
        cluster_side: K.gptr[K.i32],
        placement_errors: K.gptr[K.u32],
    ):
        crank = K.cta_id_in_cluster([2], preferred=[2])
        _, _, cluster_id = K.cta_id()
        tid = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()
        m_pair = K.cast(crank, "int32") & K.int32(1)

        roles = K.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4], when=m_pair == 0)
        ab_role = roles.role("ab_tma", warps=[5])
        sf_role = roles.role("sf_tma", warps=[6])

        smem = K.alloc_buffer((_SMEM_BYTES,), K.u8, scope="shared.dyn", align=1024)
        smbase = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))

        cluster_grid_m = K.local_scalar("int32", init=_uceil(M, K.int32(256)))
        grid_n = K.local_scalar("int32", init=_uceil(N, K.int32(256)))
        total_tiles = K.local_scalar("int32", init=cluster_grid_m * grid_n)
        full_groups = K.local_scalar("int32", init=_udiv(K_dim, K.int32(768)))
        k_rem = K.local_scalar("int32", init=K_dim - full_groups * 768)
        tail_cells = K.local_scalar("int32", init=_uceil(k_rem, K.int32(96)))
        t_sf = K.local_scalar("int32", init=_uceil(tail_cells, K.int32(2)))
        num_groups = K.local_scalar("int32", init=full_groups + K.cast(tail_cells != 0, "int32"))
        k_rem_mod96 = K.local_scalar("int32", init=_umod(k_rem, K.int32(96)))
        b64_try = K.local_scalar(
            "int32",
            init=K.Select(
                k_rem_mod96 == 64, K.int32(1), K.Select(k_rem_mod96 == 32, K.int32(2), K.int32(0))
            ),
        )
        a64 = K.local_scalar("int32", init=tail_cells - b64_try)
        b64 = K.local_scalar(
            "int32",
            init=K.Select(
                (b64_try != 0)
                & (_umod(k_rem, K.int32(32)) == 0)
                & (a64 >= 0)
                & (_umod(a64, K.int32(2)) == 0),
                b64_try,
                K.int32(0),
            ),
        )
        t_win = K.local_scalar(
            "int32",
            init=K.Select(b64 != 0, _uceil(_udiv(k_rem, K.int32(2)), K.int32(128)), K.int32(3)),
        )

        smid = K.local_scalar("uint32", init=K.cuda.mov_sreg(32, "smid"))
        with K.If(tid == 0), K.Then():
            actual_side = K.local_scalar("int32")
            planned_side = K.local_scalar("int32")
            K.ptx.ld.global_.nc.s32(actual_side, sm_side.ptr_to([smid]))
            K.ptx.ld.global_.nc.s32(planned_side, cluster_side.ptr_to([cluster_id]))
            with K.If(actual_side != planned_side), K.Then():
                old = K.local_scalar("uint32")
                K.ptx.atom.global_.add.u32(old, placement_errors.ptr_to([0]), K.uint32(1))

        with K.If((warp == 0) & (lane == 0)), K.Then():
            for stage in range(6):
                K.ptx.mbarrier.init.shared.b64(
                    K.ptr_byte_offset(smem.ptr_to([0]), _AB_READY + stage * 8, "uint64"),
                    K.uint32(1),
                )
                K.ptx.mbarrier.init.shared.b64(
                    K.ptr_byte_offset(smem.ptr_to([0]), _AB_FREE + stage * 8, "uint64"), K.uint32(1)
                )
            for stage in range(7):
                K.ptx.mbarrier.init.shared.b64(
                    K.ptr_byte_offset(smem.ptr_to([0]), _SF_READY + stage * 8, "uint64"),
                    K.uint32(1),
                )
                K.ptx.mbarrier.init.shared.b64(
                    K.ptr_byte_offset(smem.ptr_to([0]), _SF_FREE + stage * 8, "uint64"), K.uint32(1)
                )
            K.ptx.mbarrier.init.shared.b64(
                K.ptr_byte_offset(smem.ptr_to([0]), _ACC_READY, "uint64"), K.uint32(1)
            )
            K.ptx.mbarrier.init.shared.b64(
                K.ptr_byte_offset(smem.ptr_to([0]), _ACC_FREE, "uint64"), K.uint32(8)
            )
            K.ptx.mbarrier.init.shared.b64(
                K.ptr_byte_offset(smem.ptr_to([0]), _DEALLOC, "uint64"), K.uint32(32)
            )
            K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.barrier.cluster.arrive.release.aligned()
        K.ptx.barrier.cluster.wait.acquire.aligned()

        taddr = K.local_scalar("uint32", init=K.uint32(0))
        with K.If(warp <= 4), K.Then():
            with K.If(warp == 0), K.Then():
                K.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                    smbase + K.uint32(_TMEM_ADDR), K.uint32(512)
                )
            K.ptx.bar.sync(K.uint32(2), K.uint32(160))
            K.ptx.ld.shared.b32(taddr, K.ptr_byte_offset(smem.ptr_to([0]), _TMEM_ADDR, "uint32"))

        with ab_role:
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                slot = K.local_scalar("int32", init=K.int32(0))
                phase = K.local_scalar("uint32", init=K.uint32(1))
                work = K.local_scalar("int32", init=cluster_id)
                with K.While(work < total_tiles):
                    tile = K.local_scalar("int32")
                    K.ptx.ld.global_.ca.s32(tile, route_table.ptr_to([work]))
                    m_row = K.local_scalar("int32", init=_umod(tile, cluster_grid_m))
                    n_group = K.local_scalar("int32", init=_udiv(tile, cluster_grid_m))
                    m_block = K.local_scalar("int32", init=m_row * 2 + m_pair)
                    n_block = K.local_scalar("int32", init=n_group * 2 + m_pair)
                    group = K.local_scalar("int32", init=K.int32(0))
                    with K.While(group < num_groups):
                        for sub in range(3):
                            with K.If((group < full_groups) | (sub < t_win)), K.Then():
                                _wait_acquire_cta(
                                    smbase + K.uint32(_AB_FREE) + K.cast(slot * 8, "uint32"), phase
                                )
                                mbar = K.bitwise_and(
                                    smbase + K.uint32(_AB_READY) + K.cast(slot * 8, "uint32"),
                                    K.uint32(0xFEFFFFFF),
                                )
                                with K.If(m_pair == 0), K.Then():
                                    K.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                                        mbar, K.uint32(65536)
                                    )
                                x = group * 384 + sub * 128
                                a_dst = (
                                    smbase + K.uint32(_A_BASE) + K.cast(slot * _AB_STRIDE, "uint32")
                                )
                                b_dst = (
                                    smbase + K.uint32(_B_BASE) + K.cast(slot * _AB_STRIDE, "uint32")
                                )
                                K.ptx[_TMA_3D_MCAST](
                                    a_dst,
                                    K.address_of(A_tmap),
                                    K.cast(x, "int32"),
                                    K.cast(m_block * 128, "int32"),
                                    K.int32(0),
                                    mbar,
                                    K.cast(K.int32(1) << m_pair, "uint16"),
                                )
                                K.ptx[_TMA_3D](
                                    b_dst,
                                    K.address_of(B_tmap),
                                    K.cast(x, "int32"),
                                    K.cast(n_block * 128, "int32"),
                                    K.int32(0),
                                    mbar,
                                )
                                next_slot, next_phase = _ring_next(slot, phase, 6)
                                K.assign(slot, next_slot)
                                K.assign(phase, next_phase)
                        K.assign(group, group + 1)
                    K.assign(work, work + _NUM_CLUSTERS)
                last_slot, last_phase = _ring_prev(slot, phase, 6)
                _wait_acquire_cta(
                    smbase + K.uint32(_AB_FREE) + K.cast(last_slot * 8, "uint32"), last_phase
                )
                with K.If(m_pair == 0), K.Then():
                    mbar = K.bitwise_and(
                        smbase + K.uint32(_AB_READY) + K.cast(last_slot * 8, "uint32"),
                        K.uint32(0xFEFFFFFF),
                    )
                    K.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                        mbar, K.uint32(65536)
                    )

        with sf_role:
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                slot = K.local_scalar("int32", init=K.int32(0))
                phase = K.local_scalar("uint32", init=K.uint32(1))
                work = K.local_scalar("int32", init=cluster_id)
                with K.While(work < total_tiles):
                    tile = K.local_scalar("int32")
                    K.ptx.ld.global_.ca.s32(tile, route_table.ptr_to([work]))
                    m_row = K.local_scalar("int32", init=_umod(tile, cluster_grid_m))
                    n_group = K.local_scalar("int32", init=_udiv(tile, cluster_grid_m))
                    m_block = K.local_scalar("int32", init=m_row * 2 + m_pair)
                    group = K.local_scalar("int32", init=K.int32(0))
                    with K.While(group < num_groups):
                        for sub in range(4):
                            with K.If((group < full_groups) | (sub < t_sf)), K.Then():
                                _wait_acquire_cta(
                                    smbase + K.uint32(_SF_FREE) + K.cast(slot * 8, "uint32"), phase
                                )
                                mbar = K.bitwise_and(
                                    smbase + K.uint32(_SF_READY) + K.cast(slot * 8, "uint32"),
                                    K.uint32(0xFEFFFFFF),
                                )
                                with K.If(m_pair == 0), K.Then():
                                    K.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                                        mbar, K.uint32(9216)
                                    )
                                sf_group = (group * 4 + sub) * 3
                                sfa_dst = (
                                    smbase
                                    + K.uint32(_SFA_BASE)
                                    + K.cast(slot * _SFA_STRIDE, "uint32")
                                )
                                sfb_dst = (
                                    smbase
                                    + K.uint32(_SFB_BASE)
                                    + K.cast(slot * _SFB_STRIDE + m_pair * 1536, "uint32")
                                )
                                K.ptx[_TMA_4D_MCAST](
                                    sfa_dst,
                                    K.address_of(SFA_tmap),
                                    K.int32(0),
                                    K.int32(0),
                                    K.cast(sf_group, "int32"),
                                    K.cast(m_block, "int32"),
                                    mbar,
                                    K.cast(K.int32(1) << m_pair, "uint16"),
                                )
                                K.ptx[_TMA_4D_MCAST](
                                    sfb_dst,
                                    K.address_of(SFB_tmap),
                                    K.int32(0),
                                    K.int32(0),
                                    K.cast(sf_group, "int32"),
                                    K.cast(n_group * 2 + m_pair, "int32"),
                                    mbar,
                                    K.uint16(3),
                                )
                                next_slot, next_phase = _ring_next(slot, phase, 7)
                                K.assign(slot, next_slot)
                                K.assign(phase, next_phase)
                        K.assign(group, group + 1)
                    K.assign(work, work + _NUM_CLUSTERS)
                last_slot, last_phase = _ring_prev(slot, phase, 7)
                _wait_acquire_cta(
                    smbase + K.uint32(_SF_FREE) + K.cast(last_slot * 8, "uint32"), last_phase
                )
                with K.If(m_pair == 0), K.Then():
                    mbar = K.bitwise_and(
                        smbase + K.uint32(_SF_READY) + K.cast(last_slot * 8, "uint32"),
                        K.uint32(0xFEFFFFFF),
                    )
                    K.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                        mbar, K.uint32(9216)
                    )

        with mma_role:
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                d_phase = K.local_scalar("uint32", init=K.uint32(1))
                ab_slot = K.local_scalar("int32", init=K.int32(0))
                ab_phase = K.local_scalar("uint32", init=K.uint32(0))
                sf_slot = K.local_scalar("int32", init=K.int32(0))
                sf_phase = K.local_scalar("uint32", init=K.uint32(0))
                work = K.local_scalar("int32", init=cluster_id)

                # gemm9 hoists these invariant descriptor halves and TMEM
                # addresses out of its persistent tile loop. Keeping the same
                # shape here prevents repeated 64-bit reconstruction between
                # dependent tcgen05 issues.
                sfa_src = K.local_scalar("uint64", init=_sf_cp_desc(smbase, _SFA_BASE, 0, 0, 0))
                sfb_src = K.local_scalar("uint64", init=_sf_cp_desc(smbase, _SFB_BASE, 0, 0, 0))
                sfa_t0 = K.local_scalar("uint32", init=taddr + K.uint32(476))
                sfa_t1 = K.local_scalar("uint32", init=taddr + K.uint32(480))
                sfb_t0 = K.local_scalar("uint32", init=taddr + K.uint32(488))
                sfb_t1 = K.local_scalar("uint32", init=taddr + K.uint32(496))
                sfa_w1 = K.local_scalar("uint32", init=K.bitwise_or(sfa_t1, K.uint32(0x80000000)))
                sfb_w1 = K.local_scalar("uint32", init=K.bitwise_or(sfb_t1, K.uint32(0x80000000)))
                idesc_w0 = K.local_scalar(
                    "uint32", init=K.bitwise_or(K.uint32(0x90400480), _sf_id_bits(sfa_t0, sfb_t0))
                )
                idesc_w1 = K.local_scalar(
                    "uint32", init=K.bitwise_or(K.uint32(0x90400480), _sf_id_bits(sfa_w1, sfb_w1))
                )

                def stage_scale(slot, phase):
                    _wait_plain(smbase + K.uint32(_SF_READY) + K.cast(slot * 8, "uint32"), phase)
                    copies = (
                        (_SFA_BASE, 0, 96, 476),
                        (_SFA_BASE, 1, 96, 480),
                        (_SFA_BASE, 2, 96, 484),
                        (_SFB_BASE, 0, 192, 488),
                        (_SFB_BASE, 3, 192, 492),
                        (_SFB_BASE, 1, 192, 496),
                        (_SFB_BASE, 4, 192, 500),
                        (_SFB_BASE, 2, 192, 504),
                        (_SFB_BASE, 5, 192, 508),
                    )
                    for region, tile, stride, dst in copies:
                        src = sfa_src if region == _SFA_BASE else sfb_src
                        K.ptx[_TCGEN05_CP](
                            taddr + K.uint32(dst), src + K.cast(slot * stride + tile * 32, "uint64")
                        )
                    K.ptx[_TCGEN05_COMMIT](
                        smbase + K.uint32(_SF_FREE) + K.cast(slot * 8, "uint32"), K.uint16(3)
                    )
                    return _ring_next(slot, phase, 7)

                def commit_ab(slot):
                    K.ptx[_TCGEN05_COMMIT](
                        smbase + K.uint32(_AB_FREE) + K.cast(slot * 8, "uint32"), K.uint16(3)
                    )

                def issue96(cell, W0, W1, W2, d_tmem, accum, issue_pred=None):
                    if cell == 0:
                        da = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W0, 0))
                        db = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W0, 0))
                    elif cell == 1:
                        da = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W0, 3))
                        db = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W0, 3))
                    elif cell == 2:
                        da = _join_desc(
                            K.uint32(0x40104040), _ab_desc_low_straddle(smbase, _A_BASE, W0, W1, 6)
                        )
                        db = _join_desc(
                            K.uint32(0x40104040), _ab_desc_low_straddle(smbase, _B_BASE, W0, W1, 6)
                        )
                    elif cell == 3:
                        da = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W1, 1))
                        db = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W1, 1))
                    elif cell == 4:
                        da = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W1, 4))
                        db = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W1, 4))
                    elif cell == 5:
                        da = _join_desc(
                            K.uint32(0x40104040), _ab_desc_low_straddle(smbase, _A_BASE, W1, W2, 7)
                        )
                        db = _join_desc(
                            K.uint32(0x40104040), _ab_desc_low_straddle(smbase, _B_BASE, W1, W2, 7)
                        )
                    elif cell == 6:
                        da = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W2, 2))
                        db = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W2, 2))
                    else:
                        da = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W2, 5))
                        db = _join_desc(K.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W2, 5))
                    if cell % 2 == 0:
                        sfa = sfa_t0
                        sfb = sfb_t0
                        idesc = idesc_w0
                    else:
                        sfa = sfa_w1
                        sfb = sfb_w1
                        idesc = idesc_w1
                    args = (
                        d_tmem,
                        da,
                        db,
                        idesc,
                        sfa,
                        sfb,
                        K.ptx.pred(K.cast(accum != 0, "uint32")),
                    )
                    if issue_pred is None:
                        K.ptx[_TCGEN05_MMA](*args)
                    else:
                        with K.If(issue_pred), K.Then():
                            K.ptx[_TCGEN05_MMA](*args)

                def issue64(slot, atom, word, d_tmem, accum):
                    da = _join_desc(
                        K.uint32(0x40004040),
                        K.bitwise_and(
                            _ab_desc_low(smbase, _A_BASE, slot, atom), K.uint32(0x0000FFFF)
                        ),
                    )
                    db = _join_desc(
                        K.uint32(0x40004040),
                        K.bitwise_and(
                            _ab_desc_low(smbase, _B_BASE, slot, atom), K.uint32(0x0000FFFF)
                        ),
                    )
                    if word == 0:
                        sfa = sfa_t0
                        sfb = sfb_t0
                    else:
                        sfa = sfa_t1
                        sfb = sfb_t1
                    idesc = K.bitwise_or(K.uint32(0x10400480), _sf_id_bits(sfa, sfb))
                    K.ptx[_TCGEN05_MMA](
                        d_tmem, da, db, idesc, sfa, sfb, K.ptx.pred(K.cast(accum != 0, "uint32"))
                    )

                def issue_k64_variant(a_count, b_count, d_tmem, accum, need_wait, acc_wait):
                    W0 = K.local_scalar("int32", init=ab_slot)
                    ph0 = K.local_scalar("uint32", init=ab_phase)
                    W1, ph1 = _ring_next(W0, ph0, 6)
                    W2, ph2 = _ring_next(W1, ph1, 6)
                    n_cells = a_count + b_count
                    t_windows = (a_count * 96 + b_count * 64 + 255) // 256

                    specs = []
                    for cell in range(a_count):
                        specs.append(("k96", cell))
                    if a_count == 0:
                        specs.append(("k64", W0, 0, 0))
                        if b_count == 2:
                            specs.append(("k64", W0, 2, 1))
                    elif a_count == 2:
                        specs.append(("k64", W0, 6, 0))
                        if b_count == 2:
                            specs.append(("k64", W1, 0, 1))
                    elif a_count == 4:
                        specs.append(("k64", W1, 4, 0))
                        if b_count == 2:
                            specs.append(("k64", W1, 6, 1))
                    else:
                        specs.append(("k64", W2, 2, 0))
                        if b_count == 2:
                            specs.append(("k64", W2, 4, 1))

                    def issue_at(index):
                        if index < len(specs):
                            spec = specs[index]
                            if spec[0] == "k96":
                                issue96(spec[1], W0, W1, W2, d_tmem, accum)
                            else:
                                issue64(spec[1], spec[2], spec[3], d_tmem, accum)
                            K.assign(accum, K.uint32(1))

                    ns, np = stage_scale(sf_slot, sf_phase)
                    K.assign(sf_slot, ns)
                    K.assign(sf_phase, np)
                    _wait_plain(smbase + K.uint32(_AB_READY) + K.cast(W0 * 8, "uint32"), ph0)
                    with K.If(need_wait != 0), K.Then():
                        _wait_plain(smbase + K.uint32(_ACC_FREE), acc_wait)
                        K.assign(need_wait, K.uint32(0))
                    issue_at(0)
                    issue_at(1)

                    if n_cells > 2:
                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                    if t_windows >= 2:
                        _wait_plain(smbase + K.uint32(_AB_READY) + K.cast(W1 * 8, "uint32"), ph1)
                    issue_at(2)
                    commit_ab(W0)
                    issue_at(3)

                    if n_cells > 4:
                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                    issue_at(4)
                    if t_windows >= 3:
                        _wait_plain(smbase + K.uint32(_AB_READY) + K.cast(W2 * 8, "uint32"), ph2)
                    if t_windows == 1:
                        K.assign(ab_slot, W1)
                        K.assign(ab_phase, ph1)
                    elif t_windows == 2:
                        K.assign(ab_slot, W2)
                        K.assign(ab_phase, ph2)
                    else:
                        ns_ab, np_ab = _ring_next(W2, ph2, 6)
                        K.assign(ab_slot, ns_ab)
                        K.assign(ab_phase, np_ab)
                    issue_at(5)

                    if n_cells > 6:
                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                    if t_windows >= 2:
                        commit_ab(W1)
                    issue_at(6)
                    issue_at(7)
                    if t_windows >= 3:
                        commit_ab(W2)

                with K.While(work < total_tiles):
                    acc_wait = K.local_scalar("uint32", init=d_phase)
                    K.assign(d_phase, d_phase ^ K.uint32(1))
                    d_tmem = taddr + d_phase * K.uint32(220)
                    accum = K.local_scalar("uint32", init=K.uint32(0))
                    need_wait = K.local_scalar("uint32", init=K.uint32(1))
                    group = K.local_scalar("int32", init=K.int32(0))

                    with K.While(group < full_groups):
                        W0 = K.local_scalar("int32", init=ab_slot)
                        ph0 = K.local_scalar("uint32", init=ab_phase)
                        W1, ph1 = _ring_next(W0, ph0, 6)
                        W2, ph2 = _ring_next(W1, ph1, 6)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                        _wait_plain(smbase + K.uint32(_AB_READY) + K.cast(W0 * 8, "uint32"), ph0)
                        with K.If(need_wait != 0), K.Then():
                            _wait_plain(smbase + K.uint32(_ACC_FREE), acc_wait)
                            K.assign(need_wait, K.uint32(0))
                        issue96(0, W0, W1, W2, d_tmem, accum)
                        K.assign(accum, K.uint32(1))
                        issue96(1, W0, W1, W2, d_tmem, accum)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                        _wait_plain(smbase + K.uint32(_AB_READY) + K.cast(W1 * 8, "uint32"), ph1)
                        issue96(2, W0, W1, W2, d_tmem, accum)
                        commit_ab(W0)
                        issue96(3, W0, W1, W2, d_tmem, accum)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                        issue96(4, W0, W1, W2, d_tmem, accum)
                        _wait_plain(smbase + K.uint32(_AB_READY) + K.cast(W2 * 8, "uint32"), ph2)
                        ns_ab, np_ab = _ring_next(W2, ph2, 6)
                        K.assign(ab_slot, ns_ab)
                        K.assign(ab_phase, np_ab)
                        issue96(5, W0, W1, W2, d_tmem, accum)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        K.assign(sf_slot, ns)
                        K.assign(sf_phase, np)
                        commit_ab(W1)
                        issue96(6, W0, W1, W2, d_tmem, accum)
                        issue96(7, W0, W1, W2, d_tmem, accum)
                        commit_ab(W2)
                        K.assign(group, group + 1)

                    with K.If(tail_cells != 0), K.Then():
                        with K.If(b64 != 0), K.Then():
                            for ac in (0, 2, 4, 6):
                                for bc in (1, 2):
                                    with K.If((a64 == ac) & (b64 == bc)), K.Then():
                                        issue_k64_variant(
                                            ac, bc, d_tmem, accum, need_wait, acc_wait
                                        )
                        with K.If(b64 == 0), K.Then():
                            W0 = K.local_scalar("int32", init=ab_slot)
                            ph0 = K.local_scalar("uint32", init=ab_phase)
                            W1, ph1 = _ring_next(W0, ph0, 6)
                            W2, ph2 = _ring_next(W1, ph1, 6)

                            ns, np = stage_scale(sf_slot, sf_phase)
                            K.assign(sf_slot, ns)
                            K.assign(sf_phase, np)
                            _wait_plain(
                                smbase + K.uint32(_AB_READY) + K.cast(W0 * 8, "uint32"), ph0
                            )
                            with K.If(need_wait != 0), K.Then():
                                _wait_plain(smbase + K.uint32(_ACC_FREE), acc_wait)
                                K.assign(need_wait, K.uint32(0))
                            issue96(0, W0, W1, W2, d_tmem, accum)
                            K.assign(accum, K.uint32(1))
                            issue96(1, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 2)

                            with K.If(tail_cells > 2), K.Then():
                                ns, np = stage_scale(sf_slot, sf_phase)
                                K.assign(sf_slot, ns)
                                K.assign(sf_phase, np)
                            _wait_plain(
                                smbase + K.uint32(_AB_READY) + K.cast(W1 * 8, "uint32"), ph1
                            )
                            issue96(2, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 3)
                            commit_ab(W0)
                            issue96(3, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 4)

                            with K.If(tail_cells > 4), K.Then():
                                ns, np = stage_scale(sf_slot, sf_phase)
                                K.assign(sf_slot, ns)
                                K.assign(sf_phase, np)
                            issue96(4, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 5)
                            _wait_plain(
                                smbase + K.uint32(_AB_READY) + K.cast(W2 * 8, "uint32"), ph2
                            )
                            ns_ab, np_ab = _ring_next(W2, ph2, 6)
                            K.assign(ab_slot, ns_ab)
                            K.assign(ab_phase, np_ab)
                            issue96(5, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 6)

                            with K.If(tail_cells > 6), K.Then():
                                ns, np = stage_scale(sf_slot, sf_phase)
                                K.assign(sf_slot, ns)
                                K.assign(sf_phase, np)
                            commit_ab(W1)
                            issue96(6, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 7)
                            issue96(7, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 8)
                            commit_ab(W2)

                    K.ptx[_TCGEN05_COMMIT](smbase + K.uint32(_ACC_READY), K.uint16(3))
                    K.assign(work, work + _NUM_CLUSTERS)
                _wait_plain(smbase + K.uint32(_ACC_FREE), d_phase)

        with epilogue_role:
            d_buffer = K.local_scalar("int32", init=K.int32(0))
            acc_phase = K.local_scalar("uint32", init=K.uint32(0))
            work = K.local_scalar("int32", init=cluster_id)
            values = K.alloc_local((32,), "float32")
            packed = K.alloc_local((16,), "uint32")
            taddr_lane = K.local_scalar(
                "uint32", init=K.shift_left(K.cast(warp * 32, "uint32"), K.uint32(16))
            )
            even_crank = K.local_scalar(
                "uint32", init=K.bitwise_and(K.cast(crank, "uint32"), K.uint32(0xFFFFFFFE))
            )
            with K.While(work < total_tiles):
                tile_lane = K.local_scalar("int32")
                K.ptx.ld.global_.ca.s32(tile_lane, route_table.ptr_to([work]))
                tile = K.local_scalar("uint32")
                K.ptx.redux_sync.min.u32(tile, K.cast(tile_lane, "uint32"), K.uint32(0xFFFFFFFF))
                m_row = K.local_scalar("int32", init=_umod(tile, cluster_grid_m))
                n_group = K.local_scalar("int32", init=_udiv(tile, cluster_grid_m))
                off_n = K.local_scalar("int32", init=n_group * 256)
                _wait_plain(smbase + K.uint32(_ACC_READY), acc_phase)
                K.assign(acc_phase, acc_phase ^ K.uint32(1))
                tmem_base = K.local_scalar("uint32", init=taddr + K.cast(d_buffer * 220, "uint32"))
                row = K.local_scalar("int32", init=(m_row * 2 + m_pair) * 128 + warp * 32 + lane)
                row_in = K.local_scalar("uint32", init=K.cast(row < M, "uint32"))
                row_base = K.local_scalar("int64", init=K.cast(row, "int64") * N)
                c_addr = K.reinterpret("uint64", C.ptr_to([0]))
                aligned = K.local_scalar(
                    "uint32",
                    init=K.cast(
                        (K.bitwise_and(row_base, K.int32(15)) == 0)
                        & (K.bitwise_and(c_addr, K.uint64(31)) == 0),
                        "uint32",
                    ),
                )
                with K.serial(0, 8) as k:
                    band = K.Select(d_buffer == 0, 7 - k, k)
                    load_addr = tmem_base + K.cast(band * 32, "uint32") + taddr_lane
                    K.ptx[_TMEM_LD_X32](*[values[i] for i in range(32)], load_addr)
                    with K.If(k == 1), K.Then():
                        mapped_free = K.local_scalar("uint32")
                        K.ptx.mapa.shared__cluster.u32(
                            mapped_free, smbase + K.uint32(_ACC_FREE), even_crank
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        K.ptx["mbarrier.arrive.release.cta.shared::cluster.b64"](
                            mapped_free, K.uint32(1), pred=K.cast(lane == 0, "uint32")
                        )
                    with K.If(row_in != 0), K.Then():
                        for pair in range(16):
                            K.ptx.cvt.rn.f16x2.f32(
                                packed[pair], values[pair * 2 + 1], values[pair * 2]
                            )
                        col = K.local_scalar("int32", init=off_n + band * 32)
                        with K.If(aligned != 0), K.Then():
                            for q in range(2):
                                c16 = col + q * 16
                                with K.If(c16 + 16 <= N), K.Then():
                                    K.ptx["st.global.L1::no_allocate.L2::evict_first.v8.b32"](
                                        C.ptr_to([row_base + c16]),
                                        *[packed[q * 8 + i] for i in range(8)],
                                    )
                                with K.If(c16 + 16 > N), K.Then():
                                    for p in range(2):
                                        c8 = c16 + p * 8
                                        with K.If(c8 + 8 <= N), K.Then():
                                            K.ptx["st.global.L1::no_allocate.v4.b32"](
                                                C.ptr_to([row_base + c8]),
                                                *[packed[q * 8 + p * 4 + i] for i in range(4)],
                                            )
                                        with K.If(c8 + 8 > N), K.Then():
                                            for h in range(8):
                                                word = q * 8 + p * 4 + h // 2
                                                K.ptx.st.global_.b16(
                                                    C.ptr_to([row_base + c8 + h]),
                                                    _half_from_word(packed[word], h % 2),
                                                    pred=K.cast(c8 + h < N, "uint32"),
                                                )
                        with K.If(aligned == 0), K.Then():
                            for q in range(4):
                                c8 = col + q * 8
                                with K.If(c8 + 8 <= N), K.Then():
                                    K.ptx["st.global.L1::no_allocate.v4.b32"](
                                        C.ptr_to([row_base + c8]),
                                        *[packed[q * 4 + i] for i in range(4)],
                                    )
                                with K.If(c8 + 8 > N), K.Then():
                                    for h in range(8):
                                        word = q * 4 + h // 2
                                        K.ptx.st.global_.b16(
                                            C.ptr_to([row_base + c8 + h]),
                                            _half_from_word(packed[word], h % 2),
                                            pred=K.cast(c8 + h < N, "uint32"),
                                        )
                K.assign(d_buffer, K.int32(1) - d_buffer)
                K.assign(work, work + _NUM_CLUSTERS)

            K.ptx.bar.sync(K.uint32(3), K.uint32(128))
            with K.If(warp == 0), K.Then():
                K.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
                remote_dealloc = K.local_scalar("uint32")
                K.ptx.mapa.shared__cluster.u32(
                    remote_dealloc,
                    smbase + K.uint32(_DEALLOC),
                    K.cast(crank, "uint32") ^ K.uint32(1),
                )
                K.ptx["mbarrier.arrive.release.cta.shared::cluster.b64"](
                    remote_dealloc, K.uint32(1)
                )
                _wait_acquire_cta(smbase + K.uint32(_DEALLOC), K.uint32(0))
                K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](taddr, K.uint32(512))

    return fastcu_nvfp4_gemm_gb300_kernel


SWIZZLE_NONE = 0
SWIZZLE_128B = 3


class _AlignedTensorMap:
    __slots__ = ("_storage", "ptr")

    def __init__(self):
        self._storage = ctypes.create_string_buffer(192)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tiled(dtype, tensor, *, dims, strides_bytes, box, swizzle):
    descriptor = _AlignedTensorMap()
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        descriptor.ptr,
        dtype,
        len(dims),
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides_bytes,
        *box,
        *((1,) * len(dims)),
        0,
        swizzle,
        0,
        0,
    )
    return descriptor


def _build_tensor_maps(M, N, K_dim, A, B, SFA, SFB):
    row_bytes = K_dim // 2
    row_stride = (row_bytes + 15) & ~15
    sf_inner = ((K_dim + 63) // 64) * 4
    return [
        _encode_tiled(
            "uint8",
            A,
            dims=(row_bytes, M, 1),
            strides_bytes=(row_stride, M * row_stride),
            box=(128, 128, 1),
            swizzle=SWIZZLE_128B,
        ),
        _encode_tiled(
            "uint8",
            B,
            dims=(row_bytes, N, 1),
            strides_bytes=(row_stride, N * row_stride),
            box=(128, 128, 1),
            swizzle=SWIZZLE_128B,
        ),
        _encode_tiled(
            "uint8",
            SFA,
            dims=(128, 4, sf_inner // 4, (M + 127) // 128),
            strides_bytes=(128, 512, sf_inner * 128),
            box=(128, 4, 3, 1),
            swizzle=SWIZZLE_NONE,
        ),
        _encode_tiled(
            "uint8",
            SFB,
            dims=(128, 4, sf_inner // 4, (N + 127) // 128),
            strides_bytes=(128, 512, sf_inner * 128),
            box=(128, 4, 3, 1),
            swizzle=SWIZZLE_NONE,
        ),
    ]


def _fastcu_source_root() -> Path:
    override = os.environ.get("FASTCU_PATH")
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        (Path(override),)
        if override
        else (
            repo_root / ".reference-deps" / "fast-cu",
            Path("/root-vol/aarch64-ws/kernel-libs/gb300/fast.cu"),
        )
    )
    for root in candidates:
        source = root / _SOURCE_RELATIVE
        if not source.is_file():
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != SOURCE_SHA256:
            raise RuntimeError(f"frozen fast.cu source hash mismatch: {source} sha256={actual}")
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot verify fast.cu checkout {root}: {error}") from error
        if commit != SOURCE_COMMIT:
            raise RuntimeError(f"frozen fast.cu checkout mismatch: {root} commit={commit}")
        return root
    tried = ", ".join(str(root) for root in candidates)
    raise RuntimeError(f"frozen fast.cu source is unavailable; tried: {tried}")


@functools.lru_cache(maxsize=1)
def _reference_library():
    import fcntl

    source_root = _fastcu_source_root()
    nvcc = os.environ.get("CUDACXX", "nvcc")
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    digest = hashlib.sha256(
        _REFERENCE_ADAPTER.read_bytes()
        + (source_root / _SOURCE_RELATIVE).read_bytes()
        + version.encode()
        + b"-lineinfo"
    ).hexdigest()[:20]
    cache_root = Path(os.environ.get("TIRX_BENCH_CACHE_DIR", "/tmp/tirx-kernels-cache"))
    build_dir = cache_root / "fastcu" / "nvfp4_gemm_gb300"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / f"reference-{digest}.so"
    lock_path = build_dir / f"reference-{digest}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library_path.is_file():
            temporary = build_dir / f"reference-{digest}-{os.getpid()}.tmp.so"
            command = [
                nvcc,
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-lineinfo",
                "--shared",
                "-Xcompiler=-fPIC",
                "-gencode",
                "arch=compute_103a,code=sm_103a",
                f"-I{source_root / 'gb300' / 'nvfp4'}",
                str(_REFERENCE_ADAPTER),
                "-o",
                str(temporary),
                "-lcuda",
                "-lcudart",
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.replace(temporary, library_path)
            except subprocess.CalledProcessError as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    "fast.cu reference compilation failed:\n" + error.stdout + error.stderr
                ) from error
    lib = ctypes.CDLL(str(library_path))
    pointer = ctypes.c_void_p
    lib.fastcu_nvfp4_prepare_schedule.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    lib.fastcu_nvfp4_prepare_schedule.restype = ctypes.c_int
    lib.fastcu_nvfp4_create.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.fastcu_nvfp4_create.restype = pointer
    lib.fastcu_nvfp4_launch.argtypes = [pointer, pointer]
    lib.fastcu_nvfp4_launch.restype = ctypes.c_int
    lib.fastcu_nvfp4_source_placement_errors.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.fastcu_nvfp4_source_placement_errors.restype = ctypes.c_int
    lib.fastcu_nvfp4_destroy.argtypes = [pointer]
    lib.fastcu_nvfp4_destroy.restype = None
    lib.fastcu_nvfp4_ptx_version.argtypes = []
    lib.fastcu_nvfp4_ptx_version.restype = ctypes.c_int
    if lib.fastcu_nvfp4_ptx_version() != 93:
        raise RuntimeError("fast.cu reference was not built for PTX ISA 9.3")
    return lib


def _pack_vec16_scales(logical, outer, K_dim):
    import torch

    logical_inner = K_dim // 16
    sf_inner = ((K_dim + 63) // 64) * 4
    packed_outer = ((outer + 127) // 128) * 128
    packed = torch.zeros(packed_outer * sf_inner, device=logical.device, dtype=torch.uint8)
    o = torch.arange(outer, device=logical.device, dtype=torch.int64)[:, None]
    j = torch.arange(logical_inner, device=logical.device, dtype=torch.int64)[None, :]
    offset = (
        ((j // 4) * 4 + (o // 128) * sf_inner) * 128
        + (o % 32) * 16
        + ((o % 128) // 32) * 4
        + (j % 4)
    )
    packed[offset.reshape(-1)] = logical.reshape(-1)
    return packed


def prepare_data(M: int, N: int, K: int, **_: Any):
    """Create deterministic source-format packed E2M1 and VEC16 scale buffers."""
    import torch

    if M <= 0 or N <= 0 or K <= 0 or K % 16:
        raise ValueError("M and N must be positive; K must be a positive multiple of 16")
    gen = torch.Generator(device="cpu")
    gen.manual_seed((M * 1000003 + N * 1009 + K) & 0x7FFFFFFF)
    row_bytes = K // 2
    row_stride = (row_bytes + 15) & ~15
    A_host = torch.zeros((M, row_stride), dtype=torch.uint8)
    B_host = torch.zeros((N, row_stride), dtype=torch.uint8)
    A_host[:, :row_bytes] = torch.randint(0, 256, (M, row_bytes), generator=gen, dtype=torch.uint8)
    B_host[:, :row_bytes] = torch.randint(0, 256, (N, row_bytes), generator=gen, dtype=torch.uint8)
    scale_values = torch.tensor([0x30, 0x34, 0x38, 0x3C, 0x40, 0x28, 0x2C], dtype=torch.uint8)
    A_sf_logical = scale_values[torch.randint(0, len(scale_values), (M, K // 16), generator=gen)]
    B_sf_logical = scale_values[torch.randint(0, len(scale_values), (N, K // 16), generator=gen)]

    def guarded_u8(value):
        storage = torch.full((value.numel() + 256,), 0xA5, device="cuda", dtype=torch.uint8)
        view = storage[: value.numel()].view(value.shape)
        view.copy_(value)
        return view, storage[value.numel() :]

    A, A_guard = guarded_u8(A_host.cuda())
    B, B_guard = guarded_u8(B_host.cuda())
    SFA, SFA_guard = guarded_u8(_pack_vec16_scales(A_sf_logical.cuda(), M, K))
    SFB, SFB_guard = guarded_u8(_pack_vec16_scales(B_sf_logical.cuda(), N, K))
    C_storage = torch.full((M * N + 64,), 12345.0, device="cuda", dtype=torch.float16)
    C_source_storage = torch.full_like(C_storage, 12345.0)
    C = C_storage[: M * N]
    C_source = C_source_storage[: M * N]
    route = torch.zeros(4096, device="cuda", dtype=torch.int32)
    total_tiles = ((M + 255) // 256) * ((N + 255) // 256)
    route[:total_tiles] = torch.arange(total_tiles, device="cuda", dtype=torch.int32)
    sm_side = torch.zeros(256, device="cuda", dtype=torch.int32)
    cluster_side = torch.zeros(128, device="cuda", dtype=torch.int32)
    placement = torch.zeros(1, device="cuda", dtype=torch.uint32)
    return {
        "A": A,
        "B": B,
        "SFA": SFA,
        "SFB": SFB,
        "C": C,
        "C_source": C_source,
        "route": route,
        "sm_side": sm_side,
        "cluster_side": cluster_side,
        "placement": placement,
        "guards": (
            ("A", A_guard, 0xA5),
            ("B", B_guard, 0xA5),
            ("SFA", SFA_guard, 0xA5),
            ("SFB", SFB_guard, 0xA5),
            ("C", C_storage[M * N :], 12345.0),
            ("C_source", C_source_storage[M * N :], 12345.0),
        ),
    }


class _Runner:
    def __init__(self):
        previous = os.environ.get("TVM_CUDA_COMPILE_MODE")
        previous_reg_level = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
        os.environ["TVM_CUDA_COMPILE_MODE"] = "nvcc"
        if previous_reg_level is None:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = "4"
        try:
            self.lib = make_kernel().compile()
        finally:
            if previous is None:
                os.environ.pop("TVM_CUDA_COMPILE_MODE", None)
            else:
                os.environ["TVM_CUDA_COMPILE_MODE"] = previous
            if previous_reg_level is None:
                os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
            else:
                os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous_reg_level
        self._maps = None
        self._map_key = None

    def __call__(self, data, M, N, K_dim):
        key = tuple(int(data[name].data_ptr()) for name in ("A", "B", "SFA", "SFB"))
        if key != self._map_key:
            self._maps = _build_tensor_maps(
                M, N, K_dim, data["A"], data["B"], data["SFA"], data["SFB"]
            )
            self._map_key = key
        self.lib(
            *[descriptor.ptr for descriptor in self._maps],
            data["C"],
            M,
            N,
            K_dim,
            data["route"],
            data["sm_side"],
            data["cluster_side"],
            data["placement"],
        )
        return data["C"].view(M, N)


class _SourceRunner:
    def __init__(self, data, M, N, K_dim):
        import torch

        self.lib = _reference_library()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        status = self.lib.fastcu_nvfp4_prepare_schedule(
            M,
            N,
            K_dim,
            ctypes.c_void_p(data["route"].data_ptr()),
            ctypes.c_void_p(data["sm_side"].data_ptr()),
            ctypes.c_void_p(data["cluster_side"].data_ptr()),
            stream,
        )
        if status != 0:
            raise RuntimeError(f"fast.cu schedule preparation failed: CUDA error {status}")
        self.handle = self.lib.fastcu_nvfp4_create(
            *[
                ctypes.c_void_p(data[name].data_ptr())
                for name in ("A", "B", "SFA", "SFB", "C_source")
            ],
            M,
            N,
            K_dim,
        )
        if not self.handle:
            raise RuntimeError("fast.cu reference handle creation failed")

    def __call__(self):
        import torch

        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        status = self.lib.fastcu_nvfp4_launch(self.handle, stream)
        if status != 0:
            raise RuntimeError(f"fast.cu reference launch failed: CUDA error {status}")

    def placement_errors(self):
        value = ctypes.c_uint32()
        status = self.lib.fastcu_nvfp4_source_placement_errors(ctypes.byref(value))
        if status != 0:
            raise RuntimeError(f"fast.cu placement readback failed: CUDA error {status}")
        return value.value

    def close(self):
        if self.handle:
            self.lib.fastcu_nvfp4_destroy(self.handle)
            self.handle = None


_RUNNER = None


def _runner():
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = _Runner()
    return _RUNNER


def _check_guards(data):
    import torch

    for name, guard, expected in data["guards"]:
        if not bool(torch.all(guard == expected).item()):
            raise AssertionError(f"{name} allocation guard was modified")


def _check_bitwise(data, M, N):
    import torch

    actual = data["C"].view(M, N)
    expected = data["C_source"].view(M, N)
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError("source/TIRx output contains non-finite or poison values")
    if torch.equal(actual, expected):
        return {"bitwise": True, "max_abs_diff": 0.0}
    difference = (actual.float() - expected.float()).abs()
    worst = int(torch.argmax(difference).item())
    raise AssertionError(
        "fastcu_nvfp4_gemm_gb300 bitwise mismatch against frozen gemm9: "
        f"differing={int((actual != expected).sum().item())}, "
        f"max_abs_diff={float(difference.max().item())}, "
        f"actual={float(actual.reshape(-1)[worst].item())}, "
        f"expected={float(expected.reshape(-1)[worst].item())}, "
        f"flat_index={worst}"
    )


def run_test(**config: Any):
    """Compile and compare one deterministic configuration to frozen gemm9."""
    import torch

    M, N, K_dim = (int(config[name]) for name in ("M", "N", "K"))
    data = prepare_data(M, N, K_dim)
    data["C"].fill_(float("nan"))
    data["C_source"].fill_(float("nan"))
    source = _SourceRunner(data, M, N, K_dim)
    try:
        _runner()(data, M, N, K_dim)
        source()
        torch.cuda.synchronize()
        if int(data["placement"].item()) != 0 or source.placement_errors() != 0:
            raise AssertionError("source/TIRx placement audit failed")
        result = _check_bitwise(data, M, N)
        _check_guards(data)
        data["C"].fill_(float("nan"))
        _runner()(data, M, N, K_dim)
        torch.cuda.synchronize()
        if not torch.equal(data["C"], data["C_source"]):
            raise AssertionError("TIRx repeat launch is not bitwise deterministic")
        if int(data["placement"].item()) != 0:
            raise AssertionError("TIRx repeat-launch placement audit failed")
        _check_guards(data)
        return result
    finally:
        source.close()


def prepare_bench(**config: Any):
    """Compile the TIRx kernel before GPU benchmark setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    M, N, K_dim = (int(config[name]) for name in ("M", "N", "K"))
    state = {"config": {"M": M, "N": N, "K": K_dim}, "runner": _runner()}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **_):
    """Benchmark one TIRx launch and optionally the pinned gemm9 launch."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    M, N, K_dim = (prepared["config"][name] for name in ("M", "N", "K"))
    data = prepare_data(M, N, K_dim)
    data["C"].fill_(float("nan"))
    data["C_source"].fill_(float("nan"))
    source = _SourceRunner(data, M, N, K_dim)

    def tirx_launch():
        return prepared["runner"](data, M, N, K_dim)

    try:
        tirx_launch()
        with_source = external_references_enabled()
        references = None
        if with_source:
            source()
            references = {"fastcu_gemm9": lambda: source}
        torch.cuda.synchronize()
        if int(data["placement"].item()) != 0:
            raise AssertionError("TIRx placement audit failed before timing")
        if not bool(torch.isfinite(data["C"]).all()):
            raise AssertionError("TIRx output contains non-finite or poison values")
        if with_source:
            if source.placement_errors() != 0:
                raise AssertionError("source placement audit failed before timing")
            _check_bitwise(data, M, N)
        _check_guards(data)
        result = bench(
            {"tirx": tirx_launch},
            references=references,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            rounds=rounds,
            cooldown_s=cooldown_s,
        )
        torch.cuda.synchronize()
        if int(data["placement"].item()) != 0:
            raise AssertionError("TIRx placement audit failed during timing")
        if with_source and source.placement_errors() != 0:
            raise AssertionError("source placement audit failed during timing")
        _check_guards(data)
        return result
    finally:
        source.close()


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config: Any):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "make_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
