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

import tirx_kernels.tirx_lite as txl
import tvm

KERNEL_META = {
    "name": "fastcu_nvfp4_gemm_gb300",
    "category": "fastcu",
    "runtime_cuda_archs": ["sm_103a"],
}

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
# K=96 consumes six block16 scales, not four fixed scale vectors.
_TCGEN05_MMA = "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16"
_TCGEN05_COMMIT = (
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
)
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"


def _ring_next(slot, phase, size):
    wrap = txl.local_scalar("uint32", init=txl.cast(slot == size - 1, "uint32"))
    next_slot = txl.local_scalar("int32", init=txl.Select(wrap != 0, txl.int32(0), slot + 1))
    next_phase = txl.local_scalar("uint32", init=phase ^ wrap)
    return next_slot, next_phase


def _ring_prev(slot, phase, size):
    at_zero = txl.local_scalar("uint32", init=txl.cast(slot == 0, "uint32"))
    prev_slot = txl.local_scalar("int32", init=txl.Select(at_zero != 0, txl.int32(size - 1), slot - 1))
    prev_phase = txl.local_scalar("uint32", init=txl.Select(at_zero != 0, phase, phase ^ txl.uint32(1)))
    return prev_slot, prev_phase


def _wait_plain(barrier, phase):
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, txl.cast(phase, "uint32"), txl.uint32(_TRY_WAIT_TICKS)
        )


def _wait_acquire_cta(barrier, phase):
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, barrier, txl.cast(phase, "uint32"), txl.uint32(_TRY_WAIT_TICKS)
        )


def _bit_or64(*values):
    value = txl.uint64(0)
    for item in values:
        value = txl.bitwise_or(value, txl.cast(item, "uint64"))
    return value


def _ab_desc(smbase, region, slot, atom):
    base = smbase + txl.uint32(region) + txl.cast(slot * _AB_STRIDE, "uint32")
    addr = txl.cast(
        txl.bitwise_and(txl.shift_right(base, txl.uint32(4)), txl.uint32(0x7FC0)) + txl.uint32(atom), "uint64"
    )
    lbo = txl.cast(txl.bitwise_and(txl.shift_left(base, txl.uint32(12)), txl.uint32(0x7FC00000)), "uint64")
    return _bit_or64(txl.uint64(0x4010404000000000), addr, lbo)


def _ab_desc_straddle(smbase, region, addr_slot, lbo_slot, atom):
    d_addr = _ab_desc(smbase, region, addr_slot, atom)
    d_lbo = _ab_desc(smbase, region, lbo_slot, 0)
    return _bit_or64(
        txl.bitwise_and(d_addr, txl.bitwise_not(txl.uint64(0x00000000FFFF0000))),
        txl.bitwise_and(d_lbo, txl.uint64(0x00000000FFFF0000)),
    )


def _ab_desc_k64(smbase, region, slot, atom):
    return txl.bitwise_and(
        _ab_desc(smbase, region, slot, atom), txl.bitwise_not(txl.uint64(0x00100000FFFF0000))
    )


def _ab_desc_low(smbase, region, slot, atom):
    """Low word of the source K96 shared-memory operand descriptor."""
    base = smbase + txl.uint32(region) + txl.cast(slot * _AB_STRIDE, "uint32")
    return txl.bitwise_or(
        txl.bitwise_and(txl.shift_right(base, txl.uint32(4)), txl.uint32(0x7FC0)) + txl.uint32(atom),
        txl.bitwise_and(txl.shift_left(base, txl.uint32(12)), txl.uint32(0x7FC00000)),
    )


def _ab_desc_low_straddle(smbase, region, addr_slot, lbo_slot, atom):
    addr = _ab_desc_low(smbase, region, addr_slot, atom)
    lbo = _ab_desc_low(smbase, region, lbo_slot, 0)
    return txl.bitwise_or(
        txl.bitwise_and(addr, txl.uint32(0x0000FFFF)), txl.bitwise_and(lbo, txl.uint32(0xFFFF0000))
    )


def _join_desc(high, low):
    return txl.bitwise_or(txl.shift_left(txl.cast(high, "uint64"), txl.uint32(32)), txl.cast(low, "uint64"))


def _sf_cp_desc(smbase, region, slot, slot_units, tile):
    base = smbase + txl.uint32(region)
    addr = txl.cast(txl.bitwise_and(txl.shift_right(base, txl.uint32(4)), txl.uint32(0x3FC0)), "uint64")
    desc = _bit_or64(txl.uint64(0x400800010000), addr)
    return desc + txl.cast(slot * slot_units + tile * 32, "uint64")


def _sf_id_bits(sfa, sfb):
    return txl.bitwise_or(
        txl.bitwise_and(txl.shift_right(sfa, txl.uint32(1)), txl.uint32(0x60000000)),
        txl.bitwise_and(txl.shift_right(sfb, txl.uint32(26)), txl.uint32(48)),
    )


def _half_from_word(word, half):
    shifted = txl.shift_right(word, txl.cast(half * 16, "uint32"))
    return txl.cast(txl.bitwise_and(shifted, txl.uint32(0xFFFF)), "uint16")


def _udiv(x, divisor):
    """Truncating division for values known nonnegative in the source contract."""
    return txl.cast(txl.cast(x, "uint32") // txl.cast(divisor, "uint32"), "int32")


def _umod(x, divisor):
    """Remainder paired with :func:`_udiv`, without signed floor fixup."""
    return txl.cast(txl.cast(x, "uint32") % txl.cast(divisor, "uint32"), "int32")


def _uceil(x, divisor):
    """Ceiling division for a nonnegative value and positive divisor."""
    du = txl.cast(divisor, "uint32")
    return txl.cast((txl.cast(x, "uint32") + du - txl.uint32(1)) // du, "int32")


@functools.lru_cache(maxsize=1)
def make_kernel():
    """Build the fixed-topology r9 kernel with runtime M/N/K."""

    @txl.kernel(warps=7, arch="sm_103a", min_blocks_per_sm=1, grid=(2, 1, _NUM_CLUSTERS))
    def fastcu_nvfp4_gemm_gb300_kernel(
        A_tmap: txl.TensorMap,
        B_tmap: txl.TensorMap,
        SFA_tmap: txl.TensorMap,
        SFB_tmap: txl.TensorMap,
        C: txl.gptr[txl.f16],
        M: txl.i32,
        N: txl.i32,
        K_dim: txl.i32,
        route_table: txl.gptr[txl.i32],
        sm_side: txl.gptr[txl.i32],
        cluster_side: txl.gptr[txl.i32],
        placement_errors: txl.gptr[txl.u32],
    ):
        crank = txl.cta_id_in_cluster([2], preferred=[2])
        _, _, cluster_id = txl.cta_id()
        tid = txl.thread_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        m_pair = txl.cast(crank, "int32") & txl.int32(1)

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=[0, 1, 2, 3])
        mma_role = roles.role("mma", warps=[4], when=m_pair == 0)
        ab_role = roles.role("ab_tma", warps=[5])
        sf_role = roles.role("sf_tma", warps=[6])

        smem = txl.alloc_buffer((_SMEM_BYTES,), txl.u8, scope="shared.dyn", align=1024)
        smbase = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(smem.ptr_to([0])))

        cluster_grid_m = txl.local_scalar("int32", init=_uceil(M, txl.int32(256)))
        grid_n = txl.local_scalar("int32", init=_uceil(N, txl.int32(256)))
        total_tiles = txl.local_scalar("int32", init=cluster_grid_m * grid_n)
        full_groups = txl.local_scalar("int32", init=_udiv(K_dim, txl.int32(768)))
        k_rem = txl.local_scalar("int32", init=K_dim - full_groups * 768)
        tail_cells = txl.local_scalar("int32", init=_uceil(k_rem, txl.int32(96)))
        t_sf = txl.local_scalar("int32", init=_uceil(tail_cells, txl.int32(2)))
        num_groups = txl.local_scalar("int32", init=full_groups + txl.cast(tail_cells != 0, "int32"))
        k_rem_mod96 = txl.local_scalar("int32", init=_umod(k_rem, txl.int32(96)))
        b64_try = txl.local_scalar(
            "int32",
            init=txl.Select(
                k_rem_mod96 == 64, txl.int32(1), txl.Select(k_rem_mod96 == 32, txl.int32(2), txl.int32(0))
            ),
        )
        a64 = txl.local_scalar("int32", init=tail_cells - b64_try)
        b64 = txl.local_scalar(
            "int32",
            init=txl.Select(
                (b64_try != 0)
                & (_umod(k_rem, txl.int32(32)) == 0)
                & (a64 >= 0)
                & (_umod(a64, txl.int32(2)) == 0),
                b64_try,
                txl.int32(0),
            ),
        )
        t_win = txl.local_scalar(
            "int32",
            init=txl.Select(b64 != 0, _uceil(_udiv(k_rem, txl.int32(2)), txl.int32(128)), txl.int32(3)),
        )

        smid = txl.local_scalar("uint32", init=txl.cuda.mov_sreg(32, "smid"))
        with txl.If(tid == 0), txl.Then():
            actual_side = txl.local_scalar("int32")
            planned_side = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(actual_side, sm_side.ptr_to([smid]))
            txl.ptx.ld.global_.nc.s32(planned_side, cluster_side.ptr_to([cluster_id]))
            with txl.If(actual_side != planned_side), txl.Then():
                old = txl.local_scalar("uint32")
                txl.ptx.atom.global_.add.u32(old, placement_errors.ptr_to([0]), txl.uint32(1))

        with txl.If((warp == 0) & (lane == 0)), txl.Then():
            for stage in range(6):
                txl.ptx.mbarrier.init.shared.b64(
                    txl.ptr_byte_offset(smem.ptr_to([0]), _AB_READY + stage * 8, "uint64"),
                    txl.uint32(1),
                )
                txl.ptx.mbarrier.init.shared.b64(
                    txl.ptr_byte_offset(smem.ptr_to([0]), _AB_FREE + stage * 8, "uint64"), txl.uint32(1)
                )
            for stage in range(7):
                txl.ptx.mbarrier.init.shared.b64(
                    txl.ptr_byte_offset(smem.ptr_to([0]), _SF_READY + stage * 8, "uint64"),
                    txl.uint32(1),
                )
                txl.ptx.mbarrier.init.shared.b64(
                    txl.ptr_byte_offset(smem.ptr_to([0]), _SF_FREE + stage * 8, "uint64"), txl.uint32(1)
                )
            txl.ptx.mbarrier.init.shared.b64(
                txl.ptr_byte_offset(smem.ptr_to([0]), _ACC_READY, "uint64"), txl.uint32(1)
            )
            txl.ptx.mbarrier.init.shared.b64(
                txl.ptr_byte_offset(smem.ptr_to([0]), _ACC_FREE, "uint64"), txl.uint32(8)
            )
            txl.ptx.mbarrier.init.shared.b64(
                txl.ptr_byte_offset(smem.ptr_to([0]), _DEALLOC, "uint64"), txl.uint32(32)
            )
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.barrier.cluster.arrive.release.aligned()
        txl.ptx.barrier.cluster.wait.acquire.aligned()

        taddr = txl.local_scalar("uint32", init=txl.uint32(0))
        with txl.If(warp <= 4), txl.Then():
            with txl.If(warp == 0), txl.Then():
                txl.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                    smbase + txl.uint32(_TMEM_ADDR), txl.uint32(512)
                )
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(160))
            txl.ptx.ld.shared.b32(taddr, txl.ptr_byte_offset(smem.ptr_to([0]), _TMEM_ADDR, "uint32"))

        with ab_role:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                slot = txl.local_scalar("int32", init=txl.int32(0))
                phase = txl.local_scalar("uint32", init=txl.uint32(1))
                work = txl.local_scalar("int32", init=cluster_id)
                with txl.While(work < total_tiles):
                    tile = txl.local_scalar("int32")
                    txl.ptx.ld.global_.ca.s32(tile, route_table.ptr_to([work]))
                    m_row = txl.local_scalar("int32", init=_umod(tile, cluster_grid_m))
                    n_group = txl.local_scalar("int32", init=_udiv(tile, cluster_grid_m))
                    m_block = txl.local_scalar("int32", init=m_row * 2 + m_pair)
                    n_block = txl.local_scalar("int32", init=n_group * 2 + m_pair)
                    group = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(group < num_groups):
                        for sub in range(3):
                            with txl.If((group < full_groups) | (sub < t_win)), txl.Then():
                                _wait_acquire_cta(
                                    smbase + txl.uint32(_AB_FREE) + txl.cast(slot * 8, "uint32"), phase
                                )
                                mbar = txl.bitwise_and(
                                    smbase + txl.uint32(_AB_READY) + txl.cast(slot * 8, "uint32"),
                                    txl.uint32(0xFEFFFFFF),
                                )
                                with txl.If(m_pair == 0), txl.Then():
                                    txl.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                                        mbar, txl.uint32(65536)
                                    )
                                x = group * 384 + sub * 128
                                a_dst = (
                                    smbase + txl.uint32(_A_BASE) + txl.cast(slot * _AB_STRIDE, "uint32")
                                )
                                b_dst = (
                                    smbase + txl.uint32(_B_BASE) + txl.cast(slot * _AB_STRIDE, "uint32")
                                )
                                txl.ptx[_TMA_3D_MCAST](
                                    a_dst,
                                    txl.address_of(A_tmap),
                                    txl.cast(x, "int32"),
                                    txl.cast(m_block * 128, "int32"),
                                    txl.int32(0),
                                    mbar,
                                    txl.cast(txl.int32(1) << m_pair, "uint16"),
                                )
                                txl.ptx[_TMA_3D](
                                    b_dst,
                                    txl.address_of(B_tmap),
                                    txl.cast(x, "int32"),
                                    txl.cast(n_block * 128, "int32"),
                                    txl.int32(0),
                                    mbar,
                                )
                                next_slot, next_phase = _ring_next(slot, phase, 6)
                                txl.assign(slot, next_slot)
                                txl.assign(phase, next_phase)
                        txl.assign(group, group + 1)
                    txl.assign(work, work + _NUM_CLUSTERS)
                last_slot, last_phase = _ring_prev(slot, phase, 6)
                _wait_acquire_cta(
                    smbase + txl.uint32(_AB_FREE) + txl.cast(last_slot * 8, "uint32"), last_phase
                )
                with txl.If(m_pair == 0), txl.Then():
                    mbar = txl.bitwise_and(
                        smbase + txl.uint32(_AB_READY) + txl.cast(last_slot * 8, "uint32"),
                        txl.uint32(0xFEFFFFFF),
                    )
                    txl.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                        mbar, txl.uint32(65536)
                    )

        with sf_role:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                slot = txl.local_scalar("int32", init=txl.int32(0))
                phase = txl.local_scalar("uint32", init=txl.uint32(1))
                work = txl.local_scalar("int32", init=cluster_id)
                with txl.While(work < total_tiles):
                    tile = txl.local_scalar("int32")
                    txl.ptx.ld.global_.ca.s32(tile, route_table.ptr_to([work]))
                    m_row = txl.local_scalar("int32", init=_umod(tile, cluster_grid_m))
                    n_group = txl.local_scalar("int32", init=_udiv(tile, cluster_grid_m))
                    m_block = txl.local_scalar("int32", init=m_row * 2 + m_pair)
                    group = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(group < num_groups):
                        for sub in range(4):
                            with txl.If((group < full_groups) | (sub < t_sf)), txl.Then():
                                _wait_acquire_cta(
                                    smbase + txl.uint32(_SF_FREE) + txl.cast(slot * 8, "uint32"), phase
                                )
                                mbar = txl.bitwise_and(
                                    smbase + txl.uint32(_SF_READY) + txl.cast(slot * 8, "uint32"),
                                    txl.uint32(0xFEFFFFFF),
                                )
                                with txl.If(m_pair == 0), txl.Then():
                                    txl.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                                        mbar, txl.uint32(9216)
                                    )
                                sf_group = (group * 4 + sub) * 3
                                sfa_dst = (
                                    smbase
                                    + txl.uint32(_SFA_BASE)
                                    + txl.cast(slot * _SFA_STRIDE, "uint32")
                                )
                                sfb_dst = (
                                    smbase
                                    + txl.uint32(_SFB_BASE)
                                    + txl.cast(slot * _SFB_STRIDE + m_pair * 1536, "uint32")
                                )
                                txl.ptx[_TMA_4D_MCAST](
                                    sfa_dst,
                                    txl.address_of(SFA_tmap),
                                    txl.int32(0),
                                    txl.int32(0),
                                    txl.cast(sf_group, "int32"),
                                    txl.cast(m_block, "int32"),
                                    mbar,
                                    txl.cast(txl.int32(1) << m_pair, "uint16"),
                                )
                                txl.ptx[_TMA_4D_MCAST](
                                    sfb_dst,
                                    txl.address_of(SFB_tmap),
                                    txl.int32(0),
                                    txl.int32(0),
                                    txl.cast(sf_group, "int32"),
                                    txl.cast(n_group * 2 + m_pair, "int32"),
                                    mbar,
                                    txl.uint16(3),
                                )
                                next_slot, next_phase = _ring_next(slot, phase, 7)
                                txl.assign(slot, next_slot)
                                txl.assign(phase, next_phase)
                        txl.assign(group, group + 1)
                    txl.assign(work, work + _NUM_CLUSTERS)
                last_slot, last_phase = _ring_prev(slot, phase, 7)
                _wait_acquire_cta(
                    smbase + txl.uint32(_SF_FREE) + txl.cast(last_slot * 8, "uint32"), last_phase
                )
                with txl.If(m_pair == 0), txl.Then():
                    mbar = txl.bitwise_and(
                        smbase + txl.uint32(_SF_READY) + txl.cast(last_slot * 8, "uint32"),
                        txl.uint32(0xFEFFFFFF),
                    )
                    txl.ptx["mbarrier.arrive.expect_tx.release.cta.shared::cta.b64"](
                        mbar, txl.uint32(9216)
                    )

        with mma_role:
            with txl.If(txl.cuda.elect_sync() != txl.uint32(0)), txl.Then():
                d_phase = txl.local_scalar("uint32", init=txl.uint32(1))
                ab_slot = txl.local_scalar("int32", init=txl.int32(0))
                ab_phase = txl.local_scalar("uint32", init=txl.uint32(0))
                sf_slot = txl.local_scalar("int32", init=txl.int32(0))
                sf_phase = txl.local_scalar("uint32", init=txl.uint32(0))
                work = txl.local_scalar("int32", init=cluster_id)

                # gemm9 hoists these invariant descriptor halves and TMEM
                # addresses out of its persistent tile loop. Keeping the same
                # shape here prevents repeated 64-bit reconstruction between
                # dependent tcgen05 issues.
                sfa_src = txl.local_scalar("uint64", init=_sf_cp_desc(smbase, _SFA_BASE, 0, 0, 0))
                sfb_src = txl.local_scalar("uint64", init=_sf_cp_desc(smbase, _SFB_BASE, 0, 0, 0))
                sfa_t0 = txl.local_scalar("uint32", init=taddr + txl.uint32(476))
                sfa_t1 = txl.local_scalar("uint32", init=taddr + txl.uint32(480))
                sfb_t0 = txl.local_scalar("uint32", init=taddr + txl.uint32(488))
                sfb_t1 = txl.local_scalar("uint32", init=taddr + txl.uint32(496))
                sfa_w1 = txl.local_scalar("uint32", init=txl.bitwise_or(sfa_t1, txl.uint32(0x80000000)))
                sfb_w1 = txl.local_scalar("uint32", init=txl.bitwise_or(sfb_t1, txl.uint32(0x80000000)))
                idesc_w0 = txl.local_scalar(
                    "uint32", init=txl.bitwise_or(txl.uint32(0x90400480), _sf_id_bits(sfa_t0, sfb_t0))
                )
                idesc_w1 = txl.local_scalar(
                    "uint32", init=txl.bitwise_or(txl.uint32(0x90400480), _sf_id_bits(sfa_w1, sfb_w1))
                )

                def stage_scale(slot, phase):
                    _wait_plain(smbase + txl.uint32(_SF_READY) + txl.cast(slot * 8, "uint32"), phase)
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
                        txl.ptx[_TCGEN05_CP](
                            taddr + txl.uint32(dst), src + txl.cast(slot * stride + tile * 32, "uint64")
                        )
                    txl.ptx[_TCGEN05_COMMIT](
                        smbase + txl.uint32(_SF_FREE) + txl.cast(slot * 8, "uint32"), txl.uint16(3)
                    )
                    return _ring_next(slot, phase, 7)

                def commit_ab(slot):
                    txl.ptx[_TCGEN05_COMMIT](
                        smbase + txl.uint32(_AB_FREE) + txl.cast(slot * 8, "uint32"), txl.uint16(3)
                    )

                def issue96(cell, W0, W1, W2, d_tmem, accum, issue_pred=None):
                    if cell == 0:
                        da = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W0, 0))
                        db = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W0, 0))
                    elif cell == 1:
                        da = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W0, 3))
                        db = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W0, 3))
                    elif cell == 2:
                        da = _join_desc(
                            txl.uint32(0x40104040), _ab_desc_low_straddle(smbase, _A_BASE, W0, W1, 6)
                        )
                        db = _join_desc(
                            txl.uint32(0x40104040), _ab_desc_low_straddle(smbase, _B_BASE, W0, W1, 6)
                        )
                    elif cell == 3:
                        da = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W1, 1))
                        db = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W1, 1))
                    elif cell == 4:
                        da = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W1, 4))
                        db = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W1, 4))
                    elif cell == 5:
                        da = _join_desc(
                            txl.uint32(0x40104040), _ab_desc_low_straddle(smbase, _A_BASE, W1, W2, 7)
                        )
                        db = _join_desc(
                            txl.uint32(0x40104040), _ab_desc_low_straddle(smbase, _B_BASE, W1, W2, 7)
                        )
                    elif cell == 6:
                        da = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W2, 2))
                        db = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W2, 2))
                    else:
                        da = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _A_BASE, W2, 5))
                        db = _join_desc(txl.uint32(0x40104040), _ab_desc_low(smbase, _B_BASE, W2, 5))
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
                        txl.ptx.pred(txl.cast(accum != 0, "uint32")),
                    )
                    if issue_pred is None:
                        txl.ptx[_TCGEN05_MMA](*args)
                    else:
                        with txl.If(issue_pred), txl.Then():
                            txl.ptx[_TCGEN05_MMA](*args)

                def issue64(slot, atom, word, d_tmem, accum):
                    da = _join_desc(
                        txl.uint32(0x40004040),
                        txl.bitwise_and(
                            _ab_desc_low(smbase, _A_BASE, slot, atom), txl.uint32(0x0000FFFF)
                        ),
                    )
                    db = _join_desc(
                        txl.uint32(0x40004040),
                        txl.bitwise_and(
                            _ab_desc_low(smbase, _B_BASE, slot, atom), txl.uint32(0x0000FFFF)
                        ),
                    )
                    if word == 0:
                        sfa = sfa_t0
                        sfb = sfb_t0
                    else:
                        sfa = sfa_t1
                        sfb = sfb_t1
                    idesc = txl.bitwise_or(txl.uint32(0x10400480), _sf_id_bits(sfa, sfb))
                    txl.ptx[_TCGEN05_MMA](
                        d_tmem, da, db, idesc, sfa, sfb, txl.ptx.pred(txl.cast(accum != 0, "uint32"))
                    )

                def issue_k64_variant(a_count, b_count, d_tmem, accum, need_wait, acc_wait):
                    W0 = txl.local_scalar("int32", init=ab_slot)
                    ph0 = txl.local_scalar("uint32", init=ab_phase)
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
                            txl.assign(accum, txl.uint32(1))

                    ns, np = stage_scale(sf_slot, sf_phase)
                    txl.assign(sf_slot, ns)
                    txl.assign(sf_phase, np)
                    _wait_plain(smbase + txl.uint32(_AB_READY) + txl.cast(W0 * 8, "uint32"), ph0)
                    with txl.If(need_wait != 0), txl.Then():
                        _wait_plain(smbase + txl.uint32(_ACC_FREE), acc_wait)
                        txl.assign(need_wait, txl.uint32(0))
                    issue_at(0)
                    issue_at(1)

                    if n_cells > 2:
                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                    if t_windows >= 2:
                        _wait_plain(smbase + txl.uint32(_AB_READY) + txl.cast(W1 * 8, "uint32"), ph1)
                    issue_at(2)
                    commit_ab(W0)
                    issue_at(3)

                    if n_cells > 4:
                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                    issue_at(4)
                    if t_windows >= 3:
                        _wait_plain(smbase + txl.uint32(_AB_READY) + txl.cast(W2 * 8, "uint32"), ph2)
                    if t_windows == 1:
                        txl.assign(ab_slot, W1)
                        txl.assign(ab_phase, ph1)
                    elif t_windows == 2:
                        txl.assign(ab_slot, W2)
                        txl.assign(ab_phase, ph2)
                    else:
                        ns_ab, np_ab = _ring_next(W2, ph2, 6)
                        txl.assign(ab_slot, ns_ab)
                        txl.assign(ab_phase, np_ab)
                    issue_at(5)

                    if n_cells > 6:
                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                    if t_windows >= 2:
                        commit_ab(W1)
                    issue_at(6)
                    issue_at(7)
                    if t_windows >= 3:
                        commit_ab(W2)

                with txl.While(work < total_tiles):
                    acc_wait = txl.local_scalar("uint32", init=d_phase)
                    txl.assign(d_phase, d_phase ^ txl.uint32(1))
                    d_tmem = taddr + d_phase * txl.uint32(220)
                    accum = txl.local_scalar("uint32", init=txl.uint32(0))
                    need_wait = txl.local_scalar("uint32", init=txl.uint32(1))
                    group = txl.local_scalar("int32", init=txl.int32(0))

                    with txl.While(group < full_groups):
                        W0 = txl.local_scalar("int32", init=ab_slot)
                        ph0 = txl.local_scalar("uint32", init=ab_phase)
                        W1, ph1 = _ring_next(W0, ph0, 6)
                        W2, ph2 = _ring_next(W1, ph1, 6)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                        _wait_plain(smbase + txl.uint32(_AB_READY) + txl.cast(W0 * 8, "uint32"), ph0)
                        with txl.If(need_wait != 0), txl.Then():
                            _wait_plain(smbase + txl.uint32(_ACC_FREE), acc_wait)
                            txl.assign(need_wait, txl.uint32(0))
                        issue96(0, W0, W1, W2, d_tmem, accum)
                        txl.assign(accum, txl.uint32(1))
                        issue96(1, W0, W1, W2, d_tmem, accum)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                        _wait_plain(smbase + txl.uint32(_AB_READY) + txl.cast(W1 * 8, "uint32"), ph1)
                        issue96(2, W0, W1, W2, d_tmem, accum)
                        commit_ab(W0)
                        issue96(3, W0, W1, W2, d_tmem, accum)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                        issue96(4, W0, W1, W2, d_tmem, accum)
                        _wait_plain(smbase + txl.uint32(_AB_READY) + txl.cast(W2 * 8, "uint32"), ph2)
                        ns_ab, np_ab = _ring_next(W2, ph2, 6)
                        txl.assign(ab_slot, ns_ab)
                        txl.assign(ab_phase, np_ab)
                        issue96(5, W0, W1, W2, d_tmem, accum)

                        ns, np = stage_scale(sf_slot, sf_phase)
                        txl.assign(sf_slot, ns)
                        txl.assign(sf_phase, np)
                        commit_ab(W1)
                        issue96(6, W0, W1, W2, d_tmem, accum)
                        issue96(7, W0, W1, W2, d_tmem, accum)
                        commit_ab(W2)
                        txl.assign(group, group + 1)

                    with txl.If(tail_cells != 0), txl.Then():
                        with txl.If(b64 != 0), txl.Then():
                            for ac in (0, 2, 4, 6):
                                for bc in (1, 2):
                                    with txl.If((a64 == ac) & (b64 == bc)), txl.Then():
                                        issue_k64_variant(
                                            ac, bc, d_tmem, accum, need_wait, acc_wait
                                        )
                        with txl.If(b64 == 0), txl.Then():
                            W0 = txl.local_scalar("int32", init=ab_slot)
                            ph0 = txl.local_scalar("uint32", init=ab_phase)
                            W1, ph1 = _ring_next(W0, ph0, 6)
                            W2, ph2 = _ring_next(W1, ph1, 6)

                            ns, np = stage_scale(sf_slot, sf_phase)
                            txl.assign(sf_slot, ns)
                            txl.assign(sf_phase, np)
                            _wait_plain(
                                smbase + txl.uint32(_AB_READY) + txl.cast(W0 * 8, "uint32"), ph0
                            )
                            with txl.If(need_wait != 0), txl.Then():
                                _wait_plain(smbase + txl.uint32(_ACC_FREE), acc_wait)
                                txl.assign(need_wait, txl.uint32(0))
                            issue96(0, W0, W1, W2, d_tmem, accum)
                            txl.assign(accum, txl.uint32(1))
                            issue96(1, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 2)

                            with txl.If(tail_cells > 2), txl.Then():
                                ns, np = stage_scale(sf_slot, sf_phase)
                                txl.assign(sf_slot, ns)
                                txl.assign(sf_phase, np)
                            _wait_plain(
                                smbase + txl.uint32(_AB_READY) + txl.cast(W1 * 8, "uint32"), ph1
                            )
                            issue96(2, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 3)
                            commit_ab(W0)
                            issue96(3, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 4)

                            with txl.If(tail_cells > 4), txl.Then():
                                ns, np = stage_scale(sf_slot, sf_phase)
                                txl.assign(sf_slot, ns)
                                txl.assign(sf_phase, np)
                            issue96(4, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 5)
                            _wait_plain(
                                smbase + txl.uint32(_AB_READY) + txl.cast(W2 * 8, "uint32"), ph2
                            )
                            ns_ab, np_ab = _ring_next(W2, ph2, 6)
                            txl.assign(ab_slot, ns_ab)
                            txl.assign(ab_phase, np_ab)
                            issue96(5, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 6)

                            with txl.If(tail_cells > 6), txl.Then():
                                ns, np = stage_scale(sf_slot, sf_phase)
                                txl.assign(sf_slot, ns)
                                txl.assign(sf_phase, np)
                            commit_ab(W1)
                            issue96(6, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 7)
                            issue96(7, W0, W1, W2, d_tmem, accum, issue_pred=tail_cells >= 8)
                            commit_ab(W2)

                    txl.ptx[_TCGEN05_COMMIT](smbase + txl.uint32(_ACC_READY), txl.uint16(3))
                    txl.assign(work, work + _NUM_CLUSTERS)
                _wait_plain(smbase + txl.uint32(_ACC_FREE), d_phase)

        with epilogue_role:
            d_buffer = txl.local_scalar("int32", init=txl.int32(0))
            acc_phase = txl.local_scalar("uint32", init=txl.uint32(0))
            work = txl.local_scalar("int32", init=cluster_id)
            values = txl.alloc_local((32,), "float32")
            packed = txl.alloc_local((16,), "uint32")
            taddr_lane = txl.local_scalar(
                "uint32", init=txl.shift_left(txl.cast(warp * 32, "uint32"), txl.uint32(16))
            )
            even_crank = txl.local_scalar(
                "uint32", init=txl.bitwise_and(txl.cast(crank, "uint32"), txl.uint32(0xFFFFFFFE))
            )
            with txl.While(work < total_tiles):
                tile_lane = txl.local_scalar("int32")
                txl.ptx.ld.global_.ca.s32(tile_lane, route_table.ptr_to([work]))
                tile = txl.local_scalar("uint32")
                txl.ptx.redux_sync.min.u32(tile, txl.cast(tile_lane, "uint32"), txl.uint32(0xFFFFFFFF))
                m_row = txl.local_scalar("int32", init=_umod(tile, cluster_grid_m))
                n_group = txl.local_scalar("int32", init=_udiv(tile, cluster_grid_m))
                off_n = txl.local_scalar("int32", init=n_group * 256)
                _wait_plain(smbase + txl.uint32(_ACC_READY), acc_phase)
                txl.assign(acc_phase, acc_phase ^ txl.uint32(1))
                tmem_base = txl.local_scalar("uint32", init=taddr + txl.cast(d_buffer * 220, "uint32"))
                row = txl.local_scalar("int32", init=(m_row * 2 + m_pair) * 128 + warp * 32 + lane)
                row_in = txl.local_scalar("uint32", init=txl.cast(row < M, "uint32"))
                row_base = txl.local_scalar("int64", init=txl.cast(row, "int64") * N)
                c_addr = txl.reinterpret("uint64", C.ptr_to([0]))
                aligned = txl.local_scalar(
                    "uint32",
                    init=txl.cast(
                        (txl.bitwise_and(row_base, txl.int32(15)) == 0)
                        & (txl.bitwise_and(c_addr, txl.uint64(31)) == 0),
                        "uint32",
                    ),
                )
                with txl.serial(0, 8) as k:
                    band = txl.Select(d_buffer == 0, 7 - k, k)
                    load_addr = tmem_base + txl.cast(band * 32, "uint32") + taddr_lane
                    txl.ptx[_TMEM_LD_X32](*[values[i] for i in range(32)], load_addr)
                    with txl.If(k == 1), txl.Then():
                        mapped_free = txl.local_scalar("uint32")
                        txl.ptx.mapa.shared__cluster.u32(
                            mapped_free, smbase + txl.uint32(_ACC_FREE), even_crank
                        )
                        txl.ptx.tcgen05.wait__ld.sync.aligned()
                        txl.ptx["mbarrier.arrive.release.cta.shared::cluster.b64"](
                            mapped_free, txl.uint32(1), pred=txl.cast(lane == 0, "uint32")
                        )
                    with txl.If(row_in != 0), txl.Then():
                        for pair in range(16):
                            txl.ptx.cvt.rn.f16x2.f32(
                                packed[pair], values[pair * 2 + 1], values[pair * 2]
                            )
                        col = txl.local_scalar("int32", init=off_n + band * 32)
                        with txl.If(aligned != 0), txl.Then():
                            for q in range(2):
                                c16 = col + q * 16
                                with txl.If(c16 + 16 <= N), txl.Then():
                                    txl.ptx["st.global.L1::no_allocate.L2::evict_first.v8.b32"](
                                        C.ptr_to([row_base + c16]),
                                        *[packed[q * 8 + i] for i in range(8)],
                                    )
                                with txl.If(c16 + 16 > N), txl.Then():
                                    for p in range(2):
                                        c8 = c16 + p * 8
                                        with txl.If(c8 + 8 <= N), txl.Then():
                                            txl.ptx["st.global.L1::no_allocate.v4.b32"](
                                                C.ptr_to([row_base + c8]),
                                                *[packed[q * 8 + p * 4 + i] for i in range(4)],
                                            )
                                        with txl.If(c8 + 8 > N), txl.Then():
                                            for h in range(8):
                                                word = q * 8 + p * 4 + h // 2
                                                txl.ptx.st.global_.b16(
                                                    C.ptr_to([row_base + c8 + h]),
                                                    _half_from_word(packed[word], h % 2),
                                                    pred=txl.cast(c8 + h < N, "uint32"),
                                                )
                        with txl.If(aligned == 0), txl.Then():
                            for q in range(4):
                                c8 = col + q * 8
                                with txl.If(c8 + 8 <= N), txl.Then():
                                    txl.ptx["st.global.L1::no_allocate.v4.b32"](
                                        C.ptr_to([row_base + c8]),
                                        *[packed[q * 4 + i] for i in range(4)],
                                    )
                                with txl.If(c8 + 8 > N), txl.Then():
                                    for h in range(8):
                                        word = q * 4 + h // 2
                                        txl.ptx.st.global_.b16(
                                            C.ptr_to([row_base + c8 + h]),
                                            _half_from_word(packed[word], h % 2),
                                            pred=txl.cast(c8 + h < N, "uint32"),
                                        )
                txl.assign(d_buffer, txl.int32(1) - d_buffer)
                txl.assign(work, work + _NUM_CLUSTERS)

            txl.ptx.bar.sync(txl.uint32(3), txl.uint32(128))
            with txl.If(warp == 0), txl.Then():
                txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
                remote_dealloc = txl.local_scalar("uint32")
                txl.ptx.mapa.shared__cluster.u32(
                    remote_dealloc,
                    smbase + txl.uint32(_DEALLOC),
                    txl.cast(crank, "uint32") ^ txl.uint32(1),
                )
                txl.ptx["mbarrier.arrive.release.cta.shared::cluster.b64"](
                    remote_dealloc, txl.uint32(1)
                )
                _wait_acquire_cta(smbase + txl.uint32(_DEALLOC), txl.uint32(0))
                txl.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](taddr, txl.uint32(512))

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
    candidates = (Path(override),) if override else (repo_root / ".reference-deps" / "fast-cu",)
    for root in candidates:
        if (root / _SOURCE_RELATIVE).is_file():
            return root
    tried = ", ".join(str(root) for root in candidates)
    raise RuntimeError(
        f"fast.cu source is unavailable; set FASTCU_PATH to a checkout (tried: {tried})"
    )


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
