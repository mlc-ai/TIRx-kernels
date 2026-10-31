# This file is a TIRx port of code from flash-attention
# (https://github.com/Dao-AILab/flash-attention @ 00756db9), Copyright (c) 2022,
# the respective contributors, as shown by licenses/AUTHORS.flash-attention.txt
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors


"""FlashAttention-4 in native TIRx TVMScript and CUDA tile operations.

This lifts the existing FA4 implementation without changing its 128x128 tiles,
warp roles, pipeline depths, packed floating-point instructions or softmax order.
TMA, MMA, TMEM, shared accesses and supported register instructions use Tx.apply.
Warp collectives, synchronization and mask bit operations remain scalar TIRx/PTX.
Tensor maps are encoded on the host and passed in the original seven-map ABI.

The initial variant targets B200 (sm_100a), FP16, head dimension 128 and GQA
ratios 1/2/4/8. The existing flash_attention4 module remains the reference.
"""

import ctypes
import math
from typing import Any

import torch
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tirx_kernels.runner import bench, resolve_backend_config
from tvm_tirx.backend.cuda.axis import TCol, TLane
from tvm_tirx.backend.cuda.axis import tid_in_wg as AX_THREAD
from tvm_tirx.backend.cuda.lang import SMEMPool
from tvm_tirx.backend.cuda.lang.pipeline import MBarrier, Pipeline, PipelineState, TCGen05Bar
from tvm_tirx.backend.cuda.lang.tile_scheduler import (
    FlashAttentionLinearScheduler,
    FlashAttentionLPTScheduler,
)
from tvm_tirx.backend.cuda.layout import SwizzleMode, mma_shared_layout
from tvm_tirx.backend.cuda.ptx import instruction_forms
from tvm_tirx.backend.cuda.ptx.table import operand_dtypes, operand_layout
from tvm_tirx.backend.cuda.script import tile as Tx
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout

N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
BLK_M = 128
BLK_N = 128
SOFTMAX_LD_CHUNK = 32
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False
MMA_N = 128
MMA_K = 16
F16_BYTES = 2
EMU_PAIRS_CAUSAL = 2
EMU_START_CAUSAL = 0
EMU_PAIRS_NC = 2
EMU_START_NC = 1
MAX_CTAS = 148
TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
)
TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMEM_LD_16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
MAX3_F32 = "max.f32"


def _shared_instruction(opcode, *values):
    load = opcode.startswith("ld.")
    dtype = {"f32": "float32", "u32": "uint32"}[opcode.split(".")[-1]]
    count = 4 if ".v4." in opcode else 1
    pointer = values[-1] if load else values[0]
    pointer = T.reinterpret(pointer, ty=T.Ptr(dtype, pointer.ty.scope))
    shared = T.bind_(
        T.decl_tensor(
            pointer,
            ty_args=[
                T.Tensor(
                    (count,),
                    dtype,
                    scope=pointer.ty.scope,
                    align=4 * count,
                    layout=TileLayout(S[count,]),
                )
            ],
        )
    )
    regs = _register_group(values[:-1] if load else values[1:], input_only=not load)
    dst, src = (regs, shared) if load else (shared, regs)
    T.evaluate(Tx.apply(T.ptx[opcode], dst, src, scope="thread"))


def _register_group(values, *, input_only=False):
    """Expose one PTX register group as a native, thread-owned tensor view."""
    first = values[0]
    if all(
        isinstance(v, tvm_tirx.ir.TensorLoad) and v.source.same_as(first.source) for v in values
    ):
        base = first.indices[0]
        analyzer = tvm_tirx.sym.Analyzer()
        if all(
            (
                isinstance((d := analyzer.simplify(v.indices[0] - base - i)), tvm_tirx.ir.IntImm)
                and int(d) == 0
                for i, v in enumerate(values)
            )
        ):
            return T.bind_(
                T.decl_tensor(
                    T.address_of(first),
                    ty_args=[
                        T.Tensor(
                            (len(values),),
                            first.ty,
                            scope="local",
                            layout=TileLayout(S[len(values),]),
                        )
                    ],
                )
            )
    if not input_only:
        raise ValueError("Writable register groups must occupy consecutive local slots")
    temp = T.bind_(
        T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (len(values),), first.ty, scope="local", layout=TileLayout(S[len(values),])
                )
            ]
        )
    )
    for i, v in enumerate(values):
        T.assign(temp[i], v)
    return temp


def _register_instruction(opcode, *values):
    """Retain PTX rounding/packing while exposing its operands to tile lowering."""
    candidates = []
    for entry, modifiers in instruction_forms(opcode):
        groups = operand_layout(entry, modifiers)
        if sum((n for _, _, n in groups)) != len(values):
            continue
        if all(
            (
                str(v.ty.dtype) in operand_dtypes(slot, modifiers)
                for slot, start, n in groups
                for v in values[start : start + n]
            )
        ):
            candidates.append(groups)
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous register shape: {opcode}")
    groups = []
    for slot, start, n in candidates[0]:
        args = values[start : start + n]
        groups.append(
            args[0]
            if slot.rw == "r" and n == 1
            else _register_group(args, input_only=slot.rw == "r")
        )
    T.evaluate(Tx.apply(T.ptx[opcode], *groups, scope="thread"))


def ceildiv(a, b):
    return (a + b - 1) // b


def make_kernel(
    BATCH_SIZE,
    SEQ_LEN_Q,
    SEQ_LEN_KV,
    NUM_QO_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    is_causal=False,
    tmem_pipe_depth=TMEM_PIPE_DEPTH,
    smem_pipe_depth_kv=SMEM_PIPE_DEPTH_KV,
    *,
    backend_config=None,
):
    """Specialize the native TVMScript kernel; preserve the FA4 schedule."""
    TMEM_DEPTH = tmem_pipe_depth
    IS_THOR = False
    USE_2CTA = False and (not is_causal)
    cta_group = 1
    KV_DEPTH = smem_pipe_depth_kv * cta_group
    KV_ROWS = BLK_N // cta_group
    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS
    SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
    STATS_BAR_PAIRWISE = GQA_RATIO == 1
    L2_SIZE = 50 * 1024 * 1024
    SIZE_ONE_KV_HEAD = SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES
    L2_SWIZZLE = (
        1 if L2_SIZE < SIZE_ONE_KV_HEAD else 1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD))
    )
    SSCALE_TOTAL_SIZE = 2 * SMEM_PIPE_DEPTH_Q * BLK_M
    assert TMEM_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"
    num_q_blocks_total = ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE)
    num_q_blocks = ceildiv(num_q_blocks_total, SMEM_PIPE_DEPTH_Q * cta_group)
    num_total_tasks = BATCH_SIZE * NUM_KV_HEADS * num_q_blocks
    num_kv_blocks = ceildiv(SEQ_LEN_KV, BLK_N)
    EPI_ON_SOFTMAX = is_causal
    EARLY_Q_RELEASE = not is_causal
    max_ctas = MAX_CTAS
    cta_count = num_total_tasks if is_causal else min(max_ctas, num_total_tasks)
    scheduler_ctas = cta_count // cta_group
    thor_causal_d128 = False and is_causal and (HEAD_DIM == 128)
    K_SPLIT = (4 if is_causal else 6) * MMA_K
    P_SPLIT_Q = 2 if is_causal else 3
    ACC_SCALE_BASE = 0
    ROW_SUM_BASE = 0
    scale_log2 = math.log2(math.e) / math.sqrt(HEAD_DIM)
    rescale_threshold = 8.0
    NEG_INF = -float("inf")
    tma_g2s_3d = TMA_G2S_3D
    tma_g2s_4d = TMA_G2S_4D
    mma_f16 = MMA_F16
    tmem_alloc = TMEM_ALLOC
    tmem_dealloc = TMEM_DEALLOC
    tmem_relinquish = TMEM_RELINQUISH

    def n_block_max_of(m_block_idx):
        nbm = num_kv_blocks
        if not is_causal:
            return nbm
        m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
        return T.min(nbm, ceildiv(n_idx, BLK_N))

    def n_block_min_causal_of(m_block_idx):
        m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
        return T.max(0, n_idx // BLK_N)

    kv_base = SMEM_PIPE_DEPTH_Q * BLK_M * HEAD_DIM * F16_BYTES
    EPI_LD_SM = 32
    RESCALE_TILE = 16

    @T.function
    def flash_attention4(
        Q_tensor_map: T.Ptr(T.TensorMap()),
        Q_tensor_map_1: T.Ptr(T.TensorMap()),
        K_tensor_map: T.Ptr(T.TensorMap()),
        K_tensor_map_1: T.Ptr(T.TensorMap()),
        V_tensor_map: T.Ptr(T.TensorMap()),
        V_tensor_map_1: T.Ptr(T.TensorMap()),
        O_tensor_map: T.Ptr(T.TensorMap()),
    ):
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=cta_count, block=512),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=resolve_backend_config(backend_config),
        ):

            def mma_shared_at(tensor, stage):
                return T.bind_(
                    T.decl_tensor(
                        tensor.ptr_to([stage, 0, 0]),
                        ty_args=[
                            T.Tensor(
                                (128, 128),
                                "float16",
                                scope="shared.dyn",
                                align=1024,
                                layout=mma_shared_layout(
                                    "float16", SwizzleMode.SWIZZLE_128B_ATOM, (128, 128)
                                ),
                            )
                        ],
                    )
                )

            def mma_tmem(column, dtype, columns):
                return T.bind_(
                    T.cuda.decl_tmem(
                        column,
                        ty_args=[
                            T.Tensor(
                                (128, columns),
                                dtype,
                                scope="tmem",
                                layout=TileLayout(S[(128, columns) : (1 @ TLane, 1 @ TCol)]),
                            )
                        ],
                    )
                ).view(1, 1, 128, columns)

            def mma_pv(i_q, kv_stage, ki, accum):
                d = mma_tmem((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N, "float32", 128)
                a = mma_tmem(i_q * MMA_N + MMA_N // 2 + ki * (MMA_K // 2), "float16", 16)
                b = (
                    mma_shared_at(v_smem, kv_stage)
                    .view(8, 16, 128)
                    .permute(0, 2, 1)
                    .view(1, 8, 128, 16)[:, ki : ki + 1, :, :]
                )
                T.evaluate(
                    Tx.apply(
                        T.ptx[MMA_F16],
                        d,
                        a,
                        b,
                        instruction_shape=(128, 128, 16),
                        b_major="mn",
                        accum=accum,
                        scope="thread",
                    )
                )

            def tma_view(pointer, box):
                layout = ComposeLayout(3, 3, 3, TileLayout(S[box]))
                return T.bind_(
                    T.decl_tensor(
                        pointer,
                        ty_args=[
                            T.Tensor(box, "float16", scope="shared.dyn", align=1024, layout=layout)
                        ],
                    )
                )

            def tma_load(opcode, pointer, tmap, coordinates, mbar):
                box = (
                    (2, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM // 2)
                    if len(coordinates) == 4
                    else (2, BLK_N, HEAD_DIM // 2)
                )
                shared = tma_view(pointer, box)
                T.evaluate(
                    Tx.apply(
                        T.ptx[opcode],
                        shared,
                        tmap,
                        box=box,
                        coords=tuple(reversed(coordinates)),
                        swizzle=3,
                        l2_promotion=2,
                        mbar=mbar,
                        scope="thread",
                    )
                )

            def tma_store(opcode, tmap, coordinates, pointer):
                box = (
                    (2, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM // 2)
                    if len(coordinates) == 4
                    else (2, BLK_M, HEAD_DIM // 2)
                )
                shared = tma_view(pointer, box)
                T.evaluate(
                    Tx.apply(
                        T.ptx[opcode],
                        tmap,
                        shared,
                        box=box,
                        coords=tuple(reversed(coordinates)),
                        swizzle=3,
                        l2_promotion=2,
                        scope="thread",
                    )
                )

            cta_rank = T.int32(0)
            cluster_id = T.cuda.block_idx("x") // cta_group
            warp_cta = T.cuda.warp_id()
            wg_id = warp_cta >> 2
            warp_id = warp_cta & 3
            tid_in_wg = T.cuda.linear_thread_id() & 127
            smem = SMEMPool()
            q_smem = smem.alloc_tcgen05_mma_AB(
                (SMEM_PIPE_DEPTH_Q, BLK_M * cta_group, HEAD_DIM // cta_group),
                "float16",
                swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM,
            )
            k_smem = smem.alloc_tcgen05_mma_AB(
                (KV_DEPTH, BLK_N, HEAD_DIM // cta_group),
                "float16",
                swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM,
            )
            smem.move_base_to(kv_base)
            v_smem = smem.alloc_tcgen05_mma_AB(
                (KV_DEPTH, BLK_N, HEAD_DIM // cta_group),
                "float16",
                swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM,
            )
            smem.move_base_to(kv_base + KV_DEPTH * KV_ROWS * HEAD_DIM * F16_BYTES)
            o_smem = smem.alloc_tcgen05_mma_AB(
                (TMEM_DEPTH, BLK_M, HEAD_DIM), "float16", swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM
            )
            sScale = smem.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
            tmem_addr = smem.alloc((1,), "uint32")
            kv_pipe = PipelineState(KV_DEPTH, phase=0)
            softmax_epoch = PipelineState(1, phase=0)
            score_epoch = PipelineState(1, phase=0)
            tmem_epoch = PipelineState(1, phase=0)
            q_epoch = PipelineState(1, phase=0)
            o_epi_epoch = PipelineState(1, phase=0)
            q_load = Pipeline(
                smem, SMEM_PIPE_DEPTH_Q, full="tma", empty="tcgen05", empty_phase_offset=1
            )
            kv_load = Pipeline(smem, KV_DEPTH, full="tma", empty="tcgen05", empty_phase_offset=1)
            p_o_rescale = MBarrier(smem, 2)
            p_o_rescale.init(256 * cta_group)
            s_ready = TCGen05Bar(smem, 2)
            s_ready.init(1)
            o_ready = TCGen05Bar(smem, 2)
            o_ready.init(1)
            softmax_corr = Pipeline(
                smem,
                2,
                full="mbar",
                empty="mbar",
                init_full=128,
                init_empty=128,
                empty_phase_offset=1,
            )
            corr_epi = Pipeline(
                smem,
                TMEM_DEPTH,
                full="mbar",
                empty="mbar",
                init_full=128,
                init_empty=32,
                empty_phase_offset=1,
            )
            p_ready_2 = MBarrier(smem, 2)
            p_ready_2.init(128 * cta_group)
            bar_s0_s1_sequence = MBarrier(smem, 8)
            bar_s0_s1_sequence.init(32)
            smem.commit()
            q_load_full_remote = q_load.full
            kv_load_full_remote = kv_load.full
            p_o_rescale_remote = p_o_rescale
            p_ready_2_remote = p_ready_2
            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.fence.mbarrier_init.release.cluster()
            T.cuda.cta_sync()

            def tmem(col):
                return T.cast(col, ty="uint32")

            def tmem_view(address, width):
                column = address
                return T.bind_(
                    T.cuda.decl_tmem(
                        column,
                        ty_args=[
                            T.Tensor(
                                (128, width),
                                "uint32",
                                scope="tmem",
                                layout=TileLayout(S[(128, width) : (1 @ TLane, 1 @ TCol)]),
                            )
                        ],
                    )
                )

            def register_view(tensor, offset, width):
                return T.bind_(
                    T.decl_tensor(
                        tensor.ptr_to([offset]),
                        ty_args=[
                            T.Tensor(
                                (128, width),
                                tensor.ty.dtype,
                                scope="local",
                                layout=TileLayout(S[(128, width) : (1 @ AX_THREAD, 1)]),
                            )
                        ],
                    )
                )

            def tmem_load(dst, offset, address, width):
                r = register_view(dst, offset, width)
                t = tmem_view(address, width)
                T.evaluate(
                    Tx.apply(
                        T.ptx[TMEM_LD_16 if width == 16 else TMEM_LD_32],
                        r.view(4, 32, width),
                        t.view(4, 32, width),
                        scope="warpgroup",
                    )
                )

            def tmem_store(src, offset, address):
                r = register_view(src, offset, 16)
                t = tmem_view(address, 16)
                T.evaluate(
                    Tx.apply(
                        T.ptx[TMEM_ST_16], t.view(4, 32, 16), r.view(4, 32, 16), scope="warpgroup"
                    )
                )

            def elected():
                return T.cuda.elect_sync() != T.uint32(0)

            def iket_range(name, *, leader_only=False):
                token = T.bind_(T.alloc_local([1], "uint32"))
                if leader_only:
                    T.assign(token[0], T.cuda.iket.sentinel_token(name))
                    with T.if_(warp_id == 0), T.then_():
                        T.assign(token[0], T.cuda.iket.range_start(name))
                else:
                    T.assign(token[0], T.cuda.iket.range_start(name))
                return token

            def cast_f32x2_f16x2(dst_u32, src, offset):
                """One packed f32x2 -> f16x2 conversion — orig:L95-97.

                Every consumer uses packed uint32 words, so keep that register view
                directly."""
                _register_instruction(
                    "cvt.rn.f16x2.f32", dst_u32[offset // 2], src[offset + 1], src[offset]
                )

            def fma_f32x2(values, idx, multiplier, addend_value):
                """A packed f32x2 operand is one 64-bit value."""
                packed = T.bind_(T.local_scalar("uint64"))
                rhs = T.bind_(T.local_scalar("uint64"))
                addend = T.bind_(T.local_scalar("uint64"))
                _register_instruction("mov.b64", packed, values[idx], values[idx + 1])
                _register_instruction("mov.b64", rhs, multiplier, multiplier)
                _register_instruction("mov.b64", addend, addend_value, addend_value)
                _register_instruction("fma.rz.ftz.f32x2", packed, packed, rhs, addend)
                _register_instruction("mov.b64", values[idx], values[idx + 1], packed)

            def mul_f32x2(values, idx, multiplier):
                """orig:L215-223."""
                packed = T.bind_(T.local_scalar("uint64"))
                rhs = T.bind_(T.local_scalar("uint64"))
                _register_instruction("mov.b64", packed, values[idx], values[idx + 1])
                _register_instruction("mov.b64", rhs, multiplier, multiplier)
                _register_instruction("mul.rz.ftz.f32x2", packed, packed, rhs)
                _register_instruction("mov.b64", values[idx], values[idx + 1], packed)

            def reduce_max_128(out, values, accum=False):
                """SM100 three-input max tree — orig:L226-244. The outer walk stays a
                real T.serial(15) loop; only the four-element inner fan is expanded."""
                temp = T.bind_(T.alloc_local([4], "float32"))
                for i in range(4):
                    if accum and i == 0:
                        _register_instruction(
                            MAX3_F32, temp[i], values[2 * i], values[2 * i + 1], out[0]
                        )
                    else:
                        _register_instruction(
                            "mov.b32", temp[i], T.max(values[2 * i], values[2 * i + 1])
                        )
                with T.serial(15) as outer:
                    for i in range(4):
                        _register_instruction(
                            MAX3_F32,
                            temp[i],
                            temp[i],
                            values[8 * (outer + 1) + 2 * i],
                            values[8 * (outer + 1) + 2 * i + 1],
                        )
                T.assign(out[0], T.max(temp[0], temp[1]))
                _register_instruction(MAX3_F32, out[0], out[0], temp[2], temp[3])

            def reduce_sum_128(out, values, accum=False):
                """Packed add tree, accumulator insertion order preserved — orig:L247-276."""
                local_sum = T.bind_(T.alloc_local([8], "float32"))
                packed = T.bind_(T.local_scalar("uint64"))
                rhs = T.bind_(T.local_scalar("uint64"))
                for i in range(8):
                    if accum and i == 0:
                        _register_instruction("mov.b32", local_sum[i], values[i] + out[0])
                    else:
                        _register_instruction("mov.b32", local_sum[i], values[i])
                with T.serial(15) as outer:
                    for i in range(4):
                        _register_instruction(
                            "mov.b64", packed, local_sum[2 * i], local_sum[2 * i + 1]
                        )
                        _register_instruction(
                            "mov.b64",
                            rhs,
                            values[8 * (outer + 1) + 2 * i],
                            values[8 * (outer + 1) + 2 * i + 1],
                        )
                        _register_instruction("add.rn.ftz.f32x2", packed, packed, rhs)
                        _register_instruction(
                            "mov.b64", local_sum[2 * i], local_sum[2 * i + 1], packed
                        )
                for lo, hi in ((0, 2), (4, 6), (0, 4)):
                    _register_instruction("mov.b64", packed, local_sum[lo], local_sum[lo + 1])
                    _register_instruction("mov.b64", rhs, local_sum[hi], local_sum[hi + 1])
                    _register_instruction("add.rn.ftz.f32x2", packed, packed, rhs)
                    _register_instruction("mov.b64", local_sum[lo], local_sum[lo + 1], packed)
                T.assign(out[0], local_sum[0] + local_sum[1])

            def shl_u32_clamp(val, shift):
                """Left shift with PTX clamping (shift>=32 -> 0) — orig:L114-121."""
                result = T.bind_(T.local_scalar("uint32"))
                T.ptx.shl.b32(result, val, shift)
                return result

            def combine_int_frac_ex2(x_rounded, frac_ex2):
                """orig:L124-135."""
                x_rounded_i = T.bind_(T.local_scalar("int32"))
                frac_ex_i = T.bind_(T.local_scalar("int32"))
                x_rounded_e = T.bind_(T.local_scalar("int32"))
                out_i = T.bind_(T.local_scalar("int32"))
                out = T.bind_(T.local_scalar("float32"))
                _register_instruction("mov.b32", x_rounded_i, x_rounded)
                _register_instruction("mov.b32", frac_ex_i, frac_ex2)
                T.ptx.shl.b32(x_rounded_e, x_rounded_i, T.uint32(23))
                _register_instruction("add.s32", out_i, x_rounded_e, frac_ex_i)
                _register_instruction("mov.b32", out, out_i)
                return out

            POLY_EX2_DEG3 = (1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872)
            FP32_ROUND_INT = float(2**23 + 2**22)

            def ex2_emulation_2(out, idx, x, y):
                """Two-lane polynomial ex2 on the packed f32x2 datapath — orig:L155-199."""
                xy_clamped = T.bind_(T.alloc_local([2], "float32"))
                _register_instruction("mov.b32", xy_clamped[0], T.max(x, -127.0))
                _register_instruction("mov.b32", xy_clamped[1], T.max(y, -127.0))
                packed = T.bind_(T.local_scalar("uint64"))
                rhs = T.bind_(T.local_scalar("uint64"))
                addend = T.bind_(T.local_scalar("uint64"))
                xy_rounded = T.bind_(T.alloc_local([2], "float32"))
                _register_instruction("mov.b64", packed, xy_clamped[0], xy_clamped[1])
                _register_instruction(
                    "mov.b64", rhs, T.float32(FP32_ROUND_INT), T.float32(FP32_ROUND_INT)
                )
                _register_instruction("add.rm.ftz.f32x2", packed, packed, rhs)
                _register_instruction("mov.b64", xy_rounded[0], xy_rounded[1], packed)
                xy_rounded_back = T.bind_(T.alloc_local([2], "float32"))
                _register_instruction("mov.b64", packed, xy_rounded[0], xy_rounded[1])
                _register_instruction(
                    "mov.b64", rhs, T.float32(FP32_ROUND_INT), T.float32(FP32_ROUND_INT)
                )
                _register_instruction("sub.rn.ftz.f32x2", packed, packed, rhs)
                _register_instruction("mov.b64", xy_rounded_back[0], xy_rounded_back[1], packed)
                xy_frac = T.bind_(T.alloc_local([2], "float32"))
                _register_instruction("mov.b64", packed, xy_clamped[0], xy_clamped[1])
                _register_instruction("mov.b64", rhs, xy_rounded_back[0], xy_rounded_back[1])
                _register_instruction("sub.rn.ftz.f32x2", packed, packed, rhs)
                _register_instruction("mov.b64", xy_frac[0], xy_frac[1], packed)
                xy_frac_ex2 = T.bind_(T.alloc_local([2], "float32"))
                _register_instruction("mov.b32", xy_frac_ex2[0], T.float32(POLY_EX2_DEG3[3]))
                _register_instruction("mov.b32", xy_frac_ex2[1], T.float32(POLY_EX2_DEG3[3]))
                for coeff in (POLY_EX2_DEG3[2], POLY_EX2_DEG3[1], POLY_EX2_DEG3[0]):
                    _register_instruction("mov.b64", rhs, xy_frac[0], xy_frac[1])
                    _register_instruction("mov.b64", packed, xy_frac_ex2[0], xy_frac_ex2[1])
                    _register_instruction("mov.b64", addend, T.float32(coeff), T.float32(coeff))
                    _register_instruction("fma.rz.ftz.f32x2", packed, packed, rhs, addend)
                    _register_instruction("mov.b64", xy_frac_ex2[0], xy_frac_ex2[1], packed)
                _register_instruction(
                    "mov.b32", out[idx], combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
                )
                _register_instruction(
                    "mov.b32", out[idx + 1], combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])
                )

            scheduler = (
                FlashAttentionLPTScheduler(
                    "fa_scheduler",
                    num_batches=BATCH_SIZE,
                    num_heads=NUM_KV_HEADS,
                    num_m_blocks=num_q_blocks,
                    l2_swizzle=L2_SWIZZLE,
                )
                if T.constexpr(is_causal)
                else FlashAttentionLinearScheduler(
                    "fa_scheduler",
                    num_batches=BATCH_SIZE,
                    num_heads=NUM_KV_HEADS,
                    num_m_blocks=num_q_blocks,
                    num_ctas=scheduler_ctas,
                )
            )
            scheduler.init(cluster_id)
            if warp_cta == 12:
                T.ptx[tmem_alloc](T.address_of(tmem_addr[0]), T.uint32(N_COLS_TMEM))
                T.cuda.warp_sync()
            if tvm_tirx.tirx.all(wg_id == 3, warp_id == 0):
                allocated = T.local_scalar("uint32")
                _shared_instruction("ld.shared.u32", allocated, tmem_addr.ptr_to([0]))
                T.cuda.trap_when_assert_failed(allocated == T.uint32(0))
            if wg_id == 2:
                for i_q in T.unroll(2):
                    p_o_rescale_remote.arrive(i_q)
            while scheduler.valid():
                m_block_idx = scheduler.m_block_idx
                batch_idx = scheduler.batch_idx
                kv_head_idx = scheduler.head_idx

                def q_tile_start(i_q):
                    return (
                        m_block_idx * SMEM_PIPE_DEPTH_Q * cta_group + i_q * cta_group + cta_rank
                    ) * SEQ_Q_PER_TILE

                if (warp_cta >= 12) & (warp_cta <= 15):
                    T.ptx.setmaxnreg.dec.sync.aligned.u32(48)
                    if warp_cta % 4 == 1:

                        def load_q(i_q, tensor_map):
                            q_load.empty.wait(i_q, q_epoch.phase)
                            tma_q_token = iket_range("issue-tma-q")
                            with T.if_(elected()), T.then_():
                                for k_part in range(cta_group):
                                    if GQA_RATIO == 1:
                                        tma_load(
                                            tma_g2s_3d,
                                            q_smem.ptr_to([i_q, k_part * BLK_M, 0]),
                                            T.address_of(tensor_map),
                                            (
                                                T.int32(0),
                                                T.cast(q_tile_start(i_q), ty="int32"),
                                                T.cast(
                                                    (batch_idx * NUM_QO_HEADS + kv_head_idx) * 2
                                                    + 0,
                                                    ty="int32",
                                                ),
                                            ),
                                            T.cuda.cvta_generic_to_shared(
                                                q_load_full_remote.ptr_to([i_q])
                                            ),
                                        )
                                    else:
                                        tma_load(
                                            tma_g2s_4d,
                                            q_smem.ptr_to([i_q, k_part * BLK_M, 0]),
                                            T.address_of(tensor_map),
                                            (
                                                T.int32(0),
                                                T.cast(kv_head_idx * GQA_RATIO, ty="int32"),
                                                T.cast(q_tile_start(i_q), ty="int32"),
                                                T.cast(batch_idx * 2 + 0, ty="int32"),
                                            ),
                                            T.cuda.cvta_generic_to_shared(
                                                q_load_full_remote.ptr_to([i_q])
                                            ),
                                        )
                                q_load_full_remote.arrive(
                                    i_q, tx_count=cta_group * BLK_M * HEAD_DIM * F16_BYTES
                                )
                            T.cuda.iket.range_end(tma_q_token[0])

                        def load_kv(i_kv, tensor_map, event, is_v=False):
                            """One K or V box into the shared KV ring — orig:L800-840.
                            The two loads are the same instruction on the same ring;
                            only the tensormap differs."""
                            kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                            tma_kv_token = iket_range(event)
                            with T.if_(elected()), T.then_():
                                for k_part in range(1 if is_v else cta_group):
                                    tma_load(
                                        tma_g2s_3d,
                                        (v_smem if is_v else k_smem).ptr_to(
                                            [kv_pipe.stage, 0 if is_v else k_part * KV_ROWS, 0]
                                        ),
                                        T.address_of(tensor_map),
                                        (
                                            T.int32(0),
                                            T.cast(
                                                i_kv * BLK_N + cta_rank * KV_ROWS
                                                if not is_v
                                                else i_kv * BLK_N,
                                                ty="int32",
                                            ),
                                            T.cast(
                                                (batch_idx * NUM_KV_HEADS + kv_head_idx) * 2
                                                + (cta_rank if is_v and False else 0),
                                                ty="int32",
                                            ),
                                        ),
                                        T.cuda.cvta_generic_to_shared(
                                            kv_load_full_remote.ptr_to([kv_pipe.stage])
                                        ),
                                    )
                                kv_load_full_remote.arrive(
                                    kv_pipe.stage,
                                    tx_count=cta_group * KV_ROWS * HEAD_DIM * F16_BYTES,
                                )
                            T.cuda.iket.range_end(tma_kv_token[0])
                            kv_pipe.advance()

                        load_trip_count = T.local_scalar("int32")
                        T.assign(
                            load_trip_count,
                            n_block_max_of(m_block_idx)
                            if T.constexpr(is_causal)
                            else num_kv_blocks,
                        )
                        load_q(0, Q_tensor_map)
                        load_kv(load_trip_count - 1, K_tensor_map, "issue-tma-k")
                        load_q(1, Q_tensor_map_1)
                        q_epoch.advance()
                        load_kv(load_trip_count - 1, V_tensor_map, "issue-tma-v", is_v=True)
                        for _i in T.serial(load_trip_count - 1, unroll=False):
                            i_kv = load_trip_count - 2 - _i
                            load_kv(i_kv, K_tensor_map_1, "issue-tma-k")
                            load_kv(i_kv, V_tensor_map_1, "issue-tma-v", is_v=True)
                    elif warp_cta % 4 == 2:
                        corr_epi.full.wait(0, tmem_epoch.phase)
                        tma_store_token = iket_range("tma-store")
                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                            if i_q != 0:
                                corr_epi.full.wait(i_q, tmem_epoch.phase)
                            m_start_global = q_tile_start(i_q)
                            if elected():
                                if T.constexpr(GQA_RATIO == 1):
                                    tma_store(
                                        TMA_S2G_3D,
                                        T.address_of(O_tensor_map),
                                        (
                                            T.int32(0),
                                            T.cast(m_start_global, ty="int32"),
                                            T.cast(
                                                (batch_idx * NUM_QO_HEADS + kv_head_idx) * 2,
                                                ty="int32",
                                            ),
                                        ),
                                        o_smem.ptr_to([i_q, 0, 0]),
                                    )
                                else:
                                    tma_store(
                                        TMA_S2G_4D,
                                        T.address_of(O_tensor_map),
                                        (
                                            T.int32(0),
                                            T.cast(kv_head_idx * GQA_RATIO, ty="int32"),
                                            T.cast(m_start_global, ty="int32"),
                                            T.cast(batch_idx * 2, ty="int32"),
                                        ),
                                        o_smem.ptr_to([i_q, 0, 0]),
                                    )
                            T.ptx.cp.async_.bulk.commit_group()
                        T.ptx.cp.async_.bulk.wait_group(1)
                        corr_epi.empty.arrive(0)
                        T.ptx.cp.async_.bulk.wait_group(0)
                        corr_epi.empty.arrive(1)
                        T.cuda.iket.range_end(tma_store_token[0])
                        tmem_epoch.advance()
                    elif warp_cta % 4 == 0:
                        acc = T.local_scalar("int32", init=0)

                        def gemm_qk(q_stage, kv_stage):
                            for ki in range(HEAD_DIM // MMA_K):
                                with T.if_(elected()), T.then_():
                                    d = mma_tmem(q_stage * MMA_N, "float32", 128)
                                    a = (
                                        mma_shared_at(q_smem, q_stage)
                                        .view(128, 8, 16)
                                        .permute(1, 0, 2)
                                        .view(1, 8, 128, 16)[:, ki : ki + 1, :, :]
                                    )
                                    b = (
                                        mma_shared_at(k_smem, kv_stage)
                                        .view(128, 8, 16)
                                        .permute(1, 0, 2)
                                        .view(1, 8, 128, 16)[:, ki : ki + 1, :, :]
                                    )
                                    T.evaluate(
                                        Tx.apply(
                                            T.ptx[MMA_F16],
                                            d,
                                            a,
                                            b,
                                            instruction_shape=(128, 128, 16),
                                            accum=ki != 0,
                                            scope="thread",
                                        )
                                    )
                            with T.if_(elected()), T.then_():
                                s_ready.arrive(q_stage)

                        def gemm_pv_part1(i_q, kv_stage, should_accumulate):
                            for ki in range(K_SPLIT // MMA_K):
                                with T.if_(elected()), T.then_():
                                    mma_pv(
                                        i_q,
                                        kv_stage,
                                        ki,
                                        True if ki != 0 else T.cast(should_accumulate, ty="bool"),
                                    )

                        def gemm_pv_part2(i_q, kv_stage):
                            p_ready_2.wait(i_q, tmem_epoch.phase)
                            for ki in range(K_SPLIT // MMA_K, BLK_N // MMA_K):
                                with T.if_(elected()), T.then_():
                                    mma_pv(i_q, kv_stage, ki, True)

                        def gemm_pv(i_q, kv_stage, should_accumulate):
                            gemm_pv_part1(i_q, kv_stage, should_accumulate)
                            gemm_pv_part2(i_q, kv_stage)

                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                            q_load.full.wait(i_q, q_epoch.phase)
                            if i_q == 0:
                                kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                            gemm_qk(i_q, kv_pipe.stage)
                            if i_q == 1:
                                if elected():
                                    kv_load.empty.arrive(
                                        kv_pipe.stage, cta_group=cta_group, cta_mask=None
                                    )
                        kv_pipe.advance()
                        mma_trip_count = T.local_scalar("int32")
                        T.assign(
                            mma_trip_count,
                            n_block_max_of(m_block_idx)
                            if T.constexpr(is_causal)
                            else num_kv_blocks,
                        )
                        for i_kv in T.serial(mma_trip_count - 1, unroll=False):
                            stage_v = T.local_scalar("int32", init=kv_pipe.stage)
                            phase_v = T.local_scalar("int32", init=kv_pipe.phase)
                            kv_pipe.advance()
                            for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                                if i_q == 0:
                                    kv_load.full.wait(stage_v, phase_v)
                                p_o_rescale.wait(i_q, tmem_epoch.phase)
                                gemm_pv(i_q, stage_v, acc)
                                if i_q == 1:
                                    if elected():
                                        kv_load.empty.arrive(
                                            stage_v, cta_group=cta_group, cta_mask=None
                                        )
                                if i_q == 0:
                                    kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                                gemm_qk(i_q, kv_pipe.stage)
                                if T.constexpr(EARLY_Q_RELEASE):
                                    if i_kv == mma_trip_count - 2:
                                        if elected():
                                            q_load.empty.arrive(
                                                i_q, cta_group=cta_group, cta_mask=None
                                            )
                                if i_q == 1:
                                    if elected():
                                        kv_load.empty.arrive(
                                            kv_pipe.stage, cta_group=cta_group, cta_mask=None
                                        )
                            T.assign(acc, 1)
                            kv_pipe.advance()
                            tmem_epoch.advance()
                        for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                            if i_q == 0:
                                kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                            p_o_rescale.wait(i_q, tmem_epoch.phase)
                            gemm_pv(i_q, kv_pipe.stage, acc)
                            if i_q == 1:
                                if elected():
                                    kv_load.empty.arrive(
                                        kv_pipe.stage, cta_group=cta_group, cta_mask=None
                                    )
                            if elected():
                                o_ready.arrive(i_q)
                        kv_pipe.advance()
                        tmem_epoch.advance()
                        if T.constexpr(not EARLY_Q_RELEASE):
                            for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                                if elected():
                                    q_load.empty.arrive(i_q, cta_group=cta_group, cta_mask=None)
                        q_epoch.advance()
                    elif warp_cta % 4 == 3:
                        pass
                elif (warp_cta >= 0) & (warp_cta <= 7):
                    T.ptx.setmaxnreg.inc.sync.aligned.u32(200)
                    row_max = T.local_scalar("float32")
                    row_sum = T.alloc_local([1], "float32")
                    if warp_id == 0:
                        T.cuda.iket.mark("softmax-baseline")

                    def mask_r2p(s_chunk, col_limit, ncol):
                        """R2P-style column mask — orig:L1069-1103.

                        ~(0xFFFFFFFF << k) is the low-k-bits mask with no `-1`; the
                        ~ fuses into the per-column `& (1<<i)` test as ANDN, and shl
                        clamping (k>=32 -> 0) removes the min. The bit test compiles
                        to R2P."""
                        CHUNK_SIZE = 32
                        for s in range(ceildiv(ncol, CHUNK_SIZE)):
                            k_keep = T.max(col_limit - s * CHUNK_SIZE, 0)
                            mask_inv = T.bind_(T.local_scalar("uint32"))
                            T.assign(
                                mask_inv,
                                shl_u32_clamp(T.uint32(4294967295), T.cast(k_keep, ty="uint32")),
                            )
                            for i in range(CHUNK_SIZE):
                                if i < ncol - s * CHUNK_SIZE:
                                    c = s * CHUNK_SIZE + i
                                    in_bound = T.bitwise_and(
                                        T.bitwise_not(mask_inv),
                                        T.shift_left(T.uint32(1), T.uint32(i)),
                                    )
                                    _register_instruction(
                                        "mov.b32",
                                        s_chunk[c],
                                        T.Select(
                                            T.cast(in_bound, ty="bool"),
                                            s_chunk[c],
                                            T.float32(NEG_INF),
                                        ),
                                    )

                    def apply_causal_mask(s_chunk, m_blk_idx, n_blk_idx):
                        """orig:L1105-1129. `col_limit_right` is read four times
                        (once per 32-column mask chunk); the intermediates are plain
                        values, so ptxas sees the row/offset arithmetic directly."""
                        seq_pos_in_wg = tid_in_wg // GQA_RATIO
                        row_idx = (
                            m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
                            + wg_id * SEQ_Q_PER_TILE
                            + seq_pos_in_wg
                        )
                        causal_row_offset = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q
                        col_limit_right = row_idx + causal_row_offset
                        mask_r2p(s_chunk, col_limit_right, BLK_N)

                    def softmax_step(i_kv, apply_mask=False, is_first=False):
                        """One KV block of the online softmax — orig:L1131-1294."""
                        s_chunk = T.bind_(T.alloc_local([BLK_N], "float32"))
                        p_chunk = T.bind_(T.alloc_local([BLK_N // 2], "uint32"))
                        s_ready.wait(wg_id, score_epoch.phase)
                        with T.if_(warp_id == 0), T.then_():
                            T.cuda.iket.mark("softmax-phase-0")
                        softmax_max_token = iket_range("softmax-max", leader_only=True)
                        tile_max = T.bind_(T.alloc_local([1], "float32"))
                        for chunk_idx in range(BLK_N // SOFTMAX_LD_CHUNK):
                            tmem_load(
                                s_chunk,
                                chunk_idx * SOFTMAX_LD_CHUNK,
                                tmem(wg_id * MMA_N + chunk_idx * SOFTMAX_LD_CHUNK),
                                SOFTMAX_LD_CHUNK,
                            )
                        if apply_mask:
                            apply_causal_mask(s_chunk, m_block_idx, i_kv)
                        row_max_old = T.bind_(T.local_scalar("float32"))
                        if is_first:
                            reduce_max_128(tile_max, s_chunk)
                        else:
                            T.assign(row_max_old, row_max)
                            T.assign(tile_max[0], row_max_old)
                            reduce_max_128(tile_max, s_chunk, accum=True)
                        row_max_new = T.bind_(T.local_scalar("float32"))
                        acc_scale = T.bind_(T.local_scalar("float32"))
                        acc_scale_ = T.bind_(T.local_scalar("float32"))
                        row_max_safe = T.bind_(T.local_scalar("float32"))
                        T.assign(row_max_new, tile_max[0])
                        T.assign(
                            row_max_safe,
                            T.if_then_else(
                                tile_max[0] == T.float32(NEG_INF), T.float32(0.0), tile_max[0]
                            ),
                        )
                        if is_first:
                            T.assign(acc_scale, T.float32(1.0))
                        else:
                            T.assign(acc_scale_, (row_max_old - row_max_safe) * scale_log2)
                            with T.if_(acc_scale_ >= -rescale_threshold):
                                with T.then_():
                                    T.assign(row_max_new, row_max_old)
                                    T.assign(row_max_safe, row_max_old)
                                    T.assign(acc_scale, T.float32(1.0))
                                with T.else_():
                                    _register_instruction(
                                        "ex2.approx.ftz.f32", acc_scale, acc_scale_
                                    )
                        T.assign(row_max, row_max_new)
                        row_max_scaled = row_max_safe * scale_log2
                        with T.if_(warp_id == 0), T.then_():
                            T.cuda.iket.mark("softmax-phase-1")
                        T.cuda.iket.range_end(softmax_max_token[0])
                        if not is_first:
                            with T.if_(tid_in_wg < BLK_M), T.then_():
                                sScale_idx = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                                _shared_instruction(
                                    "st.shared.f32", sScale.ptr_to([sScale_idx]), acc_scale
                                )
                        if STATS_BAR_PAIRWISE:
                            T.ptx.bar.arrive(T.cast(1 + wg_id * 4 + warp_id, ty="uint32"), 64)
                        else:
                            T.ptx.bar.arrive(T.cast(1 + wg_id, ty="uint32"), 256)
                        softmax_fma_token = iket_range("softmax-fma", leader_only=True)
                        for i in range(BLK_N // 2):
                            fma_f32x2(s_chunk, 2 * i, T.float32(scale_log2), -row_max_scaled)
                        T.cuda.iket.range_end(softmax_fma_token[0])
                        softmax_exp2_token = iket_range("softmax-exp2", leader_only=True)
                        emu_pairs = EMU_PAIRS_CAUSAL if is_causal else EMU_PAIRS_NC
                        emu_start = EMU_START_CAUSAL if is_causal else EMU_START_NC
                        for frag_idx in range(4):
                            for i in range(BLK_N // 4 // 2):
                                idx = frag_idx * BLK_N // 4 + 2 * i
                                if (
                                    i * 2 % 16 < 16 - 2 * emu_pairs
                                    or frag_idx >= 4 - 1
                                    or frag_idx < emu_start
                                    or apply_mask
                                ):
                                    _register_instruction(
                                        "ex2.approx.ftz.f32", s_chunk[idx], s_chunk[idx]
                                    )
                                    _register_instruction(
                                        "ex2.approx.ftz.f32", s_chunk[idx + 1], s_chunk[idx + 1]
                                    )
                                else:
                                    ex2_emulation_2(s_chunk, idx, s_chunk[idx], s_chunk[idx + 1])
                            for i in range(BLK_N // 4 // 2):
                                idx = frag_idx * BLK_N // 4 + 2 * i
                                cast_f32x2_f16x2(p_chunk, s_chunk, idx)
                        T.cuda.iket.range_end(softmax_exp2_token[0])
                        softmax_tmem_st_token = iket_range("softmax-tmem-st", leader_only=True)
                        for i in range(P_SPLIT_Q):
                            tmem_store(
                                p_chunk,
                                i * BLK_N // 4 // 2,
                                tmem((wg_id * 2 * MMA_N + MMA_N + i * BLK_N // 4) // 2),
                            )
                        T.ptx.tcgen05.wait__st.sync.aligned()
                        p_o_rescale_remote.arrive(wg_id)
                        for i in range(4 - P_SPLIT_Q):
                            tmem_store(
                                p_chunk,
                                (P_SPLIT_Q + i) * BLK_N // 4 // 2,
                                tmem(
                                    (wg_id * 2 * MMA_N + MMA_N + (P_SPLIT_Q + i) * BLK_N // 4) // 2
                                ),
                            )
                        with T.if_(warp_id == 0), T.then_():
                            T.cuda.iket.mark("softmax-phase-2")
                        T.ptx.tcgen05.wait__st.sync.aligned()
                        p_ready_2_remote.arrive(wg_id)
                        with T.if_(warp_id == 0), T.then_():
                            T.cuda.iket.mark("softmax-phase-3")
                        T.cuda.iket.range_end(softmax_tmem_st_token[0])
                        softmax_corr.empty.wait(wg_id, softmax_epoch.phase)
                        with T.if_(warp_id == 0), T.then_():
                            T.cuda.iket.mark("softmax-phase-4")
                        softmax_sum_token = iket_range("softmax-sum", leader_only=True)
                        score_epoch.advance()
                        softmax_epoch.advance()
                        if is_first:
                            reduce_sum_128(row_sum, s_chunk)
                        else:
                            T.assign(row_sum[0], row_sum[0] * acc_scale)
                            reduce_sum_128(row_sum, s_chunk, accum=True)
                        with T.if_(warp_id == 0), T.then_():
                            T.cuda.iket.mark("softmax-phase-5")
                        T.cuda.iket.range_end(softmax_sum_token[0])

                    if T.constexpr(not EPI_ON_SOFTMAX):
                        softmax_corr.empty.wait(wg_id, softmax_epoch.phase)
                    softmax_epoch.advance()
                    n_block_max = n_block_max_of(m_block_idx)
                    n_block_min_causal = (
                        n_block_min_causal_of(m_block_idx)
                        if T.constexpr(is_causal)
                        else n_block_max
                    )
                    softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)
                    n_block_max_after_p1 = n_block_max - 1
                    num_phase2_blocks = T.max(n_block_max_after_p1 - n_block_min_causal, 0)
                    for i in T.serial(num_phase2_blocks, unroll=False):
                        softmax_step(n_block_max_after_p1 - 1 - i, apply_mask=True)
                    n_block_max_after_p2 = T.min(n_block_max_after_p1, n_block_min_causal)
                    for i in T.serial(n_block_max_after_p2, unroll=False):
                        softmax_step(n_block_max_after_p2 - 1 - i, apply_mask=False)
                    if T.constexpr(EPI_ON_SOFTMAX):
                        o_ready.wait(wg_id, o_epi_epoch.phase)
                        corr_epi.empty.wait(wg_id, o_epi_epoch.phase)
                        epi_ld_tmem_token = iket_range("epi-ld-tmem", leader_only=True)
                        acc_O_row_is_zero_or_nan = tvm_tirx.tirx.any(
                            row_sum[0] == T.float32(0.0), row_sum[0] != row_sum[0]
                        )
                        norm_scale_sm = T.local_scalar("float32")
                        _register_instruction(
                            "rcp.approx.ftz.f32",
                            norm_scale_sm,
                            T.Select(acc_O_row_is_zero_or_nan, T.float32(1.0), row_sum[0]),
                        )
                        o_row_f32_sm = T.alloc_local([EPI_LD_SM], "float32")
                        o_row_f16_sm = T.alloc_local([EPI_LD_SM // 2], "uint32")
                        for epi_q in T.unroll(2):
                            if wg_id == epi_q:
                                for d_tile in T.unroll(ceildiv(HEAD_DIM, EPI_LD_SM)):
                                    d_start = d_tile * EPI_LD_SM
                                    tmem_load(
                                        o_row_f32_sm,
                                        0,
                                        tmem((SMEM_PIPE_DEPTH_Q + epi_q) * MMA_N + d_start),
                                        EPI_LD_SM,
                                    )
                                    for i in T.unroll(EPI_LD_SM // 2):
                                        mul_f32x2(o_row_f32_sm, 2 * i, norm_scale_sm)
                                    for i in T.unroll(EPI_LD_SM // 2):
                                        cast_f32x2_f16x2(o_row_f16_sm, o_row_f32_sm, 2 * i)
                                    for i in T.unroll(EPI_LD_SM // 8):
                                        _shared_instruction(
                                            "st.shared.v4.u32",
                                            o_smem.ptr_to([epi_q, tid_in_wg, d_start + i * 8]),
                                            o_row_f16_sm[i * 4],
                                            o_row_f16_sm[i * 4 + 1],
                                            o_row_f16_sm[i * 4 + 2],
                                            o_row_f16_sm[i * 4 + 3],
                                        )
                        T.cuda.iket.range_end(epi_ld_tmem_token[0])
                        T.ptx.fence.proxy.async_.shared__cta()
                        corr_epi.full.arrive(wg_id)
                        p_o_rescale_remote.arrive(wg_id)
                        o_epi_epoch.advance()
                    else:
                        if tid_in_wg < BLK_M:
                            _shared_instruction(
                                "st.shared.f32",
                                sScale.ptr_to([ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M]),
                                row_sum[0],
                            )
                        if T.constexpr(STATS_BAR_PAIRWISE):
                            T.ptx.bar.arrive(T.cast(1 + wg_id * 4 + warp_id, ty="uint32"), 64)
                        else:
                            T.ptx.bar.arrive(T.cast(1 + wg_id, ty="uint32"), 256)
                elif (warp_cta >= 8) & (warp_cta <= 11):
                    T.ptx.setmaxnreg.dec.sync.aligned.u32(64)

                    def stats_sync(i_q):
                        if STATS_BAR_PAIRWISE:
                            T.ptx.bar.sync(T.cast(1 + i_q * 4 + warp_id, ty="uint32"), 64)
                        else:
                            T.ptx.bar.sync(T.uint32(1 + i_q), 256)

                    stats_sync(0)
                    softmax_corr.empty.arrive(0)
                    stats_sync(1)
                    softmax_epoch.advance()
                    corr_trip_count = (
                        n_block_max_of(m_block_idx) if T.constexpr(is_causal) else num_kv_blocks
                    )
                    for _i_kv in T.serial(corr_trip_count - 1, unroll=False):
                        for i_q in T.unroll(2):
                            stats_sync(i_q)
                            correction_token = iket_range("correction", leader_only=True)
                            acc_scale = T.local_scalar("float32")
                            should_rescale = T.local_scalar("int32")
                            if tid_in_wg < BLK_M:
                                _shared_instruction(
                                    "ld.shared.f32",
                                    acc_scale,
                                    sScale.ptr_to([ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]),
                                )
                                T.assign(should_rescale, T.Select(acc_scale < T.float32(1.0), 1, 0))
                            else:
                                T.assign(should_rescale, 0)
                            any_needs_rescale = T.local_scalar("uint32")
                            T.ptx.vote_sync.any.pred(
                                any_needs_rescale, T.ptx.pred(should_rescale), T.uint32(4294967295)
                            )
                            if any_needs_rescale != 0:
                                if tid_in_wg < BLK_M:
                                    o_row = T.alloc_local([RESCALE_TILE], "float32")
                                    for d_tile in T.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                        d_start = d_tile * RESCALE_TILE
                                        if d_start < HEAD_DIM:
                                            addr = tmem((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start)
                                            tmem_load(o_row, 0, addr, RESCALE_TILE)
                                            for i in T.unroll(RESCALE_TILE // 2):
                                                mul_f32x2(o_row, 2 * i, acc_scale)
                                            tmem_store(o_row, 0, addr)
                                    T.ptx.tcgen05.wait__st.sync.aligned()
                            p_o_rescale_remote.arrive(i_q)
                            softmax_corr.empty.arrive(1 - i_q)
                            T.cuda.iket.range_end(correction_token[0])
                        softmax_epoch.advance()
                    softmax_corr.empty.arrive(1)
                    if T.constexpr(not EPI_ON_SOFTMAX):
                        for i_q in T.unroll(2):
                            stats_sync(i_q)
                            row_sum_c = T.local_scalar("float32")
                            _shared_instruction(
                                "ld.shared.f32",
                                row_sum_c,
                                sScale.ptr_to([ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]),
                            )
                            softmax_corr.empty.arrive(i_q)
                            o_ready.wait(i_q, tmem_epoch.phase)
                            corr_epi.empty.wait(i_q, tmem_epoch.phase)
                            epi_ld_tmem_token = iket_range("epi-ld-tmem", leader_only=True)
                            zero_or_nan = tvm_tirx.tirx.any(
                                row_sum_c == T.float32(0.0), row_sum_c != row_sum_c
                            )
                            norm_scale = T.local_scalar("float32")
                            _register_instruction(
                                "rcp.approx.ftz.f32",
                                norm_scale,
                                T.Select(zero_or_nan, T.float32(1.0), row_sum_c),
                            )
                            o_row_f32 = T.alloc_local([TMEM_EPI_LD_SIZE], "float32")
                            o_row_f16 = T.alloc_local([TMEM_EPI_LD_SIZE // 2], "uint32")
                            for d_tile in T.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                                d_start = d_tile * TMEM_EPI_LD_SIZE
                                if d_start < HEAD_DIM:
                                    tmem_load(
                                        o_row_f32,
                                        0,
                                        tmem((SMEM_PIPE_DEPTH_Q + i_q) * MMA_N + d_start),
                                        TMEM_EPI_LD_SIZE,
                                    )
                                    for i in T.unroll(TMEM_EPI_LD_SIZE // 2):
                                        mul_f32x2(o_row_f32, 2 * i, norm_scale)
                                    for i in T.unroll(TMEM_EPI_LD_SIZE // 2):
                                        cast_f32x2_f16x2(o_row_f16, o_row_f32, 2 * i)
                                    for i in T.unroll(TMEM_EPI_LD_SIZE // 8):
                                        _shared_instruction(
                                            "st.shared.v4.u32",
                                            o_smem.ptr_to([i_q, tid_in_wg, d_start + i * 8]),
                                            o_row_f16[i * 4],
                                            o_row_f16[i * 4 + 1],
                                            o_row_f16[i * 4 + 2],
                                            o_row_f16[i * 4 + 3],
                                        )
                            T.cuda.iket.range_end(epi_ld_tmem_token[0])
                            T.ptx.fence.proxy.async_.shared__cta()
                            corr_epi.full.arrive(i_q)
                            p_o_rescale_remote.arrive(i_q)
                        tmem_epoch.advance()
                    softmax_epoch.advance()
                scheduler.next_tile()
            T.cuda.cta_sync()
            if tvm_tirx.tirx.all(wg_id == 0, warp_id == 0):
                dealloc = T.local_scalar("uint32")
                _shared_instruction("ld.shared.u32", dealloc, tmem_addr.ptr_to([0]))
                T.ptx[tmem_relinquish]()
                T.ptx[tmem_dealloc](dealloc, T.uint32(N_COLS_TMEM))

    return flash_attention4


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p(base + 63 & ~63)


def _encode(tensor, dims, strides, box):
    desc = _AlignedTensorMap()
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    tvm_ffi.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        "float16",
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *(1,) * rank,
        0,
        3,
        2,
        0,
    )
    return desc


def build_tensor_maps(
    Q,
    Kt,
    V,
    O,
    *,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    backend_config=None,
):
    """The seven maps, in the order ``make_kernel``'s parameters declare them."""
    gqa = num_qo_heads // num_kv_heads
    seq_q_per_tile = BLK_M // gqa
    use_2cta = resolve_backend_config(backend_config)["cuda"]["arch"] == "sm_110a" and (
        not is_causal
    )
    cta_group = 2 if use_2cta else 1

    def q_map(t):
        if gqa == 1:
            return _encode(
                t,
                (head_dim // 2, seq_len_q, batch_size * num_qo_heads * 2),
                (num_qo_heads * head_dim * F16_BYTES, head_dim),
                (head_dim // 2, seq_q_per_tile, 1)
                if use_2cta
                else (head_dim // 2, seq_q_per_tile, 2),
            )
        return _encode(
            t,
            (head_dim // 2, num_qo_heads, seq_len_q, batch_size * 2),
            (head_dim * F16_BYTES, num_qo_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, gqa, seq_q_per_tile, 1)
            if use_2cta
            else (head_dim // 2, gqa, seq_q_per_tile, 2),
        )

    def o_map(t):
        if gqa == 1:
            return _encode(
                t,
                (head_dim // 2, seq_len_q, batch_size * num_qo_heads * 2),
                (num_qo_heads * head_dim * F16_BYTES, head_dim),
                (head_dim // 2, seq_q_per_tile, 2),
            )
        return _encode(
            t,
            (head_dim // 2, num_qo_heads, seq_len_q, batch_size * 2),
            (head_dim * F16_BYTES, num_qo_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, gqa, seq_q_per_tile, 2),
        )

    def k_map(t):
        return _encode(
            t,
            (head_dim // 2, seq_len_kv, batch_size * num_kv_heads * 2),
            (num_kv_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, BLK_N // cta_group, 1) if use_2cta else (head_dim // 2, BLK_N, 2),
        )

    def v_map(t):
        return _encode(
            t,
            (head_dim // 2, seq_len_kv, batch_size * num_kv_heads * 2),
            (num_kv_heads * head_dim * F16_BYTES, head_dim),
            (head_dim // 2, BLK_N, 1) if use_2cta else (head_dim // 2, BLK_N, 2),
        )

    return (q_map(Q), q_map(Q), k_map(Kt), k_map(Kt), v_map(V), v_map(V), o_map(O))


def _select_reg_level(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal
):
    """Preserve the reference ptxas register-usage setting."""
    if is_causal and num_qo_heads == num_kv_heads:
        return "2"
    if is_causal:
        return "5" if seq_len_q <= 1024 else "6"
    return "10"


def get_flash_attention4_kernel(
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    *,
    backend_config=None,
):
    if resolve_backend_config(backend_config)["cuda"]["arch"] != "sm_100a":
        raise ValueError("The native FA4 variant is validated for sm_100a")
    if head_dim != 128:
        raise ValueError("The native FA4 variant requires head_dim=128")
    if num_qo_heads % num_kv_heads or num_qo_heads // num_kv_heads not in (1, 2, 4, 8):
        raise ValueError("The native FA4 variant requires GQA ratio 1, 2, 4 or 8")
    if seq_len_kv % BLK_N:
        raise ValueError("The KV sequence length must be a multiple of 128")
    backend_config = resolve_backend_config(
        backend_config,
        defaults={
            "cuda": {
                "ptxas": [
                    "-v",
                    "--warn-on-local-memory-usage",
                    "--register-usage-level="
                    + str(
                        int(
                            _select_reg_level(
                                batch_size,
                                seq_len_q,
                                seq_len_kv,
                                num_qo_heads,
                                num_kv_heads,
                                head_dim,
                                is_causal,
                            )
                        )
                    ),
                ]
            }
        },
    )
    prepare_arch = resolve_backend_config(backend_config)["cuda"]["arch"]
    deep_o = is_causal and seq_len_q <= 1024 and (prepare_arch != "sm_110a")
    return make_kernel(
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_causal=is_causal,
        tmem_pipe_depth=3 if deep_o else TMEM_PIPE_DEPTH,
        smem_pipe_depth_kv=2 if deep_o else SMEM_PIPE_DEPTH_KV,
        backend_config=backend_config,
    )


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    k = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    v = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    out = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    return (q, k, v, out)


KERNEL_META = {
    "name": "flash_attention4_tileop",
    "category": "ported.flashattention",
    "runtime_cuda_archs": ["sm_100a"],
}
CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": sl,
        "num_qo_heads": 32,
        "num_kv_heads": kv,
        "head_dim": 128,
        "is_causal": causal,
        "label": f"s{sl}_h32kv{kv}{('_causal' if causal else '')}",
    }
    for sl in [1024, 2048, 4096, 8192]
    for kv in [4, 8, 16, 32]
    for causal in [False, True]
]


def get_kernel(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    *,
    backend_config=None,
    **kwargs,
):
    return get_flash_attention4_kernel(
        batch_size,
        seq_len,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_causal=is_causal,
        backend_config=backend_config,
    )


def _build_launch(
    executable,
    q,
    k,
    v,
    out,
    *,
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    backend_config=None,
):
    tensor_maps = build_tensor_maps(
        q,
        k,
        v,
        out,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        backend_config=backend_config,
    )
    argv = tuple(desc.ptr for desc in tensor_maps)

    def launch():
        executable(*argv)

    launch._fa4_keep_alive = (q, k, v, out, tensor_maps, argv)
    return launch


def prepare_bench(*, backend_config=None, **kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {
        "config": dict(kwargs),
        "executable": compile_kernel(
            get_kernel(**kwargs, backend_config=backend_config), backend_config=backend_config
        ),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def run_test(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    *,
    backend_config=None,
    **kwargs,
):
    """Compile, run, and verify FlashAttention-4."""
    from tirx_kernels.runner import compile_kernel

    q, k, v, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    q_dev, k_dev, v_dev = (q.cuda(), k.cuda(), v.cuda())
    out = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    executable = compile_kernel(
        get_flash_attention4_kernel(
            batch_size,
            seq_len,
            seq_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            is_causal=is_causal,
            backend_config=backend_config,
        ),
        backend_config=backend_config,
    )
    launch = _build_launch(
        executable,
        q_dev,
        k_dev,
        v_dev,
        out,
        batch_size=batch_size,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        backend_config=backend_config,
    )
    launch()
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise AssertionError("output contains a non-finite value")
    _check_fp32_reference(q_dev, k_dev, v_dev, out, is_causal=is_causal)


def _check_fp32_reference(q, k, v, out, *, is_causal):
    """Check query chunks so the 8K reference fits alongside the kernel buffers."""
    seq_len, head_dim = q.shape[1], q.shape[-1]
    gqa = q.shape[2] // k.shape[2]
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            qh = q.float().permute(0, 2, 1, 3)
            kh = k.float().repeat_interleave(gqa, dim=2).permute(0, 2, 1, 3)
            vh = v.float().repeat_interleave(gqa, dim=2).permute(0, 2, 1, 3)
            columns = torch.arange(k.shape[1], device=q.device)[None, :]
            for start in range(0, seq_len, 128):
                stop = min(start + 128, seq_len)
                scores = (qh[:, :, start:stop] @ kh.transpose(-1, -2)) / math.sqrt(head_dim)
                if is_causal:
                    rows = torch.arange(start, stop, device=q.device)[:, None]
                    scores.masked_fill_(columns > rows + k.shape[1] - seq_len, -float("inf"))
                expected = (scores.softmax(-1) @ vh).permute(0, 2, 1, 3)
                torch.testing.assert_close(
                    out[:, start:stop].float(), expected, rtol=0.01, atol=0.01
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, backend_config=None, **kwargs):
    """Benchmark flash attention 4."""
    config = dict(prepared["config"])
    batch_size = config.pop("batch_size")
    seq_len = config.pop("seq_len")
    num_qo_heads = config.pop("num_qo_heads")
    num_kv_heads = config.pop("num_kv_heads")
    head_dim = config.pop("head_dim")
    is_causal = config.pop("is_causal")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    ex = executable
    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    Q_cuda = Q.cuda()
    K_cuda = K.cuda()
    V_cuda = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    launch = _build_launch(
        ex,
        Q_cuda,
        K_cuda,
        V_cuda,
        O_tir,
        batch_size=batch_size,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        backend_config=backend_config,
    )
    funcs = {"tir": launch}

    def _flashattn_sm100():
        from flash_attn.cute.interface import _flash_attn_fwd

        out_fa4 = torch.empty_like(O_tir)
        call_kwargs = {
            "q": Q_cuda,
            "k": K_cuda,
            "v": V_cuda,
            "out": out_fa4,
            "softmax_scale": 1.0 / math.sqrt(head_dim),
            "causal": is_causal,
        }

        def run():
            _flash_attn_fwd(**call_kwargs)

        run._fa4_keep_alive = (out_fa4, call_kwargs)
        return run

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": _flashattn_sm100},
        **kwargs,
    )


def run_bench(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    warmup=None,
    repeat=None,
    timer=None,
    *,
    backend_config=None,
    **kwargs,
):
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(
        batch_size=batch_size,
        seq_len=seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        **config,
        backend_config=backend_config,
    )
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)
