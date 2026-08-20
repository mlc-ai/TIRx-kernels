# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400)
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100a FP16 GDN prefill transcribed from FlashInfer's frozen CuTeDSL kernel.

The implementation follows the frozen CuTeDSL source order and preserves its
CUDA/PTX details instead of reorganizing them into a separate implementation.

Upstream source: flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py.
"""

# K.kernel traces concrete annotation objects; postponed annotations would turn them into strings.
import ctypes
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

import tirx_kernels.kern as K

D_HEAD = 128
CHUNK = 64
PAIR_TOKENS = 128  # tokens per CG0 pair-iteration
STAGE_BYTES = CHUNK * D_HEAD * 2  # one K/Q/V stage: 16 KB (f16)
TMEM_COLS = 512
DESCRIPTOR_SLOT_BYTES = 128
DESCRIPTOR_SLOTS = 4
DESCRIPTOR_BYTES_PER_CTA = DESCRIPTOR_SLOT_BYTES * DESCRIPTOR_SLOTS

# tmem column table — orig:L145-150. Hand constants, exactly as orig.
TM_STATE = 0  # recurrent state S (f32, 128 cols)
TM_QSTATE = 128  # QS, then QKV accumulator (f32, 64 cols)
TM_SINPUT = 192  # f16 state input, ts-A operand (64 cols)
TM_CG0 = 256  # KK/QK accumulators, 2 stages x 64 cols
TM_CG1 = 384  # KS / NV accumulator (f32, 64 cols)
TM_SHARED = 448  # NV (f16) at +0, decayed-V (f16) at +32

# named barrier ids / counts, exactly as orig hand-assigns them — orig:L152-155
BAR_TMEM, BAR_TMEM_N = 1, 320  # cg0 + cg1 + mma0 + mma1
BAR_INV, BAR_INV_N = 2, 128  # cg0
BAR_DEALLOC = 3  # cg1 warpgroup_sync
BAR_STATE, BAR_STATE_N = 4, 128  # cg1

# tcgen05 instruction descriptors — hardcoded u32 bit patterns (orig:L962/981/1017);
# no instruction exists to build one, the ISA defines the bit fields.
ID_KK = 0x04100010  # m64  n64  k16, f16*f16 -> f32
ID_TS = 0x08100010  # m128 n64  k16, f16*f16 -> f32
ID_KV = 0x08210010  # m128 n128 k16, f16*f16 -> f32, B col-major

MMA_SS = "tcgen05.mma.cta_group::1.kind::f16"
MMA_K8 = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
MMA_K16 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
LDM_X1 = "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
LDM_X1T = "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
LDM_X4 = "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
LDM_X4T = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
LDM_X4T_V = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
STM_X1 = "stmatrix.sync.aligned.m8n8.x1.shared.b16"
STM_X4 = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
STM_X4T_O = "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
TC_LD_256 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
TC_ST_256 = "tcgen05.st.sync.aligned.16x256b.x8.b32"
TC_ST_128 = "tcgen05.st.sync.aligned.16x128b.x8.b32"
TC_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST_32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_ST_16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TC_LD_CG0 = "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
LD_G_V4 = "ld.global.L1::no_allocate.v4.b32"
ST_G_V4 = "st.global.L1::no_allocate.v4.b32"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
TMAP_ACQ = "fence.proxy.tensormap::generic.acquire.gpu"
TMAP_REL = "fence.proxy.tensormap::generic.release.gpu"
CP_ASYNC = "cp.async.ca.shared.global"
CP_ASYNC_ARRIVE = "cp.async.mbarrier.arrive.noinc.shared::cta.b64"
BULK_WAIT = "cp.async.bulk.wait_group.read"
BULK_COMMIT = "cp.async.bulk.commit_group"


def tmem_row(n):
    """Row field of a tmem address — bits [22:16]. Plain arithmetic."""
    return n << 16


def make_kernel(HQ: int, HV: int):
    """Trace the kernel for one (HQ, HV) specialization. HQ/HV are baked."""
    SUBHEADS = HV // HQ  # value sub-heads per q head
    RANK = 3 if HQ == HV else 4  # V/O tensormap rank
    TMA_G2S_QK = (
        "cp.async.bulk.tensor.3d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    TMA_G2S_V = (
        f"cp.async.bulk.tensor.{RANK}d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    TMA_S2G_O = f"cp.async.bulk.tensor.{RANK}d.global.shared::cta.tile.bulk_group.L2::cache_hint"

    @K.kernel(
        warps=12,
        arch="sm_100a",
        min_blocks_per_sm=1,
        # orig:L1376 — the launch extent IS min(total_work, num_sms); the
        # kernel is persistent and grid-strides by exactly this.
        grid=lambda p: K.min(p["num_sequences"] * HV, p["num_sms"]),
    )
    def gdn_prefill(
        q: K.gptr[K.f16],
        k: K.gptr[K.f16],
        v: K.gptr[K.f16],
        gate: K.gptr[K.f32],
        beta: K.gptr[K.f32],
        o: K.gptr[K.f16],
        cu_seqlens: K.gptr[K.i32],
        initial_state: K.gptr[K.f32],
        final_state: K.gptr[K.f32],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        o_map: K.TensorMap,
        desc_ws: K.gptr[K.i8],  # per-CTA descriptor workspace (orig: int8)
        total_tokens: K.i64,
        num_sequences: K.i32,
        num_sms: K.i32,
        scale: K.f32,
    ):
        total_work = num_sequences * HV
        grid_x = K.min(total_work, num_sms)

        # ---------------- roles — orig:L1490+ ------------------------------
        # Explicit warp ids: the id->role map is greppable when the profiler
        # says "warp 10 is stuck". regs= is the ABSOLUTE setmaxnreg target; the
        # direction is inferred from the pinned entry allocation (65536/384 ->
        # 168 at launch). kern checks the exact partition of 0..11, contiguity,
        # regs 8-aligned in [24,256], budget <= 65536, and setmaxnreg
        # warpgroup-uniformity. orig satisfies every one, checked nowhere.

        sp = K.specialize()
        cg0 = sp.role("cg0", warps=[0, 1, 2, 3], regs=224)  # UT + inverse
        cg1 = sp.role("cg1", warps=[4, 5, 6, 7], regs=256)  # state keeper
        mma0 = sp.role("mma0", warps=[8], regs=24)  # KK/QK issuer
        tma = sp.role("tma", warps=[9], regs=24)  # Q/K/V loader
        mma1 = sp.role("mma1", warps=[10], regs=24)  # state-path issuer
        aux = sp.role("aux", warps=[11], regs=24)  # gate/beta + O store
        # ---------------- pipes — orig:L126-143 ----------------------------
        # IN-TREE lang.Pipeline / MBarrier, used raw. The ctor builds and inits
        # the full/empty pair (leader = CTA thread 0); the kind tags pick
        # TMABar/TCGen05Bar/MBarrier; init_full/init_empty are orig
        # PIPELINE_SPECS' last two columns, unchanged. Declaration order
        # reproduces orig's barrier byte layout (offsets 0..560).
        smem = K.smem_pool()
        p_k = K.Pipeline(smem, 4, full="tma", empty="tcgen05", init_empty=2)
        #   init_empty=2: mma0 AND mma1 each commit every K stage
        p_q = K.Pipeline(smem, 2, full="tma", empty="tcgen05", init_empty=2)
        p_v = K.Pipeline(smem, 3, full="tma", empty="mbar", init_empty=4)
        #   init_empty=4: lane 0 of each cg1 warp releases
        p_gate = K.Pipeline(smem, 5, full="mbar", empty="mbar", init_full=32, init_empty=256)
        #   32 aux lanes produce; 128 cg0 + 128 cg1 threads release
        p_beta = K.Pipeline(smem, 5, full="mbar", empty="mbar", init_full=32, init_empty=128)
        #   full side arrives via cp.async.mbarrier.arrive.noinc (32 lanes)
        p_qs = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_kv = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        #   state acc: TWO producers alternate in time (cg1 initial-state store
        #   vs mma1 state MMA), so init_full stays 1 — the skipping producer
        #   just advances its PipelineState past foreign rounds
        p_cg0 = K.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=128)
        p_cg1 = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)
        p_ainv = K.Pipeline(smem, 3, full="mbar", empty="tcgen05", init_full=128)
        p_qk = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=128)
        p_sinp = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=128)
        m_vks = K.MBarrier(smem, 1)  # bare mbarriers (one-way signals):
        m_nv = K.MBarrier(smem, 1)  # orig empty barriers 488/504/520
        m_dcy = K.MBarrier(smem, 1)  # are dead — no recycle side exists
        m_vks.init(128)
        m_nv.init(128)
        m_dcy.init(128)
        p_o = K.Pipeline(smem, 2, full="mbar", empty="mbar", init_full=128, init_empty=32)

        # ---------------- smem plan — orig:L113-123 ------------------------
        # ONE primitive. Stages are the leading dimension — an ordinary
        # coordinate passed to the operand constructors, no subviews.
        # swizzle=None is the IDENTITY: a plain row-major tirx buffer.
        # swizzle=K.SW128B is a COMPOSED layout, not a bare xor: the alloc's
        # last two dims are tiled into [8, 128B/elem] atoms and the xor applies
        # INSIDE each atom. All four constructors derive from that layout.
        s_tmem_addr = smem.alloc((1,), K.i32, align=4)  # tcgen05.alloc mailbox
        s_q = smem.alloc((2, CHUNK, D_HEAD), K.f16, swizzle=K.SW128B)
        s_k = smem.alloc((4, CHUNK, D_HEAD), K.f16, swizzle=K.SW128B)
        s_v = smem.alloc((3, D_HEAD, CHUNK), K.f16, swizzle=K.SW128B)
        s_ainv = smem.alloc((3, CHUNK, CHUNK), K.f16, swizzle=K.SW128B)
        s_qk = smem.alloc((2, CHUNK, CHUNK), K.f16, swizzle=K.SW128B)
        s_o = smem.alloc((2, D_HEAD, CHUNK), K.f16, swizzle=K.SW128B)
        s_cumsumlog = smem.alloc((5, CHUNK), K.f32, align=4)  # plain tirx buffers
        s_cumprod = smem.alloc((5, CHUNK), K.f32, align=4)  # (orig:L121-123)
        s_beta = smem.alloc((5, CHUNK), K.f32, align=4)

        # publish the mbarrier.init stores before any use — orig:L1477-1481
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        # named barriers — hardcoded id/count, exactly as orig hand-assigns
        def bar_tmem():  # orig id=1, count 320
            K.ptx.bar.sync(K.uint32(BAR_TMEM), K.uint32(BAR_TMEM_N))

        def bar_inv():  # orig id=2, count 128
            K.ptx.bar.sync(K.uint32(BAR_INV), K.uint32(BAR_INV_N))

        def bar_state():  # orig id=4, count 128
            K.ptx.bar.sync(K.uint32(BAR_STATE), K.uint32(BAR_STATE_N))

        # ---------------- mutable TMA descriptors — orig:L1469-1474 --------
        # A mutable descriptor is a raw pointer to a CTA-private 128B slot in
        # the global workspace; every tensormap op below is bare PTX on it.
        desc_base = K.Cast("int64", K.cta_id()) * K.int64(512)
        d_q = desc_ws.ptr_to([desc_base])
        d_k = desc_ws.ptr_to([desc_base + K.int64(128)])
        d_v = desc_ws.ptr_to([desc_base + K.int64(256)])
        d_o = desc_ws.ptr_to([desc_base + K.int64(384)])

        # ---------------- shared user closures -----------------------------
        # (the tcgen05 disable-output-lane masks are K.idioms.mma_chain's
        # default — four zero u32s, i.e. no lane disabled)

        def elect():
            return K.cuda.elect_sync()

        def elect_local():
            """One elected lane, materialized into a local.

            K.idioms.mma_chain takes `pred` rather than electing internally (G3:
            a warp-collective emitted inside the helper would land wherever the
            helper is called, including inside a guard). One of these per chain
            reproduces the frozen kernel's leader for every phase of that chain.
            """
            e = K.alloc_local([1], "uint32")
            K.assign(e[0], K.cuda.elect_sync())
            return e[0]

        def mma_desc(view, *, transpose=False):
            addr = K.cuda.cvta_generic_to_shared(view.ptr_to(0, 0))
            desc_lo = K.Cast("uint64", (addr >> 4) & K.uint32(0x3FFF))
            if transpose:
                return K.uint64(0x4000404002000000) | (desc_lo + K.uint64(0x02000000))
            return K.uint64(0x4000404000010000) | desc_lo

        def mma0_chain(d, a, b, pred):
            """The frozen MMA0 chain: one descriptor pair, eight runtime phases."""
            a_desc = mma_desc(a)
            b_desc = a_desc if a is b else mma_desc(b)
            with K.serial(8) as kp:
                phase_off = K.Cast("uint64", (kp & 3) * 2 + (kp >> 2) * 512)
                K.ptx[MMA_SS](
                    K.Cast("uint32", d),
                    a_desc + phase_off,
                    b_desc + phase_off,
                    K.uint32(ID_KK),
                    K.uint32(0),
                    K.uint32(0),
                    K.uint32(0),
                    K.uint32(0),
                    K.ptx.pred(K.Cast("uint32", kp != 0)),
                    pred=pred,
                )

        def mma1_chain(d, a, b, *, phases, transpose_b, accumulate, pred):
            """The frozen MMA1 chain, kept rolled in the emitted CUDA."""
            b_desc = mma_desc(b, transpose=transpose_b)
            with K.serial(phases) as kp:
                phase_off = (
                    K.Cast("uint64", kp * 128)
                    if transpose_b
                    else K.Cast("uint64", (kp & 3) * 2 + (kp >> 2) * 512)
                )
                enable_input_d = K.uint32(1) if accumulate else K.Cast("uint32", kp != 0)
                K.ptx[MMA_SS](
                    K.Cast("uint32", d),
                    K.Cast("uint32", a + kp * 8),
                    b_desc + phase_off,
                    K.uint32(ID_KV if transpose_b else ID_TS),
                    K.uint32(0),
                    K.uint32(0),
                    K.uint32(0),
                    K.uint32(0),
                    K.ptx.pred(enable_input_d),
                    pred=pred,
                )

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def work_coords(work):  # orig:L1509-1524
            work_u32 = K.Cast("uint32", work)
            remain = udiv_const(work_u32, HV)
            head = K.Cast("int32", work_u32 - remain * K.uint32(HV))
            # The valid-work guard proves remain < num_sequences, so the frozen
            # fast-divmod remainder has its quotient known zero.
            batch = K.Cast("int32", remain)
            bounds = K.alloc_local([2], "int32")
            # Keep these scalar: an odd batch index is only four-byte aligned.
            K.ptx.ld.global_.s32(bounds[0], cu_seqlens.ptr_to([batch]))
            K.ptx.ld.global_.s32(bounds[1], cu_seqlens.ptr_to([batch + 1]))
            lo = K.Cast("int64", bounds[0])
            hi = K.Cast("int64", bounds[1])
            return batch, head, lo, hi, K.Cast("int32", hi - lo)

        if HQ == HV:

            def head_split(head):  # orig:L2436-2440
                return head, K.int32(0)

        else:

            def head_split(head):
                qk_head = K.Cast("int32", udiv_const(K.Cast("uint32", head), SUBHEADS))
                return qk_head, head - qk_head * SUBHEADS

        # Register-level spellings this kernel uses. These are NOT library
        # material: each is one instruction, or one instruction shape used at a
        # single site, so the knowledge lives in the comment beside it rather
        # than in an API. (Structure earns an API; spelling does not — the
        # multi-instruction shapes that DID earn one are K.idioms.mma_chain,
        # warp_scan_add and cast_f16x2_to_f32x2.)
        #
        # Provenance of the one that moved out: the f16x2 widening is
        # orig:L499-508. orig spells it as pure expressions (mask/shift ->
        # reinterpret -> widen); K.idioms.cast_f16x2_to_f32x2 spells the
        # instructions (mov.b32 {lo,hi} + 2x cvt.f32.f16) and compiles to
        # byte-identical SASS -- see idioms_NOTES.md.

        def f2(a, b):
            """The packed f32x2 operand {a, b} — one 64-bit value."""
            return K.cuda.make_float2(a, b)

        def pack_f16x2(dst, a, b):
            # PTX's two-source packed conversion puts its SECOND operand in the
            # low half (CUDA's float2->half2 helper emits the same b,a order).
            K.ptx.cvt.rn.f16x2.f32(dst, b, a)

        def scale_pair(vals, i, *factors):
            """vals[2i:2i+2] *= factors, as a chain of packed f32x2 multiplies.

            mul.rn.f32x2 takes ONE 64-bit destination and two 64-bit sources —
            not a two-element float window, which is the natural-looking and
            wrong spelling (gdn_port_NOTES.md §3.2). The packed value stays in
            one uint64 across the whole chain and is unpacked once at the end
            (orig:L1615-1628): unpacking between multiplies hands the CUDA
            bridge two float2 rebuilds per step that it does not always fold
            away. Each factor is itself 64-bit — f2(...) or a uint64 loaded
            straight out of smem (orig:L1691).
            """
            t = K.alloc_local([1], "uint64")
            K.ptx.mul.rn.f32x2(t[0], f2(vals[2 * i], vals[2 * i + 1]), factors[0])
            for factor2 in factors[1:]:
                K.ptx.mul.rn.f32x2(t[0], t[0], factor2)
            K.assign(vals[2 * i], K.cuda.float2_x(t[0]))
            K.assign(vals[2 * i + 1], K.cuda.float2_y(t[0]))

        def neg_pack(dst_word, a, b):  # orig:L486-497
            # One sub.rn.f32x2 (from +0) + one cvt.rn.f16x2.f32. DPS keeps the
            # CUDA bridge from duplicating the expression (orig's own comment):
            # the value-returning form emits two standalone PTX subs.
            neg = K.alloc_local([1], "uint64")
            K.ptx.sub.rn.f32x2(neg[0], f2(K.float32(0.0), K.float32(0.0)), f2(a, b))
            pack_f16x2(dst_word, K.cuda.float2_x(neg[0]), K.cuda.float2_y(neg[0]))

        def udiv_const(value, divisor):
            """value // divisor for a NONNEGATIVE value and a constant divisor.

            orig:_udiv_u32_const hand-rolled the magic multiply
            ((n * 0xAAAAAAAB) >> 33 for /3, >> 37 for /48). Measured: that is
            pessimal. ptxas already emits the magic multiply for the spelled
            division — IMAD.HI.U32 + SHF.R.U32.HI, two instructions, where the
            hand-written 64-bit form takes five. What the frozen kernel was
            right about is the CAST: signed n/3 costs 4 instructions and n%3
            costs 6, the extras being sign correction. See idioms_NOTES.md §2.
            """
            return K.Cast("uint32", value) // K.uint32(divisor)

        def replace_desc(dsc, addr, dim1, dim2, dim3, s0, s1, s2):
            # Frozen TensorMapManager field order — orig:L380-393.
            R = "tensormap_replace.tile"
            # .b64 wants a 64-bit *value*: the tensor base address, not a
            # pointer expression.
            K.ptx[f"{R}.global_address.global.b1024.b64"](dsc, K.reinterpret("uint64", addr))
            K.ptx[f"{R}.global_dim.global.b1024.b32"](dsc, 0, K.uint32(128))
            K.ptx[f"{R}.global_dim.global.b1024.b32"](dsc, 1, K.Cast("uint32", dim1))
            K.ptx[f"{R}.global_stride.global.b1024.b64"](dsc, 0, K.Cast("uint64", s0))
            K.ptx[f"{R}.global_dim.global.b1024.b32"](dsc, 2, K.Cast("uint32", dim2))
            K.ptx[f"{R}.global_stride.global.b1024.b64"](dsc, 1, K.Cast("uint64", s1))
            K.ptx[f"{R}.global_dim.global.b1024.b32"](dsc, 3, K.Cast("uint32", dim3))
            K.ptx[f"{R}.global_stride.global.b1024.b64"](dsc, 2, K.Cast("uint64", s2))
            K.ptx[f"{R}.global_dim.global.b1024.b32"](dsc, 4, K.uint32(1))
            K.ptx[f"{R}.global_stride.global.b1024.b64"](dsc, 3, K.uint64(0))

        def replace_vo_desc(dsc, buf, hi):  # V/O share geometry
            if HQ == HV:
                replace_desc(dsc, buf, hi, HV, 1, 2 * D_HEAD * HV, 2 * D_HEAD, 0)
            else:
                replace_desc(
                    dsc, buf, hi, SUBHEADS, HQ, 2 * D_HEAD * HV, 2 * D_HEAD, 2 * D_HEAD * SUBHEADS
                )

        def copy_desc(dsc, m):  # 128B = 2x v4.u64 round trips — orig:L338-347
            tmp = K.alloc_local([4], "uint64")
            src = K.reinterpret("uint64", K.address_of(m))
            dst = K.reinterpret("uint64", dsc)
            for half in range(2):
                off = K.uint64(half * 32)
                K.ptx.ld.global_.v4.b64(
                    tmp[0], tmp[1], tmp[2], tmp[3], K.reinterpret("handle", src + off)
                )
                K.ptx.st.global_.v4.b64(
                    K.reinterpret("handle", dst + off), tmp[0], tmp[1], tmp[2], tmp[3]
                )

        def tmem_preamble():  # mailbox reload — orig:L1494-1496
            bar_tmem()
            tm = K.alloc_local([1], "int32")
            K.ptx.ld.volatile.shared.s32(tm[0], K.address_of(s_tmem_addr[0]))
            return tm

        def tmem_at(tm, col, row_bits=0):
            return K.Cast("uint32", tm[0] + col + row_bits)

        def drain(pipe, st):  # producer tail: acquire all stages before exit
            for _ in range(pipe.stages):
                pipe.empty.wait(st.stage, st.phase)
                st.advance()

        # ==================================================================
        # CG0: gate/beta -> decay transfer, KK -> (I - tril(beta*T*KK))^-1,
        #      QK epilogue.                              orig:L1491-1787
        # ==================================================================
        with cg0:
            # ring states: (depth, initial phase) — consumer full-wait starts
            # 0, producer empty-wait starts 1.
            st_gate = K.PipelineState(5, phase=0)  # consumer, p_gate
            st_beta = K.PipelineState(5, phase=0)  # consumer, p_beta
            st_acc = K.PipelineState(2, phase=0)  # consumer, p_cg0
            st_ainv = K.PipelineState(3, phase=1)  # producer, p_ainv
            st_qk = K.PipelineState(2, phase=1)  # producer, p_qk
            tmem = tmem_preamble()

            tid0 = K.tid_in_role()
            lane = K.lane_id()
            # Per-thread coordinate algebra of the CG0 fragment: USER code,
            # logical coords only — the tile applies atom tiling + the
            # intra-atom xor.                            orig:L1530-1533
            row0 = K.local_scalar("int32")
            K.assign(row0, ((tid0 >> 1) & 48) | (tid0 & 15))
            cb0 = K.local_scalar("int32")
            K.assign(cb0, tid0 & 16)
            # tmem row field: (thread << 16) & 0x600000 == warp_in_role * 32
            rowbits0 = K.local_scalar("int32")
            K.assign(rowbits0, (tid0 << 16) & 0x600000)

            def cg0_acc_ld(frag, stage):  # orig:L541-552
                # Ld16x32bx2/Repetition16 is two native x16 loads; `16` is
                # immHalfSplitoff (ISA 9.7.17.8.3, "taddr+immHalfSplitoff").
                addr = tmem_at(tmem, TM_CG0 + stage * 64, rowbits0)
                K.ptx[TC_LD_CG0](*(frag[i] for i in range(16)), addr, 16)
                K.ptx[TC_LD_CG0](*(frag[16 + i] for i in range(16)), addr + K.uint32(32), 16)

            def cg0_store_frag(view, vals):  # orig:L555-568
                words = K.alloc_local([16], "uint32")
                for p in range(16):
                    pack_f16x2(words[p], vals[2 * p], vals[2 * p + 1])
                for j, dc in enumerate((0, 8, 32, 40)):
                    K.ptx["st.shared.v4.b32"](
                        view.ptr_to(row0, cb0 + dc),
                        words[4 * j],
                        words[4 * j + 1],
                        words[4 * j + 2],
                        words[4 * j + 3],
                    )

            def cg0_load_frag(view, vals):  # orig:L571-591
                words = K.alloc_local([16], "uint32")
                for j, dc in enumerate((0, 8, 32, 40)):
                    K.ptx["ld.shared.v4.b32"](
                        words[4 * j],
                        words[4 * j + 1],
                        words[4 * j + 2],
                        words[4 * j + 3],
                        view.ptr_to(row0, cb0 + dc),
                    )
                for p in range(16):
                    K.idioms.cast_f16x2_to_f32x2(vals, p, words[p])

            # --- hierarchical 64x64 unit-lower-triangular inverse ---------
            # Register/warp choreography is irreducible; kern only removes the
            # hand-xor'd smem addresses (logical coords go through the
            # swizzled tile).                            orig:L595-759
            def invert_diag_8x8(av, block8):  # orig:L595-624
                r = block8 + (lane & 7)
                words = K.alloc_local([4], "uint32")
                K.ptx["ld.shared.v4.b32"](
                    words[0], words[1], words[2], words[3], av.ptr_to(r, block8)
                )
                row = [K.local_scalar("float32") for _ in range(8)]
                for p in range(4):
                    K.idioms.cast_f16x2_to_f32x2(row, p, words[p])
                for i in range(8):
                    with K.If((lane & 7) == i), K.Then():
                        K.assign(row[i], K.float32(1.0))
                rs = K.alloc_local([1], "float32")
                pv = K.alloc_local([1], "float32")
                for src in range(7):
                    # CuTe emits exact neg.f32 here; neg.ftz.f32 cannot be
                    # folded into its select/conversion consumers by ptxas.
                    K.ptx.neg.f32(rs[0], row[src])
                    for i in range(7):
                        if i < src:  # Python-const compare
                            # The shuffle MUST be materialized into a local
                            # here, before the guard. A traced body has no
                            # statement to pin a value-returning intrinsic to,
                            # so binding it to a Python name emits it at its
                            # USE site -- inside `(lane & 7) > src`, where the
                            # excluded lanes never reach the warp-collective
                            # __shfl_sync and the whole CTA deadlocks.
                            K.assign(pv[0], K.cuda._shfl_sync(K.uint32(0xFFFFFFFF), row[i], src, 8))
                            with K.If((lane & 7) > src), K.Then():
                                K.assign(row[i], row[i] + rs[0] * pv[0])
                    with K.If((lane & 7) > src), K.Then():
                        K.assign(row[src], rs[0])
                for p in range(4):
                    pack_f16x2(words[p], row[2 * p], row[2 * p + 1])
                K.ptx["st.shared.v4.b32"](
                    av.ptr_to(r, block8), words[0], words[1], words[2], words[3]
                )

            def ldm_x4(insn, dst, av, base_row, base_col):
                # orig:L658-673 — the .trans modifier changes each matrix's
                # load semantics, not the lane-group-to-matrix address map.
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                K.ptx[insn](dst[0], dst[1], dst[2], dst[3], av.ptr_to(row, col))

            def stm_x4(src, av, base_row, base_col):  # orig:L676-684
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                K.ptx[STM_X4](av.ptr_to(row, col), src[0], src[1], src[2], src[3])

            def mma_k8_zero(acc, a, b):
                K.ptx[MMA_K8](
                    acc[0],
                    acc[1],
                    acc[2],
                    acc[3],
                    a[0],
                    a[1],
                    b[0],
                    K.float32(0.0),
                    K.float32(0.0),
                    K.float32(0.0),
                    K.float32(0.0),
                )

            def mma_k16(acc, a, b, acc_off, b_off, accumulate):
                c = [acc[acc_off + i] for i in range(4)] if accumulate else [K.float32(0.0)] * 4
                K.ptx[MMA_K16](
                    *(acc[acc_off + i] for i in range(4)),
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    b[b_off],
                    b[b_off + 1],
                    *c,
                )

            def inverse_8_to_16(av, b16):  # orig:L627-655
                a = K.alloc_local([2], "uint32")
                b = K.alloc_local([1], "uint32")
                acc = K.alloc_local([4], "float32")
                dm = K.alloc_local([1], "uint32")
                cm = K.alloc_local([1], "uint32")
                K.ptx[LDM_X1](dm[0], av.ptr_to(b16 + 8 + (lane & 7), b16 + 8))
                K.ptx[LDM_X1T](cm[0], av.ptr_to(b16 + 8 + (lane & 7), b16))
                K.assign(a[0], dm[0])
                K.assign(a[1], dm[0])
                K.assign(b[0], cm[0])
                mma_k8_zero(acc, a, b)
                neg_pack(a[0], acc[0], acc[1])
                neg_pack(a[1], acc[2], acc[3])
                K.ptx[LDM_X1T](b[0], av.ptr_to(b16 + (lane & 7), b16))
                mma_k8_zero(acc, a, b)
                pack_f16x2(dm[0], acc[0], acc[1])
                K.ptx[STM_X1](av.ptr_to(b16 + 8 + (lane & 7), b16), dm[0])

            def inverse_16_to_32(av, b32):  # orig:L687-705
                a = K.alloc_local([4], "uint32")
                b = K.alloc_local([4], "uint32")
                acc = K.alloc_local([8], "float32")
                out = K.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a, av, b32 + 16, b32 + 16)
                ldm_x4(LDM_X4T, b, av, b32 + 16, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    neg_pack(a[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b, av, b32, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    pack_f16x2(out[p], acc[2 * p], acc[2 * p + 1])
                stm_x4(out, av, b32 + 16, b32)

            def inverse_32_to_64(av, half_warp):  # orig:L708-759
                rb = 32 + half_warp * 16
                a0 = K.alloc_local([4], "uint32")
                a1 = K.alloc_local([4], "uint32")
                b00 = K.alloc_local([4], "uint32")
                b01 = K.alloc_local([4], "uint32")
                b10 = K.alloc_local([4], "uint32")
                b11 = K.alloc_local([4], "uint32")
                acc = K.alloc_local([16], "float32")
                pa = K.alloc_local([8], "uint32")
                o0 = K.alloc_local([4], "uint32")
                o1 = K.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a0, av, rb, 32)
                ldm_x4(LDM_X4, a1, av, rb, 48)
                ldm_x4(LDM_X4T, b00, av, 32, 0)
                ldm_x4(LDM_X4T, b01, av, 32, 16)
                ldm_x4(LDM_X4T, b10, av, 48, 0)
                ldm_x4(LDM_X4T, b11, av, 48, 16)
                mma_k16(acc, a0, b00, 0, 0, False)
                mma_k16(acc, a0, b00, 4, 2, False)
                mma_k16(acc, a0, b01, 8, 0, False)
                mma_k16(acc, a0, b01, 12, 2, False)
                mma_k16(acc, a1, b10, 0, 0, True)
                mma_k16(acc, a1, b10, 4, 2, True)
                mma_k16(acc, a1, b11, 8, 0, True)
                mma_k16(acc, a1, b11, 12, 2, True)
                for p in range(8):
                    neg_pack(pa[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b00, av, 0, 0)
                ldm_x4(LDM_X4T, b01, av, 0, 16)
                ldm_x4(LDM_X4T, b10, av, 16, 0)
                ldm_x4(LDM_X4T, b11, av, 16, 16)
                for i in range(4):
                    K.assign(a0[i], pa[i])
                    K.assign(a1[i], pa[4 + i])
                mma_k16(acc, a0, b00, 0, 0, False)
                mma_k16(acc, a0, b00, 4, 2, False)
                mma_k16(acc, a0, b01, 8, 0, False)
                mma_k16(acc, a0, b01, 12, 2, False)
                mma_k16(acc, a1, b10, 0, 0, True)
                mma_k16(acc, a1, b10, 4, 2, True)
                mma_k16(acc, a1, b11, 8, 0, True)
                mma_k16(acc, a1, b11, 12, 2, True)
                for p in range(4):
                    pack_f16x2(o0[p], acc[2 * p], acc[2 * p + 1])
                    pack_f16x2(o1[p], acc[8 + 2 * p], acc[8 + 2 * p + 1])
                bar_inv()
                stm_x4(o0, av, rb, 0)
                stm_x4(o1, av, rb, 16)

            lw = K.warp_id_in_role()  # 0..3
            work = K.alloc_local([1], "int32")
            K.assign(work[0], K.cta_id())
            with K.While(work[0] < total_work):
                batch, head, lo, hi, seqlen = work_coords(work[0])
                with K.serial(K.ceildiv(seqlen, PAIR_TOKENS)) as _pair:
                    # decay transfer for both chunks: hold TWO gate stages —
                    # copy each stage, then advance.          orig:L1537-1580
                    g2 = K.alloc_local([2], "int32")
                    for j in range(2):
                        K.assign(g2[j], st_gate.stage)
                        p_gate.full.wait(g2[j], st_gate.phase)
                        st_gate.advance()
                    transfer = [K.alloc_local([32], "float32") for _ in range(2)]
                    rlog = K.alloc_local([2], "float32")
                    for j in range(2):
                        K.ptx.ld.shared.f32(rlog[j], K.address_of(s_cumsumlog[g2[j], row0]))
                    clog = K.alloc_local([2], "float32")
                    for i in range(32):
                        col = cb0 + i + (16 if i >= 16 else 0)  # orig:L1558
                        for j in range(2):
                            K.assign(transfer[j][i], K.float32(0.0))
                        # The frozen source guards the LDS + MUFU behind the
                        # Select's short-circuit; ex2's DPS shape would
                        # otherwise hoist both out unconditionally, so keep
                        # them under the same guard.        orig:L1560-1576
                        with K.If(row0 >= col), K.Then():
                            for j in range(2):
                                K.ptx.ld.shared.f32(clog[j], K.address_of(s_cumsumlog[g2[j], col]))
                            for j in range(2):
                                K.ptx.ex2.approx.ftz.f32(transfer[j][i], rlog[j] - clog[j])
                    for j in range(2):
                        p_gate.empty.arrive(g2[j])  # release

                    b2s = K.alloc_local([2], "int32")  # orig:L1582-1595
                    for j in range(2):
                        K.assign(b2s[j], st_beta.stage)
                        p_beta.full.wait(b2s[j], st_beta.phase)
                        st_beta.advance()
                    brow = K.alloc_local([2], "float32")
                    for j in range(2):
                        K.ptx.ld.shared.f32(brow[j], K.address_of(s_beta[b2s[j], row0]))

                    # KK_j -> beta*T*KK -> s_ainv stage_j     orig:L1597-1661
                    a2 = K.alloc_local([2], "int32")  # held p_ainv stages
                    kk = K.alloc_local([32], "float32")
                    for j in range(2):
                        K.assign(a2[j], st_ainv.stage)
                        p_ainv.empty.wait(a2[j], st_ainv.phase)  # acquire
                        st_ainv.advance()
                        p_cg0.full.wait(st_acc.stage, st_acc.phase)
                        cg0_acc_ld(kk, st_acc.stage)
                        K.ptx[WAIT_LD]()
                        p_cg0.empty.arrive(st_acc.stage)  # release
                        st_acc.advance()
                        for p in range(16):  # orig:L1613-1628
                            scale_pair(
                                kk,
                                p,
                                f2(transfer[j][2 * p], transfer[j][2 * p + 1]),
                                f2(brow[j], brow[j]),
                            )
                        cg0_store_frag(s_ainv[a2[j]], kk)

                    # hierarchical inverse on both stages      orig:L1663-1680
                    my = K.alloc_local([1], "int32")
                    K.assign(my[0], a2[0])
                    with K.If((lw >> 1) == 1), K.Then():
                        K.assign(my[0], a2[1])
                    bar_inv()
                    # a2[j] / my[0] are int LOCALS, not PipelineState vars: a
                    # view captures the stage EXPRESSION, so a view of
                    # st_ainv.stage would rebind under the advance() above.
                    # Copying the stage out first is what makes these views
                    # safe to build here.
                    invert_diag_8x8(s_ainv[my[0]], (((lw & 1) * 32 + lane) >> 3) * 8)
                    bar_inv()
                    inverse_8_to_16(s_ainv[a2[0]], lw * 16)
                    inverse_8_to_16(s_ainv[a2[1]], lw * 16)
                    bar_inv()
                    inverse_16_to_32(s_ainv[my[0]], (lw & 1) * 32)
                    bar_inv()
                    inverse_32_to_64(s_ainv[my[0]], lw & 1)
                    bar_inv()

                    # publish Ainv_j * beta_col, release beta  orig:L1682-1737
                    inv = K.alloc_local([32], "float32")
                    bcol = K.alloc_local([1], "uint64")
                    for j in range(2):
                        cg0_load_frag(s_ainv[a2[j]], inv)
                        for p in range(16):
                            c = cb0 + p * 2 + (16 if p >= 8 else 0)  # orig:L1687
                            K.ptx.ld.shared.u64(bcol[0], K.address_of(s_beta[b2s[j], c]))
                            scale_pair(inv, p, bcol[0])
                        cg0_store_frag(s_ainv[a2[j]], inv)
                        K.ptx[FENCE_ASYNC]()
                        p_ainv.full.arrive(a2[j])  # commit
                        p_beta.empty.arrive(b2s[j])  # release beta

                    # QK epilogue: acc * transfer_j * scale    orig:L1740-1782
                    for j in range(2):
                        p_qk.empty.wait(st_qk.stage, st_qk.phase)  # acquire
                        p_cg0.full.wait(st_acc.stage, st_acc.phase)
                        cg0_acc_ld(kk, st_acc.stage)
                        for p in range(16):
                            scale_pair(
                                kk,
                                p,
                                f2(transfer[j][2 * p], transfer[j][2 * p + 1]),
                                f2(scale, scale),
                            )
                        cg0_store_frag(s_qk[st_qk.stage], kk)
                        K.ptx[FENCE_ASYNC]()
                        K.ptx[WAIT_LD]()
                        p_cg0.empty.arrive(st_acc.stage)  # release acc
                        st_acc.advance()
                        p_qk.full.arrive(st_qk.stage)  # commit
                        st_qk.advance()
                K.assign(work[0], work[0] + grid_x)
            drain(p_ainv, st_ainv)  # orig:L1784-1787
            drain(p_qk, st_qk)

        # ==================================================================
        # CG1: recurrent-state keeper + O epilogue.       orig:L1789-2143
        # ==================================================================
        with cg1:
            st_v = K.PipelineState(3, phase=0)  # consumer, p_v
            st_g1 = K.PipelineState(5, phase=0)  # consumer, p_gate
            st_cg1c = K.PipelineState(1, phase=0)  # consumer, p_cg1
            st_kvc = K.PipelineState(1, phase=0)  # consumer, p_kv
            st_qsc = K.PipelineState(1, phase=0)  # consumer, p_qs
            st_kvp = K.PipelineState(1, phase=1)  # producer, p_kv (ring shared
            #   with mma1's producer state — see the skip advances)
            st_sinp = K.PipelineState(1, phase=1)  # producer, p_sinp
            st_op = K.PipelineState(2, phase=1)  # producer, p_o

            # tmem alloc by owner warp 0, result via mailbox — orig:L1792-1795
            with K.If(K.warp_id_in_role() == 0), K.Then():
                K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                    K.address_of(s_tmem_addr[0]), K.uint32(TMEM_COLS)
                )
            tmem = tmem_preamble()

            tid1 = K.tid_in_role()
            lane1 = K.lane_id()
            rowbits1 = (tid1 << 16) & 0x600000

            def cg1_acc_ld(fr, col):  # orig:L793-803
                addr = tmem_at(tmem, col, rowbits1)
                K.ptx[TC_LD_256](*(fr[i] for i in range(32)), addr)
                K.ptx[TC_LD_256](*(fr[32 + i] for i in range(32)), addr + K.uint32(tmem_row(16)))

            def cg1_acc_st(fr, col):  # orig:L805-815
                addr = tmem_at(tmem, col, rowbits1)
                K.ptx[TC_ST_256](addr, *(fr[i] for i in range(32)))
                K.ptx[TC_ST_256](addr + K.uint32(tmem_row(16)), *(fr[32 + i] for i in range(32)))

            def cg1_f16_st_half(col, words, half):  # orig:L817-824
                addr = tmem_at(tmem, col, rowbits1) + K.uint32(tmem_row(16 * half))
                K.ptx[TC_ST_128](addr, *(words[half * 16 + i] for i in range(16)))

            def vo_octets(view):
                """The eight x4-trans octets of a V/O fragment — orig:L833-885.

                orig hand-computes a physical byte offset with the 128B swizzle
                xor folded in, plus a sign-dependent +-32 for the second half.
                Audited (see gdn_port_NOTES.md): that byte map is EXACTLY what
                this swizzled tile produces for the logical coordinates below,
                for all 128 x 2 x 4 sites -- the xor and the +-32 are what the
                composed layout re-applies, so neither belongs in user code.
                """
                for half in range(2):
                    for band in range(4):
                        row = (tid1 & 7) | ((tid1 & 16) >> 1) | (tid1 & 64) | (band << 4)
                        col = (tid1 & 40) | (half << 4)
                        yield half * 16 + band * 4, view.ptr_to(row, col)

            def load_v_frag(vw, view):  # orig:L852-867
                for oo, addr in vo_octets(view):
                    K.ptx[LDM_X4T_V](vw[oo], vw[oo + 1], vw[oo + 2], vw[oo + 3], addr)

            def store_o_frag(words, view):  # orig:L870-885
                for oo, addr in vo_octets(view):
                    K.ptx[STM_X4T_O](addr, words[oo], words[oo + 1], words[oo + 2], words[oo + 3])

            def state_gidx(batch, head):  # orig:L894-897
                t = K.Cast("int64", tid1 & 127)
                return (K.Cast("int64", batch) * K.int64(HV) + K.Cast("int64", head)) * K.int64(
                    D_HEAD * D_HEAD
                ) + t * K.int64(D_HEAD)

            def load_initial_state(batch, head):  # orig:L888-912
                # acquire the kv ring (empty-wait), fill S, software-arrive the
                # FULL side (init_full=1: alternates with mma1's tcgen05.commit)
                p_kv.empty.wait(st_kvp.stage, st_kvp.phase)
                base = state_gidx(batch, head)
                sub = K.alloc_local([32], "uint32")
                for s in range(4):
                    for vec in range(8):
                        K.ptx[LD_G_V4](
                            sub[vec * 4],
                            sub[vec * 4 + 1],
                            sub[vec * 4 + 2],
                            sub[vec * 4 + 3],
                            initial_state.ptr_to([base + s * 32 + vec * 4]),
                        )
                    K.ptx[TC_ST_32](
                        tmem_at(tmem, TM_STATE + s * 32, rowbits1), *(sub[i] for i in range(32))
                    )
                K.ptx[WAIT_ST]()
                bar_state()
                with K.If((tid1 & 127) == 0), K.Then():
                    # p_kv's full side has TWO producers of different KINDS:
                    # mma1 commits it from the matrix engine (TCGen05Bar), cg1
                    # signals it in software here. `full="tcgen05"` names only
                    # the first, so this one is spelled as the instruction it
                    # is — orig:L911-912 (_software_commit).
                    K.ptx.mbarrier.arrive.shared.b64(
                        p_kv.full.ptr_to([st_kvp.stage]), K.uint32(1)
                    )  # guarded arrive ...
                st_kvp.advance()  # ... unguarded advance

            def store_final_state(batch, head):  # orig:L928-949
                p_kv.full.wait(st_kvc.stage, st_kvc.phase)
                base = state_gidx(batch, head)
                sub = K.alloc_local([32], "uint32")
                for s in range(4):
                    K.ptx[TC_LD_32](
                        *(sub[i] for i in range(32)), tmem_at(tmem, TM_STATE + s * 32, rowbits1)
                    )
                    # No tcgen05.wait::ld on any cg1 tmem load: the frozen kernel
                    # has none at these six sites (orig:L940/L1877/L1977/L2011/
                    # L2045/L2109) and lets the register dependency carry the
                    # ordering. Adding them cost ~0.5% and is the one place this
                    # port is knowingly *less* conservative than the ISA reads;
                    # see gdn_port_NOTES.md §7.
                    for vec in range(8):
                        K.ptx[ST_G_V4](
                            final_state.ptr_to([base + s * 32 + vec * 4]),
                            sub[vec * 4],
                            sub[vec * 4 + 1],
                            sub[vec * 4 + 2],
                            sub[vec * 4 + 3],
                        )
                p_kv.empty.arrive(st_kvc.stage)
                st_kvc.advance()

            work = K.alloc_local([1], "int32")
            K.assign(work[0], K.cta_id())
            with K.While(work[0] < total_work):
                batch, head, lo, hi, seqlen = work_coords(work[0])
                chunks = K.ceildiv(seqlen, PAIR_TOKENS) * 2
                with K.If(chunks > 0):
                    with K.Then():
                        load_initial_state(batch, head)
                        cp2 = K.alloc_local([1], "float32")
                        sv = K.alloc_local([128], "float32")
                        sw = K.alloc_local([64], "uint32")
                        cumprod_f = K.alloc_local([16], "float32")
                        decay_f = K.alloc_local([16], "float32")
                        cl2 = K.alloc_local([2], "float32")
                        ll = K.alloc_local([1], "float32")
                        dd = K.alloc_local([1], "uint64")
                        fr = K.alloc_local([64], "float32")
                        vw = K.alloc_local([32], "uint32")
                        w1 = K.alloc_local([1], "uint32")
                        nvw = K.alloc_local([32], "uint32")
                        dw = K.alloc_local([32], "uint32")
                        cbase = (tid1 << 1) & 6
                        with K.serial(chunks) as chunk:  # orig:L1850
                            with K.If((chunk & 1) == 0), K.Then():
                                # kv skip(2): mma1 produces these two rounds; a
                                # depth-1 ring makes two advances phase-neutral,
                                # kept as the visible ledger of foreign rounds
                                st_kvp.advance()
                                st_kvp.advance()

                            p_gate.full.wait(st_g1.stage, st_g1.phase)
                            K.ptx.ld.shared.f32(
                                cp2[0], K.address_of(s_cumprod[st_g1.stage, 63])
                            )  # cumprod_all — orig:L1861-1864

                            # state -> f16 state-input tmem; decay in place
                            p_kv.full.wait(st_kvc.stage, st_kvc.phase)
                            p_sinp.empty.wait(st_sinp.stage, st_sinp.phase)
                            for s in range(4):
                                K.ptx[TC_LD_32](
                                    *(sv[s * 32 + i] for i in range(32)),
                                    tmem_at(tmem, TM_STATE + s * 32, rowbits1),
                                )
                            for p in range(64):  # orig:L1884-1888
                                pack_f16x2(sw[p], sv[2 * p], sv[2 * p + 1])
                            for s in range(4):
                                K.ptx[TC_ST_16](
                                    tmem_at(tmem, TM_SINPUT + s * 16, rowbits1),
                                    *(sw[s * 16 + i] for i in range(16)),
                                )
                            K.ptx[WAIT_ST]()
                            p_sinp.full.arrive(st_sinp.stage)  # commit, all 128
                            st_sinp.advance()
                            for p in range(64):  # orig:L1896-1907
                                scale_pair(sv, p, f2(cp2[0], cp2[0]))
                            for s in range(4):
                                K.ptx[TC_ST_32](
                                    tmem_at(tmem, TM_STATE + s * 32, rowbits1),
                                    *(sv[s * 32 + i] for i in range(32)),
                                )
                            K.ptx[WAIT_ST]()
                            p_kv.empty.arrive(st_kvc.stage)  # orig:L1917
                            st_kvc.advance()

                            # per-column decay factors        orig:L1921-1960
                            K.ptx.ld.shared.f32(ll[0], K.address_of(s_cumsumlog[st_g1.stage, 63]))
                            for g in range(8):
                                col = cbase + g * 8
                                K.ptx["ld.shared.v2.f32"](
                                    cumprod_f[2 * g],
                                    cumprod_f[2 * g + 1],
                                    K.address_of(s_cumprod[st_g1.stage, col]),
                                )
                                K.ptx["ld.shared.v2.f32"](
                                    cl2[0], cl2[1], K.address_of(s_cumsumlog[st_g1.stage, col])
                                )
                                # Frozen PTX spells this as scalar negations
                                # feeding add.rn.f32x2; ptxas folds them into
                                # the packed subtraction, so express that.
                                K.ptx.sub.rn.f32x2(dd[0], f2(ll[0], ll[0]), f2(cl2[0], cl2[1]))
                                K.ptx.ex2.approx.ftz.f32(decay_f[2 * g], K.cuda.float2_x(dd[0]))
                                K.ptx.ex2.approx.ftz.f32(decay_f[2 * g + 1], K.cuda.float2_y(dd[0]))
                            p_gate.empty.arrive(st_g1.stage)  # release gate
                            st_g1.advance()

                            # VKS = V - KS*cumprod -> f16 tmem  orig:L1962-2005
                            p_v.full.wait(st_v.stage, st_v.phase)
                            load_v_frag(vw, s_v[st_v.stage])
                            p_cg1.full.wait(st_cg1c.stage, st_cg1c.phase)
                            cg1_acc_ld(fr, TM_CG1)
                            for h in range(2):
                                for g in range(8):
                                    for rep in range(2):
                                        p = h * 16 + g * 2 + rep
                                        scale_pair(
                                            fr, p, f2(cumprod_f[2 * g], cumprod_f[2 * g + 1])
                                        )
                            p_cg1.empty.arrive(st_cg1c.stage)  # release KS
                            st_cg1c.advance()
                            for p in range(32):  # orig:L1998-2002
                                pack_f16x2(w1[0], fr[2 * p], fr[2 * p + 1])
                                K.ptx.sub.f16x2(vw[p], vw[p], w1[0])
                            # VKS lands in the SHARED-input slot, not the Sf16
                            # slot. orig:L2003 stores to TMEM_SHARED_INPUT_COL;
                            # the sketch aliased it onto TM_SINPUT, which is
                            # still live -- mma1 releases p_sinp only after the
                            # QS MMA -- so the Sf16 operand got clobbered
                            # mid-MMA and chunk 0's QS came out garbage.
                            for half in range(2):
                                cg1_f16_st_half(TM_SHARED, vw, half)
                            K.ptx[WAIT_ST]()
                            m_vks.arrive(0)  # one-way signal

                            # QS *= cumprod * scale            orig:L2007-2036
                            p_qs.full.wait(st_qsc.stage, st_qsc.phase)
                            cg1_acc_ld(fr, TM_QSTATE)
                            for h in range(2):
                                for g in range(8):
                                    for rep in range(2):
                                        p = h * 16 + g * 2 + rep
                                        scale_pair(
                                            fr,
                                            p,
                                            f2(cumprod_f[2 * g], cumprod_f[2 * g + 1]),
                                            f2(scale, scale),
                                        )
                            cg1_acc_st(fr, TM_QSTATE)
                            K.ptx[WAIT_ST]()
                            p_qs.empty.arrive(st_qsc.stage)
                            st_qsc.advance()

                            # NV (f16) + decayed-V (f16) inputs orig:L2038-2097
                            p_cg1.full.wait(st_cg1c.stage, st_cg1c.phase)
                            with K.If(lane1 == 0), K.Then():
                                p_v.empty.arrive(st_v.stage)  # init_empty=4:
                            st_v.advance()  # guarded arrive, unguarded advance
                            cg1_acc_ld(fr, TM_CG1)
                            for p in range(32):
                                pack_f16x2(nvw[p], fr[2 * p], fr[2 * p + 1])
                            p_cg1.empty.arrive(st_cg1c.stage)  # release NV
                            st_cg1c.advance()
                            for h in range(2):
                                for g in range(8):
                                    for rep in range(2):
                                        p = h * 16 + g * 2 + rep
                                        scale_pair(fr, p, f2(decay_f[2 * g], decay_f[2 * g + 1]))
                            for half in range(2):  # interleaved halves
                                cg1_f16_st_half(TM_SHARED, nvw, half)
                                for p in range(16):
                                    pq = half * 16 + p
                                    pack_f16x2(dw[pq], fr[2 * pq], fr[2 * pq + 1])
                                cg1_f16_st_half(TM_SHARED + 32, dw, half)
                            K.ptx[WAIT_ST]()
                            m_nv.arrive(0)
                            m_dcy.arrive(0)

                            # O epilogue through the s_o ring   orig:L2099-2118
                            p_o.empty.wait(st_op.stage, st_op.phase)
                            p_qs.full.wait(st_qsc.stage, st_qsc.phase)
                            cg1_acc_ld(fr, TM_QSTATE)
                            for p in range(32):
                                pack_f16x2(nvw[p], fr[2 * p], fr[2 * p + 1])
                            store_o_frag(nvw, s_o[st_op.stage])
                            K.ptx[FENCE_ASYNC]()
                            p_qs.empty.arrive(st_qsc.stage)  # release QKV
                            st_qsc.advance()
                            p_o.full.arrive(st_op.stage)  # commit O
                            st_op.advance()
                        store_final_state(batch, head)
                    with K.Else():  # orig:L2132-2134
                        base = state_gidx(batch, head)
                        sval = K.alloc_local([1], "float32")
                        # The source deliberately keeps this loop rolled.
                        with K.serial(D_HEAD) as key:
                            K.ptx.ld.global_.f32(
                                sval[0], initial_state.ptr_to([base + K.Cast("int64", key)])
                            )
                            K.ptx.st.global_.f32(
                                final_state.ptr_to([base + K.Cast("int64", key)]), sval[0]
                            )
                K.assign(work[0], work[0] + grid_x)
            # tmem dealloc by owner at role exit — orig:L2136-2143
            K.cuda.warpgroup_sync(BAR_DEALLOC)
            with K.If(K.warp_id_in_role() == 0), K.Then():
                K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    K.Cast("uint32", tmem[0]), K.uint32(TMEM_COLS)
                )
            drain(p_o, st_op)
            drain(p_sinp, st_sinp)

        # ==================================================================
        # MMA issuer 0: KK0,KK1,QK0,QK1 per pair.         orig:L2144-2260
        # ==================================================================
        with mma0:
            for m in (q_map, k_map, v_map, o_map):  # orig:L1484-1488
                K.ptx.prefetch.tensormap(K.address_of(m))
            st_k = K.PipelineState(4, phase=0)  # consumer, p_k
            st_q = K.PipelineState(2, phase=0)  # consumer, p_q
            st_ap = K.PipelineState(2, phase=1)  # producer, p_cg0
            tmem = tmem_preamble()

            work = K.alloc_local([1], "int32")
            K.assign(work[0], K.cta_id())
            with K.While(work[0] < total_work):
                batch, head, lo, hi, seqlen = work_coords(work[0])
                with K.serial(K.ceildiv(seqlen, PAIR_TOKENS)) as _pair:
                    # K/Q stages stay in flight across the whole pair, so copy
                    # each stage and advance immediately; the arrays are
                    # released in orig's commit order at the end.
                    ks2 = K.alloc_local([2], "int32")
                    qs2 = K.alloc_local([2], "int32")
                    kviews = []
                    for j in range(2):  # KK_j = Kj @ Kj^T
                        p_cg0.empty.wait(st_ap.stage, st_ap.phase)  # acquire
                        K.assign(ks2[j], st_k.stage)
                        p_k.full.wait(ks2[j], st_k.phase)
                        st_k.advance()
                        # ks2[j] is an int local, so this view stays valid for
                        # the whole pair — st_k has already advanced past it.
                        # The SAME view object on both sides: K @ K^T encodes
                        # one descriptor, not two. k_range defaults to the
                        # tile's full 128-element k axis (8 phases at mma_k=16)
                        # and the B128 K-major walk comes from the tile layout
                        # — orig:L952-968.
                        kv = s_k[ks2[j]]
                        mma0_chain(tmem[0] + TM_CG0 + st_ap.stage * 64, kv, kv, elect_local())
                        p_cg0.full.arrive(st_ap.stage, pred=elect())  # commit
                        st_ap.advance()
                        kviews.append(kv)
                    for j in range(2):  # QK_j = Qj @ Kj^T
                        K.assign(qs2[j], st_q.stage)
                        p_q.full.wait(qs2[j], st_q.phase)
                        st_q.advance()
                        p_cg0.empty.wait(st_ap.stage, st_ap.phase)
                        mma0_chain(
                            tmem[0] + TM_CG0 + st_ap.stage * 64,
                            s_q[qs2[j]],
                            kviews[j],
                            elect_local(),
                        )
                        p_cg0.full.arrive(st_ap.stage, pred=elect())
                        st_ap.advance()
                    # Matrix-engine releases are commits, in Q0,Q1,K0,K1 order.
                    for j in range(2):
                        p_q.empty.arrive(qs2[j], pred=elect())
                    for j in range(2):
                        p_k.empty.arrive(ks2[j], pred=elect())
                K.assign(work[0], work[0] + grid_x)
            drain(p_cg0, st_ap)

        # ==================================================================
        # TMA loader: mutate descriptors per work item, stream K,Q,V.
        #                                                 orig:L2405-2592
        # ==================================================================
        with tma:
            st_kp = K.PipelineState(4, phase=1)  # producer, p_k
            st_qp = K.PipelineState(2, phase=1)  # producer, p_q
            st_vp = K.PipelineState(3, phase=1)  # producer, p_v
            # copy the three by-value TensorMap payloads into the CTA slots
            # once, Q -> K -> V — orig:L2416-2428
            with K.If(K.cta_id() < total_work), K.Then():
                for dsc, m in ((d_q, q_map), (d_k, k_map), (d_v, v_map)):
                    with K.If(elected()), K.Then():
                        copy_desc(dsc, m)
                    K.cuda.warp_sync()
                K.ptx.fence.acq_rel.cta()

            def load_chunk(off, chunk_idx, head, qk_head, sub):  # orig:L1098-1170
                # TMA dst is the box BASE ADDRESS (extent comes from the
                # tensormap): K/Q boxes land at column d, V boxes at row d.
                specs = (
                    (
                        d_k,
                        p_k,
                        st_kp,
                        TMA_G2S_QK,
                        lambda t, d: s_k[t].ptr_to(0, d),
                        lambda d: (K.int32(d), K.Cast("int32", off), qk_head),
                    ),
                    (
                        d_q,
                        p_q,
                        st_qp,
                        TMA_G2S_QK,
                        lambda t, d: s_q[t].ptr_to(0, d),
                        lambda d: (K.int32(d), K.Cast("int32", off), qk_head),
                    ),
                    (
                        d_v,
                        p_v,
                        st_vp,
                        TMA_G2S_V,
                        lambda t, d: s_v[t].ptr_to(d, 0),
                        lambda d: (
                            (K.int32(d), K.Cast("int32", off), sub, qk_head)
                            if RANK == 4
                            else (K.int32(d), K.Cast("int32", off), head)
                        ),
                    ),
                )
                for dsc, pipe, st, insn, dst, coords in specs:
                    pipe.empty.wait(st.stage, st.phase)  # acquire
                    with K.If(chunk_idx == 0), K.Then():  # orig:L1122-1124
                        with K.If(elected()), K.Then():
                            K.ptx[TMAP_ACQ](dsc)  # size operand is the ISA's literal 128
                    with K.If(elected()), K.Then():
                        # TMABar.arrive(tx_count=) IS arrive.expect_tx — the
                        # in-tree spelling of tx accounting.   orig:L1129
                        pipe.full.arrive(st.stage, tx_count=STAGE_BYTES)
                        for d in (0, 64):  # two boxes, one transaction
                            K.ptx[insn](
                                dst(st.stage, d),
                                dsc,
                                *coords(d),
                                pipe.full.ptr_to([st.stage]),
                                K.uint64(0),
                            )
                    st.advance()

            work = K.alloc_local([1], "int32")
            K.assign(work[0], K.cta_id())
            with K.While(work[0] < total_work):
                batch, head, lo, hi, seqlen = work_coords(work[0])
                qk_head, sub = head_split(head)
                chunks = K.ceildiv(seqlen, PAIR_TOKENS) * 2
                valid = K.ceildiv(seqlen, CHUNK)
                with K.If(chunks > 0), K.Then():
                    with K.If(elected()), K.Then():
                        K.ptx[BULK_WAIT](0)
                    K.cuda.warp_sync()  # orig:L2452-2455
                    with K.If(elected()), K.Then():  # orig:L2456-2484
                        replace_desc(
                            d_q,
                            K.address_of(q[K.int64(0)]),
                            hi,
                            HQ,
                            1,
                            2 * D_HEAD * HQ,
                            2 * D_HEAD,
                            0,
                        )
                        replace_desc(
                            d_k,
                            K.address_of(k[K.int64(0)]),
                            hi,
                            HQ,
                            1,
                            2 * D_HEAD * HQ,
                            2 * D_HEAD,
                            0,
                        )
                        replace_vo_desc(d_v, K.address_of(v[K.int64(0)]), hi)
                    K.cuda.warp_sync()
                    K.ptx[TMAP_REL]()
                    # all-but-last valid, explicit last valid, then even-pair
                    # padding; each call itself is K -> Q -> V.  orig:L2488-2586
                    with K.serial(valid - 1) as c:
                        load_chunk(lo + K.Cast("int64", c * CHUNK), c, head, qk_head, sub)
                    load_chunk(
                        lo + K.Cast("int64", (valid - 1) * CHUNK), valid - 1, head, qk_head, sub
                    )
                    with K.serial(valid, chunks) as c:  # padding chunks
                        load_chunk(lo + K.Cast("int64", c * CHUNK), c, head, qk_head, sub)
                K.assign(work[0], work[0] + grid_x)
            drain(p_q, st_qp)  # orig:L2590-2592
            drain(p_k, st_kp)
            drain(p_v, st_vp)

        # ==================================================================
        # MMA issuer 1: KS, QS, NV, QKV, state update.    orig:L2261-2404
        # ==================================================================
        with mma1:
            st_k1 = K.PipelineState(4, phase=0)  # consumer, p_k
            st_q1 = K.PipelineState(2, phase=0)  # consumer, p_q
            st_ac = K.PipelineState(3, phase=0)  # consumer, p_ainv
            st_qkc = K.PipelineState(2, phase=0)  # consumer, p_qk
            st_sc = K.PipelineState(1, phase=0)  # consumer, p_sinp
            st_vks = K.PipelineState(1, phase=0)  # consumer, m_vks
            st_nv = K.PipelineState(1, phase=0)  # consumer, m_nv
            st_dcy = K.PipelineState(1, phase=0)  # consumer, m_dcy
            st_c1p = K.PipelineState(1, phase=1)  # producer, p_cg1
            st_qsp = K.PipelineState(1, phase=1)  # producer, p_qs
            st_kvp = K.PipelineState(1, phase=1)  # producer, p_kv (shared ring
            #   with cg1's initial-state store — see the skip advance)
            tmem = tmem_preamble()

            # Every chain below is ts-form: A is a TMEM address and B a stage
            # view. Both the per-phase TMEM a-step (8 columns for f16) and the
            # phase count come from the operands, so neither appears here.
            #   orig:L971-987 (k128), orig:L990-1005 (k64), orig:L1008-1023
            #   (KV, the only MN-major B operand — its ID_KV sets trans_b, and
            #   the chain decodes that bit rather than being told).

            work = K.alloc_local([1], "int32")
            K.assign(work[0], K.cta_id())
            with K.While(work[0] < total_work):
                batch, head, lo, hi, seqlen = work_coords(work[0])
                chunks = K.ceildiv(seqlen, PAIR_TOKENS) * 2
                with K.serial(chunks) as chunk:
                    p_k.full.wait(st_k1.stage, st_k1.phase)  # held to the end
                    kv = s_k[st_k1.stage]  # ephemeral: used before st_k1 advances
                    p_q.full.wait(st_q1.stage, st_q1.phase)
                    qv = s_q[st_q1.stage]

                    p_cg1.empty.wait(st_c1p.stage, st_c1p.phase)  # KS acquire
                    p_sinp.full.wait(st_sc.stage, st_sc.phase)  # Sf16 ready
                    mma1_chain(  # KS = Sf16 @ K
                        tmem[0] + TM_CG1,
                        a=tmem[0] + TM_SINPUT,
                        b=kv,
                        phases=8,
                        transpose_b=False,
                        pred=elect_local(),
                        accumulate=False,
                    )
                    p_cg1.full.arrive(st_c1p.stage, pred=elect())  # tcgen05.commit
                    st_c1p.advance()

                    p_qs.empty.wait(st_qsp.stage, st_qsp.phase)  # QS acquire
                    mma1_chain(  # QS = Sf16 @ Q
                        tmem[0] + TM_QSTATE,
                        a=tmem[0] + TM_SINPUT,
                        b=qv,
                        phases=8,
                        transpose_b=False,
                        pred=elect_local(),
                        accumulate=False,
                    )
                    p_qs.full.arrive(st_qsp.stage, pred=elect())
                    st_qsp.advance()
                    p_sinp.empty.arrive(st_sc.stage, pred=elect())  # release
                    st_sc.advance()
                    p_q.empty.arrive(st_q1.stage, pred=elect())  # release
                    st_q1.advance()

                    p_cg1.empty.wait(st_c1p.stage, st_c1p.phase)  # NV acquire
                    m_vks.wait(0, st_vks.phase)
                    st_vks.advance()
                    p_ainv.full.wait(st_ac.stage, st_ac.phase)
                    mma1_chain(  # NV = VKS @ Ainv -- A is the SHARED slot
                        tmem[0] + TM_CG1,
                        a=tmem[0] + TM_SHARED,
                        b=s_ainv[st_ac.stage],
                        phases=4,
                        transpose_b=False,
                        pred=elect_local(),
                        accumulate=False,  # NV overwrites
                    )
                    p_cg1.full.arrive(st_c1p.stage, pred=elect())
                    st_c1p.advance()
                    p_ainv.empty.arrive(st_ac.stage, pred=elect())  # release
                    st_ac.advance()

                    p_qs.empty.wait(st_qsp.stage, st_qsp.phase)  # QKV acquire
                    p_qk.full.wait(st_qkc.stage, st_qkc.phase)
                    qkv = s_qk[st_qkc.stage]
                    m_nv.wait(0, st_nv.phase)
                    st_nv.advance()
                    mma1_chain(  # QKV += QK @ NV
                        tmem[0] + TM_QSTATE,
                        a=tmem[0] + TM_SHARED,
                        b=qkv,
                        phases=4,
                        transpose_b=False,
                        pred=elect_local(),
                        accumulate=True,  # always accumulates QS
                    )
                    p_qk.empty.arrive(st_qkc.stage, pred=elect())
                    st_qkc.advance()
                    p_qs.full.arrive(st_qsp.stage, pred=elect())
                    st_qsp.advance()

                    with K.If(chunk == 0), K.Then():  # orig:L2384-2385
                        st_kvp.advance()  # skip cg1's initial-state round
                    p_kv.empty.wait(st_kvp.stage, st_kvp.phase)
                    m_dcy.wait(0, st_dcy.phase)
                    st_dcy.advance()
                    mma1_chain(  # S += decayV @ K^T
                        # ID_KV's trans_b bit is what makes this B mn-major;
                        # the descriptor and phase walk both use that form.
                        tmem[0] + TM_STATE,
                        a=tmem[0] + TM_SHARED + 32,
                        b=s_k[st_k1.stage],
                        phases=4,
                        transpose_b=True,
                        pred=elect_local(),
                        accumulate=True,
                    )
                    p_kv.full.arrive(st_kvp.stage, pred=elect())
                    st_kvp.advance()
                    p_k.empty.arrive(st_k1.stage, pred=elect())  # release
                    st_k1.advance()
                K.assign(work[0], work[0] + grid_x)
            drain(p_cg1, st_c1p)
            drain(p_qs, st_qsp)
            drain(p_kv, st_kvp)

        # ==================================================================
        # AUX: gate/beta side-band producer + O store.    orig:L2595-2790
        # ==================================================================
        with aux:
            st_gp = K.PipelineState(5, phase=1)  # producer, p_gate
            st_bp = K.PipelineState(5, phase=1)  # producer, p_beta
            st_oc = K.PipelineState(2, phase=0)  # consumer, p_o
            lane_a = K.lane_id()
            with K.If(K.cta_id() < total_work), K.Then():  # orig:L2607-2611
                with K.If(elected()), K.Then():
                    copy_desc(d_o, o_map)
                K.cuda.warp_sync()
                K.ptx.fence.acq_rel.cta()

            def produce_gate_beta(off, head, is_last, hi):  # orig:L1173-1276
                # `is_last` is a RUNTIME predicate. Both last-tile comparisons
                # stay confined to the last-tile branch, matching the frozen
                # predicate lifetime — orig:L1194-1195.
                pos = [off + K.Cast("int64", lane_a), off + K.Cast("int64", lane_a + 32)]
                gsrc = [gate.ptr_to([p * K.int64(HV) + K.Cast("int64", head)]) for p in pos]
                bsrc = [beta.ptr_to([p * K.int64(HV) + K.Cast("int64", head)]) for p in pos]
                g = K.alloc_local([2], "float32")
                valid = K.alloc_local([2], "int32")
                for i in range(2):
                    K.assign(g[i], K.float32(1.0))
                    K.assign(valid[i], K.int32(1))
                with K.If(is_last):
                    with K.Then():
                        for i in range(2):
                            K.assign(valid[i], K.Cast("int32", pos[i] < hi))
                        for i in range(2):
                            with K.If(valid[i] != 0), K.Then():
                                K.ptx.ld.global_.f32(g[i], gsrc[i])
                    with K.Else():
                        for i in range(2):
                            K.ptx.ld.global_.f32(g[i], gsrc[i])
                for i in range(2):
                    K.ptx.lg2.approx.ftz.f32(g[i], g[i] + K.float32(1.0e-10))
                # inclusive +-scan over the warp's 64 gates — orig:L1223-1230.
                # chain=True joins the two 32-element half-scans into one
                # 64-element sequence; it is opt-in because the neutral reading
                # of "scan 2 values" is two independent scans.
                K.idioms.warp_scan_add(g, 2, lane_a)  # one 64-token sequence
                cp_ = K.alloc_local([2], "float32")
                for i in range(2):
                    K.ptx.ex2.approx.ftz.f32(cp_[i], g[i])

                # The register scan precedes stage acquisition in the frozen
                # source.
                p_gate.empty.wait(st_gp.stage, st_gp.phase)  # acquire
                for i in range(2):
                    K.ptx.st.shared.f32(
                        K.address_of(s_cumsumlog[st_gp.stage, lane_a + 32 * i]), g[i]
                    )
                for i in range(2):
                    K.ptx.st.shared.f32(
                        K.address_of(s_cumprod[st_gp.stage, lane_a + 32 * i]), cp_[i]
                    )
                p_gate.full.arrive(st_gp.stage)  # commit: 32 per-lane arrives
                st_gp.advance()

                p_beta.empty.wait(st_bp.stage, st_bp.phase)  # acquire
                bdst = [K.address_of(s_beta[st_bp.stage, lane_a + 32 * i]) for i in range(2)]
                with K.If(is_last):
                    with K.Then():
                        # The pre-store is LOAD-BEARING, not defensive. The
                        # copy below is predicated with `pred=`, which is
                        # whole-instruction @p predication and NOT the ISA's
                        # ignore-src: an out-of-range lane executes nothing at
                        # all and its destination keeps whatever was there, so
                        # the zero has to be written here. (The alternative is
                        # the ignore-src spelling, which zero-fills and needs
                        # no pre-store; orig uses this one.) K.ptx requires the
                        # trailing src-size operand precisely so that choice is
                        # visible -- src_size == cp_size means a full copy, no
                        # zero-fill.
                        for i in range(2):
                            K.ptx.st.shared.f32(bdst[i], K.float32(0.0))
                        for i in range(2):
                            K.ptx[CP_ASYNC](bdst[i], bsrc[i], 4, 4, pred=valid[i])
                    with K.Else():
                        for i in range(2):
                            K.ptx[CP_ASYNC](bdst[i], bsrc[i], 4, 4)
                # the full side commits by transaction, no software arrive:
                K.ptx[CP_ASYNC_ARRIVE](p_beta.full.ptr_to([st_bp.stage]))
                st_bp.advance()

            def store_o(off, head, qk_head, sub):  # orig:L1278-1321
                p_o.full.wait(st_oc.stage, st_oc.phase)
                with K.If(elected()), K.Then():
                    for d in (0, 64):
                        coords = (
                            (K.int32(d), K.Cast("int32", off), sub, qk_head)
                            if RANK == 4
                            else (K.int32(d), K.Cast("int32", off), head)
                        )
                        K.ptx[TMA_S2G_O](d_o, *coords, s_o[st_oc.stage].ptr_to(d, 0), K.uint64(0))
                    K.ptx[BULK_COMMIT]()
                    # wait only until the bulk ops have finished READING their
                    # source, which is all the O epilogue needs.
                    K.ptx[BULK_WAIT](0)
                p_o.empty.arrive(st_oc.stage)  # release: per-lane arrives
                st_oc.advance()

            work = K.alloc_local([1], "int32")
            K.assign(work[0], K.cta_id())
            with K.While(work[0] < total_work):
                batch, head, lo, hi, seqlen = work_coords(work[0])
                qk_head, sub = head_split(head)
                valid = K.ceildiv(seqlen, CHUNK)
                chunks = K.ceildiv(seqlen, PAIR_TOKENS) * 2
                with K.If(chunks > 0), K.Then():
                    with K.If(elected()), K.Then():
                        K.ptx[BULK_WAIT](0)
                    K.cuda.warp_sync()  # orig:L2636-2638
                    with K.If(elected()), K.Then():
                        replace_vo_desc(d_o, K.address_of(o[K.int64(0)]), hi)
                    K.cuda.warp_sync()  # orig:L2639-2663
                    K.ptx[TMAP_REL]()
                    with K.If(elected()), K.Then():
                        K.ptx[TMAP_ACQ](d_o)  # orig:L2664-2665

                    # exact fixed lookahead (0,1), then (2,3), then chunk+4
                    # immediately before the current O store.  orig:L2669-2699
                    for c in range(2):
                        produce_gate_beta(
                            lo + K.int64(c * CHUNK), head, K.int32(c) >= valid - 1, hi
                        )
                    with K.If(chunks > 2), K.Then():
                        for c in range(2, 4):
                            produce_gate_beta(
                                lo + K.int64(c * CHUNK), head, K.int32(c) >= valid - 1, hi
                            )
                    with K.serial(chunks) as c:  # steady state: +4 lookahead
                        coff = lo + K.Cast("int64", c * CHUNK)
                        with K.If(c + 4 < chunks), K.Then():
                            produce_gate_beta(
                                coff + K.int64(4 * CHUNK), head, c + 4 >= valid - 1, hi
                            )
                        store_o(coff, head, qk_head, sub)
                K.assign(work[0], work[0] + grid_x)
            drain(p_gate, st_gp)
            drain(p_beta, st_bp)

    return gdn_prefill


HEAD_PAIRS = ((2, 8), (4, 16), (8, 32), (16, 64), (16, 32), (16, 48), (16, 16), (32, 32))

SEQ_CASES = (
    ((65536,), "1x65536"),
    ((32768,), "1x32768"),
    ((16384,), "1x16384"),
    ((8192,), "1x8192"),
    ((4096,), "1x4096"),
    ((2048,), "1x2048"),
    ((6144, 2048), "6144+2048"),
    ((4096, 4096), "4096+4096"),
    ((2048, 6144), "2048+6144"),
    ((1024, 7168), "1024+7168"),
    ((2048,) * 4, "2048x4"),
    ((1024,) * 8, "1024x8"),
    ((8192,) * 8, "8192x8"),
    ((8192,) * 16, "8192x16"),
    ((8192,) * 32, "8192x32"),
)


@dataclass(frozen=True, slots=True)
class GDNPrefillSM100Config:
    label: str
    hq: int
    hv: int
    seq_lens: tuple[int, ...]
    seed: int = 0

    def validate(self) -> None:
        if (self.hq, self.hv) not in HEAD_PAIRS:
            raise ValueError(f"unsupported GDN head specialization {(self.hq, self.hv)}")
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError("seq_lens must be non-empty and positive")

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)


CONFIGS = [
    {
        "label": f"hq{hq}_hv{hv}_s{seq_label}",
        "hq": hq,
        "hv": hv,
        "seq_lens": seq_lens,
        "seed": 10000 + head_idx * len(SEQ_CASES) + seq_idx,
    }
    for head_idx, (hq, hv) in enumerate(HEAD_PAIRS)
    for seq_idx, (seq_lens, seq_label) in enumerate(SEQ_CASES)
]

KERNEL_META = {"name": "gdn_prefill_sm100", "category": "flashinfer", "compute_capability": 10}


def _cfg(**kwargs: Any) -> GDNPrefillSM100Config:
    cfg_fields = {field.name for field in fields(GDNPrefillSM100Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "seq_lens" in cfg_kwargs:
        cfg_kwargs["seq_lens"] = tuple(int(length) for length in cfg_kwargs["seq_lens"])
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = GDNPrefillSM100Config(**cfg_kwargs)
    cfg.validate()
    return cfg


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(
    tensor: torch.Tensor, *, global_dims: tuple[int, ...], global_strides_bytes: tuple[int, ...]
) -> _AlignedTensorMap:
    """Encode the frozen 64x64 FP16, B128-swizzled TMA tile."""
    import tvm

    rank = len(global_dims)
    if rank not in (3, 4):
        raise ValueError(f"GDN TensorMap rank must be 3 or 4, got {rank}")
    if len(global_strides_bytes) != rank - 1:
        raise ValueError("TensorMap global stride count must be rank - 1")

    descriptor = _AlignedTensorMap()
    box_dims = (64, 64, *((1,) * (rank - 2)))
    element_strides = (1,) * rank
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        "float16",
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        *global_strides_bytes,
        *box_dims,
        *element_strides,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        3,  # CU_TENSOR_MAP_SWIZZLE_128B
        3,  # CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def _build_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    """Build Q/K/V/O descriptors with the exact frozen logical coordinates."""
    cfg: GDNPrefillSM100Config = case["config"]
    qk_dims = (D_HEAD, cfg.total_tokens, cfg.hq)
    qk_strides = (2 * D_HEAD * cfg.hq, 2 * D_HEAD)

    if cfg.hq == cfg.hv:
        vo_dims = (D_HEAD, cfg.total_tokens, cfg.hv)
        vo_strides = (2 * D_HEAD * cfg.hv, 2 * D_HEAD)
    else:
        value_heads_per_q_head = cfg.hv // cfg.hq
        vo_dims = (D_HEAD, cfg.total_tokens, value_heads_per_q_head, cfg.hq)
        vo_strides = (2 * D_HEAD * cfg.hv, 2 * D_HEAD, 2 * D_HEAD * value_heads_per_q_head)

    return {
        "q": _encode_tensor_map(case["q"], global_dims=qk_dims, global_strides_bytes=qk_strides),
        "k": _encode_tensor_map(case["k"], global_dims=qk_dims, global_strides_bytes=qk_strides),
        "v": _encode_tensor_map(case["v"], global_dims=vo_dims, global_strides_bytes=vo_strides),
        "o": _encode_tensor_map(case["o"], global_dims=vo_dims, global_strides_bytes=vo_strides),
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    tensor_maps = case["tensor_maps"]
    return (
        case["q"].view(-1),
        case["k"].view(-1),
        case["v"].view(-1),
        case["gate"].view(-1),
        case["beta"].view(-1),
        case["o"].view(-1),
        case["cu_seqlens"],
        case["initial_state"].view(-1),
        case["final_state"].view(-1),
        tensor_maps["q"].ptr,
        tensor_maps["k"].ptr,
        tensor_maps["v"].ptr,
        tensor_maps["o"].ptr,
        case["descriptor_workspace"],
        case["total_tokens"],
        case["num_sequences"],
        case["num_sms"],
        case["scale"],
    )


@lru_cache(maxsize=1)
def _load_oracle():
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import chunk_gated_delta_rule_sm100

    return chunk_gated_delta_rule_sm100


def _run_oracle(case: dict[str, Any], output: torch.Tensor, final_state: torch.Tensor) -> None:
    oracle = _load_oracle()
    oracle(
        case["q"],
        case["k"],
        case["v"],
        case["gate"],
        case["beta"],
        output,
        case["cu_seqlens"],
        case["initial_state"],
        final_state,
        case["scale"],
    )


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for GDN prefill SM100")
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.total_tokens, cfg.hq, D_HEAD), dtype=torch.float16, device=device, generator=generator
    )
    k = F.normalize(
        torch.randn(
            (cfg.total_tokens, cfg.hq, D_HEAD),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        p=2,
        dim=-1,
    ).to(torch.float16)
    v = torch.randn(
        (cfg.total_tokens, cfg.hv, D_HEAD), dtype=torch.float16, device=device, generator=generator
    )
    gate = torch.rand(
        (cfg.total_tokens, cfg.hv), dtype=torch.float32, device=device, generator=generator
    )
    beta = torch.rand(
        (cfg.total_tokens, cfg.hv), dtype=torch.float32, device=device, generator=generator
    ).sigmoid()
    initial_state = torch.randn(
        (cfg.num_sequences, cfg.hv, D_HEAD, D_HEAD),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    final_state = torch.zeros_like(initial_state)
    o = torch.empty_like(v)

    endpoints = [0]
    for length in cfg.seq_lens:
        endpoints.append(endpoints[-1] + length)
    cu_seqlens = torch.tensor(endpoints, dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    descriptor_workspace = torch.empty(
        (num_sms * DESCRIPTOR_BYTES_PER_CTA,), dtype=torch.int8, device=device
    )

    case = {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "o": o,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "final_state": final_state,
        "descriptor_workspace": descriptor_workspace,
        "total_tokens": cfg.total_tokens,
        "num_sequences": cfg.num_sequences,
        "num_sms": num_sms,
        "scale": 1.0 / math.sqrt(D_HEAD),
    }
    case["tensor_maps"] = _build_tensor_maps(case)
    return case


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    return make_kernel(cfg.hq, cfg.hv).func


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for GDN prefill SM100")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise SkipTest(f"GDN prefill SM100 requires compute capability 10.x, got {capability}")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()

    reference_o = torch.empty_like(case["o"])
    reference_state = torch.empty_like(case["final_state"])
    _run_oracle(case, reference_o, reference_state)
    torch.cuda.synchronize()

    for name, tensor in (
        ("tirx output", case["o"]),
        ("tirx final state", case["final_state"]),
        ("frozen output", reference_o),
        ("frozen final state", reference_state),
    ):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")

    torch.testing.assert_close(case["o"], reference_o, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(case["final_state"], reference_state, atol=1e-3, rtol=1e-4)
    case["config"].validate()


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = dict(prepared["config"])
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    rounds = kwargs.pop("rounds", 5)
    cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for GDN prefill SM100 benchmark")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise SkipTest(f"GDN prefill SM100 requires compute capability 10.x, got {capability}")

    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    args = _tirx_args(case)

    def _flashinfer_cutedsl_builder():
        reference_o = torch.empty_like(case["o"])
        reference_state = torch.empty_like(case["final_state"])

        # First execution performs CuTeDSL JIT and initializes its persistent
        # workspace; subsequent executions are launch-only warmups.
        _run_oracle(case, reference_o, reference_state)
        for _ in range(2):
            _run_oracle(case, reference_o, reference_state)
        torch.cuda.synchronize()

        def launch():
            _run_oracle(case, reference_o, reference_state)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cutedsl": _flashinfer_cutedsl_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
