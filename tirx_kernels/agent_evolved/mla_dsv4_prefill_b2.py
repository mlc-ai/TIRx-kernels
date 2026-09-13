# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a DeepSeek-V4 sparse MLA prefill.

The supported contract is the varlen B=2 DSv4 sparse-MLA prefill row
``mla-dsv4-prefill-h128-swa16384-topk4x-c16384-k1024-bf16-hnd``: 386 total
query tokens over two requests with query lengths [129, 257],
``cum_seq_lens_q`` = [0, 129, 386], ``max_q_len`` = 257, H = 128, D = 512 and
K = 1152 sparse slots (128 SWA slots plus 1024 compressed slots). Both KV
pools are bf16 HND: ``swa_kv_cache`` is [130, 1, 256, 512] and
``compressed_kv_cache`` is [514, 1, 64, 512]. A single 512-dimensional latent
serves as both key and value.

Columns 0..127 of ``sparse_indices`` address flattened SWA token rows and
columns 128..1151 address flattened compressed rows; a slot participates only
when its index is nonnegative and its column is below ``sparse_topk_lens[t]``.
Per token and head the scores are ``bmm1_scale * dot(query, latent)``,
normalized by ``sum(exp(score)) + exp(sinks[h])`` -- the sink contributes a
logit but no value -- and the output is ``bmm2_scale`` times the
probability-weighted latent sum, cast to bf16.

The selected kernel is the ``dual-issuer`` frontier member of the 2026-09-12
DSv4 sparse-MLA-prefill evolution run. Everything from ``import math`` down to
the end of ``make_kernel`` is that candidate's source unchanged; this module
adds the registry interface, input generation, the independent oracle, and the
FlashInfer trtllm-gen reference arm.

Two invalid members that claimed 2.88x were removed from that run's frontier
before this port: they replaced ``softmax(QK^T)`` with uniform weights and
never read the query for scoring, so their output was query-independent and
only passed by overfitting the row's amplitude-0.05 inputs. This kernel is
exact attention.

Candidate mechanism notes, carried over from the evolution run:

Approach family: **dual-issuer over a static-balanced schedule**. The single
tcgen05 issuer of the static-balanced schedule is split into a QK-issuer warp
(8) and a PV-issuer warp (10) in the leader CTA. The QK stream waits only on
S-buffer release (``p_empty``), K-ring landing (``k_ready``) and the job's Q
copy (``q_consumed``), so it crosses job boundaries and keeps issuing while
the previous job's PVs drain; the PV warp issues each block as soon as its P
lands and owns the O accumulator. Q lives in TMEM, the latent K-split gather
uses ``cp.async.bulk.tensor.2d...tile::gather4`` on a 160 KiB gather ring, and
the two softmax warpgroups alternate blocks. The static one-wave scheduler
assigns tokens to clusters by a precomputed balanced token permutation rather
than a runtime work queue.

Register budget: output 120, gather 104, issuer 88 (the measured optimum; 128
for softmax and the 96/80/48 variants are neutral-to-worse). The epilogue
TMEM fragment is halved (x32) so ptxas reports zero spills at a 96-register
kernel cap.

The kernel is at the sm_100 hardware limit for this row: TMEM is 100% used
and shared memory sits at the measured allocation ceiling, so two CTAs per SM
-- the only way to fill the remaining tensor-core bubble -- is architecturally
unavailable (a 2-way pair split needs 448 TMEM columns per CTA and
``tcgen05.mma`` has no ``cta_group::4`` path).
"""

                                     
                                                                 
 
                                                                                           
                                                                       
                                                                            
                                                                                       
                                                                    
                                                                                     
                                                                                    
                                                
                                                                          
                                                                                     
                                                                                     
                                                                                     
                                                                                  
                                                                          
                                                                                      
                                                                                     
                                                                                      
                                                                                     
                                                                                   
                                                                                         
                                                                                        
                                                                                     

import math
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

B_H = 128
B_TOPK = 64
D_QK = 512
D_V = 512
SWA_COLS = 128                                               
NUM_UNITS = 5                                                               
UNIT_ELEMS = 64 * 256                          
LOG_2_E = math.log2(math.e)
BF16_BYTES = 2

KERNEL_NAME = "mla_dsv4_sparse_prefill_pkt_quad_static_dual_issuer_maskfirst"

LAUNCH_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_dyn_shared_memory",
)

_Q_CACHE_HINT = 0x12F0000000000000               
_KV_CACHE_HINT = 0x14F0000000000000              

_TMA_GATHER4 = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)
_TMA_Q_5D = (
    "cp.async.bulk.tensor.5d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)
_MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
_COMMIT_MC = (
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
)
_COMMIT_ONE = "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64"
_IDESC_QK = 0x08200490                                                      
_IDESC_PV = 0x08410490                           

                                                 
TMEM_O = 0                                                      
TMEM_Q = 256                                                                       
TMEM_S0 = 384                                    
TMEM_S1 = 448

SPIN_WAITS = True                                                    
POLY_EX2_DEG2 = (1.0017247200012207, 0.657636284828186, 0.3371894359588623)
FP32_ROUND_INT = float(2**23 + 2**22)


def _add_smem_desc_offset(dst, desc, offset):
    desc_lo = K.alloc_local((1,), "uint32")
    desc_hi = K.alloc_local((1,), "uint32")
    K.ptx.mov.b64(desc_lo[0], desc_hi[0], desc)
    K.ptx.add.u32(desc_lo[0], desc_lo[0], K.cast(offset, "uint32"))
    K.ptx.mov.b64(dst, desc_lo[0], desc_hi[0])


def make_kernel(s_q, topk, swa_rows, comp_rows):
    if topk % B_TOPK != 0 or topk < SWA_COLS:
        raise ValueError("topk must be a positive multiple of 64 including the 128 SWA slots")
    swa_blocks = SWA_COLS // B_TOPK
    grid_ctas = 152 if s_q == 386 else 2 * s_q

    def host_prelude(params):
        q = params["q"]
        swa = params["swa"]
        comp = params["comp"]

        def encode(data, rank, *shape):
            descriptor = K.stack_alloca("tensormap", 1)
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled", descriptor, "bfloat16", rank, data, *shape
            )
            return descriptor

        def pool_map(buf, rows):
            return encode(
                K.handle_add_byte_offset(buf.data, 0),
                2, D_QK, rows, D_QK * BF16_BYTES, 64, 1, 1, 1, 0, 3, 3, 0,
            )

        swa_tma = pool_map(swa, swa_rows)
        comp_tma = pool_map(comp, comp_rows)
                                                                                     
                                                                                                    
                                                                                            
        q_tma = encode(
            K.handle_add_byte_offset(q.data, 0),
            5,
            64, B_H, 2, 4, s_q,
            D_QK * BF16_BYTES, 256 * BF16_BYTES, 64 * BF16_BYTES, B_H * D_QK * BF16_BYTES,
            64, B_H // 2, 2, 2, 1,
            1, 1, 1, 1, 1,
            0, 3, 3, 0,
        )
        return swa_tma, comp_tma, q_tma

    @K.kernel(
        warps=20, arch="sm_100a", min_blocks_per_sm=1, grid=grid_ctas, host_prelude=host_prelude
    )
    def mla_dsv4_sparse_prefill_pkt_pingpong(
        q: K.gptr[K.bf16, (s_q, B_H, D_QK)],
        swa: K.gptr[K.bf16, (swa_rows * D_QK,)],
        comp: K.gptr[K.bf16, (comp_rows * D_QK,)],
        indices: K.gptr[K.i32, (s_q * topk,)],
        topk_lens: K.gptr[K.i32, (s_q,)],
        sinks: K.gptr[K.f32, (B_H,)],
        out: K.gptr[K.bf16, (s_q, B_H, D_V)],
        scale_log2: K.f32,
        bmm2_scale: K.f32,
        *,
        host,
    ):
        swa_tensormap, comp_tensormap, q_tma_tensormap = host
        block_idx = K.cta_id()
        K.cta_id_in_cluster([2], preferred=[2])
        thread_idx = K.thread_id()
        warp_idx = K.warp_id()
        lane_idx = K.lane_id()
        idx_in_warpgroup = K.thread_id_in_wg([128])
        cta_idx = block_idx % 2

        def prefetch(tensor_map):
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(tensor_map))

        prefetch(q_tma_tensormap)
        prefetch(swa_tensormap)
        prefetch(comp_tensormap)

        def iket_range(name):
            token = K.alloc_local((1,), "uint32")
            K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        smem = K.smem_pool()
        pool = smem.pool
                                                                                                  
        ring_smem = smem.alloc((NUM_UNITS * 64, 256), "bfloat16", swizzle=K.SW128B).buf
        s_smem_gemm = smem.alloc((2, 64, 64), "bfloat16", align=1024)            
        p_exchange = pool.alloc((2, 4, 1024), "uint32", align=128)                  
        rowwise_max_buf = pool.alloc((2, 128), "float32")                                  
        m_buf = pool.alloc((2, 64), "float32")                                            
        rowwise_li_buf = pool.alloc((2, 128), "float32")                                       
        rowwise_ref_buf = pool.alloc((2, 128), "float32")                                    
        rowwise_real_buf = pool.alloc((2, 128), "float32")                                   
        rowwise_scale_buf = pool.alloc((64,), "float32")
        is_k_valid = pool.alloc((4, 8), "int8", align=16)

        k_ready = K.TMABar(pool, NUM_UNITS)
        k_empty = K.TCGen05Bar(pool, NUM_UNITS)
        umma_ready = K.TCGen05Bar(pool, 2)
        p_empty = K.MBarrier(pool, 2)
        so_full = K.MBarrier(pool, 2)
        softmax_ready = K.TCGen05Bar(pool, 2)
        m_ready = K.MBarrier(pool, 2)
        q_consumed = K.MBarrier(pool, 1)
        tq_ready = K.TCGen05Bar(pool, 1)
        q_released = K.MBarrier(pool, 1)
        t_out_empty = K.MBarrier(pool, 1)
        li_full = K.MBarrier(pool, 1)
        li_empty = K.MBarrier(pool, 1)
        valid_full = K.MBarrier(pool, 4)
        valid_empty = K.MBarrier(pool, 4)
        clc_response_ready = K.TMABar(pool, 1)
        clc_empty = K.MBarrier(pool, 1)

        clc_response = pool.alloc((4,), "uint32", align=16)
        tmem_start_addr = pool.alloc((1,), "uint32", align=4)
        is_k_valid_byte_offset = int(is_k_valid.elem_offset)
        if is_k_valid_byte_offset % 16:
            raise ValueError("is_k_valid must be 16-byte aligned for the u32 view")
        is_k_valid_word_offset = is_k_valid_byte_offset // 4
        smem.commit()

        ring_base = K.address_of(ring_smem[0, 0])

        def ring_ptr(byte_offset):
            return K.ptr_byte_offset(ring_base, byte_offset, K.type_annotation("bfloat16"))

        class CLCJobScheduler:
            """One role-local walk over the shared CLC job stream."""

            def __init__(self):
                self.valid = K.local_scalar("int32")
                self.block_idx = K.local_scalar("int32")
                self.epoch = K.PipelineState(1, phase=0)
                K.assign(self.valid, 1)
                K.assign(self.block_idx, block_idx)

            def issue_cancel(self):
                return

            def advance(self):
                next_job = self.block_idx + grid_ctas
                with K.If(next_job >= 2 * s_q):
                    with K.Then():
                        K.assign(self.valid, 0)
                    with K.Else():
                        K.assign(self.block_idx, next_job)
                self.epoch.advance()

        def leader_bar(bar):
            """shared::cluster address of the pair leader's copy of an mbarrier."""
            mapped = K.local_scalar("uint32")
            K.ptx["mapa.shared::cluster.u32"](
                mapped, K.cuda.cvta_generic_to_shared(K.address_of(bar)), K.uint32(0)
            )
            return mapped

        def hot_wait(bar_elem, parity):
            """Wait for an mbarrier phase on a per-block handoff."""
            if not SPIN_WAITS:
                K.cuda.mbarrier_wait(K.address_of(bar_elem), parity)
                return
            done = K.local_scalar("uint32", init=K.uint32(0))
            addr = K.cuda.cvta_generic_to_shared(K.address_of(bar_elem))
            with K.While(done == K.uint32(0)):
                K.ptx["mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"](
                    done, addr, K.Cast("uint32", parity)
                )

        def ex2_emulation_2(out, idx, x, y):
            """Two-lane minimax-quadratic ex2 on the packed f32x2 datapath."""
            xy_clamped = K.alloc_local([2], "float32")
            K.ptx.mov.b32(xy_clamped[0], K.max(x, K.float32(-127.0)))
            K.ptx.mov.b32(xy_clamped[1], K.max(y, K.float32(-127.0)))
            packed = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            addend = K.local_scalar("uint64")
            xy_rounded = K.alloc_local([2], "float32")
            K.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
            K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
            K.ptx["add.rm.ftz.f32x2"](packed, packed, rhs)
            K.ptx.mov.b64(xy_rounded[0], xy_rounded[1], packed)
            xy_rounded_back = K.alloc_local([2], "float32")
            K.ptx.mov.b64(packed, xy_rounded[0], xy_rounded[1])
            K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
            K.ptx["sub.rn.ftz.f32x2"](packed, packed, rhs)
            K.ptx.mov.b64(xy_rounded_back[0], xy_rounded_back[1], packed)
            xy_frac = K.alloc_local([2], "float32")
            K.ptx.mov.b64(packed, xy_clamped[0], xy_clamped[1])
            K.ptx.mov.b64(rhs, xy_rounded_back[0], xy_rounded_back[1])
            K.ptx["sub.rn.ftz.f32x2"](packed, packed, rhs)
            K.ptx.mov.b64(xy_frac[0], xy_frac[1], packed)
            xy_frac_ex2 = K.alloc_local([2], "float32")
            K.ptx.mov.b32(xy_frac_ex2[0], K.float32(POLY_EX2_DEG2[2]))
            K.ptx.mov.b32(xy_frac_ex2[1], K.float32(POLY_EX2_DEG2[2]))
            for coeff in (POLY_EX2_DEG2[1], POLY_EX2_DEG2[0]):
                K.ptx.mov.b64(rhs, xy_frac[0], xy_frac[1])
                K.ptx.mov.b64(packed, xy_frac_ex2[0], xy_frac_ex2[1])
                K.ptx.mov.b64(addend, K.float32(coeff), K.float32(coeff))
                K.ptx["fma.rz.ftz.f32x2"](packed, packed, rhs, addend)
                K.ptx.mov.b64(xy_frac_ex2[0], xy_frac_ex2[1], packed)
            for j in range(2):
                x_rounded_i = K.local_scalar("int32")
                frac_ex_i = K.local_scalar("int32")
                x_rounded_e = K.local_scalar("int32")
                out_i = K.local_scalar("int32")
                K.ptx.mov.b32(x_rounded_i, xy_rounded[j])
                K.ptx.mov.b32(frac_ex_i, xy_frac_ex2[j])
                K.ptx.shl.b32(x_rounded_e, x_rounded_i, K.uint32(23))
                K.ptx.add.s32(out_i, x_rounded_e, frac_ex_i)
                K.ptx.mov.b32(out[idx + j], out_i)

        def num_blocks_of(s_q_idx):
            length = K.local_scalar("int32")
            K.ptx.ld.global_.s32(length, topk_lens.ptr_to([s_q_idx]))
            return K.max((length + B_TOPK - 1) // B_TOPK, 1)

        def scheduled_q_idx(job_block_idx):
            """Permute CLC jobs so its six-job workers receive the shortest tokens."""
            physical = job_block_idx // 2
            if s_q != 386:
                return physical
            schedule_round = physical // 76
            cluster = physical - schedule_round * 76
            token = K.local_scalar("int32")
            with K.If(cluster < 6):
                with K.Then():
                    K.assign(token, 350 + cluster * 6 + schedule_round)
                with K.Else():
                    regular = cluster - 6
                    with K.If(schedule_round == 0):
                        with K.Then():
                            K.assign(token, K.if_then_else(regular < 6, 64 + regular, regular - 6))
                        with K.Else():
                            with K.If(schedule_round == 1):
                                with K.Then():
                                    K.assign(
                                        token,
                                        K.if_then_else(
                                            regular < 6,
                                            70 + regular,
                                            K.if_then_else(regular < 47, 76 + regular, 100 + regular),
                                        ),
                                    )
                                with K.Else():
                                    with K.If(schedule_round == 2):
                                        with K.Then():
                                            with K.If(regular < 6):
                                                with K.Then():
                                                    K.assign(token, 76 + regular)
                                                with K.Else():
                                                    with K.If(regular < 24):
                                                        with K.Then():
                                                            K.assign(token, 192 + regular)
                                                        with K.Else():
                                                            with K.If(regular < 47):
                                                                with K.Then():
                                                                    K.assign(
                                                                        token,
                                                                        99
                                                                        + regular
                                                                        + K.Cast("int32", regular >= 29),
                                                                    )
                                                                with K.Else():
                                                                    K.assign(token, 123 + regular)
                                        with K.Else():
                                            with K.If(schedule_round == 3):
                                                with K.Then():
                                                    with K.If(regular < 6):
                                                        with K.Then():
                                                            K.assign(
                                                                token,
                                                                K.if_then_else(
                                                                    regular == 0,
                                                                    K.int32(128),
                                                                    192 + regular,
                                                                ),
                                                            )
                                                        with K.Else():
                                                            K.assign(
                                                                token,
                                                                K.if_then_else(
                                                                    regular < 24,
                                                                    210 + regular,
                                                                    K.if_then_else(
                                                                        regular < 47,
                                                                        251 + regular,
                                                                        187 + regular,
                                                                    ),
                                                                ),
                                                            )
                                                with K.Else():
                                                    K.assign(
                                                        token,
                                                        K.if_then_else(
                                                            regular < 6,
                                                            321 + regular,
                                                            K.if_then_else(
                                                                regular < 24,
                                                                251 + regular,
                                                                K.if_then_else(
                                                                    regular < 47,
                                                                    274 + regular,
                                                                    280 + regular,
                                                                ),
                                                            ),
                                                        ),
                                                    )
            return token

        def initialize_protocol():
            with K.If(warp_idx == 1):
                with K.Then():
                    with K.If(K.cuda.elect_sync()):
                        with K.Then():
                            for init_bar, arrive_count in (
                                (q_consumed, 1),
                                (tq_ready, 1),
                                (q_released, 1),
                                (t_out_empty, 256),
                                (li_full, 64),
                                (li_empty, 128),
                                (clc_response_ready, 1),
                                                                                                  
                                                                                  
                                (clc_empty, 779),
                            ):
                                with K.unroll(1) as i:
                                    K.ptx["mbarrier.init.shared.b64"](
                                        K.cuda.cvta_generic_to_shared(K.address_of(init_bar.buf[i])),
                                        K.uint32(arrive_count),
                                    )
                            with K.unroll(2) as sb:
                                for bar, count in ((umma_ready, 1), (p_empty, 256), (so_full, 256),
                                                   (softmax_ready, 1), (m_ready, 128)):
                                    K.ptx["mbarrier.init.shared.b64"](
                                        K.cuda.cvta_generic_to_shared(K.address_of(bar.buf[sb])),
                                        K.uint32(count),
                                    )
                            K.ptx["fence.mbarrier_init.release.cluster"]()
                with K.Else():
                    with K.If(warp_idx == 2):
                        with K.Then():
                            K.ptx["tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"](
                                K.cuda.cvta_generic_to_shared(K.address_of(tmem_start_addr[0])),
                                K.uint32(512),
                            )
                            allocated_tmem_addr = K.local_scalar("uint32")
                            K.ptx.ld.shared.u32(allocated_tmem_addr, tmem_start_addr.ptr_to([0]))
                            K.cuda.trap_when_assert_failed(allocated_tmem_addr == K.uint32(0))
                            K.ptx["tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"]()
                        with K.Else():
                            with K.If(warp_idx == 3), K.Then():
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    with K.unroll(NUM_UNITS) as unit:
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(K.address_of(k_ready.buf[unit])),
                                            K.uint32(1),
                                        )
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(K.address_of(k_empty.buf[unit])),
                                            K.uint32(1),
                                        )
                                    with K.unroll(4) as init_stage:
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(K.address_of(valid_full.buf[init_stage])),
                                            K.uint32(4),
                                        )
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(K.address_of(valid_empty.buf[init_stage])),
                                            K.uint32(128),
                                        )
                                    K.ptx["fence.mbarrier_init.release.cluster"]()
            K.cuda.cluster_sync()

        initialize_protocol()

                                                                                            
        def store_output(output_epoch, s_q_idx):
            """Scale O (TMEM) by the per-head softmax denominator and store bf16 to global."""
            K.cuda.mbarrier_wait(K.address_of(li_full.buf[0]), output_epoch)
            output_scale = K.local_scalar("float32")
            K.ptx.ld.shared.f32(output_scale, rowwise_scale_buf.ptr_to([idx_in_warpgroup % 64]))
            K.ptx["mbarrier.arrive.shared.b64"](
                K.cuda.cvta_generic_to_shared(K.address_of(li_empty.buf[0])), K.uint32(1)
            )
            K.cuda.mbarrier_wait(K.address_of(q_released.buf[0]), output_epoch)
            K.ptx["tcgen05.fence::after_thread_sync"]()
            head = cta_idx * 64 + idx_in_warpgroup % 64
            d_half = idx_in_warpgroup // 64
            out_row = K.ptr_byte_offset(
                out.ptr_to([s_q_idx, head, 0]), d_half * 256 * BF16_BYTES, "uint32"
            )
            output_storage = K.alloc_local((32,))
            bf16_storage = K.alloc_local((16,), "uint32")
            with K.unroll(8) as epi_k:
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                    *[output_storage[i] for i in range(32)],
                    K.cuda.get_tmem_addr(K.uint32(TMEM_O), 0, epi_k * 32),
                )
                K.ptx["tcgen05.wait::ld.sync.aligned"]()
                with K.If(epi_k == 7), K.Then():
                    K.ptx["tcgen05.fence::before_thread_sync"]()
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](leader_bar(t_out_empty.buf[0]))
                for f in range(16):
                    packed_values = K.local_scalar("uint64")
                    packed_scale = K.local_scalar("uint64")
                    K.ptx.mov.b64(packed_values, output_storage[f * 2], output_storage[f * 2 + 1])
                    K.ptx.mov.b64(packed_scale, output_scale, output_scale)
                    K.ptx["mul.rz.ftz.f32x2"](packed_values, packed_values, packed_scale)
                    K.ptx.mov.b64(output_storage[f * 2], output_storage[f * 2 + 1], packed_values)
                for f in range(16):
                    K.ptx.cvt.rn.bf16x2.f32(
                        bf16_storage[f], output_storage[f * 2 + 1], output_storage[f * 2]
                    )
                for f in range(2):
                    K.ptx["st.global.L1::no_allocate.v8.b32"](
                        K.ptr_byte_offset(out_row, (epi_k * 32 + f * 16) * BF16_BYTES, "uint32"),
                        bf16_storage[f * 8],
                        bf16_storage[f * 8 + 1],
                        bf16_storage[f * 8 + 2],
                        bf16_storage[f * 8 + 3],
                        bf16_storage[f * 8 + 4],
                        bf16_storage[f * 8 + 5],
                        bf16_storage[f * 8 + 6],
                        bf16_storage[f * 8 + 7],
                    )

        def q_load_output():
            q_o_token = iket_range("q-load-output")
            jobs = CLCJobScheduler()
            ring = K.PipelineState(NUM_UNITS, phase=0)
            last_valid = K.local_scalar("int32", init=0)
            last_s_q_idx = K.local_scalar("int32", init=0)
            with K.While(jobs.valid != 0):
                wg0_s_q_idx = scheduled_q_idx(jobs.block_idx)
                previous_epoch = K.bitwise_xor(jobs.epoch.phase, 1)
                n_blocks = num_blocks_of(wg0_s_q_idx)
                unit_lo = K.local_scalar("int32", init=ring.stage)
                phase_lo = K.local_scalar("int32", init=ring.phase)
                ring.advance()
                unit_hi = K.local_scalar("int32", init=ring.stage)
                phase_hi = K.local_scalar("int32", init=ring.phase)
                ring.advance()
                with K.If(cta_idx == 0), K.Then():
                    with K.If(warp_idx == 0), K.Then():
                        with K.If(K.cuda.elect_sync()), K.Then():
                                                                                                
                                                                                                
                                                                                                  
                                                                                              
                                                                                            
                                                                                      
                            qo_tq_tok = iket_range("qo-wait-tqready")
                            with K.If(last_valid != 0), K.Then():
                                K.cuda.mbarrier_wait(K.address_of(tq_ready.buf[0]), previous_epoch)
                            K.cuda.iket.range_end(qo_tq_tok[0])
                            for unit in (unit_lo, unit_hi):
                                K.ptx["mbarrier.arrive.expect_tx.shared.b64"](
                                    K.cuda.cvta_generic_to_shared(K.address_of(k_ready.buf[unit])),
                                    K.uint32(65536),
                                )
                            qo_qw_tok = iket_range("qo-wait-q")
                            K.cuda.mbarrier_wait(K.address_of(k_ready.buf[unit_lo]), phase_lo)
                            K.cuda.mbarrier_wait(K.address_of(k_ready.buf[unit_hi]), phase_hi)
                            K.cuda.iket.range_end(qo_qw_tok[0])
                            K.ptx["tcgen05.fence::after_thread_sync"]()
                            cp_desc = K.local_scalar("uint64")
                            K.cuda.tcgen05.encode_matrix_descriptor(
                                cp_desc.source.data, K.reinterpret(K.handle().ty, K.uint64(0)), 1, 64, 3
                            )
                            for unit_sel, unit in ((0, unit_lo), (1, unit_hi)):
                                for p_local in range(2):
                                    for kq in range(4):
                                        p_glob = 2 * unit_sel + p_local
                                        src_byte = (
                                            unit * (UNIT_ELEMS * BF16_BYTES)
                                            + p_local * 16384
                                            + kq * 32
                                        )
                                        K.ptx["tcgen05.cp.cta_group::2.128x256b"](
                                            K.Cast("uint32", TMEM_Q + p_glob * 32 + kq * 8),
                                            K.bitwise_or(
                                                K.bitwise_and(cp_desc, K.bitwise_not(K.uint64(16383))),
                                                K.Cast(
                                                    "uint64",
                                                    K.bitwise_and(
                                                        K.shift_right(
                                                            K.cuda.cvta_generic_to_shared(ring_ptr(src_byte)),
                                                            K.uint32(4),
                                                        ),
                                                        K.uint32(16383),
                                                    ),
                                                ),
                                            ),
                                        )
                                                                                                  
                            for unit in (unit_lo, unit_hi):
                                K.ptx[_COMMIT_MC](
                                    K.cuda.cvta_generic_to_shared(K.address_of(k_empty.buf[unit])),
                                    K.Cast("uint16", 3),
                                )
                            K.ptx[_COMMIT_MC](
                                K.cuda.cvta_generic_to_shared(K.address_of(q_consumed.buf[0])),
                                K.Cast("uint16", 3),
                            )
                                                                               
                                            
                ring_linear = K.local_scalar("int32", init=ring.stage + n_blocks)
                K.assign(
                    ring.phase,
                    ring.phase ^ K.bitwise_and(ring_linear // NUM_UNITS, K.int32(1)),
                )
                K.assign(ring.stage, ring_linear % NUM_UNITS)
                with K.If(last_valid != 0), K.Then():
                    qo_store_tok = iket_range("qo-store")
                    store_output(previous_epoch, last_s_q_idx)
                    K.cuda.iket.range_end(qo_store_tok[0])
                K.assign(last_valid, 1)
                K.assign(last_s_q_idx, wg0_s_q_idx)
                jobs.advance()
            with K.If(last_valid != 0), K.Then():
                last_epoch = K.bitwise_xor(jobs.epoch.phase, 1)
                store_output(last_epoch, last_s_q_idx)
            K.ptx["tcgen05.fence::before_thread_sync"]()
            K.ptx["bar.sync"](K.uint32(0), K.uint32(128))
            with K.If(warp_idx == 0), K.Then():
                K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](K.uint32(0), K.uint32(512))
            K.cuda.iket.range_end(q_o_token[0])

                                                                                            
        def kv_gather():
            kv_gather_token = iket_range("kv-gather")
            wg1_warp_idx = thread_idx // 32 - 4
            with K.If(K.cuda.elect_sync()), K.Then():
                jobs = CLCJobScheduler()
                ring = K.PipelineState(NUM_UNITS, phase=0)
                mask_pipe = K.PipelineState(4, phase=0)
                cur_indices = K.alloc_local((16,), "int32")
                nxt_indices = K.alloc_local((16,), "int32")
                cur_u32 = K.decl_buffer((16,), "int32", data=cur_indices.data, scope="local").view("uint32")
                nxt_u32 = K.decl_buffer((16,), "int32", data=nxt_indices.data, scope="local").view("uint32")

                def load_indices(dst_u32, s_q_idx, k):
                    """The 16 rows this warp gathers: {8w..8w+7} and {32+8w..32+8w+7}."""
                    with K.unroll(2) as local_row:
                        row_base = s_q_idx * topk + k * B_TOPK + local_row * 32 + wg1_warp_idx * 8
                        K.ptx["ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.u32"](
                            *[dst_u32[local_row * 8 + i] for i in range(8)],
                            K.address_of(indices[row_base]),
                        )

                with K.While(jobs.valid != 0):
                    wg1_s_q_idx = scheduled_q_idx(jobs.block_idx)
                    wg1_topk_len = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(wg1_topk_len, topk_lens.ptr_to([wg1_s_q_idx]))
                    wg1_num_k_blocks = K.max((wg1_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    load_indices(cur_u32, wg1_s_q_idx, 0)
                                                                                    
                    for unit_sel in range(2):
                        with K.If(wg1_warp_idx == 0), K.Then():
                            K.cuda.mbarrier_wait(
                                K.address_of(k_empty.buf[ring.stage]), K.bitwise_xor(ring.phase, 1)
                            )
                            K.ptx[_TMA_Q_5D](
                                K.cuda.cvta_generic_to_shared(
                                    ring_ptr(ring.stage * (UNIT_ELEMS * BF16_BYTES))
                                ),
                                K.reinterpret(K.handle().ty, K.address_of(q_tma_tensormap)),
                                0,
                                cta_idx * 64,
                                0,
                                2 * unit_sel,
                                wg1_s_q_idx,
                                leader_bar(k_ready.buf[ring.stage]),
                                K.uint64(_Q_CACHE_HINT),
                            )
                        ring.advance()
                    with K.serial(wg1_num_k_blocks, unroll=False) as k:
                                                                                           
                        with K.If(k + 1 < wg1_num_k_blocks), K.Then():
                            load_indices(nxt_u32, wg1_s_q_idx, k + 1)
                                                                                         
                        pool_rows = K.if_then_else(k < swa_blocks, K.int32(swa_rows), K.int32(comp_rows))
                        gt_mask_tok = iket_range("gt-mask")
                        K.cuda.mbarrier_wait(
                            K.address_of(valid_empty.buf[mask_pipe.stage]),
                            K.bitwise_xor(mask_pipe.phase, 1),
                        )
                        for local_row in range(2):
                            pos0 = k * B_TOPK + local_row * 32 + wg1_warp_idx * 8
                            terms = []
                            for j in range(8):
                                idx = cur_indices[local_row * 8 + j]
                                valid = K.bitwise_and(
                                    K.bitwise_and(idx >= 0, idx < pool_rows), pos0 + j < wg1_topk_len
                                )
                                terms.append(K.Select(valid, K.int32(1 << j), K.int32(0)))
                            while len(terms) > 1:
                                terms = [K.bitwise_or(terms[j], terms[j + 1]) for j in range(0, len(terms), 2)]
                            K.ptx.st.shared.b8(
                                is_k_valid.ptr_to([mask_pipe.stage, local_row * 4 + wg1_warp_idx]),
                                K.reinterpret("uint8", K.Cast("int8", terms[0])),
                            )
                        K.ptx["mbarrier.arrive.shared.b64"](
                            K.cuda.cvta_generic_to_shared(K.address_of(valid_full.buf[mask_pipe.stage])),
                            K.uint32(1),
                        )
                        K.cuda.iket.range_end(gt_mask_tok[0])
                        mask_pipe.advance()
                        gt_wait_tok = iket_range("gt-wait-empty")
                        K.cuda.mbarrier_wait(
                            K.address_of(k_empty.buf[ring.stage]), K.bitwise_xor(ring.phase, 1)
                        )
                        K.cuda.iket.range_end(gt_wait_tok[0])
                        gt_issue_tok = iket_range("gt-issue")
                        src_col = cta_idx * 256
                        mbar = leader_bar(k_ready.buf[ring.stage])

                        def issue_gather(tensor_map):
                            with K.unroll(4) as row_group:
                                with K.unroll(4) as col_atom:
                                    kv_dst_offset = (
                                        ring.stage * UNIT_ELEMS
                                        + wg1_warp_idx * 512
                                        + row_group // 2 * 2048
                                        + row_group % 2 * 256
                                        + col_atom * 4096
                                    ) * BF16_BYTES
                                    K.ptx[_TMA_GATHER4](
                                        K.cuda.cvta_generic_to_shared(ring_ptr(kv_dst_offset)),
                                        K.reinterpret(K.handle().ty, K.address_of(tensor_map)),
                                        src_col + col_atom * 64,
                                        cur_indices[row_group * 4],
                                        cur_indices[row_group * 4 + 1],
                                        cur_indices[row_group * 4 + 2],
                                        cur_indices[row_group * 4 + 3],
                                        mbar,
                                        K.uint64(_KV_CACHE_HINT),
                                    )

                        with K.If(k < swa_blocks):
                            with K.Then():
                                issue_gather(swa_tensormap)
                            with K.Else():
                                issue_gather(comp_tensormap)
                        K.cuda.iket.range_end(gt_issue_tok[0])
                        ring.advance()
                        with K.If(k + 1 < wg1_num_k_blocks), K.Then():
                            for i in range(16):
                                K.assign(cur_indices[i], nxt_indices[i])
                    jobs.advance()
            K.cuda.iket.range_end(kv_gather_token[0])

                                                                                                                        
        def qk_issuer():
            """Leader warp 8: stream the QK^T MMAs over the continuous block sequence.

            Its only waits are the S buffer release (p_empty), the K ring (k_ready) and
            the job's Q copy (q_consumed).  It never waits on PV progress, so the next
            job's first QK blocks are computed while the previous job's PVs drain and
            the softmax of the new job overlaps the old job's tail.
            """
            qk_role_token = iket_range("qk-issue-role")
            with K.If(K.cuda.elect_sync()), K.Then():
                jobs = CLCJobScheduler()
                qk_ring = K.PipelineState(NUM_UNITS, phase=0)
                qk_blk = K.PipelineState(4, phase=0)                                          

                def issue_qk(is_last):
                    s_buf = K.bitwise_and(qk_blk.stage, 1)
                    mm_wp_tok = iket_range("mm-wait-pempty")
                    hot_wait(
                        p_empty.buf[s_buf],
                        K.bitwise_xor(K.bitwise_and(K.shift_right(qk_blk.stage, 1), 1), 1),
                    )
                    K.cuda.iket.range_end(mm_wp_tok[0])
                    K.ptx["mbarrier.arrive.expect_tx.shared.b64"](
                        K.cuda.cvta_generic_to_shared(K.address_of(k_ready.buf[qk_ring.stage])),
                        K.uint32(65536),
                    )
                    mm_wk_tok = iket_range("mm-wait-kready")
                    hot_wait(k_ready.buf[qk_ring.stage], qk_ring.phase)
                    K.cuda.iket.range_end(mm_wk_tok[0])
                    mm_qk_tok = iket_range("mm-qk-issue")
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    descB_local = K.local_scalar("uint64")
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(descB_local), ring_base, 512, 64, 3
                    )
                    s_col = K.Cast("uint32", TMEM_S0 + 64 * s_buf)
                    with K.unroll(16) as ki:
                        descB_off = K.local_scalar("uint64")
                        _add_smem_desc_offset(
                            descB_off,
                            descB_local,
                            (qk_ring.stage * UNIT_ELEMS + ki // 4 * 4096 + ki % 4 * 16) // 8,
                        )
                        K.ptx[_MMA_F16](
                            s_col,
                            K.Cast("uint32", ki * 8 + TMEM_Q),
                            descB_off,
                            K.uint32(_IDESC_QK),
                            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                            K.Or(ki != 0, K.bool(False)),
                        )
                    K.ptx[_COMMIT_MC](
                        K.cuda.cvta_generic_to_shared(K.address_of(umma_ready.buf[s_buf])),
                        K.Cast("uint16", 3),
                    )
                    with K.If(is_last), K.Then():
                        K.ptx[_COMMIT_ONE](K.cuda.cvta_generic_to_shared(K.address_of(tq_ready.buf[0])))
                    K.cuda.iket.range_end(mm_qk_tok[0])
                    qk_ring.advance()
                    qk_blk.advance()


                with K.While(jobs.valid != 0):
                    qk_s_q_idx = scheduled_q_idx(jobs.block_idx)
                    n_blocks = num_blocks_of(qk_s_q_idx)
                                                                  
                    for _ in range(2):
                        qk_ring.advance()
                    mm_wq_tok = iket_range("mm-wait-qconsumed")
                    K.cuda.mbarrier_wait(K.address_of(q_consumed.buf[0]), jobs.epoch.phase)
                    K.cuda.iket.range_end(mm_wq_tok[0])
                    with K.serial(n_blocks, unroll=False) as k:
                        issue_qk(k == n_blocks - 1)
                    jobs.advance()
            K.cuda.iket.range_end(qk_role_token[0])

        def pv_issuer():
            """Leader warp 10: issue each block's PV as soon as its P lands.

            Owns the O accumulator ordering, the K ring release (k_empty: QK(k) finished
            before P(k) could exist, so the PV commit covers both readers) and the
            per-job q_released commit (all of the job's MMAs are complete once its last
            PV is, for the same reason).
            """
            pv_role_token = iket_range("qk-pv-issue")
            with K.If(K.cuda.elect_sync()), K.Then():
                jobs = CLCJobScheduler()
                pv_ring = K.PipelineState(NUM_UNITS, phase=0)
                pv_blk = K.PipelineState(4, phase=0)

                def issue_pv(is_first):
                    pbuf = K.bitwise_and(pv_blk.stage, 1)
                    mm_ws_tok = iket_range("mm-wait-sofull")
                    hot_wait(so_full.buf[pbuf], K.bitwise_and(K.shift_right(pv_blk.stage, 1), 1))
                    with K.If(is_first), K.Then():
                        K.cuda.mbarrier_wait(
                            K.address_of(t_out_empty.buf[0]), K.bitwise_xor(jobs.epoch.phase, 1)
                        )
                    K.cuda.iket.range_end(mm_ws_tok[0])
                    mm_pv_tok = iket_range("mm-pv-issue")
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    o_accumulate = K.local_scalar(
                        "uint32", init=K.if_then_else(is_first, K.uint32(0), K.uint32(1))
                    )
                    descA_local = K.local_scalar("uint64")
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(descA_local), K.address_of(s_smem_gemm[pbuf, 0, 0]), 64, 8, 0
                    )
                    descB_local = K.local_scalar("uint64")
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(descB_local), ring_base, 512, 64, 3
                    )
                    for n_half, b_half in ((0, 0), (128, 8192)):
                        with K.unroll(4) as ki:
                            descA_off = K.local_scalar("uint64")
                            _add_smem_desc_offset(descA_off, descA_local, (ki * 1024) // 8)
                            descB_off = K.local_scalar("uint64")
                            _add_smem_desc_offset(
                                descB_off,
                                descB_local,
                                (pv_ring.stage * UNIT_ELEMS + ki * 1024 + b_half) // 8,
                            )
                            K.ptx[_MMA_F16](
                                K.Cast("uint32", TMEM_O + n_half),
                                descA_off,
                                descB_off,
                                K.uint32(_IDESC_PV),
                                K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                                K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
                                K.Or(ki != 0, K.Cast("bool", o_accumulate)),
                            )
                    K.ptx[_COMMIT_MC](
                        K.cuda.cvta_generic_to_shared(K.address_of(softmax_ready.buf[pbuf])),
                        K.Cast("uint16", 3),
                    )
                    K.ptx[_COMMIT_MC](
                        K.cuda.cvta_generic_to_shared(K.address_of(k_empty.buf[pv_ring.stage])),
                        K.Cast("uint16", 3),
                    )
                    K.cuda.iket.range_end(mm_pv_tok[0])
                    pv_ring.advance()
                    pv_blk.advance()


                with K.While(jobs.valid != 0):
                    pv_s_q_idx = scheduled_q_idx(jobs.block_idx)
                    n_blocks = num_blocks_of(pv_s_q_idx)
                    for _ in range(2):
                        pv_ring.advance()
                    with K.serial(n_blocks, unroll=False) as k:
                        issue_pv(k == 0)
                    K.ptx["tcgen05.fence::before_thread_sync"]()
                    K.ptx[_COMMIT_MC](
                        K.cuda.cvta_generic_to_shared(K.address_of(q_released.buf[0])),
                        K.Cast("uint16", 3),
                    )
                    jobs.advance()
            K.cuda.iket.range_end(pv_role_token[0])

        def valid_mask():
            """Retired: the gather warps produce the validity mask."""
            return

        def clc_role(active: K.constexpr):
            clc_token = K.alloc_local((1,), "uint32")
            K.assign(clc_token[0], K.cuda.iket.sentinel_token("clc"))
            if active:
                K.assign(clc_token[0], K.cuda.iket.range_start("clc"))
                with K.If(K.cuda.elect_sync()), K.Then():
                    jobs = CLCJobScheduler()
                    with K.While(jobs.valid != 0):
                        jobs.issue_cancel()
                        jobs.advance()
            K.cuda.iket.range_end(clc_token[0])

                                                                                         
        def softmax(sel: K.constexpr):
            """Softmax warpgroup ``sel`` handles blocks k with k % 2 == sel (S/P buffer sel).

            Block k needs the running max m_{k-1} produced by the other warpgroup; it is
            handed off through ``m_buf[(k-1) % 2]`` / ``m_ready``. Each warpgroup keeps its
            own partial row sum ``li`` relative to ``m_ref`` (the last max it used) and the
            two partials are combined at the end of the job.
            """
            softmax_token = iket_range("softmax")
            local_warp_idx = warp_idx - (12 + 4 * sel)
            other = 1 - sel
            s_col = K.uint32(TMEM_S0 + 64 * sel)
            pair_bar = K.Cast("uint32", 2 + 2 * sel + K.bitwise_and(local_warp_idx, 1))
            head_in_half = idx_in_warpgroup % 64
            jobs = CLCJobScheduler()
                                                                                        
                                                                                         
                                                                                              
            valid_ring = K.RingState(4, phase=0, stage=sel, stride=2)
            sblk = K.PipelineState(1, phase=0)                                  
            c0 = K.local_scalar("int32", init=K.int32(0))                                         
            with K.While(jobs.valid != 0):
                wg3_s_q_idx = scheduled_q_idx(jobs.block_idx)
                wg3_num_k_blocks = num_blocks_of(wg3_s_q_idx)
                k_first = K.bitwise_and(c0 + sel, 1)
                m_ref = K.local_scalar("float32", init=K.float32(-1000000000000000019884624838656.0))
                li = K.local_scalar("float32", init=K.float32(0.0))
                real_mi = K.local_scalar("float32", init=K.float32("-inf"))
                scale_pair = K.local_scalar("uint64", init=K.cuda.make_float2(scale_log2, scale_log2))
                my_blocks = (wg3_num_k_blocks - k_first + 1) // 2
                with K.serial(my_blocks, unroll=False) as kk:
                    k = kk * 2 + k_first
                    v_stage = K.Cast("int32", valid_ring.stage)
                    v_phase = K.Cast("int32", valid_ring.phase)
                    sm_wait_tok = iket_range("sm-wait-s")
                    hot_wait(valid_full.buf[v_stage], v_phase)
                    p = K.alloc_local((32,), "uint32")
                    p_peer = K.alloc_local((32,), "uint32")
                    hot_wait(umma_ready.buf[sel], sblk.phase)
                    K.cuda.iket.range_end(sm_wait_tok[0])
                    sm_math_tok = iket_range("sm-math")
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    peer_col = K.if_then_else(local_warp_idx < 2, K.uint32(32), K.uint32(0))
                    own_col = K.uint32(32) - peer_col
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[p_peer[i] for i in range(32)], K.cuda.get_tmem_addr(s_col, 0, peer_col)
                    )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[p[i] for i in range(32)], K.cuda.get_tmem_addr(s_col, 0, own_col)
                    )
                    with K.unroll(8) as exchange_i:
                        exchange_offset: K.int32 = exchange_i * 32 * 4 + lane_idx * 4
                        p_peer_offset = exchange_i * 4
                        K.ptx["st.shared.v4.u32"](
                            K.cuda.cvta_generic_to_shared(
                                K.address_of(p_exchange[sel, K.bitwise_xor(local_warp_idx, 2), exchange_offset])
                            ),
                            p_peer[p_peer_offset], p_peer[p_peer_offset + 1],
                            p_peer[p_peer_offset + 2], p_peer[p_peer_offset + 3],
                        )
                    valid_word_offset = K.if_then_else(local_warp_idx >= 2, 1, 0)
                    buffer_18 = K.decl_buffer(
                        (4, 2), "uint32", data=is_k_valid.data, elem_offset=is_k_valid_word_offset,
                        scope="shared.dyn", align=16,
                    )
                    is_k_valid_u32 = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(is_k_valid_u32, buffer_18.ptr_to([v_stage, valid_word_offset]))
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx["tcgen05.fence::before_thread_sync"]()
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](leader_bar(p_empty.buf[sel]))
                    with K.If(is_k_valid_u32 != K.uint32(4294967295)), K.Then():
                        with K.unroll(32) as p_i:
                            invalid_p_predicate = K.bitwise_and(
                                K.shift_right(is_k_valid_u32, K.Cast("uint32", p_i)), K.uint32(1)
                            ) == K.uint32(0)
                            K.ptx.mov.b32(p[p_i], K.if_then_else(invalid_p_predicate, K.uint32(4286578688), p[p_i]))
                    sum_pair0 = K.local_scalar("uint64")
                    sum_pair1 = K.local_scalar("uint64")
                    mx = K.alloc_local((8,), "float32")
                    K.ptx["bar.sync"](pair_bar, K.uint32(64))
                    with K.unroll(8) as exchange_i:
                        exchange_offset: K.int32 = exchange_i * 32 * 4 + lane_idx * 4
                        p_exchange_tmp = K.alloc_local((4,), "uint32")
                        K.ptx["ld.shared.v4.u32"](
                            p_exchange_tmp[0], p_exchange_tmp[1], p_exchange_tmp[2], p_exchange_tmp[3],
                            K.cuda.cvta_generic_to_shared(K.address_of(p_exchange[sel, local_warp_idx, exchange_offset])),
                        )
                        p_pair0 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p[exchange_i * 4]), K.cuda.uint_as_float(p[exchange_i * 4 + 1])
                        )
                        peer_pair0 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p_exchange_tmp[0]), K.cuda.uint_as_float(p_exchange_tmp[1])
                        )
                        K.ptx["add.rn.f32x2"](sum_pair0, p_pair0, peer_pair0)
                        K.ptx.mov.b32(p[exchange_i * 4], K.cuda.float_as_uint(K.cuda.float2_x(sum_pair0)))
                        K.ptx.mov.b32(p[exchange_i * 4 + 1], K.cuda.float_as_uint(K.cuda.float2_y(sum_pair0)))
                        p_pair1 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p[exchange_i * 4 + 2]), K.cuda.uint_as_float(p[exchange_i * 4 + 3])
                        )
                        peer_pair1 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p_exchange_tmp[2]), K.cuda.uint_as_float(p_exchange_tmp[3])
                        )
                        K.ptx["add.rn.f32x2"](sum_pair1, p_pair1, peer_pair1)
                        K.ptx.mov.b32(p[exchange_i * 4 + 2], K.cuda.float_as_uint(K.cuda.float2_x(sum_pair1)))
                        K.ptx.mov.b32(p[exchange_i * 4 + 3], K.cuda.float_as_uint(K.cuda.float2_y(sum_pair1)))
                        K.assign(
                            mx[exchange_i],
                            K.max(
                                K.max(K.cuda.float2_x(sum_pair0), K.cuda.float2_y(sum_pair0)),
                                K.max(K.cuda.float2_x(sum_pair1), K.cuda.float2_y(sum_pair1)),
                            ),
                        )
                                                                                    
                                                                                      
                                                                 
                    for width in (4, 2, 1):
                        for i in range(width):
                            K.assign(mx[i], K.max(mx[i], mx[i + width]))
                    cur_pi_max = K.local_scalar("float32", init=mx[0] * scale_log2)
                    K.ptx.st.shared.f32(rowwise_max_buf.ptr_to([sel, idx_in_warpgroup]), cur_pi_max)
                    K.ptx["bar.sync"](pair_bar, K.uint32(64))
                    peer_pi_max = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(peer_pi_max, rowwise_max_buf.ptr_to([sel, K.bitwise_xor(idx_in_warpgroup, 64)]))
                    K.assign(cur_pi_max, K.max(cur_pi_max, peer_pi_max))
                    K.assign(real_mi, K.max(real_mi, cur_pi_max))
                                                                                     
                    m_prev = K.local_scalar("float32", init=K.float32(-1000000000000000019884624838656.0))
                    sm_wm_tok = iket_range("sm-wait-m")
                                                                                      
                                                                                             
                                                                                           
                                                                                    
                    with K.If(c0 + k > 0), K.Then():
                        hot_wait(m_ready.buf[other], K.bitwise_xor(sblk.phase, 1) if sel == 0 else sblk.phase)
                        with K.If(k > 0), K.Then():
                            K.ptx.ld.shared.f32(m_prev, m_buf.ptr_to([other, head_in_half]))
                    K.cuda.iket.range_end(sm_wm_tok[0])
                    should_scale_o = K.local_scalar("uint32")
                    K.ptx.vote_sync.any.pred(should_scale_o, cur_pi_max - m_prev > K.float32(6.0), K.uint32(4294967295))
                    new_max = K.local_scalar("float32")
                    scale_for_old = K.local_scalar("float32")
                    with K.If(should_scale_o == K.uint32(0)):
                        with K.Then():
                            K.assign(scale_for_old, K.float32(1.0))
                            K.assign(new_max, m_prev)
                        with K.Else():
                            K.assign(new_max, K.max(cur_pi_max, m_prev))
                            K.ptx["ex2.approx.ftz.f32"](scale_for_old, m_prev - new_max)
                                                                                            
                    with K.If(idx_in_warpgroup < 64), K.Then():
                        K.ptx.st.shared.f32(m_buf.ptr_to([sel, head_in_half]), new_max)
                    K.ptx["mbarrier.arrive.shared.b64"](
                        K.cuda.cvta_generic_to_shared(K.address_of(m_ready.buf[sel])), K.uint32(1)
                    )
                                                                      
                    li_scale = K.local_scalar("float32")
                    K.ptx["ex2.approx.ftz.f32"](li_scale, m_ref - new_max)
                    K.assign(m_ref, new_max)
                    s_frag = K.alloc_local((32,), "bfloat16")
                    s_pack = s_frag.view("uint32")
                    cur_sum_pair = K.local_scalar("uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0)))
                    neg_new_max_pair = K.local_scalar(
                        "uint64", init=K.cuda.make_float2(new_max * K.float32(-1.0), new_max * K.float32(-1.0))
                    )
                    fma_pair = K.local_scalar("uint64")
                    s_vals = K.alloc_local((2,), "float32")
                    for s_i in range(16):
                        p_pair = K.cuda.make_float2(K.cuda.uint_as_float(p[s_i * 2]), K.cuda.uint_as_float(p[s_i * 2 + 1]))
                        K.ptx["fma.rn.f32x2"](fma_pair, p_pair, scale_pair, neg_new_max_pair)
                                                                                         
                        if s_i % 4 == 3:
                            ex2_emulation_2(s_vals, 0, K.cuda.float2_x(fma_pair), K.cuda.float2_y(fma_pair))
                        else:
                            K.ptx["ex2.approx.ftz.f32"](s_vals[0], K.cuda.float2_x(fma_pair))
                            K.ptx["ex2.approx.ftz.f32"](s_vals[1], K.cuda.float2_y(fma_pair))
                        s_pair = K.cuda.make_float2(s_vals[0], s_vals[1])
                        K.ptx["add.rn.f32x2"](cur_sum_pair, cur_sum_pair, s_pair)
                        K.ptx.mov.b32(s_pack[s_i], K.cuda.float22bfloat162_rn(s_vals[0], s_vals[1]))
                    cur_sum = K.cuda.float2_x(cur_sum_pair) + K.cuda.float2_y(cur_sum_pair)
                    li_tmp = K.local_scalar("float32")
                    K.ptx["fma.rn.f32"](li_tmp, li, li_scale, cur_sum)
                    K.assign(li, li_tmp)
                    K.cuda.iket.range_end(sm_math_tok[0])
                    sm_wpv_tok = iket_range("sm-wait-pvdone")
                                                                                                         
                    hot_wait(softmax_ready.buf[sel], K.bitwise_xor(sblk.phase, 1))
                    K.cuda.iket.range_end(sm_wpv_tok[0])
                    sm_post_tok = iket_range("sm-pstore-rescale")
                    K.ptx["fence.proxy.async.shared::cta"]()
                    s_base: K.int32 = idx_in_warpgroup // 64 * 2048 + idx_in_warpgroup % 64 * 8
                    r_words = s_frag.view("uint32")
                    for f in range(4):
                        s_ptr = K.ptr_byte_offset(
                            K.address_of(s_smem_gemm[sel, 0, 0]), (s_base + f * 512) * BF16_BYTES, "bfloat16"
                        )
                        K.ptx["st.shared.v4.u32"](
                            K.cuda.cvta_generic_to_shared(s_ptr),
                            r_words[f * 4], r_words[f * 4 + 1], r_words[f * 4 + 2], r_words[f * 4 + 3],
                        )
                    with K.If(K.bitwise_and(k > 0, should_scale_o != K.uint32(0))), K.Then():
                                                                                              
                                                                                           
                        hot_wait(softmax_ready.buf[other], K.bitwise_xor(sblk.phase, 1) if sel == 0 else sblk.phase)
                        K.ptx["tcgen05.fence::after_thread_sync"]()
                        o_rescale = K.alloc_local((32,), "float32")
                        with K.unroll(8) as chunk_idx:
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[o_rescale[i] for i in range(32)],
                                K.cuda.get_tmem_addr(K.uint32(TMEM_O), 0, chunk_idx * 32),
                            )
                            K.ptx["tcgen05.wait::ld.sync.aligned"]()
                            for f in range(16):
                                buffer_23 = K.local_scalar("uint64")
                                buffer_24 = K.local_scalar("uint64")
                                K.ptx.mov.b64(buffer_23, o_rescale[f * 2], o_rescale[f * 2 + 1])
                                K.ptx.mov.b64(buffer_24, scale_for_old, scale_for_old)
                                K.ptx["mul.rz.ftz.f32x2"](buffer_23, buffer_23, buffer_24)
                                K.ptx.mov.b64(o_rescale[f * 2], o_rescale[f * 2 + 1], buffer_23)
                            K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                                K.cuda.get_tmem_addr(K.uint32(TMEM_O), 0, chunk_idx * 32),
                                *[o_rescale[i] for i in range(32)],
                            )
                            K.ptx["tcgen05.wait::st.sync.aligned"]()
                        K.ptx["tcgen05.fence::before_thread_sync"]()
                    K.ptx["fence.proxy.async.shared::cta"]()
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](leader_bar(so_full.buf[sel]))
                    K.ptx["mbarrier.arrive.shared.b64"](
                        K.cuda.cvta_generic_to_shared(K.address_of(valid_empty.buf[v_stage])), K.uint32(1)
                    )
                    K.cuda.iket.range_end(sm_post_tok[0])
                    valid_ring.advance()
                    sblk.advance()
                                                                             
                K.cuda.mbarrier_wait(K.address_of(li_empty.buf[0]), K.bitwise_xor(jobs.epoch.phase, 1))
                                                                                         
                K.ptx.st.shared.f32(rowwise_li_buf.ptr_to([sel, idx_in_warpgroup]), li)
                K.ptx.st.shared.f32(rowwise_ref_buf.ptr_to([sel, idx_in_warpgroup]), m_ref)
                K.ptx.st.shared.f32(rowwise_real_buf.ptr_to([sel, idx_in_warpgroup]), real_mi)
                K.ptx["bar.sync"](K.uint32(1), K.uint32(256))
                if sel == 0:
                    with K.If(idx_in_warpgroup < 64), K.Then():
                        last_wg = K.bitwise_and(c0 + wg3_num_k_blocks - 1, 1)                         
                        m_fin = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(m_fin, rowwise_ref_buf.ptr_to([last_wg, idx_in_warpgroup]))
                        li_total = K.local_scalar("float32", init=K.float32(0.0))
                        real_total = K.local_scalar("float32", init=K.float32("-inf"))
                        for w in range(2):
                            for half in range(2):
                                slot = idx_in_warpgroup + 64 * half
                                li_w = K.local_scalar("float32")
                                ref_w = K.local_scalar("float32")
                                real_w = K.local_scalar("float32")
                                K.ptx.ld.shared.f32(li_w, rowwise_li_buf.ptr_to([w, slot]))
                                K.ptx.ld.shared.f32(ref_w, rowwise_ref_buf.ptr_to([w, slot]))
                                K.ptx.ld.shared.f32(real_w, rowwise_real_buf.ptr_to([w, slot]))
                                f_w = K.local_scalar("float32")
                                K.ptx["ex2.approx.ftz.f32"](f_w, ref_w - m_fin)
                                K.assign(li_total, li_total + li_w * f_w)
                                K.assign(real_total, K.max(real_total, real_w))
                        head_idx = cta_idx * 64 + idx_in_warpgroup
                        attn_sink_value = K.local_scalar("float32")
                        K.ptx.ld.global_.f32(attn_sink_value, sinks.ptr_to([head_idx]))
                        attn_sink_log2 = attn_sink_value * K.float32(LOG_2_E)
                        sink_exp = K.local_scalar("float32")
                        K.ptx["ex2.approx.ftz.f32"](sink_exp, attn_sink_log2 - m_fin)
                        output_scale = K.local_scalar("float32", init=K.cuda.fdividef(bmm2_scale, li_total + sink_exp))
                        K.ptx.st.shared.f32(
                            rowwise_scale_buf.ptr_to([idx_in_warpgroup]),
                            K.if_then_else(
                                K.Or(real_total == K.float32("-inf"), li_total == K.float32(0.0)),
                                K.float32(0.0),
                                output_scale,
                            ),
                        )
                        K.ptx["mbarrier.arrive.shared.b64"](
                            K.cuda.cvta_generic_to_shared(K.address_of(li_full.buf[0])), K.uint32(1)
                        )
                K.assign(c0, c0 + wg3_num_k_blocks)
                jobs.advance()
            K.cuda.iket.range_end(softmax_token[0])

        roles = K.specialize(chain_dispatch=True)
        q_output = roles.role("q_load_output", warps=range(0, 4), regs=104)
        kv_load = roles.role("kv_gather", warps=range(4, 8), regs=88)
        producer = roles.warpgroup("qk_control", warps=range(8, 12), regs=48)
        qk = roles.role("qk_issue", warps=[8], when=cta_idx == 0, group=producer)
        valid = roles.role("valid_mask", warps=[9], group=producer)
        pv = roles.role("pv_issue", warps=[10], when=cta_idx == 0, group=producer)
        idle = roles.role("idle", warps=[11], group=producer)
        scale_a = roles.role("softmax_a", warps=range(12, 16), regs=120)
        scale_b = roles.role("softmax_b", warps=range(16, 20), regs=120)
        with q_output:
            q_load_output()
        with kv_load:
            kv_gather()
        with producer:
            with qk:
                qk_issuer()
            with valid:
                clc_role(False)
            with pv:
                pv_issuer()
            with idle:
                clc_role(False)
        with scale_a:
            softmax(0)
        with scale_b:
            softmax(1)

        K.cuda.cluster_sync()

    return (
        mla_dsv4_sparse_prefill_pkt_pingpong.func.with_attr("global_symbol", KERNEL_NAME)
        .with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
    )


# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_mla_dsv4_prefill_b2",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "mla-dsv4-prefill-b2",
        "selected_version": "frontier/dual-issuer",
    },
}

CONFIGS = [
    {
        "label": "b2_qsum386_h128_swa16384_c16384_k1152",
        "num_heads": 128,
        "num_seqs": 2,
        "max_q_len": 257,
        "head_dim": 512,
        "swa_page_size": 256,
        "swa_seq_len": 16384,
        "compressed_page_size": 64,
        "compressed_seq_len": 16384,
        "compressed_topk": 1024,
        "seed": 2026,
    }
]

SWA_TOPK = 128
QUERY_RANDOM_SCALE = 0.05
KV_RANDOM_SCALE = 0.05
SWA_KV_OFFSET = -0.20
COMPRESSED_KV_OFFSET = 0.25
SINK_STD = 0.05
BMM1_SCALE = 512**-0.55
BMM2_SCALE = 1.0


def _config(**config: Any) -> dict[str, Any]:
    """Validate one config against the contract this kernel implements."""
    values = {key: value for key, value in config.items() if key != "label"}
    unknown = set(values) - set(CONFIGS[0]) - {"label"}
    if unknown:
        raise ValueError(f"unsupported config keys: {sorted(unknown)}")
    resolved = {**CONFIGS[0], **values}
    resolved.pop("label", None)
    if int(resolved["num_heads"]) != B_H:
        raise ValueError(f"num_heads must be {B_H}")
    if int(resolved["head_dim"]) != D_QK:
        raise ValueError(f"head_dim must be {D_QK}")
    topk = SWA_TOPK + int(resolved["compressed_topk"])
    if topk % B_TOPK != 0:
        raise ValueError(f"total top-k must be a multiple of {B_TOPK}")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved DSv4 sparse MLA prefill")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved DSv4 sparse MLA prefill requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def _shape_key(resolved: dict[str, Any]) -> tuple[int, int, int, int]:
    """(sum_q, topk, swa_rows, comp_rows) -- the kernel's compile key."""
    num_seqs = int(resolved["num_seqs"])
    max_q_len = int(resolved["max_q_len"])
    min_len = (max_q_len + 1) // 2
    q_lens = [
        round(min_len + (max_q_len - min_len) * i / max(num_seqs - 1, 1))
        for i in range(num_seqs)
    ]
    sum_q = sum(q_lens)
    topk = SWA_TOPK + int(resolved["compressed_topk"])
    swa_page = int(resolved["swa_page_size"])
    comp_page = int(resolved["compressed_page_size"])
    swa_lens = [int(resolved["swa_seq_len"]) + swa_page * i for i in range(num_seqs)]
    swa_pages_per_seq = (max(swa_lens) + swa_page - 1) // swa_page
    swa_rows = num_seqs * swa_pages_per_seq * swa_page
    comp_base = max(int(resolved["compressed_seq_len"]), int(resolved["compressed_topk"]), max_q_len)
    comp_lens = [comp_base + comp_page * i for i in range(num_seqs)]
    comp_pages_per_seq = (max(comp_lens) + comp_page - 1) // comp_page
    comp_rows = num_seqs * comp_pages_per_seq * comp_page
    return sum_q, topk, swa_rows, comp_rows


def get_kernel(**config: Any):
    """Return the traced Kern PrimFunc for one compile key."""
    resolved = _config(**config)
    sum_q, topk, swa_rows, comp_rows = _shape_key(resolved)
    return make_kernel(sum_q, topk, swa_rows, comp_rows)


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged DSv4 sparse-MLA benchmark row: one
# generator sequence produces the query, the sinks, the two pools and their
# page permutations, then the per-token index rows. Query/KV/sink amplitudes
# are 0.05; the SWA and compressed pools add offsets -0.20 and +0.25 and clamp
# to [-1, 1]. The published builder decreases active sparse lengths by one per
# query within each request.
# ---------------------------------------------------------------------------


def _flat_indices(logical, block_table, page_size):
    """Map per-request logical token positions to flat pool rows (-1 kept)."""
    valid = logical >= 0
    safe = logical.clamp_min(0)
    physical = block_table[safe // page_size] * page_size + safe % page_size
    return torch.where(valid, physical, torch.full_like(physical, -1)).to(torch.int32)


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract inputs plus the preallocated output."""
    resolved = _config(**config)
    device = torch.device("cuda")
    num_heads = int(resolved["num_heads"])
    num_seqs = int(resolved["num_seqs"])
    max_q_len = int(resolved["max_q_len"])
    head_dim = int(resolved["head_dim"])
    swa_page = int(resolved["swa_page_size"])
    comp_page = int(resolved["compressed_page_size"])
    comp_topk = int(resolved["compressed_topk"])
    generator = torch.Generator(device=device).manual_seed(int(resolved["seed"]))

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    def randperm(n):
        return torch.randperm(n, device=device, generator=generator)

    def pool(seq_len_base, page_size, value_offset):
        seq_lens = seq_len_base + page_size * torch.arange(num_seqs, device=device)
        pages_per_seq = (int(seq_lens.max()) + page_size - 1) // page_size
        block_table = randperm(num_seqs * pages_per_seq).view(num_seqs, pages_per_seq)
        cache = (
            randn((num_seqs * pages_per_seq, 1, page_size, head_dim)) * KV_RANDOM_SCALE
        ).add_(value_offset).clamp_(-1.0, 1.0)
        return seq_lens.to(torch.int32), block_table, cache

    min_len = (max_q_len + 1) // 2
    q_lens = [
        round(min_len + (max_q_len - min_len) * i / max(num_seqs - 1, 1))
        for i in range(num_seqs)
    ]
    cum_seq_lens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    sum_q = sum(q_lens)
    query = randn((sum_q, num_heads, head_dim)) * QUERY_RANDOM_SCALE
    sinks = randn((num_heads,), torch.float32) * SINK_STD

    seq_lens, swa_table, swa_kv_cache = pool(
        int(resolved["swa_seq_len"]), swa_page, SWA_KV_OFFSET
    )
    swa_columns = torch.arange(SWA_TOPK, device=device)
    comp_base = max(int(resolved["compressed_seq_len"]), comp_topk, max_q_len)
    comp_seq_lens, comp_table, compressed_kv_cache = pool(
        comp_base, comp_page, COMPRESSED_KV_OFFSET
    )

    index_rows = []
    lens = []
    for b in range(num_seqs):
        for q_idx in range(q_lens[b]):
            token = int(seq_lens[b]) - q_lens[b] + q_idx
            num_valid = min(SWA_TOPK, token + 1)
            logical = torch.full_like(swa_columns, -1)
            logical[:num_valid] = torch.arange(
                token - num_valid + 1, token + 1, device=device
            )
            row = [_flat_indices(logical, swa_table[b], swa_page)]
            active = SWA_TOPK
            c_len = int(comp_seq_lens[b])
            sample = randperm(c_len)[:comp_topk]
            logical = torch.full((comp_topk,), -1, device=device, dtype=torch.int64)
            logical[: sample.numel()] = sample
            row.append(_flat_indices(logical, comp_table[b], comp_page))
            active += min(max(comp_topk - (comp_topk // 16) * b - q_idx, 1), comp_topk)
            index_rows.append(torch.cat(row))
            lens.append(active)
    sparse_indices = torch.stack(index_rows).contiguous()
    sparse_topk_lens = torch.tensor(lens, dtype=torch.int32, device=device)

    return {
        "config": resolved,
        "query": query,
        "swa_kv_cache": swa_kv_cache,
        "compressed_kv_cache": compressed_kv_cache,
        "sparse_indices": sparse_indices,
        "sparse_topk_lens": sparse_topk_lens,
        "seq_lens": seq_lens,
        "cum_seq_lens_q": cum_seq_lens_q,
        "max_q_len": max_q_len,
        "sinks": sinks,
        "bmm1_scale": float(BMM1_SCALE),
        "bmm2_scale": float(BMM2_SCALE),
        "kv_layout": "HND",
        "output": torch.empty(query.shape, dtype=torch.bfloat16, device=device),
    }


def _pool_rows(pool, name):
    if pool is None or pool.dim() != 4 or pool.shape[-1] != D_QK:
        raise ValueError(f"{name} must be a 4-D [pages, ., ., 512] latent pool")
    if not pool.is_contiguous() or pool.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a contiguous bf16 pool")
    return pool.shape[0] * pool.shape[1] * pool.shape[2]


def _tirx_args(case: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Bind one case to the kernel's argument list (the candidate's ``setup``)."""
    query = case["query"]
    swa = case["swa_kv_cache"]
    comp = case["compressed_kv_cache"]
    indices = case["sparse_indices"]
    lens = case["sparse_topk_lens"]
    sinks = case["sinks"]
    out = case["output"]
    sum_q, topk = int(indices.shape[0]), int(indices.shape[1])
    if tuple(query.shape) != (sum_q, B_H, D_QK) or query.dtype != torch.bfloat16:
        raise ValueError("query must be bf16 [sum_q, 128, 512]")
    if indices.dtype != torch.int32:
        raise ValueError("sparse_indices must be int32 [sum_q, K]")
    if tuple(out.shape) != (sum_q, B_H, D_V) or out.dtype != torch.bfloat16:
        raise ValueError("output must be bf16 [sum_q, 128, 512]")
    for name, tensor in (("query", query), ("sparse_indices", indices), ("output", out)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    _pool_rows(swa, "swa_kv_cache")
    _pool_rows(comp, "compressed_kv_cache")
    scale_log2 = float(case["bmm1_scale"]) * LOG_2_E
    bmm2_scale = float(case["bmm2_scale"])
    swa_flat, comp_flat = swa.view(-1), comp.view(-1)
    lens_c = lens.contiguous()
    sinks_f = sinks.to(torch.float32).contiguous()
    args = (
        query, swa_flat, comp_flat, indices.view(-1), lens_c, sinks_f,
        out, scale_log2, bmm2_scale,
    )
    keep = (query, swa, comp, swa_flat, comp_flat, indices, lens_c, sinks_f, out)
    return args, keep


# ---------------------------------------------------------------------------
# Independent oracle and correctness.
#
# This is the packaged task's reference: fp32 sparse MLA with the sink logit,
# gathering both pools by flat row index. It shares no code with the kernel.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any]) -> torch.Tensor:
    query = case["query"]
    indices = case["sparse_indices"].to(torch.int64)
    topk = indices.shape[1]
    head_dim = query.shape[2]
    columns = torch.arange(topk, device=query.device)
    active = (indices >= 0) & (
        columns[None, :] < case["sparse_topk_lens"].to(torch.int64)[:, None]
    )

    swa_rows = case["swa_kv_cache"].reshape(-1, head_dim).float()
    rows = swa_rows[indices[:, :SWA_TOPK].clamp_min(0)]
    if topk > SWA_TOPK:
        compressed_rows = case["compressed_kv_cache"].reshape(-1, head_dim).float()
        rows = torch.cat(
            [rows, compressed_rows[indices[:, SWA_TOPK:].clamp_min(0)]], dim=1
        )

    scores = torch.einsum("thd,tkd->thk", query.float(), rows) * float(case["bmm1_scale"])
    scores = scores.masked_fill(~active[:, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse[..., None])
    probs = torch.where(active[:, None, :], probs, torch.zeros_like(probs))
    sinks = case["sinks"]
    if sinks is not None:
        probs = probs / (1.0 + torch.exp(sinks.float()[None, :] - lse))[..., None]
    out = torch.einsum("thk,tkd->thd", probs, rows) * float(case["bmm2_scale"])
    return out.to(torch.bfloat16)


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    """Gate the kernel output against the oracle with the row's bf16 gate."""
    case = outputs["case"]
    actual = outputs["output"]
    expected = _reference_output(case)
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=8e-4, rtol=2e-2
    )


def run_test(**config: Any) -> None:
    """Compile, run once and gate against the oracle."""
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**config)
    executable = compile_kernel(get_kernel(**config))
    args, keep = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()
    check_correctness({"case": case, "output": case["output"]}, **config)
    del keep


# ---------------------------------------------------------------------------
# FlashInfer trtllm-gen reference arm.
#
# The packaged baseline for this row is
# `flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4` with PDL disabled,
# which is how PR #4573 reports its GPU-active sums (PDL overlap between the
# wrapper's launches would undercount).
# ---------------------------------------------------------------------------

_WORKSPACE: dict[str, Any] = {}


def _workspace(device):
    key = str(device)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return _WORKSPACE[key]


def _trtllm_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    query = case["query"]
    return dict(
        query=query,
        swa_kv_cache=case["swa_kv_cache"],
        workspace_buffer=_workspace(query.device),
        sparse_indices=case["sparse_indices"],
        compressed_kv_cache=case["compressed_kv_cache"],
        sparse_topk_lens=case["sparse_topk_lens"],
        seq_lens=case["seq_lens"],
        bmm1_scale=float(case["bmm1_scale"]),
        bmm2_scale=float(case["bmm2_scale"]),
        sinks=case["sinks"],
        kv_layout=str(case["kv_layout"]),
        cum_seq_lens_q=case["cum_seq_lens_q"],
        max_q_len=int(case["max_q_len"]),
        enable_pdl=False,
    )


def _trtllm_reference(kwargs: dict[str, Any]):
    from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

    return trtllm_batch_decode_sparse_mla_dsv4(**kwargs)


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------


def prepare_bench(**config: Any):
    """Trace and compile before bench-suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_supported_arch()
    from tirx_kernels.runner import bench

    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    executable = prepared["executable"]
    args, keep = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()

    def _trtllm_builder():
        # The workspace allocation and the JIT/cubin fetch are prepare work.
        reference_kwargs = _trtllm_kwargs(case)
        _trtllm_reference(reference_kwargs)
        return lambda: _trtllm_reference(reference_kwargs)

    results = bench(
        {"tirx": lambda: executable(*args)},
        references={"flashinfer_trtllm_dsv4": _trtllm_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )
    del keep
    return results


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **config: Any,
) -> dict[str, Any]:
    values = dict(config)
    protocol = {name: values.pop(name) for name in ("rounds", "cooldown_s") if name in values}
    prepared = prepare_bench(**values)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "check_correctness",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
