# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
from tirx_kernels.flashmla.utils._mask import pack_valid_mask8
from tirx_kernels.flashmla.utils._tma import leader_mbar

B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

LAUNCH_TAGS = ("blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

BAR_WG0_SYNC = 0

BF16_BYTES = 2
D_TQ = 384
P_TMEM_COLS = B_TOPK // 2
B_EPI = 64
WG1_NUM_WARPS = 4
WG1_ROWS_PER_WARP = (B_TOPK // 2) // 4 // WG1_NUM_WARPS
WG2_NUM_WARPS = 4
WG2_ROWS_PER_PART = (B_TOPK // 2) // 4 // WG2_NUM_WARPS

_TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_TMEM_ST_32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
_TMA_G2S_4D_CACHE = (
    "cp.async.bulk.tensor.4d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)
_TMA_GATHER4_2D_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)
_TMA_S2G_3D = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
_TCGEN_CP_64X128 = "tcgen05.cp.cta_group::2.64x128b.warpx2::02_13"
_TCGEN_COMMIT = (
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
)
_MMA_F16 = "tcgen05.mma.cta_group::2.kind::f16"
_Q_TMA_CACHE_HINT = K.uint64(0x12F0000000000000)
_KV_TMA_CACHE_HINT = K.uint64(0x14F0000000000000)


def _tmem_load(dst, tmem_col, width):
    chain = _TMEM_LD_32 if width == 32 else _TMEM_LD_64
    return K.ptx[chain](*[dst[i] for i in range(width)], tmem_col)


def _tmem_store(src, tmem_col, width=32):
    assert width == 32
    return K.ptx[_TMEM_ST_32](tmem_col, *[src[i] for i in range(width)])


def _replace_smem_desc_addr(desc, smem_ptr):
    start_addr = K.cast(
        K.bitwise_and(
            K.shift_right(K.cuda.cvta_generic_to_shared(smem_ptr), K.uint32(4)), K.uint32(0x3FFF)
        ),
        "uint64",
    )
    return K.bitwise_or(K.bitwise_and(desc, K.bitwise_not(K.uint64(0x3FFF))), start_addr)


def _recompute_smem_desc(smem_ptr, upper, matrix_start):
    start_addr = K.bitwise_and(
        K.shift_right(K.cuda.cvta_generic_to_shared(smem_ptr), K.uint32(4)), K.uint32(0x3FFF)
    )
    return K.bitwise_or(
        K.shift_left(K.uint64(upper), K.uint64(32)),
        K.cast(K.bitwise_or(K.uint32(matrix_start), start_addr), "uint64"),
    )


def _add_smem_desc_offset(desc, offset):
    """Step a descriptor by an offset that wraps in its low 32 bits.

    The same wrap as a mov.b64 unpack / add.u32 / mov.b64 repack, spelled as
    arithmetic instead, which leaves the descriptor dataflow visible to ptxas
    rather than behind an inline-asm round trip.

    The trade here is NOT the one the same change made on sparse decode, where
    it removed R2UR by unblocking uniform-datapath promotion.  On this kernel's
    two "hoist" specializations R2UR goes up (+17, +19) and UMOV and MOV come
    down further (-17, -27), which nets out to roughly 0.2-0.5% on wall time.
    R2UR count does not predict this transformation; only measurement does.
    """
    low = K.cast(K.cast(desc, "uint32") + K.cast(offset, "uint32"), "uint64")
    return K.bitwise_or(K.bitwise_and(desc, K.bitwise_not(K.uint64(0xFFFFFFFF))), low)


def _mma_f16(d_tmem, a_operand, b_desc, idesc, enable_input_d):
    return K.ptx[_MMA_F16](
        d_tmem,
        a_operand,
        b_desc,
        idesc,
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        enable_input_d,
    )


def mul_f32x2(values, idx, multiplier):
    packed = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, values[idx], values[idx + 1])
    K.ptx.mov.b64(rhs, multiplier, multiplier)
    K.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
    K.ptx.mov.b64(values[idx], values[idx + 1], packed)


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 regular phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 128")


CONFIGS = [
    {
        "label": f"bench_regular_dqk{d_qk}_hq128_s4096_kv{s_kv}_topk2048",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 2048,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_phase1",
    "category": "flashmla",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_regular_dispatch_reason(cfg: SparseFlashMLAPrefillHead128Config) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.d_qk == 512 and cfg.topk <= 1280:
        return "out_of_scope: sm100 head128 D_QK=512 topk<=1280 dispatches small-topk"
    return f"regular: sm100 head128 run_fwd_phase1_kernel<{cfg.d_qk}>"


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.s_q, cfg.h_q, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    kv = torch.randn(
        (cfg.s_kv, cfg.h_kv, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    out = torch.empty((cfg.s_q, cfg.h_q, cfg.d_v), device=device, dtype=torch.bfloat16)
    max_logits = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)
    lse = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)

    indices = torch.randint(
        low=0,
        high=cfg.s_kv,
        size=(cfg.s_q, cfg.h_kv, cfg.topk),
        device=device,
        dtype=torch.int32,
        generator=gen,
    )
    if cfg.inject_invalid_indices:
        indices[:, :, 0] = -1
        indices[:, :, 1] = cfg.s_kv
        indices[:, :, 2] = cfg.s_kv + 17
        indices[:, :, -1] = -7
    attn_sink = (
        torch.randn((cfg.h_q,), device=device, dtype=torch.float32, generator=gen)
        if cfg.have_attn_sink
        else torch.empty((cfg.h_q,), device=device, dtype=torch.float32)
    )
    if cfg.have_topk_length:
        topk_length = torch.randint(
            low=0,
            high=cfg.topk + 1,
            size=(cfg.s_q,),
            device=device,
            dtype=torch.int32,
            generator=gen,
        )
    else:
        topk_length = torch.empty((cfg.s_q,), device=device, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(cfg.d_qk)
    return {
        "config": cfg,
        "q": q,
        "kv": kv,
        "indices": indices,
        "attn_sink": attn_sink,
        "topk_length": topk_length,
        "out": out,
        "max_logits": max_logits,
        "lse": lse,
        "sm_scale": sm_scale,
        "sm_scale_div_log2": sm_scale * LOG_2_E,
        "dispatch_reason": _flashmla_regular_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    q = case["q"].float()
    kv = case["kv"][:, 0, :].float()
    indices = case["indices"][:, 0, :].to(torch.long)
    sm_scale = case["sm_scale"]
    ref_out = torch.zeros((cfg.s_q, cfg.h_q, cfg.d_v), device=q.device, dtype=torch.float32)
    ref_max_logits = torch.full((cfg.s_q, cfg.h_q), -float("inf"), device=q.device)
    ref_lse = torch.full((cfg.s_q, cfg.h_q), float("inf"), device=q.device)

    for s_q_idx in range(cfg.s_q):
        length = int(case["topk_length"][s_q_idx].item()) if cfg.have_topk_length else cfg.topk
        row_indices = indices[s_q_idx]
        pos = torch.arange(cfg.topk, device=q.device)
        valid = (pos < length) & (row_indices >= 0) & (row_indices < cfg.s_kv)
        if not torch.any(valid):
            continue
        selected = row_indices.clamp(0, cfg.s_kv - 1)
        k_full = kv[selected]
        logits = torch.matmul(q[s_q_idx], k_full[:, : cfg.d_qk].T) * sm_scale
        logits[:, ~valid] = -float("inf")
        max_logits = torch.max(logits, dim=-1).values
        exp_logits = torch.exp(logits - max_logits[:, None])
        exp_logits[:, ~valid] = 0.0
        denom = torch.sum(exp_logits, dim=-1)
        if cfg.have_attn_sink:
            sink = case["attn_sink"].float()
            denom_with_sink = denom + torch.exp(sink - max_logits)
        else:
            denom_with_sink = denom
        ref_out[s_q_idx] = torch.matmul(exp_logits, k_full[:, : cfg.d_v]) / denom_with_sink[:, None]
        ref_max_logits[s_q_idx] = max_logits
        ref_lse[s_q_idx] = max_logits + torch.log(denom)
    return ref_out.to(torch.bfloat16), ref_max_logits, ref_lse


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["q"],
        case["kv"].reshape(-1),
        case["indices"].reshape(-1),
        case["attn_sink"],
        case["topk_length"],
        case["out"],
        case["max_logits"],
        case["lse"],
    )


def make_kernel(
    s_q,
    s_kv,
    topk,
    d_qk,
    h_q,
    stride_kv_s_kv,
    stride_indices_s_q,
    have_attn_sink,
    have_topk_length,
    sm_scale_div_log2,
):
    d_sq = d_qk - D_TQ
    num_sq_tiles = (d_qk - D_TQ) // 64
    num_qk_tiles = d_qk // 64
    mma_smem_desc = (
        "recompute"
        if (d_qk == 512 and s_kv == 8192)
        else "local_hoist"
        if (d_qk == 576 and s_kv != 65536)
        else "hoist"
        if ((d_qk == 512 and s_kv == 32768) or (d_qk == 576 and s_kv == 65536))
        else "encode"
    )

    # This kernel pipelines barrier generations while reusing one physical K,
    # V, and S tile.  Derive a barrier coordinate from the algorithmic tile at
    # each ownership handoff: a moving cursor would obscure the intentional
    # k/current-QK versus k-1/PV overlap and extend its lifetime across the MMA
    # body.  Keep stage and phase as separate primitives because producer roles
    # only own the stage selected for their TMA completion barrier.
    def ring_stage(tile):
        return tile % NUM_BUFS

    def ring_phase(tile):
        return (tile // NUM_BUFS) & 1

    def host_prelude(params):
        q = params["q"]
        kv = params["kv"]
        out = params["out"]

        def encode(data, rank, *shape):
            descriptor = K.stack_alloca("tensormap", 1)
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled", descriptor, "bfloat16", rank, data, *shape
            )
            return descriptor

        kv_v_part1 = encode(
            kv.data, 2, d_qk, s_kv, stride_kv_s_kv * BF16_BYTES, 64, 1, 1, 1, 0, 3, 3, 0
        )
        kv_v_part0 = encode(
            kv.data, 2, d_qk, s_kv, stride_kv_s_kv * BF16_BYTES, 64, 1, 1, 1, 0, 3, 3, 0
        )
        kv_k_part1 = encode(
            kv.data, 2, d_qk, s_kv, stride_kv_s_kv * BF16_BYTES, 64, 1, 1, 1, 0, 3, 3, 0
        )
        kv_k_part0 = encode(
            kv.data, 2, d_qk, s_kv, stride_kv_s_kv * BF16_BYTES, 64, 1, 1, 1, 0, 3, 3, 0
        )
        out_part1 = encode(
            out.data,
            3,
            D_V,
            h_q,
            s_q,
            D_V * BF16_BYTES,
            h_q * D_V * BF16_BYTES,
            B_EPI,
            B_H // 2,
            1,
            1,
            1,
            1,
            0,
            3,
            3,
            0,
        )
        out_part0 = encode(
            out.data,
            3,
            D_V,
            h_q,
            s_q,
            D_V * BF16_BYTES,
            h_q * D_V * BF16_BYTES,
            B_EPI,
            B_H // 2,
            1,
            1,
            1,
            1,
            0,
            3,
            3,
            0,
        )
        q_tensormap = encode(
            q.data,
            4,
            64,
            h_q,
            d_qk // 64,
            s_q,
            d_qk * BF16_BYTES,
            64 * BF16_BYTES,
            h_q * d_qk * BF16_BYTES,
            64,
            B_H // 2,
            d_qk // 64,
            1,
            1,
            1,
            1,
            1,
            0,
            3,
            3,
            0,
        )
        return (kv_v_part1, kv_v_part0, kv_k_part1, kv_k_part0, out_part1, out_part0, q_tensormap)

    def sparse_flashmla_prefill_head128_phase1_kern(
        q, kv, indices, attn_sink, topk_length, out, max_logits, lse, *, host
    ):
        (
            kv_v_part1_tensormap,
            kv_v_part0_tensormap,
            kv_k_part1_tensormap,
            kv_k_part0_tensormap,
            out_part1_tensormap,
            out_part0_tensormap,
            q_tensormap,
        ) = host
        block_idx = K.cta_id()
        K.cta_id_in_cluster([2], preferred=[2])
        cta_idx = block_idx % 2
        s_q_idx = block_idx // 2
        warp_idx = K.warp_id()
        lane_idx = K.lane_id()
        idx_in_warpgroup = K.thread_id_in_wg([128])

        def iket_range(name):
            token = K.alloc_local((1,), "uint32")
            K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        if have_topk_length:
            topk_len = K.local_scalar("int32")
            K.ptx.ld.global_.nc.s32(topk_len, topk_length.ptr_to([s_q_idx]))
        else:
            topk_len = K.int32(topk)
        num_k_blocks = K.max((topk_len + B_TOPK - 1) // B_TOPK, 1)

        def prefetch(tensor_map):
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(tensor_map))

        prefetch(q_tensormap)
        prefetch(out_part0_tensormap)
        prefetch(out_part1_tensormap)
        prefetch(kv_k_part0_tensormap)
        prefetch(kv_k_part1_tensormap)
        prefetch(kv_v_part0_tensormap)
        prefetch(kv_v_part1_tensormap)

        smem = K.smem_pool()
        pool = smem.pool
        u_base = pool.offset
        q_full = smem.alloc((B_H // 2, d_qk), "bfloat16", swizzle=K.SW128B).buf
        q_cp_desc = K.local_scalar("uint64")
        K.cuda.tcgen05.encode_matrix_descriptor(
            K.address_of(q_cp_desc), K.reinterpret(K.handle().ty, K.uint64(0)), 0, 64, 3
        )
        pool.move_base_to(u_base + (B_H // 2) * d_sq * BF16_BYTES)
        v_smem = smem.alloc((D_V // 2, B_TOPK), "bfloat16", swizzle=K.SW128B).buf
        k_smem = smem.alloc((B_TOPK // 2, d_qk), "bfloat16", swizzle=K.SW128B).buf
        u_end = pool.offset
        pool.move_base_to(u_base)
        o_smem = smem.alloc((B_H // 2, D_V), "bfloat16", swizzle=K.SW128B).buf
        pool.move_base_to(u_end)
        s_smem_gemm = smem.alloc((B_H // 2, B_TOPK), "bfloat16", align=1024)
        is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
        bar_prologue_q = K.TMABar(pool, 1)
        bar_prologue_utccp = K.TCGen05Bar(pool, 1)
        bar_qk_part_done = K.TCGen05Bar(pool, NUM_BUFS)
        bar_qk_done = K.TCGen05Bar(pool, NUM_BUFS)
        bar_sv_part_done = K.TCGen05Bar(pool, NUM_BUFS)
        bar_sv_done = K.TCGen05Bar(pool, NUM_BUFS)
        bar_k_part0_ready = K.TMABar(pool, NUM_BUFS)
        bar_k_part1_ready = K.TMABar(pool, NUM_BUFS)
        bar_v_part0_ready = K.TMABar(pool, NUM_BUFS)
        bar_v_part1_ready = K.TMABar(pool, NUM_BUFS)
        bar_p_free = K.MBarrier(pool, NUM_BUFS)
        bar_so_ready = K.MBarrier(pool, NUM_BUFS)
        bar_k_valid_ready = K.MBarrier(pool, NUM_BUFS)
        bar_k_valid_free = K.MBarrier(pool, NUM_BUFS)
        tmem_start_addr = pool.alloc((1,), "uint32", align=4)
        rowwise_max_buf = pool.alloc((128,), "float32")
        rowwise_li_buf = pool.alloc((128,), "float32")
        g_indices_base = s_q_idx * stride_indices_s_q
        mma_p_accumulate = K.local_scalar("uint32")
        mma_o_accumulate = K.local_scalar("uint32")
        K.assign(mma_p_accumulate, K.uint32(0))
        K.assign(mma_o_accumulate, K.uint32(0))

        def initialize_and_load_q():
            # CUDA phase1.cuh:87-146.  Warp 0 owns barrier init, Q TMA launch,
            # and the cta_group::2 TMEM allocation.
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    bar_prologue_q.init(1)
                    bar_prologue_utccp.init(1)
                    with K.unroll(NUM_BUFS) as init_stage:
                        K.ptx.mbarrier.init.shared.b64(
                            bar_qk_part_done.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_qk_done.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_sv_part_done.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_sv_done.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_k_part0_ready.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_k_part1_ready.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_v_part0_ready.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_v_part1_ready.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_p_free.ptr_to([init_stage]), K.uint32(128 * 2)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_so_ready.ptr_to([init_stage]), K.uint32(128 * 2)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_k_valid_ready.ptr_to([init_stage]), K.uint32(16)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_k_valid_free.ptr_to([init_stage]), K.uint32(128)
                        )
                    K.ptx.fence.mbarrier_init.release.cluster()

            K.cuda.cluster_sync()

            with K.If(warp_idx == 0), K.Then():
                prologue_token = iket_range("h128-q-load")
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx[_TMA_G2S_4D_CACHE](
                        q_full.ptr_to([0, 0]),
                        K.address_of(q_tensormap),
                        K.int32(0),
                        K.cast(cta_idx * (B_H // 2), "int32"),
                        K.int32(0),
                        K.cast(s_q_idx, "int32"),
                        K.cuda.cvta_generic_to_shared(leader_mbar(bar_prologue_q.ptr_to([0]))),
                        _Q_TMA_CACHE_HINT,
                    )

                K.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                    K.address_of(tmem_start_addr[0]), K.uint32(512)
                )
                allocated_tmem_start = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(allocated_tmem_start, tmem_start_addr.ptr_to([0]))
                K.cuda.trap_when_assert_failed(allocated_tmem_start == K.uint32(0))
                K.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
                K.cuda.iket.range_end(prologue_token[0])

            K.cuda.cta_sync()

        initialize_and_load_q()

        o_tmem_col = 0
        tmem_p_col = 256
        q_tmem_col = 320

        if mma_smem_desc == "hoist":
            qk_k_part0_desc = K.SmemDescriptor()
            qk_k_part0_desc.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)
            qk_k_part1_desc = K.SmemDescriptor()
            qk_k_part1_desc.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)
            pv_a_part0_lo_desc = K.SmemDescriptor()
            pv_a_part0_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
            pv_a_part0_hi_desc = K.SmemDescriptor()
            pv_a_part0_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
            pv_a_part1_lo_desc = K.SmemDescriptor()
            pv_a_part1_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
            pv_a_part1_hi_desc = K.SmemDescriptor()
            pv_a_part1_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
            pv_b_part0_lo_desc = K.SmemDescriptor()
            pv_b_part0_lo_desc.init(v_smem.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
            pv_b_part0_hi_desc = K.SmemDescriptor()
            pv_b_part0_hi_desc.init(v_smem.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
            pv_b_part1_lo_desc = K.SmemDescriptor()
            pv_b_part1_lo_desc.init(v_smem.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
            pv_b_part1_hi_desc = K.SmemDescriptor()
            pv_b_part1_hi_desc.init(v_smem.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)

        def issue_pv_mma(dest_offset, s_offset, v_offset, hoisted_a, hoisted_b):
            if mma_smem_desc == "local_hoist":
                pv_a_local = K.SmemDescriptor()
                pv_a_local.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
                pv_b_local = K.SmemDescriptor()
                pv_b_local.init(v_smem.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
            with K.unroll(1) as mma_mi:
                with K.unroll(1) as mma_ni:
                    with K.unroll(4) as mma_ki:
                        pv_a_offset = mma_ki % 4 * 1024 + mma_mi * 512 + mma_ki // 4 * 8 + s_offset
                        pv_b_offset = mma_ki * 1024 + mma_ni * 64 + v_offset
                        if mma_smem_desc == "recompute":
                            pv_a_ptr = K.ptr_byte_offset(
                                s_smem_gemm.ptr_to([0, 0]), pv_a_offset // 8 * 16, "bfloat16"
                            )
                            pv_b_ptr = K.ptr_byte_offset(
                                v_smem.ptr_to([0, 0]), pv_b_offset // 8 * 16, "bfloat16"
                            )
                            _mma_f16(
                                K.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                _recompute_smem_desc(pv_a_ptr, 0x00004008, 0x00400000),
                                _recompute_smem_desc(pv_b_ptr, 0x40004040, 0x04000000),
                                K.uint32(0x08410490),
                                K.Or(mma_ki != 0, K.cast(mma_o_accumulate, "bool")),
                            )
                        elif mma_smem_desc == "encode":
                            pv_a_encode = K.SmemDescriptor()
                            pv_a_encode.init(
                                K.ptr_byte_offset(
                                    s_smem_gemm.ptr_to([0, 0]), pv_a_offset // 8 * 16, "bfloat16"
                                ),
                                ldo=64,
                                sdo=8,
                                swizzle=0,
                            )
                            pv_b_encode = K.SmemDescriptor()
                            pv_b_encode.init(
                                K.ptr_byte_offset(
                                    v_smem.ptr_to([0, 0]), pv_b_offset // 8 * 16, "bfloat16"
                                ),
                                ldo=1024,
                                sdo=64,
                                swizzle=3,
                            )
                            _mma_f16(
                                K.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                pv_a_encode.desc,
                                pv_b_encode.desc,
                                K.uint32(0x08410490),
                                K.Or(mma_ki != 0, K.cast(mma_o_accumulate, "bool")),
                            )
                        elif mma_smem_desc == "local_hoist":
                            _mma_f16(
                                K.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                pv_a_local.add_16B_offset(pv_a_offset // 8),
                                pv_b_local.add_16B_offset(pv_b_offset // 8),
                                K.uint32(0x08410490),
                                K.Or(mma_ki != 0, K.cast(mma_o_accumulate, "bool")),
                            )
                        else:
                            _mma_f16(
                                K.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                _add_smem_desc_offset(hoisted_a, pv_a_offset // 8),
                                _add_smem_desc_offset(hoisted_b, pv_b_offset // 8),
                                K.uint32(0x08410490),
                                K.Or(mma_ki != 0, K.cast(mma_o_accumulate, "bool")),
                            )

        def softmax_and_epilogue():
            # CUDA phase1.cuh:150-386.  Scale/exp warpgroup and epilogue.
            mi = K.local_scalar("float32", init=MAX_INIT_VAL)
            li = K.local_scalar("float32", init=0.0)
            real_mi = K.local_scalar("float32", init=K.float32(-float("inf")))
            scale_pair = K.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

            with K.serial(0, num_k_blocks, unroll=False) as k:
                softmax_token = iket_range("h128-softmax-tile")
                softmax_stage = ring_stage(k)
                softmax_phase = ring_phase(k)
                qk_wait_token = iket_range("h128-qk-wait")
                bar_qk_done.wait(softmax_stage, softmax_phase)
                K.cuda.iket.range_end(qk_wait_token[0])
                K.ptx.tcgen05.fence__after_thread_sync()

                p = K.alloc_local((P_TMEM_COLS,), "uint32")
                _tmem_load(p, K.uint32(tmem_p_col), P_TMEM_COLS)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                K.ptx.tcgen05.fence__before_thread_sync()
                bar_p_free.arrive(softmax_stage, remote=K.uint32(0))

                bar_k_valid_ready.wait(softmax_stage, softmax_phase)
                valid_word_offset = K.if_then_else(idx_in_warpgroup >= 64, B_TOPK // 8 // 2 // 4, 0)
                is_k_valid_lo = K.local_scalar("uint32")
                is_k_valid_hi = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(
                    is_k_valid_lo,
                    is_k_valid.view("uint32").ptr_to([softmax_stage, valid_word_offset]),
                )
                K.ptx.ld.shared.u32(
                    is_k_valid_hi,
                    is_k_valid.view("uint32").ptr_to([softmax_stage, valid_word_offset + 1]),
                )

                def mask_p_half(valid_word, base):
                    with K.unroll(P_TMEM_COLS // 2) as p_i:
                        invalid_p_predicate = K.bitwise_and(
                            K.shift_right(valid_word, K.uint32(p_i)), K.uint32(1)
                        ) == K.uint32(0)
                        K.ptx.mov.b32(
                            p[base + p_i],
                            K.if_then_else(
                                invalid_p_predicate, K.uint32(0xFF800000), p[base + p_i]
                            ),
                        )

                mask_p_half(is_k_valid_lo, 0)
                mask_p_half(is_k_valid_hi, P_TMEM_COLS // 2)

                cur_pi_max = K.local_scalar("float32", init=K.float32(-float("inf")))
                with K.unroll(P_TMEM_COLS) as p_i:
                    K.assign(cur_pi_max, K.max(cur_pi_max, K.cuda.uint_as_float(p[p_i])))
                K.assign(cur_pi_max, cur_pi_max * sm_scale_div_log2)
                bar_k_valid_free.arrive(softmax_stage)

                K.ptx.bar.sync(K.uint32(BAR_WG0_SYNC), 128)
                K.ptx.st.shared.f32(rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max)
                K.ptx.bar.sync(K.uint32(BAR_WG0_SYNC), 128)
                peer_pi_max = K.local_scalar("float32")
                K.ptx.ld.shared.f32(peer_pi_max, rowwise_max_buf.ptr_to([idx_in_warpgroup ^ 64]))
                K.assign(cur_pi_max, K.max(cur_pi_max, peer_pi_max))
                K.assign(real_mi, K.max(real_mi, cur_pi_max))
                should_scale_o = K.local_scalar("uint32")
                K.ptx.vote_sync.any.pred(
                    should_scale_o, cur_pi_max - mi > K.float32(6.0), K.uint32(0xFFFFFFFF)
                )

                new_max = K.local_scalar("float32")
                scale_for_old = K.local_scalar("float32")
                with K.If(should_scale_o == K.uint32(0)):
                    with K.Then():
                        K.assign(scale_for_old, 1.0)
                        K.assign(new_max, mi)
                    with K.Else():
                        K.assign(new_max, K.max(cur_pi_max, mi))
                        K.ptx.ex2.approx.ftz.f32(scale_for_old, mi - new_max)
                K.assign(mi, new_max)
                K.assign(li, li * scale_for_old)

                # Each warpgroup thread owns B_TOPK/2 consecutive bf16 values.
                s_frag = K.alloc_local((B_TOPK // 2,), "bfloat16")
                s_pack = s_frag.view("uint32")
                neg_new_max_pair = K.cuda.make_float2(-new_max, -new_max)
                fma_pair = K.local_scalar("uint64")
                with K.unroll(P_TMEM_COLS // 2) as s_i:
                    p_pair = K.cuda.make_float2(
                        K.cuda.uint_as_float(p[s_i * 2]), K.cuda.uint_as_float(p[s_i * 2 + 1])
                    )
                    K.ptx.fma.rn.f32x2(fma_pair, p_pair, scale_pair, neg_new_max_pair)
                    s_x = K.local_scalar("float32")
                    s_y = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(s_x, K.cuda.float2_x(fma_pair))
                    K.ptx.ex2.approx.ftz.f32(s_y, K.cuda.float2_y(fma_pair))
                    K.assign(li, li + s_x + s_y)
                    K.ptx.mov.b32(s_pack[s_i], K.cuda.float22bfloat162_rn(s_x, s_y))

                with K.If(k > 0), K.Then():
                    pv_stage = ring_stage(k - 1)
                    pv_phase = ring_phase(k - 1)
                    pv_wait_token = iket_range("h128-pv-wait")
                    bar_sv_done.wait(pv_stage, pv_phase)
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.cuda.iket.range_end(pv_wait_token[0])

                s_base = idx_in_warpgroup // 64 * 4096 + idx_in_warpgroup % 64 * 8
                s_words = s_frag.view("uint32")
                with K.unroll(8) as s_store_i:
                    s_ptr = K.ptr_byte_offset(
                        s_smem_gemm.ptr_to([0, 0]),
                        (s_base + s_store_i * 512) * BF16_BYTES,
                        "bfloat16",
                    )
                    s_word = s_store_i * 4
                    K.ptx.st.shared.v4.u32(
                        s_ptr,
                        s_words[s_word],
                        s_words[s_word + 1],
                        s_words[s_word + 2],
                        s_words[s_word + 3],
                    )

                with K.If((k > 0) & (should_scale_o != K.uint32(0))), K.Then():
                    K.ptx.tcgen05.fence__after_thread_sync()
                    o_rescale = K.alloc_local((32,), "float32")
                    with K.unroll((D_V // 2) // 32) as chunk_idx:
                        _tmem_load(
                            o_rescale,
                            K.cuda.get_tmem_addr(K.uint32(o_tmem_col), 0, chunk_idx * 32),
                            32,
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        with K.unroll(32 // 2) as scale_i:
                            mul_f32x2(o_rescale, scale_i * 2, scale_for_old)
                        _tmem_store(
                            o_rescale, K.cuda.get_tmem_addr(K.uint32(o_tmem_col), 0, chunk_idx * 32)
                        )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                    K.ptx.tcgen05.fence__before_thread_sync()

                K.ptx.fence.proxy.async_.shared__cta()
                bar_so_ready.arrive(softmax_stage, remote=K.uint32(0))
                K.cuda.iket.range_end(softmax_token[0])

            epilogue_token = iket_range("h128-output")
            with K.If(real_mi == K.float32(-float("inf"))), K.Then():
                K.assign(li, 0.0)
                K.assign(mi, K.float32(-float("inf")))

            K.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), li)
            K.ptx.bar.sync(K.uint32(BAR_WG0_SYNC), 128)
            peer_li = K.local_scalar("float32")
            K.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup ^ 64]))
            K.assign(li, li + peer_li)

            with K.If(idx_in_warpgroup < B_H // 2), K.Then():
                global_head = cta_idx * (B_H // 2) + idx_in_warpgroup
                cur_lse = K.local_scalar("float32")
                cur_lse_log = K.log(li)
                K.ptx.fma.rn.f32(cur_lse, mi, LN_2, cur_lse_log)
                K.assign(
                    cur_lse,
                    K.if_then_else(
                        cur_lse == K.float32(-float("inf")), K.float32(float("inf")), cur_lse
                    ),
                )
                K.ptx.st.global_.f32(
                    max_logits.ptr_to(
                        [(s_q_idx * h_q + global_head) // h_q, (s_q_idx * h_q + global_head) % h_q]
                    ),
                    real_mi * LN_2,
                )
                K.ptx.st.global_.f32(
                    lse.ptr_to(
                        [(s_q_idx * h_q + global_head) // h_q, (s_q_idx * h_q + global_head) % h_q]
                    ),
                    cur_lse,
                )

            last_k = num_k_blocks - 1
            final_pv_stage = ring_stage(last_k)
            final_pv_phase = ring_phase(last_k)
            bar_sv_done.wait(final_pv_stage, final_pv_phase)
            K.ptx.fence.proxy.async_.shared__cta()
            K.ptx.tcgen05.fence__after_thread_sync()

            if have_attn_sink:
                attn_sink_val = K.local_scalar("float32")
                K.ptx.ld.global_.nc.f32(
                    attn_sink_val,
                    attn_sink.ptr_to([cta_idx * (B_H // 2) + (idx_in_warpgroup % 64)]),
                )
                attn_sink_log2 = attn_sink_val * LOG_2_E
            else:
                attn_sink_log2 = K.float32(-float("inf"))
            sink_exp = K.local_scalar("float32")
            K.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi)
            output_scale = K.local_scalar(
                "float32", init=K.cuda.fdividef(K.float32(1.0), li + sink_exp)
            )
            o_epi = K.alloc_local((B_EPI,), "float32")
            have_valid_indices = K.local_scalar("uint32")
            K.ptx.vote_sync.any.pred(
                have_valid_indices, K.Not(li == K.float32(0.0)), K.uint32(0xFFFFFFFF)
            )
            with K.If(have_valid_indices == K.uint32(0)), K.Then():
                with K.unroll(B_EPI) as o_zero_i:
                    K.ptx.mov.b32(o_epi[o_zero_i], K.float32(0.0))
                K.assign(output_scale, 1.0)
            o_epi_bf16 = K.alloc_local((B_EPI,), "bfloat16")
            with K.unroll((D_V // 2) // B_EPI) as epi_k:
                with K.If(have_valid_indices != K.uint32(0)), K.Then():
                    _tmem_load(
                        o_epi, K.cuda.get_tmem_addr(K.uint32(o_tmem_col), 0, epi_k * B_EPI), B_EPI
                    )
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.unroll(B_EPI // 2) as scale_i:
                    mul_f32x2(o_epi, scale_i * 2, output_scale)
                o_epi_words = o_epi_bf16.view("uint32")
                with K.unroll(B_EPI // 2) as cast_i:
                    K.ptx.cvt.rn.bf16x2.f32(
                        o_epi_words[cast_i], o_epi[cast_i * 2 + 1], o_epi[cast_i * 2]
                    )
                with K.unroll(8) as o_store_i:
                    s_off = (
                        idx_in_warpgroup // 64 * 16384
                        + epi_k * 4096
                        + idx_in_warpgroup % 64 * 64
                        + K.bitwise_xor(
                            o_store_i * 8,
                            K.shift_left(
                                K.bitwise_and(
                                    idx_in_warpgroup // 64 * 256
                                    + epi_k * 64
                                    + idx_in_warpgroup % 64,
                                    7,
                                ),
                                3,
                            ),
                        )
                    )
                    s_ptr = K.ptr_byte_offset(o_smem.ptr_to([0, 0]), s_off * BF16_BYTES, "bfloat16")
                    o_word = o_store_i * 4
                    K.ptx.st.shared.v4.u32(
                        s_ptr,
                        o_epi_words[o_word],
                        o_epi_words[o_word + 1],
                        o_epi_words[o_word + 2],
                        o_epi_words[o_word + 3],
                    )

                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.bar.sync(K.uint32(BAR_WG0_SYNC), 128)
                with K.If(warp_idx == 0), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        o_part0_offset = epi_k * B_EPI * (B_H // 2) * BF16_BYTES
                        K.ptx[_TMA_S2G_3D](
                            K.address_of(out_part0_tensormap),
                            K.cast(epi_k * B_EPI, "int32"),
                            K.cast(cta_idx * (B_H // 2), "int32"),
                            K.cast(s_q_idx, "int32"),
                            K.ptr_byte_offset(o_smem.ptr_to([0, 0]), o_part0_offset, "bfloat16"),
                        )
                with K.If(warp_idx == 1), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        o_part1_offset = (epi_k * B_EPI + D_V // 2) * (B_H // 2) * BF16_BYTES
                        K.ptx[_TMA_S2G_3D](
                            K.address_of(out_part1_tensormap),
                            K.cast(epi_k * B_EPI + D_V // 2, "int32"),
                            K.cast(cta_idx * (B_H // 2), "int32"),
                            K.cast(s_q_idx, "int32"),
                            K.ptr_byte_offset(o_smem.ptr_to([0, 0]), o_part1_offset, "bfloat16"),
                        )

            with K.If(warp_idx == 0), K.Then():
                K.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(K.uint32(0), K.uint32(512))
            K.cuda.iket.range_end(epilogue_token[0])

        def k_loader():
            # CUDA phase1.cuh:387-446.  K producer warpgroup.
            k_gather_token = iket_range("h128-k-load")
            wg1_warp_idx = warp_idx - 4
            with K.If(K.cuda.elect_sync()), K.Then():
                with K.serial(0, num_k_blocks, unroll=False) as k:
                    indices_int4 = K.alloc_local((WG1_ROWS_PER_WARP, 4), "int32")
                    max_indices = K.local_scalar("int32", init=-1)
                    min_indices = K.local_scalar("int32", init=s_kv)

                    # This CTA's topk half (cta_idx), split (local_row, warp, j): one
                    # strided nc copy (auto-vectorizes to 4x v4 ld.global.nc), like head64.
                    indices_words = indices_int4.view(16).view("uint32")
                    with K.unroll(4) as indices_load_i:
                        indices_word = indices_load_i * 4
                        K.ptx.ld.global_.nc.v4.u32(
                            indices_words[indices_word],
                            indices_words[indices_word + 1],
                            indices_words[indices_word + 2],
                            indices_words[indices_word + 3],
                            indices.ptr_to(
                                [
                                    g_indices_base
                                    + k * B_TOPK
                                    + cta_idx * (B_TOPK // 2)
                                    + wg1_warp_idx * 4
                                    + indices_load_i * 16
                                ]
                            ),
                        )
                    with K.unroll(WG1_ROWS_PER_WARP) as local_row:
                        with K.unroll(4) as j:
                            idx = indices_int4[local_row, j]
                            K.assign(max_indices, K.max(max_indices, idx))
                            K.assign(min_indices, K.min(min_indices, idx))

                    is_all_rows_invalid = (min_indices == s_kv) | (max_indices == -1)
                    should_skip_tma = is_all_rows_invalid & (k >= NUM_BUFS)
                    k_stage = ring_stage(k)

                    def gather_k_part(col_start, col_count, tx_dim, bar, tensormap):
                        with K.If(K.Not(should_skip_tma)):
                            with K.Then():
                                with K.unroll(WG1_ROWS_PER_WARP) as row_group:
                                    with K.unroll(col_count) as col_atom:
                                        k_gather_offset = (
                                            (col_start + col_atom) * 64 * (B_TOPK // 2)
                                            + (wg1_warp_idx * 4 + row_group * 16) * 64
                                        ) * BF16_BYTES
                                        K.ptx[_TMA_GATHER4_2D_CACHE](
                                            K.ptr_byte_offset(
                                                k_smem.ptr_to([0, 0]), k_gather_offset, "bfloat16"
                                            ),
                                            K.address_of(tensormap),
                                            K.cast((col_start + col_atom) * 64, "int32"),
                                            indices_int4[row_group, 0],
                                            indices_int4[row_group, 1],
                                            indices_int4[row_group, 2],
                                            indices_int4[row_group, 3],
                                            K.cuda.cvta_generic_to_shared(
                                                leader_mbar(bar.ptr_to([k_stage]))
                                            ),
                                            _KV_TMA_CACHE_HINT,
                                        )
                            with K.Else():
                                _rem1 = K.local_scalar("uint64")
                                K.ptx.mapa.shared__cluster.u64(
                                    _rem1, bar.ptr_to([k_stage]), K.uint32(0)
                                )
                                K.ptx.mbarrier.complete_tx.relaxed.cluster.b64(
                                    _rem1,
                                    K.uint32(WG1_ROWS_PER_WARP * 4 * tx_dim * BF16_BYTES),
                                    pred=K.uint32(1),
                                )

                    with K.If(k > 0), K.Then():
                        prior_qk_part_stage = ring_stage(k - 1)
                        prior_qk_part_phase = ring_phase(k - 1)
                        bar_qk_part_done.wait(prior_qk_part_stage, prior_qk_part_phase)
                    gather_k_part(0, num_sq_tiles, d_sq, bar_k_part0_ready, kv_k_part0_tensormap)

                    with K.If(k > 0), K.Then():
                        prior_qk_stage = ring_stage(k - 1)
                        prior_qk_phase = ring_phase(k - 1)
                        bar_qk_done.wait(prior_qk_stage, prior_qk_phase)
                    gather_k_part(
                        num_sq_tiles,
                        num_qk_tiles - num_sq_tiles,
                        D_TQ,
                        bar_k_part1_ready,
                        kv_k_part1_tensormap,
                    )
            K.cuda.iket.range_end(k_gather_token[0])

        def v_loader():
            # CUDA phase1.cuh:447-489.  V producer warpgroup.
            v_gather_token = iket_range("h128-v-load")
            wg2_warp_idx = warp_idx - 8
            with K.If(K.cuda.elect_sync()), K.Then():
                bar_prologue_utccp.wait(0, 0)
                with K.serial(0, num_k_blocks, unroll=False) as k:
                    v_stage = ring_stage(k)
                    with K.If(k > 0), K.Then():
                        prior_pv_part_stage = ring_stage(k - 1)
                        prior_pv_part_phase = ring_phase(k - 1)
                        bar_sv_part_done.wait(prior_pv_part_stage, prior_pv_part_phase)

                    def gather_v_part(row_offset, part, token_buf, bar, tensormap):
                        # V loads all 128 tokens; the two parts map to an extent-2
                        # axis indexed by part. One strided nc copy, like head64.
                        token_words = token_buf.view(16).view("uint32")
                        with K.unroll(4) as token_load_i:
                            token_word = token_load_i * 4
                            K.ptx.ld.global_.nc.v4.u32(
                                token_words[token_word],
                                token_words[token_word + 1],
                                token_words[token_word + 2],
                                token_words[token_word + 3],
                                indices.ptr_to(
                                    [
                                        g_indices_base
                                        + k * B_TOPK
                                        + part * (B_TOPK // 2)
                                        + wg2_warp_idx * 4
                                        + token_load_i * 16
                                    ]
                                ),
                            )
                        src0 = cta_idx * 256
                        with K.unroll(WG2_ROWS_PER_PART) as row_group:
                            with K.unroll((D_V // 2) // 64) as col_atom:
                                v_gather_offset = (
                                    part * (B_TOPK // 2) * 64
                                    + col_atom * 64 * B_TOPK
                                    + (wg2_warp_idx * 4 + row_group * 16) * 64
                                ) * BF16_BYTES
                                K.ptx[_TMA_GATHER4_2D_CACHE](
                                    K.ptr_byte_offset(
                                        v_smem.ptr_to([0, 0]), v_gather_offset, "bfloat16"
                                    ),
                                    K.address_of(tensormap),
                                    K.cast(src0 + col_atom * 64, "int32"),
                                    token_buf[row_group, 0],
                                    token_buf[row_group, 1],
                                    token_buf[row_group, 2],
                                    token_buf[row_group, 3],
                                    K.cuda.cvta_generic_to_shared(
                                        leader_mbar(bar.ptr_to([v_stage]))
                                    ),
                                    _KV_TMA_CACHE_HINT,
                                )

                    token_idxs_part0 = K.alloc_local((WG2_ROWS_PER_PART, 4), "int32")
                    gather_v_part(0, 0, token_idxs_part0, bar_v_part0_ready, kv_v_part0_tensormap)

                    with K.If(k > 0), K.Then():
                        prior_pv_stage = ring_stage(k - 1)
                        prior_pv_phase = ring_phase(k - 1)
                        bar_sv_done.wait(prior_pv_stage, prior_pv_phase)
                    token_idxs_part1 = K.alloc_local((WG2_ROWS_PER_PART, 4), "int32")
                    gather_v_part(
                        WG2_ROWS_PER_PART,
                        1,
                        token_idxs_part1,
                        bar_v_part1_ready,
                        kv_v_part1_tensormap,
                    )
            K.cuda.iket.range_end(v_gather_token[0])

        def run_wg3_role(do_mma: K.constexpr):
            # CUDA phase1.cuh:490-606.  The constexpr selects one of the two
            # independently declared K roles without adding a runtime branch.
            with K.If(do_mma):
                with K.Then():
                    mma_token = iket_range("h128-qk-pv-issue")
                    with K.If(K.cuda.elect_sync()), K.Then():
                        bar_prologue_q.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                        bar_prologue_q.wait(0, 0)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        with K.unroll(48) as q_copy_flat:
                            q_copy_src = K.ptr_byte_offset(
                                q_full.ptr_to([0, 0]),
                                (d_sq * 8 + q_copy_flat % 6 * 512 + q_copy_flat // 6 % 8) * 16,
                                "bfloat16",
                            )
                            K.ptx[_TCGEN_CP_64X128](
                                K.cast(
                                    q_tmem_col + q_copy_flat % 6 * 32 + q_copy_flat // 6 % 8 * 4,
                                    "uint32",
                                ),
                                _replace_smem_desc_addr(q_cp_desc, q_copy_src),
                            )
                        K.ptx[_TCGEN_COMMIT](
                            K.cuda.cvta_generic_to_shared(bar_prologue_utccp.ptr_to([0])),
                            K.uint16(3),
                        )

                        with K.serial(0, num_k_blocks + 1, unroll=False) as k:
                            with K.If(k < num_k_blocks), K.Then():
                                qk_stage = ring_stage(k)
                                qk_phase = ring_phase(k)

                                bar_k_part0_ready.arrive(
                                    qk_stage, tx_count=B_TOPK * d_sq * BF16_BYTES
                                )
                                bar_k_part0_ready.wait(qk_stage, qk_phase)
                                with K.If(k > 0), K.Then():
                                    prior_p_stage = ring_stage(k - 1)
                                    prior_p_phase = ring_phase(k - 1)
                                    bar_p_free.wait(prior_p_stage, prior_p_phase)
                                K.ptx.tcgen05.fence__after_thread_sync()

                                K.assign(mma_p_accumulate, K.uint32(0))
                                with K.If(d_sq > 0), K.Then():
                                    if mma_smem_desc == "local_hoist":
                                        qk_part0_a_local = K.SmemDescriptor()
                                        qk_part0_a_local.init(
                                            q_full.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                        )
                                        qk_part0_b_local = K.SmemDescriptor()
                                        qk_part0_b_local.init(
                                            k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                        )
                                    if mma_smem_desc == "hoist":
                                        qk_part0_a_hoist = K.SmemDescriptor()
                                        qk_part0_a_hoist.init(
                                            q_full.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                        )
                                    with K.unroll(1) as mma_mi:
                                        with K.unroll(1) as mma_ni:
                                            with K.unroll(d_sq // 16) as mma_ki:
                                                qk_part0_offset = (
                                                    mma_ki % (d_sq // 16) // 4 * 4096
                                                    + mma_mi * 4096
                                                    + mma_ki // (d_sq // 16) * 64
                                                    + mma_ki % 4 * 16
                                                )
                                                if mma_smem_desc == "recompute":
                                                    qk_part0_a_ptr = K.ptr_byte_offset(
                                                        q_full.ptr_to([0, 0]),
                                                        qk_part0_offset // 8 * 16,
                                                        "bfloat16",
                                                    )
                                                    qk_part0_b_ptr = K.ptr_byte_offset(
                                                        k_smem.ptr_to([0, 0]),
                                                        qk_part0_offset // 8 * 16,
                                                        "bfloat16",
                                                    )
                                                    _mma_f16(
                                                        K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                        _recompute_smem_desc(
                                                            qk_part0_a_ptr, 0x40004040, 0x02000000
                                                        ),
                                                        _recompute_smem_desc(
                                                            qk_part0_b_ptr, 0x40004040, 0x02000000
                                                        ),
                                                        K.uint32(0x08200490),
                                                        K.Or(
                                                            mma_ki != 0,
                                                            K.cast(mma_p_accumulate, "bool"),
                                                        ),
                                                    )
                                                elif mma_smem_desc == "encode":
                                                    qk_part0_a_encode = K.SmemDescriptor()
                                                    qk_part0_a_encode.init(
                                                        K.ptr_byte_offset(
                                                            q_full.ptr_to([0, 0]),
                                                            qk_part0_offset // 8 * 16,
                                                            "bfloat16",
                                                        ),
                                                        ldo=512,
                                                        sdo=64,
                                                        swizzle=3,
                                                    )
                                                    qk_part0_b_encode = K.SmemDescriptor()
                                                    qk_part0_b_encode.init(
                                                        K.ptr_byte_offset(
                                                            k_smem.ptr_to([0, 0]),
                                                            qk_part0_offset // 8 * 16,
                                                            "bfloat16",
                                                        ),
                                                        ldo=512,
                                                        sdo=64,
                                                        swizzle=3,
                                                    )
                                                    _mma_f16(
                                                        K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                        qk_part0_a_encode.desc,
                                                        qk_part0_b_encode.desc,
                                                        K.uint32(0x08200490),
                                                        K.Or(
                                                            mma_ki != 0,
                                                            K.cast(mma_p_accumulate, "bool"),
                                                        ),
                                                    )
                                                elif mma_smem_desc == "local_hoist":
                                                    _mma_f16(
                                                        K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                        qk_part0_a_local.add_16B_offset(
                                                            qk_part0_offset // 8
                                                        ),
                                                        qk_part0_b_local.add_16B_offset(
                                                            qk_part0_offset // 8
                                                        ),
                                                        K.uint32(0x08200490),
                                                        K.Or(
                                                            mma_ki != 0,
                                                            K.cast(mma_p_accumulate, "bool"),
                                                        ),
                                                    )
                                                else:
                                                    _mma_f16(
                                                        K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                        qk_part0_a_hoist.add_16B_offset(
                                                            qk_part0_offset // 8
                                                        ),
                                                        qk_k_part0_desc.add_16B_offset(
                                                            qk_part0_offset // 8
                                                        ),
                                                        K.uint32(0x08200490),
                                                        K.Or(
                                                            mma_ki != 0,
                                                            K.cast(mma_p_accumulate, "bool"),
                                                        ),
                                                    )
                                    K.assign(mma_p_accumulate, K.uint32(1))
                                bar_qk_part_done.arrive(qk_stage, cta_group=2, cta_mask=3)

                                bar_k_part1_ready.arrive(
                                    qk_stage, tx_count=B_TOPK * (d_qk - d_sq) * BF16_BYTES
                                )
                                bar_k_part1_ready.wait(qk_stage, qk_phase)
                                K.ptx.tcgen05.fence__after_thread_sync()

                                if mma_smem_desc == "local_hoist":
                                    qk_part1_b_local = K.SmemDescriptor()
                                    qk_part1_b_local.init(
                                        k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                    )
                                with K.unroll(1) as mma_mi:
                                    with K.unroll(1) as mma_ni:
                                        with K.unroll(D_TQ // 16) as mma_ki:
                                            qk_part1_offset = (
                                                mma_ki % (D_TQ // 16) // 4 * 4096
                                                + mma_ni * 4096
                                                + mma_ki // (D_TQ // 16) * 64
                                                + mma_ki % 4 * 16
                                                + d_sq // 64 * 4096
                                            )
                                            if mma_smem_desc == "recompute":
                                                qk_part1_b_ptr = K.ptr_byte_offset(
                                                    k_smem.ptr_to([0, 0]),
                                                    qk_part1_offset // 8 * 16,
                                                    "bfloat16",
                                                )
                                                _mma_f16(
                                                    K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    K.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                    _recompute_smem_desc(
                                                        qk_part1_b_ptr, 0x40004040, 0x02000000
                                                    ),
                                                    K.uint32(0x08200490),
                                                    K.Or(
                                                        mma_ki != 0,
                                                        K.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            elif mma_smem_desc == "encode":
                                                qk_part1_b_encode = K.SmemDescriptor()
                                                qk_part1_b_encode.init(
                                                    K.ptr_byte_offset(
                                                        k_smem.ptr_to([0, 0]),
                                                        qk_part1_offset // 8 * 16,
                                                        "bfloat16",
                                                    ),
                                                    ldo=512,
                                                    sdo=64,
                                                    swizzle=3,
                                                )
                                                _mma_f16(
                                                    K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    K.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                    qk_part1_b_encode.desc,
                                                    K.uint32(0x08200490),
                                                    K.Or(
                                                        mma_ki != 0,
                                                        K.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            elif mma_smem_desc == "local_hoist":
                                                _mma_f16(
                                                    K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    K.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                    qk_part1_b_local.add_16B_offset(
                                                        qk_part1_offset // 8
                                                    ),
                                                    K.uint32(0x08200490),
                                                    K.Or(
                                                        mma_ki != 0,
                                                        K.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            else:
                                                _mma_f16(
                                                    K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    K.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                    qk_k_part1_desc.add_16B_offset(
                                                        qk_part1_offset // 8
                                                    ),
                                                    K.uint32(0x08200490),
                                                    K.Or(
                                                        mma_ki != 0,
                                                        K.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                K.assign(mma_p_accumulate, K.uint32(1))
                                bar_qk_done.arrive(qk_stage, cta_group=2, cta_mask=3)

                            with K.If(k > 0), K.Then():
                                pv_stage = ring_stage(k - 1)
                                pv_phase = ring_phase(k - 1)
                                bar_so_ready.wait(pv_stage, pv_phase)

                                bar_v_part0_ready.arrive(
                                    pv_stage, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                                )
                                bar_v_part0_ready.wait(pv_stage, pv_phase)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                K.assign(
                                    mma_o_accumulate,
                                    K.if_then_else(k == 1, K.uint32(0), K.uint32(1)),
                                )
                                if mma_smem_desc == "hoist":
                                    issue_pv_mma(
                                        0, 0, 0, pv_a_part0_lo_desc.desc, pv_b_part0_lo_desc.desc
                                    )
                                    issue_pv_mma(
                                        128,
                                        0,
                                        16384,
                                        pv_a_part0_hi_desc.desc,
                                        pv_b_part0_hi_desc.desc,
                                    )
                                else:
                                    issue_pv_mma(0, 0, 0, K.uint64(0), K.uint64(0))
                                    issue_pv_mma(128, 0, 16384, K.uint64(0), K.uint64(0))
                                K.assign(mma_o_accumulate, K.uint32(1))
                                bar_sv_part_done.arrive(pv_stage, cta_group=2, cta_mask=3)

                                bar_v_part1_ready.arrive(
                                    pv_stage, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                                )
                                bar_v_part1_ready.wait(pv_stage, pv_phase)
                                K.ptx.tcgen05.fence__after_thread_sync()
                                if mma_smem_desc == "hoist":
                                    issue_pv_mma(
                                        0,
                                        4096,
                                        4096,
                                        pv_a_part1_lo_desc.desc,
                                        pv_b_part1_lo_desc.desc,
                                    )
                                    issue_pv_mma(
                                        128,
                                        4096,
                                        20480,
                                        pv_a_part1_hi_desc.desc,
                                        pv_b_part1_hi_desc.desc,
                                    )
                                else:
                                    issue_pv_mma(0, 4096, 4096, K.uint64(0), K.uint64(0))
                                    issue_pv_mma(128, 4096, 20480, K.uint64(0), K.uint64(0))
                                K.assign(mma_o_accumulate, K.uint32(1))
                                bar_sv_done.arrive(pv_stage, cta_group=2, cta_mask=3)
                    K.cuda.iket.range_end(mma_token[0])

                with K.Else():
                    valid_mask_token = iket_range("h128-valid-mask")
                    with K.If(lane_idx < B_TOPK // 8), K.Then():
                        lane_indices = K.alloc_local((8,), "int32")
                        with K.serial(0, num_k_blocks, unroll=False) as k:
                            row_base = g_indices_base + k * B_TOPK + lane_idx * 8
                            lane_index_words = lane_indices.view("uint32")
                            K.ptx["ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.u32"](
                                lane_index_words[0],
                                lane_index_words[1],
                                lane_index_words[2],
                                lane_index_words[3],
                                lane_index_words[4],
                                lane_index_words[5],
                                lane_index_words[6],
                                lane_index_words[7],
                                indices.ptr_to([row_base]),
                            )
                            abs_pos_start = k * B_TOPK
                            is_ks_valid_mask = pack_valid_mask8(
                                lane_indices, abs_pos_start, lane_idx, topk_len, s_kv
                            )
                            valid_stage = ring_stage(k)
                            valid_phase = ring_phase(k)
                            bar_k_valid_free.wait(valid_stage, valid_phase ^ 1)
                            K.ptx.st.shared.b8(
                                is_k_valid.ptr_to([valid_stage, lane_idx]),
                                K.reinterpret("uint8", is_ks_valid_mask),
                            )
                            bar_k_valid_ready.arrive(valid_stage)
                    K.cuda.iket.range_end(valid_mask_token[0])

        roles = K.specialize(chain_dispatch=True)
        softmax = roles.role("softmax", warps=range(0, 4), regs=144)
        k_load = roles.role("k_loader", warps=range(4, 8), regs=96)
        v_load = roles.role("v_loader", warps=range(8, 12), regs=96)
        wg3 = roles.warpgroup("wg3", warps=range(12, 16), regs=168)
        mma = roles.role("mma", warps=[12], when=cta_idx == 0, group=wg3)
        valid = roles.role("valid", warps=[13], group=wg3)
        roles.role("idle", warps=range(14, 16), group=wg3)
        with softmax:
            softmax_and_epilogue()
        with k_load:
            k_loader()
        with v_load:
            v_loader()
        with wg3:
            with mma:
                run_wg3_role(True)
            with valid:
                run_wg3_role(False)

    sparse_flashmla_prefill_head128_phase1_kern.__annotations__ = {
        "q": K.gptr[K.bf16, (s_q, h_q, d_qk)],
        "kv": K.gptr[K.bf16, (s_kv * stride_kv_s_kv,)],
        "indices": K.gptr[K.i32, (s_q * stride_indices_s_q,)],
        "attn_sink": K.gptr[K.f32, (h_q,)],
        "topk_length": K.gptr[K.i32, (s_q,)],
        "out": K.gptr[K.bf16, (s_q, h_q, D_V)],
        "max_logits": K.gptr[K.f32, (s_q, h_q)],
        "lse": K.gptr[K.f32, (s_q, h_q)],
    }
    return K.kernel(
        warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=2 * s_q, host_prelude=host_prelude
    )(sparse_flashmla_prefill_head128_phase1_kern)


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    specialization = {
        "s_q": cfg.s_q,
        "s_kv": cfg.s_kv,
        "topk": cfg.topk,
        "d_qk": cfg.d_qk,
        "h_q": cfg.h_q,
        "stride_kv_s_kv": stride_kv_s_kv,
        "stride_indices_s_q": stride_indices_s_q,
        "have_attn_sink": cfg.have_attn_sink,
        "have_topk_length": cfg.have_topk_length,
        "sm_scale_div_log2": (1.0 / math.sqrt(cfg.d_qk)) * LOG_2_E,
    }
    return (
        make_kernel(**specialization)
        .func.with_attr("global_symbol", KERNEL_META["name"])
        .with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    # Preserve the independent oracle for all three outputs.
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    from tirx_kernels.target import prepare_cuda_arch

    if prepare_cuda_arch() == "sm_110a":
        from tirx_kernels.flashmla.utils._flashmla_bench import validate_flashmla_sparse_prefill

        validate_flashmla_sparse_prefill(
            case, (ref_out, ref_max_logits, ref_lse), output_rtol=4.01 / 128
        )
    cfg.validate()


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
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1 benchmark")

    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    ex = executable

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    args = _tirx_args(case)

    funcs = {"tirx": lambda: ex(*args)}

    from tirx_kernels.flashmla.utils._flashmla_bench import flashmla_reference_builder
    from tirx_kernels.flashmla.utils._trtllm_gen_bench import (
        trtllm_gen_config_compatible,
        trtllm_gen_reference_builder,
    )

    references = {"flashmla": lambda: flashmla_reference_builder(case)}
    if trtllm_gen_config_compatible(case["config"]):
        references["trtllm_gen"] = lambda: trtllm_gen_reference_builder(case)

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )
    from tirx_kernels.reference_variants import reference_provenance

    result["reference_variant"] = reference_provenance("flash-mla")
    return result


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
