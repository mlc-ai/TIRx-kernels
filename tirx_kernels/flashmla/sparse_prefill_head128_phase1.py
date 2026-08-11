# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025
# DeepSeek, licensed under the MIT License. The upstream sources carry no
# per-file license header; see licenses/LICENSE.flashmla.txt for the full
# license text.
#
# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# TIRx port of FlashMLA's sparse prefill phase-1 kernel, 128 q-heads.
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla.utils._mask import pack_valid_mask8
from tirx_kernels.flashmla.utils._tma import leader_mbar
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.lang.smem_desc import SmemDescriptor
from tvm.tirx.layout import S, TileLayout, laneid, wid_in_wg

B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

IKET_EVENT_NAMES = (
    "h128-q-load",
    "h128-softmax-tile",
    "h128-output",
    "h128-k-load",
    "h128-v-load",
    "h128-qk-pv-issue",
    "h128-qk-wait",
    "h128-pv-wait",
    "h128-valid-mask",
)

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
_Q_TMA_CACHE_HINT = T.uint64(0x12F0000000000000)
_KV_TMA_CACHE_HINT = T.uint64(0x14F0000000000000)


def _tmem_load(dst, tmem_col, width):
    chain = _TMEM_LD_32 if width == 32 else _TMEM_LD_64
    return T.ptx[chain](*[dst[i] for i in range(width)], tmem_col)


def _tmem_store(src, tmem_col, width=32):
    assert width == 32
    return T.ptx[_TMEM_ST_32](tmem_col, *[src[i] for i in range(width)])


def _cast_f32x2_bf16x2(dst, src, offset):
    dst_words = dst.view("uint32")
    return T.ptx.cvt.rn.bf16x2.f32(dst_words[offset // 2], src[offset + 1], src[offset])


def _replace_smem_desc_addr(desc, smem_ptr):
    start_addr = T.cast(
        T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)
        ),
        "uint64",
    )
    return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)


def _recompute_smem_desc(smem_ptr, upper, matrix_start):
    start_addr = T.bitwise_and(
        T.shift_right(T.cuda.cvta_generic_to_shared(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)
    )
    return T.bitwise_or(
        T.shift_left(T.uint64(upper), T.uint64(32)),
        T.cast(T.bitwise_or(T.uint32(matrix_start), start_addr), "uint64"),
    )


def _add_smem_desc_offset(desc, offset):
    # Descriptor offsets wrap in the low 32 bits without carrying into the
    # encoded layout fields in the high half.
    desc_lo = T.alloc_local((1,), "uint32")
    desc_hi = T.alloc_local((1,), "uint32")
    result = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mov.b64(desc_lo[0], desc_hi[0], desc))
    T.evaluate(T.ptx.add.u32(desc_lo[0], desc_lo[0], T.cast(offset, "uint32")))
    T.evaluate(T.ptx.mov.b64(result[0], desc_lo[0], desc_hi[0]))
    return result[0]


def _mma_f16(d_tmem, a_operand, b_desc, idesc, enable_input_d):
    return T.ptx[_MMA_F16](
        d_tmem,
        a_operand,
        b_desc,
        idesc,
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        enable_input_d,
    )


@T.inline
def mul_f32x2(values, idx, multiplier):
    packed: T.uint64
    rhs: T.uint64
    T.ptx.mov.b64(packed, values[idx], values[idx + 1])
    T.ptx.mov.b64(rhs, multiplier, multiplier)
    T.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
    T.ptx.mov.b64(values[idx], values[idx + 1], packed)


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
    "compute_capability": 10,
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


@T.jit
def _kernel(
    q: T.Buffer((s_q, h_q, d_qk), "bfloat16"),
    kv: T.Buffer((s_kv * stride_kv_s_kv,), "bfloat16"),
    indices: T.Buffer((s_q * stride_indices_s_q,), "int32"),
    attn_sink: T.Buffer((h_q,), "float32"),
    topk_length: T.Buffer((s_q,), "int32"),
    out: T.Buffer((s_q, h_q, D_V), "bfloat16"),
    max_logits: T.Buffer((s_q, h_q), "float32"),
    lse: T.Buffer((s_q, h_q), "float32"),
    *,
    s_q: T.constexpr,
    s_kv: T.constexpr,
    topk: T.constexpr,
    d_qk: T.constexpr,
    h_q: T.constexpr,
    stride_kv_s_kv: T.constexpr,
    stride_indices_s_q: T.constexpr,
    have_attn_sink: T.constexpr,
    have_topk_length: T.constexpr,
    sm_scale_div_log2: T.constexpr,
):
    kv_v_part1_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        kv_v_part1_tensormap,
        "bfloat16",
        2,
        kv.data,
        d_qk,
        s_kv,
        stride_kv_s_kv * BF16_BYTES,
        64,
        1,
        1,
        1,
        0,
        3,
        3,
        0,
    )
    kv_v_part0_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        kv_v_part0_tensormap,
        "bfloat16",
        2,
        kv.data,
        d_qk,
        s_kv,
        stride_kv_s_kv * BF16_BYTES,
        64,
        1,
        1,
        1,
        0,
        3,
        3,
        0,
    )
    kv_k_part1_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        kv_k_part1_tensormap,
        "bfloat16",
        2,
        kv.data,
        d_qk,
        s_kv,
        stride_kv_s_kv * BF16_BYTES,
        64,
        1,
        1,
        1,
        0,
        3,
        3,
        0,
    )
    kv_k_part0_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        kv_k_part0_tensormap,
        "bfloat16",
        2,
        kv.data,
        d_qk,
        s_kv,
        stride_kv_s_kv * BF16_BYTES,
        64,
        1,
        1,
        1,
        0,
        3,
        3,
        0,
    )
    out_part1_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        out_part1_tensormap,
        "bfloat16",
        3,
        out.data,
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
    out_part0_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        out_part0_tensormap,
        "bfloat16",
        3,
        out.data,
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
    q_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        q_tensormap,
        "bfloat16",
        4,
        q.data,
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
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    iket = IketProfiler()
    # CUDA_TRANSCRIBE_START: run_fwd_phase1_kernel line 622, then sparse_attn_fwd_kernel_devfunc
    # line 68. One CTA pair per query-row; upstream source-order TMA/MMA/softmax warp layout.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    s_q_idx: T.let = block_idx // 2
    thread_idx = T.thread_id([NUM_THREADS])
    T.warpgroup_id([NUM_THREADS // 128])
    T.warp_id_in_wg([4])
    T.lane_id([32])
    T.thread_id_in_wg([128])
    warp_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 32, 0, 32)
    lane_idx: T.let = thread_idx % 32
    topk_len: T.let = (
        T.cuda.ldg(topk_length.ptr_to([s_q_idx]), "int32") if have_topk_length else topk
    )
    num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    warpgroup_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 128, 0, 32)
    idx_in_warpgroup: T.let = thread_idx % 128
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(out_part0_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(out_part1_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(kv_k_part0_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(kv_k_part1_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(kv_v_part0_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(kv_v_part1_tensormap)))
    d_sq = T.meta_var(d_qk - D_TQ)
    num_sq_tiles = T.meta_var((d_qk - D_TQ) // 64)
    num_qk_tiles = T.meta_var(d_qk // 64)
    mma_smem_desc = T.meta_var(
        "recompute"
        if (d_qk == 512 and s_kv == 8192)
        else "local_hoist"
        if (d_qk == 576 and s_kv != 65536)
        else "hoist"
        if ((d_qk == 512 and s_kv == 32768) or (d_qk == 576 and s_kv == 65536))
        else "encode"
    )

    # CUDA phase1.cuh:84-90, config.h:93-118.  Preserve SharedMemoryPlan's
    # union offsets: q_full, {sq, v, k}, and o alias the same base.
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    q_full = pool.alloc_tcgen05_mma_AB((B_H // 2, d_qk), "bfloat16")
    q_cp_desc: T.uint64
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(q_cp_desc), T.reinterpret(T.handle().ty, T.uint64(0)), 0, 64, 3
    )
    # sQ stays live as q_full's first d_sq cols (a contiguous prefix under the
    # 64-col swizzle chunks); v/k reuse the D_TQ tail once Q has moved to TMEM.
    pool.move_base_to(u_base + (B_H // 2) * d_sq * BF16_BYTES)
    v_smem = pool.alloc_tcgen05_mma_AB((D_V // 2, B_TOPK), "bfloat16")
    k_smem = pool.alloc_tcgen05_mma_AB((B_TOPK // 2, d_qk), "bfloat16")
    u_end = T.meta_var(pool.offset)
    pool.move_base_to(u_base)
    o_smem = pool.alloc_tcgen05_mma_AB((B_H // 2, D_V), "bfloat16")
    pool.move_base_to(u_end)
    s_smem_gemm = pool.alloc_tcgen05_mma_AB(
        (B_H // 2, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
    )
    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q = TMABar(pool, 1)
    bar_prologue_utccp = TCGen05Bar(pool, 1)
    bar_qk_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_k_part0_ready = TMABar(pool, NUM_BUFS)
    bar_k_part1_ready = TMABar(pool, NUM_BUFS)
    bar_v_part0_ready = TMABar(pool, NUM_BUFS)
    bar_v_part1_ready = TMABar(pool, NUM_BUFS)
    bar_p_free = MBarrier(pool, NUM_BUFS)
    bar_so_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()
    kv_tma = kv.view(s_kv, d_qk, layout=TileLayout(S[(s_kv, d_qk) : (stride_kv_s_kv, 1)]))

    g_indices_base: T.let = s_q_idx * stride_indices_s_q
    mma_p_accumulate: T.uint32 = 0
    mma_o_accumulate: T.uint32 = 0

    # CUDA phase1.cuh:87-146.  Warp 0 owns barrier init, Q TMA launch,
    # and the cta_group::2 TMEM allocation.
    if warp_idx == 0:
        if T.cuda.elect_sync():
            bar_prologue_q.init(1)
            bar_prologue_utccp.init(1)
            for init_stage in T.unroll(NUM_BUFS):
                T.ptx.mbarrier.init.shared.b64(bar_qk_part_done.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_qk_done.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_sv_part_done.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_sv_done.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_k_part0_ready.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_k_part1_ready.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_v_part0_ready.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_v_part1_ready.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_p_free.ptr_to([init_stage]), T.uint32(128 * 2))
                T.ptx.mbarrier.init.shared.b64(bar_so_ready.ptr_to([init_stage]), T.uint32(128 * 2))
                T.ptx.mbarrier.init.shared.b64(bar_k_valid_ready.ptr_to([init_stage]), T.uint32(16))
                T.ptx.mbarrier.init.shared.b64(bar_k_valid_free.ptr_to([init_stage]), T.uint32(128))
            T.ptx.fence.mbarrier_init.release.cluster()

    T.cuda.cluster_sync()

    if warp_idx == 0:
        prologue_token = iket.range_start("h128-q-load")
        if T.cuda.elect_sync():
            T.evaluate(
                T.ptx[_TMA_G2S_4D_CACHE](
                    q_full.ptr_to([0, 0]),
                    T.address_of(q_tensormap),
                    T.int32(0),
                    T.cast(cta_idx * (B_H // 2), "int32"),
                    T.int32(0),
                    T.cast(s_q_idx, "int32"),
                    T.cuda.cvta_generic_to_shared(leader_mbar(bar_prologue_q.ptr_to([0]))),
                    _Q_TMA_CACHE_HINT,
                )
            )

        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
            T.address_of(tmem_start_addr[0]), T.uint32(512)
        )
        allocated_tmem_start: T.uint32
        T.ptx.ld.shared.u32(allocated_tmem_start, tmem_start_addr.ptr_to([0]))
        T.cuda.trap_when_assert_failed(allocated_tmem_start == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
        iket.range_end(prologue_token)

    T.cuda.cta_sync()

    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_start_addr)
    # O accumulator: one alloc; logical col halves are the B lo/hi gemm outputs,
    # read back as a (128, D_V//2) datapath-D tile via permute+reshape.
    o_tmem_col = T.meta_var(tmem_pool.offset)
    o_tmem = tmem_pool.alloc_tcgen05_mma_D(
        (B_H // 2, D_V), "float32", M=128, cta_group=2, group=(2, 2, 128)
    )
    tmem_o_lo = o_tmem.sub[:, 0 : D_V // 2]
    tmem_o_hi = o_tmem.sub[:, D_V // 2 : D_V]
    o_win = o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
    tmem_p_col = T.meta_var(tmem_pool.offset)
    tmem_p = tmem_pool.alloc_tcgen05_mma_D((B_H // 2, B_TOPK), "float32", M=128, cta_group=2)
    # Qt TMEM at real 128-lane footprint: the 64x128b.warpx2::02_13 copy mirrors rows 0-63 to
    # lane +64, so the alloc declares that replica (R[2:64@TLane]); MMA validates it at the anchor.
    q_tmem_col = T.meta_var(tmem_pool.offset)
    q_tmem = tmem_pool.alloc_tcgen05_mma_A((B_H // 2, D_TQ), "bfloat16", M=128, cta_group=2)
    v_smem_gemm = v_smem.rearrange("(x r) (z kl) -> r (z x kl)", x=2, z=2, kl=64)

    if mma_smem_desc == "hoist":
        qk_k_part0_desc = SmemDescriptor()
        qk_k_part0_desc.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)
        qk_k_part1_desc = SmemDescriptor()
        qk_k_part1_desc.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)

        pv_a_part0_lo_desc = SmemDescriptor()
        pv_a_part0_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
        pv_a_part0_hi_desc = SmemDescriptor()
        pv_a_part0_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
        pv_a_part1_lo_desc = SmemDescriptor()
        pv_a_part1_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
        pv_a_part1_hi_desc = SmemDescriptor()
        pv_a_part1_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)

        pv_b_part0_lo_desc = SmemDescriptor()
        pv_b_part0_lo_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
        pv_b_part0_hi_desc = SmemDescriptor()
        pv_b_part0_hi_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
        pv_b_part1_lo_desc = SmemDescriptor()
        pv_b_part1_lo_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
        pv_b_part1_hi_desc = SmemDescriptor()
        pv_b_part1_hi_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)

    @T.inline
    def issue_pv_mma(dest_offset, s_offset, v_offset, hoisted_a, hoisted_b):
        if mma_smem_desc == "local_hoist":
            pv_a_local = SmemDescriptor()
            pv_a_local.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
            pv_b_local = SmemDescriptor()
            pv_b_local.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
        for mma_mi in T.unroll(1):
            for mma_ni in T.unroll(1):
                for mma_ki in T.unroll(4):
                    pv_a_offset: T.let = (
                        mma_ki % 4 * 1024 + mma_mi * 512 + mma_ki // 4 * 8 + s_offset
                    )
                    pv_b_offset: T.let = mma_ki * 1024 + mma_ni * 64 + v_offset
                    if mma_smem_desc == "recompute":
                        pv_a_ptr: T.let = T.ptr_byte_offset(
                            s_smem_gemm.ptr_to([0, 0]), pv_a_offset // 8 * 16, "bfloat16"
                        )
                        pv_b_ptr: T.let = T.ptr_byte_offset(
                            v_smem_gemm.ptr_to([0, 0]), pv_b_offset // 8 * 16, "bfloat16"
                        )
                        T.evaluate(
                            _mma_f16(
                                T.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                _recompute_smem_desc(pv_a_ptr, 0x00004008, 0x00400000),
                                _recompute_smem_desc(pv_b_ptr, 0x40004040, 0x04000000),
                                T.uint32(0x08410490),
                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                            )
                        )
                    elif mma_smem_desc == "encode":
                        pv_a_encode = SmemDescriptor()
                        pv_a_encode.init(
                            T.ptr_byte_offset(
                                s_smem_gemm.ptr_to([0, 0]), pv_a_offset // 8 * 16, "bfloat16"
                            ),
                            ldo=64,
                            sdo=8,
                            swizzle=0,
                        )
                        pv_b_encode = SmemDescriptor()
                        pv_b_encode.init(
                            T.ptr_byte_offset(
                                v_smem_gemm.ptr_to([0, 0]), pv_b_offset // 8 * 16, "bfloat16"
                            ),
                            ldo=1024,
                            sdo=64,
                            swizzle=3,
                        )
                        T.evaluate(
                            _mma_f16(
                                T.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                pv_a_encode.desc,
                                pv_b_encode.desc,
                                T.uint32(0x08410490),
                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                            )
                        )
                    elif mma_smem_desc == "local_hoist":
                        T.evaluate(
                            _mma_f16(
                                T.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                pv_a_local.add_16B_offset(pv_a_offset // 8),
                                pv_b_local.add_16B_offset(pv_b_offset // 8),
                                T.uint32(0x08410490),
                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                            )
                        )
                    else:
                        T.evaluate(
                            _mma_f16(
                                T.cast(o_tmem_col + dest_offset + mma_ni * 128, "uint32"),
                                _add_smem_desc_offset(hoisted_a, pv_a_offset // 8),
                                _add_smem_desc_offset(hoisted_b, pv_b_offset // 8),
                                T.uint32(0x08410490),
                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                            )
                        )

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:150-386.  Scale/exp warpgroup and epilogue.
        T.ptx.setmaxnreg.inc.sync.aligned.u32(144)
        mi: T.float32 = MAX_INIT_VAL
        li: T.float32 = 0.0
        real_mi: T.float32 = T.float32(-float("inf"))
        scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

        for k in T.serial(0, num_k_blocks, unroll=False):
            softmax_token = iket.range_start("h128-softmax-tile")
            cur_buf: T.let = k % NUM_BUFS
            cur_phase: T.let = (k // NUM_BUFS) & 1
            qk_wait_token = iket.range_start("h128-qk-wait")
            bar_qk_done.wait(cur_buf, cur_phase)
            iket.range_end(qk_wait_token)
            T.ptx.tcgen05.fence__after_thread_sync()

            p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, P_TMEM_COLS), "uint32")
            p = p_frag.local()
            T.evaluate(_tmem_load(p, T.uint32(tmem_p_col), P_TMEM_COLS))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.ptx.tcgen05.fence__before_thread_sync()
            bar_p_free.arrive(cur_buf, remote=T.uint32(0))

            bar_k_valid_ready.wait(cur_buf, cur_phase)
            valid_word_offset: T.let = T.if_then_else(
                idx_in_warpgroup >= 64, B_TOPK // 8 // 2 // 4, 0
            )
            is_k_valid_lo: T.uint32
            is_k_valid_hi: T.uint32
            T.ptx.ld.shared.u32(
                is_k_valid_lo, is_k_valid.view("uint32").ptr_to([cur_buf, valid_word_offset])
            )
            T.ptx.ld.shared.u32(
                is_k_valid_hi, is_k_valid.view("uint32").ptr_to([cur_buf, valid_word_offset + 1])
            )

            @T.inline
            def mask_p_half(valid_word, base):
                for p_i in T.unroll(P_TMEM_COLS // 2):
                    invalid_p_predicate: T.let = T.bitwise_and(
                        T.shift_right(valid_word, T.uint32(p_i)), T.uint32(1)
                    ) == T.uint32(0)
                    p[base + p_i] = T.if_then_else(
                        invalid_p_predicate, T.uint32(0xFF800000), p[base + p_i]
                    )

            mask_p_half(is_k_valid_lo, 0)
            mask_p_half(is_k_valid_hi, P_TMEM_COLS // 2)

            cur_pi_max: T.float32 = T.float32(-float("inf"))
            for p_i in T.unroll(P_TMEM_COLS):
                cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
            cur_pi_max = cur_pi_max * sm_scale_div_log2
            bar_k_valid_free.arrive(cur_buf)

            T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
            T.ptx.st.shared.f32(rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max)
            T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
            peer_pi_max: T.float32
            T.ptx.ld.shared.f32(peer_pi_max, rowwise_max_buf.ptr_to([idx_in_warpgroup ^ 64]))
            cur_pi_max = T.max(cur_pi_max, peer_pi_max)
            real_mi = T.max(real_mi, cur_pi_max)
            should_scale_o: T.bool = (
                T.cuda.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0
            )

            new_max: T.float32
            scale_for_old: T.float32
            if not should_scale_o:
                scale_for_old = 1.0
                new_max = mi
            else:
                new_max = T.max(cur_pi_max, mi)
                T.ptx.ex2.approx.ftz.f32(scale_for_old, mi - new_max)
            mi = new_max
            li = li * scale_for_old

            # S frag: warpgroup-distributed (B_H//2, B_TOPK) tile. Thread idx owns row h = idx%64
            # and k half [64*(idx//64), +64) in 8-elem chunks (from the packing loop below).
            s_frag = T.alloc_buffer(
                (B_H // 2, B_TOPK),
                "bfloat16",
                scope="local",
                layout=TileLayout(
                    S[(2, 32, 2, B_TOPK // 2) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                ),
            )
            s_pack = s_frag.local().view("uint32")
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            fma_pair: T.uint64
            for s_i in T.unroll(P_TMEM_COLS // 2):
                p_pair: T.let = T.cuda.make_float2(
                    T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                )
                T.ptx.fma.rn.f32x2(fma_pair, p_pair, scale_pair, neg_new_max_pair)
                s_x: T.float32
                s_y: T.float32
                T.ptx.ex2.approx.ftz.f32(s_x, T.cuda.float2_x(fma_pair))
                T.ptx.ex2.approx.ftz.f32(s_y, T.cuda.float2_y(fma_pair))
                li = li + s_x + s_y
                s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)

            if k > 0:
                prev_buf: T.let = (k - 1) % NUM_BUFS
                prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                pv_wait_token = iket.range_start("h128-pv-wait")
                bar_sv_done.wait(prev_buf, prev_phase)
                T.ptx.fence.proxy.async_.shared__cta()
                iket.range_end(pv_wait_token)

            s_base: T.let = idx_in_warpgroup // 64 * 4096 + idx_in_warpgroup % 64 * 8
            s_words = s_frag.local().view("uint32")
            for s_store_i in T.unroll(8):
                s_ptr: T.let = T.ptr_byte_offset(
                    s_smem_gemm.ptr_to([0, 0]), (s_base + s_store_i * 512) * BF16_BYTES, "bfloat16"
                )
                s_word: T.let = s_store_i * 4
                T.ptx.st.shared.v4.u32(
                    s_ptr,
                    s_words[s_word],
                    s_words[s_word + 1],
                    s_words[s_word + 2],
                    s_words[s_word + 3],
                )

            if (k > 0) & should_scale_o:
                T.ptx.tcgen05.fence__after_thread_sync()
                o_rescale_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32")
                o_rescale = o_rescale_frag.local()
                for chunk_idx in T.unroll((D_V // 2) // 32):
                    T.evaluate(
                        _tmem_load(
                            o_rescale,
                            T.cuda.get_tmem_addr(T.uint32(o_tmem_col), 0, chunk_idx * 32),
                            32,
                        )
                    )
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    for scale_i in T.unroll(32 // 2):
                        mul_f32x2(o_rescale, scale_i * 2, scale_for_old)
                    T.evaluate(
                        _tmem_store(
                            o_rescale, T.cuda.get_tmem_addr(T.uint32(o_tmem_col), 0, chunk_idx * 32)
                        )
                    )
                    T.ptx.tcgen05.wait__st.sync.aligned()
                T.ptx.tcgen05.fence__before_thread_sync()

            T.ptx.fence.proxy.async_.shared__cta()
            bar_so_ready.arrive(cur_buf, remote=T.uint32(0))
            iket.range_end(softmax_token)

        epilogue_token = iket.range_start("h128-output")
        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), li)
        T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
        peer_li: T.float32
        T.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup ^ 64]))
        li = li + peer_li

        if idx_in_warpgroup < B_H // 2:
            global_head: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
            cur_lse: T.float32
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma.rn.f32(cur_lse, mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            T.ptx.st.global_.f32(max_logits.ptr_to([s_q_idx, global_head]), real_mi * LN_2)
            T.ptx.st.global_.f32(lse.ptr_to([s_q_idx, global_head]), cur_lse)

        last_k: T.let = num_k_blocks - 1
        last_buf: T.let = last_k % NUM_BUFS
        last_phase: T.let = (last_k // NUM_BUFS) & 1
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.tcgen05.fence__after_thread_sync()

        attn_sink_log2: T.let = (
            T.cuda.ldg(
                attn_sink.ptr_to([cta_idx * (B_H // 2) + (idx_in_warpgroup % 64)]), "float32"
            )
            * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        sink_exp: T.float32
        T.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi)
        output_scale: T.float32 = T.cuda.fdividef(T.float32(1.0), li + sink_exp)
        o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "float32")
        o_epi = o_epi_frag.local()
        have_valid_indices: T.let = T.cuda.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(B_EPI):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "bfloat16")
        o_epi_bf16 = o_epi_bf16_frag.local()
        o_smem_win = o_smem.rearrange("h (b r) -> (b h) r", b=2)
        for epi_k in T.unroll((D_V // 2) // B_EPI):
            if have_valid_indices:
                T.evaluate(
                    _tmem_load(
                        o_epi, T.cuda.get_tmem_addr(T.uint32(o_tmem_col), 0, epi_k * B_EPI), B_EPI
                    )
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()
            for scale_i in T.unroll(B_EPI // 2):
                mul_f32x2(o_epi, scale_i * 2, output_scale)
            for cast_i in T.unroll(B_EPI // 2):
                T.evaluate(_cast_f32x2_bf16x2(o_epi_bf16, o_epi, cast_i * 2))
            o_epi_words = o_epi_bf16.view("uint32")
            for o_store_i in T.unroll(8):
                s_off: T.let = (
                    idx_in_warpgroup // 64 * 16384
                    + epi_k * 4096
                    + idx_in_warpgroup % 64 * 64
                    + T.bitwise_xor(
                        o_store_i * 8,
                        T.shift_left(
                            T.bitwise_and(
                                idx_in_warpgroup // 64 * 256 + epi_k * 64 + idx_in_warpgroup % 64, 7
                            ),
                            3,
                        ),
                    )
                )
                s_ptr: T.let = T.ptr_byte_offset(
                    o_smem_win.ptr_to([0, 0]), s_off * BF16_BYTES, "bfloat16"
                )
                o_word: T.let = o_store_i * 4
                T.ptx.st.shared.v4.u32(
                    s_ptr,
                    o_epi_words[o_word],
                    o_epi_words[o_word + 1],
                    o_epi_words[o_word + 2],
                    o_epi_words[o_word + 3],
                )

            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
            if warp_idx == 0:
                if T.cuda.elect_sync():
                    o_part0_offset: T.let = epi_k * B_EPI * (B_H // 2) * BF16_BYTES
                    T.evaluate(
                        T.ptx[_TMA_S2G_3D](
                            T.address_of(out_part0_tensormap),
                            T.cast(epi_k * B_EPI, "int32"),
                            T.cast(cta_idx * (B_H // 2), "int32"),
                            T.cast(s_q_idx, "int32"),
                            T.ptr_byte_offset(o_smem.ptr_to([0, 0]), o_part0_offset, "bfloat16"),
                        )
                    )
            if warp_idx == 1:
                if T.cuda.elect_sync():
                    epi_k2: T.let = epi_k + (D_V // B_EPI // 2)
                    T.evaluate(epi_k2)
                    o_part1_offset: T.let = (epi_k * B_EPI + D_V // 2) * (B_H // 2) * BF16_BYTES
                    T.evaluate(
                        T.ptx[_TMA_S2G_3D](
                            T.address_of(out_part1_tensormap),
                            T.cast(epi_k * B_EPI + D_V // 2, "int32"),
                            T.cast(cta_idx * (B_H // 2), "int32"),
                            T.cast(s_q_idx, "int32"),
                            T.ptr_byte_offset(o_smem.ptr_to([0, 0]), o_part1_offset, "bfloat16"),
                        )
                    )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(T.uint32(0), T.uint32(512))
        iket.range_end(epilogue_token)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:387-446.  K producer warpgroup.
        k_gather_token = iket.range_start("h128-k-load")
        T.ptx.setmaxnreg.dec.sync.aligned.u32(96)
        wg1_warp_idx: T.let = warp_idx - 4
        if T.cuda.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                indices_int4 = T.alloc_local((WG1_ROWS_PER_WARP, 4), "int32")
                max_indices: T.int32 = -1
                min_indices: T.int32 = s_kv

                # This CTA's topk half (cta_idx), split (local_row, warp, j): one
                # strided nc copy (auto-vectorizes to 4x v4 ld.global.nc), like head64.
                idx_block = indices.view(
                    s_q, stride_indices_s_q // B_TOPK, 2, WG1_ROWS_PER_WARP, WG1_NUM_WARPS, 4
                ).sub[s_q_idx, k, cta_idx, :, wg1_warp_idx, :]
                indices_words = indices_int4.view(16).view("uint32")
                for indices_load_i in T.unroll(4):
                    indices_word: T.let = indices_load_i * 4
                    T.ptx.ld.global_.nc.v4.u32(
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
                for local_row in T.unroll(WG1_ROWS_PER_WARP):
                    for j in T.unroll(4):
                        idx: T.let = indices_int4[local_row, j]
                        max_indices = T.max(max_indices, idx)
                        min_indices = T.min(min_indices, idx)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1

                @T.inline
                def gather_k_part(col_start, col_count, tx_dim, bar, tensormap):
                    if not should_skip_tma:
                        k_gather_tile = k_smem.sub[
                            :, col_start * 64 : col_start * 64 + col_count * 64
                        ].tile(0, (-1, WG1_NUM_WARPS, 4))[:, wg1_warp_idx, :]
                        for row_group in T.unroll(WG1_ROWS_PER_WARP):
                            for col_atom in T.unroll(col_count):
                                k_gather_offset: T.let = (
                                    (col_start + col_atom) * 64 * (B_TOPK // 2)
                                    + (wg1_warp_idx * 4 + row_group * 16) * 64
                                ) * BF16_BYTES
                                T.evaluate(
                                    T.ptx[_TMA_GATHER4_2D_CACHE](
                                        T.ptr_byte_offset(
                                            k_smem.ptr_to([0, 0]), k_gather_offset, "bfloat16"
                                        ),
                                        T.address_of(tensormap),
                                        T.cast((col_start + col_atom) * 64, "int32"),
                                        indices_int4[row_group, 0],
                                        indices_int4[row_group, 1],
                                        indices_int4[row_group, 2],
                                        indices_int4[row_group, 3],
                                        T.cuda.cvta_generic_to_shared(
                                            leader_mbar(bar.ptr_to([cur_buf]))
                                        ),
                                        _KV_TMA_CACHE_HINT,
                                    )
                                )
                    else:
                        _rem1 = T.alloc_local([1], "uint64")
                        T.ptx.mapa.shared__cluster.u64(_rem1[0], bar.ptr_to([cur_buf]), T.uint32(0))
                        T.ptx.mbarrier.complete_tx.relaxed.cluster.b64(
                            _rem1[0],
                            T.uint32(WG1_ROWS_PER_WARP * 4 * tx_dim * BF16_BYTES),
                            pred=T.uint32(1),
                        )

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_part_done.wait(prev_buf, prev_phase)
                gather_k_part(0, num_sq_tiles, d_sq, bar_k_part0_ready, kv_k_part0_tensormap)

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_done.wait(prev_buf, prev_phase)
                gather_k_part(
                    num_sq_tiles,
                    num_qk_tiles - num_sq_tiles,
                    D_TQ,
                    bar_k_part1_ready,
                    kv_k_part1_tensormap,
                )
        iket.range_end(k_gather_token)

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:447-489.  V producer warpgroup.
        v_gather_token = iket.range_start("h128-v-load")
        T.ptx.setmaxnreg.dec.sync.aligned.u32(96)
        wg2_warp_idx: T.let = warp_idx - 8
        if T.cuda.elect_sync():
            bar_prologue_utccp.wait(0, 0)
            for k in T.serial(0, num_k_blocks, unroll=False):
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1
                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_part_done.wait(prev_buf, prev_phase)

                @T.inline
                def gather_v_part(row_offset, part, token_buf, bar, tensormap):
                    # V loads all 128 tokens; the two parts map to an extent-2
                    # axis indexed by part. One strided nc copy, like head64.
                    idx_block = indices.view(
                        s_q, stride_indices_s_q // B_TOPK, 2, WG2_ROWS_PER_PART, WG2_NUM_WARPS, 4
                    ).sub[s_q_idx, k, part, :, wg2_warp_idx, :]
                    token_words = token_buf.view(16).view("uint32")
                    for token_load_i in T.unroll(4):
                        token_word: T.let = token_load_i * 4
                        T.ptx.ld.global_.nc.v4.u32(
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
                    src0: T.let = cta_idx * 256
                    v_gather_tile = v_smem_gemm.tile(0, (2, -1, WG2_NUM_WARPS, 4))[
                        part, :, wg2_warp_idx, :
                    ]
                    for row_group in T.unroll(WG2_ROWS_PER_PART):
                        for col_atom in T.unroll((D_V // 2) // 64):
                            v_gather_offset: T.let = (
                                part * (B_TOPK // 2) * 64
                                + col_atom * 64 * B_TOPK
                                + (wg2_warp_idx * 4 + row_group * 16) * 64
                            ) * BF16_BYTES
                            T.evaluate(
                                T.ptx[_TMA_GATHER4_2D_CACHE](
                                    T.ptr_byte_offset(
                                        v_smem.ptr_to([0, 0]), v_gather_offset, "bfloat16"
                                    ),
                                    T.address_of(tensormap),
                                    T.cast(src0 + col_atom * 64, "int32"),
                                    token_buf[row_group, 0],
                                    token_buf[row_group, 1],
                                    token_buf[row_group, 2],
                                    token_buf[row_group, 3],
                                    T.cuda.cvta_generic_to_shared(
                                        leader_mbar(bar.ptr_to([cur_buf]))
                                    ),
                                    _KV_TMA_CACHE_HINT,
                                )
                            )

                token_idxs_part0 = T.alloc_local((WG2_ROWS_PER_PART, 4), "int32")
                gather_v_part(0, 0, token_idxs_part0, bar_v_part0_ready, kv_v_part0_tensormap)

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_done.wait(prev_buf, prev_phase)
                token_idxs_part1 = T.alloc_local((WG2_ROWS_PER_PART, 4), "int32")
                gather_v_part(
                    WG2_ROWS_PER_PART, 1, token_idxs_part1, bar_v_part1_ready, kv_v_part1_tensormap
                )
        iket.range_end(v_gather_token)

    else:
        # CUDA phase1.cuh:490-606.  MMA warp and KV-valid loading warp.
        T.ptx.setmaxnreg.inc.sync.aligned.u32(168)
        if (cta_idx == 0) & (warp_idx == 12):
            mma_token = iket.range_start("h128-qk-pv-issue")
            if T.cuda.elect_sync():
                bar_prologue_q.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                bar_prologue_q.wait(0, 0)
                T.ptx.tcgen05.fence__after_thread_sync()
                for q_copy_flat in T.unroll(48):
                    q_copy_src: T.let = T.ptr_byte_offset(
                        q_full.ptr_to([0, 0]),
                        (d_sq * 8 + q_copy_flat % 6 * 512 + q_copy_flat // 6 % 8) * 16,
                        "bfloat16",
                    )
                    T.evaluate(
                        T.ptx[_TCGEN_CP_64X128](
                            T.cast(
                                q_tmem_col + q_copy_flat % 6 * 32 + q_copy_flat // 6 % 8 * 4,
                                "uint32",
                            ),
                            _replace_smem_desc_addr(q_cp_desc, q_copy_src),
                        )
                    )
                T.evaluate(
                    T.ptx[_TCGEN_COMMIT](
                        T.cuda.cvta_generic_to_shared(bar_prologue_utccp.ptr_to([0])), T.uint16(3)
                    )
                )

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.let = k % NUM_BUFS
                        cur_phase: T.let = (k // NUM_BUFS) & 1

                        bar_k_part0_ready.arrive(cur_buf, tx_count=B_TOPK * d_sq * BF16_BYTES)
                        bar_k_part0_ready.wait(cur_buf, cur_phase)
                        if k > 0:
                            prev_buf: T.let = (k - 1) % NUM_BUFS
                            prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                            bar_p_free.wait(prev_buf, prev_phase)
                        T.ptx.tcgen05.fence__after_thread_sync()

                        mma_p_accumulate = T.uint32(0)
                        if d_sq > 0:
                            sq_smem = q_full.sub[:, :d_sq]
                            if mma_smem_desc == "local_hoist":
                                qk_part0_a_local = SmemDescriptor()
                                qk_part0_a_local.init(
                                    sq_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                )
                                qk_part0_b_local = SmemDescriptor()
                                qk_part0_b_local.init(
                                    k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                )
                            if mma_smem_desc == "hoist":
                                qk_part0_a_hoist = SmemDescriptor()
                                qk_part0_a_hoist.init(
                                    sq_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3
                                )
                            for mma_mi in T.unroll(1):
                                for mma_ni in T.unroll(1):
                                    for mma_ki in T.unroll(d_sq // 16):
                                        qk_part0_offset: T.let = (
                                            mma_ki % (d_sq // 16) // 4 * 4096
                                            + mma_mi * 4096
                                            + mma_ki // (d_sq // 16) * 64
                                            + mma_ki % 4 * 16
                                        )
                                        if mma_smem_desc == "recompute":
                                            qk_part0_a_ptr: T.let = T.ptr_byte_offset(
                                                sq_smem.ptr_to([0, 0]),
                                                qk_part0_offset // 8 * 16,
                                                "bfloat16",
                                            )
                                            qk_part0_b_ptr: T.let = T.ptr_byte_offset(
                                                k_smem.ptr_to([0, 0]),
                                                qk_part0_offset // 8 * 16,
                                                "bfloat16",
                                            )
                                            T.evaluate(
                                                _mma_f16(
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    _recompute_smem_desc(
                                                        qk_part0_a_ptr, 0x40004040, 0x02000000
                                                    ),
                                                    _recompute_smem_desc(
                                                        qk_part0_b_ptr, 0x40004040, 0x02000000
                                                    ),
                                                    T.uint32(0x08200490),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            )
                                        elif mma_smem_desc == "encode":
                                            qk_part0_a_encode = SmemDescriptor()
                                            qk_part0_a_encode.init(
                                                T.ptr_byte_offset(
                                                    sq_smem.ptr_to([0, 0]),
                                                    qk_part0_offset // 8 * 16,
                                                    "bfloat16",
                                                ),
                                                ldo=512,
                                                sdo=64,
                                                swizzle=3,
                                            )
                                            qk_part0_b_encode = SmemDescriptor()
                                            qk_part0_b_encode.init(
                                                T.ptr_byte_offset(
                                                    k_smem.ptr_to([0, 0]),
                                                    qk_part0_offset // 8 * 16,
                                                    "bfloat16",
                                                ),
                                                ldo=512,
                                                sdo=64,
                                                swizzle=3,
                                            )
                                            T.evaluate(
                                                _mma_f16(
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    qk_part0_a_encode.desc,
                                                    qk_part0_b_encode.desc,
                                                    T.uint32(0x08200490),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            )
                                        elif mma_smem_desc == "local_hoist":
                                            T.evaluate(
                                                _mma_f16(
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    qk_part0_a_local.add_16B_offset(
                                                        qk_part0_offset // 8
                                                    ),
                                                    qk_part0_b_local.add_16B_offset(
                                                        qk_part0_offset // 8
                                                    ),
                                                    T.uint32(0x08200490),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            )
                                        else:
                                            T.evaluate(
                                                _mma_f16(
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    qk_part0_a_hoist.add_16B_offset(
                                                        qk_part0_offset // 8
                                                    ),
                                                    qk_k_part0_desc.add_16B_offset(
                                                        qk_part0_offset // 8
                                                    ),
                                                    T.uint32(0x08200490),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                )
                                            )
                            mma_p_accumulate = T.uint32(1)
                        bar_qk_part_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                        bar_k_part1_ready.arrive(
                            cur_buf, tx_count=B_TOPK * (d_qk - d_sq) * BF16_BYTES
                        )
                        bar_k_part1_ready.wait(cur_buf, cur_phase)
                        T.ptx.tcgen05.fence__after_thread_sync()

                        if mma_smem_desc == "local_hoist":
                            qk_part1_b_local = SmemDescriptor()
                            qk_part1_b_local.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)
                        for mma_mi in T.unroll(1):
                            for mma_ni in T.unroll(1):
                                for mma_ki in T.unroll(D_TQ // 16):
                                    qk_part1_offset: T.let = (
                                        mma_ki % (D_TQ // 16) // 4 * 4096
                                        + mma_ni * 4096
                                        + mma_ki // (D_TQ // 16) * 64
                                        + mma_ki % 4 * 16
                                        + d_sq // 64 * 4096
                                    )
                                    if mma_smem_desc == "recompute":
                                        qk_part1_b_ptr: T.let = T.ptr_byte_offset(
                                            k_smem.ptr_to([0, 0]),
                                            qk_part1_offset // 8 * 16,
                                            "bfloat16",
                                        )
                                        T.evaluate(
                                            _mma_f16(
                                                T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                T.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                _recompute_smem_desc(
                                                    qk_part1_b_ptr, 0x40004040, 0x02000000
                                                ),
                                                T.uint32(0x08200490),
                                                T.Or(mma_ki != 0, T.cast(mma_p_accumulate, "bool")),
                                            )
                                        )
                                    elif mma_smem_desc == "encode":
                                        qk_part1_b_encode = SmemDescriptor()
                                        qk_part1_b_encode.init(
                                            T.ptr_byte_offset(
                                                k_smem.ptr_to([0, 0]),
                                                qk_part1_offset // 8 * 16,
                                                "bfloat16",
                                            ),
                                            ldo=512,
                                            sdo=64,
                                            swizzle=3,
                                        )
                                        T.evaluate(
                                            _mma_f16(
                                                T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                T.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                qk_part1_b_encode.desc,
                                                T.uint32(0x08200490),
                                                T.Or(mma_ki != 0, T.cast(mma_p_accumulate, "bool")),
                                            )
                                        )
                                    elif mma_smem_desc == "local_hoist":
                                        T.evaluate(
                                            _mma_f16(
                                                T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                T.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                qk_part1_b_local.add_16B_offset(
                                                    qk_part1_offset // 8
                                                ),
                                                T.uint32(0x08200490),
                                                T.Or(mma_ki != 0, T.cast(mma_p_accumulate, "bool")),
                                            )
                                        )
                                    else:
                                        T.evaluate(
                                            _mma_f16(
                                                T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                T.cast(q_tmem_col + mma_ki * 8, "uint32"),
                                                qk_k_part1_desc.add_16B_offset(
                                                    qk_part1_offset // 8
                                                ),
                                                T.uint32(0x08200490),
                                                T.Or(mma_ki != 0, T.cast(mma_p_accumulate, "bool")),
                                            )
                                        )
                        mma_p_accumulate = T.uint32(1)
                        bar_qk_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                    if k > 0:
                        cur_buf_prev: T.let = (k - 1) % NUM_BUFS
                        cur_phase_prev: T.let = ((k - 1) // NUM_BUFS) & 1
                        bar_so_ready.wait(cur_buf_prev, cur_phase_prev)

                        bar_v_part0_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part0_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence__after_thread_sync()
                        mma_o_accumulate = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))
                        if mma_smem_desc == "hoist":
                            issue_pv_mma(0, 0, 0, pv_a_part0_lo_desc.desc, pv_b_part0_lo_desc.desc)
                            issue_pv_mma(
                                128, 0, 16384, pv_a_part0_hi_desc.desc, pv_b_part0_hi_desc.desc
                            )
                        else:
                            issue_pv_mma(0, 0, 0, T.uint64(0), T.uint64(0))
                            issue_pv_mma(128, 0, 16384, T.uint64(0), T.uint64(0))
                        mma_o_accumulate = T.uint32(1)
                        bar_sv_part_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

                        bar_v_part1_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part1_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence__after_thread_sync()
                        if mma_smem_desc == "hoist":
                            issue_pv_mma(
                                0, 4096, 4096, pv_a_part1_lo_desc.desc, pv_b_part1_lo_desc.desc
                            )
                            issue_pv_mma(
                                128, 4096, 20480, pv_a_part1_hi_desc.desc, pv_b_part1_hi_desc.desc
                            )
                        else:
                            issue_pv_mma(0, 4096, 4096, T.uint64(0), T.uint64(0))
                            issue_pv_mma(128, 4096, 20480, T.uint64(0), T.uint64(0))
                        mma_o_accumulate = T.uint32(1)
                        bar_sv_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)
            iket.range_end(mma_token)

        elif warp_idx == 13:
            valid_mask_token = iket.range_start("h128-valid-mask")
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
                    row_base: T.let = g_indices_base + k * B_TOPK + lane_idx * 8
                    lane_index_words = lane_indices.view("uint32")
                    T.evaluate(
                        T.ptx["ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.u32"](
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
                    )
                    abs_pos_start: T.let = k * B_TOPK
                    is_ks_valid_mask: T.let = pack_valid_mask8(
                        lane_indices, abs_pos_start, lane_idx, topk_len, s_kv
                    )
                    cur_buf: T.let = k % NUM_BUFS
                    cur_phase: T.let = (k // NUM_BUFS) & 1
                    bar_k_valid_free.wait(cur_buf, cur_phase ^ 1)
                    T.ptx.st.shared.b8(
                        is_k_valid.ptr_to([cur_buf, lane_idx]),
                        T.reinterpret("uint8", is_ks_valid_mask),
                    )
                    bar_k_valid_ready.arrive(cur_buf)
            iket.range_end(valid_mask_token)


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    kernel = _kernel.specialize(
        s_q=cfg.s_q,
        s_kv=cfg.s_kv,
        topk=cfg.topk,
        d_qk=cfg.d_qk,
        h_q=cfg.h_q,
        stride_kv_s_kv=stride_kv_s_kv,
        stride_indices_s_q=stride_indices_s_q,
        have_attn_sink=cfg.have_attn_sink,
        have_topk_length=cfg.have_topk_length,
        sm_scale_div_log2=(1.0 / math.sqrt(cfg.d_qk)) * LOG_2_E,
    )
    kernel = kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
    return kernel


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
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

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

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
