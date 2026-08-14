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

from tirx_kernels.flashmla.utils._ir_builder import (
    IketProfiler,
    MBarrier,
    SmemDescriptor,
    TCGen05Bar,
    TMABar,
)
from tirx_kernels.flashmla.utils._ir_builder import builder_alloc_scalar as _builder_alloc_scalar
from tirx_kernels.flashmla.utils._ir_builder import builder_assign as _builder_assign
from tirx_kernels.flashmla.utils._ir_builder import builder_bind as _builder_bind
from tirx_kernels.flashmla.utils._ir_builder import builder_emit as _builder_emit
from tirx_kernels.flashmla.utils._ir_builder import builder_enter as _builder_enter
from tirx_kernels.flashmla.utils._ir_builder import builder_scalar as _builder_scalar
from tirx_kernels.flashmla.utils._mask import pack_valid_mask8
from tirx_kernels.flashmla.utils._tma import leader_mbar
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
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


def mul_f32x2(values, idx, multiplier):
    packed = _builder_alloc_scalar("packed", "uint64")
    rhs = _builder_alloc_scalar("rhs", "uint64")
    _builder_emit(T.ptx.mov.b64(packed, values[idx], values[idx + 1]))
    _builder_emit(T.ptx.mov.b64(rhs, multiplier, multiplier))
    _builder_emit(T.ptx.mul.rz.ftz.f32x2(packed, packed, rhs))
    _builder_emit(T.ptx.mov.b64(values[idx], values[idx + 1], packed))


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


def _build_kernel(
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
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_kernel")
            q = T.arg("q", T.buffer((s_q, h_q, d_qk), "bfloat16"))
            kv = T.arg("kv", T.buffer((s_kv * stride_kv_s_kv,), "bfloat16"))
            indices = T.arg("indices", T.buffer((s_q * stride_indices_s_q,), "int32"))
            attn_sink = T.arg("attn_sink", T.buffer((h_q,), "float32"))
            topk_length = T.arg("topk_length", T.buffer((s_q,), "int32"))
            out = T.arg("out", T.buffer((s_q, h_q, D_V), "bfloat16"))
            max_logits = T.arg("max_logits", T.buffer((s_q, h_q), "float32"))
            lse = T.arg("lse", T.buffer((s_q, h_q), "float32"))
            kv_v_part1_tensormap = _builder_bind(
                "kv_v_part1_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
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
            )
            kv_v_part0_tensormap = _builder_bind(
                "kv_v_part0_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
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
            )
            kv_k_part1_tensormap = _builder_bind(
                "kv_k_part1_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
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
            )
            kv_k_part0_tensormap = _builder_bind(
                "kv_k_part0_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
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
            )
            out_part1_tensormap = _builder_bind(
                "out_part1_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
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
            )
            out_part0_tensormap = _builder_bind(
                "out_part0_tensormap",
                T.tvm_stack_alloca("tensormap", 1),
                type_annotation=T.TensorMap(),
            )
            _builder_emit(
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
            )
            q_tensormap = _builder_bind(
                "q_tensormap", T.tvm_stack_alloca("tensormap", 1), type_annotation=T.TensorMap()
            )
            _builder_emit(
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
            )
            _builder_emit(T.device_entry())
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            iket = _builder_assign("iket", IketProfiler())
            block_idx = _builder_assign("block_idx", T.cta_id([2 * s_q]))
            _builder_emit(T.cta_id_in_cluster([2]))
            cta_idx = _builder_bind("cta_idx", block_idx % 2)
            s_q_idx = _builder_bind("s_q_idx", block_idx // 2)
            thread_idx = _builder_assign("thread_idx", T.thread_id([NUM_THREADS]))
            _builder_emit(T.warpgroup_id([NUM_THREADS // 128]))
            _builder_emit(T.warp_id_in_wg([4]))
            _builder_emit(T.lane_id([32]))
            _builder_emit(T.thread_id_in_wg([128]))
            warp_idx = _builder_bind(
                "warp_idx", T.cuda.__shfl_sync(T.uint32(4294967295), thread_idx // 32, 0, 32)
            )
            lane_idx = _builder_bind("lane_idx", thread_idx % 32)
            topk_len = _builder_bind(
                "topk_len",
                T.cuda.ldg(topk_length.ptr_to([s_q_idx]), "int32") if have_topk_length else topk,
            )
            num_k_blocks = _builder_bind(
                "num_k_blocks", T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
            )
            warpgroup_idx = _builder_bind(
                "warpgroup_idx", T.cuda.__shfl_sync(T.uint32(4294967295), thread_idx // 128, 0, 32)
            )
            idx_in_warpgroup = _builder_bind("idx_in_warpgroup", thread_idx % 128)
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_tensormap)))
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(out_part0_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(out_part1_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_k_part0_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_k_part1_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_v_part0_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_v_part1_tensormap))
                                )
                            )
            d_sq = d_qk - D_TQ
            num_sq_tiles = (d_qk - D_TQ) // 64
            num_qk_tiles = d_qk // 64
            mma_smem_desc = (
                "recompute"
                if d_qk == 512 and s_kv == 8192
                else "local_hoist"
                if d_qk == 576 and s_kv != 65536
                else "hoist"
                if (d_qk == 512 and s_kv == 32768) or (d_qk == 576 and s_kv == 65536)
                else "encode"
            )
            pool = _builder_assign("pool", T.SMEMPool())
            u_base = pool.offset
            q_full = _builder_assign(
                "q_full", pool.alloc_tcgen05_mma_AB((B_H // 2, d_qk), "bfloat16")
            )
            q_cp_desc = _builder_alloc_scalar("q_cp_desc", "uint64")
            _builder_emit(
                T.cuda.tcgen05.encode_matrix_descriptor(
                    T.address_of(q_cp_desc), T.reinterpret(T.handle().ty, T.uint64(0)), 0, 64, 3
                )
            )
            _builder_emit(pool.move_base_to(u_base + B_H // 2 * d_sq * BF16_BYTES))
            v_smem = _builder_assign(
                "v_smem", pool.alloc_tcgen05_mma_AB((D_V // 2, B_TOPK), "bfloat16")
            )
            k_smem = _builder_assign(
                "k_smem", pool.alloc_tcgen05_mma_AB((B_TOPK // 2, d_qk), "bfloat16")
            )
            u_end = pool.offset
            _builder_emit(pool.move_base_to(u_base))
            o_smem = _builder_assign(
                "o_smem", pool.alloc_tcgen05_mma_AB((B_H // 2, D_V), "bfloat16")
            )
            _builder_emit(pool.move_base_to(u_end))
            s_smem_gemm = _builder_assign(
                "s_smem_gemm",
                pool.alloc_tcgen05_mma_AB(
                    (B_H // 2, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
                ),
            )
            is_k_valid = _builder_assign("is_k_valid", pool.alloc((NUM_BUFS, B_TOPK // 8), "int8"))
            bar_prologue_q = _builder_assign("bar_prologue_q", TMABar(pool, 1))
            bar_prologue_utccp = _builder_assign("bar_prologue_utccp", TCGen05Bar(pool, 1))
            bar_qk_part_done = _builder_assign("bar_qk_part_done", TCGen05Bar(pool, NUM_BUFS))
            bar_qk_done = _builder_assign("bar_qk_done", TCGen05Bar(pool, NUM_BUFS))
            bar_sv_part_done = _builder_assign("bar_sv_part_done", TCGen05Bar(pool, NUM_BUFS))
            bar_sv_done = _builder_assign("bar_sv_done", TCGen05Bar(pool, NUM_BUFS))
            bar_k_part0_ready = _builder_assign("bar_k_part0_ready", TMABar(pool, NUM_BUFS))
            bar_k_part1_ready = _builder_assign("bar_k_part1_ready", TMABar(pool, NUM_BUFS))
            bar_v_part0_ready = _builder_assign("bar_v_part0_ready", TMABar(pool, NUM_BUFS))
            bar_v_part1_ready = _builder_assign("bar_v_part1_ready", TMABar(pool, NUM_BUFS))
            bar_p_free = _builder_assign("bar_p_free", MBarrier(pool, NUM_BUFS))
            bar_so_ready = _builder_assign("bar_so_ready", MBarrier(pool, NUM_BUFS))
            bar_k_valid_ready = _builder_assign("bar_k_valid_ready", MBarrier(pool, NUM_BUFS))
            bar_k_valid_free = _builder_assign("bar_k_valid_free", MBarrier(pool, NUM_BUFS))
            tmem_start_addr = _builder_assign(
                "tmem_start_addr", pool.alloc((1,), "uint32", align=4)
            )
            rowwise_max_buf = _builder_assign("rowwise_max_buf", pool.alloc((128,), "float32"))
            rowwise_li_buf = _builder_assign("rowwise_li_buf", pool.alloc((128,), "float32"))
            _builder_emit(pool.commit())
            kv_tma = _builder_assign(
                "kv_tma",
                kv.view(s_kv, d_qk, layout=TileLayout(S[(s_kv, d_qk) : (stride_kv_s_kv, 1)])),
            )
            g_indices_base = _builder_bind("g_indices_base", s_q_idx * stride_indices_s_q)
            mma_p_accumulate = _builder_scalar("mma_p_accumulate", 0, dtype="uint32")
            mma_o_accumulate = _builder_scalar("mma_o_accumulate", 0, dtype="uint32")
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(bar_prologue_q.init(1))
                            _builder_emit(bar_prologue_utccp.init(1))
                            with T.unroll(NUM_BUFS) as init_stage:
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_qk_part_done.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_qk_done.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_sv_part_done.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_sv_done.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_k_part0_ready.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_k_part1_ready.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_v_part0_ready.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_v_part1_ready.ptr_to([init_stage]), T.uint32(1)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_p_free.ptr_to([init_stage]), T.uint32(128 * 2)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_so_ready.ptr_to([init_stage]), T.uint32(128 * 2)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_k_valid_ready.ptr_to([init_stage]), T.uint32(16)
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mbarrier.init.shared.b64(
                                        bar_k_valid_free.ptr_to([init_stage]), T.uint32(128)
                                    )
                                )
                            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(T.cuda.cluster_sync())
            with T.If(warp_idx == 0):
                with T.Then():
                    prologue_token = _builder_assign(
                        "prologue_token", iket.range_start("h128-q-load")
                    )
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx[_TMA_G2S_4D_CACHE](
                                        q_full.ptr_to([0, 0]),
                                        T.address_of(q_tensormap),
                                        T.int32(0),
                                        T.cast(cta_idx * (B_H // 2), "int32"),
                                        T.int32(0),
                                        T.cast(s_q_idx, "int32"),
                                        T.cuda.cvta_generic_to_shared(
                                            leader_mbar(bar_prologue_q.ptr_to([0]))
                                        ),
                                        _Q_TMA_CACHE_HINT,
                                    )
                                )
                            )
                    _builder_emit(
                        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                            T.address_of(tmem_start_addr[0]), T.uint32(512)
                        )
                    )
                    allocated_tmem_start = _builder_alloc_scalar("allocated_tmem_start", "uint32")
                    _builder_emit(
                        T.ptx.ld.shared.u32(allocated_tmem_start, tmem_start_addr.ptr_to([0]))
                    )
                    _builder_emit(
                        T.cuda.trap_when_assert_failed(allocated_tmem_start == T.uint32(0))
                    )
                    _builder_emit(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned())
                    _builder_emit(iket.range_end(prologue_token))
            _builder_emit(T.cuda.cta_sync())
            tmem_pool = _builder_assign(
                "tmem_pool",
                T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_start_addr),
            )
            o_tmem_col = tmem_pool.offset
            o_tmem = _builder_assign(
                "o_tmem",
                tmem_pool.alloc_tcgen05_mma_D(
                    (B_H // 2, D_V), "float32", M=128, cta_group=2, group=(2, 2, 128)
                ),
            )
            tmem_o_lo = _builder_assign("tmem_o_lo", o_tmem.sub[:, 0 : D_V // 2])
            tmem_o_hi = _builder_assign("tmem_o_hi", o_tmem.sub[:, D_V // 2 : D_V])
            o_win = _builder_assign(
                "o_win", o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
            )
            tmem_p_col = tmem_pool.offset
            tmem_p = _builder_assign(
                "tmem_p",
                tmem_pool.alloc_tcgen05_mma_D((B_H // 2, B_TOPK), "float32", M=128, cta_group=2),
            )
            q_tmem_col = tmem_pool.offset
            q_tmem = _builder_assign(
                "q_tmem",
                tmem_pool.alloc_tcgen05_mma_A((B_H // 2, D_TQ), "bfloat16", M=128, cta_group=2),
            )
            v_smem_gemm = _builder_assign(
                "v_smem_gemm", v_smem.rearrange("(x r) (z kl) -> r (z x kl)", x=2, z=2, kl=64)
            )
            if mma_smem_desc == "hoist":
                qk_k_part0_desc = _builder_assign("qk_k_part0_desc", SmemDescriptor())
                _builder_emit(
                    qk_k_part0_desc.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)
                )
                qk_k_part1_desc = _builder_assign("qk_k_part1_desc", SmemDescriptor())
                _builder_emit(
                    qk_k_part1_desc.init(k_smem.ptr_to([0, 0]), ldo=512, sdo=64, swizzle=3)
                )
                pv_a_part0_lo_desc = _builder_assign("pv_a_part0_lo_desc", SmemDescriptor())
                _builder_emit(
                    pv_a_part0_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
                )
                pv_a_part0_hi_desc = _builder_assign("pv_a_part0_hi_desc", SmemDescriptor())
                _builder_emit(
                    pv_a_part0_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
                )
                pv_a_part1_lo_desc = _builder_assign("pv_a_part1_lo_desc", SmemDescriptor())
                _builder_emit(
                    pv_a_part1_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
                )
                pv_a_part1_hi_desc = _builder_assign("pv_a_part1_hi_desc", SmemDescriptor())
                _builder_emit(
                    pv_a_part1_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
                )
                pv_b_part0_lo_desc = _builder_assign("pv_b_part0_lo_desc", SmemDescriptor())
                _builder_emit(
                    pv_b_part0_lo_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                pv_b_part0_hi_desc = _builder_assign("pv_b_part0_hi_desc", SmemDescriptor())
                _builder_emit(
                    pv_b_part0_hi_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                pv_b_part1_lo_desc = _builder_assign("pv_b_part1_lo_desc", SmemDescriptor())
                _builder_emit(
                    pv_b_part1_lo_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                )
                pv_b_part1_hi_desc = _builder_assign("pv_b_part1_hi_desc", SmemDescriptor())
                _builder_emit(
                    pv_b_part1_hi_desc.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                )

            def issue_pv_mma(dest_offset, s_offset, v_offset, hoisted_a, hoisted_b):
                if mma_smem_desc == "local_hoist":
                    pv_a_local = _builder_assign("pv_a_local", SmemDescriptor())
                    _builder_emit(
                        pv_a_local.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
                    )
                    pv_b_local = _builder_assign("pv_b_local", SmemDescriptor())
                    _builder_emit(
                        pv_b_local.init(v_smem_gemm.ptr_to([0, 0]), ldo=1024, sdo=64, swizzle=3)
                    )
                with T.unroll(1) as mma_mi:
                    with T.unroll(1) as mma_ni:
                        with T.unroll(4) as mma_ki:
                            pv_a_offset = _builder_bind(
                                "pv_a_offset",
                                mma_ki % 4 * 1024 + mma_mi * 512 + mma_ki // 4 * 8 + s_offset,
                            )
                            pv_b_offset = _builder_bind(
                                "pv_b_offset", mma_ki * 1024 + mma_ni * 64 + v_offset
                            )
                            if mma_smem_desc == "recompute":
                                pv_a_ptr = _builder_bind(
                                    "pv_a_ptr",
                                    T.ptr_byte_offset(
                                        s_smem_gemm.ptr_to([0, 0]),
                                        pv_a_offset // 8 * 16,
                                        "bfloat16",
                                    ),
                                )
                                pv_b_ptr = _builder_bind(
                                    "pv_b_ptr",
                                    T.ptr_byte_offset(
                                        v_smem_gemm.ptr_to([0, 0]),
                                        pv_b_offset // 8 * 16,
                                        "bfloat16",
                                    ),
                                )
                                _builder_emit(
                                    T.evaluate(
                                        _mma_f16(
                                            T.cast(
                                                o_tmem_col + dest_offset + mma_ni * 128, "uint32"
                                            ),
                                            _recompute_smem_desc(pv_a_ptr, 16392, 4194304),
                                            _recompute_smem_desc(pv_b_ptr, 1073758272, 67108864),
                                            T.uint32(138478736),
                                            T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                        )
                                    )
                                )
                            elif mma_smem_desc == "encode":
                                pv_a_encode = _builder_assign("pv_a_encode", SmemDescriptor())
                                _builder_emit(
                                    pv_a_encode.init(
                                        T.ptr_byte_offset(
                                            s_smem_gemm.ptr_to([0, 0]),
                                            pv_a_offset // 8 * 16,
                                            "bfloat16",
                                        ),
                                        ldo=64,
                                        sdo=8,
                                        swizzle=0,
                                    )
                                )
                                pv_b_encode = _builder_assign("pv_b_encode", SmemDescriptor())
                                _builder_emit(
                                    pv_b_encode.init(
                                        T.ptr_byte_offset(
                                            v_smem_gemm.ptr_to([0, 0]),
                                            pv_b_offset // 8 * 16,
                                            "bfloat16",
                                        ),
                                        ldo=1024,
                                        sdo=64,
                                        swizzle=3,
                                    )
                                )
                                _builder_emit(
                                    T.evaluate(
                                        _mma_f16(
                                            T.cast(
                                                o_tmem_col + dest_offset + mma_ni * 128, "uint32"
                                            ),
                                            pv_a_encode.desc,
                                            pv_b_encode.desc,
                                            T.uint32(138478736),
                                            T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                        )
                                    )
                                )
                            elif mma_smem_desc == "local_hoist":
                                _builder_emit(
                                    T.evaluate(
                                        _mma_f16(
                                            T.cast(
                                                o_tmem_col + dest_offset + mma_ni * 128, "uint32"
                                            ),
                                            pv_a_local.add_16B_offset(pv_a_offset // 8),
                                            pv_b_local.add_16B_offset(pv_b_offset // 8),
                                            T.uint32(138478736),
                                            T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                        )
                                    )
                                )
                            else:
                                _builder_emit(
                                    T.evaluate(
                                        _mma_f16(
                                            T.cast(
                                                o_tmem_col + dest_offset + mma_ni * 128, "uint32"
                                            ),
                                            _add_smem_desc_offset(hoisted_a, pv_a_offset // 8),
                                            _add_smem_desc_offset(hoisted_b, pv_b_offset // 8),
                                            T.uint32(138478736),
                                            T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                        )
                                    )
                                )

            with T.If(warpgroup_idx == 0):
                with T.Then():
                    _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(144))
                    mi = _builder_scalar("mi", MAX_INIT_VAL, dtype="float32")
                    li = _builder_scalar("li", 0.0, dtype="float32")
                    real_mi = _builder_scalar("real_mi", T.float32(-float("inf")), dtype="float32")
                    scale_pair = _builder_bind(
                        "scale_pair", T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
                    )
                    with T.serial(0, num_k_blocks, unroll=False) as k:
                        softmax_token = _builder_assign(
                            "softmax_token", iket.range_start("h128-softmax-tile")
                        )
                        cur_buf = _builder_bind("cur_buf", k % NUM_BUFS)
                        cur_phase = _builder_bind("cur_phase", k // NUM_BUFS & 1)
                        qk_wait_token = _builder_assign(
                            "qk_wait_token", iket.range_start("h128-qk-wait")
                        )
                        _builder_emit(bar_qk_done.wait(cur_buf, cur_phase))
                        _builder_emit(iket.range_end(qk_wait_token))
                        _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                        p_frag = _builder_assign(
                            "p_frag",
                            T.alloc_tcgen05_ldst_frag("32x32b", (128, P_TMEM_COLS), "uint32"),
                        )
                        p = _builder_assign("p", p_frag.local())
                        _builder_emit(T.evaluate(_tmem_load(p, T.uint32(tmem_p_col), P_TMEM_COLS)))
                        _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                        _builder_emit(T.ptx.tcgen05.fence__before_thread_sync())
                        _builder_emit(bar_p_free.arrive(cur_buf, remote=T.uint32(0)))
                        _builder_emit(bar_k_valid_ready.wait(cur_buf, cur_phase))
                        valid_word_offset = _builder_bind(
                            "valid_word_offset",
                            T.if_then_else(idx_in_warpgroup >= 64, B_TOPK // 8 // 2 // 4, 0),
                        )
                        is_k_valid_lo = _builder_alloc_scalar("is_k_valid_lo", "uint32")
                        is_k_valid_hi = _builder_alloc_scalar("is_k_valid_hi", "uint32")
                        _builder_emit(
                            T.ptx.ld.shared.u32(
                                is_k_valid_lo,
                                is_k_valid.view("uint32").ptr_to([cur_buf, valid_word_offset]),
                            )
                        )
                        _builder_emit(
                            T.ptx.ld.shared.u32(
                                is_k_valid_hi,
                                is_k_valid.view("uint32").ptr_to([cur_buf, valid_word_offset + 1]),
                            )
                        )

                        def mask_p_half(valid_word, base):
                            with T.unroll(P_TMEM_COLS // 2) as p_i:
                                invalid_p_predicate = _builder_bind(
                                    "invalid_p_predicate",
                                    T.bitwise_and(
                                        T.shift_right(valid_word, T.uint32(p_i)), T.uint32(1)
                                    )
                                    == T.uint32(0),
                                )
                                T.buffer_store(
                                    p,
                                    T.if_then_else(
                                        invalid_p_predicate, T.uint32(4286578688), p[base + p_i]
                                    ),
                                    [base + p_i],
                                )

                        _builder_emit(mask_p_half(is_k_valid_lo, 0))
                        _builder_emit(mask_p_half(is_k_valid_hi, P_TMEM_COLS // 2))
                        cur_pi_max = _builder_scalar(
                            "cur_pi_max", T.float32(-float("inf")), dtype="float32"
                        )
                        with T.unroll(P_TMEM_COLS) as p_i:
                            T.buffer_store(
                                cur_pi_max.buffer,
                                T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i])),
                                [0],
                            )
                        T.buffer_store(cur_pi_max.buffer, cur_pi_max * sm_scale_div_log2, [0])
                        _builder_emit(bar_k_valid_free.arrive(cur_buf))
                        _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                        _builder_emit(
                            T.ptx.st.shared.f32(
                                rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max
                            )
                        )
                        _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                        peer_pi_max = _builder_alloc_scalar("peer_pi_max", "float32")
                        _builder_emit(
                            T.ptx.ld.shared.f32(
                                peer_pi_max, rowwise_max_buf.ptr_to([idx_in_warpgroup ^ 64])
                            )
                        )
                        T.buffer_store(cur_pi_max.buffer, T.max(cur_pi_max, peer_pi_max), [0])
                        T.buffer_store(real_mi.buffer, T.max(real_mi, cur_pi_max), [0])
                        should_scale_o = _builder_scalar(
                            "should_scale_o",
                            T.cuda.any_sync(T.uint32(4294967295), cur_pi_max - mi > 6.0) != 0,
                            dtype="bool",
                        )
                        new_max = _builder_alloc_scalar("new_max", "float32")
                        scale_for_old = _builder_alloc_scalar("scale_for_old", "float32")
                        with T.If(T.Not(should_scale_o)):
                            with T.Then():
                                T.buffer_store(scale_for_old.buffer, 1.0, [0])
                                T.buffer_store(new_max.buffer, mi, [0])
                            with T.Else():
                                T.buffer_store(new_max.buffer, T.max(cur_pi_max, mi), [0])
                                _builder_emit(T.ptx.ex2.approx.ftz.f32(scale_for_old, mi - new_max))
                        T.buffer_store(mi.buffer, new_max, [0])
                        T.buffer_store(li.buffer, li * scale_for_old, [0])
                        s_frag = _builder_assign(
                            "s_frag",
                            T.alloc_buffer(
                                (B_H // 2, B_TOPK),
                                "bfloat16",
                                scope="local",
                                layout=TileLayout(
                                    S[
                                        (2, 32, 2, B_TOPK // 2) : (
                                            1 @ wid_in_wg,
                                            1 @ laneid,
                                            2 @ wid_in_wg,
                                            1,
                                        )
                                    ]
                                ),
                            ),
                        )
                        s_pack = _builder_assign("s_pack", s_frag.local().view("uint32"))
                        neg_new_max_pair = _builder_bind(
                            "neg_new_max_pair", T.cuda.make_float2(-new_max, -new_max)
                        )
                        fma_pair = _builder_alloc_scalar("fma_pair", "uint64")
                        with T.unroll(P_TMEM_COLS // 2) as s_i:
                            p_pair = _builder_bind(
                                "p_pair",
                                T.cuda.make_float2(
                                    T.cuda.uint_as_float(p[s_i * 2]),
                                    T.cuda.uint_as_float(p[s_i * 2 + 1]),
                                ),
                            )
                            _builder_emit(
                                T.ptx.fma.rn.f32x2(fma_pair, p_pair, scale_pair, neg_new_max_pair)
                            )
                            s_x = _builder_alloc_scalar("s_x", "float32")
                            s_y = _builder_alloc_scalar("s_y", "float32")
                            _builder_emit(T.ptx.ex2.approx.ftz.f32(s_x, T.cuda.float2_x(fma_pair)))
                            _builder_emit(T.ptx.ex2.approx.ftz.f32(s_y, T.cuda.float2_y(fma_pair)))
                            T.buffer_store(li.buffer, li + s_x + s_y, [0])
                            T.buffer_store(s_pack, T.cuda.float22bfloat162_rn(s_x, s_y), [s_i])
                        with T.If(k > 0):
                            with T.Then():
                                prev_buf = _builder_bind("prev_buf", (k - 1) % NUM_BUFS)
                                prev_phase = _builder_bind("prev_phase", (k - 1) // NUM_BUFS & 1)
                                pv_wait_token = _builder_assign(
                                    "pv_wait_token", iket.range_start("h128-pv-wait")
                                )
                                _builder_emit(bar_sv_done.wait(prev_buf, prev_phase))
                                _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                                _builder_emit(iket.range_end(pv_wait_token))
                        s_base = _builder_bind(
                            "s_base", idx_in_warpgroup // 64 * 4096 + idx_in_warpgroup % 64 * 8
                        )
                        s_words = _builder_assign("s_words", s_frag.local().view("uint32"))
                        with T.unroll(8) as s_store_i:
                            s_ptr = _builder_bind(
                                "s_ptr",
                                T.ptr_byte_offset(
                                    s_smem_gemm.ptr_to([0, 0]),
                                    (s_base + s_store_i * 512) * BF16_BYTES,
                                    "bfloat16",
                                ),
                            )
                            s_word = _builder_bind("s_word", s_store_i * 4)
                            _builder_emit(
                                T.ptx.st.shared.v4.u32(
                                    s_ptr,
                                    s_words[s_word],
                                    s_words[s_word + 1],
                                    s_words[s_word + 2],
                                    s_words[s_word + 3],
                                )
                            )
                        with T.If((k > 0) & should_scale_o):
                            with T.Then():
                                _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                o_rescale_frag = _builder_assign(
                                    "o_rescale_frag",
                                    T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32"),
                                )
                                o_rescale = _builder_assign("o_rescale", o_rescale_frag.local())
                                with T.unroll(D_V // 2 // 32) as chunk_idx:
                                    _builder_emit(
                                        T.evaluate(
                                            _tmem_load(
                                                o_rescale,
                                                T.cuda.get_tmem_addr(
                                                    T.uint32(o_tmem_col), 0, chunk_idx * 32
                                                ),
                                                32,
                                            )
                                        )
                                    )
                                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                                    with T.unroll(32 // 2) as scale_i:
                                        _builder_emit(
                                            mul_f32x2(o_rescale, scale_i * 2, scale_for_old)
                                        )
                                    _builder_emit(
                                        T.evaluate(
                                            _tmem_store(
                                                o_rescale,
                                                T.cuda.get_tmem_addr(
                                                    T.uint32(o_tmem_col), 0, chunk_idx * 32
                                                ),
                                            )
                                        )
                                    )
                                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                                _builder_emit(T.ptx.tcgen05.fence__before_thread_sync())
                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                        _builder_emit(bar_so_ready.arrive(cur_buf, remote=T.uint32(0)))
                        _builder_emit(iket.range_end(softmax_token))
                    epilogue_token = _builder_assign(
                        "epilogue_token", iket.range_start("h128-output")
                    )
                    with T.If(real_mi == T.float32(-float("inf"))):
                        with T.Then():
                            T.buffer_store(li.buffer, 0.0, [0])
                            T.buffer_store(mi.buffer, T.float32(-float("inf")), [0])
                    _builder_emit(
                        T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), li)
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                    peer_li = _builder_alloc_scalar("peer_li", "float32")
                    _builder_emit(
                        T.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup ^ 64]))
                    )
                    T.buffer_store(li.buffer, li + peer_li, [0])
                    with T.If(idx_in_warpgroup < B_H // 2):
                        with T.Then():
                            global_head = _builder_bind(
                                "global_head", cta_idx * (B_H // 2) + idx_in_warpgroup
                            )
                            cur_lse = _builder_alloc_scalar("cur_lse", "float32")
                            cur_lse_log = _builder_bind("cur_lse_log", T.log(li))
                            _builder_emit(T.ptx.fma.rn.f32(cur_lse, mi, LN_2, cur_lse_log))
                            T.buffer_store(
                                cur_lse.buffer,
                                T.if_then_else(
                                    cur_lse == T.float32(-float("inf")),
                                    T.float32(float("inf")),
                                    cur_lse,
                                ),
                                [0],
                            )
                            _builder_emit(
                                T.ptx.st.global_.f32(
                                    max_logits.ptr_to([s_q_idx, global_head]), real_mi * LN_2
                                )
                            )
                            _builder_emit(
                                T.ptx.st.global_.f32(lse.ptr_to([s_q_idx, global_head]), cur_lse)
                            )
                    last_k = _builder_bind("last_k", num_k_blocks - 1)
                    last_buf = _builder_bind("last_buf", last_k % NUM_BUFS)
                    last_phase = _builder_bind("last_phase", last_k // NUM_BUFS & 1)
                    _builder_emit(bar_sv_done.wait(last_buf, last_phase))
                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                    _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                    attn_sink_log2 = _builder_bind(
                        "attn_sink_log2",
                        T.cuda.ldg(
                            attn_sink.ptr_to([cta_idx * (B_H // 2) + idx_in_warpgroup % 64]),
                            "float32",
                        )
                        * LOG_2_E
                        if have_attn_sink
                        else T.float32(-float("inf")),
                    )
                    sink_exp = _builder_alloc_scalar("sink_exp", "float32")
                    _builder_emit(T.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi))
                    output_scale = _builder_scalar(
                        "output_scale",
                        T.cuda.fdividef(T.float32(1.0), li + sink_exp),
                        dtype="float32",
                    )
                    o_epi_frag = _builder_assign(
                        "o_epi_frag", T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "float32")
                    )
                    o_epi = _builder_assign("o_epi", o_epi_frag.local())
                    have_valid_indices = _builder_bind(
                        "have_valid_indices", T.cuda.any_sync(T.uint32(4294967295), li != 0.0) != 0
                    )
                    with T.If(T.Not(have_valid_indices)):
                        with T.Then():
                            with T.unroll(B_EPI) as o_zero_i:
                                T.buffer_store(o_epi, 0.0, [o_zero_i])
                            T.buffer_store(output_scale.buffer, 1.0, [0])
                    o_epi_bf16_frag = _builder_assign(
                        "o_epi_bf16_frag",
                        T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "bfloat16"),
                    )
                    o_epi_bf16 = _builder_assign("o_epi_bf16", o_epi_bf16_frag.local())
                    o_smem_win = _builder_assign(
                        "o_smem_win", o_smem.rearrange("h (b r) -> (b h) r", b=2)
                    )
                    with T.unroll(D_V // 2 // B_EPI) as epi_k:
                        with T.If(have_valid_indices):
                            with T.Then():
                                _builder_emit(
                                    T.evaluate(
                                        _tmem_load(
                                            o_epi,
                                            T.cuda.get_tmem_addr(
                                                T.uint32(o_tmem_col), 0, epi_k * B_EPI
                                            ),
                                            B_EPI,
                                        )
                                    )
                                )
                                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                        with T.unroll(B_EPI // 2) as scale_i:
                            _builder_emit(mul_f32x2(o_epi, scale_i * 2, output_scale))
                        with T.unroll(B_EPI // 2) as cast_i:
                            _builder_emit(
                                T.evaluate(_cast_f32x2_bf16x2(o_epi_bf16, o_epi, cast_i * 2))
                            )
                        o_epi_words = _builder_assign("o_epi_words", o_epi_bf16.view("uint32"))
                        with T.unroll(8) as o_store_i:
                            s_off = _builder_bind(
                                "s_off",
                                idx_in_warpgroup // 64 * 16384
                                + epi_k * 4096
                                + idx_in_warpgroup % 64 * 64
                                + T.bitwise_xor(
                                    o_store_i * 8,
                                    T.shift_left(
                                        T.bitwise_and(
                                            idx_in_warpgroup // 64 * 256
                                            + epi_k * 64
                                            + idx_in_warpgroup % 64,
                                            7,
                                        ),
                                        3,
                                    ),
                                ),
                            )
                            s_ptr = _builder_bind(
                                "s_ptr",
                                T.ptr_byte_offset(
                                    o_smem_win.ptr_to([0, 0]), s_off * BF16_BYTES, "bfloat16"
                                ),
                            )
                            o_word = _builder_bind("o_word", o_store_i * 4)
                            _builder_emit(
                                T.ptx.st.shared.v4.u32(
                                    s_ptr,
                                    o_epi_words[o_word],
                                    o_epi_words[o_word + 1],
                                    o_epi_words[o_word + 2],
                                    o_epi_words[o_word + 3],
                                )
                            )
                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                        _builder_emit(T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128))
                        with T.If(warp_idx == 0):
                            with T.Then():
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        o_part0_offset = _builder_bind(
                                            "o_part0_offset",
                                            epi_k * B_EPI * (B_H // 2) * BF16_BYTES,
                                        )
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx[_TMA_S2G_3D](
                                                    T.address_of(out_part0_tensormap),
                                                    T.cast(epi_k * B_EPI, "int32"),
                                                    T.cast(cta_idx * (B_H // 2), "int32"),
                                                    T.cast(s_q_idx, "int32"),
                                                    T.ptr_byte_offset(
                                                        o_smem.ptr_to([0, 0]),
                                                        o_part0_offset,
                                                        "bfloat16",
                                                    ),
                                                )
                                            )
                                        )
                        with T.If(warp_idx == 1):
                            with T.Then():
                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        epi_k2 = _builder_bind("epi_k2", epi_k + D_V // B_EPI // 2)
                                        _builder_emit(T.evaluate(epi_k2))
                                        o_part1_offset = _builder_bind(
                                            "o_part1_offset",
                                            (epi_k * B_EPI + D_V // 2) * (B_H // 2) * BF16_BYTES,
                                        )
                                        _builder_emit(
                                            T.evaluate(
                                                T.ptx[_TMA_S2G_3D](
                                                    T.address_of(out_part1_tensormap),
                                                    T.cast(epi_k * B_EPI + D_V // 2, "int32"),
                                                    T.cast(cta_idx * (B_H // 2), "int32"),
                                                    T.cast(s_q_idx, "int32"),
                                                    T.ptr_byte_offset(
                                                        o_smem.ptr_to([0, 0]),
                                                        o_part1_offset,
                                                        "bfloat16",
                                                    ),
                                                )
                                            )
                                        )
                    with T.If(warp_idx == 0):
                        with T.Then():
                            _builder_emit(
                                T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(
                                    T.uint32(0), T.uint32(512)
                                )
                            )
                    _builder_emit(iket.range_end(epilogue_token))
                with T.Else():
                    with T.If(warpgroup_idx == 1):
                        with T.Then():
                            k_gather_token = _builder_assign(
                                "k_gather_token", iket.range_start("h128-k-load")
                            )
                            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(96))
                            wg1_warp_idx = _builder_bind("wg1_warp_idx", warp_idx - 4)
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    with T.serial(0, num_k_blocks, unroll=False) as k:
                                        indices_int4 = _builder_assign(
                                            "indices_int4",
                                            T.alloc_local((WG1_ROWS_PER_WARP, 4), "int32"),
                                        )
                                        max_indices = _builder_scalar(
                                            "max_indices", -1, dtype="int32"
                                        )
                                        min_indices = _builder_scalar(
                                            "min_indices", s_kv, dtype="int32"
                                        )
                                        idx_block = _builder_assign(
                                            "idx_block",
                                            indices.view(
                                                s_q,
                                                stride_indices_s_q // B_TOPK,
                                                2,
                                                WG1_ROWS_PER_WARP,
                                                WG1_NUM_WARPS,
                                                4,
                                            ).sub[s_q_idx, k, cta_idx, :, wg1_warp_idx, :],
                                        )
                                        indices_words = _builder_assign(
                                            "indices_words", indices_int4.view(16).view("uint32")
                                        )
                                        with T.unroll(4) as indices_load_i:
                                            indices_word = _builder_bind(
                                                "indices_word", indices_load_i * 4
                                            )
                                            _builder_emit(
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
                                            )
                                        with T.unroll(WG1_ROWS_PER_WARP) as local_row:
                                            with T.unroll(4) as j:
                                                idx = _builder_bind(
                                                    "idx", indices_int4[local_row, j]
                                                )
                                                T.buffer_store(
                                                    max_indices.buffer, T.max(max_indices, idx), [0]
                                                )
                                                T.buffer_store(
                                                    min_indices.buffer, T.min(min_indices, idx), [0]
                                                )
                                        is_all_rows_invalid = _builder_bind(
                                            "is_all_rows_invalid",
                                            (min_indices == s_kv) | (max_indices == -1),
                                        )
                                        should_skip_tma = _builder_bind(
                                            "should_skip_tma", is_all_rows_invalid & (k >= NUM_BUFS)
                                        )
                                        cur_buf = _builder_bind("cur_buf", k % NUM_BUFS)
                                        cur_phase = _builder_bind("cur_phase", k // NUM_BUFS & 1)

                                        def gather_k_part(
                                            col_start, col_count, tx_dim, bar, tensormap
                                        ):
                                            with T.If(T.Not(should_skip_tma)):
                                                with T.Then():
                                                    k_gather_tile = _builder_assign(
                                                        "k_gather_tile",
                                                        k_smem.sub[
                                                            :,
                                                            col_start * 64 : col_start * 64
                                                            + col_count * 64,
                                                        ].tile(0, (-1, WG1_NUM_WARPS, 4))[
                                                            :, wg1_warp_idx, :
                                                        ],
                                                    )
                                                    with T.unroll(WG1_ROWS_PER_WARP) as row_group:
                                                        with T.unroll(col_count) as col_atom:
                                                            k_gather_offset = _builder_bind(
                                                                "k_gather_offset",
                                                                (
                                                                    (col_start + col_atom)
                                                                    * 64
                                                                    * (B_TOPK // 2)
                                                                    + (
                                                                        wg1_warp_idx * 4
                                                                        + row_group * 16
                                                                    )
                                                                    * 64
                                                                )
                                                                * BF16_BYTES,
                                                            )
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx[_TMA_GATHER4_2D_CACHE](
                                                                        T.ptr_byte_offset(
                                                                            k_smem.ptr_to([0, 0]),
                                                                            k_gather_offset,
                                                                            "bfloat16",
                                                                        ),
                                                                        T.address_of(tensormap),
                                                                        T.cast(
                                                                            (col_start + col_atom)
                                                                            * 64,
                                                                            "int32",
                                                                        ),
                                                                        indices_int4[row_group, 0],
                                                                        indices_int4[row_group, 1],
                                                                        indices_int4[row_group, 2],
                                                                        indices_int4[row_group, 3],
                                                                        T.cuda.cvta_generic_to_shared(
                                                                            leader_mbar(
                                                                                bar.ptr_to(
                                                                                    [cur_buf]
                                                                                )
                                                                            )
                                                                        ),
                                                                        _KV_TMA_CACHE_HINT,
                                                                    )
                                                                )
                                                            )
                                                with T.Else():
                                                    _rem1 = _builder_assign(
                                                        "_rem1", T.alloc_local([1], "uint64")
                                                    )
                                                    _builder_emit(
                                                        T.ptx.mapa.shared__cluster.u64(
                                                            _rem1[0],
                                                            bar.ptr_to([cur_buf]),
                                                            T.uint32(0),
                                                        )
                                                    )
                                                    _builder_emit(
                                                        T.ptx.mbarrier.complete_tx.relaxed.cluster.b64(
                                                            _rem1[0],
                                                            T.uint32(
                                                                WG1_ROWS_PER_WARP
                                                                * 4
                                                                * tx_dim
                                                                * BF16_BYTES
                                                            ),
                                                            pred=T.uint32(1),
                                                        )
                                                    )

                                        with T.If(k > 0):
                                            with T.Then():
                                                prev_buf = _builder_bind(
                                                    "prev_buf", (k - 1) % NUM_BUFS
                                                )
                                                prev_phase = _builder_bind(
                                                    "prev_phase", (k - 1) // NUM_BUFS & 1
                                                )
                                                _builder_emit(
                                                    bar_qk_part_done.wait(prev_buf, prev_phase)
                                                )
                                        _builder_emit(
                                            gather_k_part(
                                                0,
                                                num_sq_tiles,
                                                d_sq,
                                                bar_k_part0_ready,
                                                kv_k_part0_tensormap,
                                            )
                                        )
                                        with T.If(k > 0):
                                            with T.Then():
                                                prev_buf = _builder_bind(
                                                    "prev_buf", (k - 1) % NUM_BUFS
                                                )
                                                prev_phase = _builder_bind(
                                                    "prev_phase", (k - 1) // NUM_BUFS & 1
                                                )
                                                _builder_emit(
                                                    bar_qk_done.wait(prev_buf, prev_phase)
                                                )
                                        _builder_emit(
                                            gather_k_part(
                                                num_sq_tiles,
                                                num_qk_tiles - num_sq_tiles,
                                                D_TQ,
                                                bar_k_part1_ready,
                                                kv_k_part1_tensormap,
                                            )
                                        )
                            _builder_emit(iket.range_end(k_gather_token))
                        with T.Else():
                            with T.If(warpgroup_idx == 2):
                                with T.Then():
                                    v_gather_token = _builder_assign(
                                        "v_gather_token", iket.range_start("h128-v-load")
                                    )
                                    _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(96))
                                    wg2_warp_idx = _builder_bind("wg2_warp_idx", warp_idx - 8)
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            _builder_emit(bar_prologue_utccp.wait(0, 0))
                                            with T.serial(0, num_k_blocks, unroll=False) as k:
                                                cur_buf = _builder_bind("cur_buf", k % NUM_BUFS)
                                                cur_phase = _builder_bind(
                                                    "cur_phase", k // NUM_BUFS & 1
                                                )
                                                with T.If(k > 0):
                                                    with T.Then():
                                                        prev_buf = _builder_bind(
                                                            "prev_buf", (k - 1) % NUM_BUFS
                                                        )
                                                        prev_phase = _builder_bind(
                                                            "prev_phase", (k - 1) // NUM_BUFS & 1
                                                        )
                                                        _builder_emit(
                                                            bar_sv_part_done.wait(
                                                                prev_buf, prev_phase
                                                            )
                                                        )

                                                def gather_v_part(
                                                    row_offset, part, token_buf, bar, tensormap
                                                ):
                                                    idx_block = _builder_assign(
                                                        "idx_block",
                                                        indices.view(
                                                            s_q,
                                                            stride_indices_s_q // B_TOPK,
                                                            2,
                                                            WG2_ROWS_PER_PART,
                                                            WG2_NUM_WARPS,
                                                            4,
                                                        ).sub[s_q_idx, k, part, :, wg2_warp_idx, :],
                                                    )
                                                    token_words = _builder_assign(
                                                        "token_words",
                                                        token_buf.view(16).view("uint32"),
                                                    )
                                                    with T.unroll(4) as token_load_i:
                                                        token_word = _builder_bind(
                                                            "token_word", token_load_i * 4
                                                        )
                                                        _builder_emit(
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
                                                        )
                                                    src0 = _builder_bind("src0", cta_idx * 256)
                                                    v_gather_tile = _builder_assign(
                                                        "v_gather_tile",
                                                        v_smem_gemm.tile(
                                                            0, (2, -1, WG2_NUM_WARPS, 4)
                                                        )[part, :, wg2_warp_idx, :],
                                                    )
                                                    with T.unroll(WG2_ROWS_PER_PART) as row_group:
                                                        with T.unroll(D_V // 2 // 64) as col_atom:
                                                            v_gather_offset = _builder_bind(
                                                                "v_gather_offset",
                                                                (
                                                                    part * (B_TOPK // 2) * 64
                                                                    + col_atom * 64 * B_TOPK
                                                                    + (
                                                                        wg2_warp_idx * 4
                                                                        + row_group * 16
                                                                    )
                                                                    * 64
                                                                )
                                                                * BF16_BYTES,
                                                            )
                                                            _builder_emit(
                                                                T.evaluate(
                                                                    T.ptx[_TMA_GATHER4_2D_CACHE](
                                                                        T.ptr_byte_offset(
                                                                            v_smem.ptr_to([0, 0]),
                                                                            v_gather_offset,
                                                                            "bfloat16",
                                                                        ),
                                                                        T.address_of(tensormap),
                                                                        T.cast(
                                                                            src0 + col_atom * 64,
                                                                            "int32",
                                                                        ),
                                                                        token_buf[row_group, 0],
                                                                        token_buf[row_group, 1],
                                                                        token_buf[row_group, 2],
                                                                        token_buf[row_group, 3],
                                                                        T.cuda.cvta_generic_to_shared(
                                                                            leader_mbar(
                                                                                bar.ptr_to(
                                                                                    [cur_buf]
                                                                                )
                                                                            )
                                                                        ),
                                                                        _KV_TMA_CACHE_HINT,
                                                                    )
                                                                )
                                                            )

                                                token_idxs_part0 = _builder_assign(
                                                    "token_idxs_part0",
                                                    T.alloc_local((WG2_ROWS_PER_PART, 4), "int32"),
                                                )
                                                _builder_emit(
                                                    gather_v_part(
                                                        0,
                                                        0,
                                                        token_idxs_part0,
                                                        bar_v_part0_ready,
                                                        kv_v_part0_tensormap,
                                                    )
                                                )
                                                with T.If(k > 0):
                                                    with T.Then():
                                                        prev_buf = _builder_bind(
                                                            "prev_buf", (k - 1) % NUM_BUFS
                                                        )
                                                        prev_phase = _builder_bind(
                                                            "prev_phase", (k - 1) // NUM_BUFS & 1
                                                        )
                                                        _builder_emit(
                                                            bar_sv_done.wait(prev_buf, prev_phase)
                                                        )
                                                token_idxs_part1 = _builder_assign(
                                                    "token_idxs_part1",
                                                    T.alloc_local((WG2_ROWS_PER_PART, 4), "int32"),
                                                )
                                                _builder_emit(
                                                    gather_v_part(
                                                        WG2_ROWS_PER_PART,
                                                        1,
                                                        token_idxs_part1,
                                                        bar_v_part1_ready,
                                                        kv_v_part1_tensormap,
                                                    )
                                                )
                                    _builder_emit(iket.range_end(v_gather_token))
                                with T.Else():
                                    _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(168))
                                    with T.If((cta_idx == 0) & (warp_idx == 12)):
                                        with T.Then():
                                            mma_token = _builder_assign(
                                                "mma_token", iket.range_start("h128-qk-pv-issue")
                                            )
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(
                                                        bar_prologue_q.arrive(
                                                            0, tx_count=B_H * d_qk * BF16_BYTES
                                                        )
                                                    )
                                                    _builder_emit(bar_prologue_q.wait(0, 0))
                                                    _builder_emit(
                                                        T.ptx.tcgen05.fence__after_thread_sync()
                                                    )
                                                    with T.unroll(48) as q_copy_flat:
                                                        q_copy_src = _builder_bind(
                                                            "q_copy_src",
                                                            T.ptr_byte_offset(
                                                                q_full.ptr_to([0, 0]),
                                                                (
                                                                    d_sq * 8
                                                                    + q_copy_flat % 6 * 512
                                                                    + q_copy_flat // 6 % 8
                                                                )
                                                                * 16,
                                                                "bfloat16",
                                                            ),
                                                        )
                                                        _builder_emit(
                                                            T.evaluate(
                                                                T.ptx[_TCGEN_CP_64X128](
                                                                    T.cast(
                                                                        q_tmem_col
                                                                        + q_copy_flat % 6 * 32
                                                                        + q_copy_flat // 6 % 8 * 4,
                                                                        "uint32",
                                                                    ),
                                                                    _replace_smem_desc_addr(
                                                                        q_cp_desc, q_copy_src
                                                                    ),
                                                                )
                                                            )
                                                        )
                                                    _builder_emit(
                                                        T.evaluate(
                                                            T.ptx[_TCGEN_COMMIT](
                                                                T.cuda.cvta_generic_to_shared(
                                                                    bar_prologue_utccp.ptr_to([0])
                                                                ),
                                                                T.uint16(3),
                                                            )
                                                        )
                                                    )
                                                    with T.serial(
                                                        0, num_k_blocks + 1, unroll=False
                                                    ) as k:
                                                        with T.If(k < num_k_blocks):
                                                            with T.Then():
                                                                cur_buf = _builder_bind(
                                                                    "cur_buf", k % NUM_BUFS
                                                                )
                                                                cur_phase = _builder_bind(
                                                                    "cur_phase", k // NUM_BUFS & 1
                                                                )
                                                                _builder_emit(
                                                                    bar_k_part0_ready.arrive(
                                                                        cur_buf,
                                                                        tx_count=B_TOPK
                                                                        * d_sq
                                                                        * BF16_BYTES,
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_k_part0_ready.wait(
                                                                        cur_buf, cur_phase
                                                                    )
                                                                )
                                                                with T.If(k > 0):
                                                                    with T.Then():
                                                                        prev_buf = _builder_bind(
                                                                            "prev_buf",
                                                                            (k - 1) % NUM_BUFS,
                                                                        )
                                                                        prev_phase = _builder_bind(
                                                                            "prev_phase",
                                                                            (k - 1) // NUM_BUFS & 1,
                                                                        )
                                                                        _builder_emit(
                                                                            bar_p_free.wait(
                                                                                prev_buf, prev_phase
                                                                            )
                                                                        )
                                                                _builder_emit(
                                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                                )
                                                                T.buffer_store(
                                                                    mma_p_accumulate.buffer,
                                                                    T.uint32(0),
                                                                    [0],
                                                                )
                                                                if d_sq > 0:
                                                                    sq_smem = _builder_assign(
                                                                        "sq_smem",
                                                                        q_full.sub[:, :d_sq],
                                                                    )
                                                                    if (
                                                                        mma_smem_desc
                                                                        == "local_hoist"
                                                                    ):
                                                                        qk_part0_a_local = (
                                                                            _builder_assign(
                                                                                "qk_part0_a_local",
                                                                                SmemDescriptor(),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            qk_part0_a_local.init(
                                                                                sq_smem.ptr_to(
                                                                                    [0, 0]
                                                                                ),
                                                                                ldo=512,
                                                                                sdo=64,
                                                                                swizzle=3,
                                                                            )
                                                                        )
                                                                        qk_part0_b_local = (
                                                                            _builder_assign(
                                                                                "qk_part0_b_local",
                                                                                SmemDescriptor(),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            qk_part0_b_local.init(
                                                                                k_smem.ptr_to(
                                                                                    [0, 0]
                                                                                ),
                                                                                ldo=512,
                                                                                sdo=64,
                                                                                swizzle=3,
                                                                            )
                                                                        )
                                                                    if mma_smem_desc == "hoist":
                                                                        qk_part0_a_hoist = (
                                                                            _builder_assign(
                                                                                "qk_part0_a_hoist",
                                                                                SmemDescriptor(),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            qk_part0_a_hoist.init(
                                                                                sq_smem.ptr_to(
                                                                                    [0, 0]
                                                                                ),
                                                                                ldo=512,
                                                                                sdo=64,
                                                                                swizzle=3,
                                                                            )
                                                                        )
                                                                    with T.unroll(1) as mma_mi:
                                                                        with T.unroll(1) as mma_ni:
                                                                            with T.unroll(
                                                                                d_sq // 16
                                                                            ) as mma_ki:
                                                                                qk_part0_offset = _builder_bind(
                                                                                    "qk_part0_offset",
                                                                                    mma_ki
                                                                                    % (d_sq // 16)
                                                                                    // 4
                                                                                    * 4096
                                                                                    + mma_mi * 4096
                                                                                    + mma_ki
                                                                                    // (d_sq // 16)
                                                                                    * 64
                                                                                    + mma_ki
                                                                                    % 4
                                                                                    * 16,
                                                                                )
                                                                                if (
                                                                                    mma_smem_desc
                                                                                    == "recompute"
                                                                                ):
                                                                                    qk_part0_a_ptr = _builder_bind(
                                                                                        "qk_part0_a_ptr",
                                                                                        T.ptr_byte_offset(
                                                                                            sq_smem.ptr_to(
                                                                                                [
                                                                                                    0,
                                                                                                    0,
                                                                                                ]
                                                                                            ),
                                                                                            qk_part0_offset
                                                                                            // 8
                                                                                            * 16,
                                                                                            "bfloat16",
                                                                                        ),
                                                                                    )
                                                                                    qk_part0_b_ptr = _builder_bind(
                                                                                        "qk_part0_b_ptr",
                                                                                        T.ptr_byte_offset(
                                                                                            k_smem.ptr_to(
                                                                                                [
                                                                                                    0,
                                                                                                    0,
                                                                                                ]
                                                                                            ),
                                                                                            qk_part0_offset
                                                                                            // 8
                                                                                            * 16,
                                                                                            "bfloat16",
                                                                                        ),
                                                                                    )
                                                                                    _builder_emit(
                                                                                        T.evaluate(
                                                                                            _mma_f16(
                                                                                                T.cast(
                                                                                                    tmem_p_col
                                                                                                    + mma_ni
                                                                                                    * 64,
                                                                                                    "uint32",
                                                                                                ),
                                                                                                _recompute_smem_desc(
                                                                                                    qk_part0_a_ptr,
                                                                                                    1073758272,
                                                                                                    33554432,
                                                                                                ),
                                                                                                _recompute_smem_desc(
                                                                                                    qk_part0_b_ptr,
                                                                                                    1073758272,
                                                                                                    33554432,
                                                                                                ),
                                                                                                T.uint32(
                                                                                                    136316048
                                                                                                ),
                                                                                                T.Or(
                                                                                                    mma_ki
                                                                                                    != 0,
                                                                                                    T.cast(
                                                                                                        mma_p_accumulate,
                                                                                                        "bool",
                                                                                                    ),
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                elif (
                                                                                    mma_smem_desc
                                                                                    == "encode"
                                                                                ):
                                                                                    qk_part0_a_encode = _builder_assign(
                                                                                        "qk_part0_a_encode",
                                                                                        SmemDescriptor(),
                                                                                    )
                                                                                    _builder_emit(
                                                                                        qk_part0_a_encode.init(
                                                                                            T.ptr_byte_offset(
                                                                                                sq_smem.ptr_to(
                                                                                                    [
                                                                                                        0,
                                                                                                        0,
                                                                                                    ]
                                                                                                ),
                                                                                                qk_part0_offset
                                                                                                // 8
                                                                                                * 16,
                                                                                                "bfloat16",
                                                                                            ),
                                                                                            ldo=512,
                                                                                            sdo=64,
                                                                                            swizzle=3,
                                                                                        )
                                                                                    )
                                                                                    qk_part0_b_encode = _builder_assign(
                                                                                        "qk_part0_b_encode",
                                                                                        SmemDescriptor(),
                                                                                    )
                                                                                    _builder_emit(
                                                                                        qk_part0_b_encode.init(
                                                                                            T.ptr_byte_offset(
                                                                                                k_smem.ptr_to(
                                                                                                    [
                                                                                                        0,
                                                                                                        0,
                                                                                                    ]
                                                                                                ),
                                                                                                qk_part0_offset
                                                                                                // 8
                                                                                                * 16,
                                                                                                "bfloat16",
                                                                                            ),
                                                                                            ldo=512,
                                                                                            sdo=64,
                                                                                            swizzle=3,
                                                                                        )
                                                                                    )
                                                                                    _builder_emit(
                                                                                        T.evaluate(
                                                                                            _mma_f16(
                                                                                                T.cast(
                                                                                                    tmem_p_col
                                                                                                    + mma_ni
                                                                                                    * 64,
                                                                                                    "uint32",
                                                                                                ),
                                                                                                qk_part0_a_encode.desc,
                                                                                                qk_part0_b_encode.desc,
                                                                                                T.uint32(
                                                                                                    136316048
                                                                                                ),
                                                                                                T.Or(
                                                                                                    mma_ki
                                                                                                    != 0,
                                                                                                    T.cast(
                                                                                                        mma_p_accumulate,
                                                                                                        "bool",
                                                                                                    ),
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                elif (
                                                                                    mma_smem_desc
                                                                                    == "local_hoist"
                                                                                ):
                                                                                    _builder_emit(
                                                                                        T.evaluate(
                                                                                            _mma_f16(
                                                                                                T.cast(
                                                                                                    tmem_p_col
                                                                                                    + mma_ni
                                                                                                    * 64,
                                                                                                    "uint32",
                                                                                                ),
                                                                                                qk_part0_a_local.add_16B_offset(
                                                                                                    qk_part0_offset
                                                                                                    // 8
                                                                                                ),
                                                                                                qk_part0_b_local.add_16B_offset(
                                                                                                    qk_part0_offset
                                                                                                    // 8
                                                                                                ),
                                                                                                T.uint32(
                                                                                                    136316048
                                                                                                ),
                                                                                                T.Or(
                                                                                                    mma_ki
                                                                                                    != 0,
                                                                                                    T.cast(
                                                                                                        mma_p_accumulate,
                                                                                                        "bool",
                                                                                                    ),
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                else:
                                                                                    _builder_emit(
                                                                                        T.evaluate(
                                                                                            _mma_f16(
                                                                                                T.cast(
                                                                                                    tmem_p_col
                                                                                                    + mma_ni
                                                                                                    * 64,
                                                                                                    "uint32",
                                                                                                ),
                                                                                                qk_part0_a_hoist.add_16B_offset(
                                                                                                    qk_part0_offset
                                                                                                    // 8
                                                                                                ),
                                                                                                qk_k_part0_desc.add_16B_offset(
                                                                                                    qk_part0_offset
                                                                                                    // 8
                                                                                                ),
                                                                                                T.uint32(
                                                                                                    136316048
                                                                                                ),
                                                                                                T.Or(
                                                                                                    mma_ki
                                                                                                    != 0,
                                                                                                    T.cast(
                                                                                                        mma_p_accumulate,
                                                                                                        "bool",
                                                                                                    ),
                                                                                                ),
                                                                                            )
                                                                                        )
                                                                                    )
                                                                    T.buffer_store(
                                                                        mma_p_accumulate.buffer,
                                                                        T.uint32(1),
                                                                        [0],
                                                                    )
                                                                _builder_emit(
                                                                    bar_qk_part_done.arrive(
                                                                        cur_buf,
                                                                        cta_group=2,
                                                                        cta_mask=3,
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_k_part1_ready.arrive(
                                                                        cur_buf,
                                                                        tx_count=B_TOPK
                                                                        * (d_qk - d_sq)
                                                                        * BF16_BYTES,
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_k_part1_ready.wait(
                                                                        cur_buf, cur_phase
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                                )
                                                                if mma_smem_desc == "local_hoist":
                                                                    qk_part1_b_local = (
                                                                        _builder_assign(
                                                                            "qk_part1_b_local",
                                                                            SmemDescriptor(),
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        qk_part1_b_local.init(
                                                                            k_smem.ptr_to([0, 0]),
                                                                            ldo=512,
                                                                            sdo=64,
                                                                            swizzle=3,
                                                                        )
                                                                    )
                                                                with T.unroll(1) as mma_mi:
                                                                    with T.unroll(1) as mma_ni:
                                                                        with T.unroll(
                                                                            D_TQ // 16
                                                                        ) as mma_ki:
                                                                            qk_part1_offset = _builder_bind(
                                                                                "qk_part1_offset",
                                                                                mma_ki
                                                                                % (D_TQ // 16)
                                                                                // 4
                                                                                * 4096
                                                                                + mma_ni * 4096
                                                                                + mma_ki
                                                                                // (D_TQ // 16)
                                                                                * 64
                                                                                + mma_ki % 4 * 16
                                                                                + d_sq // 64 * 4096,
                                                                            )
                                                                            if (
                                                                                mma_smem_desc
                                                                                == "recompute"
                                                                            ):
                                                                                qk_part1_b_ptr = _builder_bind(
                                                                                    "qk_part1_b_ptr",
                                                                                    T.ptr_byte_offset(
                                                                                        k_smem.ptr_to(
                                                                                            [0, 0]
                                                                                        ),
                                                                                        qk_part1_offset
                                                                                        // 8
                                                                                        * 16,
                                                                                        "bfloat16",
                                                                                    ),
                                                                                )
                                                                                _builder_emit(
                                                                                    T.evaluate(
                                                                                        _mma_f16(
                                                                                            T.cast(
                                                                                                tmem_p_col
                                                                                                + mma_ni
                                                                                                * 64,
                                                                                                "uint32",
                                                                                            ),
                                                                                            T.cast(
                                                                                                q_tmem_col
                                                                                                + mma_ki
                                                                                                * 8,
                                                                                                "uint32",
                                                                                            ),
                                                                                            _recompute_smem_desc(
                                                                                                qk_part1_b_ptr,
                                                                                                1073758272,
                                                                                                33554432,
                                                                                            ),
                                                                                            T.uint32(
                                                                                                136316048
                                                                                            ),
                                                                                            T.Or(
                                                                                                mma_ki
                                                                                                != 0,
                                                                                                T.cast(
                                                                                                    mma_p_accumulate,
                                                                                                    "bool",
                                                                                                ),
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                            elif (
                                                                                mma_smem_desc
                                                                                == "encode"
                                                                            ):
                                                                                qk_part1_b_encode = _builder_assign(
                                                                                    "qk_part1_b_encode",
                                                                                    SmemDescriptor(),
                                                                                )
                                                                                _builder_emit(
                                                                                    qk_part1_b_encode.init(
                                                                                        T.ptr_byte_offset(
                                                                                            k_smem.ptr_to(
                                                                                                [
                                                                                                    0,
                                                                                                    0,
                                                                                                ]
                                                                                            ),
                                                                                            qk_part1_offset
                                                                                            // 8
                                                                                            * 16,
                                                                                            "bfloat16",
                                                                                        ),
                                                                                        ldo=512,
                                                                                        sdo=64,
                                                                                        swizzle=3,
                                                                                    )
                                                                                )
                                                                                _builder_emit(
                                                                                    T.evaluate(
                                                                                        _mma_f16(
                                                                                            T.cast(
                                                                                                tmem_p_col
                                                                                                + mma_ni
                                                                                                * 64,
                                                                                                "uint32",
                                                                                            ),
                                                                                            T.cast(
                                                                                                q_tmem_col
                                                                                                + mma_ki
                                                                                                * 8,
                                                                                                "uint32",
                                                                                            ),
                                                                                            qk_part1_b_encode.desc,
                                                                                            T.uint32(
                                                                                                136316048
                                                                                            ),
                                                                                            T.Or(
                                                                                                mma_ki
                                                                                                != 0,
                                                                                                T.cast(
                                                                                                    mma_p_accumulate,
                                                                                                    "bool",
                                                                                                ),
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                            elif (
                                                                                mma_smem_desc
                                                                                == "local_hoist"
                                                                            ):
                                                                                _builder_emit(
                                                                                    T.evaluate(
                                                                                        _mma_f16(
                                                                                            T.cast(
                                                                                                tmem_p_col
                                                                                                + mma_ni
                                                                                                * 64,
                                                                                                "uint32",
                                                                                            ),
                                                                                            T.cast(
                                                                                                q_tmem_col
                                                                                                + mma_ki
                                                                                                * 8,
                                                                                                "uint32",
                                                                                            ),
                                                                                            qk_part1_b_local.add_16B_offset(
                                                                                                qk_part1_offset
                                                                                                // 8
                                                                                            ),
                                                                                            T.uint32(
                                                                                                136316048
                                                                                            ),
                                                                                            T.Or(
                                                                                                mma_ki
                                                                                                != 0,
                                                                                                T.cast(
                                                                                                    mma_p_accumulate,
                                                                                                    "bool",
                                                                                                ),
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                            else:
                                                                                _builder_emit(
                                                                                    T.evaluate(
                                                                                        _mma_f16(
                                                                                            T.cast(
                                                                                                tmem_p_col
                                                                                                + mma_ni
                                                                                                * 64,
                                                                                                "uint32",
                                                                                            ),
                                                                                            T.cast(
                                                                                                q_tmem_col
                                                                                                + mma_ki
                                                                                                * 8,
                                                                                                "uint32",
                                                                                            ),
                                                                                            qk_k_part1_desc.add_16B_offset(
                                                                                                qk_part1_offset
                                                                                                // 8
                                                                                            ),
                                                                                            T.uint32(
                                                                                                136316048
                                                                                            ),
                                                                                            T.Or(
                                                                                                mma_ki
                                                                                                != 0,
                                                                                                T.cast(
                                                                                                    mma_p_accumulate,
                                                                                                    "bool",
                                                                                                ),
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )
                                                                T.buffer_store(
                                                                    mma_p_accumulate.buffer,
                                                                    T.uint32(1),
                                                                    [0],
                                                                )
                                                                _builder_emit(
                                                                    bar_qk_done.arrive(
                                                                        cur_buf,
                                                                        cta_group=2,
                                                                        cta_mask=3,
                                                                    )
                                                                )
                                                        with T.If(k > 0):
                                                            with T.Then():
                                                                cur_buf_prev = _builder_bind(
                                                                    "cur_buf_prev",
                                                                    (k - 1) % NUM_BUFS,
                                                                )
                                                                cur_phase_prev = _builder_bind(
                                                                    "cur_phase_prev",
                                                                    (k - 1) // NUM_BUFS & 1,
                                                                )
                                                                _builder_emit(
                                                                    bar_so_ready.wait(
                                                                        cur_buf_prev, cur_phase_prev
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_v_part0_ready.arrive(
                                                                        cur_buf_prev,
                                                                        tx_count=B_TOPK
                                                                        // 2
                                                                        * D_V
                                                                        * BF16_BYTES,
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_v_part0_ready.wait(
                                                                        cur_buf_prev, cur_phase_prev
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                                )
                                                                T.buffer_store(
                                                                    mma_o_accumulate.buffer,
                                                                    T.if_then_else(
                                                                        k == 1,
                                                                        T.uint32(0),
                                                                        T.uint32(1),
                                                                    ),
                                                                    [0],
                                                                )
                                                                if mma_smem_desc == "hoist":
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            0,
                                                                            0,
                                                                            0,
                                                                            pv_a_part0_lo_desc.desc,
                                                                            pv_b_part0_lo_desc.desc,
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            128,
                                                                            0,
                                                                            16384,
                                                                            pv_a_part0_hi_desc.desc,
                                                                            pv_b_part0_hi_desc.desc,
                                                                        )
                                                                    )
                                                                else:
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            0,
                                                                            0,
                                                                            0,
                                                                            T.uint64(0),
                                                                            T.uint64(0),
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            128,
                                                                            0,
                                                                            16384,
                                                                            T.uint64(0),
                                                                            T.uint64(0),
                                                                        )
                                                                    )
                                                                T.buffer_store(
                                                                    mma_o_accumulate.buffer,
                                                                    T.uint32(1),
                                                                    [0],
                                                                )
                                                                _builder_emit(
                                                                    bar_sv_part_done.arrive(
                                                                        cur_buf_prev,
                                                                        cta_group=2,
                                                                        cta_mask=3,
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_v_part1_ready.arrive(
                                                                        cur_buf_prev,
                                                                        tx_count=B_TOPK
                                                                        // 2
                                                                        * D_V
                                                                        * BF16_BYTES,
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_v_part1_ready.wait(
                                                                        cur_buf_prev, cur_phase_prev
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                                )
                                                                if mma_smem_desc == "hoist":
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            0,
                                                                            4096,
                                                                            4096,
                                                                            pv_a_part1_lo_desc.desc,
                                                                            pv_b_part1_lo_desc.desc,
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            128,
                                                                            4096,
                                                                            20480,
                                                                            pv_a_part1_hi_desc.desc,
                                                                            pv_b_part1_hi_desc.desc,
                                                                        )
                                                                    )
                                                                else:
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            0,
                                                                            4096,
                                                                            4096,
                                                                            T.uint64(0),
                                                                            T.uint64(0),
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        issue_pv_mma(
                                                                            128,
                                                                            4096,
                                                                            20480,
                                                                            T.uint64(0),
                                                                            T.uint64(0),
                                                                        )
                                                                    )
                                                                T.buffer_store(
                                                                    mma_o_accumulate.buffer,
                                                                    T.uint32(1),
                                                                    [0],
                                                                )
                                                                _builder_emit(
                                                                    bar_sv_done.arrive(
                                                                        cur_buf_prev,
                                                                        cta_group=2,
                                                                        cta_mask=3,
                                                                    )
                                                                )
                                            _builder_emit(iket.range_end(mma_token))
                                        with T.Else():
                                            with T.If(warp_idx == 13):
                                                with T.Then():
                                                    valid_mask_token = _builder_assign(
                                                        "valid_mask_token",
                                                        iket.range_start("h128-valid-mask"),
                                                    )
                                                    with T.If(lane_idx < B_TOPK // 8):
                                                        with T.Then():
                                                            lane_indices = _builder_assign(
                                                                "lane_indices",
                                                                T.alloc_local((8,), "int32"),
                                                            )
                                                            with T.serial(
                                                                0, num_k_blocks, unroll=False
                                                            ) as k:
                                                                row_base = _builder_bind(
                                                                    "row_base",
                                                                    g_indices_base
                                                                    + k * B_TOPK
                                                                    + lane_idx * 8,
                                                                )
                                                                lane_index_words = _builder_assign(
                                                                    "lane_index_words",
                                                                    lane_indices.view("uint32"),
                                                                )
                                                                _builder_emit(
                                                                    T.evaluate(
                                                                        T.ptx[
                                                                            "ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.u32"
                                                                        ](
                                                                            lane_index_words[0],
                                                                            lane_index_words[1],
                                                                            lane_index_words[2],
                                                                            lane_index_words[3],
                                                                            lane_index_words[4],
                                                                            lane_index_words[5],
                                                                            lane_index_words[6],
                                                                            lane_index_words[7],
                                                                            indices.ptr_to(
                                                                                [row_base]
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                                abs_pos_start = _builder_bind(
                                                                    "abs_pos_start", k * B_TOPK
                                                                )
                                                                is_ks_valid_mask = _builder_bind(
                                                                    "is_ks_valid_mask",
                                                                    pack_valid_mask8(
                                                                        lane_indices,
                                                                        abs_pos_start,
                                                                        lane_idx,
                                                                        topk_len,
                                                                        s_kv,
                                                                    ),
                                                                )
                                                                cur_buf = _builder_bind(
                                                                    "cur_buf", k % NUM_BUFS
                                                                )
                                                                cur_phase = _builder_bind(
                                                                    "cur_phase", k // NUM_BUFS & 1
                                                                )
                                                                _builder_emit(
                                                                    bar_k_valid_free.wait(
                                                                        cur_buf, cur_phase ^ 1
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    T.ptx.st.shared.b8(
                                                                        is_k_valid.ptr_to(
                                                                            [cur_buf, lane_idx]
                                                                        ),
                                                                        T.reinterpret(
                                                                            "uint8",
                                                                            is_ks_valid_mask,
                                                                        ),
                                                                    )
                                                                )
                                                                _builder_emit(
                                                                    bar_k_valid_ready.arrive(
                                                                        cur_buf
                                                                    )
                                                                )
                                                    _builder_emit(iket.range_end(valid_mask_token))
    return builder.get()


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    kernel = _build_kernel(
        s_q=cfg.s_q,
        s_kv=cfg.s_kv,
        topk=cfg.topk,
        d_qk=cfg.d_qk,
        h_q=cfg.h_q,
        stride_kv_s_kv=stride_kv_s_kv,
        stride_indices_s_q=stride_indices_s_q,
        have_attn_sink=cfg.have_attn_sink,
        have_topk_length=cfg.have_topk_length,
        sm_scale_div_log2=1.0 / math.sqrt(cfg.d_qk) * LOG_2_E,
    )
    kernel = kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
    return kernel


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
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
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

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
