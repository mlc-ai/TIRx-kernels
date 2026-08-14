# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla.utils._ir_builder import builder_alloc_scalar as _builder_alloc_scalar
from tirx_kernels.flashmla.utils._ir_builder import builder_assign as _builder_assign
from tirx_kernels.flashmla.utils._ir_builder import builder_bind as _builder_bind
from tirx_kernels.flashmla.utils._ir_builder import builder_emit as _builder_emit
from tirx_kernels.flashmla.utils._ir_builder import builder_scalar as _builder_scalar
from tirx_kernels.flashmla.utils._ir_builder import (
    query_cancel_first_ctaid_x as _query_cancel_first_ctaid_x,
)
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
from tvm.tirx.layout import Axis

B_H = 128
B_TOPK = 64
D_QK = 512
D_V = 512
LOG_2_E = math.log2(math.e)

BF16_BYTES = 2

IKET_EVENT_NAMES = (
    "h128-small-q-load-output",
    "h128-small-kv-load",
    "h128-small-qk-pv-issue",
    "h128-small-valid-mask",
    "h128-small-clc",
    "h128-small-softmax",
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


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128SmallTopKConfig:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int = D_QK
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 small-topk phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 small-topk phase1 requires h_kv == 1")
        if self.d_qk != D_QK:
            raise ValueError("head128 small-topk phase1 is scoped to d_qk == 512")
        if self.d_v != D_V:
            raise ValueError("head128 small-topk phase1 requires d_v == 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("small-topk phase1 requires topk to be a multiple of 64")
        if self.topk > 1280:
            raise ValueError("topk > 1280 dispatches outside the small-topk phase1 scope")


CONFIGS = [
    {
        "label": f"bench_smalltopk_dqk512_hq128_s4096_kv{s_kv}_topk1280",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 1280,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_small_topk_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128SmallTopKConfig:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128SmallTopKConfig)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128SmallTopKConfig(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_small_topk_dispatch_reason(cfg: SparseFlashMLAPrefillHead128SmallTopKConfig) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.h_kv != 1:
        return "out_of_scope: h_kv != 1 violates FlashMLA sparse prefill phase1 assumptions"
    if cfg.d_qk != D_QK:
        return "out_of_scope: small-topk head128 supports only D_QK=512"
    if cfg.d_v != D_V:
        return "out_of_scope: d_v != 512"
    if cfg.topk > 1280:
        return "out_of_scope: topk > 1280 dispatches to regular head128 when supported"
    return "small_topk: sm100 head128 run_fwd_for_small_topk_phase1_kernel<Prefill, 512>"


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
        "dispatch_reason": _flashmla_small_topk_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
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


# The dispatcher-selected SM100 form is frozen below as explicit TMA,
# tcgen05, TMEM, packed-register, and PTX memory operations.
# fmt: off
@cache
def _make_low_level_kernel(
    s_q: int,
    s_kv: int,
    topk: int,
    stride_kv_s_kv: int,
    stride_indices_s_q: int,
    have_attn_sink: bool,
    have_topk_length: bool,
    sm_scale_div_log2: float,
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name('_kernel')
            q = T.arg('q', T.buffer((s_q, 128, 512), 'bfloat16'))
            kv = T.arg('kv', T.buffer((s_kv * stride_kv_s_kv,), 'bfloat16'))
            indices = T.arg('indices', T.buffer((s_q * stride_indices_s_q,), 'int32'))
            attn_sink = T.arg('attn_sink', T.buffer((128,), 'float32'))
            topk_length = T.arg('topk_length', T.buffer((s_q,), 'int32'))
            out = T.arg('out', T.buffer((s_q, 128, 512), 'bfloat16'))
            max_logits = T.arg('max_logits', T.buffer((s_q, 128), 'float32'))
            lse = T.arg('lse', T.buffer((s_q, 128), 'float32'))
            _builder_emit(T.func_attr({'tirx.kernel_launch_params': ['blockIdx.x', 'clusterCtaIdx.x', 'threadIdx.x', 'tirx.use_programtic_dependent_launch', 'tirx.use_dyn_shared_memory']}))
            kv_tma_tensormap = _builder_bind('kv_tma_tensormap', T.tvm_stack_alloca('tensormap', 1), type_annotation=T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', kv_tma_tensormap, 'bfloat16', 2, T.handle_add_byte_offset(kv.data, 0), 512, s_kv, stride_kv_s_kv * 2, 64, 1, 1, 1, 0, 3, 3, 0))
            out_tensormap = _builder_bind('out_tensormap', T.tvm_stack_alloca('tensormap', 1), type_annotation=T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', out_tensormap, 'bfloat16', 4, T.handle_add_byte_offset(out.data, 0), 64, 128, 8, s_q, 1024, 128, 131072, 64, 64, 8, 1, 1, 1, 1, 1, 0, 3, 3, 0))
            out_tensormap_1 = _builder_bind('out_tensormap_1', T.tvm_stack_alloca('tensormap', 1), type_annotation=T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', out_tensormap_1, 'bfloat16', 4, T.handle_add_byte_offset(out.data, 0), 64, 128, 8, s_q, 1024, 128, 131072, 64, 64, 8, 1, 1, 1, 1, 1, 0, 3, 3, 0))
            q_tma_tensormap = _builder_bind('q_tma_tensormap', T.tvm_stack_alloca('tensormap', 1), type_annotation=T.TensorMap())
            _builder_emit(T.call_packed('runtime.cuTensorMapEncodeTiled', q_tma_tensormap, 'bfloat16', 5, T.handle_add_byte_offset(q.data, 0), 64, 128, 2, 4, s_q, 1024, 512, 128, 131072, 64, 64, 2, 4, 1, 1, 1, 1, 1, 1, 0, 3, 3, 0))
            with T.launch_thread('clusterCtaIdx.x', 2) as clusterCtaIdx_x:
                blockIdx_x = _builder_assign('blockIdx_x', T.launch_thread('blockIdx.x', 2 * s_q))
                threadIdx_x = _builder_assign('threadIdx_x', T.launch_thread('threadIdx.x', 512))
                warp_id_in_cta = _builder_bind('warp_id_in_cta', T.tvm_warp_shuffle(T.uint32(4294967295), threadIdx_x // 32, 0, 32, 32), type_annotation=T.int32)
                block_idx = _builder_bind('block_idx', blockIdx_x, type_annotation=T.int32)
                v = _builder_bind('v', clusterCtaIdx_x, type_annotation=T.int32)
                thread_idx = _builder_bind('thread_idx', threadIdx_x, type_annotation=T.int32)
                v_1 = _builder_bind('v_1', warp_id_in_cta // 4, type_annotation=T.int32)
                v_2 = _builder_bind('v_2', warp_id_in_cta % 4, type_annotation=T.int32)
                v_3 = _builder_bind('v_3', threadIdx_x % 32, type_annotation=T.int32)
                v_4 = _builder_bind('v_4', threadIdx_x % 128, type_annotation=T.int32)
                v_5 = _builder_bind('v_5', threadIdx_x % 128, type_annotation=T.int32)
                v_6 = _builder_bind('v_6', threadIdx_x % 128, type_annotation=T.int32)
                v_7 = _builder_bind('v_7', threadIdx_x % 128, type_annotation=T.int32)
                _builder_emit(T.evaluate(v))
                _builder_emit(T.evaluate(v_1))
                _builder_emit(T.evaluate(v_2))
                _builder_emit(T.evaluate(v_3))
                _builder_emit(T.evaluate(v_4))
                _builder_emit(T.evaluate(v_5))
                _builder_emit(T.evaluate(v_6))
                _builder_emit(T.evaluate(v_7))
                _builder_if_33 = T.If(warp_id_in_cta == 0)
                _builder_if_33.__enter__()
                _builder_then_33 = T.Then()
                _builder_then_33.__enter__()
                _builder_if_34 = T.If(T.cuda.elect_sync() != T.uint32(0))
                _builder_if_34.__enter__()
                _builder_then_34 = T.Then()
                _builder_then_34.__enter__()
                _builder_emit(T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(q_tma_tensormap)), '', '', '', 'tensormap', ''))
                _builder_then_34.__exit__(None, None, None)
                _builder_if_34.__exit__(None, None, None)
                _builder_then_33.__exit__(None, None, None)
                _builder_if_33.__exit__(None, None, None)
                _builder_if_36 = T.If(warp_id_in_cta == 0)
                _builder_if_36.__enter__()
                _builder_then_36 = T.Then()
                _builder_then_36.__enter__()
                _builder_if_37 = T.If(T.cuda.elect_sync() != T.uint32(0))
                _builder_if_37.__enter__()
                _builder_then_37 = T.Then()
                _builder_then_37.__enter__()
                _builder_emit(T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(out_tensormap_1)), '', '', '', 'tensormap', ''))
                _builder_then_37.__exit__(None, None, None)
                _builder_if_37.__exit__(None, None, None)
                _builder_then_36.__exit__(None, None, None)
                _builder_if_36.__exit__(None, None, None)
                _builder_if_39 = T.If(warp_id_in_cta == 0)
                _builder_if_39.__enter__()
                _builder_then_39 = T.Then()
                _builder_then_39.__enter__()
                _builder_if_40 = T.If(T.cuda.elect_sync() != T.uint32(0))
                _builder_if_40.__enter__()
                _builder_then_40 = T.Then()
                _builder_then_40.__enter__()
                _builder_emit(T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(out_tensormap)), '', '', '', 'tensormap', ''))
                _builder_then_40.__exit__(None, None, None)
                _builder_if_40.__exit__(None, None, None)
                _builder_then_39.__exit__(None, None, None)
                _builder_if_39.__exit__(None, None, None)
                _builder_if_42 = T.If(warp_id_in_cta == 0)
                _builder_if_42.__enter__()
                _builder_then_42 = T.Then()
                _builder_then_42.__enter__()
                _builder_if_43 = T.If(T.cuda.elect_sync() != T.uint32(0))
                _builder_if_43.__enter__()
                _builder_then_43 = T.Then()
                _builder_then_43.__enter__()
                _builder_emit(T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(kv_tma_tensormap)), '', '', '', 'tensormap', ''))
                _builder_then_43.__exit__(None, None, None)
                _builder_if_43.__exit__(None, None, None)
                _builder_then_42.__exit__(None, None, None)
                _builder_if_42.__exit__(None, None, None)
                with T.attr({'tirx.launch_bounds_min_blocks_per_sm': 1}):
                    cta_idx = _builder_bind('cta_idx', block_idx % 2, type_annotation=T.int32)
                    warp_idx = _builder_bind('warp_idx', T.cuda.__shfl_sync(T.uint32(4294967295), thread_idx // 32, 0, 32), type_annotation=T.int32)
                    lane_idx = _builder_bind('lane_idx', thread_idx % 32, type_annotation=T.int32)
                    warpgroup_idx = _builder_bind('warpgroup_idx', T.cuda.__shfl_sync(T.uint32(4294967295), thread_idx // 128, 0, 32), type_annotation=T.int32)
                    idx_in_warpgroup = _builder_bind('idx_in_warpgroup', thread_idx % 128, type_annotation=T.int32)
                    pool_buf = _builder_assign('pool_buf', T.alloc_buffer((0,), 'uint8', scope='shared.dyn'))
                    q_smem = _builder_assign('q_smem', T.decl_buffer((64, 512), 'bfloat16', data=pool_buf.data, scope='shared.dyn', align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 8, 64):(64, 4096, 1)]))))
                    k_smem = _builder_assign('k_smem', T.decl_buffer((256, 256), 'bfloat16', data=pool_buf.data, elem_offset=32768, scope='shared.dyn', align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(256, 4, 64):(64, 16384, 1)]))))
                    s_smem_gemm = _builder_assign('s_smem_gemm', T.decl_buffer((64, 64), 'bfloat16', data=pool_buf.data, elem_offset=98304, scope='shared.dyn', align=1024, layout=T.TileLayout(T.S[(64, 8, 8):(8, 512, 1)])))
                    p_exchange = _builder_assign('p_exchange', T.decl_buffer((4, 1024), 'uint32', data=pool_buf.data, elem_offset=51200, scope='shared.dyn'))
                    rowwise_max_buf = _builder_assign('rowwise_max_buf', T.decl_buffer((128,), data=pool_buf.data, elem_offset=55296, scope='shared.dyn'))
                    rowwise_li_buf = _builder_assign('rowwise_li_buf', T.decl_buffer((128,), data=pool_buf.data, elem_offset=55424, scope='shared.dyn'))
                    is_k_valid = _builder_assign('is_k_valid', T.decl_buffer((4, 8), 'int8', data=pool_buf.data, elem_offset=222208, scope='shared.dyn', align=16))
                    buffer = _builder_assign('buffer', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27780, scope='shared.dyn', align=8))
                    buffer_1 = _builder_assign('buffer_1', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27781, scope='shared.dyn', align=8))
                    buffer_2 = _builder_assign('buffer_2', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27782, scope='shared.dyn', align=8))
                    buffer_3 = _builder_assign('buffer_3', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27783, scope='shared.dyn', align=8))
                    bar_tOut_empty_buf = _builder_assign('bar_tOut_empty_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27784, scope='shared.dyn', align=8))
                    buffer_4 = _builder_assign('buffer_4', T.decl_buffer((4,), 'uint64', data=pool_buf.data, elem_offset=27785, scope='shared.dyn', align=8))
                    buffer_5 = _builder_assign('buffer_5', T.decl_buffer((4,), 'uint64', data=pool_buf.data, elem_offset=27789, scope='shared.dyn', align=8))
                    bar_P_empty_buf = _builder_assign('bar_P_empty_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27793, scope='shared.dyn', align=8))
                    buffer_6 = _builder_assign('buffer_6', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27794, scope='shared.dyn', align=8))
                    buffer_7 = _builder_assign('buffer_7', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27795, scope='shared.dyn', align=8))
                    bar_S_O_full_buf = _builder_assign('bar_S_O_full_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27796, scope='shared.dyn', align=8))
                    bar_li_full_buf = _builder_assign('bar_li_full_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27797, scope='shared.dyn', align=8))
                    bar_li_empty_buf = _builder_assign('bar_li_empty_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27798, scope='shared.dyn', align=8))
                    bar_valid_coord_scales_full_buf = _builder_assign('bar_valid_coord_scales_full_buf', T.decl_buffer((4,), 'uint64', data=pool_buf.data, elem_offset=27799, scope='shared.dyn', align=8))
                    bar_valid_coord_scales_empty_buf = _builder_assign('bar_valid_coord_scales_empty_buf', T.decl_buffer((4,), 'uint64', data=pool_buf.data, elem_offset=27803, scope='shared.dyn', align=8))
                    buffer_8 = _builder_assign('buffer_8', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27807, scope='shared.dyn', align=8))
                    bar_clc_empty_buf = _builder_assign('bar_clc_empty_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27808, scope='shared.dyn', align=8))
                    bar_tQ_consumed_buf = _builder_assign('bar_tQ_consumed_buf', T.decl_buffer((1,), 'uint64', data=pool_buf.data, elem_offset=27809, scope='shared.dyn', align=8))
                    clc_response = _builder_assign('clc_response', T.decl_buffer((4,), 'uint32', data=pool_buf.data, elem_offset=55620, scope='shared.dyn', align=16))
                    tmem_start_addr = _builder_assign('tmem_start_addr', T.decl_buffer((1,), 'uint32', data=pool_buf.data, elem_offset=55624, scope='shared.dyn', align=4))
                    with T.attr({'tirx.dyn_smem_bytes': T.int64(222500)}):
                        _builder_emit(T.evaluate(0))
                    o_tmem = _builder_assign('o_tmem', T.decl_buffer((64, 512), scope='tmem', layout=T.TileLayout(T.S[(64, 2, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0))
                    buffer_9 = _builder_assign('buffer_9', T.decl_buffer((64, 2, 2, 128), scope='tmem', layout=T.TileLayout(T.S[(64, 2, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0))
                    buffer_10 = _builder_assign('buffer_10', T.decl_buffer((2, 64, 2, 128), scope='tmem', layout=T.TileLayout(T.S[(2, 64, 2, 128):(64 @ Axis.TLane, 1 @ Axis.TLane, 128 @ Axis.TCol, 1 @ Axis.TCol)]), allocated_addr=0))
                    o_win = _builder_assign('o_win', T.decl_buffer((128, 256), scope='tmem', layout=T.TileLayout(T.S[(2, 64, 2, 128):(64 @ Axis.TLane, 1 @ Axis.TLane, 128 @ Axis.TCol, 1 @ Axis.TCol)]), allocated_addr=0))
                    q_tmem_fold = _builder_assign('q_tmem_fold', T.decl_buffer((2, 64, 256), 'bfloat16', scope='tmem', layout=T.TileLayout(T.S[(128, 256):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256))
                    tmem_p = _builder_assign('tmem_p', T.decl_buffer((64, 128), scope='tmem', layout=T.TileLayout(T.S[(64, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                    buffer_11 = _builder_assign('buffer_11', k_smem.view(4, 64, 4, 64))
                    buffer_12 = _builder_assign('buffer_12', T.decl_buffer((4, 64, 4, 64), 'bfloat16', data=buffer_11.data, elem_offset=32768, scope='shared.dyn', align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(4, 64, 4, 64):(16384, 64, 4096, 1)]))))
                    k_smem_gemm = _builder_assign('k_smem_gemm', buffer_12.view(4, 64, 256))
                    _builder_if_89 = T.If(warp_idx == 1)
                    _builder_if_89.__enter__()
                    _builder_then_89 = T.Then()
                    _builder_then_89.__enter__()
                    _builder_if_90 = T.If(T.cuda.elect_sync())
                    _builder_if_90.__enter__()
                    _builder_then_90 = T.Then()
                    _builder_then_90.__enter__()
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_1[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_2[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_3[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_tOut_empty_buf[i])), T.uint32(256), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_P_empty_buf[i])), T.uint32(256), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_6[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_7[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_S_O_full_buf[i])), T.uint32(256), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_full_buf[i])), T.uint32(64), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_empty_buf[i])), T.uint32(128), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_8[i])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_clc_empty_buf[i])), T.uint32(539), 'init', 'shared', 'b64', ''))
                    # One elected arrival from each WG0 warp in both CTAs.
                    with T.unroll(1) as i:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_tQ_consumed_buf[i])), T.uint32(8), 'init', 'shared', 'b64', ''))
                    _builder_emit(T.ptx.fence('mbarrier_init', 'release', 'cluster', ''))
                    _builder_then_90.__exit__(None, None, None)
                    _builder_if_90.__exit__(None, None, None)
                    _builder_then_89.__exit__(None, None, None)
                    _builder_else_89 = T.Else()
                    _builder_else_89.__enter__()
                    _builder_if_119 = T.If(warp_idx == 2)
                    _builder_if_119.__enter__()
                    _builder_then_119 = T.Then()
                    _builder_then_119.__enter__()
                    _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(tmem_start_addr[0])), T.uint32(512), 'alloc', 'cta_group::2', 'sync', 'aligned', 'shared::cta', 'b32', ''))
                    allocated_tmem_addr = _builder_alloc_scalar('allocated_tmem_addr', 'uint32')
                    _builder_emit(T.ptx.ld.shared.u32(allocated_tmem_addr, tmem_start_addr.ptr_to([0])))
                    _builder_emit(T.cuda.trap_when_assert_failed(allocated_tmem_addr == T.uint32(0)))
                    _builder_emit(T.ptx.tcgen05('relinquish_alloc_permit', 'cta_group::2', 'sync', 'aligned', ''))
                    _builder_then_119.__exit__(None, None, None)
                    _builder_else_119 = T.Else()
                    _builder_else_119.__enter__()
                    _builder_if_126 = T.If(warp_idx == 3)
                    _builder_if_126.__enter__()
                    _builder_then_126 = T.Then()
                    _builder_then_126.__enter__()
                    _builder_if_127 = T.If(T.cuda.elect_sync())
                    _builder_if_127.__enter__()
                    _builder_then_127 = T.Then()
                    _builder_then_127.__enter__()
                    with T.unroll(4) as init_stage:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_4[init_stage])), T.uint32(1), 'init', 'shared', 'b64', ''))
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_5[init_stage])), T.uint32(1), 'init', 'shared', 'b64', ''))
                    with T.unroll(4) as init_stage:
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_full_buf[init_stage])), T.uint32(8), 'init', 'shared', 'b64', ''))
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_empty_buf[init_stage])), T.uint32(128), 'init', 'shared', 'b64', ''))
                    _builder_emit(T.ptx.fence('mbarrier_init', 'release', 'cluster', ''))
                    _builder_then_127.__exit__(None, None, None)
                    _builder_if_127.__exit__(None, None, None)
                    _builder_then_126.__exit__(None, None, None)
                    _builder_if_126.__exit__(None, None, None)
                    _builder_else_119.__exit__(None, None, None)
                    _builder_if_119.__exit__(None, None, None)
                    _builder_else_89.__exit__(None, None, None)
                    _builder_if_89.__exit__(None, None, None)
                    _builder_emit(T.cuda.cluster_sync())
                    _builder_if_136 = T.If(warpgroup_idx == 0)
                    _builder_if_136.__enter__()
                    _builder_then_136 = T.Then()
                    _builder_then_136.__enter__()
                    q_o_token = _builder_scalar('q_o_token', T.cuda.iket.range_start('h128-small-q-load-output'), dtype='uint32')
                    _builder_emit(T.ptx.setmaxnreg(160, 'inc', 'sync', 'aligned', 'u32', ''))
                    wg0_job_valid = _builder_scalar('wg0_job_valid', 1, dtype='int32')
                    wg0_job_block_idx = _builder_scalar('wg0_job_block_idx', block_idx, dtype='int32')
                    wg0_outer_loop_phase = _builder_scalar('wg0_outer_loop_phase', 0, dtype='int32')
                    last_valid = _builder_scalar('last_valid', 0, dtype='int32')
                    last_s_q_idx = _builder_scalar('last_s_q_idx', 0, dtype='int32')
                    last_outer_loop_phase = _builder_scalar('last_outer_loop_phase', 0, dtype='int32')
                    with T.While(wg0_job_valid != 0):
                        wg0_s_q_idx = _builder_bind('wg0_s_q_idx', wg0_job_block_idx // 2, type_annotation=T.int32)
                        _builder_if_147 = T.If(warp_idx == 0)
                        _builder_if_147.__enter__()
                        _builder_then_147 = T.Then()
                        _builder_then_147.__enter__()
                        _builder_if_148 = T.If(T.cuda.elect_sync())
                        _builder_if_148.__enter__()
                        _builder_then_148 = T.Then()
                        _builder_then_148.__enter__()
                        _builder_emit(T.ptx.cp(0, 'async', 'bulk', 'wait_group', '', ''))
                        buffer_13 = _builder_assign('buffer_13', q.view(s_q, 128, 2, 4, 64))
                        buffer_14 = _builder_assign('buffer_14', buffer_13.view(64, 128, 2, 4, s_q, layout=T.TileLayout(T.S[(64, 128, 2, 4, s_q):(1, 512, 256, 64, 65536)])))
                        q_tma = _builder_assign('q_tma', T.decl_buffer((64, 128, 2, 4, s_q), 'bfloat16', data=buffer_14.data, layout=T.TileLayout(T.S[(64, 128, 2, 4, s_q):(1, 512, 256, 64, 65536)])))
                        buffer_15 = _builder_assign('buffer_15', q_smem.view(64, 4, 2, 64))
                        buffer_16 = _builder_assign('buffer_16', buffer_15.view(64, 64, 2, 4, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 64, 2, 4):(1, 64, 4096, 8192)]))))
                        q_smem_tma = _builder_assign('q_smem_tma', T.decl_buffer((64, 64, 2, 4), 'bfloat16', data=buffer_16.data, scope='shared.dyn', align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 64, 2, 4):(1, 64, 4096, 8192)]))))
                        buffer_17 = _builder_alloc_scalar('buffer_17', 'uint64')
                        _builder_emit(T.ptx.mapa(buffer_17, T.address_of(buffer[0]), T.uint32(0), '', 'u64', ''))
                        buffer_18 = _builder_assign('buffer_18', T.decl_scalar(T.bfloat16, data=q_smem_tma.data, elem_offset=0, scope='shared.dyn'))
                        _builder_emit(T.ptx.cp(T.cuda.cvta_generic_to_shared(T.address_of(buffer_18)), T.reinterpret(T.handle().ty, T.address_of(q_tma_tensormap)), 0, block_idx % 2 * 64, 0, 0, wg0_job_block_idx // 2, T.cuda.cvta_generic_to_shared(T.reinterpret(T.handle().ty, buffer_17)), T.uint64(1364590687093260288), 'async', 'bulk', 'tensor', '5d', 'shared::cluster', 'global', '', 'mbarrier::complete_tx::bytes', '', 'cta_group::2', 'L2::cache_hint', ''))
                        _builder_if_160 = T.If(cta_idx == 0)
                        _builder_if_160.__enter__()
                        _builder_then_160 = T.Then()
                        _builder_then_160.__enter__()
                        # Do not republish tQ-full until both CTAs consumed its prior phase.
                        with T.If(last_valid != 0):
                            with T.Then():
                                _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_tQ_consumed_buf[0]), T.bitwise_xor(last_outer_loop_phase, 0)))
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer[0])), T.uint32(131072), 'arrive', 'expect_tx', '', '', 'shared', 'b64', ''))
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer[0]), T.bitwise_xor(wg0_outer_loop_phase, 0)))
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_1[0]), T.bitwise_xor(T.bitwise_xor(wg0_outer_loop_phase, 1), 0)))
                        _builder_emit(T.ptx.tcgen05('fence::after_thread_sync', ''))
                        buffer_19 = _builder_assign('buffer_19', T.decl_buffer((2, 64, 4, 64), 'bfloat16', scope='tmem', layout=T.TileLayout(T.S[(128, 256):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256))
                        buffer_20 = _builder_assign('buffer_20', T.decl_buffer((64, 4, 2, 64), 'bfloat16', scope='tmem', layout=T.TileLayout(T.S[(64, 4, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256))
                        q_tmem_cp = _builder_assign('q_tmem_cp', T.decl_buffer((64, 4, 2, 64), 'bfloat16', scope='tmem', layout=T.TileLayout(T.S[(64, 4, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256))
                        buffer_21 = _builder_assign('buffer_21', q_smem.view(64, 4, 2, 64))
                        cp_desc = _builder_alloc_scalar('cp_desc', 'uint64')
                        _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(cp_desc.buffer.data, T.reinterpret(T.handle().ty, T.uint64(0)), 1, 64, 3))
                        with T.unroll(16) as flat:
                            _builder_emit(T.ptx.tcgen05(T.Cast('uint32', 256 + (flat % 4 * 32 + flat // 4 % 4 * 8)), T.bitwise_or(T.bitwise_and(cp_desc, T.bitwise_not(T.uint64(16383))), T.Cast('uint64', T.bitwise_and(T.shift_right(T.cuda.cvta_generic_to_shared(T.ptr_byte_offset(T.address_of(buffer_21[0, 0, 0, 0]), (flat % 4 * 1024 + flat // 4 % 4 * 2) * 16, T.type_annotation('bfloat16'))), T.uint32(4)), T.uint32(16383)))), 'cp', 'cta_group::2', '128x256b', '', '', '', ''))
                        _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_2[0])), T.Cast('uint16', 3), 'commit', 'cta_group::2', 'mbarrier::arrive::one', 'shared::cluster', 'multicast::cluster', 'b64', ''))
                        _builder_then_160.__exit__(None, None, None)
                        _builder_if_160.__exit__(None, None, None)
                        _builder_then_148.__exit__(None, None, None)
                        _builder_if_148.__exit__(None, None, None)
                        _builder_then_147.__exit__(None, None, None)
                        _builder_if_147.__exit__(None, None, None)
                        _builder_if_174 = T.If(last_valid != 0)
                        _builder_if_174.__enter__()
                        _builder_then_174 = T.Then()
                        _builder_then_174.__enter__()
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_li_full_buf[0]), T.bitwise_xor(last_outer_loop_phase, 0)))
                        output_scale = _builder_alloc_scalar('output_scale', 'float32')
                        _builder_emit(T.ptx.ld.shared.f32(output_scale, rowwise_li_buf.ptr_to([idx_in_warpgroup % 64])))
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_empty_buf[0])), T.uint32(1), 'arrive', '', '', 'shared', 'b64', ''))
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_3[0]), T.bitwise_xor(last_outer_loop_phase, 0)))
                        buffer_13 = _builder_assign('buffer_13', T.alloc_local((64,)))
                        o_epi_frag = _builder_assign('o_epi_frag', buffer_13.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)])))
                        o_epi = _builder_assign('o_epi', o_epi_frag.local())
                        buffer_14 = _builder_assign('buffer_14', T.alloc_local((64,), 'bfloat16'))
                        o_epi_bf16_frag = _builder_assign('o_epi_bf16_frag', buffer_14.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)])))
                        buffer_15 = _builder_assign('buffer_15', q_smem.view(64, 2, 256))
                        buffer_16 = _builder_assign('buffer_16', buffer_15.view(2, 64, 256, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 64, 4, 64):(16384, 64, 4096, 1)]))))
                        q_smem_win = _builder_assign('q_smem_win', buffer_16.view(128, 256))
                        with T.unroll(4) as epi_k:
                            local_storage = _builder_assign('local_storage', o_epi_frag.local())
                            _builder_emit(T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], local_storage[32], local_storage[33], local_storage[34], local_storage[35], local_storage[36], local_storage[37], local_storage[38], local_storage[39], local_storage[40], local_storage[41], local_storage[42], local_storage[43], local_storage[44], local_storage[45], local_storage[46], local_storage[47], local_storage[48], local_storage[49], local_storage[50], local_storage[51], local_storage[52], local_storage[53], local_storage[54], local_storage[55], local_storage[56], local_storage[57], local_storage[58], local_storage[59], local_storage[60], local_storage[61], local_storage[62], local_storage[63], T.cuda.get_tmem_addr(T.uint32(0), 0, epi_k * 64), 'ld', 'sync', 'aligned', '32x32b', 'x64', '', 'b32', ''))
                            _builder_emit(T.ptx.tcgen05('wait::ld', 'sync', 'aligned', ''))
                            _builder_if_192 = T.If(epi_k == 0)
                            _builder_if_192.__enter__()
                            _builder_then_192 = T.Then()
                            _builder_then_192.__enter__()
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(T.bitwise_xor(last_outer_loop_phase, 1), 0)))
                            _builder_emit(T.ptx.fence('proxy', 'async', 'shared::cta', ''))
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    tq_consumed_remote = _builder_alloc_scalar('tq_consumed_remote', 'uint32')
                                    _builder_emit(T.ptx.mapa(tq_consumed_remote, T.cuda.cvta_generic_to_shared(T.address_of(bar_tQ_consumed_buf[0])), T.uint32(0), 'shared::cluster', 'u32', ''))
                                    _builder_emit(T.ptx.mbarrier(tq_consumed_remote, 'arrive', '', '', 'shared::cluster', 'b64', ''))
                            _builder_then_192.__exit__(None, None, None)
                            _builder_if_192.__exit__(None, None, None)
                            _builder_if_195 = T.If(epi_k == 3)
                            _builder_if_195.__enter__()
                            _builder_then_195 = T.Then()
                            _builder_then_195.__enter__()
                            buffer_17 = _builder_alloc_scalar('buffer_17', 'uint32')
                            _builder_emit(T.ptx.mapa(buffer_17, T.cuda.cvta_generic_to_shared(T.address_of(bar_tOut_empty_buf[0])), T.uint32(0), 'shared::cluster', 'u32', ''))
                            _builder_emit(T.ptx.mbarrier(buffer_17, 'arrive', '', '', 'shared::cluster', 'b64', ''))
                            _builder_then_195.__exit__(None, None, None)
                            _builder_if_195.__exit__(None, None, None)
                            buffer_17 = _builder_assign('buffer_17', o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                            buffer_18 = _builder_assign('buffer_18', o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                            with T.serial(32) as f:
                                dst_lane_indices_0_0 = _builder_scalar('dst_lane_indices_0_0', f * 2, dtype='int32')
                                dst_lane_indices_1_0 = _builder_scalar('dst_lane_indices_1_0', f * 2 + 1, dtype='int32')
                                buffer_19 = _builder_alloc_scalar('buffer_19', 'uint64')
                                buffer_20 = _builder_alloc_scalar('buffer_20', 'uint64')
                                _builder_emit(T.ptx.mov(buffer_19, buffer_18[f * 2], buffer_18[f * 2 + 1], 'b64', ''))
                                _builder_emit(T.ptx.mov(buffer_20, output_scale, output_scale, 'b64', ''))
                                _builder_emit(T.ptx.mul(buffer_19, buffer_19, buffer_20, 'rz', 'ftz', '', 'f32x2', ''))
                                _builder_emit(T.ptx.mov(buffer_17[f * 2], buffer_17[f * 2 + 1], buffer_19, 'b64', ''))
                            buffer_19 = _builder_assign('buffer_19', o_epi_bf16_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                            buffer_19_words = _builder_assign('buffer_19_words', buffer_19.view('uint32'))
                            buffer_20 = _builder_assign('buffer_20', o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                            with T.serial(32) as f:
                                dst_lane_indices_0_0 = _builder_scalar('dst_lane_indices_0_0', f * 2, dtype='int32')
                                dst_lane_indices_1_0 = _builder_scalar('dst_lane_indices_1_0', f * 2 + 1, dtype='int32')
                                _builder_emit(T.ptx.cvt.rn.bf16x2.f32(buffer_19_words[f], buffer_20[f * 2 + 1], buffer_20[f * 2]))
                            r_local = _builder_assign('r_local', o_epi_bf16_frag.local())
                            r_words = _builder_assign('r_words', r_local.view('uint32'))
                            with T.serial(8) as f:
                                ds = _builder_scalar('ds', f % 8 * 8, dtype='int32')
                                dr = _builder_scalar('dr', f % 8 * 8, dtype='int32')
                                s_off = _builder_scalar('s_off', v_5 // 64 * 16384 + epi_k % 4 * 4096 + v_5 % 64 * 64 + T.bitwise_xor(f * 8, T.shift_left(T.bitwise_and(v_5 // 64 * 256 + epi_k % 4 * 64 + v_5 % 64, 7), 3)), dtype='int32')
                                s_ptr = _builder_bind('s_ptr', T.ptr_byte_offset(T.address_of(q_smem_win[0, 0]), s_off * BF16_BYTES, 'bfloat16'))
                                r_w = _builder_scalar('r_w', dr // 2, dtype='int32')
                                _builder_emit(T.ptx.st(T.cuda.cvta_generic_to_shared(s_ptr), r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], '', '', 'shared', '', '', '', 'v4', 'u32', ''))
                        _builder_emit(T.ptx.fence('proxy', 'async', 'shared::cta', ''))
                        _builder_emit(T.ptx.bar(T.uint32(0), T.uint32(128), '', 'sync', ''))
                        _builder_if_232 = T.If(warp_idx == 0)
                        _builder_if_232.__enter__()
                        _builder_then_232 = T.Then()
                        _builder_then_232.__enter__()
                        _builder_if_233 = T.If(T.cuda.elect_sync())
                        _builder_if_233.__enter__()
                        _builder_then_233 = T.Then()
                        _builder_then_233.__enter__()
                        buffer_17 = _builder_assign('buffer_17', T.decl_scalar(T.bfloat16, data=q_smem.data, elem_offset=0, scope='shared.dyn'))
                        _builder_emit(T.ptx.cp(T.reinterpret(T.handle().ty, T.address_of(out_tensormap_1)), 0, block_idx % 2 * 64, 0, last_s_q_idx, T.cuda.cvta_generic_to_shared(T.address_of(buffer_17)), 'async', 'bulk', 'tensor', '4d', 'global', 'shared::cta', 'tile', 'bulk_group', '', ''))
                        _builder_emit(T.ptx.cp('async', 'bulk', 'commit_group', ''))
                        _builder_then_233.__exit__(None, None, None)
                        _builder_if_233.__exit__(None, None, None)
                        _builder_then_232.__exit__(None, None, None)
                        _builder_if_232.__exit__(None, None, None)
                        _builder_then_174.__exit__(None, None, None)
                        _builder_else_174 = T.Else()
                        _builder_else_174.__enter__()
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(wg0_outer_loop_phase, 0)))
                        with T.If(T.cuda.elect_sync()):
                            with T.Then():
                                tq_consumed_remote = _builder_alloc_scalar('tq_consumed_remote', 'uint32')
                                _builder_emit(T.ptx.mapa(tq_consumed_remote, T.cuda.cvta_generic_to_shared(T.address_of(bar_tQ_consumed_buf[0])), T.uint32(0), 'shared::cluster', 'u32', ''))
                                _builder_emit(T.ptx.mbarrier(tq_consumed_remote, 'arrive', '', '', 'shared::cluster', 'b64', ''))
                        _builder_else_174.__exit__(None, None, None)
                        _builder_if_174.__exit__(None, None, None)
                        T.buffer_store(last_valid.buffer, 1, [0])
                        T.buffer_store(last_s_q_idx.buffer, wg0_s_q_idx, [0])
                        T.buffer_store(last_outer_loop_phase.buffer, wg0_outer_loop_phase, [0])
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(wg0_outer_loop_phase, 0)))
                        wg0_next_job = _builder_alloc_scalar('wg0_next_job', 'uint32')
                        _builder_emit(_query_cancel_first_ctaid_x(wg0_next_job, T.address_of(clc_response[0])))
                        _rem1 = _builder_alloc_scalar('_rem1', 'uint64')
                        _builder_emit(T.ptx.mapa(_rem1, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), 'shared::cluster', 'u64', ''))
                        _builder_emit(T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem1), T.uint32(1), T.bool(True), 'arrive', '', '', '', 'b64', 'pred'))
                        _builder_if_248 = T.If(wg0_next_job == T.uint32(4294967295))
                        _builder_if_248.__enter__()
                        _builder_then_248 = T.Then()
                        _builder_then_248.__enter__()
                        T.buffer_store(wg0_job_valid.buffer, 0, [0])
                        _builder_then_248.__exit__(None, None, None)
                        _builder_else_248 = T.Else()
                        _builder_else_248.__enter__()
                        T.buffer_store(wg0_job_block_idx.buffer, T.Cast('int32', wg0_next_job), [0])
                        _builder_else_248.__exit__(None, None, None)
                        _builder_if_248.__exit__(None, None, None)
                        T.buffer_store(wg0_outer_loop_phase.buffer, T.bitwise_xor(wg0_outer_loop_phase, 1), [0])
                    _builder_if_253 = T.If(last_valid != 0)
                    _builder_if_253.__enter__()
                    _builder_then_253 = T.Then()
                    _builder_then_253.__enter__()
                    _builder_if_254 = T.If(warp_idx == 0)
                    _builder_if_254.__enter__()
                    _builder_then_254 = T.Then()
                    _builder_then_254.__enter__()
                    _builder_if_255 = T.If(T.cuda.elect_sync())
                    _builder_if_255.__enter__()
                    _builder_then_255 = T.Then()
                    _builder_then_255.__enter__()
                    _builder_emit(T.ptx.cp(0, 'async', 'bulk', 'wait_group', '', ''))
                    _builder_then_255.__exit__(None, None, None)
                    _builder_if_255.__exit__(None, None, None)
                    _builder_then_254.__exit__(None, None, None)
                    _builder_if_254.__exit__(None, None, None)
                    _builder_emit(T.ptx.bar(T.uint32(0), T.uint32(128), '', 'sync', ''))
                    _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_li_full_buf[0]), T.bitwise_xor(last_outer_loop_phase, 0)))
                    output_scale = _builder_alloc_scalar('output_scale', 'float32')
                    _builder_emit(T.ptx.ld.shared.f32(output_scale, rowwise_li_buf.ptr_to([idx_in_warpgroup % 64])))
                    _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_empty_buf[0])), T.uint32(1), 'arrive', '', '', 'shared', 'b64', ''))
                    _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_3[0]), T.bitwise_xor(last_outer_loop_phase, 0)))
                    _builder_if_263 = T.If(T.cuda.elect_sync())
                    _builder_if_263.__enter__()
                    _builder_then_263 = T.Then()
                    _builder_then_263.__enter__()
                    _builder_emit(T.ptx.griddepcontrol('launch_dependents', ''))
                    _builder_then_263.__exit__(None, None, None)
                    _builder_if_263.__exit__(None, None, None)
                    buffer_13 = _builder_assign('buffer_13', T.alloc_local((64,)))
                    o_epi_frag = _builder_assign('o_epi_frag', buffer_13.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)])))
                    o_epi = _builder_assign('o_epi', o_epi_frag.local())
                    buffer_14 = _builder_assign('buffer_14', T.alloc_local((64,), 'bfloat16'))
                    o_epi_bf16_frag = _builder_assign('o_epi_bf16_frag', buffer_14.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)])))
                    buffer_15 = _builder_assign('buffer_15', q_smem.view(64, 2, 256))
                    buffer_16 = _builder_assign('buffer_16', buffer_15.view(2, 64, 256, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 64, 4, 64):(16384, 64, 4096, 1)]))))
                    q_smem_win = _builder_assign('q_smem_win', buffer_16.view(128, 256))
                    with T.unroll(4) as epi_k:
                        local_storage = _builder_assign('local_storage', o_epi_frag.local())
                        _builder_emit(T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], local_storage[32], local_storage[33], local_storage[34], local_storage[35], local_storage[36], local_storage[37], local_storage[38], local_storage[39], local_storage[40], local_storage[41], local_storage[42], local_storage[43], local_storage[44], local_storage[45], local_storage[46], local_storage[47], local_storage[48], local_storage[49], local_storage[50], local_storage[51], local_storage[52], local_storage[53], local_storage[54], local_storage[55], local_storage[56], local_storage[57], local_storage[58], local_storage[59], local_storage[60], local_storage[61], local_storage[62], local_storage[63], T.cuda.get_tmem_addr(T.uint32(0), 0, epi_k * 64), 'ld', 'sync', 'aligned', '32x32b', 'x64', '', 'b32', ''))
                        _builder_emit(T.ptx.tcgen05('wait::ld', 'sync', 'aligned', ''))
                        _builder_if_277 = T.If(epi_k == 0)
                        _builder_if_277.__enter__()
                        _builder_then_277 = T.Then()
                        _builder_then_277.__enter__()
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(last_outer_loop_phase, 0)))
                        _builder_emit(T.ptx.fence('proxy', 'async', 'shared::cta', ''))
                        _builder_then_277.__exit__(None, None, None)
                        _builder_if_277.__exit__(None, None, None)
                        _builder_if_280 = T.If(epi_k == 3)
                        _builder_if_280.__enter__()
                        _builder_then_280 = T.Then()
                        _builder_then_280.__enter__()
                        buffer_17 = _builder_alloc_scalar('buffer_17', 'uint32')
                        _builder_emit(T.ptx.mapa(buffer_17, T.cuda.cvta_generic_to_shared(T.address_of(bar_tOut_empty_buf[0])), T.uint32(0), 'shared::cluster', 'u32', ''))
                        _builder_emit(T.ptx.mbarrier(buffer_17, 'arrive', '', '', 'shared::cluster', 'b64', ''))
                        _builder_then_280.__exit__(None, None, None)
                        _builder_if_280.__exit__(None, None, None)
                        buffer_17 = _builder_assign('buffer_17', o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                        buffer_18 = _builder_assign('buffer_18', o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                        with T.serial(32) as f:
                            dst_lane_indices_0_0 = _builder_scalar('dst_lane_indices_0_0', f * 2, dtype='int32')
                            dst_lane_indices_1_0 = _builder_scalar('dst_lane_indices_1_0', f * 2 + 1, dtype='int32')
                            buffer_19 = _builder_alloc_scalar('buffer_19', 'uint64')
                            buffer_20 = _builder_alloc_scalar('buffer_20', 'uint64')
                            _builder_emit(T.ptx.mov(buffer_19, buffer_18[f * 2], buffer_18[f * 2 + 1], 'b64', ''))
                            _builder_emit(T.ptx.mov(buffer_20, output_scale, output_scale, 'b64', ''))
                            _builder_emit(T.ptx.mul(buffer_19, buffer_19, buffer_20, 'rz', 'ftz', '', 'f32x2', ''))
                            _builder_emit(T.ptx.mov(buffer_17[f * 2], buffer_17[f * 2 + 1], buffer_19, 'b64', ''))
                        buffer_19 = _builder_assign('buffer_19', o_epi_bf16_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                        buffer_19_words = _builder_assign('buffer_19_words', buffer_19.view('uint32'))
                        buffer_20 = _builder_assign('buffer_20', o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)])))
                        with T.serial(32) as f:
                            dst_lane_indices_0_0 = _builder_scalar('dst_lane_indices_0_0', f * 2, dtype='int32')
                            dst_lane_indices_1_0 = _builder_scalar('dst_lane_indices_1_0', f * 2 + 1, dtype='int32')
                            _builder_emit(T.ptx.cvt.rn.bf16x2.f32(buffer_19_words[f], buffer_20[f * 2 + 1], buffer_20[f * 2]))
                        r_local = _builder_assign('r_local', o_epi_bf16_frag.local())
                        r_words = _builder_assign('r_words', r_local.view('uint32'))
                        with T.serial(8) as f:
                            ds = _builder_scalar('ds', f % 8 * 8, dtype='int32')
                            dr = _builder_scalar('dr', f % 8 * 8, dtype='int32')
                            s_off = _builder_scalar('s_off', v_6 // 64 * 16384 + epi_k % 4 * 4096 + v_6 % 64 * 64 + T.bitwise_xor(f * 8, T.shift_left(T.bitwise_and(v_6 // 64 * 256 + epi_k % 4 * 64 + v_6 % 64, 7), 3)), dtype='int32')
                            s_ptr = _builder_bind('s_ptr', T.ptr_byte_offset(T.address_of(q_smem_win[0, 0]), s_off * BF16_BYTES, 'bfloat16'))
                            r_w = _builder_scalar('r_w', dr // 2, dtype='int32')
                            _builder_emit(T.ptx.st(T.cuda.cvta_generic_to_shared(s_ptr), r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], '', '', 'shared', '', '', '', 'v4', 'u32', ''))
                    _builder_emit(T.ptx.fence('proxy', 'async', 'shared::cta', ''))
                    _builder_emit(T.ptx.bar(T.uint32(0), T.uint32(128), '', 'sync', ''))
                    _builder_if_317 = T.If(warp_idx == 0)
                    _builder_if_317.__enter__()
                    _builder_then_317 = T.Then()
                    _builder_then_317.__enter__()
                    _builder_if_318 = T.If(T.cuda.elect_sync())
                    _builder_if_318.__enter__()
                    _builder_then_318 = T.Then()
                    _builder_then_318.__enter__()
                    buffer_17 = _builder_assign('buffer_17', T.decl_scalar(T.bfloat16, data=q_smem.data, elem_offset=0, scope='shared.dyn'))
                    _builder_emit(T.ptx.cp(T.reinterpret(T.handle().ty, T.address_of(out_tensormap)), 0, block_idx % 2 * 64, 0, last_s_q_idx, T.cuda.cvta_generic_to_shared(T.address_of(buffer_17)), 'async', 'bulk', 'tensor', '4d', 'global', 'shared::cta', 'tile', 'bulk_group', '', ''))
                    _builder_emit(T.ptx.cp('async', 'bulk', 'commit_group', ''))
                    _builder_then_318.__exit__(None, None, None)
                    _builder_if_318.__exit__(None, None, None)
                    _builder_then_317.__exit__(None, None, None)
                    _builder_if_317.__exit__(None, None, None)
                    _builder_then_253.__exit__(None, None, None)
                    _builder_if_253.__exit__(None, None, None)
                    _builder_if_322 = T.If(warp_idx == 0)
                    _builder_if_322.__enter__()
                    _builder_then_322 = T.Then()
                    _builder_then_322.__enter__()
                    _builder_emit(T.ptx.tcgen05(T.uint32(0), T.uint32(512), 'dealloc', 'cta_group::2', 'sync', 'aligned', 'b32', ''))
                    _builder_then_322.__exit__(None, None, None)
                    _builder_if_322.__exit__(None, None, None)
                    _builder_emit(T.cuda.iket.range_end(q_o_token))
                    _builder_then_136.__exit__(None, None, None)
                    _builder_else_136 = T.Else()
                    _builder_else_136.__enter__()
                    _builder_if_326 = T.If(warpgroup_idx == 1)
                    _builder_if_326.__enter__()
                    _builder_then_326 = T.Then()
                    _builder_then_326.__enter__()
                    kv_gather_token = _builder_scalar('kv_gather_token', T.cuda.iket.range_start('h128-small-kv-load'), dtype='uint32')
                    _builder_emit(T.ptx.setmaxnreg(80, 'dec', 'sync', 'aligned', 'u32', ''))
                    wg1_warp_idx = _builder_bind('wg1_warp_idx', thread_idx // 32 - 4, type_annotation=T.int32)
                    _builder_if_330 = T.If(T.cuda.elect_sync())
                    _builder_if_330.__enter__()
                    _builder_then_330 = T.Then()
                    _builder_then_330.__enter__()
                    wg1_job_valid = _builder_scalar('wg1_job_valid', 1, dtype='int32')
                    wg1_job_block_idx = _builder_scalar('wg1_job_block_idx', block_idx, dtype='int32')
                    wg1_outer_loop_phase = _builder_scalar('wg1_outer_loop_phase', 0, dtype='int32')
                    wg1_rs = _builder_scalar('wg1_rs', 0, dtype='int32')
                    with T.While(wg1_job_valid != 0):
                        wg1_s_q_idx = _builder_bind('wg1_s_q_idx', wg1_job_block_idx // 2, type_annotation=T.int32)
                        wg1_topk_len = _builder_scalar('wg1_topk_len', topk, dtype='int32')
                        if have_topk_length:
                            _builder_emit(T.ptx.ld.global_.s32(wg1_topk_len, topk_length.ptr_to([wg1_s_q_idx])))
                        wg1_num_k_blocks = _builder_bind('wg1_num_k_blocks', T.max((wg1_topk_len + 64 - 1) // 64, 1), type_annotation=T.int32)
                        wg1_g_indices_base = _builder_bind('wg1_g_indices_base', wg1_s_q_idx * stride_indices_s_q, type_annotation=T.int32)
                        with T.serial(wg1_num_k_blocks, unroll=False) as k:
                            k_buf_idx = _builder_bind('k_buf_idx', wg1_rs % 4, type_annotation=T.int32)
                            k_bar_phase = _builder_bind('k_bar_phase', T.bitwise_and(wg1_rs // 4, 1), type_annotation=T.int32)
                            cur_indices = _builder_assign('cur_indices', T.alloc_local((16,), 'int32'))
                            with T.unroll(2) as local_row:
                                row = _builder_bind('row', local_row * 32 + wg1_warp_idx * 8, type_annotation=T.int32)
                                row_base = _builder_bind('row_base', wg1_g_indices_base + k * 64 + row, type_annotation=T.int32)
                                buffer_13 = _builder_assign('buffer_13', T.decl_buffer((16,), 'int32', data=cur_indices.data, scope='local'))
                                buffer_14 = _builder_assign('buffer_14', buffer_13.view('uint32'))
                                _builder_emit(T.ptx.ld(buffer_14[local_row * 8], buffer_14[local_row * 8 + 1], buffer_14[local_row * 8 + 2], buffer_14[local_row * 8 + 3], buffer_14[local_row * 8 + 4], buffer_14[local_row * 8 + 5], buffer_14[local_row * 8 + 6], buffer_14[local_row * 8 + 7], T.address_of(indices[row_base]), '', '', 'global', '', 'nc', 'L1::no_allocate', 'L2::evict_first', 'L2::256B', 'v8', 'u32', ''))
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_5[k_buf_idx]), T.bitwise_xor(T.bitwise_xor(k_bar_phase, 1), 0)))
                            k_smem_gemm_cur = _builder_assign('k_smem_gemm_cur', T.decl_buffer((64, 256), 'bfloat16', data=k_smem_gemm.data, elem_offset=32768 + k_buf_idx * 16384, scope='shared.dyn', align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 4, 64):(64, 4096, 1)]))))
                            src_col = _builder_bind('src_col', cta_idx * 256, type_annotation=T.int32)
                            buffer_13 = _builder_assign('buffer_13', k_smem_gemm_cur.view(64, 4, 64))
                            buffer_14 = _builder_assign('buffer_14', buffer_13.view(2, 4, 2, 4, 4, 64))
                            buffer_15 = _builder_assign('buffer_15', T.decl_buffer((2, 2, 4, 4, 64), 'bfloat16', data=buffer_14.data, elem_offset=32768 + k_buf_idx * 16384 + wg1_warp_idx * 512, scope='shared.dyn', align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 2, 4, 4, 64):(2048, 256, 64, 4096, 1)]))))
                            k_gather_tile = _builder_assign('k_gather_tile', buffer_15.view(16, 4, 64))
                            kv_tma = _builder_assign('kv_tma', kv.view(s_kv, 512, layout=T.TileLayout(T.S[(s_kv, 512):(stride_kv_s_kv, 1)])))
                            k_gather_tile_2d = _builder_assign('k_gather_tile_2d', k_gather_tile.view(16, 256))
                            with T.unroll(4) as row_group:
                                with T.unroll(4) as col_atom:
                                    buffer_16 = _builder_alloc_scalar('buffer_16', 'uint64')
                                    _builder_emit(T.ptx.mapa(buffer_16, T.address_of(buffer_4[k_buf_idx]), T.uint32(0), '', 'u64', ''))
                                    kv_dst_offset = _builder_bind('kv_dst_offset', (k_buf_idx * 16384 + wg1_warp_idx * 512 + row_group // 2 * 2048 + row_group % 2 * 256 + col_atom * 4096) * BF16_BYTES, type_annotation=T.int32)
                                    _builder_emit(T.ptx.cp(T.cuda.cvta_generic_to_shared(T.ptr_byte_offset(T.address_of(k_smem[0, 0]), kv_dst_offset, T.type_annotation('bfloat16'))), T.reinterpret(T.handle().ty, T.address_of(kv_tma_tensormap)), src_col + col_atom * 64, cur_indices[row_group * 4], cur_indices[row_group * 4 + 1], cur_indices[row_group * 4 + 2], cur_indices[row_group * 4 + 3], T.cuda.cvta_generic_to_shared(T.reinterpret(T.handle().ty, buffer_16)), T.uint64(1508705875169116160), 'async', 'bulk', 'tensor', '2d', 'shared::cluster', 'global', 'tile::gather4', 'mbarrier::complete_tx::bytes', '', 'cta_group::2', 'L2::cache_hint', ''))
                            T.buffer_store(wg1_rs.buffer, wg1_rs + 1, [0])
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(wg1_outer_loop_phase, 0)))
                        wg1_next_job = _builder_alloc_scalar('wg1_next_job', 'uint32')
                        _builder_emit(_query_cancel_first_ctaid_x(wg1_next_job, T.address_of(clc_response[0])))
                        _rem2 = _builder_alloc_scalar('_rem2', 'uint64')
                        _builder_emit(T.ptx.mapa(_rem2, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), 'shared::cluster', 'u64', ''))
                        _builder_emit(T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem2), T.uint32(1), T.bool(True), 'arrive', '', '', '', 'b64', 'pred'))
                        _builder_if_374 = T.If(wg1_next_job == T.uint32(4294967295))
                        _builder_if_374.__enter__()
                        _builder_then_374 = T.Then()
                        _builder_then_374.__enter__()
                        T.buffer_store(wg1_job_valid.buffer, 0, [0])
                        _builder_then_374.__exit__(None, None, None)
                        _builder_else_374 = T.Else()
                        _builder_else_374.__enter__()
                        T.buffer_store(wg1_job_block_idx.buffer, T.Cast('int32', wg1_next_job), [0])
                        _builder_else_374.__exit__(None, None, None)
                        _builder_if_374.__exit__(None, None, None)
                        T.buffer_store(wg1_outer_loop_phase.buffer, T.bitwise_xor(wg1_outer_loop_phase, 1), [0])
                    _builder_then_330.__exit__(None, None, None)
                    _builder_if_330.__exit__(None, None, None)
                    _builder_emit(T.cuda.iket.range_end(kv_gather_token))
                    _builder_then_326.__exit__(None, None, None)
                    _builder_else_326 = T.Else()
                    _builder_else_326.__enter__()
                    _builder_if_381 = T.If(warpgroup_idx == 2)
                    _builder_if_381.__enter__()
                    _builder_then_381 = T.Then()
                    _builder_then_381.__enter__()
                    _builder_emit(T.ptx.setmaxnreg(80, 'dec', 'sync', 'aligned', 'u32', ''))
                    _builder_if_383 = T.If(T.bitwise_and(warp_idx == 8, cta_idx == 0))
                    _builder_if_383.__enter__()
                    _builder_then_383 = T.Then()
                    _builder_then_383.__enter__()
                    mma_token = _builder_scalar('mma_token', T.cuda.iket.range_start('h128-small-qk-pv-issue'), dtype='uint32')
                    _builder_if_385 = T.If(T.cuda.elect_sync())
                    _builder_if_385.__enter__()
                    _builder_then_385 = T.Then()
                    _builder_then_385.__enter__()
                    umma_job_valid = _builder_scalar('umma_job_valid', 1, dtype='int32')
                    umma_job_block_idx = _builder_scalar('umma_job_block_idx', block_idx, dtype='int32')
                    umma_outer_loop_phase = _builder_scalar('umma_outer_loop_phase', 0, dtype='int32')
                    umma_rs = _builder_scalar('umma_rs', 0, dtype='int32')
                    with T.While(umma_job_valid != 0):
                        umma_s_q_idx = _builder_bind('umma_s_q_idx', umma_job_block_idx // 2, type_annotation=T.int32)
                        umma_topk_len = _builder_scalar('umma_topk_len', topk, dtype='int32')
                        if have_topk_length:
                            _builder_emit(T.ptx.ld.global_.s32(umma_topk_len, topk_length.ptr_to([umma_s_q_idx])))
                        umma_num_k_blocks = _builder_bind('umma_num_k_blocks', T.max((umma_topk_len + 64 - 1) // 64, 1), type_annotation=T.int32)
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(umma_outer_loop_phase, 0)))
                        with T.serial(umma_num_k_blocks + 1, unroll=False) as k:
                            _builder_if_398 = T.If(k < umma_num_k_blocks)
                            _builder_if_398.__enter__()
                            _builder_then_398 = T.Then()
                            _builder_then_398.__enter__()
                            k_buf_idx = _builder_bind('k_buf_idx', umma_rs % 4, type_annotation=T.int32)
                            k_bar_phase = _builder_bind('k_bar_phase', T.bitwise_and(umma_rs // 4, 1), type_annotation=T.int32)
                            p_bar_phase = _builder_bind('p_bar_phase', T.bitwise_and(umma_rs, 1), type_annotation=T.int32)
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_P_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(p_bar_phase, 1), 0)))
                            _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_4[k_buf_idx])), T.uint32(65536), 'arrive', 'expect_tx', '', '', 'shared', 'b64', ''))
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_4[k_buf_idx]), T.bitwise_xor(k_bar_phase, 0)))
                            _builder_emit(T.ptx.tcgen05('fence::after_thread_sync', ''))
                            qk_accumulate = _builder_scalar('qk_accumulate', T.uint32(0), dtype='uint32')
                            descB_local = _builder_alloc_scalar('descB_local', 'uint64')
                            _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3))
                            with T.unroll(1) as mi:
                                with T.unroll(1) as ni:
                                    with T.unroll(16) as ki:
                                        _builder_emit(T.ptx.tcgen05(T.Cast('uint32', ni * 64 + 384), T.Cast('uint32', ki * 8 + 256), _add_smem_desc_offset(descB_local, (ki // 1024 * 16384 + ni * 16384 + k_buf_idx * 16384 + ki % 16 // 4 * 4096 + ki % 1024 // 16 * 64 + ki % 4 * 16) // 8), T.uint32(136316048), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.Or(ki != 0, T.Cast('bool', qk_accumulate)), 'mma', 'cta_group::2', 'kind::f16', 'p12'))
                            T.buffer_store(qk_accumulate.buffer, T.uint32(1), [0])
                            _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_6[0])), T.Cast('uint16', 3), 'commit', 'cta_group::2', 'mbarrier::arrive::one', 'shared::cluster', 'multicast::cluster', 'b64', ''))
                            _builder_if_415 = T.If(k == umma_num_k_blocks - 1)
                            _builder_if_415.__enter__()
                            _builder_then_415 = T.Then()
                            _builder_then_415.__enter__()
                            _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_1[0])), 'commit', 'cta_group::2', 'mbarrier::arrive::one', 'shared::cluster', 'b64', ''))
                            _builder_then_415.__exit__(None, None, None)
                            _builder_if_415.__exit__(None, None, None)
                            _builder_then_398.__exit__(None, None, None)
                            _builder_if_398.__exit__(None, None, None)
                            _builder_if_417 = T.If(k > 0)
                            _builder_if_417.__enter__()
                            _builder_then_417 = T.Then()
                            _builder_then_417.__enter__()
                            prev_k = _builder_bind('prev_k', k - 1, type_annotation=T.int32)
                            prev_rs = _builder_bind('prev_rs', umma_rs - 1, type_annotation=T.int32)
                            prev_buf = _builder_bind('prev_buf', prev_rs % 4, type_annotation=T.int32)
                            prev_s_o_phase = _builder_bind('prev_s_o_phase', T.bitwise_and(prev_rs, 1), type_annotation=T.int32)
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_S_O_full_buf[0]), T.bitwise_xor(prev_s_o_phase, 0)))
                            _builder_if_423 = T.If(prev_k == 0)
                            _builder_if_423.__enter__()
                            _builder_then_423 = T.Then()
                            _builder_then_423.__enter__()
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_tOut_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(umma_outer_loop_phase, 1), 0)))
                            _builder_then_423.__exit__(None, None, None)
                            _builder_if_423.__exit__(None, None, None)
                            _builder_emit(T.ptx.tcgen05('fence::after_thread_sync', ''))
                            o_accumulate = _builder_scalar('o_accumulate', T.if_then_else(prev_k == 0, T.uint32(0), T.uint32(1)), dtype='uint32')
                            buffer_13 = _builder_assign('buffer_13', T.decl_buffer((64, 256), scope='tmem', layout=T.TileLayout(T.S[(64, 1, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0))
                            descB_local = _builder_alloc_scalar('descB_local', 'uint64')
                            _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3))
                            descA_local = _builder_alloc_scalar('descA_local', 'uint64')
                            _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA_local), T.address_of(s_smem_gemm[0, 0]), 64, 8, 0))
                            with T.unroll(1) as mi:
                                with T.unroll(1) as ni:
                                    with T.unroll(4) as ki:
                                        _builder_emit(T.ptx.tcgen05(T.Cast('uint32', ni * 128), _add_smem_desc_offset(descA_local, (ki % 4 * 1024 + mi * 512 + ki // 4 * 8) // 8), _add_smem_desc_offset(descB_local, ((ki * 16 + ni) // 64 * 16384 + prev_buf * 16384 + (ki * 16 + ni) % 64 * 64) // 8), T.uint32(138478736), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.Or(ki != 0, T.Cast('bool', o_accumulate)), 'mma', 'cta_group::2', 'kind::f16', 'p12'))
                            buffer_14 = _builder_assign('buffer_14', T.decl_buffer((64, 256), scope='tmem', layout=T.TileLayout(T.S[(64, 1, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=128))
                            descB_local_1 = _builder_alloc_scalar('descB_local_1', 'uint64')
                            _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local_1), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3))
                            descA_local_1 = _builder_alloc_scalar('descA_local_1', 'uint64')
                            _builder_emit(T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA_local_1), T.address_of(s_smem_gemm[0, 0]), 64, 8, 0))
                            with T.unroll(1) as mi:
                                with T.unroll(1) as ni:
                                    with T.unroll(4) as ki:
                                        _builder_emit(T.ptx.tcgen05(T.Cast('uint32', ni * 128 + 128), _add_smem_desc_offset(descA_local_1, (ki % 4 * 1024 + mi * 512 + ki // 4 * 8) // 8), _add_smem_desc_offset(descB_local_1, ((ki * 16 + ni) // 64 * 16384 + prev_buf * 16384 + (ki * 16 + ni) % 64 * 64 + 8192) // 8), T.uint32(138478736), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.Or(ki != 0, T.Cast('bool', o_accumulate)), 'mma', 'cta_group::2', 'kind::f16', 'p12'))
                            T.buffer_store(o_accumulate.buffer, T.uint32(1), [0])
                            _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_7[0])), T.Cast('uint16', 3), 'commit', 'cta_group::2', 'mbarrier::arrive::one', 'shared::cluster', 'multicast::cluster', 'b64', ''))
                            _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_5[prev_buf])), T.Cast('uint16', 3), 'commit', 'cta_group::2', 'mbarrier::arrive::one', 'shared::cluster', 'multicast::cluster', 'b64', ''))
                            _builder_then_417.__exit__(None, None, None)
                            _builder_if_417.__exit__(None, None, None)
                            _builder_if_448 = T.If(k != umma_num_k_blocks)
                            _builder_if_448.__enter__()
                            _builder_then_448 = T.Then()
                            _builder_then_448.__enter__()
                            T.buffer_store(umma_rs.buffer, umma_rs + 1, [0])
                            _builder_then_448.__exit__(None, None, None)
                            _builder_if_448.__exit__(None, None, None)
                        _builder_emit(T.ptx.tcgen05('fence::before_thread_sync', ''))
                        _builder_emit(T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_3[0])), T.Cast('uint16', 3), 'commit', 'cta_group::2', 'mbarrier::arrive::one', 'shared::cluster', 'multicast::cluster', 'b64', ''))
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(umma_outer_loop_phase, 0)))
                        umma_next_job = _builder_alloc_scalar('umma_next_job', 'uint32')
                        _builder_emit(_query_cancel_first_ctaid_x(umma_next_job, T.address_of(clc_response[0])))
                        _rem3 = _builder_alloc_scalar('_rem3', 'uint64')
                        _builder_emit(T.ptx.mapa(_rem3, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), 'shared::cluster', 'u64', ''))
                        _builder_emit(T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem3), T.uint32(1), T.bool(True), 'arrive', '', '', '', 'b64', 'pred'))
                        _builder_if_458 = T.If(umma_next_job == T.uint32(4294967295))
                        _builder_if_458.__enter__()
                        _builder_then_458 = T.Then()
                        _builder_then_458.__enter__()
                        T.buffer_store(umma_job_valid.buffer, 0, [0])
                        _builder_then_458.__exit__(None, None, None)
                        _builder_else_458 = T.Else()
                        _builder_else_458.__enter__()
                        T.buffer_store(umma_job_block_idx.buffer, T.Cast('int32', umma_next_job), [0])
                        _builder_else_458.__exit__(None, None, None)
                        _builder_if_458.__exit__(None, None, None)
                        T.buffer_store(umma_outer_loop_phase.buffer, T.bitwise_xor(umma_outer_loop_phase, 1), [0])
                    _builder_then_385.__exit__(None, None, None)
                    _builder_if_385.__exit__(None, None, None)
                    _builder_emit(T.cuda.iket.range_end(mma_token))
                    _builder_then_383.__exit__(None, None, None)
                    _builder_else_383 = T.Else()
                    _builder_else_383.__enter__()
                    _builder_if_465 = T.If(warp_idx == 9)
                    _builder_if_465.__enter__()
                    _builder_then_465 = T.Then()
                    _builder_then_465.__enter__()
                    valid_mask_token = _builder_scalar('valid_mask_token', T.cuda.iket.range_start('h128-small-valid-mask'), dtype='uint32')
                    _builder_if_467 = T.If(lane_idx < 8)
                    _builder_if_467.__enter__()
                    _builder_then_467 = T.Then()
                    _builder_then_467.__enter__()
                    lane_indices = _builder_assign('lane_indices', T.alloc_local((8,), 'int32'))
                    valid_job_valid = _builder_scalar('valid_job_valid', 1, dtype='int32')
                    valid_job_block_idx = _builder_scalar('valid_job_block_idx', block_idx, dtype='int32')
                    valid_outer_loop_phase = _builder_scalar('valid_outer_loop_phase', 0, dtype='int32')
                    valid_rs = _builder_scalar('valid_rs', 0, dtype='int32')
                    with T.While(valid_job_valid != 0):
                        valid_s_q_idx = _builder_bind('valid_s_q_idx', valid_job_block_idx // 2, type_annotation=T.int32)
                        valid_topk_len = _builder_scalar('valid_topk_len', topk, dtype='int32')
                        if have_topk_length:
                            _builder_emit(T.ptx.ld.global_.s32(valid_topk_len, topk_length.ptr_to([valid_s_q_idx])))
                        valid_num_k_blocks = _builder_bind('valid_num_k_blocks', T.max((valid_topk_len + 64 - 1) // 64, 1), type_annotation=T.int32)
                        valid_g_indices_base = _builder_bind('valid_g_indices_base', valid_s_q_idx * stride_indices_s_q, type_annotation=T.int32)
                        with T.serial(valid_num_k_blocks, unroll=False) as k:
                            row_base = _builder_bind('row_base', valid_g_indices_base + k * 64 + lane_idx * 8, type_annotation=T.int32)
                            buffer_13 = _builder_assign('buffer_13', T.decl_buffer((8,), 'int32', data=lane_indices.data, scope='local'))
                            buffer_14 = _builder_assign('buffer_14', buffer_13.view('uint32'))
                            _builder_emit(T.ptx.ld(buffer_14[0], buffer_14[1], buffer_14[2], buffer_14[3], buffer_14[4], buffer_14[5], buffer_14[6], buffer_14[7], T.address_of(indices[row_base]), '', '', 'global', '', 'nc', 'L1::no_allocate', 'L2::evict_normal', 'L2::256B', 'v8', 'u32', ''))
                            abs_pos_start = _builder_bind('abs_pos_start', k * 64, type_annotation=T.int32)
                            mask = _builder_bind('mask', T.Cast('int8', T.bitwise_or(T.bitwise_or(T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[0] >= 0, lane_indices[0] < s_kv), abs_pos_start + lane_idx * 8 < valid_topk_len), 1, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[1] >= 0, lane_indices[1] < s_kv), abs_pos_start + lane_idx * 8 + 1 < valid_topk_len), 2, 0)), T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[2] >= 0, lane_indices[2] < s_kv), abs_pos_start + lane_idx * 8 + 2 < valid_topk_len), 4, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[3] >= 0, lane_indices[3] < s_kv), abs_pos_start + lane_idx * 8 + 3 < valid_topk_len), 8, 0))), T.bitwise_or(T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[4] >= 0, lane_indices[4] < s_kv), abs_pos_start + lane_idx * 8 + 4 < valid_topk_len), 16, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[5] >= 0, lane_indices[5] < s_kv), abs_pos_start + lane_idx * 8 + 5 < valid_topk_len), 32, 0)), T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[6] >= 0, lane_indices[6] < s_kv), abs_pos_start + lane_idx * 8 + 6 < valid_topk_len), 64, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[7] >= 0, lane_indices[7] < s_kv), abs_pos_start + lane_idx * 8 + 7 < valid_topk_len), 128, 0))))), type_annotation=T.int8)
                            index_buf_idx = _builder_bind('index_buf_idx', valid_rs % 4, type_annotation=T.int32)
                            index_bar_phase = _builder_bind('index_bar_phase', T.bitwise_and(valid_rs // 4, 1), type_annotation=T.int32)
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_valid_coord_scales_empty_buf[index_buf_idx]), T.bitwise_xor(T.bitwise_xor(index_bar_phase, 1), 0)))
                            _builder_emit(T.ptx.st.shared.b8(is_k_valid.ptr_to([index_buf_idx, lane_idx]), T.reinterpret('uint8', mask)))
                            _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_full_buf[index_buf_idx])), T.uint32(1), 'arrive', '', '', 'shared', 'b64', ''))
                            T.buffer_store(valid_rs.buffer, valid_rs + 1, [0])
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(valid_outer_loop_phase, 0)))
                        valid_next_job = _builder_alloc_scalar('valid_next_job', 'uint32')
                        _builder_emit(_query_cancel_first_ctaid_x(valid_next_job, T.address_of(clc_response[0])))
                        _rem4 = _builder_alloc_scalar('_rem4', 'uint64')
                        _builder_emit(T.ptx.mapa(_rem4, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), 'shared::cluster', 'u64', ''))
                        _builder_emit(T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem4), T.uint32(1), T.bool(True), 'arrive', '', '', '', 'b64', 'pred'))
                        _builder_if_499 = T.If(valid_next_job == T.uint32(4294967295))
                        _builder_if_499.__enter__()
                        _builder_then_499 = T.Then()
                        _builder_then_499.__enter__()
                        T.buffer_store(valid_job_valid.buffer, 0, [0])
                        _builder_then_499.__exit__(None, None, None)
                        _builder_else_499 = T.Else()
                        _builder_else_499.__enter__()
                        T.buffer_store(valid_job_block_idx.buffer, T.Cast('int32', valid_next_job), [0])
                        _builder_else_499.__exit__(None, None, None)
                        _builder_if_499.__exit__(None, None, None)
                        T.buffer_store(valid_outer_loop_phase.buffer, T.bitwise_xor(valid_outer_loop_phase, 1), [0])
                    _builder_then_467.__exit__(None, None, None)
                    _builder_if_467.__exit__(None, None, None)
                    _builder_emit(T.cuda.iket.range_end(valid_mask_token))
                    _builder_then_465.__exit__(None, None, None)
                    _builder_else_465 = T.Else()
                    _builder_else_465.__enter__()
                    _builder_if_506 = T.If(warp_idx >= 10)
                    _builder_if_506.__enter__()
                    _builder_then_506 = T.Then()
                    _builder_then_506.__enter__()
                    clc_token = _builder_scalar('clc_token', T.cuda.iket.sentinel_token('h128-small-clc'), dtype='uint32')
                    _builder_if_508 = T.If(warp_idx == 10)
                    _builder_if_508.__enter__()
                    _builder_then_508 = T.Then()
                    _builder_then_508.__enter__()
                    T.buffer_store(clc_token.buffer, T.cuda.iket.range_start('h128-small-clc'), [0])
                    _builder_then_508.__exit__(None, None, None)
                    _builder_if_508.__exit__(None, None, None)
                    _builder_if_510 = T.If(T.cuda.elect_sync())
                    _builder_if_510.__enter__()
                    _builder_then_510 = T.Then()
                    _builder_then_510.__enter__()
                    _builder_if_511 = T.If(warp_idx == 10)
                    _builder_if_511.__enter__()
                    _builder_then_511 = T.Then()
                    _builder_then_511.__enter__()
                    clc_job_valid = _builder_scalar('clc_job_valid', 1, dtype='int32')
                    clc_outer_loop_phase = _builder_scalar('clc_outer_loop_phase', 0, dtype='int32')
                    with T.While(clc_job_valid != 0):
                        _builder_if_515 = T.If(cta_idx == 0)
                        _builder_if_515.__enter__()
                        _builder_then_515 = T.Then()
                        _builder_then_515.__enter__()
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_clc_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(clc_outer_loop_phase, 1), 0)))
                        _builder_emit(T.ptx.clusterlaunchcontrol(T.cuda.cvta_generic_to_shared(T.address_of(clc_response[0])), T.cuda.cvta_generic_to_shared(T.address_of(buffer_8[0])), 'try_cancel', 'async', 'shared::cta', 'mbarrier::complete_tx::bytes', 'multicast::cluster::all', 'b128', ''))
                        _builder_then_515.__exit__(None, None, None)
                        _builder_if_515.__exit__(None, None, None)
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_8[0])), T.uint32(16), 'arrive', 'expect_tx', '', '', 'shared', 'b64', ''))
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(clc_outer_loop_phase, 0)))
                        clc_next_job = _builder_alloc_scalar('clc_next_job', 'uint32')
                        _builder_emit(_query_cancel_first_ctaid_x(clc_next_job, T.address_of(clc_response[0])))
                        _rem5 = _builder_alloc_scalar('_rem5', 'uint64')
                        _builder_emit(T.ptx.mapa(_rem5, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), 'shared::cluster', 'u64', ''))
                        _builder_emit(T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem5), T.uint32(1), T.bool(True), 'arrive', '', '', '', 'b64', 'pred'))
                        _builder_if_525 = T.If(clc_next_job == T.uint32(4294967295))
                        _builder_if_525.__enter__()
                        _builder_then_525 = T.Then()
                        _builder_then_525.__enter__()
                        T.buffer_store(clc_job_valid.buffer, 0, [0])
                        _builder_then_525.__exit__(None, None, None)
                        _builder_if_525.__exit__(None, None, None)
                        T.buffer_store(clc_outer_loop_phase.buffer, T.bitwise_xor(clc_outer_loop_phase, 1), [0])
                    _builder_then_511.__exit__(None, None, None)
                    _builder_if_511.__exit__(None, None, None)
                    _builder_then_510.__exit__(None, None, None)
                    _builder_if_510.__exit__(None, None, None)
                    _builder_emit(T.cuda.iket.range_end(clc_token))
                    _builder_then_506.__exit__(None, None, None)
                    _builder_if_506.__exit__(None, None, None)
                    _builder_else_465.__exit__(None, None, None)
                    _builder_if_465.__exit__(None, None, None)
                    _builder_else_383.__exit__(None, None, None)
                    _builder_if_383.__exit__(None, None, None)
                    _builder_then_381.__exit__(None, None, None)
                    _builder_else_381 = T.Else()
                    _builder_else_381.__enter__()
                    softmax_token = _builder_scalar('softmax_token', T.cuda.iket.range_start('h128-small-softmax'), dtype='uint32')
                    _builder_emit(T.ptx.setmaxnreg(160, 'inc', 'sync', 'aligned', 'u32', ''))
                    local_warp_idx = _builder_bind('local_warp_idx', warp_idx - 12, type_annotation=T.int32)
                    wg3_job_valid = _builder_scalar('wg3_job_valid', 1, dtype='int32')
                    wg3_job_block_idx = _builder_scalar('wg3_job_block_idx', block_idx, dtype='int32')
                    wg3_outer_loop_phase = _builder_scalar('wg3_outer_loop_phase', 0, dtype='int32')
                    wg3_rs = _builder_scalar('wg3_rs', 0, dtype='int32')
                    with T.While(wg3_job_valid != 0):
                        wg3_s_q_idx = _builder_bind('wg3_s_q_idx', wg3_job_block_idx // 2, type_annotation=T.int32)
                        wg3_topk_len = _builder_scalar('wg3_topk_len', topk, dtype='int32')
                        if have_topk_length:
                            _builder_emit(T.ptx.ld.global_.s32(wg3_topk_len, topk_length.ptr_to([wg3_s_q_idx])))
                        wg3_num_k_blocks = _builder_bind('wg3_num_k_blocks', T.max((wg3_topk_len + 64 - 1) // 64, 1), type_annotation=T.int32)
                        mi = _builder_scalar('mi', T.float32(-1e+30), dtype='float32')
                        li = _builder_scalar('li', T.float32(0.0), dtype='float32')
                        real_mi = _builder_scalar('real_mi', T.float32('-inf'), dtype='float32')
                        scale_pair = _builder_bind('scale_pair', T.cuda.make_float2(T.float32(sm_scale_div_log2), T.float32(sm_scale_div_log2)), type_annotation=T.uint64)
                        with T.serial(wg3_num_k_blocks, unroll=False) as k:
                            k_buf_idx = _builder_bind('k_buf_idx', wg3_rs % 4, type_annotation=T.int32)
                            k_bar_phase = _builder_bind('k_bar_phase', T.bitwise_and(wg3_rs // 4, 1), type_annotation=T.int32)
                            index_buf_idx = _builder_bind('index_buf_idx', wg3_rs % 4, type_annotation=T.int32)
                            index_bar_phase = _builder_bind('index_bar_phase', T.bitwise_and(wg3_rs // 4, 1), type_annotation=T.int32)
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_valid_coord_scales_full_buf[index_buf_idx]), T.bitwise_xor(index_bar_phase, 0)))
                            buffer_13 = _builder_assign('buffer_13', T.alloc_local((32,)))
                            p_frag = _builder_assign('p_frag', buffer_13.view(128, 32, layout=T.TileLayout(T.S[(128, 32):(1 @ Axis.tid_in_wg, 1)])))
                            buffer_14 = _builder_assign('buffer_14', T.alloc_local((32,)))
                            p_peer_frag = _builder_assign('p_peer_frag', buffer_14.view(128, 32, layout=T.TileLayout(T.S[(128, 32):(1 @ Axis.tid_in_wg, 1)])))
                            buffer_15 = _builder_assign('buffer_15', p_frag.local())
                            p = _builder_assign('p', buffer_15.view('uint32'))
                            buffer_16 = _builder_assign('buffer_16', p_peer_frag.local())
                            p_peer = _builder_assign('p_peer', buffer_16.view('uint32'))
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_6[0]), T.bitwise_xor(T.bitwise_and(wg3_rs, 1), 0)))
                            _builder_emit(T.ptx.tcgen05('fence::after_thread_sync', ''))
                            _builder_if_563 = T.If(local_warp_idx < 2)
                            _builder_if_563.__enter__()
                            _builder_then_563 = T.Then()
                            _builder_then_563.__enter__()
                            buffer_17 = _builder_assign('buffer_17', T.decl_buffer((64, 2, 64), scope='tmem', layout=T.TileLayout(T.S[(64, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                            buffer_18 = _builder_assign('buffer_18', T.decl_buffer((2, 64, 64), scope='tmem', layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                            p_win = _builder_assign('p_win', T.decl_buffer((128, 64), scope='tmem', layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                            local_storage = _builder_assign('local_storage', p_frag.local())
                            _builder_emit(T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], T.uint32(384), 'ld', 'sync', 'aligned', '32x32b', 'x32', '', 'b32', ''))
                            local_storage_1 = _builder_assign('local_storage_1', p_peer_frag.local())
                            _builder_emit(T.ptx.tcgen05(local_storage_1[0], local_storage_1[1], local_storage_1[2], local_storage_1[3], local_storage_1[4], local_storage_1[5], local_storage_1[6], local_storage_1[7], local_storage_1[8], local_storage_1[9], local_storage_1[10], local_storage_1[11], local_storage_1[12], local_storage_1[13], local_storage_1[14], local_storage_1[15], local_storage_1[16], local_storage_1[17], local_storage_1[18], local_storage_1[19], local_storage_1[20], local_storage_1[21], local_storage_1[22], local_storage_1[23], local_storage_1[24], local_storage_1[25], local_storage_1[26], local_storage_1[27], local_storage_1[28], local_storage_1[29], local_storage_1[30], local_storage_1[31], T.cuda.get_tmem_addr(T.uint32(384), 0, 32), 'ld', 'sync', 'aligned', '32x32b', 'x32', '', 'b32', ''))
                            _builder_then_563.__exit__(None, None, None)
                            _builder_else_563 = T.Else()
                            _builder_else_563.__enter__()
                            buffer_17 = _builder_assign('buffer_17', T.decl_buffer((64, 2, 64), scope='tmem', layout=T.TileLayout(T.S[(64, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                            buffer_18 = _builder_assign('buffer_18', T.decl_buffer((2, 64, 64), scope='tmem', layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                            p_win = _builder_assign('p_win', T.decl_buffer((128, 64), scope='tmem', layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384))
                            local_storage = _builder_assign('local_storage', p_peer_frag.local())
                            _builder_emit(T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], T.uint32(384), 'ld', 'sync', 'aligned', '32x32b', 'x32', '', 'b32', ''))
                            local_storage_1 = _builder_assign('local_storage_1', p_frag.local())
                            _builder_emit(T.ptx.tcgen05(local_storage_1[0], local_storage_1[1], local_storage_1[2], local_storage_1[3], local_storage_1[4], local_storage_1[5], local_storage_1[6], local_storage_1[7], local_storage_1[8], local_storage_1[9], local_storage_1[10], local_storage_1[11], local_storage_1[12], local_storage_1[13], local_storage_1[14], local_storage_1[15], local_storage_1[16], local_storage_1[17], local_storage_1[18], local_storage_1[19], local_storage_1[20], local_storage_1[21], local_storage_1[22], local_storage_1[23], local_storage_1[24], local_storage_1[25], local_storage_1[26], local_storage_1[27], local_storage_1[28], local_storage_1[29], local_storage_1[30], local_storage_1[31], T.cuda.get_tmem_addr(T.uint32(384), 0, 32), 'ld', 'sync', 'aligned', '32x32b', 'x32', '', 'b32', ''))
                            _builder_else_563.__exit__(None, None, None)
                            _builder_if_563.__exit__(None, None, None)
                            _builder_emit(T.ptx.tcgen05('wait::ld', 'sync', 'aligned', ''))
                            _builder_emit(T.ptx.tcgen05('fence::before_thread_sync', ''))
                            buffer_17 = _builder_alloc_scalar('buffer_17', 'uint32')
                            _builder_emit(T.ptx.mapa(buffer_17, T.cuda.cvta_generic_to_shared(T.address_of(bar_P_empty_buf[0])), T.uint32(0), 'shared::cluster', 'u32', ''))
                            _builder_emit(T.ptx.mbarrier(buffer_17, 'arrive', '', '', 'shared::cluster', 'b64', ''))
                            valid_word_offset = _builder_bind('valid_word_offset', T.if_then_else(local_warp_idx >= 2, 1, 0), type_annotation=T.int32)
                            buffer_18 = _builder_assign('buffer_18', T.decl_buffer((4, 2), 'uint32', data=is_k_valid.data, elem_offset=55552, scope='shared.dyn', align=16))
                            is_k_valid_u32 = _builder_alloc_scalar('is_k_valid_u32', 'uint32')
                            _builder_emit(T.ptx.ld.shared.u32(is_k_valid_u32, buffer_18.ptr_to([index_buf_idx, valid_word_offset])))
                            with T.unroll(32) as p_i:
                                invalid_p_predicate = _builder_bind('invalid_p_predicate', T.bitwise_and(T.shift_right(is_k_valid_u32, T.Cast('uint32', p_i)), T.uint32(1)) == T.uint32(0), type_annotation=T.bool)
                                T.buffer_store(p, T.if_then_else(invalid_p_predicate, T.uint32(4286578688), p[p_i]), [p_i])
                            sum_pair0 = _builder_alloc_scalar('sum_pair0', 'uint64')
                            sum_pair1 = _builder_alloc_scalar('sum_pair1', 'uint64')
                            with T.unroll(8) as exchange_i:
                                exchange_offset = _builder_scalar('exchange_offset', exchange_i * 32 * 4 + lane_idx * 4, dtype='int32')
                                p_peer_offset = _builder_bind('p_peer_offset', exchange_i * 4, type_annotation=T.int32)
                                buffer_19 = _builder_assign('buffer_19', T.decl_buffer((32,), 'uint32', data=p_peer.data, scope='local', layout=T.TileLayout(T.S[(32, 1):(1, 1)])))
                                buffer_20 = _builder_assign('buffer_20', T.decl_buffer((32,), 'uint32', data=buffer_19.data, scope='local', layout=T.TileLayout(T.S[(32, 1, 1):(1, 1, 1)])))
                                _builder_emit(T.ptx.st(T.cuda.cvta_generic_to_shared(T.address_of(p_exchange[T.bitwise_xor(local_warp_idx, 2), exchange_offset])), buffer_20[p_peer_offset], buffer_20[p_peer_offset + 1], buffer_20[p_peer_offset + 2], buffer_20[p_peer_offset + 3], '', '', 'shared', '', '', '', 'v4', 'u32', ''))
                            _builder_emit(T.ptx.bar(T.Cast('uint32', 2 + T.bitwise_and(local_warp_idx, 1)), T.uint32(64), '', 'sync', ''))
                            with T.unroll(8) as exchange_i:
                                exchange_offset = _builder_scalar('exchange_offset', exchange_i * 32 * 4 + lane_idx * 4, dtype='int32')
                                p_exchange_tmp = _builder_assign('p_exchange_tmp', T.alloc_local((4,), 'uint32'))
                                buffer_19 = _builder_assign('buffer_19', T.decl_buffer((4,), 'uint32', data=p_exchange_tmp.data, scope='local'))
                                buffer_20 = _builder_assign('buffer_20', T.decl_buffer((4,), 'uint32', data=buffer_19.data, scope='local', layout=T.TileLayout(T.S[(4, 1):(1, 1)])))
                                _builder_emit(T.ptx.ld(buffer_20[0], buffer_20[1], buffer_20[2], buffer_20[3], T.cuda.cvta_generic_to_shared(T.address_of(p_exchange[local_warp_idx, exchange_offset])), '', '', 'shared', '', '', '', '', '', 'v4', 'u32', ''))
                                p_pair0 = _builder_bind('p_pair0', T.cuda.make_float2(T.cuda.uint_as_float(p[exchange_i * 4]), T.cuda.uint_as_float(p[exchange_i * 4 + 1])), type_annotation=T.uint64)
                                peer_pair0 = _builder_bind('peer_pair0', T.cuda.make_float2(T.cuda.uint_as_float(p_exchange_tmp[0]), T.cuda.uint_as_float(p_exchange_tmp[1])), type_annotation=T.uint64)
                                _builder_emit(T.ptx.add(sum_pair0, p_pair0, peer_pair0, 'rn', '', '', 'f32x2', '', ''))
                                T.buffer_store(p, T.cuda.float_as_uint(T.cuda.float2_x(sum_pair0)), [exchange_i * 4])
                                T.buffer_store(p, T.cuda.float_as_uint(T.cuda.float2_y(sum_pair0)), [exchange_i * 4 + 1])
                                p_pair1 = _builder_bind('p_pair1', T.cuda.make_float2(T.cuda.uint_as_float(p[exchange_i * 4 + 2]), T.cuda.uint_as_float(p[exchange_i * 4 + 3])), type_annotation=T.uint64)
                                peer_pair1 = _builder_bind('peer_pair1', T.cuda.make_float2(T.cuda.uint_as_float(p_exchange_tmp[2]), T.cuda.uint_as_float(p_exchange_tmp[3])), type_annotation=T.uint64)
                                _builder_emit(T.ptx.add(sum_pair1, p_pair1, peer_pair1, 'rn', '', '', 'f32x2', '', ''))
                                T.buffer_store(p, T.cuda.float_as_uint(T.cuda.float2_x(sum_pair1)), [exchange_i * 4 + 2])
                                T.buffer_store(p, T.cuda.float_as_uint(T.cuda.float2_y(sum_pair1)), [exchange_i * 4 + 3])
                            cur_pi_max = _builder_scalar('cur_pi_max', T.float32('-inf'), dtype='float32')
                            with T.unroll(32) as p_i:
                                T.buffer_store(cur_pi_max.buffer, T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i])), [0])
                            T.buffer_store(cur_pi_max.buffer, cur_pi_max * T.float32(sm_scale_div_log2), [0])
                            _builder_emit(T.ptx.st.shared.f32(rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max))
                            _builder_emit(T.ptx.bar(T.Cast('uint32', 2 + T.bitwise_and(local_warp_idx, 1)), T.uint32(64), '', 'sync', ''))
                            peer_pi_max = _builder_alloc_scalar('peer_pi_max', 'float32')
                            _builder_emit(T.ptx.ld.shared.f32(peer_pi_max, rowwise_max_buf.ptr_to([T.bitwise_xor(idx_in_warpgroup, 64)])))
                            T.buffer_store(cur_pi_max.buffer, T.max(cur_pi_max, peer_pi_max), [0])
                            T.buffer_store(real_mi.buffer, T.max(real_mi, cur_pi_max), [0])
                            should_scale_o = _builder_bind('should_scale_o', T.cuda.any_sync(T.uint32(4294967295), cur_pi_max - mi > T.float32(6.0)) != 0, type_annotation=T.bool)
                            new_max = _builder_alloc_scalar('new_max', 'float32')
                            scale_for_old = _builder_alloc_scalar('scale_for_old', 'float32')
                            _builder_if_629 = T.If(T.Not(should_scale_o))
                            _builder_if_629.__enter__()
                            _builder_then_629 = T.Then()
                            _builder_then_629.__enter__()
                            T.buffer_store(scale_for_old.buffer, T.float32(1.0), [0])
                            T.buffer_store(new_max.buffer, mi, [0])
                            _builder_then_629.__exit__(None, None, None)
                            _builder_else_629 = T.Else()
                            _builder_else_629.__enter__()
                            T.buffer_store(new_max.buffer, T.max(cur_pi_max, mi), [0])
                            _builder_emit(T.ptx.ex2(scale_for_old, mi - new_max, 'approx', 'ftz', 'f32', ''))
                            _builder_else_629.__exit__(None, None, None)
                            _builder_if_629.__exit__(None, None, None)
                            T.buffer_store(mi.buffer, new_max, [0])
                            s_frag = _builder_assign('s_frag', T.alloc_local((64, 64), 'bfloat16', layout=T.TileLayout(T.S[(2, 32, 2, 32):(1 @ Axis.wid_in_wg, 1 @ Axis.laneid, 2 @ Axis.wid_in_wg, 1)])))
                            buffer_19 = _builder_assign('buffer_19', s_frag.local())
                            s_pack = _builder_assign('s_pack', buffer_19.view('uint32'))
                            cur_sum_pair = _builder_scalar('cur_sum_pair', T.cuda.make_float2(T.float32(0.0), T.float32(0.0)), dtype='uint64')
                            neg_new_max_pair = _builder_bind('neg_new_max_pair', T.cuda.make_float2(new_max * T.float32(-1.0), new_max * T.float32(-1.0)), type_annotation=T.uint64)
                            fma_pair = _builder_alloc_scalar('fma_pair', 'uint64')
                            with T.unroll(16) as s_i:
                                p_pair = _builder_bind('p_pair', T.cuda.make_float2(T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])), type_annotation=T.uint64)
                                _builder_emit(T.ptx.fma(fma_pair, p_pair, scale_pair, neg_new_max_pair, 'rn', '', '', 'f32x2', '', ''))
                                s_x = _builder_alloc_scalar('s_x', 'float32')
                                s_y = _builder_alloc_scalar('s_y', 'float32')
                                _builder_emit(T.ptx.ex2(s_x, T.cuda.float2_x(fma_pair), 'approx', 'ftz', 'f32', ''))
                                _builder_emit(T.ptx.ex2(s_y, T.cuda.float2_y(fma_pair), 'approx', 'ftz', 'f32', ''))
                                s_pair = _builder_bind('s_pair', T.cuda.make_float2(s_x, s_y), type_annotation=T.uint64)
                                _builder_emit(T.ptx.add(cur_sum_pair, cur_sum_pair, s_pair, 'rn', '', '', 'f32x2', '', ''))
                                T.buffer_store(s_pack, T.cuda.float22bfloat162_rn(s_x, s_y), [s_i])
                            cur_sum = _builder_bind('cur_sum', T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair), type_annotation=T.float32)
                            li_tmp = _builder_alloc_scalar('li_tmp', 'float32')
                            _builder_emit(T.ptx.fma(li_tmp, li, scale_for_old, cur_sum, 'rn', '', '', 'f32', '', ''))
                            T.buffer_store(li.buffer, li_tmp, [0])
                            _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_7[0]), T.bitwise_xor(T.bitwise_xor(T.bitwise_and(wg3_rs, 1), 1), 0)))
                            _builder_emit(T.ptx.fence('proxy', 'async', 'shared::cta', ''))
                            s_base = _builder_scalar('s_base', v_7 // 64 * 2048 + v_7 % 64 * 8, dtype='int32')
                            r_local = _builder_assign('r_local', s_frag.local())
                            r_words = _builder_assign('r_words', r_local.view('uint32'))
                            with T.serial(4) as f:
                                ds = _builder_scalar('ds', f % 4 * 512, dtype='int32')
                                dr = _builder_scalar('dr', f % 4 * 8, dtype='int32')
                                s_ptr = _builder_bind('s_ptr', T.ptr_byte_offset(T.address_of(s_smem_gemm[0, 0]), (s_base + ds) * BF16_BYTES, 'bfloat16'))
                                r_w = _builder_scalar('r_w', dr // 2, dtype='int32')
                                _builder_emit(T.ptx.st(T.cuda.cvta_generic_to_shared(s_ptr), r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], '', '', 'shared', '', '', '', 'v4', 'u32', ''))
                            _builder_if_671 = T.If(T.bitwise_and(k > 0, should_scale_o))
                            _builder_if_671.__enter__()
                            _builder_then_671 = T.Then()
                            _builder_then_671.__enter__()
                            _builder_emit(T.ptx.tcgen05('fence::after_thread_sync', ''))
                            buffer_20 = _builder_assign('buffer_20', T.alloc_local((32,)))
                            o_rescale_frag = _builder_assign('o_rescale_frag', buffer_20.view(128, 32, layout=T.TileLayout(T.S[(128, 32):(1 @ Axis.tid_in_wg, 1)])))
                            with T.unroll(8) as chunk_idx:
                                local_storage = _builder_assign('local_storage', o_rescale_frag.local())
                                _builder_emit(T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], T.cuda.get_tmem_addr(T.uint32(0), 0, chunk_idx * 32), 'ld', 'sync', 'aligned', '32x32b', 'x32', '', 'b32', ''))
                                _builder_emit(T.ptx.tcgen05('wait::ld', 'sync', 'aligned', ''))
                                buffer_21 = _builder_assign('buffer_21', o_rescale_frag.local(layout=T.TileLayout(T.S[(16, 2):(2, 1)])))
                                buffer_22 = _builder_assign('buffer_22', o_rescale_frag.local(layout=T.TileLayout(T.S[(16, 2):(2, 1)])))
                                with T.serial(16) as f:
                                    dst_lane_indices_0_0 = _builder_scalar('dst_lane_indices_0_0', f * 2, dtype='int32')
                                    dst_lane_indices_1_0 = _builder_scalar('dst_lane_indices_1_0', f * 2 + 1, dtype='int32')
                                    buffer_23 = _builder_alloc_scalar('buffer_23', 'uint64')
                                    buffer_24 = _builder_alloc_scalar('buffer_24', 'uint64')
                                    _builder_emit(T.ptx.mov(buffer_23, buffer_22[f * 2], buffer_22[f * 2 + 1], 'b64', ''))
                                    _builder_emit(T.ptx.mov(buffer_24, scale_for_old, scale_for_old, 'b64', ''))
                                    _builder_emit(T.ptx.mul(buffer_23, buffer_23, buffer_24, 'rz', 'ftz', '', 'f32x2', ''))
                                    _builder_emit(T.ptx.mov(buffer_21[f * 2], buffer_21[f * 2 + 1], buffer_23, 'b64', ''))
                                local_storage_1 = _builder_assign('local_storage_1', o_rescale_frag.local())
                                _builder_emit(T.ptx.tcgen05(T.cuda.get_tmem_addr(T.uint32(0), 0, chunk_idx * 32), local_storage_1[0], local_storage_1[1], local_storage_1[2], local_storage_1[3], local_storage_1[4], local_storage_1[5], local_storage_1[6], local_storage_1[7], local_storage_1[8], local_storage_1[9], local_storage_1[10], local_storage_1[11], local_storage_1[12], local_storage_1[13], local_storage_1[14], local_storage_1[15], local_storage_1[16], local_storage_1[17], local_storage_1[18], local_storage_1[19], local_storage_1[20], local_storage_1[21], local_storage_1[22], local_storage_1[23], local_storage_1[24], local_storage_1[25], local_storage_1[26], local_storage_1[27], local_storage_1[28], local_storage_1[29], local_storage_1[30], local_storage_1[31], 'st', 'sync', 'aligned', '32x32b', 'x32', '', 'b32', ''))
                                _builder_emit(T.ptx.tcgen05('wait::st', 'sync', 'aligned', ''))
                            _builder_emit(T.ptx.tcgen05('fence::before_thread_sync', ''))
                            _builder_then_671.__exit__(None, None, None)
                            _builder_if_671.__exit__(None, None, None)
                            _builder_emit(T.ptx.fence('proxy', 'async', 'shared::cta', ''))
                            buffer_20 = _builder_alloc_scalar('buffer_20', 'uint32')
                            _builder_emit(T.ptx.mapa(buffer_20, T.cuda.cvta_generic_to_shared(T.address_of(bar_S_O_full_buf[0])), T.uint32(0), 'shared::cluster', 'u32', ''))
                            _builder_emit(T.ptx.mbarrier(buffer_20, 'arrive', '', '', 'shared::cluster', 'b64', ''))
                            _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_empty_buf[index_buf_idx])), T.uint32(1), 'arrive', '', '', 'shared', 'b64', ''))
                            T.buffer_store(wg3_rs.buffer, wg3_rs + 1, [0])
                        _builder_if_700 = T.If(real_mi == T.float32('-inf'))
                        _builder_if_700.__enter__()
                        _builder_then_700 = T.Then()
                        _builder_then_700.__enter__()
                        T.buffer_store(li.buffer, T.float32(0.0), [0])
                        T.buffer_store(mi.buffer, T.float32('-inf'), [0])
                        _builder_then_700.__exit__(None, None, None)
                        _builder_if_700.__exit__(None, None, None)
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(bar_li_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(wg3_outer_loop_phase, 1), 0)))
                        _builder_emit(T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([T.bitwise_xor(idx_in_warpgroup, 64)]), li))
                        _builder_emit(T.ptx.bar(T.uint32(1), T.uint32(128), '', 'sync', ''))
                        peer_li = _builder_alloc_scalar('peer_li', 'float32')
                        _builder_emit(T.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup])))
                        T.buffer_store(li.buffer, li + peer_li, [0])
                        _builder_if_709 = T.If(idx_in_warpgroup < 64)
                        _builder_if_709.__enter__()
                        _builder_then_709 = T.Then()
                        _builder_then_709.__enter__()
                        head_idx = _builder_bind('head_idx', cta_idx * 64 + idx_in_warpgroup, type_annotation=T.int32)
                        attn_sink_value = _builder_scalar('attn_sink_value', T.float32(-float('inf')), dtype='float32')
                        if have_attn_sink:
                            _builder_emit(T.ptx.ld.global_.f32(attn_sink_value, attn_sink.ptr_to([head_idx])))
                        attn_sink_log2 = _builder_bind('attn_sink_log2', attn_sink_value * T.float32(1.4426950408889634), type_annotation=T.float32)
                        sink_exp = _builder_alloc_scalar('sink_exp', 'float32')
                        _builder_emit(T.ptx.ex2(sink_exp, attn_sink_log2 - mi, 'approx', 'ftz', 'f32', ''))
                        output_scale = _builder_bind('output_scale', T.cuda.fdividef(T.float32(1.0), li + sink_exp), type_annotation=T.float32)
                        _builder_emit(T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), T.if_then_else(li == T.float32(0.0), T.float32(0.0), output_scale)))
                        _builder_emit(T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_full_buf[0])), T.uint32(1), 'arrive', '', '', 'shared', 'b64', ''))
                        cur_lse = _builder_alloc_scalar('cur_lse', 'float32')
                        _builder_emit(T.ptx.fma(cur_lse, mi, T.float32(0.6931471805599453), T.log(li), 'rn', '', '', 'f32', '', ''))
                        T.buffer_store(cur_lse.buffer, T.if_then_else(cur_lse == T.float32('-inf'), T.float32('inf'), cur_lse), [0])
                        _builder_emit(T.ptx.st.global_.f32(max_logits.ptr_to([wg3_s_q_idx, head_idx]), real_mi * T.float32(0.6931471805599453)))
                        _builder_emit(T.ptx.st.global_.f32(lse.ptr_to([wg3_s_q_idx, head_idx]), cur_lse))
                        _builder_then_709.__exit__(None, None, None)
                        _builder_if_709.__exit__(None, None, None)
                        _builder_emit(T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(wg3_outer_loop_phase, 0)))
                        wg3_next_job = _builder_alloc_scalar('wg3_next_job', 'uint32')
                        _builder_emit(_query_cancel_first_ctaid_x(wg3_next_job, T.address_of(clc_response[0])))
                        _rem6 = _builder_alloc_scalar('_rem6', 'uint64')
                        _builder_emit(T.ptx.mapa(_rem6, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), 'shared::cluster', 'u64', ''))
                        _builder_emit(T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem6), T.uint32(1), T.bool(True), 'arrive', '', '', '', 'b64', 'pred'))
                        _builder_if_731 = T.If(wg3_next_job == T.uint32(4294967295))
                        _builder_if_731.__enter__()
                        _builder_then_731 = T.Then()
                        _builder_then_731.__enter__()
                        T.buffer_store(wg3_job_valid.buffer, 0, [0])
                        _builder_then_731.__exit__(None, None, None)
                        _builder_else_731 = T.Else()
                        _builder_else_731.__enter__()
                        T.buffer_store(wg3_job_block_idx.buffer, T.Cast('int32', wg3_next_job), [0])
                        _builder_else_731.__exit__(None, None, None)
                        _builder_if_731.__exit__(None, None, None)
                        T.buffer_store(wg3_outer_loop_phase.buffer, T.bitwise_xor(wg3_outer_loop_phase, 1), [0])
                    _builder_emit(T.cuda.iket.range_end(softmax_token))
                    _builder_else_381.__exit__(None, None, None)
                    _builder_if_381.__exit__(None, None, None)
                    _builder_else_326.__exit__(None, None, None)
                    _builder_if_326.__exit__(None, None, None)
                    _builder_else_136.__exit__(None, None, None)
                    _builder_if_136.__exit__(None, None, None)
                    _builder_emit(T.cuda.cluster_sync())
    return builder.get()
# fmt: on


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    return _make_low_level_kernel(
        cfg.s_q,
        cfg.s_kv,
        cfg.topk,
        stride_kv_s_kv,
        stride_indices_s_q,
        cfg.have_attn_sink,
        cfg.have_topk_length,
        (1.0 / math.sqrt(cfg.d_qk)) * LOG_2_E,
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
    if not case["dispatch_reason"].startswith("small_topk:"):
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
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1 benchmark")

    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("small_topk:"):
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
