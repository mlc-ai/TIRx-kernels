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

from tvm.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm.script import tirx as T
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
    @T.prim_func
    def _kernel(q: T.Buffer((s_q, 128, 512), "bfloat16"), kv: T.Buffer((s_kv * stride_kv_s_kv,), "bfloat16"), indices: T.Buffer((s_q * stride_indices_s_q,), "int32"), attn_sink: T.Buffer((128,), "float32"), topk_length: T.Buffer((s_q,), "int32"), out: T.Buffer((s_q, 128, 512), "bfloat16"), max_logits: T.Buffer((s_q, 128), "float32"), lse: T.Buffer((s_q, 128), "float32")):
        T.func_attr({"tirx.kernel_launch_params": ["blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_programtic_dependent_launch", "tirx.use_dyn_shared_memory"]})
        kv_tma_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", kv_tma_tensormap, "bfloat16", 2, T.handle_add_byte_offset(kv.data, 0), 512, s_kv, stride_kv_s_kv * 2, 64, 1, 1, 1, 0, 3, 3, 0)
        out_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", out_tensormap, "bfloat16", 4, T.handle_add_byte_offset(out.data, 0), 64, 128, 8, s_q, 1024, 128, 131072, 64, 64, 8, 1, 1, 1, 1, 1, 0, 3, 3, 0)
        out_tensormap_1: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", out_tensormap_1, "bfloat16", 4, T.handle_add_byte_offset(out.data, 0), 64, 128, 8, s_q, 1024, 128, 131072, 64, 64, 8, 1, 1, 1, 1, 1, 0, 3, 3, 0)
        q_tma_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", q_tma_tensormap, "bfloat16", 5, T.handle_add_byte_offset(q.data, 0), 64, 128, 2, 4, s_q, 1024, 512, 128, 131072, 64, 64, 2, 4, 1, 1, 1, 1, 1, 1, 0, 3, 3, 0)
        with T.launch_thread("clusterCtaIdx.x", 2) as clusterCtaIdx_x:
            blockIdx_x = T.launch_thread("blockIdx.x", 2 * s_q)
            threadIdx_x = T.launch_thread("threadIdx.x", 512)
            warp_id_in_cta: T.let[T.int32] = T.tvm_warp_shuffle(T.uint32(4294967295), threadIdx_x // 32, 0, 32, 32)
            block_idx: T.let[T.int32] = blockIdx_x
            v: T.let[T.int32] = clusterCtaIdx_x
            thread_idx: T.let[T.int32] = threadIdx_x
            v_1: T.let[T.int32] = warp_id_in_cta // 4
            v_2: T.let[T.int32] = warp_id_in_cta % 4
            v_3: T.let[T.int32] = threadIdx_x % 32
            v_4: T.let[T.int32] = threadIdx_x % 128
            v_5: T.let[T.int32] = threadIdx_x % 128
            v_6: T.let[T.int32] = threadIdx_x % 128
            v_7: T.let[T.int32] = threadIdx_x % 128
            T.evaluate(v)
            T.evaluate(v_1)
            T.evaluate(v_2)
            T.evaluate(v_3)
            T.evaluate(v_4)
            T.evaluate(v_5)
            T.evaluate(v_6)
            T.evaluate(v_7)
            if warp_id_in_cta == 0:
                if T.cuda.elect_sync() != T.uint32(0):
                    T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(q_tma_tensormap)), "", "", "", "tensormap", "")
            if warp_id_in_cta == 0:
                if T.cuda.elect_sync() != T.uint32(0):
                    T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(out_tensormap_1)), "", "", "", "tensormap", "")
            if warp_id_in_cta == 0:
                if T.cuda.elect_sync() != T.uint32(0):
                    T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(out_tensormap)), "", "", "", "tensormap", "")
            if warp_id_in_cta == 0:
                if T.cuda.elect_sync() != T.uint32(0):
                    T.ptx.prefetch(T.reinterpret(T.handle().ty, T.address_of(kv_tma_tensormap)), "", "", "", "tensormap", "")
            with T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}):
                cta_idx: T.let[T.int32] = block_idx % 2
                warp_idx: T.let[T.int32] = T.cuda.__shfl_sync(T.uint32(4294967295), thread_idx // 32, 0, 32)
                lane_idx: T.let[T.int32] = thread_idx % 32
                warpgroup_idx: T.let[T.int32] = T.cuda.__shfl_sync(T.uint32(4294967295), thread_idx // 128, 0, 32)
                idx_in_warpgroup: T.let[T.int32] = thread_idx % 128
                pool_buf = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
                q_smem = T.decl_buffer((64, 512), "bfloat16", data=pool_buf.data, scope="shared.dyn", align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 8, 64):(64, 4096, 1)])))
                k_smem = T.decl_buffer((256, 256), "bfloat16", data=pool_buf.data, elem_offset=32768, scope="shared.dyn", align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(256, 4, 64):(64, 16384, 1)])))
                s_smem_gemm = T.decl_buffer((64, 64), "bfloat16", data=pool_buf.data, elem_offset=98304, scope="shared.dyn", align=1024, layout=T.TileLayout(T.S[(64, 8, 8):(8, 512, 1)]))
                p_exchange = T.decl_buffer((4, 1024), "uint32", data=pool_buf.data, elem_offset=51200, scope="shared.dyn")
                rowwise_max_buf = T.decl_buffer((128,), data=pool_buf.data, elem_offset=55296, scope="shared.dyn")
                rowwise_li_buf = T.decl_buffer((128,), data=pool_buf.data, elem_offset=55424, scope="shared.dyn")
                is_k_valid = T.decl_buffer((4, 8), "int8", data=pool_buf.data, elem_offset=222208, scope="shared.dyn", align=16)
                buffer = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27780, scope="shared.dyn", align=8)
                buffer_1 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27781, scope="shared.dyn", align=8)
                buffer_2 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27782, scope="shared.dyn", align=8)
                buffer_3 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27783, scope="shared.dyn", align=8)
                bar_tOut_empty_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27784, scope="shared.dyn", align=8)
                buffer_4 = T.decl_buffer((4,), "uint64", data=pool_buf.data, elem_offset=27785, scope="shared.dyn", align=8)
                buffer_5 = T.decl_buffer((4,), "uint64", data=pool_buf.data, elem_offset=27789, scope="shared.dyn", align=8)
                bar_P_empty_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27793, scope="shared.dyn", align=8)
                buffer_6 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27794, scope="shared.dyn", align=8)
                buffer_7 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27795, scope="shared.dyn", align=8)
                bar_S_O_full_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27796, scope="shared.dyn", align=8)
                bar_li_full_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27797, scope="shared.dyn", align=8)
                bar_li_empty_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27798, scope="shared.dyn", align=8)
                bar_valid_coord_scales_full_buf = T.decl_buffer((4,), "uint64", data=pool_buf.data, elem_offset=27799, scope="shared.dyn", align=8)
                bar_valid_coord_scales_empty_buf = T.decl_buffer((4,), "uint64", data=pool_buf.data, elem_offset=27803, scope="shared.dyn", align=8)
                buffer_8 = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27807, scope="shared.dyn", align=8)
                bar_clc_empty_buf = T.decl_buffer((1,), "uint64", data=pool_buf.data, elem_offset=27808, scope="shared.dyn", align=8)
                clc_response = T.decl_buffer((4,), "uint32", data=pool_buf.data, elem_offset=55620, scope="shared.dyn", align=16)
                tmem_start_addr = T.decl_buffer((1,), "uint32", data=pool_buf.data, elem_offset=55624, scope="shared.dyn", align=4)
                with T.attr({"tirx.dyn_smem_bytes": T.int64(222500)}):
                    T.evaluate(0)
                o_tmem = T.decl_buffer((64, 512), scope="tmem", layout=T.TileLayout(T.S[(64, 2, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0)
                buffer_9 = T.decl_buffer((64, 2, 2, 128), scope="tmem", layout=T.TileLayout(T.S[(64, 2, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0)
                buffer_10 = T.decl_buffer((2, 64, 2, 128), scope="tmem", layout=T.TileLayout(T.S[(2, 64, 2, 128):(64 @ Axis.TLane, 1 @ Axis.TLane, 128 @ Axis.TCol, 1 @ Axis.TCol)]), allocated_addr=0)
                o_win = T.decl_buffer((128, 256), scope="tmem", layout=T.TileLayout(T.S[(2, 64, 2, 128):(64 @ Axis.TLane, 1 @ Axis.TLane, 128 @ Axis.TCol, 1 @ Axis.TCol)]), allocated_addr=0)
                q_tmem_fold = T.decl_buffer((2, 64, 256), "bfloat16", scope="tmem", layout=T.TileLayout(T.S[(128, 256):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256)
                tmem_p = T.decl_buffer((64, 128), scope="tmem", layout=T.TileLayout(T.S[(64, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                buffer_11 = k_smem.view(4, 64, 4, 64)
                buffer_12 = T.decl_buffer((4, 64, 4, 64), "bfloat16", data=buffer_11.data, elem_offset=32768, scope="shared.dyn", align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(4, 64, 4, 64):(16384, 64, 4096, 1)])))
                k_smem_gemm = buffer_12.view(4, 64, 256)
                if warp_idx == 1:
                    if T.cuda.elect_sync():
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_1[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_2[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_3[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_tOut_empty_buf[i])), T.uint32(256), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_P_empty_buf[i])), T.uint32(256), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_6[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_7[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_S_O_full_buf[i])), T.uint32(256), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_full_buf[i])), T.uint32(64), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_empty_buf[i])), T.uint32(128), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_8[i])), T.uint32(1), "init", "shared", "b64", "")
                        for i in T.unroll(1):
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_clc_empty_buf[i])), T.uint32(539), "init", "shared", "b64", "")
                        T.ptx.fence("mbarrier_init", "release", "cluster", "")
                else:
                    if warp_idx == 2:
                        T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(tmem_start_addr[0])), T.uint32(512), "alloc", "cta_group::2", "sync", "aligned", "shared::cta", "b32", "")
                        allocated_tmem_addr: T.uint32
                        T.ptx.ld.shared.u32(allocated_tmem_addr, tmem_start_addr.ptr_to([0]))
                        T.cuda.trap_when_assert_failed(allocated_tmem_addr == T.uint32(0))
                        T.ptx.tcgen05("relinquish_alloc_permit", "cta_group::2", "sync", "aligned", "")
                    else:
                        if warp_idx == 3:
                            if T.cuda.elect_sync():
                                for init_stage in T.unroll(4):
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_4[init_stage])), T.uint32(1), "init", "shared", "b64", "")
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_5[init_stage])), T.uint32(1), "init", "shared", "b64", "")
                                for init_stage in T.unroll(4):
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_full_buf[init_stage])), T.uint32(8), "init", "shared", "b64", "")
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_empty_buf[init_stage])), T.uint32(128), "init", "shared", "b64", "")
                                T.ptx.fence("mbarrier_init", "release", "cluster", "")
                T.cuda.cluster_sync()
                if warpgroup_idx == 0:
                    q_o_token: T.uint32 = T.cuda.iket.range_start("h128-small-q-load-output")
                    T.ptx.setmaxnreg(160, "inc", "sync", "aligned", "u32", "")
                    wg0_job_valid: T.int32 = 1
                    wg0_job_block_idx: T.int32 = block_idx
                    wg0_outer_loop_phase: T.int32 = 0
                    last_valid: T.int32 = 0
                    last_s_q_idx: T.int32 = 0
                    last_outer_loop_phase: T.int32 = 0
                    while wg0_job_valid != 0:
                        wg0_s_q_idx: T.let[T.int32] = wg0_job_block_idx // 2
                        if warp_idx == 0:
                            if T.cuda.elect_sync():
                                T.ptx.cp(0, "async", "bulk", "wait_group", "", "")
                                buffer_13 = q.view(s_q, 128, 2, 4, 64)
                                buffer_14 = buffer_13.view(64, 128, 2, 4, s_q, layout=T.TileLayout(T.S[(64, 128, 2, 4, s_q):(1, 512, 256, 64, 65536)]))
                                q_tma = T.decl_buffer((64, 128, 2, 4, s_q), "bfloat16", data=buffer_14.data, layout=T.TileLayout(T.S[(64, 128, 2, 4, s_q):(1, 512, 256, 64, 65536)]))
                                buffer_15 = q_smem.view(64, 4, 2, 64)
                                buffer_16 = buffer_15.view(64, 64, 2, 4, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 64, 2, 4):(1, 64, 4096, 8192)])))
                                q_smem_tma = T.decl_buffer((64, 64, 2, 4), "bfloat16", data=buffer_16.data, scope="shared.dyn", align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 64, 2, 4):(1, 64, 4096, 8192)])))
                                buffer_17: T.uint64
                                T.ptx.mapa(buffer_17, T.address_of(buffer[0]), T.uint32(0), "", "u64", "")
                                buffer_18 = T.decl_scalar(T.bfloat16, data=q_smem_tma.data, elem_offset=0, scope="shared.dyn")
                                T.ptx.cp(T.cuda.cvta_generic_to_shared(T.address_of(buffer_18)), T.reinterpret(T.handle().ty, T.address_of(q_tma_tensormap)), 0, block_idx % 2 * 64, 0, 0, wg0_job_block_idx // 2, T.cuda.cvta_generic_to_shared(T.reinterpret(T.handle().ty, buffer_17)), T.uint64(1364590687093260288), "async", "bulk", "tensor", "5d", "shared::cluster", "global", "", "mbarrier::complete_tx::bytes", "", "cta_group::2", "L2::cache_hint", "")
                                if cta_idx == 0:
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer[0])), T.uint32(131072), "arrive", "expect_tx", "", "", "shared", "b64", "")
                                    T.cuda.mbarrier_wait(T.address_of(buffer[0]), T.bitwise_xor(wg0_outer_loop_phase, 0))
                                    T.cuda.mbarrier_wait(T.address_of(buffer_1[0]), T.bitwise_xor(T.bitwise_xor(wg0_outer_loop_phase, 1), 0))
                                    T.ptx.tcgen05("fence::after_thread_sync", "")
                                    buffer_19 = T.decl_buffer((2, 64, 4, 64), "bfloat16", scope="tmem", layout=T.TileLayout(T.S[(128, 256):(1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256)
                                    buffer_20 = T.decl_buffer((64, 4, 2, 64), "bfloat16", scope="tmem", layout=T.TileLayout(T.S[(64, 4, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256)
                                    q_tmem_cp = T.decl_buffer((64, 4, 2, 64), "bfloat16", scope="tmem", layout=T.TileLayout(T.S[(64, 4, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=256)
                                    buffer_21 = q_smem.view(64, 4, 2, 64)
                                    cp_desc: T.uint64
                                    T.cuda.tcgen05.encode_matrix_descriptor(cp_desc.buffer.data, T.reinterpret(T.handle().ty, T.uint64(0)), 1, 64, 3)
                                    for flat in T.unroll(16):
                                        T.ptx.tcgen05(T.Cast("uint32", 256 + (flat % 4 * 32 + flat // 4 % 4 * 8)), T.bitwise_or(T.bitwise_and(cp_desc, T.bitwise_not(T.uint64(16383))), T.Cast("uint64", T.bitwise_and(T.shift_right(T.cuda.cvta_generic_to_shared(T.ptr_byte_offset(T.address_of(buffer_21[0, 0, 0, 0]), (flat % 4 * 1024 + flat // 4 % 4 * 2) * 16, T.type_annotation("bfloat16"))), T.uint32(4)), T.uint32(16383)))), "cp", "cta_group::2", "128x256b", "", "", "", "")
                                    T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_2[0])), T.Cast("uint16", 3), "commit", "cta_group::2", "mbarrier::arrive::one", "shared::cluster", "multicast::cluster", "b64", "")
                        if last_valid != 0:
                            T.cuda.mbarrier_wait(T.address_of(bar_li_full_buf[0]), T.bitwise_xor(last_outer_loop_phase, 0))
                            output_scale: T.float32
                            T.ptx.ld.shared.f32(output_scale, rowwise_li_buf.ptr_to([idx_in_warpgroup % 64]))
                            T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_empty_buf[0])), T.uint32(1), "arrive", "", "", "shared", "b64", "")
                            T.cuda.mbarrier_wait(T.address_of(buffer_3[0]), T.bitwise_xor(last_outer_loop_phase, 0))
                            buffer_13 = T.alloc_local((64,))
                            o_epi_frag = buffer_13.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)]))
                            o_epi = o_epi_frag.local()
                            buffer_14 = T.alloc_local((64,), "bfloat16")
                            o_epi_bf16_frag = buffer_14.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)]))
                            buffer_15 = q_smem.view(64, 2, 256)
                            buffer_16 = buffer_15.view(2, 64, 256, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 64, 4, 64):(16384, 64, 4096, 1)])))
                            q_smem_win = buffer_16.view(128, 256)
                            for epi_k in T.unroll(4):
                                local_storage = o_epi_frag.local()
                                T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], local_storage[32], local_storage[33], local_storage[34], local_storage[35], local_storage[36], local_storage[37], local_storage[38], local_storage[39], local_storage[40], local_storage[41], local_storage[42], local_storage[43], local_storage[44], local_storage[45], local_storage[46], local_storage[47], local_storage[48], local_storage[49], local_storage[50], local_storage[51], local_storage[52], local_storage[53], local_storage[54], local_storage[55], local_storage[56], local_storage[57], local_storage[58], local_storage[59], local_storage[60], local_storage[61], local_storage[62], local_storage[63], T.cuda.get_tmem_addr(T.uint32(0), 0, epi_k * 64), "ld", "sync", "aligned", "32x32b", "x64", "", "b32", "")
                                T.ptx.tcgen05("wait::ld", "sync", "aligned", "")
                                if epi_k == 0:
                                    T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(T.bitwise_xor(last_outer_loop_phase, 1), 0))
                                    T.ptx.fence("proxy", "async", "shared::cta", "")
                                if epi_k == 3:
                                    buffer_17: T.uint32
                                    T.ptx.mapa(buffer_17, T.cuda.cvta_generic_to_shared(T.address_of(bar_tOut_empty_buf[0])), T.uint32(0), "shared::cluster", "u32", "")
                                    T.ptx.mbarrier(buffer_17, "arrive", "", "", "shared::cluster", "b64", "")
                                buffer_17 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                                buffer_18 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                                for f in range(32):
                                    dst_lane_indices_0_0: T.int32 = f * 2
                                    dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                    buffer_19: T.uint64
                                    buffer_20: T.uint64
                                    T.ptx.mov(buffer_19, buffer_18[f * 2], buffer_18[f * 2 + 1], "b64", "")
                                    T.ptx.mov(buffer_20, output_scale, output_scale, "b64", "")
                                    T.ptx.mul(buffer_19, buffer_19, buffer_20, "rz", "ftz", "", "f32x2", "")
                                    T.ptx.mov(buffer_17[f * 2], buffer_17[f * 2 + 1], buffer_19, "b64", "")
                                buffer_19 = o_epi_bf16_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                                buffer_19_words = buffer_19.view("uint32")
                                buffer_20 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                                for f in range(32):
                                    dst_lane_indices_0_0: T.int32 = f * 2
                                    dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                    T.ptx.cvt.rn.bf16x2.f32(
                                        buffer_19_words[f], buffer_20[f * 2 + 1], buffer_20[f * 2]
                                    )
                                r_local = o_epi_bf16_frag.local()
                                r_words = r_local.view("uint32")
                                for f in range(8):
                                    ds: T.int32 = f % 8 * 8
                                    dr: T.int32 = f % 8 * 8
                                    s_off: T.int32 = v_5 // 64 * 16384 + epi_k % 4 * 4096 + v_5 % 64 * 64 + T.bitwise_xor(f * 8, T.shift_left(T.bitwise_and(v_5 // 64 * 256 + epi_k % 4 * 64 + v_5 % 64, 7), 3))
                                    s_ptr: T.let = T.ptr_byte_offset(
                                        T.address_of(q_smem_win[0, 0]), s_off * BF16_BYTES, "bfloat16"
                                    )
                                    r_w: T.int32 = dr // 2
                                    T.ptx.st(T.cuda.cvta_generic_to_shared(s_ptr), r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], "", "", "shared", "", "", "", "v4", "u32", "")
                            T.ptx.fence("proxy", "async", "shared::cta", "")
                            T.ptx.bar(T.uint32(0), T.uint32(128), "", "sync", "")
                            if warp_idx == 0:
                                if T.cuda.elect_sync():
                                    buffer_17 = T.decl_scalar(T.bfloat16, data=q_smem.data, elem_offset=0, scope="shared.dyn")
                                    T.ptx.cp(T.reinterpret(T.handle().ty, T.address_of(out_tensormap_1)), 0, block_idx % 2 * 64, 0, last_s_q_idx, T.cuda.cvta_generic_to_shared(T.address_of(buffer_17)), "async", "bulk", "tensor", "4d", "global", "shared::cta", "tile", "bulk_group", "", "")
                                    T.ptx.cp("async", "bulk", "commit_group", "")
                        else:
                            T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(wg0_outer_loop_phase, 0))
                        last_valid = 1
                        last_s_q_idx = wg0_s_q_idx
                        last_outer_loop_phase = wg0_outer_loop_phase
                        T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(wg0_outer_loop_phase, 0))
                        wg0_next_job: T.uint32
                        query_cancel_first_ctaid_x(wg0_next_job, T.address_of(clc_response[0]))
                        _rem1: T.uint64
                        T.ptx.mapa(_rem1, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), "shared::cluster", "u64", "")
                        T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem1), T.uint32(1), T.bool(True), "arrive", "", "", "", "b64", "pred")
                        if wg0_next_job == T.uint32(4294967295):
                            wg0_job_valid = 0
                        else:
                            wg0_job_block_idx = T.Cast("int32", wg0_next_job)
                        wg0_outer_loop_phase = T.bitwise_xor(wg0_outer_loop_phase, 1)
                    if last_valid != 0:
                        if warp_idx == 0:
                            if T.cuda.elect_sync():
                                T.ptx.cp(0, "async", "bulk", "wait_group", "", "")
                        T.ptx.bar(T.uint32(0), T.uint32(128), "", "sync", "")
                        T.cuda.mbarrier_wait(T.address_of(bar_li_full_buf[0]), T.bitwise_xor(last_outer_loop_phase, 0))
                        output_scale: T.float32
                        T.ptx.ld.shared.f32(output_scale, rowwise_li_buf.ptr_to([idx_in_warpgroup % 64]))
                        T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_empty_buf[0])), T.uint32(1), "arrive", "", "", "shared", "b64", "")
                        T.cuda.mbarrier_wait(T.address_of(buffer_3[0]), T.bitwise_xor(last_outer_loop_phase, 0))
                        if T.cuda.elect_sync():
                            T.ptx.griddepcontrol("launch_dependents", "")
                        buffer_13 = T.alloc_local((64,))
                        o_epi_frag = buffer_13.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)]))
                        o_epi = o_epi_frag.local()
                        buffer_14 = T.alloc_local((64,), "bfloat16")
                        o_epi_bf16_frag = buffer_14.view(128, 64, layout=T.TileLayout(T.S[(128, 64):(1 @ Axis.tid_in_wg, 1)]))
                        buffer_15 = q_smem.view(64, 2, 256)
                        buffer_16 = buffer_15.view(2, 64, 256, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 64, 4, 64):(16384, 64, 4096, 1)])))
                        q_smem_win = buffer_16.view(128, 256)
                        for epi_k in T.unroll(4):
                            local_storage = o_epi_frag.local()
                            T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], local_storage[32], local_storage[33], local_storage[34], local_storage[35], local_storage[36], local_storage[37], local_storage[38], local_storage[39], local_storage[40], local_storage[41], local_storage[42], local_storage[43], local_storage[44], local_storage[45], local_storage[46], local_storage[47], local_storage[48], local_storage[49], local_storage[50], local_storage[51], local_storage[52], local_storage[53], local_storage[54], local_storage[55], local_storage[56], local_storage[57], local_storage[58], local_storage[59], local_storage[60], local_storage[61], local_storage[62], local_storage[63], T.cuda.get_tmem_addr(T.uint32(0), 0, epi_k * 64), "ld", "sync", "aligned", "32x32b", "x64", "", "b32", "")
                            T.ptx.tcgen05("wait::ld", "sync", "aligned", "")
                            if epi_k == 0:
                                T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(last_outer_loop_phase, 0))
                                T.ptx.fence("proxy", "async", "shared::cta", "")
                            if epi_k == 3:
                                buffer_17: T.uint32
                                T.ptx.mapa(buffer_17, T.cuda.cvta_generic_to_shared(T.address_of(bar_tOut_empty_buf[0])), T.uint32(0), "shared::cluster", "u32", "")
                                T.ptx.mbarrier(buffer_17, "arrive", "", "", "shared::cluster", "b64", "")
                            buffer_17 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                            buffer_18 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                            for f in range(32):
                                dst_lane_indices_0_0: T.int32 = f * 2
                                dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                buffer_19: T.uint64
                                buffer_20: T.uint64
                                T.ptx.mov(buffer_19, buffer_18[f * 2], buffer_18[f * 2 + 1], "b64", "")
                                T.ptx.mov(buffer_20, output_scale, output_scale, "b64", "")
                                T.ptx.mul(buffer_19, buffer_19, buffer_20, "rz", "ftz", "", "f32x2", "")
                                T.ptx.mov(buffer_17[f * 2], buffer_17[f * 2 + 1], buffer_19, "b64", "")
                            buffer_19 = o_epi_bf16_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                            buffer_19_words = buffer_19.view("uint32")
                            buffer_20 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                            for f in range(32):
                                dst_lane_indices_0_0: T.int32 = f * 2
                                dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                T.ptx.cvt.rn.bf16x2.f32(
                                    buffer_19_words[f], buffer_20[f * 2 + 1], buffer_20[f * 2]
                                )
                            r_local = o_epi_bf16_frag.local()
                            r_words = r_local.view("uint32")
                            for f in range(8):
                                ds: T.int32 = f % 8 * 8
                                dr: T.int32 = f % 8 * 8
                                s_off: T.int32 = v_6 // 64 * 16384 + epi_k % 4 * 4096 + v_6 % 64 * 64 + T.bitwise_xor(f * 8, T.shift_left(T.bitwise_and(v_6 // 64 * 256 + epi_k % 4 * 64 + v_6 % 64, 7), 3))
                                s_ptr: T.let = T.ptr_byte_offset(
                                    T.address_of(q_smem_win[0, 0]), s_off * BF16_BYTES, "bfloat16"
                                )
                                r_w: T.int32 = dr // 2
                                T.ptx.st(T.cuda.cvta_generic_to_shared(s_ptr), r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], "", "", "shared", "", "", "", "v4", "u32", "")
                        T.ptx.fence("proxy", "async", "shared::cta", "")
                        T.ptx.bar(T.uint32(0), T.uint32(128), "", "sync", "")
                        if warp_idx == 0:
                            if T.cuda.elect_sync():
                                buffer_17 = T.decl_scalar(T.bfloat16, data=q_smem.data, elem_offset=0, scope="shared.dyn")
                                T.ptx.cp(T.reinterpret(T.handle().ty, T.address_of(out_tensormap)), 0, block_idx % 2 * 64, 0, last_s_q_idx, T.cuda.cvta_generic_to_shared(T.address_of(buffer_17)), "async", "bulk", "tensor", "4d", "global", "shared::cta", "tile", "bulk_group", "", "")
                                T.ptx.cp("async", "bulk", "commit_group", "")
                    if warp_idx == 0:
                        T.ptx.tcgen05(T.uint32(0), T.uint32(512), "dealloc", "cta_group::2", "sync", "aligned", "b32", "")
                    T.cuda.iket.range_end(q_o_token)
                else:
                    if warpgroup_idx == 1:
                        kv_gather_token: T.uint32 = T.cuda.iket.range_start("h128-small-kv-load")
                        T.ptx.setmaxnreg(80, "dec", "sync", "aligned", "u32", "")
                        wg1_warp_idx: T.let[T.int32] = thread_idx // 32 - 4
                        if T.cuda.elect_sync():
                            wg1_job_valid: T.int32 = 1
                            wg1_job_block_idx: T.int32 = block_idx
                            wg1_outer_loop_phase: T.int32 = 0
                            wg1_rs: T.int32 = 0
                            while wg1_job_valid != 0:
                                wg1_s_q_idx: T.let[T.int32] = wg1_job_block_idx // 2
                                wg1_topk_len: T.int32 = topk
                                if have_topk_length:
                                    T.ptx.ld.global_.s32(wg1_topk_len, topk_length.ptr_to([wg1_s_q_idx]))
                                wg1_num_k_blocks: T.let[T.int32] = T.max((wg1_topk_len + 64 - 1) // 64, 1)
                                wg1_g_indices_base: T.let[T.int32] = wg1_s_q_idx * stride_indices_s_q
                                for k in T.serial(wg1_num_k_blocks, unroll=False):
                                    k_buf_idx: T.let[T.int32] = wg1_rs % 4
                                    k_bar_phase: T.let[T.int32] = T.bitwise_and(wg1_rs // 4, 1)
                                    cur_indices = T.alloc_local((16,), "int32")
                                    for local_row in T.unroll(2):
                                        row: T.let[T.int32] = local_row * 32 + wg1_warp_idx * 8
                                        row_base: T.let[T.int32] = wg1_g_indices_base + k * 64 + row
                                        buffer_13 = T.decl_buffer((16,), "int32", data=cur_indices.data, scope="local")
                                        buffer_14 = buffer_13.view("uint32")
                                        T.ptx.ld(buffer_14[local_row * 8], buffer_14[local_row * 8 + 1], buffer_14[local_row * 8 + 2], buffer_14[local_row * 8 + 3], buffer_14[local_row * 8 + 4], buffer_14[local_row * 8 + 5], buffer_14[local_row * 8 + 6], buffer_14[local_row * 8 + 7], T.address_of(indices[row_base]), "", "", "global", "", "nc", "L1::no_allocate", "L2::evict_first", "L2::256B", "v8", "u32", "")
                                    T.cuda.mbarrier_wait(T.address_of(buffer_5[k_buf_idx]), T.bitwise_xor(T.bitwise_xor(k_bar_phase, 1), 0))
                                    k_smem_gemm_cur = T.decl_buffer((64, 256), "bfloat16", data=k_smem_gemm.data, elem_offset=32768 + k_buf_idx * 16384, scope="shared.dyn", align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 4, 64):(64, 4096, 1)])))
                                    src_col: T.let[T.int32] = cta_idx * 256
                                    buffer_13 = k_smem_gemm_cur.view(64, 4, 64)
                                    buffer_14 = buffer_13.view(2, 4, 2, 4, 4, 64)
                                    buffer_15 = T.decl_buffer((2, 2, 4, 4, 64), "bfloat16", data=buffer_14.data, elem_offset=32768 + k_buf_idx * 16384 + wg1_warp_idx * 512, scope="shared.dyn", align=1024, layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 2, 4, 4, 64):(2048, 256, 64, 4096, 1)])))
                                    k_gather_tile = buffer_15.view(16, 4, 64)
                                    kv_tma = kv.view(s_kv, 512, layout=T.TileLayout(T.S[(s_kv, 512):(stride_kv_s_kv, 1)]))
                                    k_gather_tile_2d = k_gather_tile.view(16, 256)
                                    for row_group in T.unroll(4):
                                        for col_atom in T.unroll(4):
                                            buffer_16: T.uint64
                                            T.ptx.mapa(buffer_16, T.address_of(buffer_4[k_buf_idx]), T.uint32(0), "", "u64", "")
                                            kv_dst_offset: T.let[T.int32] = (k_buf_idx * 16384 + wg1_warp_idx * 512 + row_group // 2 * 2048 + row_group % 2 * 256 + col_atom * 4096) * BF16_BYTES
                                            T.ptx.cp(T.cuda.cvta_generic_to_shared(T.ptr_byte_offset(T.address_of(k_smem[0, 0]), kv_dst_offset, T.type_annotation("bfloat16"))), T.reinterpret(T.handle().ty, T.address_of(kv_tma_tensormap)), src_col + col_atom * 64, cur_indices[row_group * 4], cur_indices[row_group * 4 + 1], cur_indices[row_group * 4 + 2], cur_indices[row_group * 4 + 3], T.cuda.cvta_generic_to_shared(T.reinterpret(T.handle().ty, buffer_16)), T.uint64(1508705875169116160), "async", "bulk", "tensor", "2d", "shared::cluster", "global", "tile::gather4", "mbarrier::complete_tx::bytes", "", "cta_group::2", "L2::cache_hint", "")
                                    wg1_rs = wg1_rs + 1
                                T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(wg1_outer_loop_phase, 0))
                                wg1_next_job: T.uint32
                                query_cancel_first_ctaid_x(wg1_next_job, T.address_of(clc_response[0]))
                                _rem2: T.uint64
                                T.ptx.mapa(_rem2, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), "shared::cluster", "u64", "")
                                T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem2), T.uint32(1), T.bool(True), "arrive", "", "", "", "b64", "pred")
                                if wg1_next_job == T.uint32(4294967295):
                                    wg1_job_valid = 0
                                else:
                                    wg1_job_block_idx = T.Cast("int32", wg1_next_job)
                                wg1_outer_loop_phase = T.bitwise_xor(wg1_outer_loop_phase, 1)
                        T.cuda.iket.range_end(kv_gather_token)
                    else:
                        if warpgroup_idx == 2:
                            T.ptx.setmaxnreg(80, "dec", "sync", "aligned", "u32", "")
                            if T.bitwise_and(warp_idx == 8, cta_idx == 0):
                                mma_token: T.uint32 = T.cuda.iket.range_start("h128-small-qk-pv-issue")
                                if T.cuda.elect_sync():
                                    umma_job_valid: T.int32 = 1
                                    umma_job_block_idx: T.int32 = block_idx
                                    umma_outer_loop_phase: T.int32 = 0
                                    umma_rs: T.int32 = 0
                                    while umma_job_valid != 0:
                                        umma_s_q_idx: T.let[T.int32] = umma_job_block_idx // 2
                                        umma_topk_len: T.int32 = topk
                                        if have_topk_length:
                                            T.ptx.ld.global_.s32(umma_topk_len, topk_length.ptr_to([umma_s_q_idx]))
                                        umma_num_k_blocks: T.let[T.int32] = T.max((umma_topk_len + 64 - 1) // 64, 1)
                                        T.cuda.mbarrier_wait(T.address_of(buffer_2[0]), T.bitwise_xor(umma_outer_loop_phase, 0))
                                        for k in T.serial(umma_num_k_blocks + 1, unroll=False):
                                            if k < umma_num_k_blocks:
                                                k_buf_idx: T.let[T.int32] = umma_rs % 4
                                                k_bar_phase: T.let[T.int32] = T.bitwise_and(umma_rs // 4, 1)
                                                p_bar_phase: T.let[T.int32] = T.bitwise_and(umma_rs, 1)
                                                T.cuda.mbarrier_wait(T.address_of(bar_P_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(p_bar_phase, 1), 0))
                                                T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_4[k_buf_idx])), T.uint32(65536), "arrive", "expect_tx", "", "", "shared", "b64", "")
                                                T.cuda.mbarrier_wait(T.address_of(buffer_4[k_buf_idx]), T.bitwise_xor(k_bar_phase, 0))
                                                T.ptx.tcgen05("fence::after_thread_sync", "")
                                                qk_accumulate: T.uint32 = T.uint32(0)
                                                descB_local: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3)
                                                for mi in T.unroll(1):
                                                    for ni in T.unroll(1):
                                                        for ki in T.unroll(16):
                                                            T.ptx.tcgen05(T.Cast("uint32", ni * 64 + 384), T.Cast("uint32", ki * 8 + 256), _add_smem_desc_offset(descB_local, (ki // 1024 * 16384 + ni * 16384 + k_buf_idx * 16384 + ki % 16 // 4 * 4096 + ki % 1024 // 16 * 64 + ki % 4 * 16) // 8), T.uint32(136316048), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), ki != 0 or T.Cast("bool", qk_accumulate), "mma", "cta_group::2", "kind::f16", "p12")
                                                qk_accumulate = T.uint32(1)
                                                T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_6[0])), T.Cast("uint16", 3), "commit", "cta_group::2", "mbarrier::arrive::one", "shared::cluster", "multicast::cluster", "b64", "")
                                                if k == umma_num_k_blocks - 1:
                                                    T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_1[0])), "commit", "cta_group::2", "mbarrier::arrive::one", "shared::cluster", "b64", "")
                                            if k > 0:
                                                prev_k: T.let[T.int32] = k - 1
                                                prev_rs: T.let[T.int32] = umma_rs - 1
                                                prev_buf: T.let[T.int32] = prev_rs % 4
                                                prev_s_o_phase: T.let[T.int32] = T.bitwise_and(prev_rs, 1)
                                                T.cuda.mbarrier_wait(T.address_of(bar_S_O_full_buf[0]), T.bitwise_xor(prev_s_o_phase, 0))
                                                if prev_k == 0:
                                                    T.cuda.mbarrier_wait(T.address_of(bar_tOut_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(umma_outer_loop_phase, 1), 0))
                                                T.ptx.tcgen05("fence::after_thread_sync", "")
                                                o_accumulate: T.uint32 = T.if_then_else(prev_k == 0, T.uint32(0), T.uint32(1))
                                                buffer_13 = T.decl_buffer((64, 256), scope="tmem", layout=T.TileLayout(T.S[(64, 1, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=0)
                                                descB_local: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3)
                                                descA_local: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA_local), T.address_of(s_smem_gemm[0, 0]), 64, 8, 0)
                                                for mi in T.unroll(1):
                                                    for ni in T.unroll(1):
                                                        for ki in T.unroll(4):
                                                            T.ptx.tcgen05(T.Cast("uint32", ni * 128), _add_smem_desc_offset(descA_local, (ki % 4 * 1024 + mi * 512 + ki // 4 * 8) // 8), _add_smem_desc_offset(descB_local, ((ki * 16 + ni) // 64 * 16384 + prev_buf * 16384 + (ki * 16 + ni) % 64 * 64) // 8), T.uint32(138478736), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), ki != 0 or T.Cast("bool", o_accumulate), "mma", "cta_group::2", "kind::f16", "p12")
                                                buffer_14 = T.decl_buffer((64, 256), scope="tmem", layout=T.TileLayout(T.S[(64, 1, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=128)
                                                descB_local_1: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local_1), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3)
                                                descA_local_1: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA_local_1), T.address_of(s_smem_gemm[0, 0]), 64, 8, 0)
                                                for mi in T.unroll(1):
                                                    for ni in T.unroll(1):
                                                        for ki in T.unroll(4):
                                                            T.ptx.tcgen05(T.Cast("uint32", ni * 128 + 128), _add_smem_desc_offset(descA_local_1, (ki % 4 * 1024 + mi * 512 + ki // 4 * 8) // 8), _add_smem_desc_offset(descB_local_1, ((ki * 16 + ni) // 64 * 16384 + prev_buf * 16384 + (ki * 16 + ni) % 64 * 64 + 8192) // 8), T.uint32(138478736), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), ki != 0 or T.Cast("bool", o_accumulate), "mma", "cta_group::2", "kind::f16", "p12")
                                                o_accumulate = T.uint32(1)
                                                T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_7[0])), T.Cast("uint16", 3), "commit", "cta_group::2", "mbarrier::arrive::one", "shared::cluster", "multicast::cluster", "b64", "")
                                                T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_5[prev_buf])), T.Cast("uint16", 3), "commit", "cta_group::2", "mbarrier::arrive::one", "shared::cluster", "multicast::cluster", "b64", "")
                                            if k != umma_num_k_blocks:
                                                umma_rs = umma_rs + 1
                                        T.ptx.tcgen05("fence::before_thread_sync", "")
                                        T.ptx.tcgen05(T.cuda.cvta_generic_to_shared(T.address_of(buffer_3[0])), T.Cast("uint16", 3), "commit", "cta_group::2", "mbarrier::arrive::one", "shared::cluster", "multicast::cluster", "b64", "")
                                        T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(umma_outer_loop_phase, 0))
                                        umma_next_job: T.uint32
                                        query_cancel_first_ctaid_x(umma_next_job, T.address_of(clc_response[0]))
                                        _rem3: T.uint64
                                        T.ptx.mapa(_rem3, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), "shared::cluster", "u64", "")
                                        T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem3), T.uint32(1), T.bool(True), "arrive", "", "", "", "b64", "pred")
                                        if umma_next_job == T.uint32(4294967295):
                                            umma_job_valid = 0
                                        else:
                                            umma_job_block_idx = T.Cast("int32", umma_next_job)
                                        umma_outer_loop_phase = T.bitwise_xor(umma_outer_loop_phase, 1)
                                T.cuda.iket.range_end(mma_token)
                            else:
                                if warp_idx == 9:
                                    valid_mask_token: T.uint32 = T.cuda.iket.range_start("h128-small-valid-mask")
                                    if lane_idx < 8:
                                        lane_indices = T.alloc_local((8,), "int32")
                                        valid_job_valid: T.int32 = 1
                                        valid_job_block_idx: T.int32 = block_idx
                                        valid_outer_loop_phase: T.int32 = 0
                                        valid_rs: T.int32 = 0
                                        while valid_job_valid != 0:
                                            valid_s_q_idx: T.let[T.int32] = valid_job_block_idx // 2
                                            valid_topk_len: T.int32 = topk
                                            if have_topk_length:
                                                T.ptx.ld.global_.s32(valid_topk_len, topk_length.ptr_to([valid_s_q_idx]))
                                            valid_num_k_blocks: T.let[T.int32] = T.max((valid_topk_len + 64 - 1) // 64, 1)
                                            valid_g_indices_base: T.let[T.int32] = valid_s_q_idx * stride_indices_s_q
                                            for k in T.serial(valid_num_k_blocks, unroll=False):
                                                row_base: T.let[T.int32] = valid_g_indices_base + k * 64 + lane_idx * 8
                                                buffer_13 = T.decl_buffer((8,), "int32", data=lane_indices.data, scope="local")
                                                buffer_14 = buffer_13.view("uint32")
                                                T.ptx.ld(buffer_14[0], buffer_14[1], buffer_14[2], buffer_14[3], buffer_14[4], buffer_14[5], buffer_14[6], buffer_14[7], T.address_of(indices[row_base]), "", "", "global", "", "nc", "L1::no_allocate", "L2::evict_normal", "L2::256B", "v8", "u32", "")
                                                abs_pos_start: T.let[T.int32] = k * 64
                                                mask: T.let[T.int8] = T.Cast("int8", T.bitwise_or(T.bitwise_or(T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[0] >= 0, lane_indices[0] < s_kv), abs_pos_start + lane_idx * 8 < valid_topk_len), 1, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[1] >= 0, lane_indices[1] < s_kv), abs_pos_start + lane_idx * 8 + 1 < valid_topk_len), 2, 0)), T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[2] >= 0, lane_indices[2] < s_kv), abs_pos_start + lane_idx * 8 + 2 < valid_topk_len), 4, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[3] >= 0, lane_indices[3] < s_kv), abs_pos_start + lane_idx * 8 + 3 < valid_topk_len), 8, 0))), T.bitwise_or(T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[4] >= 0, lane_indices[4] < s_kv), abs_pos_start + lane_idx * 8 + 4 < valid_topk_len), 16, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[5] >= 0, lane_indices[5] < s_kv), abs_pos_start + lane_idx * 8 + 5 < valid_topk_len), 32, 0)), T.bitwise_or(T.Select(T.bitwise_and(T.bitwise_and(lane_indices[6] >= 0, lane_indices[6] < s_kv), abs_pos_start + lane_idx * 8 + 6 < valid_topk_len), 64, 0), T.Select(T.bitwise_and(T.bitwise_and(lane_indices[7] >= 0, lane_indices[7] < s_kv), abs_pos_start + lane_idx * 8 + 7 < valid_topk_len), 128, 0)))))
                                                index_buf_idx: T.let[T.int32] = valid_rs % 4
                                                index_bar_phase: T.let[T.int32] = T.bitwise_and(valid_rs // 4, 1)
                                                T.cuda.mbarrier_wait(T.address_of(bar_valid_coord_scales_empty_buf[index_buf_idx]), T.bitwise_xor(T.bitwise_xor(index_bar_phase, 1), 0))
                                                T.ptx.st.shared.b8(is_k_valid.ptr_to([index_buf_idx, lane_idx]), T.reinterpret("uint8", mask))
                                                T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_full_buf[index_buf_idx])), T.uint32(1), "arrive", "", "", "shared", "b64", "")
                                                valid_rs = valid_rs + 1
                                            T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(valid_outer_loop_phase, 0))
                                            valid_next_job: T.uint32
                                            query_cancel_first_ctaid_x(valid_next_job, T.address_of(clc_response[0]))
                                            _rem4: T.uint64
                                            T.ptx.mapa(_rem4, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), "shared::cluster", "u64", "")
                                            T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem4), T.uint32(1), T.bool(True), "arrive", "", "", "", "b64", "pred")
                                            if valid_next_job == T.uint32(4294967295):
                                                valid_job_valid = 0
                                            else:
                                                valid_job_block_idx = T.Cast("int32", valid_next_job)
                                            valid_outer_loop_phase = T.bitwise_xor(valid_outer_loop_phase, 1)
                                    T.cuda.iket.range_end(valid_mask_token)
                                else:
                                    if warp_idx >= 10:
                                        clc_token: T.uint32 = T.cuda.iket.sentinel_token("h128-small-clc")
                                        if warp_idx == 10:
                                            clc_token = T.cuda.iket.range_start("h128-small-clc")
                                        if T.cuda.elect_sync():
                                            if warp_idx == 10:
                                                clc_job_valid: T.int32 = 1
                                                clc_outer_loop_phase: T.int32 = 0
                                                while clc_job_valid != 0:
                                                    if cta_idx == 0:
                                                        T.cuda.mbarrier_wait(T.address_of(bar_clc_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(clc_outer_loop_phase, 1), 0))
                                                        T.ptx.clusterlaunchcontrol(T.cuda.cvta_generic_to_shared(T.address_of(clc_response[0])), T.cuda.cvta_generic_to_shared(T.address_of(buffer_8[0])), "try_cancel", "async", "shared::cta", "mbarrier::complete_tx::bytes", "multicast::cluster::all", "b128", "")
                                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(buffer_8[0])), T.uint32(16), "arrive", "expect_tx", "", "", "shared", "b64", "")
                                                    T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(clc_outer_loop_phase, 0))
                                                    clc_next_job: T.uint32
                                                    query_cancel_first_ctaid_x(clc_next_job, T.address_of(clc_response[0]))
                                                    _rem5: T.uint64
                                                    T.ptx.mapa(_rem5, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), "shared::cluster", "u64", "")
                                                    T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem5), T.uint32(1), T.bool(True), "arrive", "", "", "", "b64", "pred")
                                                    if clc_next_job == T.uint32(4294967295):
                                                        clc_job_valid = 0
                                                    clc_outer_loop_phase = T.bitwise_xor(clc_outer_loop_phase, 1)
                                        T.cuda.iket.range_end(clc_token)
                        else:
                            softmax_token: T.uint32 = T.cuda.iket.range_start("h128-small-softmax")
                            T.ptx.setmaxnreg(160, "inc", "sync", "aligned", "u32", "")
                            local_warp_idx: T.let[T.int32] = warp_idx - 12
                            wg3_job_valid: T.int32 = 1
                            wg3_job_block_idx: T.int32 = block_idx
                            wg3_outer_loop_phase: T.int32 = 0
                            wg3_rs: T.int32 = 0
                            while wg3_job_valid != 0:
                                wg3_s_q_idx: T.let[T.int32] = wg3_job_block_idx // 2
                                wg3_topk_len: T.int32 = topk
                                if have_topk_length:
                                    T.ptx.ld.global_.s32(wg3_topk_len, topk_length.ptr_to([wg3_s_q_idx]))
                                wg3_num_k_blocks: T.let[T.int32] = T.max((wg3_topk_len + 64 - 1) // 64, 1)
                                mi: T.float32 = T.float32(-1000000000000000019884624838656.0)
                                li: T.float32 = T.float32(0.0)
                                real_mi: T.float32 = T.float32("-inf")
                                scale_pair: T.let[T.uint64] = T.cuda.make_float2(T.float32(sm_scale_div_log2), T.float32(sm_scale_div_log2))
                                for k in T.serial(wg3_num_k_blocks, unroll=False):
                                    k_buf_idx: T.let[T.int32] = wg3_rs % 4
                                    k_bar_phase: T.let[T.int32] = T.bitwise_and(wg3_rs // 4, 1)
                                    index_buf_idx: T.let[T.int32] = wg3_rs % 4
                                    index_bar_phase: T.let[T.int32] = T.bitwise_and(wg3_rs // 4, 1)
                                    T.cuda.mbarrier_wait(T.address_of(bar_valid_coord_scales_full_buf[index_buf_idx]), T.bitwise_xor(index_bar_phase, 0))
                                    buffer_13 = T.alloc_local((32,))
                                    p_frag = buffer_13.view(128, 32, layout=T.TileLayout(T.S[(128, 32):(1 @ Axis.tid_in_wg, 1)]))
                                    buffer_14 = T.alloc_local((32,))
                                    p_peer_frag = buffer_14.view(128, 32, layout=T.TileLayout(T.S[(128, 32):(1 @ Axis.tid_in_wg, 1)]))
                                    buffer_15 = p_frag.local()
                                    p = buffer_15.view("uint32")
                                    buffer_16 = p_peer_frag.local()
                                    p_peer = buffer_16.view("uint32")
                                    T.cuda.mbarrier_wait(T.address_of(buffer_6[0]), T.bitwise_xor(T.bitwise_and(wg3_rs, 1), 0))
                                    T.ptx.tcgen05("fence::after_thread_sync", "")
                                    if local_warp_idx < 2:
                                        buffer_17 = T.decl_buffer((64, 2, 64), scope="tmem", layout=T.TileLayout(T.S[(64, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                                        buffer_18 = T.decl_buffer((2, 64, 64), scope="tmem", layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                                        p_win = T.decl_buffer((128, 64), scope="tmem", layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                                        local_storage = p_frag.local()
                                        T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], T.uint32(384), "ld", "sync", "aligned", "32x32b", "x32", "", "b32", "")
                                        local_storage_1 = p_peer_frag.local()
                                        T.ptx.tcgen05(local_storage_1[0], local_storage_1[1], local_storage_1[2], local_storage_1[3], local_storage_1[4], local_storage_1[5], local_storage_1[6], local_storage_1[7], local_storage_1[8], local_storage_1[9], local_storage_1[10], local_storage_1[11], local_storage_1[12], local_storage_1[13], local_storage_1[14], local_storage_1[15], local_storage_1[16], local_storage_1[17], local_storage_1[18], local_storage_1[19], local_storage_1[20], local_storage_1[21], local_storage_1[22], local_storage_1[23], local_storage_1[24], local_storage_1[25], local_storage_1[26], local_storage_1[27], local_storage_1[28], local_storage_1[29], local_storage_1[30], local_storage_1[31], T.cuda.get_tmem_addr(T.uint32(384), 0, 32), "ld", "sync", "aligned", "32x32b", "x32", "", "b32", "")
                                    else:
                                        buffer_17 = T.decl_buffer((64, 2, 64), scope="tmem", layout=T.TileLayout(T.S[(64, 2, 64):(1 @ Axis.TLane, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                                        buffer_18 = T.decl_buffer((2, 64, 64), scope="tmem", layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                                        p_win = T.decl_buffer((128, 64), scope="tmem", layout=T.TileLayout(T.S[(2, 64, 64):(64 @ Axis.TLane, 1 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=384)
                                        local_storage = p_peer_frag.local()
                                        T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], T.uint32(384), "ld", "sync", "aligned", "32x32b", "x32", "", "b32", "")
                                        local_storage_1 = p_frag.local()
                                        T.ptx.tcgen05(local_storage_1[0], local_storage_1[1], local_storage_1[2], local_storage_1[3], local_storage_1[4], local_storage_1[5], local_storage_1[6], local_storage_1[7], local_storage_1[8], local_storage_1[9], local_storage_1[10], local_storage_1[11], local_storage_1[12], local_storage_1[13], local_storage_1[14], local_storage_1[15], local_storage_1[16], local_storage_1[17], local_storage_1[18], local_storage_1[19], local_storage_1[20], local_storage_1[21], local_storage_1[22], local_storage_1[23], local_storage_1[24], local_storage_1[25], local_storage_1[26], local_storage_1[27], local_storage_1[28], local_storage_1[29], local_storage_1[30], local_storage_1[31], T.cuda.get_tmem_addr(T.uint32(384), 0, 32), "ld", "sync", "aligned", "32x32b", "x32", "", "b32", "")
                                    T.ptx.tcgen05("wait::ld", "sync", "aligned", "")
                                    T.ptx.tcgen05("fence::before_thread_sync", "")
                                    buffer_17: T.uint32
                                    T.ptx.mapa(buffer_17, T.cuda.cvta_generic_to_shared(T.address_of(bar_P_empty_buf[0])), T.uint32(0), "shared::cluster", "u32", "")
                                    T.ptx.mbarrier(buffer_17, "arrive", "", "", "shared::cluster", "b64", "")
                                    valid_word_offset: T.let[T.int32] = T.if_then_else(local_warp_idx >= 2, 1, 0)
                                    buffer_18 = T.decl_buffer((4, 2), "uint32", data=is_k_valid.data, elem_offset=55552, scope="shared.dyn", align=16)
                                    is_k_valid_u32: T.uint32
                                    T.ptx.ld.shared.u32(is_k_valid_u32, buffer_18.ptr_to([index_buf_idx, valid_word_offset]))
                                    for p_i in T.unroll(32):
                                        invalid_p_predicate: T.let[T.bool] = T.bitwise_and(T.shift_right(is_k_valid_u32, T.Cast("uint32", p_i)), T.uint32(1)) == T.uint32(0)
                                        p[p_i] = T.if_then_else(invalid_p_predicate, T.uint32(4286578688), p[p_i])
                                    sum_pair0: T.uint64
                                    sum_pair1: T.uint64
                                    for exchange_i in T.unroll(8):
                                        exchange_offset: T.int32 = exchange_i * 32 * 4 + lane_idx * 4
                                        p_peer_offset: T.let[T.int32] = exchange_i * 4
                                        buffer_19 = T.decl_buffer((32,), "uint32", data=p_peer.data, scope="local", layout=T.TileLayout(T.S[(32, 1):(1, 1)]))
                                        buffer_20 = T.decl_buffer((32,), "uint32", data=buffer_19.data, scope="local", layout=T.TileLayout(T.S[(32, 1, 1):(1, 1, 1)]))
                                        T.ptx.st(T.cuda.cvta_generic_to_shared(T.address_of(p_exchange[T.bitwise_xor(local_warp_idx, 2), exchange_offset])), buffer_20[p_peer_offset], buffer_20[p_peer_offset + 1], buffer_20[p_peer_offset + 2], buffer_20[p_peer_offset + 3], "", "", "shared", "", "", "", "v4", "u32", "")
                                    T.ptx.bar(T.Cast("uint32", 2 + T.bitwise_and(local_warp_idx, 1)), T.uint32(64), "", "sync", "")
                                    for exchange_i in T.unroll(8):
                                        exchange_offset: T.int32 = exchange_i * 32 * 4 + lane_idx * 4
                                        p_exchange_tmp = T.alloc_local((4,), "uint32")
                                        buffer_19 = T.decl_buffer((4,), "uint32", data=p_exchange_tmp.data, scope="local")
                                        buffer_20 = T.decl_buffer((4,), "uint32", data=buffer_19.data, scope="local", layout=T.TileLayout(T.S[(4, 1):(1, 1)]))
                                        T.ptx.ld(buffer_20[0], buffer_20[1], buffer_20[2], buffer_20[3], T.cuda.cvta_generic_to_shared(T.address_of(p_exchange[local_warp_idx, exchange_offset])), "", "", "shared", "", "", "", "", "", "v4", "u32", "")
                                        p_pair0: T.let[T.uint64] = T.cuda.make_float2(T.cuda.uint_as_float(p[exchange_i * 4]), T.cuda.uint_as_float(p[exchange_i * 4 + 1]))
                                        peer_pair0: T.let[T.uint64] = T.cuda.make_float2(T.cuda.uint_as_float(p_exchange_tmp[0]), T.cuda.uint_as_float(p_exchange_tmp[1]))
                                        T.ptx.add(sum_pair0, p_pair0, peer_pair0, "rn", "", "", "f32x2", "", "")
                                        p[exchange_i * 4] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair0))
                                        p[exchange_i * 4 + 1] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair0))
                                        p_pair1: T.let[T.uint64] = T.cuda.make_float2(T.cuda.uint_as_float(p[exchange_i * 4 + 2]), T.cuda.uint_as_float(p[exchange_i * 4 + 3]))
                                        peer_pair1: T.let[T.uint64] = T.cuda.make_float2(T.cuda.uint_as_float(p_exchange_tmp[2]), T.cuda.uint_as_float(p_exchange_tmp[3]))
                                        T.ptx.add(sum_pair1, p_pair1, peer_pair1, "rn", "", "", "f32x2", "", "")
                                        p[exchange_i * 4 + 2] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair1))
                                        p[exchange_i * 4 + 3] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair1))
                                    cur_pi_max: T.float32 = T.float32("-inf")
                                    for p_i in T.unroll(32):
                                        cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
                                    cur_pi_max = cur_pi_max * T.float32(sm_scale_div_log2)
                                    T.ptx.st.shared.f32(rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max)
                                    T.ptx.bar(T.Cast("uint32", 2 + T.bitwise_and(local_warp_idx, 1)), T.uint32(64), "", "sync", "")
                                    peer_pi_max: T.float32
                                    T.ptx.ld.shared.f32(peer_pi_max, rowwise_max_buf.ptr_to([T.bitwise_xor(idx_in_warpgroup, 64)]))
                                    cur_pi_max = T.max(cur_pi_max, peer_pi_max)
                                    real_mi = T.max(real_mi, cur_pi_max)
                                    should_scale_o: T.let[T.bool] = T.cuda.any_sync(T.uint32(4294967295), cur_pi_max - mi > T.float32(6.0)) != 0
                                    new_max: T.float32
                                    scale_for_old: T.float32
                                    if not should_scale_o:
                                        scale_for_old = T.float32(1.0)
                                        new_max = mi
                                    else:
                                        new_max = T.max(cur_pi_max, mi)
                                        T.ptx.ex2(scale_for_old, mi - new_max, "approx", "ftz", "f32", "")
                                    mi = new_max
                                    s_frag = T.alloc_local((64, 64), "bfloat16", layout=T.TileLayout(T.S[(2, 32, 2, 32):(1 @ Axis.wid_in_wg, 1 @ Axis.laneid, 2 @ Axis.wid_in_wg, 1)]))
                                    buffer_19 = s_frag.local()
                                    s_pack = buffer_19.view("uint32")
                                    cur_sum_pair: T.uint64 = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
                                    neg_new_max_pair: T.let[T.uint64] = T.cuda.make_float2(new_max * T.float32(-1.0), new_max * T.float32(-1.0))
                                    fma_pair: T.uint64
                                    for s_i in T.unroll(16):
                                        p_pair: T.let[T.uint64] = T.cuda.make_float2(T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1]))
                                        T.ptx.fma(fma_pair, p_pair, scale_pair, neg_new_max_pair, "rn", "", "", "f32x2", "", "")
                                        s_x: T.float32
                                        s_y: T.float32
                                        T.ptx.ex2(s_x, T.cuda.float2_x(fma_pair), "approx", "ftz", "f32", "")
                                        T.ptx.ex2(s_y, T.cuda.float2_y(fma_pair), "approx", "ftz", "f32", "")
                                        s_pair: T.let[T.uint64] = T.cuda.make_float2(s_x, s_y)
                                        T.ptx.add(cur_sum_pair, cur_sum_pair, s_pair, "rn", "", "", "f32x2", "", "")
                                        s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)
                                    cur_sum: T.let[T.float32] = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair)
                                    li_tmp: T.float32
                                    T.ptx.fma(li_tmp, li, scale_for_old, cur_sum, "rn", "", "", "f32", "", "")
                                    li = li_tmp
                                    T.cuda.mbarrier_wait(T.address_of(buffer_7[0]), T.bitwise_xor(T.bitwise_xor(T.bitwise_and(wg3_rs, 1), 1), 0))
                                    T.ptx.fence("proxy", "async", "shared::cta", "")
                                    s_base: T.int32 = v_7 // 64 * 2048 + v_7 % 64 * 8
                                    r_local = s_frag.local()
                                    r_words = r_local.view("uint32")
                                    for f in range(4):
                                        ds: T.int32 = f % 4 * 512
                                        dr: T.int32 = f % 4 * 8
                                        s_ptr: T.let = T.ptr_byte_offset(
                                            T.address_of(s_smem_gemm[0, 0]),
                                            (s_base + ds) * BF16_BYTES,
                                            "bfloat16",
                                        )
                                        r_w: T.int32 = dr // 2
                                        T.ptx.st(T.cuda.cvta_generic_to_shared(s_ptr), r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], "", "", "shared", "", "", "", "v4", "u32", "")
                                    if T.bitwise_and(k > 0, should_scale_o):
                                        T.ptx.tcgen05("fence::after_thread_sync", "")
                                        buffer_20 = T.alloc_local((32,))
                                        o_rescale_frag = buffer_20.view(128, 32, layout=T.TileLayout(T.S[(128, 32):(1 @ Axis.tid_in_wg, 1)]))
                                        for chunk_idx in T.unroll(8):
                                            local_storage = o_rescale_frag.local()
                                            T.ptx.tcgen05(local_storage[0], local_storage[1], local_storage[2], local_storage[3], local_storage[4], local_storage[5], local_storage[6], local_storage[7], local_storage[8], local_storage[9], local_storage[10], local_storage[11], local_storage[12], local_storage[13], local_storage[14], local_storage[15], local_storage[16], local_storage[17], local_storage[18], local_storage[19], local_storage[20], local_storage[21], local_storage[22], local_storage[23], local_storage[24], local_storage[25], local_storage[26], local_storage[27], local_storage[28], local_storage[29], local_storage[30], local_storage[31], T.cuda.get_tmem_addr(T.uint32(0), 0, chunk_idx * 32), "ld", "sync", "aligned", "32x32b", "x32", "", "b32", "")
                                            T.ptx.tcgen05("wait::ld", "sync", "aligned", "")
                                            buffer_21 = o_rescale_frag.local(layout=T.TileLayout(T.S[(16, 2):(2, 1)]))
                                            buffer_22 = o_rescale_frag.local(layout=T.TileLayout(T.S[(16, 2):(2, 1)]))
                                            for f in range(16):
                                                dst_lane_indices_0_0: T.int32 = f * 2
                                                dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                                buffer_23: T.uint64
                                                buffer_24: T.uint64
                                                T.ptx.mov(buffer_23, buffer_22[f * 2], buffer_22[f * 2 + 1], "b64", "")
                                                T.ptx.mov(buffer_24, scale_for_old, scale_for_old, "b64", "")
                                                T.ptx.mul(buffer_23, buffer_23, buffer_24, "rz", "ftz", "", "f32x2", "")
                                                T.ptx.mov(buffer_21[f * 2], buffer_21[f * 2 + 1], buffer_23, "b64", "")
                                            local_storage_1 = o_rescale_frag.local()
                                            T.ptx.tcgen05(T.cuda.get_tmem_addr(T.uint32(0), 0, chunk_idx * 32), local_storage_1[0], local_storage_1[1], local_storage_1[2], local_storage_1[3], local_storage_1[4], local_storage_1[5], local_storage_1[6], local_storage_1[7], local_storage_1[8], local_storage_1[9], local_storage_1[10], local_storage_1[11], local_storage_1[12], local_storage_1[13], local_storage_1[14], local_storage_1[15], local_storage_1[16], local_storage_1[17], local_storage_1[18], local_storage_1[19], local_storage_1[20], local_storage_1[21], local_storage_1[22], local_storage_1[23], local_storage_1[24], local_storage_1[25], local_storage_1[26], local_storage_1[27], local_storage_1[28], local_storage_1[29], local_storage_1[30], local_storage_1[31], "st", "sync", "aligned", "32x32b", "x32", "", "b32", "")
                                            T.ptx.tcgen05("wait::st", "sync", "aligned", "")
                                        T.ptx.tcgen05("fence::before_thread_sync", "")
                                    T.ptx.fence("proxy", "async", "shared::cta", "")
                                    buffer_20: T.uint32
                                    T.ptx.mapa(buffer_20, T.cuda.cvta_generic_to_shared(T.address_of(bar_S_O_full_buf[0])), T.uint32(0), "shared::cluster", "u32", "")
                                    T.ptx.mbarrier(buffer_20, "arrive", "", "", "shared::cluster", "b64", "")
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_valid_coord_scales_empty_buf[index_buf_idx])), T.uint32(1), "arrive", "", "", "shared", "b64", "")
                                    wg3_rs = wg3_rs + 1
                                if real_mi == T.float32("-inf"):
                                    li = T.float32(0.0)
                                    mi = T.float32("-inf")
                                T.cuda.mbarrier_wait(T.address_of(bar_li_empty_buf[0]), T.bitwise_xor(T.bitwise_xor(wg3_outer_loop_phase, 1), 0))
                                T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([T.bitwise_xor(idx_in_warpgroup, 64)]), li)
                                T.ptx.bar(T.uint32(1), T.uint32(128), "", "sync", "")
                                peer_li: T.float32
                                T.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup]))
                                li = li + peer_li
                                if idx_in_warpgroup < 64:
                                    head_idx: T.let[T.int32] = cta_idx * 64 + idx_in_warpgroup
                                    attn_sink_value: T.float32 = T.float32(-float("inf"))
                                    if have_attn_sink:
                                        T.ptx.ld.global_.f32(attn_sink_value, attn_sink.ptr_to([head_idx]))
                                    attn_sink_log2: T.let[T.float32] = attn_sink_value * T.float32(1.4426950408889634)
                                    sink_exp: T.float32
                                    T.ptx.ex2(sink_exp, attn_sink_log2 - mi, "approx", "ftz", "f32", "")
                                    output_scale: T.let[T.float32] = T.cuda.fdividef(T.float32(1.0), li + sink_exp)
                                    T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), T.if_then_else(li == T.float32(0.0), T.float32(0.0), output_scale))
                                    T.ptx.mbarrier(T.cuda.cvta_generic_to_shared(T.address_of(bar_li_full_buf[0])), T.uint32(1), "arrive", "", "", "shared", "b64", "")
                                    cur_lse: T.float32
                                    T.ptx.fma(cur_lse, mi, T.float32(0.69314718055994529), T.log(li), "rn", "", "", "f32", "", "")
                                    cur_lse = T.if_then_else(cur_lse == T.float32("-inf"), T.float32("inf"), cur_lse)
                                    T.ptx.st.global_.f32(max_logits.ptr_to([wg3_s_q_idx, head_idx]), real_mi * T.float32(0.69314718055994529))
                                    T.ptx.st.global_.f32(lse.ptr_to([wg3_s_q_idx, head_idx]), cur_lse)
                                T.cuda.mbarrier_wait(T.address_of(buffer_8[0]), T.bitwise_xor(wg3_outer_loop_phase, 0))
                                wg3_next_job: T.uint32
                                query_cancel_first_ctaid_x(wg3_next_job, T.address_of(clc_response[0]))
                                _rem6: T.uint64
                                T.ptx.mapa(_rem6, T.address_of(bar_clc_empty_buf[0]), T.uint32(0), "shared::cluster", "u64", "")
                                T.ptx.mbarrier(T.reinterpret(T.handle().ty, _rem6), T.uint32(1), T.bool(True), "arrive", "", "", "", "b64", "pred")
                                if wg3_next_job == T.uint32(4294967295):
                                    wg3_job_valid = 0
                                else:
                                    wg3_job_block_idx = T.Cast("int32", wg3_next_job)
                                wg3_outer_loop_phase = T.bitwise_xor(wg3_outer_loop_phase, 1)
                            T.cuda.iket.range_end(softmax_token)
                T.cuda.cluster_sync()
    return _kernel
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


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("small_topk:"):
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
