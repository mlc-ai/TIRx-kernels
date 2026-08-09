from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import cache, partial
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla._gemm import tcgen05_config
from tirx_kernels.flashmla._mask import pack_valid_mask8
from tirx_kernels.flashmla._tma import leader_mbar, tma_config
from tvm.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.layout import Axis, S, TileLayout, laneid, wid_in_wg

B_H = 128
B_TOPK = 64
D_QK = 512
D_V = 512
NUM_THREADS = 512
NUM_K_BUFS = 4
NUM_INDEX_BUFS = 4
NUM_WORKER_THREADS = (128 + 4 + (B_TOPK // 8) + 1 + 128) * 2 + 1
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

IKET_EVENT_NAMES = (
    "h128-small-q-load-output",
    "h128-small-kv-load",
    "h128-small-qk-pv-issue",
    "h128-small-valid-mask",
    "h128-small-clc",
    "h128-small-softmax",
)

LAUNCH_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
    "tirx.use_dyn_shared_memory",
)

BF16_BYTES = 2
B_EPI = 64

BAR_WG0_SYNC = 0
BAR_WG2_SYNC = 1
BAR_WG2_WARP02 = 2

WG1_ROWS_PER_WARP = B_TOPK // 4
WG3_ELEMS_PER_THREAD = B_TOPK // 2

# KV gather4 TMA knobs shared by the gather call sites.
_mma_config = partial(tcgen05_config, cta_group=2, smem_desc="local_hoist")
_kv_gather_tma = partial(
    tma_config,
    dispatch="tma_explicit",
    cta_group=2,
    cta_mask=T.uint16(1),
    cache_hint=T.uint64(0x14F0000000000000),
)


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
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    iket = IketProfiler()
    # CUDA_TRANSCRIBE_START: phase1.cuh:24, scoped to KernelTemplate<Prefill, 512>.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    thread_idx = T.thread_id([NUM_THREADS])
    T.warpgroup_id([NUM_THREADS // 128])
    T.warp_id_in_wg([4])
    T.lane_id([32])
    T.thread_id_in_wg([128])
    warp_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 32, 0, 32)
    lane_idx: T.let = thread_idx % 32
    warpgroup_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 128, 0, 32)
    idx_in_warpgroup: T.let = thread_idx % 128

    pool = T.SMEMPool()
    q_smem = pool.alloc_tcgen05_mma_AB((B_H // 2, D_QK), "bfloat16")
    k_smem = pool.alloc_tcgen05_mma_AB((NUM_K_BUFS * B_TOPK, D_QK // 2), "bfloat16")
    s_smem_gemm = pool.alloc_tcgen05_mma_AB(
        (B_H // 2, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
    )
    p_exchange = pool.alloc((4, (B_H // 2 // 2) * (B_TOPK // 2)), "uint32")
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    is_k_valid = pool.alloc((NUM_INDEX_BUFS, B_TOPK // 8), "int8", align=16)

    bar_sQ_full = TMABar(pool, 1, leader=True)
    bar_tQ_empty = TCGen05Bar(pool, 1, leader=True)
    bar_tQ_full = TCGen05Bar(pool, 1, leader=True)
    bar_tOut_full = TCGen05Bar(pool, 1, leader=True)
    bar_tOut_empty = MBarrier(pool, 1, leader=True)
    bar_KV_full = TMABar(pool, NUM_K_BUFS, leader=True)
    bar_KV_empty = TCGen05Bar(pool, NUM_K_BUFS, leader=True)
    bar_P_empty = MBarrier(pool, 1, leader=True)
    bar_QK_done = TCGen05Bar(pool, 1, leader=True)
    bar_SV_done = TCGen05Bar(pool, 1, leader=True)
    bar_S_O_full = MBarrier(pool, 1, leader=True)
    bar_li_full = MBarrier(pool, 1, leader=True)
    bar_li_empty = MBarrier(pool, 1, leader=True)
    bar_valid_coord_scales_full = MBarrier(pool, NUM_INDEX_BUFS, leader=True)
    bar_valid_coord_scales_empty = MBarrier(pool, NUM_INDEX_BUFS, leader=True)
    bar_clc_full = TMABar(pool, 1, leader=True)
    bar_clc_empty = MBarrier(pool, 1, leader=True)
    clc_response = pool.alloc((4,), "uint32", align=16)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    pool.commit()
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_start_addr)
    # O accumulator: one alloc; col halves = B lo/hi gemm outputs (physical col 0-127/128-255),
    # read back as a (128, D_V//2) datapath-D tile via permute+reshape.
    o_tmem = tmem_pool.alloc_tcgen05_mma_D(
        (B_H // 2, D_V), "float32", M=128, cta_group=2, group=(2, 2, 128)
    )
    o_win = o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
    # Q TMEM: one alloc at real 128-lane footprint, batched [2,M,K] head-dim fold (batch==lane-half);
    # q_tmem_fold[b,h,k]=Q[h,256b+k], lane-half b = contiguous D half [256b,+256) matching K gather.
    q_tmem_fold = tmem_pool.alloc_tcgen05_mma_A(
        (2, B_H // 2, D_QK // 2), "bfloat16", M=128, cta_group=2
    )
    tmem_p_col = T.meta_var(tmem_pool.offset)
    tmem_p = tmem_pool.alloc_tcgen05_mma_D((B_H // 2, B_TOPK * 2), "float32", M=128, cta_group=2)
    k_smem_gemm = k_smem.rearrange(
        "(kh row) (buf kl) -> buf row (kh kl)", kh=(D_QK // 2) // 64, buf=NUM_K_BUFS, kl=64
    )

    if warp_idx == 1:
        if T.cuda.elect_sync():
            bar_sQ_full.init(1)
            bar_tQ_empty.init(1)
            bar_tQ_full.init(1)
            bar_tOut_full.init(1)
            bar_tOut_empty.init(256)
            bar_P_empty.init(256)
            bar_QK_done.init(1)
            bar_SV_done.init(1)
            bar_S_O_full.init(256)
            bar_li_full.init(B_H // 2)
            bar_li_empty.init(128)
            bar_clc_full.init(1)
            bar_clc_empty.init(NUM_WORKER_THREADS)
            T.ptx.fence.mbarrier_init.release.cluster()
    elif warp_idx == 2:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
            T.address_of(tmem_start_addr[0]), T.uint32(512)
        )
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
    elif warp_idx == 3:
        if T.cuda.elect_sync():
            for init_stage in T.unroll(NUM_K_BUFS):
                T.ptx.mbarrier.init.shared.b64(bar_KV_full.ptr_to([init_stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(bar_KV_empty.ptr_to([init_stage]), T.uint32(1))
            for init_stage in T.unroll(NUM_INDEX_BUFS):
                T.ptx.mbarrier.init.shared.b64(
                    bar_valid_coord_scales_full.ptr_to([init_stage]), T.uint32(B_TOPK // 8)
                )
                T.ptx.mbarrier.init.shared.b64(
                    bar_valid_coord_scales_empty.ptr_to([init_stage]), T.uint32(128)
                )
            T.ptx.fence.mbarrier_init.release.cluster()

    T.cuda.cluster_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:192-396. Q fetching and O write-back warpgroup.
        q_o_token = iket.range_start("h128-small-q-load-output")
        T.ptx.setmaxnreg.inc.sync.aligned.u32(160)

        @T.inline
        def issue_q_copy(q_s_q_idx, q_outer_loop_phase):
            if warp_idx == 0:
                if T.cuda.elect_sync():
                    T.ptx.cp.async_.bulk.wait_group(0)
                    # Q's head-dim halves interleave per 64-elem chunk, matching the cp fold.
                    q_tma = q.rearrange(
                        "s h (half chunk inner) -> inner h half chunk s",
                        half=2,
                        chunk=D_QK // 64 // 2,
                        inner=64,
                    )
                    q_smem_tma = q_smem.rearrange(
                        "m (chunk c d0) -> d0 m c chunk", chunk=D_QK // 64 // 2, c=2
                    )
                    Tx.copy_async(
                        q_smem_tma[:, :, :, :],
                        q_tma.chunk((None, 2, None, None, None))[:, cta_idx, :, :, q_s_q_idx],
                        **tma_config(
                            mbar=leader_mbar(bar_sQ_full.ptr_to([0])),
                            cta_group=2,
                            cache_hint=T.uint64(0x12F0000000000000),
                        ),
                    )
                    if cta_idx == 0:
                        bar_sQ_full.arrive(0, tx_count=B_H * D_QK * BF16_BYTES)
                        bar_sQ_full.wait(0, q_outer_loop_phase)
                        bar_tQ_empty.wait(0, q_outer_loop_phase ^ 1)
                        T.ptx.tcgen05.fence__after_thread_sync()
                        q_tmem_cp = q_tmem_fold.rearrange("b h (dc di) -> h dc b di", di=64)
                        Tx.copy_async(
                            q_tmem_cp[:, :, :, :],
                            q_smem.view(B_H // 2, D_QK // 128, 2, 64)[:, :, :, :],
                            shape="128x256b",
                            cta_group=2,
                        )
                        bar_tQ_full.arrive(0, cta_group=2, cta_mask=3)

        @T.inline
        def perform_o_copy_out(o_s_q_idx, o_outer_loop_phase, is_last_o: T.constexpr):
            bar_li_full.wait(0, o_outer_loop_phase)
            output_scale: T.let = rowwise_li_buf[idx_in_warpgroup % 64]
            bar_li_empty.arrive(0)

            bar_tOut_full.wait(0, o_outer_loop_phase)
            if is_last_o:
                if T.cuda.elect_sync():
                    T.ptx.griddepcontrol.launch_dependents()

            o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "float32")
            o_epi = o_epi_frag.local()
            o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "bfloat16")
            q_smem_win = q_smem.rearrange("h (b r) -> (b h) r", b=2)
            for epi_k in T.unroll((D_V // 2) // B_EPI):
                Tx.wg.copy_async(
                    o_epi_frag[:, :], o_win.chunk((None, (D_V // 2) // B_EPI))[:, epi_k]
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                if epi_k == 0:
                    if is_last_o:
                        bar_tQ_full.wait(0, o_outer_loop_phase)
                    else:
                        bar_tQ_full.wait(0, o_outer_loop_phase ^ 1)
                    T.ptx.fence.proxy.async_.shared__cta()
                if epi_k == ((D_V // 2) // B_EPI) - 1:
                    bar_tOut_empty.arrive(0, remote=T.uint32(0))
                Tx.wg.mul(o_epi_frag[:, :], o_epi_frag[:, :], output_scale)
                Tx.wg.cast(o_epi_bf16_frag[:, :], o_epi_frag[:, :])
                Tx.wg.copy(
                    q_smem_win.chunk((None, (D_V // 2) // B_EPI))[:, epi_k], o_epi_bf16_frag[:, :]
                )

            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
            if warp_idx == 0:
                if T.cuda.elect_sync():
                    Tx.copy_async(
                        out.chunk((None, 2, None))[o_s_q_idx, cta_idx, :],
                        q_smem[:, :],
                        **tma_config(),
                    )
                    T.ptx.cp.async_.bulk.commit_group()

        wg0_job_valid: T.int32 = 1
        wg0_job_block_idx: T.int32 = block_idx
        wg0_outer_loop_phase: T.int32 = 0
        last_valid: T.int32 = 0
        last_s_q_idx: T.int32 = 0
        last_outer_loop_phase: T.int32 = 0

        while wg0_job_valid != 0:
            wg0_s_q_idx: T.let = wg0_job_block_idx // 2
            issue_q_copy(wg0_s_q_idx, wg0_outer_loop_phase)

            if last_valid != 0:
                perform_o_copy_out(last_s_q_idx, last_outer_loop_phase, False)
            else:
                bar_tQ_full.wait(0, wg0_outer_loop_phase)
            last_valid = 1
            last_s_q_idx = wg0_s_q_idx
            last_outer_loop_phase = wg0_outer_loop_phase

            bar_clc_full.wait(0, wg0_outer_loop_phase)
            wg0_next_job = T.local_scalar("uint32")
            query_cancel_first_ctaid_x(wg0_next_job, T.address_of(clc_response[0]))
            _rem1 = T.alloc_local([1], "uint64")
            T.ptx.mapa.shared__cluster.u64(_rem1[0], bar_clc_empty.ptr_to([0]), T.uint32(0))
            T.ptx.mbarrier.arrive.b64(_rem1[0], T.uint32(1), pred=T.bool(True))
            if wg0_next_job == T.uint32(0xFFFFFFFF):
                wg0_job_valid = 0
            else:
                wg0_job_block_idx = T.cast(wg0_next_job, "int32")
            wg0_outer_loop_phase = wg0_outer_loop_phase ^ 1

        if last_valid != 0:
            if warp_idx == 0:
                if T.cuda.elect_sync():
                    T.ptx.cp.async_.bulk.wait_group(0)
            T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
            perform_o_copy_out(last_s_q_idx, last_outer_loop_phase, True)

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(T.uint32(0), T.uint32(512))
        iket.range_end(q_o_token)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:397-451. Prefill KV gather producer.
        kv_gather_token = iket.range_start("h128-small-kv-load")
        T.ptx.setmaxnreg.dec.sync.aligned.u32(80)
        # Source uses canonical_warp_idx() here, not canonical_warp_idx_sync().
        wg1_warp_idx: T.let = thread_idx // 32 - 4
        if T.cuda.elect_sync():
            wg1_job_valid: T.int32 = 1
            wg1_job_block_idx: T.int32 = block_idx
            wg1_outer_loop_phase: T.int32 = 0
            wg1_rs: T.int32 = 0
            while wg1_job_valid != 0:
                wg1_s_q_idx: T.let = wg1_job_block_idx // 2
                wg1_topk_len: T.let = (
                    T.cuda.ldg(topk_length.ptr_to([wg1_s_q_idx]), "int32")
                    if have_topk_length
                    else topk
                )
                wg1_num_k_blocks: T.let = T.max((wg1_topk_len + B_TOPK - 1) // B_TOPK, 1)
                wg1_g_indices_base: T.let = wg1_s_q_idx * stride_indices_s_q

                for k in T.serial(0, wg1_num_k_blocks, unroll=False):
                    k_buf_idx: T.let = wg1_rs % NUM_K_BUFS
                    k_bar_phase: T.let = (wg1_rs // NUM_K_BUFS) & 1
                    cur_indices = T.alloc_local((WG1_ROWS_PER_WARP,), "int32")
                    for local_row in T.unroll(WG1_ROWS_PER_WARP // 8):
                        row: T.let = local_row * (4 * 8) + wg1_warp_idx * 8
                        row_base: T.let = wg1_g_indices_base + k * B_TOPK + row
                        Tx.copy(
                            cur_indices[local_row * 8 : local_row * 8 + 8],
                            indices[row_base : row_base + 8],
                            dispatch="vec_256b",
                            cache="nc",
                            l1_evict="L1::no_allocate",
                            l2_evict="L2::evict_first",
                            prefetch_size="L2::256B",
                        )
                    bar_KV_empty.wait(k_buf_idx, k_bar_phase ^ 1)
                    k_smem_gemm_cur = k_smem_gemm.sub[k_buf_idx]
                    src_col: T.let = cta_idx * (D_QK // 2)
                    # Rows interleave (row_group, warp, pair, lane4): this warp's 8-row stripes of each
                    # 64-col chunk. Col reshaped to (chunk,64); row picks this warp's rows via rank-preserving tile.
                    k_gather_tile = (
                        k_smem_gemm_cur.view(B_TOPK, (D_QK // 2) // 64, 64).tile(0, (-1, 4, 2, 4))
                    )[:, wg1_warp_idx, :, :]
                    kv_tma = kv.view(
                        s_kv, D_QK, layout=TileLayout(S[(s_kv, D_QK) : (stride_kv_s_kv, 1)])
                    )
                    k_gather_tile_2d = k_gather_tile.view(WG1_ROWS_PER_WARP, D_QK // 2)
                    for row_group in T.unroll(WG1_ROWS_PER_WARP // 4):
                        for col_atom in T.unroll((D_QK // 2) // 64):
                            col = T.meta_var(src_col + col_atom * 64)
                            Tx.copy_async(
                                k_gather_tile_2d[
                                    row_group * 4 : row_group * 4 + 4,
                                    col_atom * 64 : col_atom * 64 + 64,
                                ],
                                kv_tma[0:1, col : col + 64],
                                **_kv_gather_tma(
                                    mbar=leader_mbar(bar_KV_full.ptr_to([k_buf_idx])),
                                    gather4=[
                                        cur_indices[row_group * 4 + lane] for lane in range(4)
                                    ],
                                ),
                            )
                    wg1_rs = wg1_rs + 1

                bar_clc_full.wait(0, wg1_outer_loop_phase)
                wg1_next_job = T.local_scalar("uint32")
                query_cancel_first_ctaid_x(wg1_next_job, T.address_of(clc_response[0]))
                _rem2 = T.alloc_local([1], "uint64")
                T.ptx.mapa.shared__cluster.u64(_rem2[0], bar_clc_empty.ptr_to([0]), T.uint32(0))
                T.ptx.mbarrier.arrive.b64(_rem2[0], T.uint32(1), pred=T.bool(True))
                if wg1_next_job == T.uint32(0xFFFFFFFF):
                    wg1_job_valid = 0
                else:
                    wg1_job_block_idx = T.cast(wg1_next_job, "int32")
                wg1_outer_loop_phase = wg1_outer_loop_phase ^ 1
        iket.range_end(kv_gather_token)

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:533-787. UMMA, valid-mask loading, and CLC producer.
        T.ptx.setmaxnreg.dec.sync.aligned.u32(80)

        if (warp_idx == 8) & (cta_idx == 0):
            mma_token = iket.range_start("h128-small-qk-pv-issue")
            if T.cuda.elect_sync():
                umma_job_valid: T.int32 = 1
                umma_job_block_idx: T.int32 = block_idx
                umma_outer_loop_phase: T.int32 = 0
                umma_rs: T.int32 = 0
                while umma_job_valid != 0:
                    umma_s_q_idx: T.let = umma_job_block_idx // 2
                    umma_topk_len: T.let = (
                        T.cuda.ldg(topk_length.ptr_to([umma_s_q_idx]), "int32")
                        if have_topk_length
                        else topk
                    )
                    umma_num_k_blocks: T.let = T.max((umma_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    bar_tQ_full.wait(0, umma_outer_loop_phase)

                    for k in T.serial(0, umma_num_k_blocks + 1, unroll=False):
                        if k < umma_num_k_blocks:
                            k_buf_idx: T.let = umma_rs % NUM_K_BUFS
                            k_bar_phase: T.let = (umma_rs // NUM_K_BUFS) & 1
                            p_bar_phase: T.let = umma_rs & 1
                            bar_P_empty.wait(0, p_bar_phase ^ 1)
                            bar_KV_full.arrive(k_buf_idx, tx_count=B_TOPK * D_QK * BF16_BYTES)
                            bar_KV_full.wait(k_buf_idx, k_bar_phase)
                            T.ptx.tcgen05.fence__after_thread_sync()
                            qk_accumulate: T.uint32 = 0
                            Tx.gemm_async(
                                tmem_p[:, :],
                                q_tmem_fold[:, :, :],
                                k_smem_gemm[k_buf_idx, :, :],
                                **_mma_config(accum=qk_accumulate),
                            )
                            qk_accumulate = T.uint32(1)
                            bar_QK_done.arrive(0, cta_group=2, cta_mask=3)
                            if k == umma_num_k_blocks - 1:
                                T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
                                    bar_tQ_empty.ptr_to([0])
                                )

                        if k > 0:
                            prev_k: T.let = k - 1
                            prev_rs: T.let = umma_rs - 1
                            prev_buf: T.let = prev_rs % NUM_K_BUFS
                            prev_s_o_phase: T.let = prev_rs & 1
                            bar_S_O_full.wait(0, prev_s_o_phase)
                            if prev_k == 0:
                                bar_tOut_empty.wait(0, umma_outer_loop_phase ^ 1)
                            T.ptx.tcgen05.fence__after_thread_sync()
                            o_accumulate: T.uint32 = T.if_then_else(
                                prev_k == 0, T.uint32(0), T.uint32(1)
                            )

                            @T.inline
                            def gemm_o(dst, col_lo, col_hi):
                                Tx.gemm_async(
                                    dst[:, :],
                                    s_smem_gemm[:, :],
                                    k_smem_gemm[prev_buf, :, col_lo:col_hi],
                                    transB=True,
                                    **_mma_config(accum=o_accumulate),
                                )

                            gemm_o(o_tmem.sub[:, 0 : D_V // 2], 0, D_V // 4)
                            gemm_o(o_tmem.sub[:, D_V // 2 : D_V], D_V // 4, D_V // 2)
                            o_accumulate = T.uint32(1)
                            bar_SV_done.arrive(0, cta_group=2, cta_mask=3)
                            bar_KV_empty.arrive(prev_buf, cta_group=2, cta_mask=3)

                        if k != umma_num_k_blocks:
                            umma_rs = umma_rs + 1

                    T.ptx.tcgen05.fence__before_thread_sync()
                    bar_tOut_full.arrive(0, cta_group=2, cta_mask=3)

                    bar_clc_full.wait(0, umma_outer_loop_phase)
                    umma_next_job = T.local_scalar("uint32")
                    query_cancel_first_ctaid_x(umma_next_job, T.address_of(clc_response[0]))
                    _rem3 = T.alloc_local([1], "uint64")
                    T.ptx.mapa.shared__cluster.u64(_rem3[0], bar_clc_empty.ptr_to([0]), T.uint32(0))
                    T.ptx.mbarrier.arrive.b64(_rem3[0], T.uint32(1), pred=T.bool(True))
                    if umma_next_job == T.uint32(0xFFFFFFFF):
                        umma_job_valid = 0
                    else:
                        umma_job_block_idx = T.cast(umma_next_job, "int32")
                    umma_outer_loop_phase = umma_outer_loop_phase ^ 1
            iket.range_end(mma_token)

        elif warp_idx == 9:
            valid_mask_token = iket.range_start("h128-small-valid-mask")
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                valid_job_valid: T.int32 = 1
                valid_job_block_idx: T.int32 = block_idx
                valid_outer_loop_phase: T.int32 = 0
                valid_rs: T.int32 = 0
                while valid_job_valid != 0:
                    valid_s_q_idx: T.let = valid_job_block_idx // 2
                    valid_topk_len: T.let = (
                        T.cuda.ldg(topk_length.ptr_to([valid_s_q_idx]), "int32")
                        if have_topk_length
                        else topk
                    )
                    valid_num_k_blocks: T.let = T.max((valid_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    valid_g_indices_base: T.let = valid_s_q_idx * stride_indices_s_q
                    for k in T.serial(0, valid_num_k_blocks, unroll=False):
                        row_base: T.let = valid_g_indices_base + k * B_TOPK + lane_idx * 8
                        Tx.copy(
                            lane_indices[0:8],
                            indices[row_base : row_base + 8],
                            dispatch="vec_256b",
                            cache="nc",
                            l1_evict="L1::no_allocate",
                            l2_evict="L2::evict_normal",
                            prefetch_size="L2::256B",
                        )
                        abs_pos_start: T.let = k * B_TOPK
                        mask: T.let = pack_valid_mask8(
                            lane_indices, abs_pos_start, lane_idx, valid_topk_len, s_kv
                        )
                        index_buf_idx: T.let = valid_rs % NUM_INDEX_BUFS
                        index_bar_phase: T.let = (valid_rs // NUM_INDEX_BUFS) & 1
                        bar_valid_coord_scales_empty.wait(index_buf_idx, index_bar_phase ^ 1)
                        is_k_valid[index_buf_idx, lane_idx] = mask
                        bar_valid_coord_scales_full.arrive(index_buf_idx)
                        valid_rs = valid_rs + 1

                    bar_clc_full.wait(0, valid_outer_loop_phase)
                    valid_next_job = T.local_scalar("uint32")
                    query_cancel_first_ctaid_x(valid_next_job, T.address_of(clc_response[0]))
                    _rem4 = T.alloc_local([1], "uint64")
                    T.ptx.mapa.shared__cluster.u64(_rem4[0], bar_clc_empty.ptr_to([0]), T.uint32(0))
                    T.ptx.mbarrier.arrive.b64(_rem4[0], T.uint32(1), pred=T.bool(True))
                    if valid_next_job == T.uint32(0xFFFFFFFF):
                        valid_job_valid = 0
                    else:
                        valid_job_block_idx = T.cast(valid_next_job, "int32")
                    valid_outer_loop_phase = valid_outer_loop_phase ^ 1
            iket.range_end(valid_mask_token)

        elif warp_idx >= 10:
            clc_token = iket.sentinel_token("h128-small-clc")
            if warp_idx == 10:
                clc_token = iket.range_start("h128-small-clc")
            if T.cuda.elect_sync():
                if warp_idx == 10:
                    clc_job_valid: T.int32 = 1
                    clc_outer_loop_phase: T.int32 = 0
                    while clc_job_valid != 0:
                        if cta_idx == 0:
                            bar_clc_empty.wait(0, clc_outer_loop_phase ^ 1)
                            T.ptx[
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            ](T.address_of(clc_response[0]), bar_clc_full.ptr_to([0]))
                        bar_clc_full.arrive(0, tx_count=16)

                        bar_clc_full.wait(0, clc_outer_loop_phase)
                        clc_next_job = T.local_scalar("uint32")
                        query_cancel_first_ctaid_x(clc_next_job, T.address_of(clc_response[0]))
                        _rem5 = T.alloc_local([1], "uint64")
                        T.ptx.mapa.shared__cluster.u64(
                            _rem5[0], bar_clc_empty.ptr_to([0]), T.uint32(0)
                        )
                        T.ptx.mbarrier.arrive.b64(_rem5[0], T.uint32(1), pred=T.bool(True))
                        if clc_next_job == T.uint32(0xFFFFFFFF):
                            clc_job_valid = 0
                        clc_outer_loop_phase = clc_outer_loop_phase ^ 1
            iket.range_end(clc_token)

    else:
        # CUDA phase1.cuh:788-921. Scale/exp warpgroup.
        softmax_token = iket.range_start("h128-small-softmax")
        T.ptx.setmaxnreg.inc.sync.aligned.u32(160)
        local_warp_idx: T.let = warp_idx - 12
        wg3_job_valid: T.int32 = 1
        wg3_job_block_idx: T.int32 = block_idx
        wg3_outer_loop_phase: T.int32 = 0
        wg3_rs: T.int32 = 0
        while wg3_job_valid != 0:
            wg3_s_q_idx: T.let = wg3_job_block_idx // 2
            wg3_topk_len: T.let = (
                T.cuda.ldg(topk_length.ptr_to([wg3_s_q_idx]), "int32") if have_topk_length else topk
            )
            wg3_num_k_blocks: T.let = T.max((wg3_topk_len + B_TOPK - 1) // B_TOPK, 1)
            mi: T.float32 = MAX_INIT_VAL
            li: T.float32 = 0.0
            real_mi: T.float32 = T.float32(-float("inf"))
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

            for k in T.serial(0, wg3_num_k_blocks, unroll=False):
                k_buf_idx: T.let = wg3_rs % NUM_K_BUFS
                k_bar_phase: T.let = (wg3_rs // NUM_K_BUFS) & 1
                index_buf_idx: T.let = wg3_rs % NUM_INDEX_BUFS
                index_bar_phase: T.let = (wg3_rs // NUM_INDEX_BUFS) & 1
                bar_valid_coord_scales_full.wait(index_buf_idx, index_bar_phase)
                p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, WG3_ELEMS_PER_THREAD), "float32")
                p_peer_frag = T.alloc_tcgen05_ldst_frag(
                    "32x32b", (128, WG3_ELEMS_PER_THREAD), "float32"
                )
                p = p_frag.local().view("uint32")
                p_peer = p_peer_frag.local().view("uint32")
                bar_QK_done.wait(0, wg3_rs & 1)
                T.ptx.tcgen05.fence__after_thread_sync()

                @T.inline
                def load_p(lo_dst, hi_dst):
                    # datapath-B P read back as (128, B_TOPK) identity: merge the
                    # two lane-halves into 128 rows.
                    p_win = tmem_p.rearrange("h (b t) -> (b h) t", b=2)
                    Tx.wg.copy_async(lo_dst[:, :], p_win.chunk((None, 2))[:, 0])
                    Tx.wg.copy_async(hi_dst[:, :], p_win.chunk((None, 2))[:, 1])

                if local_warp_idx < 2:
                    load_p(p_frag, p_peer_frag)
                else:
                    load_p(p_peer_frag, p_frag)
                T.ptx.tcgen05.wait__ld.sync.aligned()
                T.ptx.tcgen05.fence__before_thread_sync()
                bar_P_empty.arrive(0, remote=T.uint32(0))

                valid_word_offset: T.let = T.if_then_else(
                    local_warp_idx >= 2, WG3_ELEMS_PER_THREAD // 32, 0
                )
                is_k_valid_u32: T.let = is_k_valid.view("uint32")[index_buf_idx, valid_word_offset]
                for p_i in T.unroll(WG3_ELEMS_PER_THREAD):
                    invalid_p_predicate: T.let = T.bitwise_and(
                        T.shift_right(is_k_valid_u32, T.uint32(p_i)), T.uint32(1)
                    ) == T.uint32(0)
                    p[p_i] = T.if_then_else(invalid_p_predicate, T.uint32(0xFF800000), p[p_i])

                sum_pair0: T.uint64
                sum_pair1: T.uint64
                for exchange_i in T.unroll(WG3_ELEMS_PER_THREAD // 4):
                    exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                    p_peer_offset: T.let = exchange_i * 4
                    Tx.copy(
                        p_exchange[local_warp_idx ^ 2, exchange_offset : exchange_offset + 4],
                        p_peer[p_peer_offset : p_peer_offset + 4],
                        dispatch="vec_128b",
                    )
                T.ptx.bar.sync(T.uint32(BAR_WG2_WARP02 + (local_warp_idx & 1)), 64)
                for exchange_i in T.unroll(WG3_ELEMS_PER_THREAD // 4):
                    exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                    p_exchange_tmp = T.alloc_local((4,), "uint32")
                    Tx.copy(
                        p_exchange_tmp[0:4],
                        p_exchange[local_warp_idx, exchange_offset : exchange_offset + 4],
                        dispatch="vec_128b",
                    )
                    p_pair0: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[exchange_i * 4]),
                        T.cuda.uint_as_float(p[exchange_i * 4 + 1]),
                    )
                    peer_pair0: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p_exchange_tmp[0]),
                        T.cuda.uint_as_float(p_exchange_tmp[1]),
                    )
                    T.ptx.add.rn.f32x2(sum_pair0, p_pair0, peer_pair0)
                    p[exchange_i * 4] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair0))
                    p[exchange_i * 4 + 1] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair0))
                    p_pair1: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[exchange_i * 4 + 2]),
                        T.cuda.uint_as_float(p[exchange_i * 4 + 3]),
                    )
                    peer_pair1: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p_exchange_tmp[2]),
                        T.cuda.uint_as_float(p_exchange_tmp[3]),
                    )
                    T.ptx.add.rn.f32x2(sum_pair1, p_pair1, peer_pair1)
                    p[exchange_i * 4 + 2] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair1))
                    p[exchange_i * 4 + 3] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair1))

                cur_pi_max: T.float32 = T.float32(-float("inf"))
                for p_i in T.unroll(WG3_ELEMS_PER_THREAD):
                    cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
                cur_pi_max = cur_pi_max * sm_scale_div_log2
                rowwise_max_buf[idx_in_warpgroup] = cur_pi_max
                T.ptx.bar.sync(T.uint32(BAR_WG2_WARP02 + (local_warp_idx & 1)), 64)
                cur_pi_max = T.max(cur_pi_max, rowwise_max_buf[idx_in_warpgroup ^ 64])
                real_mi = T.max(real_mi, cur_pi_max)
                should_scale_o: T.let = (
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

                # S frag: warpgroup-distributed (B_H//2, B_TOPK) tile. Thread idx owns row h = idx%64
                # and k half [32*(idx//64), +32) in 8-elem chunks (from the packing loop below).
                s_frag = T.alloc_buffer(
                    (B_H // 2, B_TOPK),
                    "bfloat16",
                    scope="local",
                    layout=TileLayout(
                        S[(2, 32, 2, B_TOPK // 2) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                    ),
                )
                s_pack = s_frag.local().view("uint32")
                cur_sum_pair: T.uint64 = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
                neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
                fma_pair: T.uint64
                for s_i in T.unroll(WG3_ELEMS_PER_THREAD // 2):
                    p_pair: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                    )
                    T.ptx.fma.rn.f32x2(fma_pair, p_pair, scale_pair, neg_new_max_pair)
                    s_x: T.float32
                    s_y: T.float32
                    T.ptx.ex2.approx.ftz.f32(s_x, T.cuda.float2_x(fma_pair))
                    T.ptx.ex2.approx.ftz.f32(s_y, T.cuda.float2_y(fma_pair))
                    s_pair: T.let = T.cuda.make_float2(s_x, s_y)
                    T.ptx.add.rn.f32x2(cur_sum_pair, cur_sum_pair, s_pair)
                    s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)
                cur_sum: T.let = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair)
                li_tmp: T.float32
                T.ptx.fma.rn.f32(li_tmp, li, scale_for_old, cur_sum)
                li = li_tmp

                bar_SV_done.wait(0, (wg3_rs & 1) ^ 1)
                T.ptx.fence.proxy.async_.shared__cta()
                Tx.wg.copy(s_smem_gemm[:, :], s_frag[:, :])

                if (k > 0) & should_scale_o:
                    T.ptx.tcgen05.fence__after_thread_sync()
                    o_rescale_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32")
                    for chunk_idx in T.unroll((D_V // 2) // 32):
                        Tx.wg.copy_async(
                            o_rescale_frag[:, :],
                            o_win.chunk((None, (D_V // 2) // 32))[:, chunk_idx],
                        )
                        T.ptx.tcgen05.wait__ld.sync.aligned()
                        Tx.wg.mul(o_rescale_frag[:, :], o_rescale_frag[:, :], scale_for_old)
                        Tx.wg.copy_async(
                            o_win.chunk((None, (D_V // 2) // 32))[:, chunk_idx],
                            o_rescale_frag[:, :],
                        )
                        T.ptx.tcgen05.wait__st.sync.aligned()
                    T.ptx.tcgen05.fence__before_thread_sync()

                T.ptx.fence.proxy.async_.shared__cta()
                bar_S_O_full.arrive(0, remote=T.uint32(0))
                bar_valid_coord_scales_empty.arrive(index_buf_idx)
                wg3_rs = wg3_rs + 1

            if real_mi == T.float32(-float("inf")):
                li = 0.0
                mi = T.float32(-float("inf"))

            bar_li_empty.wait(0, wg3_outer_loop_phase ^ 1)
            rowwise_li_buf[idx_in_warpgroup ^ 64] = li
            T.ptx.bar.sync(T.uint32(BAR_WG2_SYNC), 128)
            li = li + rowwise_li_buf[idx_in_warpgroup]

            if idx_in_warpgroup < B_H // 2:
                head_idx: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
                attn_sink_log2: T.let = (
                    T.cuda.ldg(attn_sink.ptr_to([head_idx]), "float32") * LOG_2_E
                    if have_attn_sink
                    else T.float32(-float("inf"))
                )
                sink_exp: T.float32
                T.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi)
                output_scale: T.let = T.cuda.fdividef(T.float32(1.0), li + sink_exp)
                rowwise_li_buf[idx_in_warpgroup] = T.if_then_else(li == 0.0, 0.0, output_scale)
                bar_li_full.arrive(0)
                cur_lse: T.float32
                T.ptx.fma.rn.f32(cur_lse, mi, LN_2, T.log(li))
                cur_lse = T.if_then_else(
                    cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
                )
                max_logits[wg3_s_q_idx, head_idx] = real_mi * LN_2
                lse[wg3_s_q_idx, head_idx] = cur_lse

            bar_clc_full.wait(0, wg3_outer_loop_phase)
            wg3_next_job = T.local_scalar("uint32")
            query_cancel_first_ctaid_x(wg3_next_job, T.address_of(clc_response[0]))
            _rem6 = T.alloc_local([1], "uint64")
            T.ptx.mapa.shared__cluster.u64(_rem6[0], bar_clc_empty.ptr_to([0]), T.uint32(0))
            T.ptx.mbarrier.arrive.b64(_rem6[0], T.uint32(1), pred=T.bool(True))
            if wg3_next_job == T.uint32(0xFFFFFFFF):
                wg3_job_valid = 0
            else:
                wg3_job_block_idx = T.cast(wg3_next_job, "int32")
            wg3_outer_loop_phase = wg3_outer_loop_phase ^ 1
        iket.range_end(softmax_token)

    T.cuda.cluster_sync()


# The dispatcher-selected SM100 forms are frozen below as explicit TMA,
# tcgen05, TMEM, packed-register, and PTX memory operations.  The tile
# implementation above remains an internal A/B reference only.
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
                                buffer_20 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                                for f in range(32):
                                    dst_lane_indices_0_0: T.int32 = f * 2
                                    dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                    T.cuda.func_call("tvm_builtin_cast_float32x2_bfloat16x2", T.address_of(buffer_19[f * 2]), T.address_of(buffer_20[f * 2]), source_code="\n__forceinline__ __device__ void tvm_builtin_cast_float32x2_bfloat16x2(void* dst, void* src) {\n    ((nv_bfloat162*)dst)[0] = __float22bfloat162_rn(((float2*)src)[0]);\n}\n")
                                r_local = o_epi_bf16_frag.local()
                                r_words = r_local.view("uint32")
                                for f in range(8):
                                    ds: T.int32 = f % 8 * 8
                                    dr: T.int32 = f % 8 * 8
                                    s_off: T.int32 = v_5 // 64 * 16384 + epi_k % 4 * 4096 + v_5 % 64 * 64 + T.bitwise_xor(f * 8, T.shift_left(T.bitwise_and(v_5 // 64 * 256 + epi_k % 4 * 64 + v_5 % 64, 7), 3))
                                    s_ptr: T.let[T.handle] = T.cuda.func_call("tvm_builtin_pointer_offset", T.address_of(q_smem_win[0, 0]), s_off, source_code="\ntemplate <typename T>\n__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {\n    return ptr + offset;\n}\n", return_type=T.handle().ty)
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
                            buffer_20 = o_epi_frag.local(layout=T.TileLayout(T.S[(32, 2):(2, 1)]))
                            for f in range(32):
                                dst_lane_indices_0_0: T.int32 = f * 2
                                dst_lane_indices_1_0: T.int32 = f * 2 + 1
                                T.cuda.func_call("tvm_builtin_cast_float32x2_bfloat16x2", T.address_of(buffer_19[f * 2]), T.address_of(buffer_20[f * 2]), source_code="\n__forceinline__ __device__ void tvm_builtin_cast_float32x2_bfloat16x2(void* dst, void* src) {\n    ((nv_bfloat162*)dst)[0] = __float22bfloat162_rn(((float2*)src)[0]);\n}\n")
                            r_local = o_epi_bf16_frag.local()
                            r_words = r_local.view("uint32")
                            for f in range(8):
                                ds: T.int32 = f % 8 * 8
                                dr: T.int32 = f % 8 * 8
                                s_off: T.int32 = v_6 // 64 * 16384 + epi_k % 4 * 4096 + v_6 % 64 * 64 + T.bitwise_xor(f * 8, T.shift_left(T.bitwise_and(v_6 // 64 * 256 + epi_k % 4 * 64 + v_6 % 64, 7), 3))
                                s_ptr: T.let[T.handle] = T.cuda.func_call("tvm_builtin_pointer_offset", T.address_of(q_smem_win[0, 0]), s_off, source_code="\ntemplate <typename T>\n__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {\n    return ptr + offset;\n}\n", return_type=T.handle().ty)
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
                                                            T.ptx.tcgen05(T.Cast("uint32", ni * 64 + 384), T.Cast("uint32", ki * 8 + 256), T.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descB_local, (ki // 1024 * 16384 + ni * 16384 + k_buf_idx * 16384 + ki % 16 // 4 * 4096 + ki % 1024 // 16 * 64 + ki % 4 * 16) // 8, source_code="\n__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {\n    SmemDescriptor desc;\n    desc.desc_ = desc_base;\n    desc.lo += static_cast<uint32_t>(offset);\n    return desc.desc_;\n}\n", return_type="uint64"), T.uint32(136316048), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), ki != 0 or T.Cast("bool", qk_accumulate), "mma", "cta_group::2", "kind::f16", "p12")
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
                                                            T.ptx.tcgen05(T.Cast("uint32", ni * 128), T.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descA_local, (ki % 4 * 1024 + mi * 512 + ki // 4 * 8) // 8, source_code="\n__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {\n    SmemDescriptor desc;\n    desc.desc_ = desc_base;\n    desc.lo += static_cast<uint32_t>(offset);\n    return desc.desc_;\n}\n", return_type="uint64"), T.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descB_local, ((ki * 16 + ni) // 64 * 16384 + prev_buf * 16384 + (ki * 16 + ni) % 64 * 64) // 8, source_code="\n__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {\n    SmemDescriptor desc;\n    desc.desc_ = desc_base;\n    desc.lo += static_cast<uint32_t>(offset);\n    return desc.desc_;\n}\n", return_type="uint64"), T.uint32(138478736), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), ki != 0 or T.Cast("bool", o_accumulate), "mma", "cta_group::2", "kind::f16", "p12")
                                                buffer_14 = T.decl_buffer((64, 256), scope="tmem", layout=T.TileLayout(T.S[(64, 1, 2, 128):(1 @ Axis.TLane, 128 @ Axis.TCol, 64 @ Axis.TLane, 1 @ Axis.TCol)]), allocated_addr=128)
                                                descB_local_1: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB_local_1), T.address_of(k_smem_gemm[0, 0, 0]), 512, 64, 3)
                                                descA_local_1: T.uint64
                                                T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA_local_1), T.address_of(s_smem_gemm[0, 0]), 64, 8, 0)
                                                for mi in T.unroll(1):
                                                    for ni in T.unroll(1):
                                                        for ki in T.unroll(4):
                                                            T.ptx.tcgen05(T.Cast("uint32", ni * 128 + 128), T.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descA_local_1, (ki % 4 * 1024 + mi * 512 + ki // 4 * 8) // 8, source_code="\n__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {\n    SmemDescriptor desc;\n    desc.desc_ = desc_base;\n    desc.lo += static_cast<uint32_t>(offset);\n    return desc.desc_;\n}\n", return_type="uint64"), T.cuda.func_call("tvm_builtin_smem_desc_add_16B_offset", descB_local_1, ((ki * 16 + ni) // 64 * 16384 + prev_buf * 16384 + (ki * 16 + ni) % 64 * 64 + 8192) // 8, source_code="\n__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(uint64_t desc_base, int32_t offset) {\n    SmemDescriptor desc;\n    desc.desc_ = desc_base;\n    desc.lo += static_cast<uint32_t>(offset);\n    return desc.desc_;\n}\n", return_type="uint64"), T.uint32(138478736), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), ki != 0 or T.Cast("bool", o_accumulate), "mma", "cta_group::2", "kind::f16", "p12")
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
                                        s_ptr: T.let[T.handle] = T.cuda.func_call("tvm_builtin_pointer_offset", T.address_of(s_smem_gemm[0, 0]), s_base + ds, source_code="\ntemplate <typename T>\n__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {\n    return ptr + offset;\n}\n", return_type=T.handle().ty)
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

    from tirx_kernels.flashmla._flashmla_bench import flashmla_reference_builder
    from tirx_kernels.flashmla._trtllm_gen_bench import (
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
