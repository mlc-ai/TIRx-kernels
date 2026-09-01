# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import math
from dataclasses import dataclass, fields
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

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

LAUNCH_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
    "tirx.use_dyn_shared_memory",
)


def _add_smem_desc_offset(dst, desc, offset):
    # Descriptor offsets wrap in the low 32 bits without carrying into the
    # encoded layout fields in the high half.
    desc_lo = K.alloc_local((1,), "uint32")
    desc_hi = K.alloc_local((1,), "uint32")
    K.ptx.mov.b64(desc_lo[0], desc_hi[0], desc)
    K.ptx.add.u32(desc_lo[0], desc_lo[0], K.cast(offset, "uint32"))
    K.ptx.mov.b64(dst, desc_lo[0], desc_hi[0])


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


# The dispatcher-selected SM100 form remains explicit PTX, with K owning
# entry structure, storage, barriers, TMEM bookkeeping, and warp roles.
@cache
def make_kernel(
    s_q,
    s_kv,
    topk,
    stride_kv_s_kv,
    stride_indices_s_q,
    have_attn_sink,
    have_topk_length,
    sm_scale_div_log2,
):
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

        kv_tma = encode(
            K.handle_add_byte_offset(kv.data, 0),
            2,
            D_QK,
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
        # Both output descriptors encode the same tiling; the kernel keeps two so the
        # steady-state and drain stores address separate tensormaps.
        out_shape = (
            64,
            B_H,
            D_V // 64,
            s_q,
            D_V * BF16_BYTES,
            64 * BF16_BYTES,
            B_H * D_V * BF16_BYTES,
            64,
            B_H // 2,
            D_V // 64,
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
        out_tma = encode(K.handle_add_byte_offset(out.data, 0), 4, *out_shape)
        out_tma_1 = encode(K.handle_add_byte_offset(out.data, 0), 4, *out_shape)
        q_tma = encode(
            K.handle_add_byte_offset(q.data, 0),
            5,
            64,
            B_H,
            2,
            4,
            s_q,
            D_QK * BF16_BYTES,
            256 * BF16_BYTES,
            64 * BF16_BYTES,
            B_H * D_QK * BF16_BYTES,
            64,
            B_H // 2,
            2,
            4,
            1,
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
        return kv_tma, out_tma, out_tma_1, q_tma

    @K.kernel(
        warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=2 * s_q, host_prelude=host_prelude
    )
    def sparse_flashmla_prefill_head128_small_topk_phase1_kern(
        q: K.gptr[K.bf16, (s_q, B_H, D_QK)],
        kv: K.gptr[K.bf16, (s_kv * stride_kv_s_kv,)],
        indices: K.gptr[K.i32, (s_q * stride_indices_s_q,)],
        attn_sink: K.gptr[K.f32, (B_H,)],
        topk_length: K.gptr[K.i32, (s_q,)],
        out: K.gptr[K.bf16, (s_q, B_H, D_V)],
        max_logits: K.gptr[K.f32, (s_q, B_H)],
        lse: K.gptr[K.f32, (s_q, B_H)],
        *,
        host,
    ):
        kv_tma_tensormap, out_tensormap, out_tensormap_1, q_tma_tensormap = host
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
        prefetch(out_tensormap_1)
        prefetch(out_tensormap)
        prefetch(kv_tma_tensormap)

        def iket_range(name):
            token = K.alloc_local((1,), "uint32")
            K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        smem = K.smem_pool()
        pool = smem.pool
        q_smem = smem.alloc((64, 512), "bfloat16", swizzle=K.SW128B).buf
        k_smem = smem.alloc((256, 256), "bfloat16", swizzle=K.SW128B).buf
        s_smem_gemm = smem.alloc((64, 64), "bfloat16", align=1024)
        p_exchange = pool.alloc((4, 1024), "uint32")
        rowwise_max_buf = pool.alloc((128,), "float32")
        rowwise_li_buf = pool.alloc((128,), "float32")
        is_k_valid = pool.alloc((4, 8), "int8", align=16)

        q_full = K.TMABar(pool, 1)
        tq_ready = K.TCGen05Bar(pool, 1)
        q_consumed = K.MBarrier(pool, 1)
        q_released = K.MBarrier(pool, 1)
        t_out_empty = K.MBarrier(pool, 1)
        k_ready = K.TMABar(pool, 4)
        k_empty = K.TCGen05Bar(pool, 4)
        p_empty = K.MBarrier(pool, 1)
        umma_ready = K.MBarrier(pool, 1)
        softmax_ready = K.MBarrier(pool, 1)
        so_full = K.MBarrier(pool, 1)
        li_full = K.MBarrier(pool, 1)
        li_empty = K.MBarrier(pool, 1)
        valid_full = K.MBarrier(pool, 4)
        valid_empty = K.MBarrier(pool, 4)
        clc_response_ready = K.TMABar(pool, 1)
        clc_empty = K.MBarrier(pool, 1)
        tq_consumed = K.MBarrier(pool, 1)

        clc_response = pool.alloc((4,), "uint32", align=16)
        tmem_start_addr = pool.alloc((1,), "uint32", align=4)
        assert pool.offset == 222500
        smem.commit()

        buffer_11 = k_smem.view(4, 64, 4, 64)
        buffer_12 = K.decl_buffer(
            (4, 64, 4, 64),
            "bfloat16",
            data=buffer_11.data,
            elem_offset=32768,
            scope="shared.dyn",
            align=1024,
        )
        k_smem_gemm = buffer_12.view(4, 64, 256)

        class CLCJobScheduler:
            """One role-local walk over the shared CLC job stream."""

            def __init__(self):
                self.valid = K.local_scalar("int32")
                self.block_idx = K.local_scalar("int32")
                self.epoch = K.PipelineState(1, phase=0)
                self.init()

            def init(self):
                K.assign(self.valid, 1)
                K.assign(self.block_idx, block_idx)

            def issue_cancel(self):
                with K.If(cta_idx == 0), K.Then():
                    K.cuda.mbarrier_wait(
                        K.address_of(clc_empty.buf[0]), K.bitwise_xor(self.epoch.phase, 1)
                    )
                    K.ptx[
                        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                    ](
                        K.cuda.cvta_generic_to_shared(K.address_of(clc_response[0])),
                        K.cuda.cvta_generic_to_shared(K.address_of(clc_response_ready.buf[0])),
                    )
                K.ptx["mbarrier.arrive.expect_tx.shared.b64"](
                    K.cuda.cvta_generic_to_shared(K.address_of(clc_response_ready.buf[0])),
                    K.uint32(16),
                )

            def advance(self):
                K.cuda.mbarrier_wait(K.address_of(clc_response_ready.buf[0]), self.epoch.phase)
                next_job = K.local_scalar("uint32")
                K.query_cancel_first_ctaid_x(next_job, K.address_of(clc_response[0]))
                remote_empty = K.local_scalar("uint64")
                K.ptx["mapa.shared::cluster.u64"](
                    remote_empty, K.address_of(clc_empty.buf[0]), K.uint32(0)
                )
                K.ptx["mbarrier.arrive.b64"](
                    K.reinterpret(K.handle().ty, remote_empty), K.uint32(1), pred=K.bool(True)
                )
                with K.If(next_job == K.uint32(4294967295)):
                    with K.Then():
                        K.assign(self.valid, 0)
                    with K.Else():
                        K.assign(self.block_idx, K.Cast("int32", next_job))
                self.epoch.advance()

        def initialize_protocol():
            with K.If(warp_idx == 1):
                with K.Then():
                    with K.If(K.cuda.elect_sync()):
                        with K.Then():
                            for init_bar, arrive_count in (
                                (q_full, 1),
                                (tq_ready, 1),
                                (q_consumed, 1),
                                (q_released, 1),
                                (t_out_empty, 256),
                                (p_empty, 256),
                                (umma_ready, 1),
                                (softmax_ready, 1),
                                (so_full, 256),
                                (li_full, 64),
                                (li_empty, 128),
                                (clc_response_ready, 1),
                                (clc_empty, 539),
                                # One elected arrival from each WG0 warp in both CTAs.
                                (tq_consumed, 8),
                            ):
                                with K.unroll(1) as i:
                                    K.ptx["mbarrier.init.shared.b64"](
                                        K.cuda.cvta_generic_to_shared(
                                            K.address_of(init_bar.buf[i])
                                        ),
                                        K.uint32(arrive_count),
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
                                    with K.unroll(4) as init_stage:
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(
                                                K.address_of(k_ready.buf[init_stage])
                                            ),
                                            K.uint32(1),
                                        )
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(
                                                K.address_of(k_empty.buf[init_stage])
                                            ),
                                            K.uint32(1),
                                        )
                                    with K.unroll(4) as init_stage:
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(
                                                K.address_of(valid_full.buf[init_stage])
                                            ),
                                            K.uint32(8),
                                        )
                                        K.ptx["mbarrier.init.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(
                                                K.address_of(valid_empty.buf[init_stage])
                                            ),
                                            K.uint32(128),
                                        )
                                    K.ptx["fence.mbarrier_init.release.cluster"]()
            K.cuda.cluster_sync()

        initialize_protocol()

        def store_output(
            output_epoch,
            q_epoch,
            tensor_map,
            s_q_idx,
            signal_q_consumed: K.constexpr,
            launch_dependents: K.constexpr,
        ):
            K.cuda.mbarrier_wait(K.address_of(li_full.buf[0]), output_epoch)
            output_scale = K.local_scalar("float32")
            K.ptx.ld.shared.f32(output_scale, rowwise_li_buf.ptr_to([idx_in_warpgroup % 64]))
            K.ptx["mbarrier.arrive.shared.b64"](
                K.cuda.cvta_generic_to_shared(K.address_of(li_empty.buf[0])), K.uint32(1)
            )
            K.cuda.mbarrier_wait(K.address_of(q_released.buf[0]), output_epoch)
            if launch_dependents:
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx["griddepcontrol.launch_dependents"]()

            output_storage = K.alloc_local((64,))
            bf16_storage = K.alloc_local((32,), "uint32")
            q_smem_win = q_smem.view(64, 2, 256).view(2, 64, 256).view(128, 256)
            with K.unroll(4) as epi_k:
                local_storage = output_storage
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"](
                    *[local_storage[i] for i in range(64)],
                    K.cuda.get_tmem_addr(K.uint32(0), 0, epi_k * 64),
                )
                K.ptx["tcgen05.wait::ld.sync.aligned"]()
                with K.If(epi_k == 0), K.Then():
                    K.cuda.mbarrier_wait(K.address_of(q_consumed.buf[0]), q_epoch)
                    K.ptx["fence.proxy.async.shared::cta"]()
                    if signal_q_consumed:
                        with K.If(K.cuda.elect_sync()), K.Then():
                            tq_consumed_remote = K.local_scalar("uint32")
                            K.ptx["mapa.shared::cluster.u32"](
                                tq_consumed_remote,
                                K.cuda.cvta_generic_to_shared(K.address_of(tq_consumed.buf[0])),
                                K.uint32(0),
                            )
                            K.ptx["mbarrier.arrive.shared::cluster.b64"](tq_consumed_remote)
                with K.If(epi_k == 3), K.Then():
                    output_empty = K.local_scalar("uint32")
                    K.ptx["mapa.shared::cluster.u32"](
                        output_empty,
                        K.cuda.cvta_generic_to_shared(K.address_of(t_out_empty.buf[0])),
                        K.uint32(0),
                    )
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](output_empty)

                scaled = output_storage
                unscaled = output_storage
                for f in range(32):
                    packed_values = K.local_scalar("uint64")
                    packed_scale = K.local_scalar("uint64")
                    K.ptx.mov.b64(packed_values, unscaled[f * 2], unscaled[f * 2 + 1])
                    K.ptx.mov.b64(packed_scale, output_scale, output_scale)
                    K.ptx["mul.rz.ftz.f32x2"](packed_values, packed_values, packed_scale)
                    K.ptx.mov.b64(scaled[f * 2], scaled[f * 2 + 1], packed_values)

                bf16_words = bf16_storage
                scaled_for_convert = output_storage
                for f in range(32):
                    K.ptx.cvt.rn.bf16x2.f32(
                        bf16_words[f], scaled_for_convert[f * 2 + 1], scaled_for_convert[f * 2]
                    )

                r_words = bf16_storage
                for f in range(8):
                    s_off: K.int32 = (
                        idx_in_warpgroup // 64 * 16384
                        + epi_k % 4 * 4096
                        + idx_in_warpgroup % 64 * 64
                        + K.bitwise_xor(
                            f * 8,
                            K.shift_left(
                                K.bitwise_and(
                                    idx_in_warpgroup // 64 * 256
                                    + epi_k % 4 * 64
                                    + idx_in_warpgroup % 64,
                                    7,
                                ),
                                3,
                            ),
                        )
                    )
                    s_ptr = K.ptr_byte_offset(
                        K.address_of(q_smem_win[0, 0]), s_off * BF16_BYTES, "bfloat16"
                    )
                    r_w: K.int32 = f * 4
                    K.ptx["st.shared.v4.u32"](
                        K.cuda.cvta_generic_to_shared(s_ptr),
                        r_words[r_w],
                        r_words[r_w + 1],
                        r_words[r_w + 2],
                        r_words[r_w + 3],
                    )

            K.ptx["fence.proxy.async.shared::cta"]()
            K.ptx["bar.sync"](K.uint32(0), K.uint32(128))
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx["cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"](
                        K.reinterpret(K.handle().ty, K.address_of(tensor_map)),
                        0,
                        cta_idx * 64,
                        0,
                        s_q_idx,
                        K.cuda.cvta_generic_to_shared(q_smem.ptr_to([0, 0])),
                    )
                K.ptx["cp.async.bulk.commit_group"]()

        def q_load_output():
            q_o_token = iket_range("h128-small-q-load-output")
            jobs = CLCJobScheduler()
            last_valid = K.local_scalar("int32", init=0)
            last_s_q_idx = K.local_scalar("int32", init=0)
            with K.While(jobs.valid != 0):
                wg0_s_q_idx = jobs.block_idx // 2
                previous_epoch = K.bitwise_xor(jobs.epoch.phase, 1)
                with K.If(warp_idx == 0), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx["cp.async.bulk.wait_group"](0)
                        buffer_17 = K.local_scalar("uint64")
                        K.ptx["mapa.u64"](buffer_17, K.address_of(q_full.buf[0]), K.uint32(0))
                        K.ptx[
                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                        ](
                            K.cuda.cvta_generic_to_shared(q_smem.ptr_to([0, 0])),
                            K.reinterpret(K.handle().ty, K.address_of(q_tma_tensormap)),
                            0,
                            cta_idx * 64,
                            0,
                            0,
                            jobs.block_idx // 2,
                            K.cuda.cvta_generic_to_shared(K.reinterpret(K.handle().ty, buffer_17)),
                            K.uint64(1364590687093260288),
                        )
                        with K.If(cta_idx == 0), K.Then():
                            # Do not republish tQ-full until both CTAs consumed its prior phase.
                            with K.If(last_valid != 0), K.Then():
                                K.cuda.mbarrier_wait(
                                    K.address_of(tq_consumed.buf[0]), previous_epoch
                                )
                            K.ptx["mbarrier.arrive.expect_tx.shared.b64"](
                                K.cuda.cvta_generic_to_shared(K.address_of(q_full.buf[0])),
                                K.uint32(131072),
                            )
                            K.cuda.mbarrier_wait(K.address_of(q_full.buf[0]), jobs.epoch.phase)
                            K.cuda.mbarrier_wait(K.address_of(tq_ready.buf[0]), previous_epoch)
                            K.ptx["tcgen05.fence::after_thread_sync"]()
                            cp_desc = K.local_scalar("uint64")
                            K.cuda.tcgen05.encode_matrix_descriptor(
                                cp_desc.source.data,
                                K.reinterpret(K.handle().ty, K.uint64(0)),
                                1,
                                64,
                                3,
                            )
                            with K.unroll(16) as flat:
                                K.ptx["tcgen05.cp.cta_group::2.128x256b"](
                                    K.Cast("uint32", 256 + (flat % 4 * 32 + flat // 4 % 4 * 8)),
                                    K.bitwise_or(
                                        K.bitwise_and(cp_desc, K.bitwise_not(K.uint64(16383))),
                                        K.Cast(
                                            "uint64",
                                            K.bitwise_and(
                                                K.shift_right(
                                                    K.cuda.cvta_generic_to_shared(
                                                        K.ptr_byte_offset(
                                                            q_smem.ptr_to([0, 0]),
                                                            (flat % 4 * 1024 + flat // 4 % 4 * 2)
                                                            * 16,
                                                            K.type_annotation("bfloat16"),
                                                        )
                                                    ),
                                                    K.uint32(4),
                                                ),
                                                K.uint32(16383),
                                            ),
                                        ),
                                    ),
                                )
                            K.ptx[
                                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                            ](
                                K.cuda.cvta_generic_to_shared(K.address_of(q_consumed.buf[0])),
                                K.Cast("uint16", 3),
                            )
                with K.If(last_valid != 0):
                    with K.Then():
                        store_output(
                            previous_epoch,
                            jobs.epoch.phase,
                            out_tensormap_1,
                            last_s_q_idx,
                            signal_q_consumed=True,
                            launch_dependents=False,
                        )
                    with K.Else():
                        K.cuda.mbarrier_wait(K.address_of(q_consumed.buf[0]), jobs.epoch.phase)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            buffer_13 = K.local_scalar("uint32")
                            K.ptx["mapa.shared::cluster.u32"](
                                buffer_13,
                                K.cuda.cvta_generic_to_shared(K.address_of(tq_consumed.buf[0])),
                                K.uint32(0),
                            )
                            K.ptx["mbarrier.arrive.shared::cluster.b64"](buffer_13)
                K.assign(last_valid, 1)
                K.assign(last_s_q_idx, wg0_s_q_idx)
                jobs.advance()
            with K.If(last_valid != 0), K.Then():
                last_epoch = K.bitwise_xor(jobs.epoch.phase, 1)
                with K.If(warp_idx == 0), K.Then():
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx["cp.async.bulk.wait_group"](0)
                K.ptx["bar.sync"](K.uint32(0), K.uint32(128))
                store_output(
                    last_epoch,
                    last_epoch,
                    out_tensormap,
                    last_s_q_idx,
                    signal_q_consumed=False,
                    launch_dependents=True,
                )
            with K.If(warp_idx == 0), K.Then():
                K.ptx["tcgen05.dealloc.cta_group::2.sync.aligned.b32"](K.uint32(0), K.uint32(512))
            K.cuda.iket.range_end(q_o_token[0])

        def kv_gather():
            kv_gather_token = iket_range("h128-small-kv-load")
            wg1_warp_idx = thread_idx // 32 - 4
            with K.If(K.cuda.elect_sync()), K.Then():
                jobs = CLCJobScheduler()
                k_pipe = K.PipelineState(4, phase=0)
                with K.While(jobs.valid != 0):
                    wg1_s_q_idx = jobs.block_idx // 2
                    wg1_topk_len = K.local_scalar("int32", init=topk)
                    with K.If(have_topk_length), K.Then():
                        K.ptx.ld.global_.s32(wg1_topk_len, topk_length.ptr_to([wg1_s_q_idx]))
                    wg1_num_k_blocks = K.max((wg1_topk_len + 64 - 1) // 64, 1)
                    wg1_g_indices_base = wg1_s_q_idx * stride_indices_s_q
                    with K.serial(wg1_num_k_blocks, unroll=False) as k:
                        cur_indices = K.alloc_local((16,), "int32")
                        with K.unroll(2) as local_row:
                            row = local_row * 32 + wg1_warp_idx * 8
                            row_base = wg1_g_indices_base + k * 64 + row
                            buffer_13 = K.decl_buffer(
                                (16,), "int32", data=cur_indices.data, scope="local"
                            )
                            buffer_14 = buffer_13.view("uint32")
                            K.ptx["ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.u32"](
                                *[buffer_14[local_row * 8 + i] for i in range(8)],
                                K.address_of(indices[row_base]),
                            )
                        K.cuda.mbarrier_wait(
                            K.address_of(k_empty.buf[k_pipe.stage]), K.bitwise_xor(k_pipe.phase, 1)
                        )
                        src_col = cta_idx * 256
                        with K.unroll(4) as row_group:
                            with K.unroll(4) as col_atom:
                                buffer_16 = K.local_scalar("uint64")
                                K.ptx["mapa.u64"](
                                    buffer_16, K.address_of(k_ready.buf[k_pipe.stage]), K.uint32(0)
                                )
                                kv_dst_offset = (
                                    k_pipe.stage * 16384
                                    + wg1_warp_idx * 512
                                    + row_group // 2 * 2048
                                    + row_group % 2 * 256
                                    + col_atom * 4096
                                ) * BF16_BYTES
                                K.ptx[
                                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                ](
                                    K.cuda.cvta_generic_to_shared(
                                        K.ptr_byte_offset(
                                            K.address_of(k_smem[0, 0]),
                                            kv_dst_offset,
                                            K.type_annotation("bfloat16"),
                                        )
                                    ),
                                    K.reinterpret(K.handle().ty, K.address_of(kv_tma_tensormap)),
                                    src_col + col_atom * 64,
                                    cur_indices[row_group * 4],
                                    cur_indices[row_group * 4 + 1],
                                    cur_indices[row_group * 4 + 2],
                                    cur_indices[row_group * 4 + 3],
                                    K.cuda.cvta_generic_to_shared(
                                        K.reinterpret(K.handle().ty, buffer_16)
                                    ),
                                    K.uint64(1508705875169116160),
                                )
                        k_pipe.advance()
                    jobs.advance()
            K.cuda.iket.range_end(kv_gather_token[0])

        def producer_role(role: K.constexpr):
            with K.If(role == 0):
                with K.Then():
                    mma_token = iket_range("h128-small-qk-pv-issue")
                    with K.If(K.cuda.elect_sync()), K.Then():
                        jobs = CLCJobScheduler()
                        k_pipe = K.PipelineState(4, phase=0)
                        with K.While(jobs.valid != 0):
                            umma_s_q_idx = jobs.block_idx // 2
                            umma_topk_len = K.local_scalar("int32", init=topk)
                            with K.If(have_topk_length), K.Then():
                                K.ptx.ld.global_.s32(
                                    umma_topk_len, topk_length.ptr_to([umma_s_q_idx])
                                )
                            umma_num_k_blocks = K.max((umma_topk_len + 64 - 1) // 64, 1)
                            K.cuda.mbarrier_wait(K.address_of(q_consumed.buf[0]), jobs.epoch.phase)
                            with K.serial(umma_num_k_blocks + 1, unroll=False) as k:
                                with K.If(k < umma_num_k_blocks), K.Then():
                                    K.cuda.mbarrier_wait(
                                        K.address_of(p_empty.buf[0]),
                                        K.bitwise_xor(K.bitwise_and(k_pipe.stage, 1), 1),
                                    )
                                    K.ptx["mbarrier.arrive.expect_tx.shared.b64"](
                                        K.cuda.cvta_generic_to_shared(
                                            K.address_of(k_ready.buf[k_pipe.stage])
                                        ),
                                        K.uint32(65536),
                                    )
                                    K.cuda.mbarrier_wait(
                                        K.address_of(k_ready.buf[k_pipe.stage]), k_pipe.phase
                                    )
                                    K.ptx["tcgen05.fence::after_thread_sync"]()
                                    qk_accumulate = K.local_scalar("uint32", init=K.uint32(0))
                                    descB_local = K.local_scalar("uint64")
                                    K.cuda.tcgen05.encode_matrix_descriptor(
                                        K.address_of(descB_local),
                                        K.address_of(k_smem_gemm[0, 0, 0]),
                                        512,
                                        64,
                                        3,
                                    )
                                    with K.unroll(1) as mi:
                                        with K.unroll(1) as ni:
                                            with K.unroll(16) as ki:
                                                descB_off = K.local_scalar("uint64")
                                                _add_smem_desc_offset(
                                                    descB_off,
                                                    descB_local,
                                                    (
                                                        ki // 1024 * 16384
                                                        + ni * 16384
                                                        + k_pipe.stage * 16384
                                                        + ki % 16 // 4 * 4096
                                                        + ki % 1024 // 16 * 64
                                                        + ki % 4 * 16
                                                    )
                                                    // 8,
                                                )
                                                K.ptx["tcgen05.mma.cta_group::2.kind::f16"](
                                                    K.Cast("uint32", ni * 64 + 384),
                                                    K.Cast("uint32", ki * 8 + 256),
                                                    descB_off,
                                                    K.uint32(136316048),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.uint32(0),
                                                    K.Or(ki != 0, K.Cast("bool", qk_accumulate)),
                                                )
                                    K.assign(qk_accumulate, K.uint32(1))
                                    K.ptx[
                                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                                    ](
                                        K.cuda.cvta_generic_to_shared(
                                            K.address_of(umma_ready.buf[0])
                                        ),
                                        K.Cast("uint16", 3),
                                    )
                                    with K.If(k == umma_num_k_blocks - 1), K.Then():
                                        K.ptx[
                                            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64"
                                        ](
                                            K.cuda.cvta_generic_to_shared(
                                                K.address_of(tq_ready.buf[0])
                                            )
                                        )
                                with K.If(k > 0), K.Then():
                                    prev_k = k - 1
                                    prev_buf = (k_pipe.stage + 3) % 4
                                    prev_s_o_phase = K.bitwise_and(prev_buf, 1)
                                    K.cuda.mbarrier_wait(
                                        K.address_of(so_full.buf[0]),
                                        K.bitwise_xor(prev_s_o_phase, 0),
                                    )
                                    with K.If(prev_k == 0), K.Then():
                                        K.cuda.mbarrier_wait(
                                            K.address_of(t_out_empty.buf[0]),
                                            K.bitwise_xor(jobs.epoch.phase, 1),
                                        )
                                    K.ptx["tcgen05.fence::after_thread_sync"]()
                                    o_accumulate = K.local_scalar(
                                        "uint32",
                                        init=K.if_then_else(prev_k == 0, K.uint32(0), K.uint32(1)),
                                    )
                                    # The two PV MMA issues differ only in the N-half they accumulate
                                    # into and the matching 8KB descB offset.
                                    for n_half, b_half in ((0, 0), (128, 8192)):
                                        descB_local = K.local_scalar("uint64")
                                        K.cuda.tcgen05.encode_matrix_descriptor(
                                            K.address_of(descB_local),
                                            K.address_of(k_smem_gemm[0, 0, 0]),
                                            512,
                                            64,
                                            3,
                                        )
                                        descA_local = K.local_scalar("uint64")
                                        K.cuda.tcgen05.encode_matrix_descriptor(
                                            K.address_of(descA_local),
                                            K.address_of(s_smem_gemm[0, 0]),
                                            64,
                                            8,
                                            0,
                                        )
                                        with K.unroll(1) as mi:
                                            with K.unroll(1) as ni:
                                                with K.unroll(4) as ki:
                                                    # `if` guards keep the first half free of `+ 0` terms.
                                                    n_acc = ni * 128
                                                    if n_half:
                                                        n_acc = n_acc + n_half
                                                    b_off = (
                                                        (ki * 16 + ni) // 64 * 16384
                                                        + prev_buf * 16384
                                                        + (ki * 16 + ni) % 64 * 64
                                                    )
                                                    if b_half:
                                                        b_off = b_off + b_half
                                                    descA_off = K.local_scalar("uint64")
                                                    _add_smem_desc_offset(
                                                        descA_off,
                                                        descA_local,
                                                        (ki % 4 * 1024 + mi * 512 + ki // 4 * 8)
                                                        // 8,
                                                    )
                                                    descB_off = K.local_scalar("uint64")
                                                    _add_smem_desc_offset(
                                                        descB_off, descB_local, b_off // 8
                                                    )
                                                    K.ptx["tcgen05.mma.cta_group::2.kind::f16"](
                                                        K.Cast("uint32", n_acc),
                                                        descA_off,
                                                        descB_off,
                                                        K.uint32(138478736),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.uint32(0),
                                                        K.Or(ki != 0, K.Cast("bool", o_accumulate)),
                                                    )
                                    K.assign(o_accumulate, K.uint32(1))
                                    K.ptx[
                                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                                    ](
                                        K.cuda.cvta_generic_to_shared(
                                            K.address_of(softmax_ready.buf[0])
                                        ),
                                        K.Cast("uint16", 3),
                                    )
                                    K.ptx[
                                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                                    ](
                                        K.cuda.cvta_generic_to_shared(
                                            K.address_of(k_empty.buf[prev_buf])
                                        ),
                                        K.Cast("uint16", 3),
                                    )
                                with K.If(k != umma_num_k_blocks), K.Then():
                                    k_pipe.advance()
                            K.ptx["tcgen05.fence::before_thread_sync"]()
                            K.ptx[
                                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                            ](
                                K.cuda.cvta_generic_to_shared(K.address_of(q_released.buf[0])),
                                K.Cast("uint16", 3),
                            )
                            jobs.advance()
                    K.cuda.iket.range_end(mma_token[0])
                with K.Else():
                    with K.If(role == 1):
                        with K.Then():
                            valid_mask_token = iket_range("h128-small-valid-mask")
                            with K.If(lane_idx < 8), K.Then():
                                lane_indices = K.alloc_local((8,), "int32")
                                jobs = CLCJobScheduler()
                                index_pipe = K.PipelineState(4, phase=0)
                                with K.While(jobs.valid != 0):
                                    valid_s_q_idx = jobs.block_idx // 2
                                    valid_topk_len = K.local_scalar("int32", init=topk)
                                    with K.If(have_topk_length), K.Then():
                                        K.ptx.ld.global_.s32(
                                            valid_topk_len, topk_length.ptr_to([valid_s_q_idx])
                                        )
                                    valid_num_k_blocks = K.max((valid_topk_len + 64 - 1) // 64, 1)
                                    valid_g_indices_base = valid_s_q_idx * stride_indices_s_q
                                    with K.serial(valid_num_k_blocks, unroll=False) as k:
                                        row_base = valid_g_indices_base + k * 64 + lane_idx * 8
                                        buffer_13 = K.decl_buffer(
                                            (8,), "int32", data=lane_indices.data, scope="local"
                                        )
                                        buffer_14 = buffer_13.view("uint32")
                                        K.ptx[
                                            "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.u32"
                                        ](
                                            *[buffer_14[i] for i in range(8)],
                                            K.address_of(indices[row_base]),
                                        )
                                        abs_pos_start = k * 64

                                        def valid_bit(j, bit):
                                            # j == 0 keeps the bare base so the emitted expression matches the
                                            # other seven lanes' `base + j` form without a folded `+ 0`.
                                            pos = abs_pos_start + lane_idx * 8
                                            if j:
                                                pos = pos + j
                                            return K.Select(
                                                K.bitwise_and(
                                                    K.bitwise_and(
                                                        lane_indices[j] >= 0, lane_indices[j] < s_kv
                                                    ),
                                                    pos < valid_topk_len,
                                                ),
                                                bit,
                                                0,
                                            )

                                        # Balanced pairwise reduction: reproduces the hand-written or-tree shape.
                                        terms = [valid_bit(j, 1 << j) for j in range(8)]
                                        while len(terms) > 1:
                                            terms = [
                                                K.bitwise_or(terms[j], terms[j + 1])
                                                for j in range(0, len(terms), 2)
                                            ]
                                        mask = K.Cast("int8", terms[0])
                                        K.cuda.mbarrier_wait(
                                            K.address_of(valid_empty.buf[index_pipe.stage]),
                                            K.bitwise_xor(index_pipe.phase, 1),
                                        )
                                        K.ptx.st.shared.b8(
                                            is_k_valid.ptr_to([index_pipe.stage, lane_idx]),
                                            K.reinterpret("uint8", mask),
                                        )
                                        K.ptx["mbarrier.arrive.shared.b64"](
                                            K.cuda.cvta_generic_to_shared(
                                                K.address_of(valid_full.buf[index_pipe.stage])
                                            ),
                                            K.uint32(1),
                                        )
                                        index_pipe.advance()
                                    jobs.advance()
                            K.cuda.iket.range_end(valid_mask_token[0])
                        with K.Else():
                            with K.If(role >= 2), K.Then():
                                clc_token = K.alloc_local((1,), "uint32")
                                K.assign(clc_token[0], K.cuda.iket.sentinel_token("h128-small-clc"))
                                with K.If(role == 2), K.Then():
                                    K.assign(
                                        clc_token[0], K.cuda.iket.range_start("h128-small-clc")
                                    )
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    with K.If(role == 2), K.Then():
                                        jobs = CLCJobScheduler()
                                        with K.While(jobs.valid != 0):
                                            jobs.issue_cancel()
                                            jobs.advance()
                                K.cuda.iket.range_end(clc_token[0])

        def softmax():
            softmax_token = iket_range("h128-small-softmax")
            local_warp_idx = warp_idx - 12
            jobs = CLCJobScheduler()
            index_pipe = K.PipelineState(4, phase=0)
            with K.While(jobs.valid != 0):
                wg3_s_q_idx = jobs.block_idx // 2
                wg3_topk_len = K.local_scalar("int32", init=topk)
                with K.If(have_topk_length), K.Then():
                    K.ptx.ld.global_.s32(wg3_topk_len, topk_length.ptr_to([wg3_s_q_idx]))
                wg3_num_k_blocks = K.max((wg3_topk_len + 64 - 1) // 64, 1)
                mi = K.local_scalar("float32", init=K.float32(-1000000000000000019884624838656.0))
                li = K.local_scalar("float32", init=K.float32(0.0))
                real_mi = K.local_scalar("float32", init=K.float32("-inf"))
                scale_pair = K.local_scalar(
                    "uint64",
                    init=K.cuda.make_float2(
                        K.float32(sm_scale_div_log2), K.float32(sm_scale_div_log2)
                    ),
                )
                with K.serial(wg3_num_k_blocks, unroll=False) as k:
                    K.cuda.mbarrier_wait(
                        K.address_of(valid_full.buf[index_pipe.stage]), index_pipe.phase
                    )
                    p = K.alloc_local((32,), "uint32")
                    p_peer = K.alloc_local((32,), "uint32")
                    K.cuda.mbarrier_wait(
                        K.address_of(umma_ready.buf[0]), K.bitwise_and(index_pipe.stage, 1)
                    )
                    K.ptx["tcgen05.fence::after_thread_sync"]()
                    with K.If(local_warp_idx < 2):
                        with K.Then():
                            local_storage = p
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[local_storage[i] for i in range(32)], K.uint32(384)
                            )
                            local_storage_1 = p_peer
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[local_storage_1[i] for i in range(32)],
                                K.cuda.get_tmem_addr(K.uint32(384), 0, 32),
                            )
                        with K.Else():
                            local_storage = p_peer
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[local_storage[i] for i in range(32)], K.uint32(384)
                            )
                            local_storage_1 = p
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[local_storage_1[i] for i in range(32)],
                                K.cuda.get_tmem_addr(K.uint32(384), 0, 32),
                            )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx["tcgen05.fence::before_thread_sync"]()
                    buffer_17 = K.local_scalar("uint32")
                    K.ptx["mapa.shared::cluster.u32"](
                        buffer_17,
                        K.cuda.cvta_generic_to_shared(K.address_of(p_empty.buf[0])),
                        K.uint32(0),
                    )
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](buffer_17)
                    valid_word_offset = K.if_then_else(local_warp_idx >= 2, 1, 0)
                    buffer_18 = K.decl_buffer(
                        (4, 2),
                        "uint32",
                        data=is_k_valid.data,
                        elem_offset=55552,
                        scope="shared.dyn",
                        align=16,
                    )
                    is_k_valid_u32 = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(
                        is_k_valid_u32, buffer_18.ptr_to([index_pipe.stage, valid_word_offset])
                    )
                    with K.unroll(32) as p_i:
                        invalid_p_predicate = K.bitwise_and(
                            K.shift_right(is_k_valid_u32, K.Cast("uint32", p_i)), K.uint32(1)
                        ) == K.uint32(0)
                        K.ptx.mov.b32(
                            p[p_i],
                            K.if_then_else(invalid_p_predicate, K.uint32(4286578688), p[p_i]),
                        )
                    sum_pair0 = K.local_scalar("uint64")
                    sum_pair1 = K.local_scalar("uint64")
                    with K.unroll(8) as exchange_i:
                        exchange_offset: K.int32 = exchange_i * 32 * 4 + lane_idx * 4
                        p_peer_offset = exchange_i * 4
                        K.ptx["st.shared.v4.u32"](
                            K.cuda.cvta_generic_to_shared(
                                K.address_of(
                                    p_exchange[K.bitwise_xor(local_warp_idx, 2), exchange_offset]
                                )
                            ),
                            p_peer[p_peer_offset],
                            p_peer[p_peer_offset + 1],
                            p_peer[p_peer_offset + 2],
                            p_peer[p_peer_offset + 3],
                        )
                    K.ptx["bar.sync"](
                        K.Cast("uint32", 2 + K.bitwise_and(local_warp_idx, 1)), K.uint32(64)
                    )
                    with K.unroll(8) as exchange_i:
                        exchange_offset: K.int32 = exchange_i * 32 * 4 + lane_idx * 4
                        p_exchange_tmp = K.alloc_local((4,), "uint32")
                        K.ptx["ld.shared.v4.u32"](
                            p_exchange_tmp[0],
                            p_exchange_tmp[1],
                            p_exchange_tmp[2],
                            p_exchange_tmp[3],
                            K.cuda.cvta_generic_to_shared(
                                K.address_of(p_exchange[local_warp_idx, exchange_offset])
                            ),
                        )
                        p_pair0 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p[exchange_i * 4]),
                            K.cuda.uint_as_float(p[exchange_i * 4 + 1]),
                        )
                        peer_pair0 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p_exchange_tmp[0]),
                            K.cuda.uint_as_float(p_exchange_tmp[1]),
                        )
                        K.ptx["add.rn.f32x2"](sum_pair0, p_pair0, peer_pair0)
                        K.ptx.mov.b32(
                            p[exchange_i * 4], K.cuda.float_as_uint(K.cuda.float2_x(sum_pair0))
                        )
                        K.ptx.mov.b32(
                            p[exchange_i * 4 + 1], K.cuda.float_as_uint(K.cuda.float2_y(sum_pair0))
                        )
                        p_pair1 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p[exchange_i * 4 + 2]),
                            K.cuda.uint_as_float(p[exchange_i * 4 + 3]),
                        )
                        peer_pair1 = K.cuda.make_float2(
                            K.cuda.uint_as_float(p_exchange_tmp[2]),
                            K.cuda.uint_as_float(p_exchange_tmp[3]),
                        )
                        K.ptx["add.rn.f32x2"](sum_pair1, p_pair1, peer_pair1)
                        K.ptx.mov.b32(
                            p[exchange_i * 4 + 2], K.cuda.float_as_uint(K.cuda.float2_x(sum_pair1))
                        )
                        K.ptx.mov.b32(
                            p[exchange_i * 4 + 3], K.cuda.float_as_uint(K.cuda.float2_y(sum_pair1))
                        )
                    cur_pi_max = K.local_scalar("float32", init=K.float32("-inf"))
                    with K.unroll(32) as p_i:
                        K.assign(cur_pi_max, K.max(cur_pi_max, K.cuda.uint_as_float(p[p_i])))
                    K.assign(cur_pi_max, cur_pi_max * K.float32(sm_scale_div_log2))
                    K.ptx.st.shared.f32(rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max)
                    K.ptx["bar.sync"](
                        K.Cast("uint32", 2 + K.bitwise_and(local_warp_idx, 1)), K.uint32(64)
                    )
                    peer_pi_max = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(
                        peer_pi_max, rowwise_max_buf.ptr_to([K.bitwise_xor(idx_in_warpgroup, 64)])
                    )
                    K.assign(cur_pi_max, K.max(cur_pi_max, peer_pi_max))
                    K.assign(real_mi, K.max(real_mi, cur_pi_max))
                    should_scale_o = K.local_scalar("uint32")
                    K.ptx.vote_sync.any.pred(
                        should_scale_o, cur_pi_max - mi > K.float32(6.0), K.uint32(4294967295)
                    )
                    new_max = K.local_scalar("float32")
                    scale_for_old = K.local_scalar("float32")
                    with K.If(should_scale_o == K.uint32(0)):
                        with K.Then():
                            K.assign(scale_for_old, K.float32(1.0))
                            K.assign(new_max, mi)
                        with K.Else():
                            K.assign(new_max, K.max(cur_pi_max, mi))
                            K.ptx["ex2.approx.ftz.f32"](scale_for_old, mi - new_max)
                    K.assign(mi, new_max)
                    s_frag = K.alloc_local((32,), "bfloat16")
                    s_pack = s_frag.view("uint32")
                    cur_sum_pair = K.local_scalar(
                        "uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0))
                    )
                    neg_new_max_pair = K.local_scalar(
                        "uint64",
                        init=K.cuda.make_float2(
                            new_max * K.float32(-1.0), new_max * K.float32(-1.0)
                        ),
                    )
                    fma_pair = K.local_scalar("uint64")
                    with K.unroll(16) as s_i:
                        p_pair = K.cuda.make_float2(
                            K.cuda.uint_as_float(p[s_i * 2]), K.cuda.uint_as_float(p[s_i * 2 + 1])
                        )
                        K.ptx["fma.rn.f32x2"](fma_pair, p_pair, scale_pair, neg_new_max_pair)
                        s_x = K.local_scalar("float32")
                        s_y = K.local_scalar("float32")
                        K.ptx["ex2.approx.ftz.f32"](s_x, K.cuda.float2_x(fma_pair))
                        K.ptx["ex2.approx.ftz.f32"](s_y, K.cuda.float2_y(fma_pair))
                        s_pair = K.cuda.make_float2(s_x, s_y)
                        K.ptx["add.rn.f32x2"](cur_sum_pair, cur_sum_pair, s_pair)
                        K.ptx.mov.b32(s_pack[s_i], K.cuda.float22bfloat162_rn(s_x, s_y))
                    cur_sum = K.cuda.float2_x(cur_sum_pair) + K.cuda.float2_y(cur_sum_pair)
                    li_tmp = K.local_scalar("float32")
                    K.ptx["fma.rn.f32"](li_tmp, li, scale_for_old, cur_sum)
                    K.assign(li, li_tmp)
                    K.cuda.mbarrier_wait(
                        K.address_of(softmax_ready.buf[0]),
                        K.bitwise_xor(K.bitwise_and(index_pipe.stage, 1), 1),
                    )
                    K.ptx["fence.proxy.async.shared::cta"]()
                    s_base: K.int32 = idx_in_warpgroup // 64 * 2048 + idx_in_warpgroup % 64 * 8
                    r_words = s_frag.view("uint32")
                    for f in range(4):
                        ds: K.int32 = f % 4 * 512
                        dr: K.int32 = f % 4 * 8
                        s_ptr = K.ptr_byte_offset(
                            K.address_of(s_smem_gemm[0, 0]), (s_base + ds) * BF16_BYTES, "bfloat16"
                        )
                        r_w: K.int32 = dr // 2
                        K.ptx["st.shared.v4.u32"](
                            K.cuda.cvta_generic_to_shared(s_ptr),
                            r_words[r_w],
                            r_words[r_w + 1],
                            r_words[r_w + 2],
                            r_words[r_w + 3],
                        )
                    with K.If(K.bitwise_and(k > 0, should_scale_o != K.uint32(0))), K.Then():
                        K.ptx["tcgen05.fence::after_thread_sync"]()
                        o_rescale = K.alloc_local((32,), "float32")
                        with K.unroll(8) as chunk_idx:
                            K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[o_rescale[i] for i in range(32)],
                                K.cuda.get_tmem_addr(K.uint32(0), 0, chunk_idx * 32),
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
                                K.cuda.get_tmem_addr(K.uint32(0), 0, chunk_idx * 32),
                                *[o_rescale[i] for i in range(32)],
                            )
                            K.ptx["tcgen05.wait::st.sync.aligned"]()
                        K.ptx["tcgen05.fence::before_thread_sync"]()
                    K.ptx["fence.proxy.async.shared::cta"]()
                    buffer_20 = K.local_scalar("uint32")
                    K.ptx["mapa.shared::cluster.u32"](
                        buffer_20,
                        K.cuda.cvta_generic_to_shared(K.address_of(so_full.buf[0])),
                        K.uint32(0),
                    )
                    K.ptx["mbarrier.arrive.shared::cluster.b64"](buffer_20)
                    K.ptx["mbarrier.arrive.shared.b64"](
                        K.cuda.cvta_generic_to_shared(
                            K.address_of(valid_empty.buf[index_pipe.stage])
                        ),
                        K.uint32(1),
                    )
                    index_pipe.advance()
                with K.If(real_mi == K.float32("-inf")), K.Then():
                    K.assign(li, K.float32(0.0))
                    K.assign(mi, K.float32("-inf"))
                K.cuda.mbarrier_wait(
                    K.address_of(li_empty.buf[0]), K.bitwise_xor(jobs.epoch.phase, 1)
                )
                K.ptx.st.shared.f32(
                    rowwise_li_buf.ptr_to([K.bitwise_xor(idx_in_warpgroup, 64)]), li
                )
                K.ptx["bar.sync"](K.uint32(1), K.uint32(128))
                peer_li = K.local_scalar("float32")
                K.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup]))
                K.assign(li, li + peer_li)
                with K.If(idx_in_warpgroup < 64), K.Then():
                    head_idx = cta_idx * 64 + idx_in_warpgroup
                    attn_sink_value = K.local_scalar("float32", init=K.float32(-float("inf")))
                    with K.If(have_attn_sink), K.Then():
                        K.ptx.ld.global_.f32(attn_sink_value, attn_sink.ptr_to([head_idx]))
                    attn_sink_log2 = attn_sink_value * K.float32(1.4426950408889634)
                    sink_exp = K.local_scalar("float32")
                    K.ptx["ex2.approx.ftz.f32"](sink_exp, attn_sink_log2 - mi)
                    output_scale = K.local_scalar(
                        "float32", init=K.cuda.fdividef(K.float32(1.0), li + sink_exp)
                    )
                    K.ptx.st.shared.f32(
                        rowwise_li_buf.ptr_to([idx_in_warpgroup]),
                        K.if_then_else(li == K.float32(0.0), K.float32(0.0), output_scale),
                    )
                    K.ptx["mbarrier.arrive.shared.b64"](
                        K.cuda.cvta_generic_to_shared(K.address_of(li_full.buf[0])), K.uint32(1)
                    )
                    cur_lse = K.local_scalar("float32")
                    K.ptx["fma.rn.f32"](cur_lse, mi, K.float32(0.69314718055994529), K.log(li))
                    K.assign(
                        cur_lse,
                        K.if_then_else(cur_lse == K.float32("-inf"), K.float32("inf"), cur_lse),
                    )
                    K.ptx.st.global_.f32(
                        max_logits.ptr_to([wg3_s_q_idx, head_idx]),
                        real_mi * K.float32(0.69314718055994529),
                    )
                    K.ptx.st.global_.f32(lse.ptr_to([wg3_s_q_idx, head_idx]), cur_lse)
                jobs.advance()
            K.cuda.iket.range_end(softmax_token[0])

        roles = K.specialize(chain_dispatch=True)
        q_output = roles.role("q_load_output", warps=range(0, 4), regs=160)
        kv_load = roles.role("kv_gather", warps=range(4, 8), regs=80)
        producer = roles.warpgroup("qk_control", warps=range(8, 12), regs=80)
        mma = roles.role("qk_pv_issue", warps=[8], when=cta_idx == 0, group=producer)
        valid = roles.role("valid_mask", warps=[9], group=producer)
        clc = roles.role("clc", warps=[10], group=producer)
        idle = roles.role("idle", warps=[11], group=producer)
        scale = roles.role("softmax", warps=range(12, 16), regs=160)
        with q_output:
            q_load_output()
        with kv_load:
            kv_gather()
        with producer:
            with mma:
                producer_role(0)
            with valid:
                producer_role(1)
            with clc:
                producer_role(2)
            with idle:
                producer_role(3)
        with scale:
            softmax()

        K.cuda.cluster_sync()

    return sparse_flashmla_prefill_head128_small_topk_phase1_kern.func.with_attr(
        "global_symbol", KERNEL_META["name"]
    ).with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    return make_kernel(
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
    # Torch oracle retained by design: no library exposes phase-1's split
    # intermediates (out/max_logits/lse per split), so nothing upstream can
    # arbitrate them.
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
