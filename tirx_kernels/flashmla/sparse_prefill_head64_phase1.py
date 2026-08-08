# This file contains code ported from FlashMLA (https://github.com/deepseek-ai/FlashMLA),
# copyright (c) 2025 DeepSeek, licensed under the MIT License.
# See THIRD_PARTY_LICENSES.md for the full license text.

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla.utils._mask import pack_valid_mask8
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.lang.smem_desc import SmemDescriptor
from tvm.tirx.layout import S, TileLayout, laneid, wid_in_wg

B_H = 64
B_TOPK = 64
D_V = 512
NUM_BUFS = 3
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

IKET_EVENT_NAMES = (
    "h64-q-load",
    "h64-softmax-tile",
    "h64-output",
    "h64-kv-nope-load",
    "h64-qk-pv-issue",
    "h64-qk-wait",
    "h64-pv-wait",
    "h64-valid-mask",
    "h64-k-rope-load",
)

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

BAR_WG0_SYNC = 0
BAR_WG0_WARP02 = 1

BF16_BYTES = 2
Q_ROPE_DIM = 64
WG1_NUM_WARPS = 4
WG1_ROWS_PER_WARP = (B_TOPK // 4) // WG1_NUM_WARPS

_TMA_G2S_4D_CACHE = (
    "cp.async.bulk.tensor.4d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_GATHER4_2D_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
_TMA_S2G_2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
_CP_ASYNC_CG_16B = "cp.async.cg.shared.global.L2::128B"
_TCGEN_CP_128X256 = "tcgen05.cp.cta_group::1.128x256b"
_TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_MMA_WS_F16 = "tcgen05.mma.ws.cta_group::1.kind::f16"
_TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
_TMEM_ST_32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
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
    return T.ptx.cvt.rn.bf16x2.f32(
        dst_words[offset // 2], src[offset + 1], src[offset]
    )


def _replace_smem_desc_addr(desc, smem_ptr):
    start_addr = T.cast(
        T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)
        ),
        "uint64",
    )
    return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)


@T.inline
def mul_f32x2(values, idx, multiplier):
    packed: T.uint64
    rhs: T.uint64
    T.ptx.mov.b64(packed, values[idx], values[idx + 1])
    T.ptx.mov.b64(rhs, multiplier, multiplier)
    T.ptx.mul.rz.ftz.f32x2(packed, packed, rhs)
    T.ptx.mov.b64(values[idx], values[idx + 1], packed)


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead64Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int = 576
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head64 regular phase1 requires h_q == 64")
        if self.h_kv != 1:
            raise ValueError("head64 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 64")


# Cover the two upstream fwd/head64 phase1 instantiations:
# D_QK=512 and D_QK=576, h_q=64, topk=512 at the scoped s_kv values.
CONFIGS = [
    {
        "label": f"bench_dqk{d_qk}_hq64_s4096_kv{s_kv}_topk512",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 512,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 49152, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head64_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead64Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead64Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead64Config(**cfg_kwargs)
    cfg.validate()
    return cfg


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
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
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


def _ring_mod3(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_mod3 = T.uint32(0x10210210)
        shift = T.cast(value, "uint32") * T.uint32(4)
        return T.cast(T.bitwise_and(T.shift_right(packed_mod3, shift), T.uint32(0xF)), "int32")

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = value - max_offset
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = T.Select(value < offset, value - (offset - NUM_BUFS), result)
    return result


def _ring_phase_parity(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_phase = T.uint32(0x38)
        return T.cast(
            T.bitwise_and(T.shift_right(packed_phase, T.cast(value, "uint32")), T.uint32(1)),
            "int32",
        )

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = T.int32((max_offset // NUM_BUFS) & 1)
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = T.Select(value < offset, T.int32(((offset - NUM_BUFS) // NUM_BUFS) & 1), result)
    return result


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
    kv_part1_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        kv_part1_tensormap,
        "bfloat16",
        2,
        T.handle_add_byte_offset(kv.data, D_V // 2 * BF16_BYTES),
        D_V // 2,
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
    kv_part0_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        kv_part0_tensormap,
        "bfloat16",
        2,
        kv.data,
        D_V // 2,
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
        2,
        out.data,
        D_V,
        s_q * h_q,
        D_V * BF16_BYTES,
        64,
        B_H,
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
        2,
        out.data,
        D_V,
        s_q * h_q,
        D_V * BF16_BYTES,
        64,
        B_H,
        1,
        1,
        0,
        3,
        3,
        0,
    )
    q_nope_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        q_nope_tensormap,
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
        B_H,
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
    if d_qk > D_V:
        q_rope_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            q_rope_tensormap,
            "bfloat16",
            4,
            q.data,
            32,
            h_q,
            d_qk // 32,
            s_q,
            d_qk * BF16_BYTES,
            32 * BF16_BYTES,
            h_q * d_qk * BF16_BYTES,
            32,
            B_H,
            Q_ROPE_DIM // 32,
            1,
            1,
            1,
            1,
            1,
            0,
            2,
            3,
            0,
        )
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    iket = IketProfiler()
    # CUDA_TRANSCRIBE_START: sparse_attn_fwd_kernel lines 65-71. One CTA per query row;
    # warp 0 owns Q TMA, warps 0-1 own O TMA, warpgroup 0 also does softmax/epilogue.
    s_q_idx = T.cta_id([s_q])
    warpgroup_idx = T.warpgroup_id([3])
    warp_idx_in_wg = T.warp_id_in_wg([4])
    lane_idx = T.lane_id([32])
    idx_in_warpgroup = T.thread_id_in_wg([128])
    warp_idx: T.let = warpgroup_idx * 4 + warp_idx_in_wg
    max_k_blocks = T.meta_var(topk // B_TOPK)
    if have_topk_length:
        topk_len: T.int32
        T.ptx.ld.global_.s32(topk_len, topk_length.ptr_to([s_q_idx]))
        num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    else:
        topk_len = T.meta_var(topk)
        num_k_blocks = T.meta_var(max_k_blocks)
    have_rope = T.meta_var(d_qk == 576)
    if have_rope:
        if warp_idx == 0:
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_rope_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_nope_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(out_part0_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(out_part1_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(kv_part0_tensormap)))
    if warp_idx == 0:
        if T.cuda.elect_sync():
            T.evaluate(T.ptx.prefetch.tensormap(T.address_of(kv_part1_tensormap)))

    # CUDA phase1.cuh:73-78, config.h:111-139. Reserve SharedMemoryPlan offsets now;
    # instantiate bf16 MMA views only at their use sites (unused ones trip BF16 legalization).
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    k_rope = pool.alloc_tcgen05_mma_AB(
        (B_TOPK, Q_ROPE_DIM), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
    )
    # *_tiled_mma refolds stay here: the gemm B-operand descriptor hoists to the decl site.
    k_rope_tiled_mma = k_rope.rearrange("r (h c) -> (h r) c", h=2)
    k_nope = pool.alloc_tcgen05_mma_AB((NUM_BUFS, B_TOPK, D_V), "bfloat16")
    k_nope_tiled_mma = k_nope.rearrange("b r (dc h ci) -> b (h r) (dc ci)", dc=4, h=2, ci=64)
    u_end = T.meta_var(pool.offset)
    # q_nope aliases the last k_nope stage: Q moves to TMEM before that stage is used.
    pool.move_base_to(u_end - B_H * D_V * BF16_BYTES)
    q_nope = pool.alloc_tcgen05_mma_AB((B_H, D_V), "bfloat16")
    # o_smem aliases the front of the region: O is only written in the epilogue.
    pool.move_base_to(u_base)
    o_smem = pool.alloc_tcgen05_mma_AB((B_H, D_V), "bfloat16")
    pool.move_base_to(u_end)

    p_exchange_buf = pool.alloc((4, 32 * (B_TOPK // 2)), "float32")
    s_q_rope_base = T.meta_var(pool.offset)
    q_rope = pool.alloc_tcgen05_mma_AB(
        (B_H, Q_ROPE_DIM), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
    )
    q_rope_end = T.meta_var(pool.offset)
    # s_smem_gemm aliases q_rope: Q RoPE moves to TMEM before the first S tile is stored.
    pool.move_base_to(s_q_rope_base)
    s_smem_gemm = pool.alloc_tcgen05_mma_AB(
        (B_H, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
    )
    pool.move_base_to(q_rope_end)

    local_mma_desc = T.meta_var(d_qk > D_V and s_kv == 8192)
    if not local_mma_desc:
        if have_rope:
            qk_rope_desc = SmemDescriptor()
            qk_rope_desc.init(k_rope_tiled_mma.ptr_to([0, 0]), ldo=0, sdo=32, swizzle=2)
        pv_b_lo_desc = SmemDescriptor()
        pv_b_lo_desc.init(k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3)
        pv_b_hi_desc = SmemDescriptor()
        pv_b_hi_desc.init(k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3)
        qk_nope_desc = SmemDescriptor()
        qk_nope_desc.init(k_nope_tiled_mma.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
    if have_rope:
        q_rope_cp_desc: T.uint64
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(q_rope_cp_desc), T.reinterpret(T.handle().ty, T.uint64(0)), 1, 32, 2
        )
    if not local_mma_desc:
        pv_a_lo_desc = SmemDescriptor()
        pv_a_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
        pv_a_hi_desc = SmemDescriptor()
        pv_a_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)

    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q_nope = TMABar(pool, 1)
    bar_prologue_q_rope = TMABar(pool, 1)
    bar_prologue_utccp_nope = TCGen05Bar(pool, 1)
    bar_prologue_utccp_rope = TCGen05Bar(pool, 1)
    bar_qk_nope_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_rope_done = TCGen05Bar(pool, 1)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_kv_nope_ready_part0 = TMABar(pool, NUM_BUFS)
    bar_kv_nope_ready_part1 = TMABar(pool, NUM_BUFS)
    bar_kv_rope_ready = MBarrier(pool, 1)
    bar_p_free = MBarrier(pool, 1)
    bar_so_ready = MBarrier(pool, 1)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()

    # CUDA phase1.cuh:77. h_kv is fixed to 1, so the row pointer is
    # params.indices + s_q_idx * params.stride_indices_s_q.
    g_indices_base: T.let = s_q_idx * stride_indices_s_q
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=1, tmem_addr=tmem_start_addr)
    # O accumulator: one alloc. Col halves [0:256)/[256:512) = E lo/hi gemm outputs;
    # reads back as a plain (128, 256) datapath-D tile.
    o_tmem_col = T.meta_var(tmem_pool.offset)
    _o_tmem = tmem_pool.alloc_tcgen05_mma_D(
        (B_H, D_V), "float32", M=64, cta_group=1, ws=True, group=(2, 2, 128)
    )
    q_nope_tmem_col = T.meta_var(tmem_pool.offset)
    _q_nope_tmem = tmem_pool.alloc_tcgen05_mma_A(
        (2, B_H, D_V // 2), "bfloat16", M=64, cta_group=1, ws=True
    )
    q_rope_tmem_col = T.meta_var(tmem_pool.offset)
    _q_rope_tmem = tmem_pool.alloc_tcgen05_mma_A(
        (2, B_H, Q_ROPE_DIM // 2), "bfloat16", M=64, cta_group=1, ws=True
    )
    tmem_p_col = T.meta_var(tmem_pool.offset)
    # .ws logits gemm C: two batched 64x64 lane-half partials.
    _tmem_p = tmem_pool.alloc_tcgen05_mma_D((2, B_H, B_TOPK), "float32", M=64, cta_group=1, ws=True)
    mma_p_accumulate: T.uint32 = 0
    mma_o_accumulate: T.uint32 = 0

    # CUDA phase1.cuh:100-150.  Warp 0 performs descriptor prefetch, Q TMA
    # launch, prologue barrier init, and TMEM allocation.
    if warp_idx == 0:
        prologue_token = iket.range_start("h64-q-load")
        if T.cuda.elect_sync():
            bar_prologue_q_nope.init(1)
            bar_prologue_q_rope.init(1)
            T.ptx.fence.mbarrier_init.release.cluster()

            if have_rope:
                T.evaluate(
                    T.ptx[_TMA_G2S_4D_CACHE](
                        q_rope.ptr_to([0, 0]),
                        T.address_of(q_rope_tensormap),
                        T.int32(0),
                        T.int32(0),
                        T.int32(D_V // 32),
                        T.cast(s_q_idx, "int32"),
                        T.cuda.cvta_generic_to_shared(bar_prologue_q_rope.ptr_to([0])),
                        _Q_TMA_CACHE_HINT,
                    )
                )

            T.evaluate(
                T.ptx[_TMA_G2S_4D_CACHE](
                    q_nope.ptr_to([0, 0]),
                    T.address_of(q_nope_tensormap),
                    T.int32(0),
                    T.int32(0),
                    T.int32(0),
                    T.cast(s_q_idx, "int32"),
                    T.cuda.cvta_generic_to_shared(bar_prologue_q_nope.ptr_to([0])),
                    _Q_TMA_CACHE_HINT,
                )
            )
            bar_prologue_utccp_rope.init(1)
            bar_prologue_utccp_nope.init(1)
            if bar_qk_nope_done.leader:
                for init_stage in T.unroll(NUM_BUFS):
                    T.ptx.mbarrier.init.shared.b64(
                        bar_qk_nope_done.ptr_to([init_stage]), T.uint32(1)
                    )
                    T.ptx.mbarrier.init.shared.b64(bar_sv_done.ptr_to([init_stage]), T.uint32(1))
                    T.ptx.mbarrier.init.shared.b64(
                        bar_kv_nope_ready_part0.ptr_to([init_stage]), T.uint32(1)
                    )
                    T.ptx.mbarrier.init.shared.b64(
                        bar_kv_nope_ready_part1.ptr_to([init_stage]), T.uint32(1)
                    )
                    T.ptx.mbarrier.init.shared.b64(
                        bar_k_valid_ready.ptr_to([init_stage]), T.uint32(B_TOPK // 8)
                    )
                    T.ptx.mbarrier.init.shared.b64(
                        bar_k_valid_free.ptr_to([init_stage]), T.uint32(128)
                    )
            bar_p_free.init(128)
            bar_so_ready.init(128)
            bar_qk_rope_done.init(1)
            bar_kv_rope_ready.init(64)
            T.ptx.fence.mbarrier_init.release.cluster()

        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(tmem_start_addr[0]), T.uint32(512)
        )
        allocated_tmem_start: T.uint32
        T.ptx.ld.shared.u32(allocated_tmem_start, tmem_start_addr.ptr_to([0]))
        T.cuda.trap_when_assert_failed(allocated_tmem_start == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
        iket.range_end(prologue_token)

    T.cuda.cta_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:152-168.  Scale/exp warpgroup state.
        mi: T.float32 = MAX_INIT_VAL
        li: T.float32 = 0.0
        real_mi: T.float32 = T.float32(-float("inf"))

        # CUDA phase1.cuh:169-244. Scale/exp loop: P TMEM read/mask/reduce, row max,
        # S generation, S shared store, conditional O rescale.
        for k in T.serial(0, num_k_blocks, unroll=False):
            softmax_token = iket.range_start("h64-softmax-tile")
            T.ptx.bar.sync(T.uint32(BAR_WG0_WARP02 + T.bitwise_and(warp_idx, T.int32(1))), 64)
            cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
            cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
            qk_wait_token = iket.range_start("h64-qk-wait")
            bar_qk_nope_done.wait(cur_buf, cur_phase)
            iket.range_end(qk_wait_token)
            bar_k_valid_ready.wait(cur_buf, cur_phase)
            T.ptx.tcgen05.fence__after_thread_sync()

            # CUDA common_subroutine.h:75-134 retrieve_mask_and_reduce_p.
            p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, (B_TOPK // 2)), "float32")
            p_peer_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, (B_TOPK // 2)), "float32")
            p = p_frag.local()
            p_peer = p_peer_frag.local()
            if warp_idx < 2:
                T.evaluate(_tmem_load(p, T.uint32(tmem_p_col), B_TOPK // 2))
                T.evaluate(
                    _tmem_load(
                        p_peer,
                        T.cuda.get_tmem_addr(T.uint32(tmem_p_col), 0, B_TOPK // 2),
                        B_TOPK // 2,
                    )
                )
            else:
                T.evaluate(_tmem_load(p_peer, T.uint32(tmem_p_col), B_TOPK // 2))
                T.evaluate(
                    _tmem_load(
                        p, T.cuda.get_tmem_addr(T.uint32(tmem_p_col), 0, B_TOPK // 2), B_TOPK // 2
                    )
                )
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.ptx.tcgen05.fence__before_thread_sync()
            bar_p_free.arrive(0)

            valid_word_offset: T.int32 = T.if_then_else(warp_idx >= 2, (B_TOPK // 2) // 32, 0)
            is_k_valid_u32: T.uint32
            T.ptx.ld.shared.u32(
                is_k_valid_u32, is_k_valid.view("uint32").ptr_to([cur_buf, valid_word_offset])
            )
            for p_i in T.unroll(B_TOPK // 2):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_u32, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i] = T.cuda.uint_as_float(
                    T.if_then_else(
                        invalid_p_predicate, T.uint32(0xFF800000), T.cuda.float_as_uint(p[p_i])
                    )
                )

            for exchange_i in T.unroll((B_TOPK // 2) // 4):
                exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                p_peer_offset: T.let = exchange_i * 4
                p_peer_u32 = p_peer.view("uint32")
                T.ptx.st.shared.v4.u32(
                    p_exchange_buf.ptr_to([warp_idx ^ 2, exchange_offset]),
                    p_peer_u32[p_peer_offset],
                    p_peer_u32[p_peer_offset + 1],
                    p_peer_u32[p_peer_offset + 2],
                    p_peer_u32[p_peer_offset + 3],
                )
            T.ptx.bar.sync(T.uint32(BAR_WG0_WARP02 + T.bitwise_and(warp_idx, T.int32(1))), 64)
            p_add_pair0: T.uint64
            p_add_pair1: T.uint64
            for exchange_i in T.unroll((B_TOPK // 2) // 4):
                exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                p_exchange_tmp = T.alloc_local((4,), "float32")
                p_exchange_tmp_u32 = p_exchange_tmp.view("uint32")
                T.ptx.ld.shared.v4.u32(
                    p_exchange_tmp_u32[0],
                    p_exchange_tmp_u32[1],
                    p_exchange_tmp_u32[2],
                    p_exchange_tmp_u32[3],
                    p_exchange_buf.ptr_to([warp_idx, exchange_offset]),
                )
                p_pair0: T.let = T.cuda.make_float2(p[exchange_i * 4], p[exchange_i * 4 + 1])
                peer_pair0: T.let = T.cuda.make_float2(p_exchange_tmp[0], p_exchange_tmp[1])
                T.ptx.add.rn.f32x2(p_add_pair0, p_pair0, peer_pair0)
                p[exchange_i * 4] = T.cuda.float2_x(p_add_pair0)
                p[exchange_i * 4 + 1] = T.cuda.float2_y(p_add_pair0)
                p_pair1: T.let = T.cuda.make_float2(p[exchange_i * 4 + 2], p[exchange_i * 4 + 3])
                peer_pair1: T.let = T.cuda.make_float2(p_exchange_tmp[2], p_exchange_tmp[3])
                T.ptx.add.rn.f32x2(p_add_pair1, p_pair1, peer_pair1)
                p[exchange_i * 4 + 2] = T.cuda.float2_x(p_add_pair1)
                p[exchange_i * 4 + 3] = T.cuda.float2_y(p_add_pair1)

            bar_k_valid_free.arrive(cur_buf)

            cur_pi_max: T.float32 = T.float32(-float("inf"))
            for p_i in T.unroll(B_TOPK // 2):
                cur_pi_max = T.max(cur_pi_max, p[p_i])
            cur_pi_max = cur_pi_max * sm_scale_div_log2
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

            # S frag: warpgroup-distributed (B_H, B_TOPK) tile.
            s_frag = T.alloc_buffer(
                (B_H, B_TOPK),
                "bfloat16",
                scope="local",
                layout=TileLayout(
                    S[(2, 32, 2, 32) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                ),
            )
            s_pack = s_frag.local().view("uint32")
            cur_sum_pair: T.uint64 = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
            fma_pair: T.uint64
            for s_i in T.unroll((B_TOPK // 2) // 2):
                p_pair: T.let = T.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1])
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

            if k > 0:
                prev_buf: T.int32 = _ring_mod3(k - 1, max_k_blocks)
                prev_phase: T.int32 = _ring_phase_parity(k - 1, max_k_blocks)
                pv_wait_token = iket.range_start("h64-pv-wait")
                bar_sv_done.wait(prev_buf, prev_phase)
                iket.range_end(pv_wait_token)

            # On the first iteration s_smem_gemm aliases q_rope, which was
            # read by an async TCGEN05 copy.  On later iterations it is the
            # completed SxV MMA stage.  Order either async read before the
            # generic S write below.
            T.ptx.fence.proxy.async_.shared__cta()

            # CUDA phase1.cuh:229-232 S store (vectorized by the reg copy path).
            s_base: T.let = idx_in_warpgroup // 64 * 2048 + idx_in_warpgroup % 64 * 8
            s_words = s_frag.local().view("uint32")
            for s_store_i in T.unroll(4):
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
                # CUDA common_subroutine.h:147-168 rescale_O.
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
            bar_so_ready.arrive(0)
            iket.range_end(softmax_token)

        # CUDA phase1.cuh:246-357.  Epilogue scalar exchange, O TMEM readback,
        # output scaling/bf16 staging, and the two elected-warp O TMA stores.
        epilogue_token = iket.range_start("h64-output")
        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        T.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), li)
        T.ptx.bar.sync(T.uint32(BAR_WG0_SYNC), 128)
        peer_li: T.float32
        T.ptx.ld.shared.f32(peer_li, rowwise_li_buf.ptr_to([idx_in_warpgroup ^ 64]))
        li = li + peer_li

        if idx_in_warpgroup < B_H:
            cur_lse: T.float32
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma.rn.f32(cur_lse, mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            T.ptx.st.global_.f32(max_logits.ptr_to([s_q_idx, idx_in_warpgroup]), real_mi * LN_2)
            T.ptx.st.global_.f32(lse.ptr_to([s_q_idx, idx_in_warpgroup]), cur_lse)

        last_k: T.int32 = num_k_blocks - 1
        last_buf: T.int32 = _ring_mod3(last_k, max_k_blocks)
        last_phase: T.int32 = _ring_phase_parity(last_k, max_k_blocks)
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.tcgen05.fence__after_thread_sync()
        # bar_sv_done makes the final TCGEN05 read complete; cross back from
        # the async proxy before o_smem aliases and overwrites its K stage.
        T.ptx.fence.proxy.async_.shared__cta()

        attn_sink_log2: T.let = (
            T.cuda.ldg(attn_sink.ptr_to([idx_in_warpgroup % B_H]), "float32") * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        sink_exp: T.float32
        T.ptx.ex2.approx.ftz.f32(sink_exp, attn_sink_log2 - mi)
        output_scale: T.float32 = T.cuda.fdividef(T.float32(1.0), li + sink_exp)

        o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "float32")
        o_epi = o_epi_frag.local()
        o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "bfloat16")
        o_epi_bf16 = o_epi_bf16_frag.local()
        have_valid_indices: T.let = T.cuda.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(64):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        for epi_c in T.unroll(2):
            for epi_k in T.unroll((D_V // 4) // 64):
                if have_valid_indices:
                    # CUDA phase1.cuh:314-317: TMEM O load/fence.
                    T.evaluate(
                        _tmem_load(
                            o_epi,
                            T.cuda.get_tmem_addr(T.uint32(o_tmem_col), 0, epi_c * 128 + epi_k * 64),
                            64,
                        )
                    )
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                for scale_i in T.unroll(64 // 2):
                    mul_f32x2(o_epi, scale_i * 2, output_scale)
                for cast_i in T.unroll(64 // 2):
                    T.evaluate(_cast_f32x2_bf16x2(o_epi_bf16, o_epi, cast_i * 2))
                o_epi_words = o_epi_bf16.view("uint32")
                for o_store_i in T.unroll(8):
                    s_off: T.let = (
                        (epi_k // 2 + epi_c) % 2 * 16384
                        + idx_in_warpgroup // 64 * 8192
                        + epi_k % 2 * 4096
                        + idx_in_warpgroup % 64 * 64
                        + T.bitwise_xor(
                            o_store_i * 8,
                            T.shift_left(
                                T.bitwise_and(
                                    (epi_k // 2 + epi_c) % 2 * 256
                                    + idx_in_warpgroup // 64 * 128
                                    + epi_k % 2 * 64
                                    + idx_in_warpgroup % 64,
                                    7,
                                ),
                                3,
                            ),
                        )
                    )
                    s_ptr: T.let = T.ptr_byte_offset(
                        o_smem.ptr_to([0, 0]), s_off * BF16_BYTES, "bfloat16"
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
                        # CUDA phase1.cuh:335-342: first half O TMA store.
                        T.evaluate(
                            T.ptx[_TMA_S2G_2D](
                                T.address_of(out_part0_tensormap),
                                T.cast(epi_c * 256 + epi_k * 64, "int32"),
                                T.cast(s_q_idx * B_H, "int32"),
                                T.ptr_byte_offset(
                                    o_smem.ptr_to([0, 0]),
                                    (epi_c * 256 + epi_k * 64) * B_H * BF16_BYTES,
                                    "bfloat16",
                                ),
                            )
                        )
                if warp_idx == 1:
                    if T.cuda.elect_sync():
                        # CUDA phase1.cuh:343-350: second half O TMA store.
                        T.evaluate(
                            T.ptx[_TMA_S2G_2D](
                                T.address_of(out_part1_tensormap),
                                T.cast(epi_c * 256 + epi_k * 64 + 128, "int32"),
                                T.cast(s_q_idx * B_H, "int32"),
                                T.ptr_byte_offset(
                                    o_smem.ptr_to([0, 0]),
                                    (epi_c * 256 + epi_k * 64 + 128) * B_H * BF16_BYTES,
                                    "bfloat16",
                                ),
                            )
                        )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), T.uint32(512))
        iket.range_end(epilogue_token)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:358-412. KV NoPE producer. Scalar index loads + skip
        # decisions transcribed; gather4 kept at its source-order call site.
        kv_nope_token = iket.range_start("h64-kv-nope-load")
        wg1_warp_idx: T.let = warp_idx - 4
        # This warp's 16 interleaved NoPE rows: split the 64-row dim into
        # (stripe, warp, row) and pick this warp, merging stripe x row.
        k_nope_warp = k_nope.tile((1, (-1, WG1_NUM_WARPS, 4)))[:, wg1_warp_idx, :]
        if T.cuda.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                selected_idx = T.alloc_local((WG1_ROWS_PER_WARP, 4), "int32")
                max_indices: T.int32 = -1
                min_indices: T.int32 = s_kv
                # This warp's 16 indices from the (local_row, warp, j) split.
                selected_words = selected_idx.view(16).view("uint32")
                for selected_load_i in T.unroll(4):
                    selected_word: T.let = selected_load_i * 4
                    T.ptx.ld.global_.nc.v4.u32(
                        selected_words[selected_word],
                        selected_words[selected_word + 1],
                        selected_words[selected_word + 2],
                        selected_words[selected_word + 3],
                        indices.ptr_to(
                            [g_indices_base + k * B_TOPK + wg1_warp_idx * 4 + selected_load_i * 16]
                        ),
                    )
                for local_row in T.unroll(WG1_ROWS_PER_WARP):
                    for j in T.unroll(4):
                        idx: T.let = selected_idx[local_row, j]
                        max_indices = T.max(max_indices, idx)
                        min_indices = T.min(min_indices, idx)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)

                if k == 2:
                    bar_prologue_utccp_nope.wait(0, 0)

                cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                bar_sv_done.wait(cur_buf, T.bitwise_xor(cur_phase, T.int32(1)))

                if not should_skip_tma:
                    dst_part0 = k_nope_warp.sub[cur_buf, :, 0 : D_V // 2]
                    for row_group in T.unroll(WG1_ROWS_PER_WARP):
                        for col_atom in T.unroll((D_V // 2) // 64):
                            dst_part0_offset: T.let = (
                                cur_buf * B_TOPK * D_V
                                + col_atom * 64 * B_TOPK
                                + (wg1_warp_idx * 4 + row_group * 16) * 64
                            ) * BF16_BYTES
                            T.evaluate(
                                T.ptx[_TMA_GATHER4_2D_CACHE](
                                    T.ptr_byte_offset(
                                        k_nope.ptr_to([0, 0, 0]), dst_part0_offset, "bfloat16"
                                    ),
                                    T.address_of(kv_part0_tensormap),
                                    T.cast(col_atom * 64, "int32"),
                                    selected_idx[row_group, 0],
                                    selected_idx[row_group, 1],
                                    selected_idx[row_group, 2],
                                    selected_idx[row_group, 3],
                                    T.cuda.cvta_generic_to_shared(
                                        bar_kv_nope_ready_part0.ptr_to([cur_buf])
                                    ),
                                    _KV_TMA_CACHE_HINT,
                                )
                            )
                    dst_part1 = k_nope_warp.sub[cur_buf, :, D_V // 2 : D_V]
                    for row_group in T.unroll(WG1_ROWS_PER_WARP):
                        for col_atom in T.unroll((D_V // 2) // 64):
                            dst_part1_offset: T.let = (
                                cur_buf * B_TOPK * D_V
                                + (D_V // 2 + col_atom * 64) * B_TOPK
                                + (wg1_warp_idx * 4 + row_group * 16) * 64
                            ) * BF16_BYTES
                            T.evaluate(
                                T.ptx[_TMA_GATHER4_2D_CACHE](
                                    T.ptr_byte_offset(
                                        k_nope.ptr_to([0, 0, 0]), dst_part1_offset, "bfloat16"
                                    ),
                                    T.address_of(kv_part1_tensormap),
                                    T.cast(col_atom * 64, "int32"),
                                    selected_idx[row_group, 0],
                                    selected_idx[row_group, 1],
                                    selected_idx[row_group, 2],
                                    selected_idx[row_group, 3],
                                    T.cuda.cvta_generic_to_shared(
                                        bar_kv_nope_ready_part1.ptr_to([cur_buf])
                                    ),
                                    _KV_TMA_CACHE_HINT,
                                )
                            )
                else:
                    tx_bytes = T.uint32(WG1_ROWS_PER_WARP * 4 * (D_V // 2) * BF16_BYTES)
                    T.ptx.mbarrier.complete_tx.relaxed.cluster.shared__cluster.b64(
                        bar_kv_nope_ready_part0.ptr_to([cur_buf]), T.uint32(tx_bytes)
                    )
                    T.ptx.mbarrier.complete_tx.relaxed.cluster.shared__cluster.b64(
                        bar_kv_nope_ready_part1.ptr_to([cur_buf]), T.uint32(tx_bytes)
                    )
        iket.range_end(kv_nope_token)

    else:
        # CUDA phase1.cuh:413-572. MMA warpgroup. Keep every low-level async
        # issue and its completion barrier in source order on the elected thread.
        if warp_idx == 8:
            mma_token = iket.range_start("h64-qk-pv-issue")
            if T.cuda.elect_sync():
                if have_rope:
                    bar_prologue_q_rope.arrive(0, tx_count=B_H * (d_qk - D_V) * BF16_BYTES)
                    bar_prologue_q_rope.wait(0, 0)
                    T.ptx.tcgen05.fence__after_thread_sync()
                    for q_rope_flat in T.unroll(2):
                        q_rope_src: T.let = T.ptr_byte_offset(
                            q_rope.ptr_to([0, 0]), q_rope_flat % 2 * 2 * 16, "bfloat16"
                        )
                        T.evaluate(
                            T.ptx[_TCGEN_CP_128X256](
                                T.cast(q_rope_tmem_col + q_rope_flat % 2 * 8, "uint32"),
                                _replace_smem_desc_addr(q_rope_cp_desc, q_rope_src),
                            )
                        )
                    T.evaluate(
                        T.ptx[_TCGEN_COMMIT](
                            T.cuda.cvta_generic_to_shared(bar_prologue_utccp_rope.ptr_to([0]))
                        )
                    )

                bar_prologue_q_nope.arrive(0, tx_count=B_H * D_V * BF16_BYTES)
                bar_prologue_q_nope.wait(0, 0)
                T.ptx.tcgen05.fence__after_thread_sync()
                q_nope_cp_desc: T.uint64
                T.cuda.tcgen05.encode_matrix_descriptor(
                    T.address_of(q_nope_cp_desc),
                    T.reinterpret(T.handle().ty, T.uint64(0)),
                    1,
                    64,
                    3,
                )
                for q_nope_flat in T.unroll(16):
                    q_nope_src: T.let = T.ptr_byte_offset(
                        q_nope.ptr_to([0, 0]),
                        (q_nope_flat % 4 * 1024 + q_nope_flat // 4 % 4 * 2) * 16,
                        "bfloat16",
                    )
                    T.evaluate(
                        T.ptx[_TCGEN_CP_128X256](
                            T.cast(
                                q_nope_tmem_col + (q_nope_flat % 4 * 32 + q_nope_flat // 4 % 4 * 8),
                                "uint32",
                            ),
                            _replace_smem_desc_addr(q_nope_cp_desc, q_nope_src),
                        )
                    )
                T.evaluate(
                    T.ptx[_TCGEN_COMMIT](
                        T.cuda.cvta_generic_to_shared(bar_prologue_utccp_nope.ptr_to([0]))
                    )
                )

                if have_rope:
                    bar_prologue_utccp_rope.wait(0, 0)

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                        cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                        bar_p_free.wait(0, T.bitwise_xor(T.bitwise_and(k, T.int32(1)), T.int32(1)))
                        T.ptx.tcgen05.fence__after_thread_sync()

                        if have_rope:
                            bar_kv_rope_ready.wait(0, T.bitwise_and(k, T.int32(1)))
                            T.ptx.tcgen05.fence__after_thread_sync()
                            # CUDA phase1.cuh:489 Q RoPE x K RoPE MMA.
                            mma_p_accumulate = T.uint32(0)
                            if local_mma_desc:
                                qk_rope_desc_local = SmemDescriptor()
                                qk_rope_desc_local.init(
                                    k_rope_tiled_mma.ptr_to([0, 0]), ldo=0, sdo=32, swizzle=2
                                )
                                for mma_mi in T.unroll(1):
                                    for mma_ni in T.unroll(1):
                                        for mma_ki in T.unroll(2):
                                            T.evaluate(
                                                T.ptx[_MMA_WS_F16](
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    T.cast(q_rope_tmem_col + mma_ki * 8, "uint32"),
                                                    qk_rope_desc_local.add_16B_offset(
                                                        (mma_ni * 4096 + mma_ki * 16) // 8
                                                    ),
                                                    T.uint32(69207184),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                    T.uint64(0),
                                                )
                                            )
                            else:
                                for mma_mi in T.unroll(1):
                                    for mma_ni in T.unroll(1):
                                        for mma_ki in T.unroll(2):
                                            T.evaluate(
                                                T.ptx[_MMA_WS_F16](
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    T.cast(q_rope_tmem_col + mma_ki * 8, "uint32"),
                                                    qk_rope_desc.add_16B_offset(
                                                        (mma_ni * 4096 + mma_ki * 16) // 8
                                                    ),
                                                    T.uint32(69207184),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                    T.uint64(0),
                                                )
                                            )
                            T.evaluate(
                                T.ptx[_TCGEN_COMMIT](
                                    T.cuda.cvta_generic_to_shared(bar_qk_rope_done.ptr_to([0]))
                                )
                            )

                        if k == 0:
                            bar_prologue_utccp_nope.wait(0, 0)

                        for kv_nope_part_idx in T.unroll(2):
                            tx_bytes: T.let = B_TOPK * (D_V // 2) * BF16_BYTES
                            if kv_nope_part_idx == 0:
                                bar_kv_nope_ready_part0.arrive(cur_buf, tx_count=tx_bytes)
                                bar_kv_nope_ready_part0.wait(cur_buf, cur_phase)
                            else:
                                bar_kv_nope_ready_part1.arrive(cur_buf, tx_count=tx_bytes)
                                bar_kv_nope_ready_part1.wait(cur_buf, cur_phase)
                            T.ptx.tcgen05.fence__after_thread_sync()
                            # CUDA phase1.cuh:505-506 Q NoPE x K NoPE MMA.
                            clear_nope_accum: T.let = (not have_rope) & (kv_nope_part_idx == 0)
                            mma_p_accumulate = T.if_then_else(
                                clear_nope_accum, T.uint32(0), T.uint32(1)
                            )
                            if local_mma_desc:
                                qk_nope_desc_local = SmemDescriptor()
                                qk_nope_desc_local.init(
                                    k_nope_tiled_mma.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3
                                )
                                for mma_mi in T.unroll(1):
                                    for mma_ni in T.unroll(1):
                                        for mma_ki in T.unroll(8):
                                            qk_nope_offset: T.let = (
                                                mma_ki // 1024 * 32768
                                                + mma_ni * 32768
                                                + cur_buf * 32768
                                                + kv_nope_part_idx % 2 * 16384
                                                + mma_ki % 8 // 4 * 8192
                                                + mma_ki % 1024 // 8 * 64
                                                + mma_ki % 4 * 16
                                            ) // 8
                                            T.evaluate(
                                                T.ptx[_MMA_WS_F16](
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    T.cast(
                                                        q_nope_tmem_col
                                                        + kv_nope_part_idx * 64
                                                        + mma_ki * 8,
                                                        "uint32",
                                                    ),
                                                    qk_nope_desc_local.add_16B_offset(
                                                        qk_nope_offset
                                                    ),
                                                    T.uint32(69207184),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                    T.uint64(0),
                                                )
                                            )
                            else:
                                for mma_mi in T.unroll(1):
                                    for mma_ni in T.unroll(1):
                                        for mma_ki in T.unroll(8):
                                            qk_nope_offset: T.let = (
                                                mma_ki // 1024 * 32768
                                                + mma_ni * 32768
                                                + cur_buf * 32768
                                                + kv_nope_part_idx % 2 * 16384
                                                + mma_ki % 8 // 4 * 8192
                                                + mma_ki % 1024 // 8 * 64
                                                + mma_ki % 4 * 16
                                            ) // 8
                                            T.evaluate(
                                                T.ptx[_MMA_WS_F16](
                                                    T.cast(tmem_p_col + mma_ni * 64, "uint32"),
                                                    T.cast(
                                                        q_nope_tmem_col
                                                        + kv_nope_part_idx * 64
                                                        + mma_ki * 8,
                                                        "uint32",
                                                    ),
                                                    qk_nope_desc.add_16B_offset(qk_nope_offset),
                                                    T.uint32(69207184),
                                                    T.Or(
                                                        mma_ki != 0,
                                                        T.cast(mma_p_accumulate, "bool"),
                                                    ),
                                                    T.uint64(0),
                                                )
                                            )
                        T.evaluate(
                            T.ptx[_TCGEN_COMMIT](
                                T.cuda.cvta_generic_to_shared(bar_qk_nope_done.ptr_to([cur_buf]))
                            )
                        )

                    if k > 0:
                        cur_buf_prev: T.int32 = _ring_mod3(k - 1, max_k_blocks)
                        bar_so_ready.wait(0, T.bitwise_and(k - 1, T.int32(1)))
                        T.ptx.tcgen05.fence__after_thread_sync()
                        # CUDA phase1.cuh:521-523 S(i-1) x V(i-1) MMA.
                        mma_o_accumulate = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))
                        if local_mma_desc:
                            pv_b_lo_desc_local = SmemDescriptor()
                            pv_b_lo_desc_local.init(
                                k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3
                            )
                            pv_a_lo_desc_local = SmemDescriptor()
                            pv_a_lo_desc_local.init(
                                s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0
                            )
                            for mma_mi in T.unroll(1):
                                for mma_ni in T.unroll(1):
                                    for mma_ki in T.unroll(4):
                                        T.evaluate(
                                            T.ptx[_MMA_WS_F16](
                                                T.cast(o_tmem_col + mma_ni * 128, "uint32"),
                                                pv_a_lo_desc_local.add_16B_offset(
                                                    (
                                                        mma_ki % 4 * 1024
                                                        + mma_mi * 512
                                                        + mma_ki // 4 * 8
                                                    )
                                                    // 8
                                                ),
                                                pv_b_lo_desc_local.add_16B_offset(
                                                    (
                                                        (mma_ki * 16 + mma_ni) // 64 * 32768
                                                        + cur_buf_prev * 32768
                                                        + (mma_ki * 16 + mma_ni) % 64 * 64
                                                    )
                                                    // 8
                                                ),
                                                T.uint32(71369872),
                                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                                T.uint64(0),
                                            )
                                        )
                            pv_b_hi_desc_local = SmemDescriptor()
                            pv_b_hi_desc_local.init(
                                k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3
                            )
                            pv_a_hi_desc_local = SmemDescriptor()
                            pv_a_hi_desc_local.init(
                                s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0
                            )
                            for mma_mi in T.unroll(1):
                                for mma_ni in T.unroll(1):
                                    for mma_ki in T.unroll(4):
                                        T.evaluate(
                                            T.ptx[_MMA_WS_F16](
                                                T.cast(o_tmem_col + 128 + mma_ni * 128, "uint32"),
                                                pv_a_hi_desc_local.add_16B_offset(
                                                    (
                                                        mma_ki % 4 * 1024
                                                        + mma_mi * 512
                                                        + mma_ki // 4 * 8
                                                    )
                                                    // 8
                                                ),
                                                pv_b_hi_desc_local.add_16B_offset(
                                                    (
                                                        (mma_ki * 16 + mma_ni) // 64 * 32768
                                                        + cur_buf_prev * 32768
                                                        + (mma_ki * 16 + mma_ni) % 64 * 64
                                                        + 16384
                                                    )
                                                    // 8
                                                ),
                                                T.uint32(71369872),
                                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                                T.uint64(0),
                                            )
                                        )
                        else:
                            for mma_mi in T.unroll(1):
                                for mma_ni in T.unroll(1):
                                    for mma_ki in T.unroll(4):
                                        T.evaluate(
                                            T.ptx[_MMA_WS_F16](
                                                T.cast(o_tmem_col + mma_ni * 128, "uint32"),
                                                pv_a_lo_desc.add_16B_offset(
                                                    (
                                                        mma_ki % 4 * 1024
                                                        + mma_mi * 512
                                                        + mma_ki // 4 * 8
                                                    )
                                                    // 8
                                                ),
                                                pv_b_lo_desc.add_16B_offset(
                                                    (
                                                        (mma_ki * 16 + mma_ni) // 64 * 32768
                                                        + cur_buf_prev * 32768
                                                        + (mma_ki * 16 + mma_ni) % 64 * 64
                                                    )
                                                    // 8
                                                ),
                                                T.uint32(71369872),
                                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                                T.uint64(0),
                                            )
                                        )
                            for mma_mi in T.unroll(1):
                                for mma_ni in T.unroll(1):
                                    for mma_ki in T.unroll(4):
                                        T.evaluate(
                                            T.ptx[_MMA_WS_F16](
                                                T.cast(o_tmem_col + 128 + mma_ni * 128, "uint32"),
                                                pv_a_hi_desc.add_16B_offset(
                                                    (
                                                        mma_ki % 4 * 1024
                                                        + mma_mi * 512
                                                        + mma_ki // 4 * 8
                                                    )
                                                    // 8
                                                ),
                                                pv_b_hi_desc.add_16B_offset(
                                                    (
                                                        (mma_ki * 16 + mma_ni) // 64 * 32768
                                                        + cur_buf_prev * 32768
                                                        + (mma_ki * 16 + mma_ni) % 64 * 64
                                                        + 16384
                                                    )
                                                    // 8
                                                ),
                                                T.uint32(71369872),
                                                T.Or(mma_ki != 0, T.cast(mma_o_accumulate, "bool")),
                                                T.uint64(0),
                                            )
                                        )
                        mma_o_accumulate = T.uint32(1)
                        T.evaluate(
                            T.ptx[_TCGEN_COMMIT](
                                T.cuda.cvta_generic_to_shared(bar_sv_done.ptr_to([cur_buf_prev]))
                            )
                        )
            iket.range_end(mma_token)

        elif warp_idx == 9:
            # CUDA common_subroutine.h:14-44 load_indices_and_generate_mask.
            valid_mask_token = iket.range_start("h64-valid-mask")
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
                    abs_pos_start: T.let = k * B_TOPK
                    row_base: T.let = g_indices_base + k * B_TOPK + lane_idx * 8
                    lane_index_words = lane_indices.view("uint32")
                    T.evaluate(
                        T.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.u32"](
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
                    is_ks_valid_mask: T.int8 = pack_valid_mask8(
                        lane_indices, abs_pos_start, lane_idx, topk_len, s_kv
                    )

                    cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                    cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                    bar_k_valid_free.wait(cur_buf, T.bitwise_xor(cur_phase, T.int32(1)))
                    T.ptx.st.shared.b8(
                        is_k_valid.ptr_to([cur_buf, lane_idx]),
                        T.reinterpret("uint8", is_ks_valid_mask),
                    )
                    bar_k_valid_ready.arrive(cur_buf)
            iket.range_end(valid_mask_token)

        elif (warp_idx == 10) | (warp_idx == 11):
            kv_rope_token = iket.range_start("h64-k-rope-load")
            if have_rope:
                thread_idx: T.let = (warp_idx - 10) * 32 + lane_idx
                group_idx: T.let = thread_idx // 8
                idx_in_group: T.let = thread_idx % 8
                for k in T.serial(0, num_k_blocks, unroll=False):
                    rope_indices = T.alloc_local(((B_TOPK // (64 // 8)),), "int32")
                    for local_row in T.unroll(B_TOPK // (64 // 8)):
                        rope_indices[local_row] = T.cuda.ldg(
                            indices.ptr_to(
                                [g_indices_base + k * B_TOPK + group_idx + local_row * (64 // 8)]
                            ),
                            "int32",
                        )
                    bar_qk_rope_done.wait(
                        0, T.bitwise_xor(T.bitwise_and(k, T.int32(1)), T.int32(1))
                    )
                    for local_row in T.unroll(B_TOPK // (64 // 8)):
                        index = rope_indices[local_row]
                        is_valid_index: T.let = (index >= 0) & (index < s_kv)
                        kv_off: T.let = index * stride_kv_s_kv + D_V + idx_in_group * 8
                        T.evaluate(
                            T.ptx[_CP_ASYNC_CG_16B](
                                k_rope.ptr_to(
                                    [group_idx + local_row * (64 // 8), idx_in_group * 8]
                                ),
                                kv.ptr_to([kv_off]),
                                T.int32(16),
                                T.if_then_else(is_valid_index, T.uint32(16), T.uint32(0)),
                            )
                        )
                    T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(
                        bar_kv_rope_ready.ptr_to([0])
                    )
            iket.range_end(kv_rope_token)


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
    return kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=3.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    case = prepare_data(**kwargs)
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
