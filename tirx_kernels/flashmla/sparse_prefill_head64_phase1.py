# This file is a TIRx port of code from FlashMLA
# (https://github.com/deepseek-ai/FlashMLA @ 9241ae3e), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
from tirx_kernels.flashmla.utils._mask import pack_valid_mask8

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

TMA_G2S_4D_CACHE = (
    "cp.async.bulk.tensor.4d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
TMA_GATHER4_2D_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
)
TMA_S2G_2D = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
CP_ASYNC_CG_16B = "cp.async.cg.shared.global.L2::128B"
TCGEN_CP_128X256 = "tcgen05.cp.cta_group::1.128x256b"
TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
MMA_WS_F16 = "tcgen05.mma.ws.cta_group::1.kind::f16"
TMEM_LD_32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_LD_64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
TMEM_ST_32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
Q_TMA_CACHE_HINT = K.uint64(0x12F0000000000000)
KV_TMA_CACHE_HINT = K.uint64(0x14F0000000000000)


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


def _ring_mod3(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_mod3 = K.uint32(0x10210210)
        shift = K.cast(value, "uint32") * K.uint32(4)
        return K.cast(K.bitwise_and(K.shift_right(packed_mod3, shift), K.uint32(0xF)), "int32")

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = value - max_offset
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = K.Select(value < offset, value - (offset - NUM_BUFS), result)
    return result


def _ring_phase_parity(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_phase = K.uint32(0x38)
        return K.cast(
            K.bitwise_and(K.shift_right(packed_phase, K.cast(value, "uint32")), K.uint32(1)),
            "int32",
        )

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = K.int32((max_offset // NUM_BUFS) & 1)
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = K.Select(value < offset, K.int32(((offset - NUM_BUFS) // NUM_BUFS) & 1), result)
    return result


LD_INDICES_V8 = "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.u32"
ID_QK = 0x04200490
ID_PV = 0x04410490


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
    """Trace the kernel for one specialization.

    Every ``K.constexpr`` / ``K.meta_var`` of the original is a plain Python
    constant here — the host language is the macro system (design doc §1).
    """
    # orig:469-477.
    max_k_blocks = topk // B_TOPK
    have_rope = d_qk == 576
    # orig:531. The one trace-time axis this kernel has that FA4 does not: for
    # exactly one of the eight repo configs the MMA chains re-encode their
    # descriptors *inside* the k-loop; the other seven hoist to kernel scope.
    # Both arms are reachable and both are ported (NOTES §3).
    local_mma_desc = d_qk > D_V and s_kv == 8192

    def ring_mod3(value):
        """orig:277-287. `max_k_blocks` is 8, so only the nibble path traces."""
        return _ring_mod3(value, max_k_blocks)

    def ring_phase_parity(value):
        """orig:290-302. Same."""
        return _ring_phase_parity(value, max_k_blocks)

    def host_prelude(params):
        q = params["q"]
        kv = params["kv"]
        out = params["out"]

        def encode(data, rank, *shape):
            descriptor = K.Bind(K.tvm_stack_alloca("tensormap", 1))
            K.evaluate(
                K.call_packed(
                    "runtime.cuTensorMapEncodeTiled", descriptor, "bfloat16", rank, data, *shape
                )
            )
            return descriptor

        kv_part1 = encode(
            K.handle_add_byte_offset(kv.data, D_V // 2 * BF16_BYTES),
            2,
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
        kv_part0 = encode(
            kv.data, 2, D_V // 2, s_kv, stride_kv_s_kv * BF16_BYTES, 64, 1, 1, 1, 0, 3, 3, 0
        )
        out_part1 = encode(out.data, 2, D_V, s_q * h_q, D_V * BF16_BYTES, 64, B_H, 1, 1, 0, 3, 3, 0)
        out_part0 = encode(out.data, 2, D_V, s_q * h_q, D_V * BF16_BYTES, 64, B_H, 1, 1, 0, 3, 3, 0)
        q_nope = encode(
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
        q_rope = None
        if have_rope:
            q_rope = encode(
                q.data,
                4,
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
        return kv_part1, kv_part0, out_part1, out_part0, q_nope, q_rope

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=s_q, host_prelude=host_prelude)
    def sparse_flashmla_prefill_head64_phase1_kern(
        q: K.gptr[K.bf16, (s_q, h_q, d_qk)],
        kv: K.gptr[K.bf16, (s_kv * stride_kv_s_kv,)],
        indices: K.gptr[K.i32, (s_q * stride_indices_s_q,)],
        attn_sink: K.gptr[K.f32, (h_q,)],
        topk_length: K.gptr[K.i32, (s_q,)],
        out: K.gptr[K.bf16, (s_q, h_q, D_V)],
        max_logits: K.gptr[K.f32, (s_q, h_q)],
        lse: K.gptr[K.f32, (s_q, h_q)],
        *,
        host,
    ):
        (
            kv_part1_tensormap,
            kv_part0_tensormap,
            out_part1_tensormap,
            out_part0_tensormap,
            q_nope_tensormap,
            q_rope_tensormap,
        ) = host
        # ---- CTA coordinates — orig:463-468 ------------------------------
        # The original's own spelling. The entry already declares the
        # cta->warp->thread chain; `warpgroup_id`/`warp_id_in_wg` derive from
        # the warp-uniform `warp_id_in_cta` broadcast (which is also what feeds
        # the role dispatch), and `thread_id_in_wg` correctly does not
        # broadcast because it indexes per thread. `warp_idx = warpgroup_idx *
        # 4 + warp_idx_in_wg` (orig:468) folds back to the entry's own warp id,
        # so the port reads that rather than rebuilding it.
        warp_idx = K.warp_id()
        lane_idx = K.lane_id()
        idx_in_warpgroup = K.thread_id_in_wg([128])
        s_q_idx = K.cta_id()
        topk_len_buf = K.alloc_local([1], "int32")
        K.assign(topk_len_buf[0], K.int32(topk))
        if have_topk_length:
            K.assign(topk_len_buf[0], K.cuda.ldg(topk_length.ptr_to([s_q_idx]), "int32"))
        num_k_blocks_buf = K.alloc_local([1], "int32")
        K.assign(num_k_blocks_buf[0], (topk_len_buf[0] + B_TOPK - 1) // B_TOPK)

        # ---- descriptor prefetch — orig:478-496 --------------------------
        # Six separate `if warp_idx == 0: if elect_sync():` guards, kept as six
        # rather than merged: a collective keeps the original's branch and loop
        # placement exactly (KERN_PORTING.md §8 G3).
        def prefetch(tensor_map):
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(tensor_map))

        if have_rope:
            prefetch(q_rope_tensormap)
        prefetch(q_nope_tensormap)
        prefetch(out_part0_tensormap)
        prefetch(out_part1_tensormap)
        prefetch(kv_part0_tensormap)
        prefetch(kv_part1_tensormap)

        # ---- shared memory — orig:498-571 --------------------------------
        # Use the arena directly for the three union aliases and their explicit
        # high-water marks; swizzled allocations still go through smem.alloc.
        smem = K.smem_pool()
        pool = smem.pool

        u_base = pool.offset
        k_rope = smem.alloc((B_TOPK, Q_ROPE_DIM), "bfloat16", swizzle=K.SW64B).buf
        k_nope = smem.alloc((NUM_BUFS, B_TOPK, D_V), "bfloat16", swizzle=K.SW128B).buf
        u_end = pool.offset
        # q_nope aliases the last k_nope stage: Q moves to TMEM before that
        # stage is used.
        pool.move_base_to(u_end - B_H * D_V * BF16_BYTES)
        q_nope = smem.alloc((B_H, D_V), "bfloat16", swizzle=K.SW128B).buf
        # o_smem aliases the front of the region: O is only written in the
        # epilogue.
        pool.move_base_to(u_base)
        o_smem = smem.alloc((B_H, D_V), "bfloat16", swizzle=K.SW128B).buf
        pool.move_base_to(u_end)

        p_exchange_buf = pool.alloc((4, 32 * (B_TOPK // 2)), "float32")
        s_q_rope_base = pool.offset
        q_rope = smem.alloc((B_H, Q_ROPE_DIM), "bfloat16", swizzle=K.SW64B).buf
        q_rope_end = pool.offset
        # s_smem_gemm aliases q_rope: Q RoPE moves to TMEM before the first S
        # tile is stored.
        pool.move_base_to(s_q_rope_base)
        s_smem_gemm = smem.alloc((B_H, B_TOPK), "bfloat16", align=1024)
        pool.move_base_to(q_rope_end)

        # ---- hoisted matrix descriptors — orig:531-551 -------------------
        # `pv_b_lo`/`pv_b_hi` and `pv_a_lo`/`pv_a_hi` are byte-identical
        # encodes, DELIBERATELY duplicated so each chain owns its own register
        # pair. A port that CSEs them changes register pressure, so both are
        # reproduced (NOTES §3).
        if not local_mma_desc:
            if have_rope:
                qk_rope_desc = K.SmemDescriptor()
                qk_rope_desc.init(k_rope.ptr_to([0, 0]), ldo=0, sdo=32, swizzle=2)
            pv_b_lo_desc = K.SmemDescriptor()
            pv_b_lo_desc.init(k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3)
            pv_b_hi_desc = K.SmemDescriptor()
            pv_b_hi_desc.init(k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3)
            qk_nope_desc = K.SmemDescriptor()
            qk_nope_desc.init(k_nope.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3)
        if have_rope:
            q_rope_cp_desc = K.alloc_local([1], "uint64")
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(q_rope_cp_desc[0]), K.reinterpret(K.handle().ty, K.uint64(0)), 1, 32, 2
            )
        if not local_mma_desc:
            pv_a_lo_desc = K.SmemDescriptor()
            pv_a_lo_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)
            pv_a_hi_desc = K.SmemDescriptor()
            pv_a_hi_desc.init(s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0)

        is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
        bar_prologue_q_nope = K.TMABar(pool, 1)
        bar_prologue_q_rope = K.TMABar(pool, 1)
        bar_prologue_utccp_nope = K.TCGen05Bar(pool, 1)
        bar_prologue_utccp_rope = K.TCGen05Bar(pool, 1)
        bar_qk_nope_done = K.TCGen05Bar(pool, NUM_BUFS)
        bar_qk_rope_done = K.TCGen05Bar(pool, 1)
        bar_sv_done = K.TCGen05Bar(pool, NUM_BUFS)
        bar_kv_nope_ready_part0 = K.TMABar(pool, NUM_BUFS)
        bar_kv_nope_ready_part1 = K.TMABar(pool, NUM_BUFS)
        bar_kv_rope_ready = K.MBarrier(pool, 1)
        bar_p_free = K.MBarrier(pool, 1)
        bar_so_ready = K.MBarrier(pool, 1)
        bar_k_valid_ready = K.MBarrier(pool, NUM_BUFS)
        bar_k_valid_free = K.MBarrier(pool, NUM_BUFS)
        tmem_start_addr = pool.alloc((1,), "uint32", align=4)
        rowwise_max_buf = pool.alloc((128,), "float32")
        rowwise_li_buf = pool.alloc((128,), "float32")
        smem.commit()

        # orig:573-575. h_kv is fixed to 1, so the row pointer is
        # params.indices + s_q_idx * params.stride_indices_s_q.
        g_indices_base = K.Bind(s_q_idx * stride_indices_s_q)

        # ---- TMEM — fixed physical column map ----------------------------
        o_tmem_col = 0
        q_nope_tmem_col = 256
        q_rope_tmem_col = 384
        tmem_p_col = 400
        mma_p_accumulate = K.alloc_local([1], "uint32")
        K.assign(mma_p_accumulate[0], K.uint32(0))
        mma_o_accumulate = K.alloc_local([1], "uint32")
        K.assign(mma_o_accumulate[0], K.uint32(0))

        # ---- shared closures ---------------------------------------------

        def tmem_load(dst, tmem_col, width):
            """orig:66-68."""
            chain = TMEM_LD_32 if width == 32 else TMEM_LD_64
            K.ptx[chain](*(dst[i] for i in range(width)), tmem_col)

        def tmem_store(src, tmem_col, width=32):
            """orig:71-73."""
            assert width == 32
            K.ptx[TMEM_ST_32](tmem_col, *(src[i] for i in range(width)))

        def cast_f32x2_bf16x2(dst, src, offset):
            """orig:76-80."""
            dst_words = dst.view("uint32")
            K.ptx.cvt.rn.bf16x2.f32(dst_words[offset // 2], src[offset + 1], src[offset])

        def replace_smem_desc_addr(desc, smem_ptr):
            """orig:83-90. Splice a shared address into an encoded descriptor."""
            start_addr = K.cast(
                K.bitwise_and(
                    K.shift_right(K.cuda.cvta_generic_to_shared(smem_ptr), K.uint32(4)),
                    K.uint32(0x3FFF),
                ),
                "uint64",
            )
            return K.bitwise_or(K.bitwise_and(desc, K.bitwise_not(K.uint64(0x3FFF))), start_addr)

        def mul_f32x2(values, idx, multiplier):
            """orig:93-100, the `@K.inline mul_f32x2`, as a kernel-local closure.

            A packed f32x2 operand is ONE 64-bit value, not a two-element float
            window (GDN port notes §3.2).
            """
            packed = K.alloc_local([1], "uint64")
            rhs = K.alloc_local([1], "uint64")
            K.ptx.mov.b64(packed[0], values[idx], values[idx + 1])
            K.ptx.mov.b64(rhs[0], multiplier, multiplier)
            K.ptx.mul.rz.ftz.f32x2(packed[0], packed[0], rhs[0])
            K.ptx.mov.b64(values[idx], values[idx + 1], packed[0])

        def commit(bar, stage):
            """The matrix engine's arrive on a one-way barrier."""
            K.ptx[TCGEN_COMMIT](K.cuda.cvta_generic_to_shared(bar.ptr_to([stage])))

        def wg0_pair_sync():
            """orig:679/739. The 64-thread named barrier pairing warps {0,2} and
            {1,3} — the cross-warp P exchange's rendezvous. No FA4 analogue: it
            exists because the `.ws` M=64 gemm writes two batched 64x64
            lane-half partials that must be summed across the warp pair."""
            K.ptx.bar.sync(K.uint32(BAR_WG0_WARP02 + K.bitwise_and(warp_idx, K.int32(1))), 64)

        def iket_range(name):
            token = K.alloc_local([1], "uint32")
            K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        # ---- prologue: warp 0 — orig:597-665 ------------------------------
        # Not a role: warp 0 is a sub-range of the softmax role's warps and a
        # role owns a contiguous *disjoint* range. The original spells this on
        # the raw warp id too.
        with K.If(warp_idx == 0), K.Then():
            prologue_token = iket_range("h64-q-load")
            with K.If(K.cuda.elect_sync()), K.Then():
                bar_prologue_q_nope.init(1)
                bar_prologue_q_rope.init(1)
                K.ptx.fence.mbarrier_init.release.cluster()

                if have_rope:
                    K.ptx[TMA_G2S_4D_CACHE](
                        q_rope.ptr_to([0, 0]),
                        K.address_of(q_rope_tensormap),
                        K.int32(0),
                        K.int32(0),
                        K.int32(D_V // 32),
                        K.cast(s_q_idx, "int32"),
                        K.cuda.cvta_generic_to_shared(bar_prologue_q_rope.ptr_to([0])),
                        K.uint64(Q_TMA_CACHE_HINT),
                    )

                K.ptx[TMA_G2S_4D_CACHE](
                    q_nope.ptr_to([0, 0]),
                    K.address_of(q_nope_tensormap),
                    K.int32(0),
                    K.int32(0),
                    K.int32(0),
                    K.cast(s_q_idx, "int32"),
                    K.cuda.cvta_generic_to_shared(bar_prologue_q_nope.ptr_to([0])),
                    K.uint64(Q_TMA_CACHE_HINT),
                )
                bar_prologue_utccp_rope.init(1)
                bar_prologue_utccp_nope.init(1)
                # The five NUM_BUFS rings are initialized by hand under one
                # shared leader predicate rather than through five `bar.init()`
                # calls; transcribed as written (orig:634-651).
                with K.If(bar_qk_nope_done.leader), K.Then():
                    for init_stage in range(NUM_BUFS):
                        K.ptx.mbarrier.init.shared.b64(
                            bar_qk_nope_done.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_sv_done.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_kv_nope_ready_part0.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_kv_nope_ready_part1.ptr_to([init_stage]), K.uint32(1)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_k_valid_ready.ptr_to([init_stage]), K.uint32(B_TOPK // 8)
                        )
                        K.ptx.mbarrier.init.shared.b64(
                            bar_k_valid_free.ptr_to([init_stage]), K.uint32(128)
                        )
                bar_p_free.init(128)
                bar_so_ready.init(128)
                bar_qk_rope_done.init(1)
                bar_kv_rope_ready.init(64)
                K.ptx.fence.mbarrier_init.release.cluster()

            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                K.address_of(tmem_start_addr[0]), K.uint32(512)
            )
            allocated_tmem_start = K.alloc_local([1], "uint32")
            K.ptx.ld.shared.u32(allocated_tmem_start[0], tmem_start_addr.ptr_to([0]))
            K.cuda.trap_when_assert_failed(allocated_tmem_start[0] == K.uint32(0))
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            K.cuda.iket.range_end(prologue_token[0])

        K.cuda.cta_sync()

        # ==================================================================
        # Role bodies. Defined as closures so the five `with role:` blocks
        # below stay *back to back* with nothing at CTA scope between them —
        # adjacency is what lets K.specialize fold them into one else-if
        # chain (KERN_PORTING.md §8).
        # ==================================================================

        def softmax_and_epilogue():
            """warps 0-3 — orig:669-1007."""
            # orig:670-673. Scale/exp warpgroup state.
            mi = K.alloc_local([1], "float32")
            K.assign(mi[0], K.float32(MAX_INIT_VAL))
            li = K.alloc_local([1], "float32")
            K.assign(li[0], K.float32(0.0))
            real_mi = K.alloc_local([1], "float32")
            K.assign(real_mi[0], K.float32(-float("inf")))

            # orig:675-872. Scale/exp loop: P TMEM read/mask/reduce, row max,
            # S generation, S shared store, conditional O rescale.
            with K.serial(num_k_blocks_buf[0], unroll=False) as k:
                softmax_token = iket_range("h64-softmax-tile")
                wg0_pair_sync()
                cur_buf = K.alloc_local([1], "int32")
                K.assign(cur_buf[0], ring_mod3(k))
                cur_phase = K.alloc_local([1], "int32")
                K.assign(cur_phase[0], ring_phase_parity(k))
                qk_wait_token = iket_range("h64-qk-wait")
                bar_qk_nope_done.wait(cur_buf[0], cur_phase[0])
                K.cuda.iket.range_end(qk_wait_token[0])
                bar_k_valid_ready.wait(cur_buf[0], cur_phase[0])
                K.ptx.tcgen05.fence__after_thread_sync()

                # orig:688-762, CUDA common_subroutine.h:75-134
                # retrieve_mask_and_reduce_p.
                p = K.alloc_local((B_TOPK // 2,), "float32")
                p_peer = K.alloc_local((B_TOPK // 2,), "float32")
                with K.If(warp_idx < 2):
                    with K.Then():
                        tmem_load(p, K.uint32(tmem_p_col), B_TOPK // 2)
                        tmem_load(
                            p_peer,
                            K.cuda.get_tmem_addr(K.uint32(tmem_p_col), 0, B_TOPK // 2),
                            B_TOPK // 2,
                        )
                    with K.Else():
                        tmem_load(p_peer, K.uint32(tmem_p_col), B_TOPK // 2)
                        tmem_load(
                            p,
                            K.cuda.get_tmem_addr(K.uint32(tmem_p_col), 0, B_TOPK // 2),
                            B_TOPK // 2,
                        )
                K.ptx.tcgen05.wait__ld.sync.aligned()
                K.ptx.tcgen05.fence__before_thread_sync()
                bar_p_free.arrive(0)

                valid_word_offset = K.alloc_local([1], "int32")
                K.assign(
                    valid_word_offset[0], K.if_then_else(warp_idx >= 2, (B_TOPK // 2) // 32, 0)
                )
                is_k_valid_u32 = K.alloc_local([1], "uint32")
                K.ptx.ld.shared.u32(
                    is_k_valid_u32[0],
                    is_k_valid.view("uint32").ptr_to([cur_buf[0], valid_word_offset[0]]),
                )
                for p_i in range(B_TOPK // 2):
                    invalid_p_predicate = K.Bind(
                        K.bitwise_and(K.shift_right(is_k_valid_u32[0], K.uint32(p_i)), K.uint32(1))
                        == K.uint32(0)
                    )
                    K.ptx.mov.b32(
                        p[p_i],
                        K.cuda.uint_as_float(
                            K.if_then_else(
                                invalid_p_predicate,
                                K.uint32(0xFF800000),
                                K.cuda.float_as_uint(p[p_i]),
                            )
                        ),
                    )

                for exchange_i in range((B_TOPK // 2) // 4):
                    # An UNANNOTATED assignment in the original (orig:729), which
                    # is a declared one-element local, not a `K.let` -- the
                    # call-site census caught this as 16 missing `alignas(64)
                    # int` declarations (KERN_PORTING.md §8, "read the
                    # assignments, not the annotations").
                    exchange_offset = K.alloc_local([1], "int32")
                    K.assign(exchange_offset[0], exchange_i * 32 * 4 + lane_idx * 4)
                    p_peer_offset = exchange_i * 4
                    p_peer_u32 = p_peer.view("uint32")
                    K.ptx.st.shared.v4.u32(
                        p_exchange_buf.ptr_to([K.bitwise_xor(warp_idx, 2), exchange_offset[0]]),
                        p_peer_u32[p_peer_offset],
                        p_peer_u32[p_peer_offset + 1],
                        p_peer_u32[p_peer_offset + 2],
                        p_peer_u32[p_peer_offset + 3],
                    )
                wg0_pair_sync()
                p_add_pair0 = K.alloc_local([1], "uint64")
                p_add_pair1 = K.alloc_local([1], "uint64")
                for exchange_i in range((B_TOPK // 2) // 4):
                    # orig:743, unannotated again -- see the note above.
                    exchange_offset = K.alloc_local([1], "int32")
                    K.assign(exchange_offset[0], exchange_i * 32 * 4 + lane_idx * 4)
                    p_exchange_tmp = K.alloc_local((4,), "float32")
                    p_exchange_tmp_u32 = p_exchange_tmp.view("uint32")
                    K.ptx.ld.shared.v4.u32(
                        p_exchange_tmp_u32[0],
                        p_exchange_tmp_u32[1],
                        p_exchange_tmp_u32[2],
                        p_exchange_tmp_u32[3],
                        p_exchange_buf.ptr_to([warp_idx, exchange_offset[0]]),
                    )
                    p_pair0 = K.Bind(K.cuda.make_float2(p[exchange_i * 4], p[exchange_i * 4 + 1]))
                    peer_pair0 = K.Bind(K.cuda.make_float2(p_exchange_tmp[0], p_exchange_tmp[1]))
                    K.ptx.add.rn.f32x2(p_add_pair0[0], p_pair0, peer_pair0)
                    K.ptx.mov.b32(p[exchange_i * 4], K.cuda.float2_x(p_add_pair0[0]))
                    K.ptx.mov.b32(p[exchange_i * 4 + 1], K.cuda.float2_y(p_add_pair0[0]))
                    p_pair1 = K.Bind(
                        K.cuda.make_float2(p[exchange_i * 4 + 2], p[exchange_i * 4 + 3])
                    )
                    peer_pair1 = K.Bind(K.cuda.make_float2(p_exchange_tmp[2], p_exchange_tmp[3]))
                    K.ptx.add.rn.f32x2(p_add_pair1[0], p_pair1, peer_pair1)
                    K.ptx.mov.b32(p[exchange_i * 4 + 2], K.cuda.float2_x(p_add_pair1[0]))
                    K.ptx.mov.b32(p[exchange_i * 4 + 3], K.cuda.float2_y(p_add_pair1[0]))

                bar_k_valid_free.arrive(cur_buf[0])

                cur_pi_max = K.alloc_local([1], "float32")
                K.assign(cur_pi_max[0], K.float32(-float("inf")))
                for p_i in range(B_TOPK // 2):
                    K.assign(cur_pi_max[0], K.max(cur_pi_max[0], p[p_i]))
                K.assign(cur_pi_max[0], cur_pi_max[0] * sm_scale_div_log2)
                K.ptx.st.shared.f32(rowwise_max_buf.ptr_to([idx_in_warpgroup]), cur_pi_max[0])
                K.ptx.bar.sync(K.uint32(BAR_WG0_SYNC), 128)
                peer_pi_max = K.alloc_local([1], "float32")
                K.ptx.ld.shared.f32(
                    peer_pi_max[0], rowwise_max_buf.ptr_to([K.bitwise_xor(idx_in_warpgroup, 64)])
                )
                K.assign(cur_pi_max[0], K.max(cur_pi_max[0], peer_pi_max[0]))
                K.assign(real_mi[0], K.max(real_mi[0], cur_pi_max[0]))
                # G3: the warp collective lands in a local *before* the guard
                # that reads it, which is where the original's typed
                # declaration puts it too (orig:776-778).
                should_scale_o = K.alloc_local([1], "bool")
                K.assign(
                    should_scale_o[0],
                    (K.cuda.any_sync(K.uint32(0xFFFFFFFF), cur_pi_max[0] - mi[0] > 6.0) != 0),
                )
                new_max = K.alloc_local([1], "float32")
                scale_for_old = K.alloc_local([1], "float32")
                with K.If(K.Not(should_scale_o[0])):
                    with K.Then():
                        K.assign(scale_for_old[0], K.float32(1.0))
                        K.assign(new_max[0], mi[0])
                    with K.Else():
                        K.assign(new_max[0], K.max(cur_pi_max[0], mi[0]))
                        K.ptx.ex2.approx.ftz.f32(scale_for_old[0], mi[0] - new_max[0])
                K.assign(mi[0], new_max[0])

                # Each warpgroup thread owns B_TOPK/2 consecutive bf16 values.
                s_frag = K.alloc_local((B_TOPK // 2,), "bfloat16")
                s_pack = s_frag.view("uint32")
                cur_sum_pair = K.alloc_local([1], "uint64")
                K.assign(cur_sum_pair[0], K.cuda.make_float2(K.float32(0.0), K.float32(0.0)))
                neg_new_max_pair = K.Bind(K.cuda.make_float2(-new_max[0], -new_max[0]))
                scale_pair = K.Bind(K.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2))
                fma_pair = K.alloc_local([1], "uint64")
                for s_i in range((B_TOPK // 2) // 2):
                    p_pair = K.Bind(K.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1]))
                    K.ptx.fma.rn.f32x2(fma_pair[0], p_pair, scale_pair, neg_new_max_pair)
                    s_x = K.alloc_local([1], "float32")
                    s_y = K.alloc_local([1], "float32")
                    K.ptx.ex2.approx.ftz.f32(s_x[0], K.cuda.float2_x(fma_pair[0]))
                    K.ptx.ex2.approx.ftz.f32(s_y[0], K.cuda.float2_y(fma_pair[0]))
                    s_pair = K.Bind(K.cuda.make_float2(s_x[0], s_y[0]))
                    K.ptx.add.rn.f32x2(cur_sum_pair[0], cur_sum_pair[0], s_pair)
                    K.ptx.mov.b32(s_pack[s_i], K.cuda.float22bfloat162_rn(s_x[0], s_y[0]))
                cur_sum = K.Bind(
                    K.cuda.float2_x(cur_sum_pair[0]) + K.cuda.float2_y(cur_sum_pair[0])
                )
                li_tmp = K.alloc_local([1], "float32")
                K.ptx.fma.rn.f32(li_tmp[0], li[0], scale_for_old[0], cur_sum)
                K.assign(li[0], li_tmp[0])

                with K.If(k > 0), K.Then():
                    prev_buf = K.alloc_local([1], "int32")
                    K.assign(prev_buf[0], ring_mod3(k - 1))
                    prev_phase = K.alloc_local([1], "int32")
                    K.assign(prev_phase[0], ring_phase_parity(k - 1))
                    pv_wait_token = iket_range("h64-pv-wait")
                    bar_sv_done.wait(prev_buf[0], prev_phase[0])
                    K.cuda.iket.range_end(pv_wait_token[0])

                # On the first iteration s_smem_gemm aliases q_rope, which was
                # read by an async TCGEN05 copy. On later iterations it is the
                # completed SxV MMA stage. Order either async read before the
                # generic S write below.
                K.ptx.fence.proxy.async_.shared__cta()

                # orig:831-845 S store (vectorized by the reg copy path).
                s_base = K.Bind(idx_in_warpgroup // 64 * 2048 + idx_in_warpgroup % 64 * 8)
                s_words = s_frag.view("uint32")
                for s_store_i in range(4):
                    s_ptr = K.Bind(
                        K.ptr_byte_offset(
                            s_smem_gemm.ptr_to([0, 0]),
                            (s_base + s_store_i * 512) * BF16_BYTES,
                            "bfloat16",
                        )
                    )
                    s_word = s_store_i * 4
                    K.ptx.st.shared.v4.u32(
                        s_ptr,
                        s_words[s_word],
                        s_words[s_word + 1],
                        s_words[s_word + 2],
                        s_words[s_word + 3],
                    )
                with K.If((k > 0) & should_scale_o[0]), K.Then():
                    K.ptx.tcgen05.fence__after_thread_sync()
                    # CUDA common_subroutine.h:147-168 rescale_O.
                    o_rescale = K.alloc_local((32,), "float32")
                    for chunk_idx in range((D_V // 2) // 32):
                        tmem_load(
                            o_rescale,
                            K.cuda.get_tmem_addr(K.uint32(o_tmem_col), 0, chunk_idx * 32),
                            32,
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        for scale_i in range(32 // 2):
                            mul_f32x2(o_rescale, scale_i * 2, scale_for_old[0])
                        tmem_store(
                            o_rescale, K.cuda.get_tmem_addr(K.uint32(o_tmem_col), 0, chunk_idx * 32)
                        )
                        K.ptx.tcgen05.wait__st.sync.aligned()
                    K.ptx.tcgen05.fence__before_thread_sync()

                K.ptx.fence.proxy.async_.shared__cta()
                bar_so_ready.arrive(0)
                K.cuda.iket.range_end(softmax_token[0])

            # orig:874-1007. Epilogue scalar exchange, O TMEM readback, output
            # scaling/bf16 staging, and the two elected-warp O TMA stores.
            epilogue_token = iket_range("h64-output")
            with K.If(real_mi[0] == K.float32(-float("inf"))), K.Then():
                K.assign(li[0], K.float32(0.0))
                K.assign(mi[0], K.float32(-float("inf")))

            K.ptx.st.shared.f32(rowwise_li_buf.ptr_to([idx_in_warpgroup]), li[0])
            K.ptx.bar.sync(K.uint32(BAR_WG0_SYNC), 128)
            peer_li = K.alloc_local([1], "float32")
            K.ptx.ld.shared.f32(
                peer_li[0], rowwise_li_buf.ptr_to([K.bitwise_xor(idx_in_warpgroup, 64)])
            )
            K.assign(li[0], li[0] + peer_li[0])

            with K.If(idx_in_warpgroup < B_H), K.Then():
                cur_lse = K.alloc_local([1], "float32")
                cur_lse_log = K.Bind(K.log(li[0]))
                K.ptx.fma.rn.f32(cur_lse[0], mi[0], K.float32(LN_2), cur_lse_log)
                K.assign(
                    cur_lse[0],
                    K.if_then_else(
                        cur_lse[0] == K.float32(-float("inf")), K.float32(float("inf")), cur_lse[0]
                    ),
                )
                logits_offset = s_q_idx * h_q + idx_in_warpgroup
                K.ptx.st.global_.f32(
                    max_logits.ptr_to([logits_offset // h_q, logits_offset % h_q]),
                    real_mi[0] * K.float32(LN_2),
                )
                K.ptx.st.global_.f32(
                    lse.ptr_to([logits_offset // h_q, logits_offset % h_q]), cur_lse[0]
                )

            last_k = K.alloc_local([1], "int32")
            K.assign(last_k[0], K.int32(num_k_blocks_buf[0] - 1))
            last_buf = K.alloc_local([1], "int32")
            K.assign(last_buf[0], ring_mod3(last_k[0]))
            last_phase = K.alloc_local([1], "int32")
            K.assign(last_phase[0], ring_phase_parity(last_k[0]))
            bar_sv_done.wait(last_buf[0], last_phase[0])
            K.ptx.tcgen05.fence__after_thread_sync()
            # bar_sv_done makes the final TCGEN05 read complete; cross back
            # from the async proxy before o_smem aliases and overwrites its K
            # stage.
            K.ptx.fence.proxy.async_.shared__cta()

            if have_attn_sink:
                attn_sink_log2 = K.Bind(
                    K.cuda.ldg(attn_sink.ptr_to([idx_in_warpgroup % B_H]), "float32") * LOG_2_E
                )
            else:
                attn_sink_log2 = K.float32(-float("inf"))
            sink_exp = K.alloc_local([1], "float32")
            K.ptx.ex2.approx.ftz.f32(sink_exp[0], attn_sink_log2 - mi[0])
            output_scale = K.alloc_local([1], "float32")
            K.assign(output_scale[0], K.cuda.fdividef(K.float32(1.0), li[0] + sink_exp[0]))

            o_epi = K.alloc_local((64,), "float32")
            o_epi_bf16 = K.alloc_local((64,), "bfloat16")
            # G3 again, and this one is the FA4 §6.5 shape exactly: K.Bind is
            # the traced spelling of `K.let`, and binding the collective
            # materialises it before the guard below reads it.
            have_valid_indices = K.Bind(K.cuda.any_sync(K.uint32(0xFFFFFFFF), li[0] != 0.0) != 0)
            with K.If(K.Not(have_valid_indices)), K.Then():
                for o_zero_i in range(64):
                    K.ptx.mov.b32(o_epi[o_zero_i], K.float32(0.0))
                K.assign(output_scale[0], K.float32(1.0))
            for epi_c in range(2):
                for epi_k in range((D_V // 4) // 64):
                    with K.If(have_valid_indices), K.Then():
                        # orig:927-935, CUDA phase1.cuh:314-317 TMEM O load/fence.
                        tmem_load(
                            o_epi,
                            K.cuda.get_tmem_addr(K.uint32(o_tmem_col), 0, epi_c * 128 + epi_k * 64),
                            64,
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                    for scale_i in range(64 // 2):
                        mul_f32x2(o_epi, scale_i * 2, output_scale[0])
                    for cast_i in range(64 // 2):
                        cast_f32x2_bf16x2(o_epi_bf16, o_epi, cast_i * 2)
                    o_epi_words = o_epi_bf16.view("uint32")
                    for o_store_i in range(8):
                        s_off = K.Bind(
                            (epi_k // 2 + epi_c) % 2 * 16384
                            + idx_in_warpgroup // 64 * 8192
                            + epi_k % 2 * 4096
                            + idx_in_warpgroup % 64 * 64
                            + K.bitwise_xor(
                                o_store_i * 8,
                                K.shift_left(
                                    K.bitwise_and(
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
                        s_ptr = K.Bind(
                            K.ptr_byte_offset(o_smem.ptr_to([0, 0]), s_off * BF16_BYTES, "bfloat16")
                        )
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
                            # orig:976-988, CUDA phase1.cuh:335-342 first half.
                            K.ptx[TMA_S2G_2D](
                                K.address_of(out_part0_tensormap),
                                K.cast(epi_c * 256 + epi_k * 64, "int32"),
                                K.cast(s_q_idx * B_H, "int32"),
                                K.ptr_byte_offset(
                                    o_smem.ptr_to([0, 0]),
                                    (epi_c * 256 + epi_k * 64) * B_H * BF16_BYTES,
                                    "bfloat16",
                                ),
                            )
                    with K.If(warp_idx == 1), K.Then():
                        with K.If(K.cuda.elect_sync()), K.Then():
                            # orig:991-1003, CUDA phase1.cuh:343-350 second half.
                            K.ptx[TMA_S2G_2D](
                                K.address_of(out_part1_tensormap),
                                K.cast(epi_c * 256 + epi_k * 64 + 128, "int32"),
                                K.cast(s_q_idx * B_H, "int32"),
                                K.ptr_byte_offset(
                                    o_smem.ptr_to([0, 0]),
                                    (epi_c * 256 + epi_k * 64 + 128) * B_H * BF16_BYTES,
                                    "bfloat16",
                                ),
                            )

            with K.If(warp_idx == 0), K.Then():
                K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(K.uint32(0), K.uint32(512))
            K.cuda.iket.range_end(epilogue_token[0])

        def kv_nope_producer():
            """warps 4-7 — orig:1009-1110. KV NoPE `tile::gather4` producer."""
            kv_nope_token = iket_range("h64-kv-nope-load")
            wg1_warp_idx = K.Bind(warp_idx - 4)
            # This warp's 16 interleaved NoPE rows: split the 64-row dim into
            # (stripe, warp, row) and pick this warp, merging stripe x row.
            with K.If(K.cuda.elect_sync()), K.Then():
                with K.serial(num_k_blocks_buf[0], unroll=False) as k:
                    selected_idx = K.alloc_local((WG1_ROWS_PER_WARP, 4), "int32")
                    max_indices = K.alloc_local([1], "int32")
                    K.assign(max_indices[0], K.int32(-1))
                    min_indices = K.alloc_local([1], "int32")
                    K.assign(min_indices[0], K.int32(s_kv))
                    # This warp's 16 indices from the (local_row, warp, j) split.
                    selected_words = selected_idx.view(16).view("uint32")
                    for selected_load_i in range(4):
                        selected_word = selected_load_i * 4
                        K.ptx.ld.global_.nc.v4.u32(
                            selected_words[selected_word],
                            selected_words[selected_word + 1],
                            selected_words[selected_word + 2],
                            selected_words[selected_word + 3],
                            indices.ptr_to(
                                [
                                    g_indices_base
                                    + k * B_TOPK
                                    + wg1_warp_idx * 4
                                    + selected_load_i * 16
                                ]
                            ),
                        )
                    for local_row in range(WG1_ROWS_PER_WARP):
                        for j in range(4):
                            idx = K.Bind(selected_idx[local_row, j])
                            K.assign(max_indices[0], K.max(max_indices[0], idx))
                            K.assign(min_indices[0], K.min(min_indices[0], idx))

                    is_all_rows_invalid = K.Bind((min_indices[0] == s_kv) | (max_indices[0] == -1))
                    should_skip_tma = K.Bind(is_all_rows_invalid & (k >= NUM_BUFS))

                    with K.If(k == 2), K.Then():
                        bar_prologue_utccp_nope.wait(0, 0)

                    cur_buf = K.alloc_local([1], "int32")
                    K.assign(cur_buf[0], ring_mod3(k))
                    cur_phase = K.alloc_local([1], "int32")
                    K.assign(cur_phase[0], ring_phase_parity(k))
                    bar_sv_done.wait(cur_buf[0], K.bitwise_xor(cur_phase[0], K.int32(1)))

                    with K.If(K.Not(should_skip_tma)):
                        with K.Then():
                            for row_group in range(WG1_ROWS_PER_WARP):
                                for col_atom in range((D_V // 2) // 64):
                                    dst_part0_offset = K.Bind(
                                        (
                                            cur_buf[0] * B_TOPK * D_V
                                            + col_atom * 64 * B_TOPK
                                            + (wg1_warp_idx * 4 + row_group * 16) * 64
                                        )
                                        * BF16_BYTES
                                    )
                                    K.ptx[TMA_GATHER4_2D_CACHE](
                                        K.ptr_byte_offset(
                                            k_nope.ptr_to([0, 0, 0]), dst_part0_offset, "bfloat16"
                                        ),
                                        K.address_of(kv_part0_tensormap),
                                        K.cast(col_atom * 64, "int32"),
                                        selected_idx[row_group, 0],
                                        selected_idx[row_group, 1],
                                        selected_idx[row_group, 2],
                                        selected_idx[row_group, 3],
                                        K.cuda.cvta_generic_to_shared(
                                            bar_kv_nope_ready_part0.ptr_to([cur_buf[0]])
                                        ),
                                        K.uint64(KV_TMA_CACHE_HINT),
                                    )
                            for row_group in range(WG1_ROWS_PER_WARP):
                                for col_atom in range((D_V // 2) // 64):
                                    dst_part1_offset = K.Bind(
                                        (
                                            cur_buf[0] * B_TOPK * D_V
                                            + (D_V // 2 + col_atom * 64) * B_TOPK
                                            + (wg1_warp_idx * 4 + row_group * 16) * 64
                                        )
                                        * BF16_BYTES
                                    )
                                    K.ptx[TMA_GATHER4_2D_CACHE](
                                        K.ptr_byte_offset(
                                            k_nope.ptr_to([0, 0, 0]), dst_part1_offset, "bfloat16"
                                        ),
                                        K.address_of(kv_part1_tensormap),
                                        K.cast(col_atom * 64, "int32"),
                                        selected_idx[row_group, 0],
                                        selected_idx[row_group, 1],
                                        selected_idx[row_group, 2],
                                        selected_idx[row_group, 3],
                                        K.cuda.cvta_generic_to_shared(
                                            bar_kv_nope_ready_part1.ptr_to([cur_buf[0]])
                                        ),
                                        K.uint64(KV_TMA_CACHE_HINT),
                                    )
                        with K.Else():
                            # orig:1103, unannotated: a declared uint local, and
                            # the original re-casts it at both use sites.
                            tx_bytes = K.alloc_local([1], "uint32")
                            K.assign(
                                tx_bytes[0],
                                K.uint32(WG1_ROWS_PER_WARP * 4 * (D_V // 2) * BF16_BYTES),
                            )
                            K.ptx.mbarrier.complete_tx.relaxed.cluster.shared__cluster.b64(
                                bar_kv_nope_ready_part0.ptr_to([cur_buf[0]]), K.uint32(tx_bytes[0])
                            )
                            K.ptx.mbarrier.complete_tx.relaxed.cluster.shared__cluster.b64(
                                bar_kv_nope_ready_part1.ptr_to([cur_buf[0]]), K.uint32(tx_bytes[0])
                            )
            K.cuda.iket.range_end(kv_nope_token[0])

        def mma_issuer():
            """warp 8 — orig:1115-1464. All four `tcgen05.mma.ws` chains.

            The chains are hand-written closures, not `K.idioms.mma_chain`:
            the ring stage reaches the MMA as a *runtime* 16-byte descriptor
            offset (`+ cur_buf * 32768`), which is the stage-in-offset model
            design-cluster ruling (a) put outside the chain helper's scope.
            """
            mma_token = iket_range("h64-qk-pv-issue")
            with K.If(K.cuda.elect_sync()), K.Then():
                if have_rope:
                    bar_prologue_q_rope.arrive(0, tx_count=B_H * (d_qk - D_V) * BF16_BYTES)
                    bar_prologue_q_rope.wait(0, 0)
                    K.ptx.tcgen05.fence__after_thread_sync()
                    for q_rope_flat in range(2):
                        q_rope_src = K.Bind(
                            K.ptr_byte_offset(
                                q_rope.ptr_to([0, 0]), q_rope_flat % 2 * 2 * 16, "bfloat16"
                            )
                        )
                        K.ptx[TCGEN_CP_128X256](
                            K.cast(q_rope_tmem_col + q_rope_flat % 2 * 8, "uint32"),
                            replace_smem_desc_addr(q_rope_cp_desc[0], q_rope_src),
                        )
                    commit(bar_prologue_utccp_rope, 0)

                bar_prologue_q_nope.arrive(0, tx_count=B_H * D_V * BF16_BYTES)
                bar_prologue_q_nope.wait(0, 0)
                K.ptx.tcgen05.fence__after_thread_sync()
                q_nope_cp_desc = K.alloc_local([1], "uint64")
                K.cuda.tcgen05.encode_matrix_descriptor(
                    K.address_of(q_nope_cp_desc[0]),
                    K.reinterpret(K.handle().ty, K.uint64(0)),
                    1,
                    64,
                    3,
                )
                for q_nope_flat in range(16):
                    q_nope_src = K.Bind(
                        K.ptr_byte_offset(
                            q_nope.ptr_to([0, 0]),
                            (q_nope_flat % 4 * 1024 + q_nope_flat // 4 % 4 * 2) * 16,
                            "bfloat16",
                        )
                    )
                    K.ptx[TCGEN_CP_128X256](
                        K.cast(
                            q_nope_tmem_col + (q_nope_flat % 4 * 32 + q_nope_flat // 4 % 4 * 8),
                            "uint32",
                        ),
                        replace_smem_desc_addr(q_nope_cp_desc[0], q_nope_src),
                    )
                commit(bar_prologue_utccp_nope, 0)

                if have_rope:
                    bar_prologue_utccp_rope.wait(0, 0)

                with K.serial(num_k_blocks_buf[0] + 1, unroll=False) as k:
                    with K.If(k < num_k_blocks_buf[0]), K.Then():
                        cur_buf = K.alloc_local([1], "int32")
                        K.assign(cur_buf[0], ring_mod3(k))
                        cur_phase = K.alloc_local([1], "int32")
                        K.assign(cur_phase[0], ring_phase_parity(k))
                        bar_p_free.wait(0, K.bitwise_xor(K.bitwise_and(k, K.int32(1)), K.int32(1)))
                        K.ptx.tcgen05.fence__after_thread_sync()

                        if have_rope:
                            bar_kv_rope_ready.wait(0, K.bitwise_and(k, K.int32(1)))
                            K.ptx.tcgen05.fence__after_thread_sync()
                            # orig:1183-1231, CUDA phase1.cuh:489 QRoPE x KRoPE.
                            K.assign(mma_p_accumulate[0], K.uint32(0))
                            if local_mma_desc:
                                qk_rope_desc_local = K.SmemDescriptor()
                                qk_rope_desc_local.init(
                                    k_rope.ptr_to([0, 0]), ldo=0, sdo=32, swizzle=2
                                )
                                _qk_rope_chain(qk_rope_desc_local)
                            else:
                                _qk_rope_chain(qk_rope_desc)
                            commit(bar_qk_rope_done, 0)

                        with K.If(k == 0), K.Then():
                            bar_prologue_utccp_nope.wait(0, 0)

                        for kv_nope_part_idx in range(2):
                            tx_bytes = B_TOPK * (D_V // 2) * BF16_BYTES
                            if kv_nope_part_idx == 0:
                                bar_kv_nope_ready_part0.arrive(cur_buf[0], tx_count=tx_bytes)
                                bar_kv_nope_ready_part0.wait(cur_buf[0], cur_phase[0])
                            else:
                                bar_kv_nope_ready_part1.arrive(cur_buf[0], tx_count=tx_bytes)
                                bar_kv_nope_ready_part1.wait(cur_buf[0], cur_phase[0])
                            K.ptx.tcgen05.fence__after_thread_sync()
                            # orig:1245-1317, CUDA phase1.cuh:505-506 QNoPE x KNoPE.
                            clear_nope_accum = (not have_rope) and (kv_nope_part_idx == 0)
                            K.assign(
                                mma_p_accumulate[0],
                                K.uint32(0) if clear_nope_accum else K.uint32(1),
                            )
                            if local_mma_desc:
                                qk_nope_desc_local = K.SmemDescriptor()
                                qk_nope_desc_local.init(
                                    k_nope.ptr_to([0, 0, 0]), ldo=1024, sdo=64, swizzle=3
                                )
                                _qk_nope_chain(qk_nope_desc_local, cur_buf[0], kv_nope_part_idx)
                            else:
                                _qk_nope_chain(qk_nope_desc, cur_buf[0], kv_nope_part_idx)
                        commit(bar_qk_nope_done, cur_buf[0])

                    with K.If(k > 0), K.Then():
                        cur_buf_prev = K.alloc_local([1], "int32")
                        K.assign(cur_buf_prev[0], ring_mod3(k - 1))
                        bar_so_ready.wait(0, K.bitwise_and(k - 1, K.int32(1)))
                        K.ptx.tcgen05.fence__after_thread_sync()
                        # orig:1328-1457, CUDA phase1.cuh:521-523 S(i-1) x V(i-1).
                        K.assign(
                            mma_o_accumulate[0], K.if_then_else(k == 1, K.uint32(0), K.uint32(1))
                        )
                        if local_mma_desc:
                            pv_b_lo_desc_local = K.SmemDescriptor()
                            pv_b_lo_desc_local.init(
                                k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3
                            )
                            pv_a_lo_desc_local = K.SmemDescriptor()
                            pv_a_lo_desc_local.init(
                                s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0
                            )
                            _pv_chain(
                                pv_a_lo_desc_local, pv_b_lo_desc_local, cur_buf_prev[0], half=0
                            )
                            pv_b_hi_desc_local = K.SmemDescriptor()
                            pv_b_hi_desc_local.init(
                                k_nope.ptr_to([0, 0, 0]), ldo=512, sdo=64, swizzle=3
                            )
                            pv_a_hi_desc_local = K.SmemDescriptor()
                            pv_a_hi_desc_local.init(
                                s_smem_gemm.ptr_to([0, 0]), ldo=64, sdo=8, swizzle=0
                            )
                            _pv_chain(
                                pv_a_hi_desc_local, pv_b_hi_desc_local, cur_buf_prev[0], half=1
                            )
                        else:
                            _pv_chain(pv_a_lo_desc, pv_b_lo_desc, cur_buf_prev[0], half=0)
                            _pv_chain(pv_a_hi_desc, pv_b_hi_desc, cur_buf_prev[0], half=1)
                        K.assign(mma_o_accumulate[0], K.uint32(1))
                        commit(bar_sv_done, cur_buf_prev[0])
            K.cuda.iket.range_end(mma_token[0])

        # --- the four MMA chains, as closures (ruling (a)) -----------------
        # Each is exactly the original's triple-nested `K.unroll` walk with the
        # descriptor operand parameterised, so the local-encode and hoisted
        # arms of `local_mma_desc` issue *the same* instruction sequence and
        # differ only in where the descriptor was encoded.

        def _qk_rope_chain(desc):
            """orig:1190-1226."""
            for mma_mi in range(1):
                for mma_ni in range(1):
                    for mma_ki in range(2):
                        K.ptx[MMA_WS_F16](
                            K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                            K.cast(q_rope_tmem_col + mma_ki * 8, "uint32"),
                            desc.add_16B_offset((mma_ni * 4096 + mma_ki * 16) // 8),
                            K.uint32(ID_QK),
                            True if mma_ki != 0 else K.cast(mma_p_accumulate[0], "bool"),
                            K.uint64(0),
                        )

        def _qk_nope_chain(desc, cur_buf_v, kv_nope_part_idx):
            """orig:1255-1317."""
            for mma_mi in range(1):
                for mma_ni in range(1):
                    for mma_ki in range(8):
                        qk_nope_offset = K.Bind(
                            (
                                mma_ki // 1024 * 32768
                                + mma_ni * 32768
                                + cur_buf_v * 32768
                                + kv_nope_part_idx % 2 * 16384
                                + mma_ki % 8 // 4 * 8192
                                + mma_ki % 1024 // 8 * 64
                                + mma_ki % 4 * 16
                            )
                            // 8
                        )
                        K.ptx[MMA_WS_F16](
                            K.cast(tmem_p_col + mma_ni * 64, "uint32"),
                            K.cast(q_nope_tmem_col + kv_nope_part_idx * 64 + mma_ki * 8, "uint32"),
                            desc.add_16B_offset(qk_nope_offset),
                            K.uint32(ID_QK),
                            True if mma_ki != 0 else K.cast(mma_p_accumulate[0], "bool"),
                            K.uint64(0),
                        )

        def _pv_chain(a_desc, b_desc, cur_buf_prev_v, half):
            """orig:1339-1401 (local arm) / orig:1403-1457 (hoisted arm).

            `half` selects the lo (0) / hi (1) 256-column output group: the hi
            chain adds 128 to the tmem column and 16384 to the B operand's byte
            offset, and is otherwise the same walk.
            """
            for mma_mi in range(1):
                for mma_ni in range(1):
                    for mma_ki in range(4):
                        K.ptx[MMA_WS_F16](
                            K.cast(o_tmem_col + half * 128 + mma_ni * 128, "uint32"),
                            a_desc.add_16B_offset(
                                (mma_ki % 4 * 1024 + mma_mi * 512 + mma_ki // 4 * 8) // 8
                            ),
                            b_desc.add_16B_offset(
                                (
                                    (mma_ki * 16 + mma_ni) // 64 * 32768
                                    + cur_buf_prev_v * 32768
                                    + (mma_ki * 16 + mma_ni) % 64 * 64
                                    + half * 16384
                                )
                                // 8
                            ),
                            K.uint32(ID_PV),
                            True if mma_ki != 0 else K.cast(mma_o_accumulate[0], "bool"),
                            K.uint64(0),
                        )

        def valid_mask_producer():
            """warp 9 — orig:1466-1500, CUDA common_subroutine.h:14-44."""
            valid_mask_token = iket_range("h64-valid-mask")
            with K.If(lane_idx < B_TOPK // 8), K.Then():
                lane_indices = K.alloc_local((8,), "int32")
                with K.serial(num_k_blocks_buf[0], unroll=False) as k:
                    abs_pos_start = K.Bind(k * B_TOPK)
                    row_base = K.Bind(g_indices_base + k * B_TOPK + lane_idx * 8)
                    lane_index_words = lane_indices.view("uint32")
                    K.ptx[LD_INDICES_V8](
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
                    is_ks_valid_mask = K.alloc_local([1], "int8")
                    K.assign(
                        is_ks_valid_mask[0],
                        pack_valid_mask8(
                            lane_indices, abs_pos_start, lane_idx, topk_len_buf[0], s_kv
                        ),
                    )

                    cur_buf = K.alloc_local([1], "int32")
                    K.assign(cur_buf[0], ring_mod3(k))
                    cur_phase = K.alloc_local([1], "int32")
                    K.assign(cur_phase[0], ring_phase_parity(k))
                    bar_k_valid_free.wait(cur_buf[0], K.bitwise_xor(cur_phase[0], K.int32(1)))
                    K.ptx.st.shared.b8(
                        is_k_valid.ptr_to([cur_buf[0], lane_idx]),
                        K.reinterpret("uint8", is_ks_valid_mask[0]),
                    )
                    bar_k_valid_ready.arrive(cur_buf[0])
            K.cuda.iket.range_end(valid_mask_token[0])

        def k_rope_loader():
            """warps 10-11 — orig:1502-1537. Only live when `have_rope`."""
            kv_rope_token = iket_range("h64-k-rope-load")
            if have_rope:
                thread_idx = K.Bind((warp_idx - 10) * 32 + lane_idx)
                group_idx = K.Bind(thread_idx // 8)
                idx_in_group = K.Bind(thread_idx % 8)
                with K.serial(num_k_blocks_buf[0], unroll=False) as k:
                    rope_indices = K.alloc_local(((B_TOPK // (64 // 8)),), "int32")
                    for local_row in range(B_TOPK // (64 // 8)):
                        K.ptx.mov.b32(
                            rope_indices[local_row],
                            K.cuda.ldg(
                                indices.ptr_to(
                                    [
                                        g_indices_base
                                        + k * B_TOPK
                                        + group_idx
                                        + local_row * (64 // 8)
                                    ]
                                ),
                                "int32",
                            ),
                        )
                    bar_qk_rope_done.wait(
                        0, K.bitwise_xor(K.bitwise_and(k, K.int32(1)), K.int32(1))
                    )
                    for local_row in range(B_TOPK // (64 // 8)):
                        # orig:1521, unannotated: a declared one-element local.
                        index = K.alloc_local([1], "int32")
                        K.assign(index[0], rope_indices[local_row])
                        is_valid_index = K.Bind((index[0] >= 0) & (index[0] < s_kv))
                        kv_off = K.Bind(index[0] * stride_kv_s_kv + D_V + idx_in_group * 8)
                        K.ptx[CP_ASYNC_CG_16B](
                            k_rope.ptr_to([group_idx + local_row * (64 // 8), idx_in_group * 8]),
                            kv.ptr_to([kv_off]),
                            K.int32(16),
                            K.if_then_else(is_valid_index, K.uint32(16), K.uint32(0)),
                        )
                    K.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(
                        bar_kv_rope_ready.ptr_to([0])
                    )
            K.cuda.iket.range_end(kv_rope_token[0])

        # ---- roles — orig:669-1537 ---------------------------------------
        # Five roles partitioning warps 0..11, entered back-to-back with
        # nothing at CTA scope between them so K.specialize's chain_dispatch
        # folds them into one else-if chain. No `regs=` anywhere: the original
        # issues no `setmaxnreg` at all (scout §2), so every role leaves the
        # partition to ptxas exactly as today, and Specialize.finalize accounts
        # the unset roles at `entry_regs`.
        sp = K.specialize(chain_dispatch=True)
        r_softmax = sp.role("softmax", warps=[0, 1, 2, 3])
        r_kv_nope = sp.role("kv_nope", warps=[4, 5, 6, 7])
        r_mma = sp.role("mma", warps=[8])
        r_valid_mask = sp.role("valid_mask", warps=[9])
        r_k_rope = sp.role("k_rope", warps=[10, 11])

        with r_softmax:
            softmax_and_epilogue()
        with r_kv_nope:
            kv_nope_producer()
        with r_mma:
            mma_issuer()
        with r_valid_mask:
            valid_mask_producer()
        with r_k_rope:
            k_rope_loader()

    return sparse_flashmla_prefill_head64_phase1_kern


def _make_kernel_for(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    return make_kernel(
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


def get_kernel(**kwargs: Any):
    return (
        _make_kernel_for(**kwargs)
        .func.with_attr("global_symbol", KERNEL_META["name"])
        .with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
    )


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


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


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
        raise SkipTest("CUDA is required for sparse FlashMLA phase1 benchmark")

    from tirx_kernels.runner import bench

    ex = executable

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


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
