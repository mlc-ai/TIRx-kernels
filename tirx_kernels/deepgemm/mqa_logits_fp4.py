# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of DeepGEMM's MQA logits kernel, FP4 variant.

Upstream source: deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh.
"""

import ctypes
import os
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
import tvm

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_TEST_DIFF_THRESHOLD = 5e-6
_COMPILE_CACHE_NAMESPACE = "deepgemm.mqa_logits_fp4.compile"


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(192)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(dtype, rank, tensor, dims, strides, box, swizzle):
    descriptor = _AlignedTensorMap()
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        descriptor.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *(1 for _ in range(rank)),
        0,
        swizzle,
        2,
        0,
    )
    return descriptor


def _build_tirx_tensor_maps(data: dict[str, Any]):
    config = data["config"]
    q_fp4, sf_q = data["q_in"]
    kv_fp4, sf_kv = data["kv_in"]
    packed = config.head_dim // 2
    q = q_fp4.reshape(config.seq_len * config.num_heads, packed).contiguous().view(torch.uint8)
    kv = kv_fp4.contiguous().view(torch.uint8)
    sf_q = sf_q.contiguous().view(torch.uint32)
    sf_kv = sf_kv.contiguous().view(torch.uint32).view(1, -1)
    maps = (
        _encode_tensor_map("uint32", 1, sf_kv, (config.seq_len_kv,), (), (config.block_kv,), 0),
        _encode_tensor_map(
            "uint8",
            2,
            kv,
            (packed, config.seq_len_kv),
            (packed,),
            (packed, config.block_kv),
            K.SW64B.value,
        ),
        _encode_tensor_map(
            "float32",
            2,
            data["weights"],
            (config.num_heads, config.seq_len),
            (config.num_heads * 4,),
            (config.num_heads, config.block_q),
            0,
        ),
        _encode_tensor_map(
            "uint32",
            2,
            sf_q,
            (config.num_heads, config.seq_len),
            (config.num_heads * 4,),
            (config.num_heads, config.block_q),
            0,
        ),
        _encode_tensor_map(
            "uint8",
            2,
            q,
            (packed, config.seq_len * config.num_heads),
            (packed,),
            (packed, config.block_q * config.num_heads),
            K.SW64B.value,
        ),
    )
    return maps


@dataclass(frozen=True)
class MQALogitsConfig:
    seq_len: int = 32
    seq_len_kv: int = 256
    num_heads: int = 64
    head_dim: int = 128
    logits_dtype: str = "float32"
    compressed_logits: bool = False
    disable_cp: bool = True
    seed: int = 0
    num_sms: int = 148
    logits_stride_override: int | None = None

    @property
    def block_q(self) -> int:
        return 128 // self.num_heads

    @property
    def block_kv(self) -> int:
        return 256

    @property
    def max_seqlen_k(self) -> int:
        return 0 if not self.compressed_logits else self.seq_len_kv

    @property
    def aligned_seq_len(self) -> int:
        return _align_up(self.seq_len, self.block_q)

    @property
    def logits_stride(self) -> int:
        if self.logits_stride_override is not None:
            return self.logits_stride_override
        if self.compressed_logits:
            return _align_up(self.max_seqlen_k, self.block_kv)
        return _align_up(self.seq_len_kv + self.block_kv, 8)

    def validate(self) -> None:
        if self.num_heads not in (32, 64):
            raise ValueError("num_heads must be 32 or 64")
        if self.head_dim != 128:
            raise ValueError("head_dim must be 128 for the SM100 FP4 MQA logits kernel")
        if 128 % self.num_heads != 0:
            raise ValueError("128 must be divisible by num_heads")
        if self.seq_len <= 0 or self.seq_len_kv <= 0:
            raise ValueError("sequence lengths must be positive")
        if self.logits_dtype not in ("float32", "bfloat16"):
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if self.num_sms <= 0:
            raise ValueError("num_sms must be positive")
        if self.logits_stride_override is not None and self.logits_stride_override <= 0:
            raise ValueError("logits_stride_override must be positive when provided")
        if not self.disable_cp and (self.seq_len_kv % self.seq_len != 0 or self.seq_len % 2 != 0):
            raise ValueError(
                "CP-style schedule generation requires seq_len_kv % seq_len == 0 and even seq_len"
            )


def _make_config(**kwargs: Any) -> MQALogitsConfig:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = MQALogitsConfig(**kwargs)
    config.validate()
    return config


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _torch_logits_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported logits_dtype: {dtype}")


def _config_label(config: dict[str, Any]) -> str:
    dtype = "f32" if config["logits_dtype"] == "float32" else "bf16"
    mode = "compressed" if config["compressed_logits"] else "dense"
    cp = "nocp" if config["disable_cp"] else "cp"
    return (
        f"s{config['seq_len']}_skv{config['seq_len_kv']}_"
        f"h{config['num_heads']}_d{config['head_dim']}_{dtype}_{mode}_{cp}"
    )


def _make_case(
    *,
    seq_len: int,
    seq_len_kv: int,
    logits_dtype: str,
    compressed_logits: bool,
    disable_cp: bool,
    seed: int,
) -> dict[str, Any]:
    config = {
        "seq_len": seq_len,
        "seq_len_kv": seq_len_kv,
        "num_heads": 64,
        "head_dim": 128,
        "logits_dtype": logits_dtype,
        "compressed_logits": compressed_logits,
        "disable_cp": disable_cp,
        "seed": seed,
    }
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_fp4_mqa_logits",
    "category": "deepgemm",
    "compute_capability": 10,
}

DEEPGEMM_TEST_COVERAGE = [
    _make_case(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        logits_dtype=logits_dtype,
        compressed_logits=compressed_logits,
        disable_cp=disable_cp,
        seed=1000 + seed,
    )
    for seed, (logits_dtype, compressed_logits, seq_len, seq_len_kv, disable_cp) in enumerate(
        (logits_dtype, compressed_logits, seq_len, seq_len_kv, disable_cp)
        for logits_dtype in ("float32", "bfloat16")
        for compressed_logits in (False, True)
        for seq_len in (2048, 4096)
        for seq_len_kv in (4096, 8192)
        for disable_cp in (False, True)
    )
]

CONFIGS = DEEPGEMM_TEST_COVERAGE


def load_deep_gemm_mqa() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM MQA logits runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "fp8_fp4_mqa_logits"):
        raise SkipTest("DeepGEMM MQA logits runtime unavailable: missing fp8_fp4_mqa_logits")
    return module, "installed"


def _generate_ks_ke(config: MQALogitsConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.disable_cp:
        ks = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
        ke = torch.arange(config.seq_len, dtype=torch.int32, device="cuda")
        ke = ke + (config.seq_len_kv - config.seq_len)
        return ks, ke

    chunk_size = config.seq_len // 2
    cp_size = config.seq_len_kv // config.seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
    ke = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    q_f32 = q.float()
    kv_f32 = kv.float()
    mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seq_len_k_start[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seq_len_k_end[:, None]
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q_f32, kv_f32)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_mqa()
    config = _make_config(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 FP4 MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP4 MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    q = torch.randn(
        config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = _generate_ks_ke(config)

    q_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        q.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_in = (
        q_fp4[0].view(config.seq_len, config.num_heads, config.head_dim // 2).contiguous(),
        q_fp4[1].view(config.seq_len, config.num_heads).contiguous(),
    )
    kv_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        kv.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_in = (
        kv_fp4[0].view(config.seq_len_kv, config.head_dim // 2).contiguous(),
        kv_fp4[1].view(config.seq_len_kv).contiguous(),
    )

    q_simulated = deep_gemm.utils.cast_back_from_fp4(
        q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.seq_len, config.num_heads, config.head_dim)
    kv_simulated = deep_gemm.utils.cast_back_from_fp4(
        kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.seq_len_kv, config.head_dim)
    reference = _ref_mqa_logits(
        q_simulated.to(torch.bfloat16), kv_simulated.to(torch.bfloat16), weights, ks, ke
    )
    max_seqlen_k = int((ke - ks).max().item()) if config.compressed_logits else 0
    runtime_config = MQALogitsConfig(
        **{
            **asdict(config),
            "num_sms": int(getattr(deep_gemm, "get_num_sms", lambda: config.num_sms)()),
        }
    )
    return {
        "config": runtime_config,
        "reference_source": source,
        "q": q,
        "kv": kv,
        "q_in": q_in,
        "kv_in": kv_in,
        "weights": weights,
        "cu_seq_len_k_start": ks,
        "cu_seq_len_k_end": ke,
        "max_seqlen_k": max_seqlen_k,
        "reference": reference,
        "deep_gemm": deep_gemm,
    }


def _weighted_relu_reduce(accum, weights, weight_row, num_values):
    """Packed weighted-ReLU reduction over the per-row accumulators.

    relu-like term (x + |x|) accumulates as f32x2 pairs across two
    interleaved sums; the final scalar halves the doubled result.
    """
    regs = K.alloc_local((7,), "uint64")
    scalars = K.alloc_local((3,), "float32")
    sum_0, sum_1 = regs[0], regs[1]
    accum_pair, abs_pair, relu_pair, weight_pair = regs[2], regs[3], regs[4], regs[5]
    total = regs[6]
    abs_lo, abs_hi, result = scalars[0], scalars[1], scalars[2]
    K.ptx.mov.b64(sum_0, K.float32(0), K.float32(0))
    K.ptx.mov.b64(sum_1, K.float32(0), K.float32(0))
    for head_group in range(num_values // 4):
        head = head_group * 4
        K.ptx.mov.b64(accum_pair, accum[head], accum[head + 1])
        K.ptx.abs.f32(abs_lo, accum[head])
        K.ptx.abs.f32(abs_hi, accum[head + 1])
        K.ptx.mov.b64(abs_pair, abs_lo, abs_hi)
        K.ptx.add.rn.f32x2(relu_pair, accum_pair, abs_pair)
        K.ptx.mov.b64(weight_pair, weights[weight_row, head], weights[weight_row, head + 1])
        K.ptx.fma.rn.f32x2(sum_0, relu_pair, weight_pair, sum_0)

        K.ptx.mov.b64(accum_pair, accum[head + 2], accum[head + 3])
        K.ptx.abs.f32(abs_lo, accum[head + 2])
        K.ptx.abs.f32(abs_hi, accum[head + 3])
        K.ptx.mov.b64(abs_pair, abs_lo, abs_hi)
        K.ptx.add.rn.f32x2(relu_pair, accum_pair, abs_pair)
        K.ptx.mov.b64(weight_pair, weights[weight_row, head + 2], weights[weight_row, head + 3])
        K.ptx.fma.rn.f32x2(sum_1, relu_pair, weight_pair, sum_1)
    K.ptx.add.rn.f32x2(total, sum_0, sum_1)
    K.ptx.mov.b64(abs_lo, abs_hi, total)
    K.ptx.add.rn.f32(result, abs_lo, abs_hi)
    K.ptx.mul.rn.f32(result, result, K.float32(0.5))
    return result


def get_kernel(**kwargs: Any):
    config = _make_config(**kwargs)
    num_heads = config.num_heads
    head_dim = config.head_dim
    block_q = config.block_q
    block_kv = config.block_kv
    num_q_stages = 3
    # DeepGEMM pipeline depth for FP4 (e2m1 KV is half-width).
    num_kv_stages = 10
    num_tmem_stages = 3
    num_specialized_threads = 128
    num_math_threads = 256
    num_math_warpgroups = num_math_threads // 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    num_utccp_aligned_elems = 128
    umma_m = 128
    umma_n = block_q * num_heads
    umma_k = 64
    num_sfq = _align_up(block_q * num_heads, num_utccp_aligned_elems)
    num_sfkv = _align_up(block_kv, num_utccp_aligned_elems)
    real_num_sfq = block_q * num_heads
    swizzle_alignment = 8 * (head_dim // 2)
    smem_q_size_per_stage = block_q * num_heads * (head_dim // 2)
    smem_kv_size_per_stage = block_kv * (head_dim // 2)
    smem_sf_kv_size_per_stage = num_sfkv * 4
    smem_weight_size_per_stage = block_q * num_heads * 4
    num_accum_tmem_cols = block_q * num_heads * num_tmem_stages
    num_sfa_tmem_cols = num_sfq // 32
    num_sfb_tmem_cols = num_sfkv // 32
    num_tmem_cols = 32
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 32:
        num_tmem_cols = 64
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 64:
        num_tmem_cols = 128
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 128:
        num_tmem_cols = 256
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 256:
        num_tmem_cols = 512
    tmem_start_col_of_sfq = num_accum_tmem_cols
    tmem_start_col_of_sfkv = num_accum_tmem_cols + num_sfa_tmem_cols
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    TMA_G2S_1D = (
        "cp.async.bulk.tensor.1d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    TMA_G2S_2D = (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    TCGEN05_CP = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    TCGEN05_MMA = "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"
    TCGEN05_LD_X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
    # `__launch_bounds__(kNumThreads, 1)`: without .minnctapersm ptxas ignores
    # the setmaxnreg 56/224 budget (C7508); bf16 runs faster without it. The
    # original attaches this attribute for float32 only (orig:L552-555).
    min_blocks = 1 if config.logits_dtype == "float32" else None

    @K.kernel(warps=num_warps, arch="sm_100a", min_blocks_per_sm=min_blocks, grid=config.num_sms)
    def sm100_fp4_mqa_logits(
        seq_len: K.u32,
        seq_len_kv: K.u32,
        max_seqlen_k: K.u32,
        logits_stride: K.u32,
        cu_seq_len_k_start: K.gptr[K.i32],
        cu_seq_len_k_end: K.gptr[K.i32],
        logits_flat: K.gptr[logits_tir_dtype],
        sf_kv_map: K.TensorMap,
        kv_map: K.TensorMap,
        weights_map: K.TensorMap,
        sf_q_map: K.TensorMap,
        q_map: K.TensorMap,
    ):
        cache_policy_evict_normal = K.uint64(0x1000000000000000)

        def replace_smem_desc_addr(desc, smem_ptr):
            shared_addr = K.cuda.cvta_generic_to_shared(smem_ptr)
            start_addr = K.Cast(
                "uint64", K.bitwise_and(K.shift_right(shared_addr, K.uint32(4)), K.uint32(0x3FFF))
            )
            return K.bitwise_or(K.bitwise_and(desc, K.bitwise_not(K.uint64(0x3FFF))), start_addr)

        def recompute_fp4_smem_desc(smem_ptr):
            # ldo=0, sdo=32, 64B swizzle.  This is the exact uniform descriptor
            # form produced by the former ``smem_desc="recompute"`` dispatch.
            shared_addr = K.cuda.cvta_generic_to_shared(smem_ptr)
            start_addr = K.Cast(
                "uint64", K.bitwise_and(K.shift_right(shared_addr, K.uint32(4)), K.uint32(0x3FFF))
            )
            return K.bitwise_or(K.shift_left(K.uint64(0x80004020), K.uint64(32)), start_addr)

        def emit_sf_transpose(buf, dst, lane, stage_idx, elem_base):
            # DeepGEMM's st.shared.v4 SF transpose, out-of-place into staging
            # (no in-place WAR warp_sync; elect.sync covers the cross-lane barrier).
            # ptx destinations are declared registers the instruction writes into.
            v = K.alloc_local([4], "uint32")
            for i in range(4):
                K.ptx.ld.shared.u32(v[i], buf.ptr_to([stage_idx, elem_base + i * 32 + lane]))
            K.ptx.st.shared.v4.u32(
                dst.ptr_to([stage_idx, elem_base + lane * 4]), v[0], v[1], v[2], v[3]
            )

        sm_idx = K.cta_id()
        thread_idx = K.thread_id()
        warp_idx = K.warp_id()
        warpgroup_idx = K.warpgroup_id([num_warps // 4])
        lane_idx = K.lane_id()
        # Keep the former dispatcher placement: one warp-elected prefetch for
        # each map, issued before any pipeline work.
        for tensor_map in (q_map, sf_q_map, weights_map, kv_map, sf_kv_map):
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(tensor_map))

        # SMEMPool owns the smem offsets; q/kv carry a 64B-atom swizzle (head_dim//2
        # B/row).  Allocation ORDER is the smem layout, so it is transcribed exactly.
        pool = K.smem_pool()
        smem_q = pool.alloc(
            (num_q_stages, block_q * num_heads, head_dim // 2),
            K.u8,
            swizzle=K.SW64B,
            align=swizzle_alignment,
        )
        smem_kv = pool.alloc(
            (num_kv_stages, block_kv, head_dim // 2), K.u8, swizzle=K.SW64B, align=swizzle_alignment
        )
        smem_sf_q = pool.alloc((num_q_stages, num_sfq), K.u32, align=16)
        smem_sf_q_t = pool.alloc((num_q_stages, num_sfq), K.u32, align=16)
        # 2D view for the copy_async(tma) dst: a fresh decl_buffer, NOT .view(shape)
        # (the shape-view keeps the rank-2 layout and mis-maps the 3D TMA write).
        smem_sf_q_2d = K.decl_buffer(
            (num_q_stages, block_q, num_heads),
            "uint32",
            data=smem_sf_q.data,
            scope="shared.dyn",
            elem_offset=smem_sf_q.elem_offset,
            align=16,
        )
        smem_sf_kv = pool.alloc((num_kv_stages, num_sfkv), K.u32, align=16)
        smem_sf_kv_t = pool.alloc((num_kv_stages, num_sfkv), K.u32, align=16)
        smem_weights = pool.alloc((num_q_stages, block_q, num_heads), K.f32, align=16)
        # Producer/consumer barrier pairs as Pipeline objects (full = data ready, empty
        # = slot free); each Pipeline runs mbarrier.init itself, no separate init loop.
        q_pipe = K.Pipeline(
            pool,
            num_q_stages,
            full="tma",
            empty="mbar",
            init_full=1,
            init_empty=num_math_threads + 32,
        )
        kv_pipe = K.Pipeline(
            pool, num_kv_stages, full="tma", empty="tcgen05", init_full=1, init_empty=1
        )
        tmem_pipe = K.Pipeline(
            pool, num_tmem_stages, full="tcgen05", empty="mbar", init_full=1, init_empty=128
        )
        # Per-stage handoff from the transpose worker; staging reuse is
        # protected by the kv pipe.
        sf_ready = K.MBarrier(pool, num_kv_stages)
        sf_ready.init(1)
        tmem_ptr_in_smem = pool.alloc((1,), K.u32, align=4)
        pool.commit()
        # TMEM operands are raw column addresses.  The fixed map is accumulator
        # columns first, then SFQ and SFKV; no TIR TMEM buffer view.
        tmem_col = 0
        sfq_tmem_col = tmem_start_col_of_sfq
        sfkv_tmem_col = tmem_start_col_of_sfkv
        seq_k_start = K.alloc_local([block_q], "uint32")
        seq_k_end = K.alloc_local([block_q], "uint32")
        schedule_result = K.alloc_local([2], "uint32")

        def store_logits(flat_offset, value):
            if config.logits_dtype == "float32":
                K.ptx.st.global_.f32(logits_flat.ptr_to([flat_offset]), value)
            else:
                K.ptx.st.global_.b16(logits_flat.ptr_to([flat_offset]), value)

        def load_schedule(q_idx):
            schedule_start = K.local_scalar("uint32", init=K.uint32(0xFFFFFFFF))
            schedule_end = K.local_scalar("uint32", init=K.uint32(0))
            for schedule_i in range(block_q):
                row_idx = K.local_scalar(
                    "uint32",
                    init=K.min(
                        q_idx * K.uint32(block_q) + K.uint32(schedule_i), seq_len - K.uint32(1)
                    ),
                )
                row_start = K.local_scalar("int32")
                row_end = K.local_scalar("int32")
                K.ptx.ld.global_.s32(
                    row_start, cu_seq_len_k_start.ptr_to([K.Cast("int32", row_idx)])
                )
                K.ptx.ld.global_.s32(row_end, cu_seq_len_k_end.ptr_to([K.Cast("int32", row_idx)]))
                K.ptx.mov.b32(
                    seq_k_start[schedule_i], K.min(K.Cast("uint32", row_start), seq_len_kv)
                )
                K.ptx.mov.b32(seq_k_end[schedule_i], K.min(K.Cast("uint32", row_end), seq_len_kv))
                K.assign(schedule_start, K.min(schedule_start, seq_k_start[schedule_i]))
                K.assign(schedule_end, K.max(schedule_end, seq_k_end[schedule_i]))
            K.assign(schedule_start, schedule_start // K.uint32(4) * K.uint32(4))
            num_kv_blocks = K.local_scalar(
                "uint32",
                init=(schedule_end - schedule_start + K.uint32(block_kv - 1)) // K.uint32(block_kv),
            )
            K.ptx.mov.b32(schedule_result[0], schedule_start)
            K.ptx.mov.b32(schedule_result[1], num_kv_blocks)

        # Pipeline constructors already ran mbarrier.init; fence + cta_sync publish them.
        K.ptx.fence.mbarrier_init.release.cluster()

        # The tmem allocation is warp 10's, but it is NOT the mma role block:
        # it sits before the CTA-wide sync, with CTA-scope code after it, so it
        # is spelled as the original spells it -- a plain guard, no setmaxnreg.
        with K.If(warp_idx == spec_warp_start + 2), K.Then():
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                K.address_of(tmem_ptr_in_smem[0]), K.uint32(num_tmem_cols)
            )
        K.cuda.cta_sync()

        num_q_blocks = K.local_scalar(
            "uint32", init=(seq_len + K.uint32(block_q - 1)) // K.uint32(block_q)
        )
        K.ptx.griddepcontrol.wait()

        # ---------------- roles ----------------------------------------
        #
        # The entry is deliberately unpinned, so each role states the original
        # setmaxnreg direction explicitly instead of fabricating an entry
        # allocation from a launch bound the kernel does not have.
        sp = K.specialize()
        producer_regs = {"regs": 56} if min_blocks is not None else {"regs": 56, "direction": "dec"}
        math_regs = {"regs": 224} if min_blocks is not None else {"regs": 224, "direction": "inc"}
        q_tma = sp.role("q_tma", warps=[spec_warp_start], **producer_regs)
        kv_tma = sp.role("kv_tma", warps=[spec_warp_start + 1], **producer_regs)
        mma = sp.role("mma", warps=[spec_warp_start + 2], **producer_regs)
        sf_transpose = sp.role("sf_transpose", warps=[spec_warp_start + 3], **producer_regs)
        math = sp.role("math", warps=list(range(spec_warp_start)), **math_regs)

        # ---------------- warp 8: Q + SFQ + weights ---------------------
        with q_tma:
            # elect_sync wraps the WHOLE loop, as the original does: the ring
            # cursors stay elect-lane locals on the uniform datapath. G3's
            # loop-level rule means this placement is preserved exactly.
            with K.If(K.cuda.elect_sync()), K.Then():
                q_state = K.RingState(num_q_stages)
                q_idx = K.local_scalar("uint32", init=sm_idx)
                with K.While(q_idx < num_q_blocks):
                    q_pipe.empty.wait(q_state.stage, q_state.phase ^ K.uint32(1))
                    # u32 row base -- the copy_async(tma) gmem-layout grouping now
                    # handles unsigned shape extents (no int32 cast needed).
                    q_row0 = K.local_scalar("uint32", init=q_idx * K.uint32(block_q * num_heads))
                    K.ptx[TMA_G2S_2D](
                        smem_q[q_state.stage].ptr_to(0, 0),
                        K.address_of(q_map),
                        K.int32(0),
                        K.Cast("int32", q_row0),
                        q_pipe.full.ptr_to([q_state.stage]),
                        cache_policy_evict_normal,
                    )
                    q_blk0 = K.local_scalar("uint32", init=q_idx * K.uint32(block_q))
                    K.ptx[TMA_G2S_2D](
                        smem_sf_q_2d.ptr_to([q_state.stage, 0, 0]),
                        K.address_of(sf_q_map),
                        K.int32(0),
                        K.Cast("int32", q_blk0),
                        q_pipe.full.ptr_to([q_state.stage]),
                        cache_policy_evict_normal,
                    )
                    K.ptx[TMA_G2S_2D](
                        smem_weights.ptr_to([q_state.stage, 0, 0]),
                        K.address_of(weights_map),
                        K.int32(0),
                        K.Cast("int32", q_blk0),
                        q_pipe.full.ptr_to([q_state.stage]),
                        cache_policy_evict_normal,
                    )
                    q_pipe.full.arrive(
                        q_state.stage,
                        tx_count=smem_q_size_per_stage
                        + real_num_sfq * 4
                        + smem_weight_size_per_stage,
                    )
                    K.assign(q_idx, q_idx + K.uint32(config.num_sms))
                    q_state.advance()
            K.cuda.warp_sync()

        # ---------------- warp 9: KV + SFKV -----------------------------
        with kv_tma:
            with K.If(K.cuda.elect_sync()), K.Then():
                kv_state = K.RingState(num_kv_stages)
                q_idx = K.local_scalar("uint32", init=sm_idx)
                with K.While(q_idx < num_q_blocks):
                    load_schedule(q_idx)
                    kv_start = K.local_scalar("uint32", init=schedule_result[0])
                    num_kv_blocks = K.local_scalar("uint32", init=schedule_result[1])
                    kv_idx = K.local_scalar("uint32", init=K.uint32(0))
                    with K.While(kv_idx < num_kv_blocks):
                        kv_pipe.empty.wait(kv_state.stage, kv_state.phase ^ K.uint32(1))
                        kv_row0 = K.local_scalar(
                            "uint32", init=kv_start + kv_idx * K.uint32(block_kv)
                        )
                        K.ptx[TMA_G2S_2D](
                            smem_kv[kv_state.stage].ptr_to(0, 0),
                            K.address_of(kv_map),
                            K.int32(0),
                            K.Cast("int32", kv_row0),
                            kv_pipe.full.ptr_to([kv_state.stage]),
                            cache_policy_evict_normal,
                        )
                        K.ptx[TMA_G2S_1D](
                            smem_sf_kv.ptr_to([kv_state.stage, 0]),
                            K.address_of(sf_kv_map),
                            K.Cast("int32", kv_row0),
                            kv_pipe.full.ptr_to([kv_state.stage]),
                            cache_policy_evict_normal,
                        )
                        kv_pipe.full.arrive(
                            kv_state.stage,
                            tx_count=smem_kv_size_per_stage + smem_sf_kv_size_per_stage,
                        )
                        K.assign(kv_idx, kv_idx + K.uint32(1))
                        kv_state.advance()
                    K.assign(q_idx, q_idx + K.uint32(config.num_sms))

        # ---------------- warp 10: UTCCP + block-scaled MMA issuer -------
        with mma:
            tmem_allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_allocated, tmem_ptr_in_smem.ptr_to([0]))
            K.cuda.trap_when_assert_failed(tmem_allocated == K.uint32(0))
            desc_i = K.local_scalar("uint32")
            # Encode the block-scaled instruction descriptor once; each K phase
            # rotates its SF id from this same base descriptor below.
            K.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                K.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float4_e2m1fn",
                b_dtype="float4_e2m1fn",
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=sfkv_tmem_col,
                sfb_tmem_addr=sfq_tmem_col,
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            # REGION D operand views: fp4 over the packed-uint8 smem bytes; .view also
            # carries elem_offset, giving the true buffer start under the pool.
            smem_kv_fp4 = smem_kv.buf.view("float4_e2m1fn")
            smem_q_fp4 = smem_q.buf.view("float4_e2m1fn")
            desc_sf = K.local_scalar("uint64")
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc_sf), K.reinterpret("handle", K.uint64(0)), ldo=0, sdo=8, swizzle=0
            )
            q_state = K.RingState(num_q_stages)
            kv_state = K.RingState(num_kv_stages)
            tmem_state = K.RingState(num_tmem_stages)
            # The SFQ overwrite waits only when a previous MMA commit exists;
            # its stage/phase come from the K-owned current cursor.
            has_tmem_issue = K.local_scalar("uint32", init=K.uint32(0))
            q_idx = K.local_scalar("uint32", init=sm_idx)
            with K.While(q_idx < num_q_blocks):
                load_schedule(q_idx)
                kv_start = K.local_scalar("uint32", init=schedule_result[0])
                num_kv_blocks = K.local_scalar("uint32", init=schedule_result[1])
                q_pipe.full.wait(q_state.stage, q_state.phase)
                emit_sf_transpose(smem_sf_q, smem_sf_q_t, lane_idx, q_state.stage, 0)
                K.cuda.warp_sync()
                K.ptx.fence.proxy.async_.shared__cta()
                # Each tmem_pipe.full arrive commits all prior asynchronous
                # TCGEN work from this issuer. Wait for the final commit from
                # the preceding q block before overwriting its SFQ TMEM input.
                with K.If(has_tmem_issue != K.uint32(0)), K.Then():
                    previous_stage = K.Select(
                        tmem_state.stage == K.uint32(0),
                        K.uint32(num_tmem_stages - 1),
                        tmem_state.stage - K.uint32(1),
                    )
                    previous_phase = K.Select(
                        tmem_state.stage == K.uint32(0),
                        tmem_state.phase ^ K.uint32(1),
                        tmem_state.phase,
                    )
                    tmem_pipe.full.wait(previous_stage, previous_phase)
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx[TCGEN05_CP](
                        K.uint32(sfq_tmem_col),
                        replace_smem_desc_addr(desc_sf, smem_sf_q_t.ptr_to([q_state.stage, 0])),
                    )
                K.cuda.warp_sync()
                kv_idx = K.local_scalar("uint32", init=K.uint32(0))
                with K.While(kv_idx < num_kv_blocks):
                    # UMMA reads smem_kv (kv full); staged SF comes from the
                    # transpose worker (sf_ready).
                    kv_pipe.full.wait(kv_state.stage, kv_state.phase)
                    sf_ready.wait(kv_state.stage, kv_state.phase)
                    # cp + MMA share ONE elect scope: drops a redundant elect.sync per
                    # kv-iter and lets the cp overlap the MMA setup.
                    with K.If(K.cuda.elect_sync()), K.Then():
                        for sfkv_i in range(num_sfkv // num_utccp_aligned_elems):
                            K.ptx[TCGEN05_CP](
                                K.uint32(sfkv_tmem_col + sfkv_i * 4),
                                replace_smem_desc_addr(
                                    desc_sf,
                                    smem_sf_kv_t.ptr_to(
                                        [kv_state.stage, sfkv_i * num_utccp_aligned_elems]
                                    ),
                                ),
                            )
                        for math_wg_i in range(num_math_warpgroups):
                            tmem_addr = K.local_scalar(
                                "uint32",
                                init=K.uint32(tmem_col) + tmem_state.stage * K.uint32(umma_n),
                            )
                            tmem_pipe.empty.wait(tmem_state.stage, tmem_state.phase ^ K.uint32(1))
                            # REGION D: block-scaled FP4 UMMA, D = KV @ Q^K.  The
                            # first K=64 phase overwrites and the second accumulates;
                            # scale ids 0/2 select the two block-32 factors.
                            desc_i_local = K.local_scalar("uint32", init=desc_i)
                            for ki in range(head_dim // umma_k):
                                sf_linear = K.local_scalar(
                                    "int32", init=K.int32(ki * (umma_k // 32))
                                )
                                K.cuda.runtime_instr_desc(
                                    K.address_of(desc_i_local), sf_linear % (umma_k // 16)
                                )
                                K.ptx[TCGEN05_MMA](
                                    tmem_addr,
                                    recompute_fp4_smem_desc(
                                        smem_kv_fp4.ptr_to(
                                            [kv_state.stage, math_wg_i * umma_m, ki * umma_k]
                                        )
                                    ),
                                    recompute_fp4_smem_desc(
                                        smem_q_fp4.ptr_to([q_state.stage, 0, ki * umma_k])
                                    ),
                                    desc_i_local,
                                    K.cuda.get_tmem_addr(
                                        sfkv_tmem_col + math_wg_i * 4,
                                        sf_linear % 128 // 4,
                                        sf_linear // 128,
                                    ),
                                    K.cuda.get_tmem_addr(
                                        sfq_tmem_col, sf_linear % 128 // 4, sf_linear // 128
                                    ),
                                    K.ptx.pred(tvm.tirx.const(ki != 0, "bool")),
                                )
                            tmem_pipe.full.arrive(tmem_state.stage)
                            K.assign(has_tmem_issue, K.uint32(1))
                            tmem_state.advance()
                    with K.If(K.cuda.elect_sync()), K.Then():
                        kv_pipe.empty.arrive(kv_state.stage, cta_group=1)
                    K.assign(kv_idx, kv_idx + K.uint32(1))
                    kv_state.advance()
                q_pipe.empty.arrive(q_state.stage)
                K.assign(q_idx, q_idx + K.uint32(config.num_sms))
                q_state.advance()

        # ---------------- warp 11: SF-KV transpose worker ---------------
        with sf_transpose:
            # Overlaps transpose(k+1) with the tcgen05 warp's UTCCP+MMA of block k.
            kv_state = K.RingState(num_kv_stages)
            t_q_idx = K.local_scalar("uint32", init=sm_idx)
            with K.While(t_q_idx < num_q_blocks):
                load_schedule(t_q_idx)
                t_num_kv = K.local_scalar("uint32", init=schedule_result[1])
                t_kv_i = K.local_scalar("uint32", init=K.uint32(0))
                with K.While(t_kv_i < t_num_kv):
                    kv_pipe.full.wait(kv_state.stage, kv_state.phase)
                    # The preceding tcgen05.cp read of this transposed stage
                    # completes through kv_pipe.empty before the TMA producer
                    # can refill the stage.  Reconnect that async-proxy read
                    # to the generic-proxy stores which now reuse the same
                    # physical shared-memory backing.
                    K.ptx.fence.proxy.async_.shared__cta()
                    emit_sf_transpose(smem_sf_kv, smem_sf_kv_t, lane_idx, kv_state.stage, 0)
                    emit_sf_transpose(
                        smem_sf_kv, smem_sf_kv_t, lane_idx, kv_state.stage, num_utccp_aligned_elems
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                    with K.If(K.cuda.elect_sync()), K.Then():
                        sf_ready.arrive(kv_state.stage)
                    K.assign(t_kv_i, t_kv_i + K.uint32(1))
                    kv_state.advance()
                K.assign(t_q_idx, t_q_idx + K.uint32(config.num_sms))

        # ---------------- warps 0-7: math + epilogue --------------------
        with math:
            accum = K.alloc_local([num_heads], "float32")
            cached_weights = K.alloc_local([block_q, num_heads], "float32")
            # f32-dense store offsets, hoisted and chained (+block_kv per split) to keep
            # nvrtc's u64 IMAD.WIDE out of the hot loop; bf16-dense keeps per-iter form.
            token_store_off = K.alloc_local([block_q], "uint64")
            q_state = K.RingState(num_q_stages)
            tmem_state = K.RingState(
                num_tmem_stages, stage=warpgroup_idx, stride=num_math_warpgroups
            )
            math_thread_idx = K.Cast("uint32", K.tid_in_role())
            q_idx = K.local_scalar("uint32", init=sm_idx)
            with K.While(q_idx < num_q_blocks):
                load_schedule(q_idx)
                kv_start = K.local_scalar("uint32", init=schedule_result[0])
                num_kv_blocks = K.local_scalar("uint32", init=schedule_result[1])
                q_pipe.full.wait(q_state.stage, q_state.phase)
                with K.If(num_kv_blocks > K.uint32(0)), K.Then():
                    for weight_i in range(block_q):
                        for weight_j in range(num_heads // 4):
                            weight_col = K.local_scalar("int32", init=K.int32(weight_j * 4))
                            K.ptx.ld.shared.v4.f32(
                                cached_weights[weight_i, weight_col],
                                cached_weights[weight_i, weight_col + 1],
                                cached_weights[weight_i, weight_col + 2],
                                cached_weights[weight_i, weight_col + 3],
                                smem_weights.ptr_to([q_state.stage, weight_i, weight_col]),
                            )
                    # Publish the generic-proxy weight reads before this
                    # consumer releases the Q stage for a later TMA overwrite.
                    K.ptx.fence.proxy.async_.shared__cta()
                    if not config.compressed_logits and config.logits_dtype == "float32":
                        for tb_i in range(block_q):
                            K.ptx.mov.b64(
                                token_store_off[tb_i],
                                K.Cast("uint64", q_idx * K.uint32(block_q) + K.uint32(tb_i))
                                * K.Cast("uint64", logits_stride)
                                + K.Cast("uint64", kv_start + math_thread_idx),
                            )
                    kv_idx = K.local_scalar("uint32", init=K.uint32(0))
                    with K.While(kv_idx < num_kv_blocks):
                        kv_offset = K.local_scalar(
                            "uint32", init=kv_start + kv_idx * K.uint32(block_kv) + math_thread_idx
                        )
                        tmem_pipe.full.wait(tmem_state.stage, tmem_state.phase)
                        for q_inner_i in range(block_q):
                            tmem_addr = K.local_scalar(
                                "uint32",
                                init=K.uint32(tmem_col)
                                + tmem_state.stage * K.uint32(umma_n)
                                + K.uint32(q_inner_i * num_heads),
                            )
                            # REGION E: TMEM->register read as 2x tcgen05.ld.32x32b.x32
                            # (DeepGEMM's 64-head shape); accum stays flat for the reduce.
                            K.ptx[TCGEN05_LD_X32](
                                *[accum[head_i] for head_i in range(num_heads // 2)],
                                K.cuda.get_tmem_addr(K.uint32(tmem_col), 0, tmem_addr),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            tmem_addr_hi = K.local_scalar(
                                "uint32", init=tmem_addr + K.uint32(num_heads // 2)
                            )
                            K.ptx[TCGEN05_LD_X32](
                                *[
                                    accum[num_heads // 2 + head_i]
                                    for head_i in range(num_heads // 2)
                                ],
                                K.cuda.get_tmem_addr(K.uint32(tmem_col), 0, tmem_addr_hi),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            if q_inner_i == block_q - 1:
                                tmem_pipe.empty.arrive(tmem_state.stage)
                            result_f32 = _weighted_relu_reduce(
                                accum, cached_weights, q_inner_i, num_heads
                            )
                            result = K.local_scalar(
                                logits_tir_dtype, init=K.Cast(logits_tir_dtype, result_f32)
                            )
                            if config.compressed_logits:
                                q_offset = K.local_scalar(
                                    "uint64",
                                    init=K.Cast(
                                        "uint64", q_idx * K.uint32(block_q) + K.uint32(q_inner_i)
                                    )
                                    * K.Cast("uint64", logits_stride),
                                )
                                row_k_start = K.local_scalar("uint32", init=seq_k_start[q_inner_i])
                                row_k_end = K.local_scalar("uint32", init=seq_k_end[q_inner_i])
                                # Range-guarded store: if-converts to a predicated @P STG
                                # for this kernel (unlike fp8's clamp-to-padding variant).
                                with (
                                    K.If(
                                        tvm.tirx.all(
                                            row_k_start <= kv_offset, kv_offset < row_k_end
                                        )
                                    ),
                                    K.Then(),
                                ):
                                    store_logits(
                                        q_offset
                                        + K.Cast("uint64", kv_offset)
                                        - K.Cast("uint64", row_k_start),
                                        result,
                                    )
                            elif config.logits_dtype == "float32":
                                store_logits(token_store_off[q_inner_i], result)
                            else:
                                # bf16-dense: per-iter offset; see token_store_off note.
                                q_offset_bf16 = K.local_scalar(
                                    "uint64",
                                    init=K.Cast(
                                        "uint64", q_idx * K.uint32(block_q) + K.uint32(q_inner_i)
                                    )
                                    * K.Cast("uint64", logits_stride),
                                )
                                store_logits(q_offset_bf16 + K.Cast("uint64", kv_offset), result)
                        if not config.compressed_logits and config.logits_dtype == "float32":
                            for tb_i in range(block_q):
                                K.ptx.mov.b64(
                                    token_store_off[tb_i],
                                    token_store_off[tb_i] + K.uint64(block_kv),
                                )
                        K.assign(kv_idx, kv_idx + K.uint32(1))
                        tmem_state.advance()
                q_pipe.empty.arrive(q_state.stage)
                K.assign(q_idx, q_idx + K.uint32(config.num_sms))
                q_state.advance()
            K.ptx.bar.sync(8, K.uint32(num_math_threads))
            with K.If(warp_idx == 0), K.Then():
                K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    K.uint32(tmem_col), K.uint32(num_tmem_cols)
                )

    # `@K.kernel` has no `attrs=`, so the launch metadata the original sets on
    # its PrimFunc is applied to the traced one here. `Kernel.func` is a plain
    # attribute (entry.py), and `Kernel.mod` reads it, so this reaches compile.
    main = sm100_fp4_mqa_logits.func.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )
    return tvm.IRModule({"main": main})


def _compile_tirx_mqa_for_config(
    *,
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    logits_dtype: str,
    compressed_logits: bool,
    disable_cp: bool,
    num_sms: int,
    logits_stride_override: int | None,
) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    mod = get_kernel(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        logits_dtype=logits_dtype,
        compressed_logits=compressed_logits,
        disable_cp=disable_cp,
        num_sms=num_sms,
        logits_stride_override=logits_stride_override,
    )
    with target:
        # --ftz=false lets abs fold into FADD2 operand modifiers (ftz blocks it).
        os.environ["TVM_CUDA_NVRTC_EXTRA_OPTS"] = "--ftz=false"
        os.environ["TVM_CUDA_PTXAS_EXTRA_OPTS"] = "--allow-expensive-optimizations=true"
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


_compile_tirx_mqa_for_config = cache(_compile_tirx_mqa_for_config)


def _compile_tirx_mqa_kwargs(config: MQALogitsConfig) -> dict[str, Any]:
    return {
        "seq_len": config.block_q,
        "seq_len_kv": config.block_kv,
        "num_heads": config.num_heads,
        "head_dim": config.head_dim,
        "logits_dtype": config.logits_dtype,
        "compressed_logits": config.compressed_logits,
        "disable_cp": True,
        "num_sms": config.num_sms,
        "logits_stride_override": None,
    }


def _compile_tirx_mqa_key(config: MQALogitsConfig) -> tuple[tuple[str, Any], ...]:
    return tuple(_compile_tirx_mqa_kwargs(config).items())


def _compile_tirx_mqa(config: MQALogitsConfig, max_seqlen_k: int) -> Any:
    # The kernel is independent of seq_len/seq_len_kv/disable_cp/logits_stride (all
    # runtime): canonical values let the cache dedup to one kernel per structural config.
    del max_seqlen_k

    compile_kwargs = _compile_tirx_mqa_kwargs(config)
    return _compile_tirx_mqa_for_config(**compile_kwargs)


def _logits_storage_shape(config: MQALogitsConfig, max_seqlen_k: int) -> tuple[int, int]:
    if config.compressed_logits:
        stride = _align_up(max_seqlen_k, config.block_kv)
    else:
        stride = _align_up(config.seq_len_kv + config.block_kv, 8)
    return config.aligned_seq_len, stride


def _allocate_logits(config: MQALogitsConfig, max_seqlen_k: int) -> torch.Tensor:
    storage_shape = _logits_storage_shape(config, max_seqlen_k)
    return torch.full(
        storage_shape, float("-inf"), device="cuda", dtype=_torch_logits_dtype(config.logits_dtype)
    )


def _prepare_global_barrier(executable: Any) -> None:
    try:
        prepare_global_barrier = executable.mod.get_function("__tvm_prepare_global_barrier")
    except AttributeError:
        prepare_global_barrier = None
    if prepare_global_barrier is not None:
        prepare_global_barrier()


def _prepare_tirx_invocation(
    data: dict[str, Any], logits: torch.Tensor | None = None, *, executable: Any | None = None
) -> dict[str, Any]:
    config: MQALogitsConfig = data["config"]
    if logits is None:
        logits = _allocate_logits(config, data["max_seqlen_k"])
    if executable is None:
        executable = _compile_tirx_mqa(config, data["max_seqlen_k"])
    return {
        "executable": executable,
        "logits": logits,
        "tensor_maps": _build_tirx_tensor_maps(data),
    }


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    executable = invocation["executable"]
    logits = invocation["logits"]
    tensor_maps = invocation["tensor_maps"]
    _prepare_global_barrier(executable)
    executable.mod(
        config.seq_len,
        config.seq_len_kv,
        data["max_seqlen_k"],
        logits.stride(0),
        data["cu_seq_len_k_start"].view(-1),
        data["cu_seq_len_k_end"].view(-1),
        logits.view(-1),
        *(tensor_map.ptr for tensor_map in tensor_maps),
    )
    return logits


def _launch_tirx_mqa(data: dict[str, Any], logits: torch.Tensor | None = None) -> torch.Tensor:
    return _run_tirx_invocation(data, _prepare_tirx_invocation(data, logits))


def _run_deepgemm_mqa(data: dict[str, Any], *, clean_logits: bool) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    return data["deep_gemm"].fp8_fp4_mqa_logits(
        q=data["q_in"],
        kv=data["kv_in"],
        weights=data["weights"],
        cu_seq_len_k_start=data["cu_seq_len_k_start"],
        cu_seq_len_k_end=data["cu_seq_len_k_end"],
        clean_logits=clean_logits,
        max_seqlen_k=data["max_seqlen_k"],
        logits_dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _expand_compressed_logits(logits: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    if not config.compressed_logits:
        return logits[: config.seq_len, : config.seq_len_kv]

    expanded = torch.full(
        (config.seq_len, config.seq_len_kv), float("-inf"), device="cuda", dtype=logits.dtype
    )
    ks = data["cu_seq_len_k_start"]
    ke = data["cu_seq_len_k_end"]
    for row_idx in range(config.seq_len):
        start = int(ks[row_idx].item())
        end = int(ke[row_idx].item())
        expanded[row_idx, start:end] = logits[row_idx, : end - start]
    return expanded


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _assert_correct(data: dict[str, Any], logits: torch.Tensor, *, name: str) -> float:
    reference = data["reference"]
    observed = _expand_compressed_logits(logits, data)
    ref_neginf_mask = reference == float("-inf")
    observed = observed.masked_fill(ref_neginf_mask, 0)
    reference = reference.masked_fill(ref_neginf_mask, 0)
    diff = _calc_diff(observed, reference)
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} simulated diff {diff:.6g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    config: MQALogitsConfig = data["config"]
    clean_logits = not config.compressed_logits
    deepgemm_logits = _run_deepgemm_mqa(data, clean_logits=clean_logits)
    # Library-anchored: the torch ref is a yardstick, not the arbiter --
    # DeepGEMM's own diff on the same inputs bounds what TIRx must achieve.
    deepgemm_diff = _assert_correct(data, deepgemm_logits, name="DeepGEMM")
    tirx_logits = _launch_tirx_mqa(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.6g} is worse than DeepGEMM diff {deepgemm_diff:.6g}"
        )


def prepare_bench(**kwargs: Any):
    """Compile the TIRx executable without allocating CUDA data."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = _make_config(**kwargs)
    executable = _compile_tirx_mqa(config, 0)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs), "executable": executable})


def run_gpu(prepared, **kwargs: Any) -> dict[str, Any]:
    kwargs = {**prepared["config"], **kwargs}
    from tirx_kernels.runner import bench

    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    timer = kwargs.pop("timer", None)  # None inherits the global default (proton)
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    tirx_executable = prepared["executable"]

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    data = prepare_data(**config_kwargs)
    invocation = _prepare_tirx_invocation(data, executable=tirx_executable)

    # Correctness gate before timing (preserves the old validate_case behavior).
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    max_diff = _assert_correct(data, tirx_logits, name="TIRx")

    funcs = {"tirx": lambda: _run_tirx_invocation(data, invocation)}

    def _deepgemm():
        return lambda: _run_deepgemm_mqa(data, clean_logits=False)

    references = {"deepgemm": _deepgemm}

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )
    result["max_diff"] = max_diff
    return result


def run_bench(**kwargs: Any) -> dict[str, Any]:
    protocol = {
        name: kwargs.pop(name)
        for name in ("warmup", "repeat", "timer", "rounds", "cooldown_s")
        if name in kwargs
    }
    return prepare_bench(**kwargs).run_gpu(**protocol)


__all__ = [
    "CONFIGS",
    "DEEPGEMM_TEST_COVERAGE",
    "KERNEL_META",
    "MQALogitsConfig",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
