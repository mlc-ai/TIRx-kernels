# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of DeepGEMM's MQA logits kernel, FP8 variant.

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
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6
_COMPILE_CACHE_NAMESPACE = "deepgemm.mqa_logits_fp8.compile"


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
    kv_fp8, kv_scales = data["kv_in"]
    q = data["q_in"].view(torch.uint8).reshape(config.seq_len * config.num_heads, config.head_dim)
    swizzle = {32: K.SW32B, 64: K.SW64B, 128: K.SW128B}[config.head_dim].value
    maps = (
        _encode_tensor_map(
            "float32", 1, kv_scales, (config.seq_len_kv,), (), (config.block_kv,), 0
        ),
        _encode_tensor_map(
            "uint8",
            2,
            kv_fp8.view(torch.uint8),
            (config.head_dim, config.seq_len_kv),
            (config.head_dim,),
            (config.head_dim, config.block_kv),
            swizzle,
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
            "uint8",
            2,
            q,
            (config.head_dim, config.seq_len * config.num_heads),
            (config.head_dim,),
            (config.head_dim, config.block_q * config.num_heads),
            swizzle,
        ),
    )
    return maps


@dataclass(frozen=True)
class MQALogitsFP8Config:
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
        if self.head_dim not in (32, 64, 128):
            raise ValueError("head_dim must be 32, 64, or 128 for the SM100 FP8 MQA logits kernel")
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


def _make_config(**kwargs: Any) -> MQALogitsFP8Config:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = MQALogitsFP8Config(**kwargs)
    config.validate()
    return config


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
    "name": "deepgemm_sm100_fp8_mqa_logits",
    "category": "deepgemm",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "deep-gemm",
            "git": {
                "url": "https://github.com/deepseek-ai/DeepGEMM.git",
                "commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
            },
            "import": "deep_gemm",
        },
    ),
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


def _generate_ks_ke(config: MQALogitsFP8Config) -> tuple[torch.Tensor, torch.Tensor]:
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
    cols = torch.arange(0, seq_len_kv, device="cuda")
    logits = torch.empty((q.shape[0], seq_len_kv), device="cuda", dtype=torch.float32)
    chunk_size = 128
    for chunk_start in range(0, q.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, q.shape[0])
        score = torch.einsum("mhd,nd->hmn", q_f32[chunk_start:chunk_end], kv_f32)
        chunk_logits = (
            score.relu() * weights[chunk_start:chunk_end].unsqueeze(-1).transpose(0, 1)
        ).sum(dim=0)
        mask_lo = cols[None, :] >= cu_seq_len_k_start[chunk_start:chunk_end, None]
        mask_hi = cols[None, :] < cu_seq_len_k_end[chunk_start:chunk_end, None]
        logits[chunk_start:chunk_end] = chunk_logits.masked_fill(
            ~(mask_lo & mask_hi), float("-inf")
        )
    return logits


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_mqa()
    config = _make_config(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 FP8 MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP8 MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    q = torch.randn(
        config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = _generate_ks_ke(config)

    q_in = q.to(torch.float8_e4m3fn).contiguous()
    kv_in = deep_gemm.utils.per_custom_dims_cast_to_fp8(kv, (0,), False)

    q_simulated = q_in.to(torch.bfloat16)
    kv_simulated = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
    reference = _ref_mqa_logits(
        q_simulated.to(torch.bfloat16), kv_simulated.to(torch.bfloat16), weights, ks, ke
    )
    max_seqlen_k = int((ke - ks).max().item()) if config.compressed_logits else 0
    runtime_config = MQALogitsFP8Config(
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


def get_kernel(**kwargs: Any):
    config = _make_config(**kwargs)
    num_heads = config.num_heads
    head_dim = config.head_dim
    block_q = config.block_q
    block_kv = config.block_kv
    num_q_stages = 3
    num_kv_stages = 5
    num_tmem_stages = 3
    num_specialized_threads = 128
    num_math_threads = 256
    num_math_warpgroups = num_math_threads // 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    umma_m = 128
    umma_k = 32
    umma_n = block_q * num_heads
    smem_q_size_per_stage = block_q * num_heads * head_dim
    smem_weight_size_per_stage = block_q * num_heads * 4
    smem_kv_size_per_stage = block_kv * head_dim
    smem_kv_scale_size_per_stage = block_kv * 4
    _SWZ = {32: K.SW32B, 64: K.SW64B, 128: K.SW128B}[head_dim]
    swizzle_alignment = 8 * head_dim

    num_accum_tmem_cols = block_q * num_heads * num_tmem_stages
    num_tmem_cols = 32
    for bound in (32, 64, 128, 256):
        if num_accum_tmem_cols > bound:
            num_tmem_cols = bound * 2
    if num_tmem_cols > 512:
        raise ValueError(f"tensor memory columns {num_tmem_cols} exceeds SM100 single-CTA limit")
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    TMA_G2S_1D = (
        "cp.async.bulk.tensor.1d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    TMA_G2S_2D = (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    MMA = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    TC_LD = f"tcgen05.ld.sync.aligned.32x32b.x{num_heads // 2}.b32"
    desc_sdo = head_dim // 2
    desc_swizzle = {32: 1, 64: 2, 128: 3}[head_dim]

    @K.kernel(warps=num_warps, arch="sm_100f", min_blocks_per_sm=1, grid=config.num_sms)
    def sm100_fp8_mqa_logits(
        seq_len: K.u32,
        seq_len_kv: K.u32,
        max_seqlen_k: K.u32,
        num_q_blocks: K.u32,
        logits_stride: K.u32,
        cu_seq_len_k_start: K.gptr[K.i32],
        cu_seq_len_k_end: K.gptr[K.i32],
        logits_flat: K.gptr[logits_tir_dtype],
        kv_scales_map: K.TensorMap,
        kv_map: K.TensorMap,
        weights_map: K.TensorMap,
        q_map: K.TensorMap,
    ):
        cache_policy_evict_normal = K.uint64(0x1000000000000000)
        sm_idx_u32 = K.Cast("uint32", K.cta_id())
        warp_idx = K.warp_id()
        warpgroup_idx = K.warpgroup_id([num_warps // 4])
        lane_idx_u32 = K.Cast("uint32", K.lane_id())

        # One elected lane of warp 0 prefetches every descriptor before any
        # pipeline traffic (the former dispatcher's placement).
        for m in (q_map, weights_map, kv_map, kv_scales_map):
            with K.If(warp_idx == 0), K.Then():
                with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                    K.ptx.prefetch.tensormap(K.address_of(m))

        smem = K.smem_pool()
        smem_q = smem.alloc(
            (num_q_stages, block_q * num_heads, head_dim),
            K.u8,
            swizzle=_SWZ,
            align=swizzle_alignment,
        )
        smem_weights = smem.alloc((num_q_stages, block_q, num_heads), K.f32, align=16)
        smem_kv = smem.alloc(
            (num_kv_stages, block_kv, head_dim), K.u8, swizzle=_SWZ, align=swizzle_alignment
        )
        smem_kv_scales = smem.alloc((num_kv_stages, block_kv), K.f32, align=16)
        q_pipe = K.Pipeline(
            smem,
            num_q_stages,
            full="tma",
            empty="mbar",
            init_full=1,
            init_empty=num_math_threads + 1,
        )
        kv_pipe = K.Pipeline(
            smem, num_kv_stages, full="tma", empty="mbar", init_full=1, init_empty=num_math_threads
        )
        tmem_pipe = K.Pipeline(
            smem, num_tmem_stages, full="tcgen05", empty="mbar", init_full=1, init_empty=128
        )
        tmem_ptr_in_smem = smem.alloc((1,), K.u32, align=4)
        # TMEM D is addressed by its fixed column base.  The allocator is
        # asserted to return base zero below, so no TIR tmem buffer is needed.
        tmem_col = 0

        seq_k_start = K.alloc_local([block_q], "uint32")
        seq_k_end = K.alloc_local([block_q], "uint32")
        schedule_result = K.alloc_local([2], "uint32")

        def store_logits(flat_offset, value):
            # Scalar predicated store: per-thread non-contiguous output, so
            # TMA/bulk does not apply.
            if config.logits_dtype == "float32":
                K.ptx.st.global_.f32(logits_flat.ptr_to([flat_offset]), value)
            else:
                K.ptx.st.global_.b16(logits_flat.ptr_to([flat_offset]), value)

        def load_schedule(q_idx):
            schedule_start = K.local_scalar("uint32")
            schedule_end = K.local_scalar("uint32")
            K.assign(schedule_start, K.uint32(0xFFFFFFFF))
            K.assign(schedule_end, K.uint32(0))
            for schedule_i in range(block_q):
                row_idx = K.min(
                    q_idx * K.uint32(block_q) + K.uint32(schedule_i), seq_len - K.uint32(1)
                )
                row = K.alloc_local([2], "int32")
                K.ptx.ld.global_.s32(row[0], cu_seq_len_k_start.ptr_to([K.Cast("int32", row_idx)]))
                K.ptx.mov.b32(seq_k_start[schedule_i], K.min(K.Cast("uint32", row[0]), seq_len_kv))
                K.ptx.ld.global_.s32(row[1], cu_seq_len_k_end.ptr_to([K.Cast("int32", row_idx)]))
                K.ptx.mov.b32(seq_k_end[schedule_i], K.min(K.Cast("uint32", row[1]), seq_len_kv))
                K.assign(schedule_start, K.min(schedule_start, seq_k_start[schedule_i]))
                K.assign(schedule_end, K.max(schedule_end, seq_k_end[schedule_i]))
            K.assign(schedule_start, schedule_start // K.uint32(4) * K.uint32(4))
            K.ptx.mov.b32(schedule_result[0], schedule_start)
            K.ptx.mov.b32(
                schedule_result[1],
                (schedule_end - schedule_start + K.uint32(block_kv - 1)) // K.uint32(block_kv),
            )

        def wrelu_reduce(accum, weights, row):
            """0.5 * sum_h w[row,h] * (a[h] + |a[h]|) — orig:L346-383.

            Packed f32x2 throughout: one uint64 carries each pair, and the
            two running sums are only unpacked at the end. Kernel-specific
            math, so a local closure (design doc principle 4).
            """
            sum_0 = K.alloc_local([1], "uint64")
            sum_1 = K.alloc_local([1], "uint64")
            accum_pair = K.local_scalar("uint64")
            abs_pair = K.local_scalar("uint64")
            relu_pair = K.local_scalar("uint64")
            weight_pair = K.local_scalar("uint64")
            abs_lo = K.local_scalar("float32")
            abs_hi = K.local_scalar("float32")
            total = K.local_scalar("uint64")
            total_lo = K.local_scalar("float32")
            total_hi = K.local_scalar("float32")
            result = K.local_scalar("float32")
            K.ptx.mov.b64(sum_0[0], K.float32(0), K.float32(0))
            K.ptx.mov.b64(sum_1[0], K.float32(0), K.float32(0))
            for head in range(0, num_heads, 4):
                for pair, acc_sum in ((head, sum_0), (head + 2, sum_1)):
                    K.ptx.mov.b64(accum_pair, accum[pair], accum[pair + 1])
                    K.ptx.abs.f32(abs_lo, accum[pair])
                    K.ptx.abs.f32(abs_hi, accum[pair + 1])
                    K.ptx.mov.b64(abs_pair, abs_lo, abs_hi)
                    K.ptx.add.rn.f32x2(relu_pair, accum_pair, abs_pair)
                    K.ptx.mov.b64(weight_pair, weights[row, pair], weights[row, pair + 1])
                    K.ptx.fma.rn.f32x2(acc_sum[0], relu_pair, weight_pair, acc_sum[0])
            K.ptx.add.rn.f32x2(total, sum_0[0], sum_1[0])
            K.ptx.mov.b64(total_lo, total_hi, total)
            K.ptx.add.rn.f32(result, total_lo, total_hi)
            K.ptx.mul.rn.f32(result, result, K.float32(0.5))
            return result

        K.ptx.fence.mbarrier_init.release.cluster()
        # Keep the collective under the entry-owned warp-uniform id.  The
        # generic pool guard declares a second warp scope, which is not a safe
        # lowering boundary for this persistent kernel.
        with K.If(warp_idx == spec_warp_start + 2), K.Then():
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                K.address_of(tmem_ptr_in_smem[0]), K.uint32(num_tmem_cols)
            )
        K.cuda.cta_sync()
        K.ptx.griddepcontrol.wait()

        # ---------------- roles ----------------------------------------
        sp = K.specialize()
        math = sp.role("math", warps=list(range(spec_warp_start)), regs=224)
        q_tma = sp.role("q_tma", warps=[spec_warp_start], regs=56)
        kv_tma = sp.role("kv_tma", warps=[spec_warp_start + 1], regs=56)
        mma = sp.role("mma", warps=[spec_warp_start + 2], regs=56)
        idle = sp.role("idle", warps=[spec_warp_start + 3], regs=56)

        # ---------------- warp 8: Q + weights ---------------------------
        with q_tma:
            # elect_sync wraps the WHOLE loop, as the original does: the ring
            # cursors stay elect-lane locals on the uniform datapath. G3's
            # loop-level rule means this placement is preserved exactly.
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                q_state = K.RingState(num_q_stages)
                q_idx = K.local_scalar("uint32", init=sm_idx_u32)
                with K.While(q_idx < num_q_blocks):
                    q_pipe.empty.wait(q_state.stage, q_state.phase ^ K.uint32(1))
                    K.ptx[TMA_G2S_2D](
                        smem_q[q_state.stage].ptr_to(0, 0),
                        K.address_of(q_map),
                        K.int32(0),
                        K.Cast("int32", q_idx * K.uint32(block_q * num_heads)),
                        q_pipe.full.ptr_to([q_state.stage]),
                        cache_policy_evict_normal,
                    )
                    K.ptx[TMA_G2S_2D](
                        smem_weights.ptr_to([q_state.stage, 0, 0]),
                        K.address_of(weights_map),
                        K.int32(0),
                        K.Cast("int32", q_idx * K.uint32(block_q)),
                        q_pipe.full.ptr_to([q_state.stage]),
                        cache_policy_evict_normal,
                    )
                    q_pipe.full.arrive(
                        q_state.stage, tx_count=smem_q_size_per_stage + smem_weight_size_per_stage
                    )
                    K.assign(q_idx, q_idx + K.uint32(config.num_sms))
                    q_state.advance()
            K.cuda.warp_sync()

        # ---------------- warp 9: KV + KV scales ------------------------
        with kv_tma:
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                kv_state = K.RingState(num_kv_stages)
                q_idx = K.local_scalar("uint32")
                kv_idx = K.local_scalar("uint32")
                K.assign(q_idx, sm_idx_u32)
                with K.While(q_idx < num_q_blocks):
                    load_schedule(q_idx)
                    kv_start = schedule_result[0]
                    K.assign(kv_idx, K.uint32(0))
                    with K.While(kv_idx < schedule_result[1]):
                        kv_pipe.empty.wait(kv_state.stage, kv_state.phase ^ K.uint32(1))
                        kv_row0 = kv_start + kv_idx * K.uint32(block_kv)
                        K.ptx[TMA_G2S_2D](
                            smem_kv[kv_state.stage].ptr_to(0, 0),
                            K.address_of(kv_map),
                            K.int32(0),
                            K.Cast("int32", kv_row0),
                            kv_pipe.full.ptr_to([kv_state.stage]),
                            cache_policy_evict_normal,
                        )
                        K.ptx[TMA_G2S_1D](
                            smem_kv_scales.ptr_to([kv_state.stage, 0]),
                            K.address_of(kv_scales_map),
                            K.Cast("int32", kv_row0),
                            kv_pipe.full.ptr_to([kv_state.stage]),
                            cache_policy_evict_normal,
                        )
                        kv_pipe.full.arrive(
                            kv_state.stage,
                            tx_count=smem_kv_size_per_stage + smem_kv_scale_size_per_stage,
                        )
                        K.assign(kv_idx, kv_idx + K.uint32(1))
                        kv_state.advance()
                    K.assign(q_idx, q_idx + K.uint32(config.num_sms))

        # ---------------- warp 10: MMA issuer ---------------------------
        with mma:
            tmem_allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_allocated, tmem_ptr_in_smem.ptr_to([0]))
            K.cuda.trap_when_assert_failed(tmem_allocated == K.uint32(0))
            desc_i = K.local_scalar("uint32")
            K.cuda.tcgen05.encode_instr_descriptor(
                K.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float8_e4m3fn",
                b_dtype="float8_e4m3fn",
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            desc_a = K.local_scalar("uint64")
            desc_b = K.local_scalar("uint64")
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                q_state = K.RingState(num_q_stages)
                kv_state = K.RingState(num_kv_stages)
                tmem_state = K.RingState(num_tmem_stages)
                q_idx = K.local_scalar("uint32")
                kv_idx = K.local_scalar("uint32")
                K.assign(q_idx, sm_idx_u32)
                with K.While(q_idx < num_q_blocks):
                    load_schedule(q_idx)
                    q_pipe.full.wait(q_state.stage, q_state.phase)
                    K.assign(kv_idx, K.uint32(0))
                    with K.While(kv_idx < schedule_result[1]):
                        kv_pipe.full.wait(kv_state.stage, kv_state.phase)
                        for math_wg_i in range(num_math_warpgroups):
                            tmem_pipe.empty.wait(tmem_state.stage, tmem_state.phase ^ K.uint32(1))
                            K.cuda.tcgen05.encode_matrix_descriptor(
                                K.address_of(desc_a),
                                smem_kv[kv_state.stage].ptr_to(math_wg_i * umma_m, 0),
                                ldo=0,
                                sdo=desc_sdo,
                                swizzle=desc_swizzle,
                            )
                            K.cuda.tcgen05.encode_matrix_descriptor(
                                K.address_of(desc_b),
                                smem_q[q_state.stage].ptr_to(0, 0),
                                ldo=0,
                                sdo=desc_sdo,
                                swizzle=desc_swizzle,
                            )
                            for ki in range(head_dim // umma_k):
                                offset = ki * umma_k // 16
                                K.ptx[MMA](
                                    K.uint32(tmem_col)
                                    + K.Cast("uint32", tmem_state.stage) * K.uint32(umma_n),
                                    K.smem_desc_add_16B_offset(desc_a, offset),
                                    K.smem_desc_add_16B_offset(desc_b, offset),
                                    desc_i,
                                    K.uint32(0),
                                    K.uint32(0),
                                    K.uint32(0),
                                    K.uint32(0),
                                    K.ptx.pred(K.uint32(ki)),
                                )
                            tmem_pipe.full.arrive(tmem_state.stage)
                            tmem_state.advance()
                        K.assign(kv_idx, kv_idx + K.uint32(1))
                        kv_state.advance()
                    q_pipe.empty.arrive(q_state.stage)
                    K.assign(q_idx, q_idx + K.uint32(config.num_sms))
                    q_state.advance()
            K.cuda.warp_sync()

        with idle:
            pass

        # ---------------- warps 0-7: math + epilogue --------------------
        with math:
            math_thread_idx = K.Cast("uint32", K.tid_in_role())
            accum = K.alloc_local([num_heads], "float32")
            cached_weights = K.alloc_local([block_q, num_heads], "float32")
            q_row_off_base = K.local_scalar("uint64")
            q_state = K.RingState(num_q_stages)
            kv_state = K.RingState(num_kv_stages)
            tmem_state = K.RingState(
                num_tmem_stages, stage=warpgroup_idx, stride=num_math_warpgroups
            )
            q_idx = K.local_scalar("uint32")
            kv_idx = K.local_scalar("uint32")
            kv_offset = K.local_scalar("uint32")
            K.assign(q_idx, sm_idx_u32)
            with K.While(q_idx < num_q_blocks):
                load_schedule(q_idx)
                q_pipe.full.wait(q_state.stage, q_state.phase)
                with K.If(schedule_result[1] > K.uint32(0)), K.Then():
                    for weight_i in range(block_q):
                        for weight_j in range(num_heads // 4):
                            wc = weight_j * 4
                            K.ptx.ld.shared.v4.f32(
                                cached_weights[weight_i, wc],
                                cached_weights[weight_i, wc + 1],
                                cached_weights[weight_i, wc + 2],
                                cached_weights[weight_i, wc + 3],
                                smem_weights.ptr_to([q_state.stage, weight_i, wc]),
                            )
                    K.assign(
                        q_row_off_base,
                        K.Cast("uint64", q_idx * K.uint32(block_q))
                        * K.Cast("uint64", logits_stride),
                    )
                    # Publish the generic-proxy weight reads before this
                    # consumer releases the Q stage for a later TMA overwrite.
                    K.ptx.fence.proxy.async_.shared__cta()
                    K.assign(kv_offset, schedule_result[0] + math_thread_idx)
                    K.assign(kv_idx, K.uint32(0))
                    with K.While(kv_idx < schedule_result[1]):
                        kv_pipe.full.wait(kv_state.stage, kv_state.phase)
                        scale_kv = K.local_scalar("float32")
                        K.ptx.ld.shared.f32(
                            scale_kv, smem_kv_scales.ptr_to([kv_state.stage, math_thread_idx])
                        )
                        tmem_pipe.full.wait(tmem_state.stage, tmem_state.phase)
                        K.ptx.fence.proxy.async_.shared__cta()
                        kv_pipe.empty.arrive(kv_state.stage)
                        tmem_stage_base = K.uint32(tmem_col) + K.Cast(
                            "uint32", tmem_state.stage
                        ) * K.uint32(umma_n)
                        for q_inner_i in range(block_q):
                            tmem_addr = tmem_stage_base + K.uint32(q_inner_i * num_heads)
                            K.ptx[TC_LD](
                                *[accum[h] for h in range(num_heads // 2)],
                                K.cuda.get_tmem_addr(K.uint32(0), 0, tmem_addr),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            K.ptx[TC_LD](
                                *[accum[num_heads // 2 + h] for h in range(num_heads // 2)],
                                K.cuda.get_tmem_addr(
                                    K.uint32(0), 0, tmem_addr + K.uint32(num_heads // 2)
                                ),
                            )
                            K.ptx.tcgen05.wait__ld.sync.aligned()
                            reduced = wrelu_reduce(accum, cached_weights, q_inner_i)
                            result = K.Cast(logits_tir_dtype, scale_kv * reduced)
                            q_offset = q_row_off_base + K.Cast(
                                "uint64", K.uint32(q_inner_i)
                            ) * K.Cast("uint64", logits_stride)
                            if config.compressed_logits:
                                # Unconditional store with the column clamped
                                # into the row's stride padding; a range guard
                                # would become a BSSY/BRA region.
                                col = K.min(
                                    kv_offset - seq_k_start[q_inner_i], logits_stride - K.uint32(1)
                                )
                                store_logits(q_offset + K.Cast("uint64", col), result)
                            else:
                                store_logits(q_offset + K.Cast("uint64", kv_offset), result)
                        # Release the tmem stage once per kv block AFTER the
                        # token loop; inside the last token ptxas fuses it with
                        # the compressed guard branch.
                        tmem_pipe.empty.arrive(tmem_state.stage)
                        K.assign(kv_idx, kv_idx + K.uint32(1))
                        K.assign(kv_offset, kv_offset + K.uint32(block_kv))
                        kv_state.advance()
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
    sm100_fp8_mqa_logits.func = sm100_fp8_mqa_logits.func.with_attr(
        "tirx.persistent_kernel", True
    ).with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )
    return sm100_fp8_mqa_logits.func


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

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100f"})
    kernel = get_kernel(
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
        mod = tvm.IRModule({"main": kernel})
        # --ftz=false lets abs fold into FADD2 operand modifiers (ftz blocks it).
        os.environ["TVM_CUDA_NVRTC_EXTRA_OPTS"] = "--ftz=false"
        os.environ["TVM_CUDA_PTXAS_EXTRA_OPTS"] = "--allow-expensive-optimizations=true"
        # Level 6 avoids math-loop spills on the bf16 shapes (swept 4-10).
        os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = "6"
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


_compile_tirx_mqa_for_config = cache(_compile_tirx_mqa_for_config)


def _compile_tirx_mqa_kwargs(config: MQALogitsFP8Config) -> dict[str, Any]:
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


def _compile_tirx_mqa_key(config: MQALogitsFP8Config) -> tuple[tuple[str, Any], ...]:
    return tuple(_compile_tirx_mqa_kwargs(config).items())


def _compile_tirx_mqa(config: MQALogitsFP8Config, max_seqlen_k: int) -> Any:
    # The kernel is independent of seq_len/seq_len_kv/disable_cp/logits_stride (all
    # runtime): canonical values let the cache dedup to one kernel per structural config.
    del max_seqlen_k

    compile_kwargs = _compile_tirx_mqa_kwargs(config)
    return _compile_tirx_mqa_for_config(**compile_kwargs)


def _logits_storage_shape(config: MQALogitsFP8Config, max_seqlen_k: int) -> tuple[int, int]:
    if config.compressed_logits:
        # One extra block_kv of stride padding so len <= max_seqlen_k < stride always
        # holds — required by the kernel's clamp-to-padding compressed store.
        stride = _align_up(max_seqlen_k + config.block_kv, config.block_kv)
    else:
        stride = _align_up(config.seq_len_kv + config.block_kv, 8)
    return config.aligned_seq_len, stride


def _allocate_logits(config: MQALogitsFP8Config, max_seqlen_k: int) -> torch.Tensor:
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
    config: MQALogitsFP8Config = data["config"]
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
    config: MQALogitsFP8Config = data["config"]
    executable = invocation["executable"]
    logits = invocation["logits"]
    tensor_maps = invocation["tensor_maps"]
    _prepare_global_barrier(executable)
    executable.mod(
        config.seq_len,
        config.seq_len_kv,
        data["max_seqlen_k"],
        config.aligned_seq_len // config.block_q,
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
    config: MQALogitsFP8Config = data["config"]
    return data["deep_gemm"].fp8_fp4_mqa_logits(
        q=(data["q_in"], None),
        kv=data["kv_in"],
        weights=data["weights"],
        cu_seq_len_k_start=data["cu_seq_len_k_start"],
        cu_seq_len_k_end=data["cu_seq_len_k_end"],
        clean_logits=clean_logits,
        max_seqlen_k=data["max_seqlen_k"],
        logits_dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _expand_compressed_logits(logits: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsFP8Config = data["config"]
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
    config: MQALogitsFP8Config = data["config"]
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
    torch.cuda.empty_cache()

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
    "MQALogitsFP8Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
