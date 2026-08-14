# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of DeepGEMM's MQA logits kernel, FP8 variant.

Upstream source: deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh.
"""

import os
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

import tvm
from tirx_kernels.flashmla.utils._ir_builder import MBarrier, TCGen05Bar, TMABar
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6
_COMPILE_CACHE_NAMESPACE = "deepgemm.mqa_logits_fp8.compile"


_BUILDER_MISSING = object()


def _builder_runtime_condition(value):
    return value


def _builder_enter(frame):
    frames = frame.frames if hasattr(frame, "frames") else [frame]
    prim_func = next(
        candidate
        for candidate in reversed(IRBuilder.current().frames)
        if type(candidate).__name__ == "PrimFuncFrame"
    )
    for item in frames:
        prim_func.add_callback(lambda item=item: item.__exit__(None, None, None))
        item.__enter__()


def _builder_emit(value):
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if isinstance(value, IRBuilderFrame) or (
        hasattr(value, "frames") and hasattr(value, "__enter__")
    ):
        _builder_enter(value)
    elif tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)
    elif isinstance(value, int | bool):
        T.evaluate(tvm.tirx.const(value))


def _builder_alloc_scalar(name, dtype):
    scalar = T.local_scalar(dtype)
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def _builder_scalar(name, value, dtype):
    scalar = _builder_alloc_scalar(name, dtype)
    T.buffer_store(scalar.buffer, value, scalar.indices)
    return scalar


def _builder_buffer(name, shape, dtype):
    buffer = T.alloc_local(shape, dtype)
    IRBuilder.name(name, buffer)
    return buffer


def _builder_bind(name, value, type_annotation=None):
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_assign(name, value, previous=_BUILDER_MISSING):
    if isinstance(value, I.meta_var):
        return value.value
    if previous is not _BUILDER_MISSING:
        if isinstance(previous, T.scalar_wrapper | tvm.tirx.expr.BufferLoad):
            target = previous.scalar if isinstance(previous, T.scalar_wrapper) else previous
            T.buffer_store(target.buffer, value, target.indices)
            return target
        if (
            is_buffer_var(previous)
            and len(previous.ty.shape) == 1
            and bool(previous.ty.shape[0] == 1)
        ):
            try:
                T.buffer_store(previous, value, [0])
                return previous
            except TypeError:
                pass
    if getattr(type(value), "_is_meta_class", False):
        name_meta_class_value(name, value)
        return value
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _builder_assign(f"{name}_{index}", item)
        return value
    if is_buffer_var(value) or isinstance(value, IterVar | Layout):
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Var):
        if isinstance(value.ty, tvm.ir.PointerType):
            return _builder_bind(name, value, value.ty)
        IRBuilder.name(name, value)
        return value
    if isinstance(value, tvm.ir.Expr) and isinstance(
        getattr(value, "ty", None), tvm.ir.PointerType
    ):
        return _builder_bind(name, value, value.ty)
    if isinstance(value, tvm.ir.Expr) and tvm.ir.is_prim_expr(value):
        return _builder_scalar(name, value, str(value.ty.dtype))
    if isinstance(value, tvm.tirx.expr.ExprOp):
        return _builder_scalar(name, value, "bool")
    return value


def _builder_assign_many(names, values, previous):
    return tuple(
        _builder_assign(name, value, old) for name, value, old in zip(names, values, previous)
    )


@T.meta_class
class Pipeline:
    """Builder-native full/empty barrier pair matching TVM's pipeline helper."""

    def __init__(
        self,
        pool,
        stages,
        *,
        full,
        empty,
        init_full=1,
        init_empty=1,
        empty_phase_offset=0,
        leader=None,
    ):
        barrier_kinds = {"tma": TMABar, "tcgen05": TCGen05Bar, "mbar": MBarrier}
        self.stages = stages
        self.full = barrier_kinds[full](pool, stages, leader=leader)
        self.full.init(init_full)
        self.empty = barrier_kinds[empty](
            pool, stages, phase_offset=empty_phase_offset, leader=leader
        )
        self.empty.init(init_empty)


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
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

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
    # q/kv carry a 128B MMA swizzle layout (head_dim * 1B = 128 B/row); the
    # SWIZZLE_128B_ATOM is 8 rows * 1024 B, so their SMEM base must be 1024-aligned.
    _SWZ = {
        32: SwizzleMode.SWIZZLE_32B_ATOM,
        64: SwizzleMode.SWIZZLE_64B_ATOM,
        128: SwizzleMode.SWIZZLE_128B_ATOM,
    }[head_dim]
    swizzle_alignment = 8 * head_dim

    # TMEM accumulator: umma_n cols/stage, 3 stages round-robin between the two math
    # warpgroups; alloc rounds up to the hw bucket (384 -> 512, full SM100 TMEM).
    num_accum_tmem_cols = block_q * num_heads * num_tmem_stages
    num_tmem_cols = 32
    if num_accum_tmem_cols > 32:
        num_tmem_cols = 64
    if num_accum_tmem_cols > 64:
        num_tmem_cols = 128
    if num_accum_tmem_cols > 128:
        num_tmem_cols = 256
    if num_accum_tmem_cols > 256:
        num_tmem_cols = 512
    if num_tmem_cols > 512:
        raise ValueError(f"tensor memory columns {num_tmem_cols} exceeds SM100 single-CTA limit")
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"
    cache_policy_evict_normal = T.uint64(0x1000000000000000)
    tma_g2s_1d = (
        "cp.async.bulk.tensor.1d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    tma_g2s_2d = (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    tcgen05_mma = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    tcgen05_ld = f"tcgen05.ld.sync.aligned.32x32b.x{num_heads // 2}.b32"
    desc_sdo = head_dim // 2
    desc_swizzle = {32: 1, 64: 2, 128: 3}[head_dim]

    def wrelu_reduce(accum, weights, row):
        sum_0 = T.alloc_local((1,), "uint64")
        sum_1 = T.alloc_local((1,), "uint64")
        accum_pair = T.alloc_local((1,), "uint64")
        abs_pair = T.alloc_local((1,), "uint64")
        relu_pair = T.alloc_local((1,), "uint64")
        weight_pair = T.alloc_local((1,), "uint64")
        abs_lo = T.alloc_local((1,), "float32")
        abs_hi = T.alloc_local((1,), "float32")
        total = T.alloc_local((1,), "uint64")
        total_lo = T.alloc_local((1,), "float32")
        total_hi = T.alloc_local((1,), "float32")
        result = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.mov.b64(sum_0[0], T.float32(0), T.float32(0)))
        T.evaluate(T.ptx.mov.b64(sum_1[0], T.float32(0), T.float32(0)))
        for head in range(0, num_heads, 4):
            T.evaluate(T.ptx.mov.b64(accum_pair[0], accum[head], accum[head + 1]))
            T.evaluate(T.ptx.abs.f32(abs_lo[0], accum[head]))
            T.evaluate(T.ptx.abs.f32(abs_hi[0], accum[head + 1]))
            T.evaluate(T.ptx.mov.b64(abs_pair[0], abs_lo[0], abs_hi[0]))
            T.evaluate(T.ptx.add.rn.f32x2(relu_pair[0], accum_pair[0], abs_pair[0]))
            T.evaluate(T.ptx.mov.b64(weight_pair[0], weights[row, head], weights[row, head + 1]))
            T.evaluate(T.ptx.fma.rn.f32x2(sum_0[0], relu_pair[0], weight_pair[0], sum_0[0]))

            T.evaluate(T.ptx.mov.b64(accum_pair[0], accum[head + 2], accum[head + 3]))
            T.evaluate(T.ptx.abs.f32(abs_lo[0], accum[head + 2]))
            T.evaluate(T.ptx.abs.f32(abs_hi[0], accum[head + 3]))
            T.evaluate(T.ptx.mov.b64(abs_pair[0], abs_lo[0], abs_hi[0]))
            T.evaluate(T.ptx.add.rn.f32x2(relu_pair[0], accum_pair[0], abs_pair[0]))
            T.evaluate(
                T.ptx.mov.b64(weight_pair[0], weights[row, head + 2], weights[row, head + 3])
            )
            T.evaluate(T.ptx.fma.rn.f32x2(sum_1[0], relu_pair[0], weight_pair[0], sum_1[0]))
        T.evaluate(T.ptx.add.rn.f32x2(total[0], sum_0[0], sum_1[0]))
        T.evaluate(T.ptx.mov.b64(total_lo[0], total_hi[0], total[0]))
        T.evaluate(T.ptx.add.rn.f32(result[0], total_lo[0], total_hi[0]))
        T.evaluate(T.ptx.mul.rn.f32(result[0], result[0], T.float32(0.5)))
        return result[0]

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def named_barrier_sync_8(count):
        T.evaluate(T.ptx.bar.sync(8, T.uint32(count)))

    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("sm100_fp8_mqa_logits")
            seq_len = T.arg("seq_len", T.uint32())
            seq_len_kv = T.arg("seq_len_kv", T.uint32())
            max_seqlen_k = T.arg("max_seqlen_k", T.uint32())
            logits_stride = T.arg("logits_stride", T.uint32())
            cu_seq_len_k_start_h = T.arg("cu_seq_len_k_start_h", T.handle())
            cu_seq_len_k_end_h = T.arg("cu_seq_len_k_end_h", T.handle())
            logits_h = T.arg("logits_h", T.handle())
            q_gmem_h = T.arg("q_gmem_h", T.handle())
            kv_gmem_h = T.arg("kv_gmem_h", T.handle())
            kv_scales_gmem_h = T.arg("kv_scales_gmem_h", T.handle())
            weights_gmem_h = T.arg("weights_gmem_h", T.handle())
            # seq_len / seq_len_kv are RUNTIME like DeepGEMM: one compiled kernel serves any
            # length; structure stays compile-time. match_buffer must precede device_entry.
            cu_seq_len_k_start = _builder_assign(
                "cu_seq_len_k_start",
                T.match_buffer(cu_seq_len_k_start_h, (seq_len,), "int32"),
                locals().get("cu_seq_len_k_start", _BUILDER_MISSING),
            )
            cu_seq_len_k_end = _builder_assign(
                "cu_seq_len_k_end",
                T.match_buffer(cu_seq_len_k_end_h, (seq_len,), "int32"),
                locals().get("cu_seq_len_k_end", _BUILDER_MISSING),
            )
            logits = _builder_assign(
                "logits",
                T.match_buffer(
                    logits_h,
                    (
                        (T.cast(seq_len, "int32") + T.int32(block_q - 1))
                        // T.int32(block_q)
                        * T.int32(block_q),
                        T.cast(logits_stride, "int32"),
                    ),
                    config.logits_dtype,
                ),
                locals().get("logits", _BUILDER_MISSING),
            )
            q_gmem = _builder_assign(
                "q_gmem",
                T.match_buffer(q_gmem_h, (seq_len * num_heads, head_dim), "uint8"),
                locals().get("q_gmem", _BUILDER_MISSING),
            )
            kv_gmem = _builder_assign(
                "kv_gmem",
                T.match_buffer(kv_gmem_h, (seq_len_kv, head_dim), "uint8"),
                locals().get("kv_gmem", _BUILDER_MISSING),
            )
            kv_scales_gmem = _builder_assign(
                "kv_scales_gmem",
                T.match_buffer(kv_scales_gmem_h, (seq_len_kv,), "float32"),
                locals().get("kv_scales_gmem", _BUILDER_MISSING),
            )
            weights_gmem = _builder_assign(
                "weights_gmem",
                T.match_buffer(weights_gmem_h, (seq_len, num_heads), "float32"),
                locals().get("weights_gmem", _BUILDER_MISSING),
            )

            # Runtime lengths remain part of the public kernel ABI, so construct
            # the four tensor maps in the host prologue instead of specializing
            # sequence lengths into the PrimFunc.  Q/KV retain the dispatcher's
            # head-dimension swizzle; weights and per-token KV scales are linear.
            kv_scales_gmem_tensormap = _builder_bind(
                "kv_scales_gmem_tensormap", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    kv_scales_gmem_tensormap,
                    "float32",
                    1,
                    kv_scales_gmem.data,
                    seq_len_kv,
                    block_kv,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )
            kv_gmem_tensormap = _builder_bind(
                "kv_gmem_tensormap", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    kv_gmem_tensormap,
                    "uint8",
                    2,
                    kv_gmem.data,
                    head_dim,
                    seq_len_kv,
                    head_dim,
                    head_dim,
                    block_kv,
                    1,
                    1,
                    0,
                    _SWZ.value,
                    2,
                    0,
                )
            )
            weights_gmem_tensormap = _builder_bind(
                "weights_gmem_tensormap", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    weights_gmem_tensormap,
                    "float32",
                    2,
                    weights_gmem.data,
                    num_heads,
                    seq_len,
                    num_heads * 4,
                    num_heads,
                    block_q,
                    1,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )
            q_gmem_tensormap = _builder_bind(
                "q_gmem_tensormap", T.tvm_stack_alloca("tensormap", 1), T.TensorMap()
            )
            _builder_emit(
                T.call_packed(
                    "runtime.cuTensorMapEncodeTiled",
                    q_gmem_tensormap,
                    "uint8",
                    2,
                    q_gmem.data,
                    head_dim,
                    seq_len * num_heads,
                    head_dim,
                    head_dim,
                    block_q * num_heads,
                    1,
                    1,
                    0,
                    _SWZ.value,
                    2,
                    0,
                )
            )
            _builder_emit(T.device_entry())
            _builder_emit(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            # TIRX_TRANSCRIBE_START sm100_fp8_mqa_logits
            aligned_sl = _builder_scalar(
                "aligned_sl",
                (
                    (T.cast(seq_len, "int32") + T.int32(block_q - 1))
                    // T.int32(block_q)
                    * T.int32(block_q)
                ),
                "int32",
            )
            logits_flat = _builder_assign(
                "logits_flat",
                T.decl_buffer(
                    (aligned_sl * T.cast(logits_stride, "int32"),),
                    logits_tir_dtype,
                    data=logits.data,
                    scope="global",
                ),
                locals().get("logits_flat", _BUILDER_MISSING),
            )
            num_q_blocks = _builder_scalar(
                "num_q_blocks", (seq_len + T.uint32(block_q - 1)) // T.uint32(block_q), "uint32"
            )
            sm_idx = _builder_assign(
                "sm_idx", T.cta_id([config.num_sms]), locals().get("sm_idx", _BUILDER_MISSING)
            )
            sm_idx_u32 = _builder_bind("sm_idx_u32", T.cast(sm_idx, "uint32"), None)
            warp_idx = _builder_assign(
                "warp_idx", T.warp_id([num_warps]), locals().get("warp_idx", _BUILDER_MISSING)
            )
            warp_idx_u32 = _builder_bind("warp_idx_u32", T.cast(warp_idx, "uint32"), None)
            warpgroup_idx = _builder_assign(
                "warpgroup_idx",
                T.warpgroup_id([num_warps // 4]),
                locals().get("warpgroup_idx", _BUILDER_MISSING),
            )
            lane_idx = _builder_assign(
                "lane_idx", T.lane_id([32]), locals().get("lane_idx", _BUILDER_MISSING)
            )
            lane_idx_u32 = _builder_bind("lane_idx_u32", T.cast(lane_idx, "uint32"), None)

            # Match the former dispatcher placement: one elected lane of warp 0
            # prefetches every descriptor before pipeline traffic begins.
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_gmem_tensormap)))
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(weights_gmem_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_gmem_tensormap))
                                )
                            )
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            _builder_emit(
                                T.evaluate(
                                    T.ptx.prefetch.tensormap(T.address_of(kv_scales_gmem_tensormap))
                                )
                            )

            # SMEMPool owns the smem offsets; q/kv carry the 128B MMA swizzle layout. Under
            # the pool .data is the arena base — use .ptr_to([0,...]) for a buffer's start.
            pool = _builder_assign("pool", T.SMEMPool(), locals().get("pool", _BUILDER_MISSING))
            smem_q = _builder_assign(
                "smem_q",
                pool.alloc(
                    (num_q_stages, block_q * num_heads, head_dim),
                    "uint8",
                    scope="shared.dyn",
                    align=swizzle_alignment,
                    layout=mma_shared_layout(
                        "uint8", _SWZ, (num_q_stages, block_q * num_heads, head_dim)
                    ),
                ),
                locals().get("smem_q", _BUILDER_MISSING),
            )
            smem_weights = _builder_assign(
                "smem_weights",
                pool.alloc((num_q_stages, block_q, num_heads), "float32", align=16),
                locals().get("smem_weights", _BUILDER_MISSING),
            )
            smem_kv = _builder_assign(
                "smem_kv",
                pool.alloc(
                    (num_kv_stages, block_kv, head_dim),
                    "uint8",
                    scope="shared.dyn",
                    align=swizzle_alignment,
                    layout=mma_shared_layout("uint8", _SWZ, (num_kv_stages, block_kv, head_dim)),
                ),
                locals().get("smem_kv", _BUILDER_MISSING),
            )
            smem_kv_scales = _builder_assign(
                "smem_kv_scales",
                pool.alloc((num_kv_stages, block_kv), "float32", align=16),
                locals().get("smem_kv_scales", _BUILDER_MISSING),
            )
            # Producer/consumer barrier pairs as Pipeline objects (full = data ready, empty
            # = slot free); each Pipeline runs mbarrier.init itself, no separate init loop.
            q_pipe = _builder_assign(
                "q_pipe",
                Pipeline(
                    pool,
                    num_q_stages,
                    full="tma",
                    empty="mbar",
                    init_full=1,
                    init_empty=num_math_threads + 1,
                ),
                locals().get("q_pipe", _BUILDER_MISSING),
            )
            kv_pipe = _builder_assign(
                "kv_pipe",
                Pipeline(
                    pool,
                    num_kv_stages,
                    full="tma",
                    empty="mbar",
                    init_full=1,
                    init_empty=num_math_threads,
                ),
                locals().get("kv_pipe", _BUILDER_MISSING),
            )
            tmem_pipe = _builder_assign(
                "tmem_pipe",
                Pipeline(
                    pool, num_tmem_stages, full="tcgen05", empty="mbar", init_full=1, init_empty=128
                ),
                locals().get("tmem_pipe", _BUILDER_MISSING),
            )
            tmem_ptr_in_smem = _builder_assign(
                "tmem_ptr_in_smem",
                pool.alloc((1,), "uint32", align=4),
                locals().get("tmem_ptr_in_smem", _BUILDER_MISSING),
            )
            _builder_emit(pool.commit())
            # TMEMPool gives a CONSTANT 0-based col_start so tmem addressing folds the base
            # into the col offset; manual tcgen05.alloc/dealloc below keep the lifecycle.
            tmem_pool = _builder_assign(
                "tmem_pool",
                T.TMEMPool(pool, total_cols=num_tmem_cols, cta_group=1, tmem_addr=tmem_ptr_in_smem),
                locals().get("tmem_pool", _BUILDER_MISSING),
            )
            tmem = _builder_assign(
                "tmem",
                tmem_pool.alloc(
                    (128, num_tmem_cols), "float32", layout=tmem_layout, cols=num_tmem_cols
                ),
                locals().get("tmem", _BUILDER_MISSING),
            )
            seq_k_start = _builder_assign(
                "seq_k_start",
                T.alloc_local((block_q,), "uint32"),
                locals().get("seq_k_start", _BUILDER_MISSING),
            )
            seq_k_end = _builder_assign(
                "seq_k_end",
                T.alloc_local((block_q,), "uint32"),
                locals().get("seq_k_end", _BUILDER_MISSING),
            )
            schedule_result = _builder_assign(
                "schedule_result",
                T.alloc_local((2,), "uint32"),
                locals().get("schedule_result", _BUILDER_MISSING),
            )

            def store_logits(flat_offset, value):
                # Scalar predicated store: per-thread non-contiguous output, so TMA/bulk
                # does not apply.  Both dtypes use explicit PTX global stores.
                if config.logits_dtype == "float32":
                    _builder_emit(T.ptx.st.global_.f32(logits_flat.ptr_to([flat_offset]), value))
                else:
                    _builder_emit(T.ptx.st.global_.b16(logits_flat.ptr_to([flat_offset]), value))

            def load_schedule(q_idx):
                schedule_start = _builder_scalar("schedule_start", T.uint32(0xFFFFFFFF), "uint32")
                schedule_end = _builder_scalar("schedule_end", T.uint32(0), "uint32")
                with T.unroll(0, block_q) as schedule_i:
                    IRBuilder.name("schedule_i", schedule_i)
                    row_idx = _builder_scalar(
                        "row_idx",
                        T.min(
                            q_idx * T.uint32(block_q) + T.uint32(schedule_i), seq_len - T.uint32(1)
                        ),
                        "uint32",
                    )
                    row_start = _builder_alloc_scalar("row_start", "int32")
                    row_end = _builder_alloc_scalar("row_end", "int32")
                    _builder_emit(
                        T.ptx.ld.global_.s32(
                            row_start, cu_seq_len_k_start.ptr_to([T.cast(row_idx, "int32")])
                        )
                    )
                    T.buffer_store(
                        seq_k_start, T.min(T.cast(row_start, "uint32"), seq_len_kv), [schedule_i]
                    )
                    _builder_emit(
                        T.ptx.ld.global_.s32(
                            row_end, cu_seq_len_k_end.ptr_to([T.cast(row_idx, "int32")])
                        )
                    )
                    T.buffer_store(
                        seq_k_end, T.min(T.cast(row_end, "uint32"), seq_len_kv), [schedule_i]
                    )
                    schedule_start = _builder_assign(
                        "schedule_start",
                        T.min(schedule_start, seq_k_start[schedule_i]),
                        locals().get("schedule_start", _BUILDER_MISSING),
                    )
                    schedule_end = _builder_assign(
                        "schedule_end",
                        T.max(schedule_end, seq_k_end[schedule_i]),
                        locals().get("schedule_end", _BUILDER_MISSING),
                    )
                schedule_start = _builder_assign(
                    "schedule_start",
                    schedule_start // T.uint32(4) * T.uint32(4),
                    locals().get("schedule_start", _BUILDER_MISSING),
                )
                num_kv_blocks = _builder_assign(
                    "num_kv_blocks",
                    (schedule_end - schedule_start + T.uint32(block_kv - 1)) // T.uint32(block_kv),
                    locals().get("num_kv_blocks", _BUILDER_MISSING),
                )
                T.buffer_store(schedule_result, schedule_start, [0])
                T.buffer_store(schedule_result, num_kv_blocks, [1])

            # Pipeline constructors already ran mbarrier.init; fence + cta_sync publish them.
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            with T.If(warp_idx == spec_warp_start + 2):
                with T.Then():
                    _builder_emit(
                        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                            T.address_of(tmem_ptr_in_smem[0]), T.uint32(num_tmem_cols)
                        )
                    )
            _builder_emit(T.cuda.cta_sync())

            _builder_emit(cuda_grid_dependency_synchronize())

            with T.If(warp_idx == spec_warp_start):
                with T.Then():
                    _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            # Ring cursors with subtract-wrap (DeepGEMM RingPipeline): avoids ptxas
                            # magic-number division for `% kNumStages` on these hot paths.
                            q_stage_idx = _builder_scalar("q_stage_idx", T.uint32(0), "uint32")
                            q_phase = _builder_scalar("q_phase", T.uint32(0), "uint32")
                            q_idx = _builder_scalar("q_idx", sm_idx_u32, "uint32")
                            with T.While(q_idx < num_q_blocks):
                                _builder_emit(q_pipe.empty.wait(q_stage_idx, q_phase ^ T.uint32(1)))
                                # Keep scheduler row bases in u32; cast only the TensorMap
                                # coordinates consumed by the explicit TMA instruction.
                                q_row0 = _builder_scalar(
                                    "q_row0", q_idx * T.uint32(block_q * num_heads), "uint32"
                                )
                                _builder_emit(
                                    T.ptx[tma_g2s_2d](
                                        smem_q.ptr_to([q_stage_idx, 0, 0]),
                                        T.address_of(q_gmem_tensormap),
                                        T.int32(0),
                                        T.cast(q_row0, "int32"),
                                        q_pipe.full.ptr_to([q_stage_idx]),
                                        cache_policy_evict_normal,
                                    )
                                )
                                q_blk0 = _builder_scalar(
                                    "q_blk0", q_idx * T.uint32(block_q), "uint32"
                                )
                                _builder_emit(
                                    T.ptx[tma_g2s_2d](
                                        smem_weights.ptr_to([q_stage_idx, 0, 0]),
                                        T.address_of(weights_gmem_tensormap),
                                        T.int32(0),
                                        T.cast(q_blk0, "int32"),
                                        q_pipe.full.ptr_to([q_stage_idx]),
                                        cache_policy_evict_normal,
                                    )
                                )
                                _builder_emit(
                                    q_pipe.full.arrive(
                                        q_stage_idx,
                                        tx_count=smem_q_size_per_stage + smem_weight_size_per_stage,
                                    )
                                )
                                q_idx = _builder_assign(
                                    "q_idx",
                                    q_idx + T.uint32(config.num_sms),
                                    locals().get("q_idx", _BUILDER_MISSING),
                                )
                                q_stage_idx = _builder_assign(
                                    "q_stage_idx",
                                    q_stage_idx + T.uint32(1),
                                    locals().get("q_stage_idx", _BUILDER_MISSING),
                                )
                                with T.If(q_stage_idx >= T.uint32(num_q_stages)):
                                    with T.Then():
                                        q_stage_idx = _builder_assign(
                                            "q_stage_idx",
                                            q_stage_idx - T.uint32(num_q_stages),
                                            locals().get("q_stage_idx", _BUILDER_MISSING),
                                        )
                                        q_phase = _builder_assign(
                                            "q_phase",
                                            q_phase ^ T.uint32(1),
                                            locals().get("q_phase", _BUILDER_MISSING),
                                        )
                    _builder_emit(T.cuda.warp_sync())
                with T.Else():
                    with T.If(warp_idx == spec_warp_start + 1):
                        with T.Then():
                            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                            with T.If(T.cuda.elect_sync()):
                                with T.Then():
                                    kv_stage_idx = _builder_scalar(
                                        "kv_stage_idx", T.uint32(0), "uint32"
                                    )
                                    kv_phase = _builder_scalar("kv_phase", T.uint32(0), "uint32")
                                    q_idx = _builder_scalar("q_idx", sm_idx_u32, "uint32")
                                    with T.While(q_idx < num_q_blocks):
                                        _builder_emit(load_schedule(q_idx))
                                        kv_start = _builder_scalar(
                                            "kv_start", schedule_result[0], "uint32"
                                        )
                                        num_kv_blocks = _builder_scalar(
                                            "num_kv_blocks", schedule_result[1], "uint32"
                                        )
                                        kv_idx = _builder_scalar("kv_idx", T.uint32(0), "uint32")
                                        with T.While(kv_idx < num_kv_blocks):
                                            _builder_emit(
                                                kv_pipe.empty.wait(
                                                    kv_stage_idx, kv_phase ^ T.uint32(1)
                                                )
                                            )
                                            kv_row0 = _builder_scalar(
                                                "kv_row0",
                                                kv_start + kv_idx * T.uint32(block_kv),
                                                "uint32",
                                            )
                                            _builder_emit(
                                                T.ptx[tma_g2s_2d](
                                                    smem_kv.ptr_to([kv_stage_idx, 0, 0]),
                                                    T.address_of(kv_gmem_tensormap),
                                                    T.int32(0),
                                                    T.cast(kv_row0, "int32"),
                                                    kv_pipe.full.ptr_to([kv_stage_idx]),
                                                    cache_policy_evict_normal,
                                                )
                                            )
                                            _builder_emit(
                                                T.ptx[tma_g2s_1d](
                                                    smem_kv_scales.ptr_to([kv_stage_idx, 0]),
                                                    T.address_of(kv_scales_gmem_tensormap),
                                                    T.cast(kv_row0, "int32"),
                                                    kv_pipe.full.ptr_to([kv_stage_idx]),
                                                    cache_policy_evict_normal,
                                                )
                                            )
                                            _builder_emit(
                                                kv_pipe.full.arrive(
                                                    kv_stage_idx,
                                                    tx_count=smem_kv_size_per_stage
                                                    + smem_kv_scale_size_per_stage,
                                                )
                                            )
                                            kv_idx = _builder_assign(
                                                "kv_idx",
                                                kv_idx + T.uint32(1),
                                                locals().get("kv_idx", _BUILDER_MISSING),
                                            )
                                            kv_stage_idx = _builder_assign(
                                                "kv_stage_idx",
                                                kv_stage_idx + T.uint32(1),
                                                locals().get("kv_stage_idx", _BUILDER_MISSING),
                                            )
                                            with T.If(kv_stage_idx >= T.uint32(num_kv_stages)):
                                                with T.Then():
                                                    kv_stage_idx = _builder_assign(
                                                        "kv_stage_idx",
                                                        kv_stage_idx - T.uint32(num_kv_stages),
                                                        locals().get(
                                                            "kv_stage_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    kv_phase = _builder_assign(
                                                        "kv_phase",
                                                        kv_phase ^ T.uint32(1),
                                                        locals().get("kv_phase", _BUILDER_MISSING),
                                                    )
                                        q_idx = _builder_assign(
                                            "q_idx",
                                            q_idx + T.uint32(config.num_sms),
                                            locals().get("q_idx", _BUILDER_MISSING),
                                        )
                        with T.Else():
                            with T.If(warp_idx == spec_warp_start + 2):
                                with T.Then():
                                    _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                                    tmem_allocated = _builder_alloc_scalar(
                                        "tmem_allocated", "uint32"
                                    )
                                    _builder_emit(
                                        T.ptx.ld.shared.u32(
                                            tmem_allocated, tmem_ptr_in_smem.ptr_to([0])
                                        )
                                    )
                                    _builder_emit(
                                        T.cuda.trap_when_assert_failed(
                                            tmem_allocated == T.uint32(0)
                                        )
                                    )
                                    # Dense E4M3 UMMA metadata is invariant across the persistent loop.
                                    # Keep the runtime descriptor encoding that the former dispatcher
                                    # produced, then update only the per-stage matrix descriptors.
                                    desc_i = _builder_alloc_scalar("desc_i", "uint32")
                                    _builder_emit(
                                        T.cuda.tcgen05.encode_instr_descriptor(
                                            T.address_of(desc_i),
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
                                    )
                                    runtime_instr_desc = _builder_scalar(
                                        "runtime_instr_desc",
                                        T.shift_left(T.cast(desc_i, "uint64"), T.uint64(32)),
                                        "uint64",
                                    )
                                    runtime_instr_desc_hi = _builder_scalar(
                                        "runtime_instr_desc_hi",
                                        T.cast(
                                            T.shift_right(runtime_instr_desc, T.uint64(32)),
                                            "uint32",
                                        ),
                                        "uint32",
                                    )
                                    desc_a = _builder_alloc_scalar("desc_a", "uint64")
                                    desc_b = _builder_alloc_scalar("desc_b", "uint64")
                                    # FP8 (E4M3) views over the head-dimension-swizzled uint8 SMEM.
                                    smem_q_fp8 = _builder_assign(
                                        "smem_q_fp8",
                                        smem_q.view("float8_e4m3fn"),
                                        locals().get("smem_q_fp8", _BUILDER_MISSING),
                                    )
                                    smem_kv_fp8 = _builder_assign(
                                        "smem_kv_fp8",
                                        smem_kv.view("float8_e4m3fn"),
                                        locals().get("smem_kv_fp8", _BUILDER_MISSING),
                                    )
                                    # Whole MMA-warp loop in one elect scope: ring cursors stay elect-lane
                                    # locals on the uniform datapath (no R2UR per use).
                                    with T.If(T.cuda.elect_sync()):
                                        with T.Then():
                                            q_stage_idx = _builder_scalar(
                                                "q_stage_idx", T.uint32(0), "uint32"
                                            )
                                            q_phase = _builder_scalar(
                                                "q_phase", T.uint32(0), "uint32"
                                            )
                                            kv_stage_idx = _builder_scalar(
                                                "kv_stage_idx", T.uint32(0), "uint32"
                                            )
                                            kv_phase = _builder_scalar(
                                                "kv_phase", T.uint32(0), "uint32"
                                            )
                                            tmem_stage_idx = _builder_scalar(
                                                "tmem_stage_idx", T.uint32(0), "uint32"
                                            )
                                            tmem_phase = _builder_scalar(
                                                "tmem_phase", T.uint32(0), "uint32"
                                            )
                                            q_idx = _builder_scalar("q_idx", sm_idx_u32, "uint32")
                                            with T.While(q_idx < num_q_blocks):
                                                _builder_emit(load_schedule(q_idx))
                                                num_kv_blocks = _builder_scalar(
                                                    "num_kv_blocks", schedule_result[1], "uint32"
                                                )
                                                _builder_emit(
                                                    q_pipe.full.wait(q_stage_idx, q_phase)
                                                )
                                                kv_idx = _builder_scalar(
                                                    "kv_idx", T.uint32(0), "uint32"
                                                )
                                                with T.While(kv_idx < num_kv_blocks):
                                                    _builder_emit(
                                                        kv_pipe.full.wait(kv_stage_idx, kv_phase)
                                                    )
                                                    with T.unroll(
                                                        0, num_math_warpgroups
                                                    ) as math_wg_i:
                                                        IRBuilder.name("math_wg_i", math_wg_i)
                                                        tmem_addr = _builder_scalar(
                                                            "tmem_addr",
                                                            tmem_stage_idx * T.uint32(umma_n),
                                                            "uint32",
                                                        )
                                                        _builder_emit(
                                                            tmem_pipe.empty.wait(
                                                                tmem_stage_idx,
                                                                tmem_phase ^ T.uint32(1),
                                                            )
                                                        )
                                                        # D = KV @ Q^T.  Issue K=32 phases in increasing order;
                                                        # phase zero overwrites and later phases accumulate.
                                                        with T.unroll(0, head_dim // umma_k) as ki:
                                                            IRBuilder.name("ki", ki)
                                                            _builder_emit(
                                                                T.cuda.tcgen05.encode_matrix_descriptor(
                                                                    T.address_of(desc_a),
                                                                    smem_kv_fp8.ptr_to(
                                                                        [
                                                                            kv_stage_idx,
                                                                            math_wg_i * umma_m,
                                                                            ki * umma_k,
                                                                        ]
                                                                    ),
                                                                    ldo=0,
                                                                    sdo=desc_sdo,
                                                                    swizzle=desc_swizzle,
                                                                )
                                                            )
                                                            _builder_emit(
                                                                T.cuda.tcgen05.encode_matrix_descriptor(
                                                                    T.address_of(desc_b),
                                                                    smem_q_fp8.ptr_to(
                                                                        [
                                                                            q_stage_idx,
                                                                            0,
                                                                            ki * umma_k,
                                                                        ]
                                                                    ),
                                                                    ldo=0,
                                                                    sdo=desc_sdo,
                                                                    swizzle=desc_swizzle,
                                                                )
                                                            )
                                                            _builder_emit(
                                                                T.ptx[tcgen05_mma](
                                                                    tmem_addr,
                                                                    desc_a,
                                                                    desc_b,
                                                                    runtime_instr_desc_hi,
                                                                    T.uint32(0),
                                                                    T.uint32(0),
                                                                    T.uint32(0),
                                                                    T.uint32(0),
                                                                    T.cast(ki != 0, "bool"),
                                                                )
                                                            )
                                                        _builder_emit(
                                                            tmem_pipe.full.arrive(tmem_stage_idx)
                                                        )
                                                        tmem_stage_idx = _builder_assign(
                                                            "tmem_stage_idx",
                                                            tmem_stage_idx + T.uint32(1),
                                                            locals().get(
                                                                "tmem_stage_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(
                                                            tmem_stage_idx
                                                            >= T.uint32(num_tmem_stages)
                                                        ):
                                                            with T.Then():
                                                                tmem_stage_idx = _builder_assign(
                                                                    "tmem_stage_idx",
                                                                    tmem_stage_idx
                                                                    - T.uint32(num_tmem_stages),
                                                                    locals().get(
                                                                        "tmem_stage_idx",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                tmem_phase = _builder_assign(
                                                                    "tmem_phase",
                                                                    tmem_phase ^ T.uint32(1),
                                                                    locals().get(
                                                                        "tmem_phase",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                    kv_idx = _builder_assign(
                                                        "kv_idx",
                                                        kv_idx + T.uint32(1),
                                                        locals().get("kv_idx", _BUILDER_MISSING),
                                                    )
                                                    kv_stage_idx = _builder_assign(
                                                        "kv_stage_idx",
                                                        kv_stage_idx + T.uint32(1),
                                                        locals().get(
                                                            "kv_stage_idx", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    with T.If(
                                                        kv_stage_idx >= T.uint32(num_kv_stages)
                                                    ):
                                                        with T.Then():
                                                            kv_stage_idx = _builder_assign(
                                                                "kv_stage_idx",
                                                                kv_stage_idx
                                                                - T.uint32(num_kv_stages),
                                                                locals().get(
                                                                    "kv_stage_idx", _BUILDER_MISSING
                                                                ),
                                                            )
                                                            kv_phase = _builder_assign(
                                                                "kv_phase",
                                                                kv_phase ^ T.uint32(1),
                                                                locals().get(
                                                                    "kv_phase", _BUILDER_MISSING
                                                                ),
                                                            )
                                                _builder_emit(q_pipe.empty.arrive(q_stage_idx))
                                                q_idx = _builder_assign(
                                                    "q_idx",
                                                    q_idx + T.uint32(config.num_sms),
                                                    locals().get("q_idx", _BUILDER_MISSING),
                                                )
                                                q_stage_idx = _builder_assign(
                                                    "q_stage_idx",
                                                    q_stage_idx + T.uint32(1),
                                                    locals().get("q_stage_idx", _BUILDER_MISSING),
                                                )
                                                with T.If(q_stage_idx >= T.uint32(num_q_stages)):
                                                    with T.Then():
                                                        q_stage_idx = _builder_assign(
                                                            "q_stage_idx",
                                                            q_stage_idx - T.uint32(num_q_stages),
                                                            locals().get(
                                                                "q_stage_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        q_phase = _builder_assign(
                                                            "q_phase",
                                                            q_phase ^ T.uint32(1),
                                                            locals().get(
                                                                "q_phase", _BUILDER_MISSING
                                                            ),
                                                        )
                                    _builder_emit(T.cuda.warp_sync())
                                with T.Else():
                                    with T.If(warp_idx == spec_warp_start + 3):
                                        with T.Then():
                                            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(56))
                                        with T.Else():
                                            with T.If(warp_idx < spec_warp_start):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.ptx.setmaxnreg.inc.sync.aligned.u32(224)
                                                    )
                                                    math_thread_idx = _builder_scalar(
                                                        "math_thread_idx",
                                                        warp_idx_u32 * T.uint32(32) + lane_idx_u32,
                                                        "uint32",
                                                    )
                                                    accum = _builder_assign(
                                                        "accum",
                                                        T.alloc_local((num_heads,), "float32"),
                                                        locals().get("accum", _BUILDER_MISSING),
                                                    )
                                                    cached_weights = _builder_assign(
                                                        "cached_weights",
                                                        T.alloc_local(
                                                            (block_q, num_heads), "float32"
                                                        ),
                                                        locals().get(
                                                            "cached_weights", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    # Per-q-row logits base offset (= q_row * logits_stride): invariant across
                                                    # the kv loop, so compute once per q block.
                                                    # Single u64 row base; ptxas spilled the block_q-wide u64 array,
                                                    # costing an LDL reload per clamped store.
                                                    q_row_off_base = _builder_assign(
                                                        "q_row_off_base",
                                                        T.alloc_local((1,), "uint64"),
                                                        locals().get(
                                                            "q_row_off_base", _BUILDER_MISSING
                                                        ),
                                                    )
                                                    q_stage_idx = _builder_scalar(
                                                        "q_stage_idx", T.uint32(0), "uint32"
                                                    )
                                                    q_phase = _builder_scalar(
                                                        "q_phase", T.uint32(0), "uint32"
                                                    )
                                                    kv_stage_idx = _builder_scalar(
                                                        "kv_stage_idx", T.uint32(0), "uint32"
                                                    )
                                                    kv_phase = _builder_scalar(
                                                        "kv_phase", T.uint32(0), "uint32"
                                                    )
                                                    tmem_stage_idx = _builder_scalar(
                                                        "tmem_stage_idx",
                                                        T.cast(warpgroup_idx, "uint32"),
                                                        "uint32",
                                                    )
                                                    tmem_phase = _builder_scalar(
                                                        "tmem_phase", T.uint32(0), "uint32"
                                                    )
                                                    q_idx = _builder_scalar(
                                                        "q_idx", sm_idx_u32, "uint32"
                                                    )
                                                    with T.While(q_idx < num_q_blocks):
                                                        _builder_emit(load_schedule(q_idx))
                                                        kv_start = _builder_scalar(
                                                            "kv_start", schedule_result[0], "uint32"
                                                        )
                                                        num_kv_blocks = _builder_scalar(
                                                            "num_kv_blocks",
                                                            schedule_result[1],
                                                            "uint32",
                                                        )
                                                        _builder_emit(
                                                            q_pipe.full.wait(q_stage_idx, q_phase)
                                                        )
                                                        with T.If(num_kv_blocks > T.uint32(0)):
                                                            with T.Then():
                                                                with T.unroll(
                                                                    0, block_q
                                                                ) as weight_i:
                                                                    IRBuilder.name(
                                                                        "weight_i", weight_i
                                                                    )
                                                                    with T.unroll(
                                                                        0, num_heads // 4
                                                                    ) as weight_j:
                                                                        IRBuilder.name(
                                                                            "weight_j", weight_j
                                                                        )
                                                                        weight_col = _builder_assign(
                                                                            "weight_col",
                                                                            weight_j * 4,
                                                                            locals().get(
                                                                                "weight_col",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx.ld.shared.v4.f32(
                                                                                cached_weights[
                                                                                    weight_i,
                                                                                    weight_col,
                                                                                ],
                                                                                cached_weights[
                                                                                    weight_i,
                                                                                    weight_col + 1,
                                                                                ],
                                                                                cached_weights[
                                                                                    weight_i,
                                                                                    weight_col + 2,
                                                                                ],
                                                                                cached_weights[
                                                                                    weight_i,
                                                                                    weight_col + 3,
                                                                                ],
                                                                                smem_weights.ptr_to(
                                                                                    [
                                                                                        q_stage_idx,
                                                                                        weight_i,
                                                                                        weight_col,
                                                                                    ]
                                                                                ),
                                                                            )
                                                                        )
                                                                T.buffer_store(
                                                                    q_row_off_base,
                                                                    T.cast(
                                                                        q_idx * T.uint32(block_q),
                                                                        "uint64",
                                                                    )
                                                                    * T.cast(
                                                                        logits_stride, "uint64"
                                                                    ),
                                                                    [0],
                                                                )
                                                                # Publish the generic-proxy weight reads before this consumer
                                                                # eventually releases the Q stage for a subsequent TMA overwrite.
                                                                _builder_emit(
                                                                    T.ptx.fence.proxy.async_.shared__cta()
                                                                )
                                                                kv_offset = _builder_scalar(
                                                                    "kv_offset",
                                                                    kv_start + math_thread_idx,
                                                                    "uint32",
                                                                )
                                                                kv_idx = _builder_scalar(
                                                                    "kv_idx", T.uint32(0), "uint32"
                                                                )
                                                                with T.While(
                                                                    kv_idx < num_kv_blocks
                                                                ):
                                                                    _builder_emit(
                                                                        kv_pipe.full.wait(
                                                                            kv_stage_idx, kv_phase
                                                                        )
                                                                    )
                                                                    scale_kv = (
                                                                        _builder_alloc_scalar(
                                                                            "scale_kv", "float32"
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        T.ptx.ld.shared.f32(
                                                                            scale_kv,
                                                                            smem_kv_scales.ptr_to(
                                                                                [
                                                                                    kv_stage_idx,
                                                                                    math_thread_idx,
                                                                                ]
                                                                            ),
                                                                        )
                                                                    )
                                                                    _builder_emit(
                                                                        tmem_pipe.full.wait(
                                                                            tmem_stage_idx,
                                                                            tmem_phase,
                                                                        )
                                                                    )
                                                                    # Release the kv stage only after the tmem accumulator is
                                                                    # ready, and fence the generic-proxy scale_kv read first.
                                                                    _builder_emit(
                                                                        T.ptx.fence.proxy.async_.shared__cta()
                                                                    )
                                                                    _builder_emit(
                                                                        kv_pipe.empty.arrive(
                                                                            kv_stage_idx
                                                                        )
                                                                    )
                                                                    tmem_stage_base = (
                                                                        _builder_scalar(
                                                                            "tmem_stage_base",
                                                                            tmem_stage_idx
                                                                            * T.uint32(umma_n),
                                                                            "uint32",
                                                                        )
                                                                    )
                                                                    with T.unroll(
                                                                        0, block_q
                                                                    ) as q_inner_i:
                                                                        IRBuilder.name(
                                                                            "q_inner_i", q_inner_i
                                                                        )
                                                                        tmem_addr = _builder_scalar(
                                                                            "tmem_addr",
                                                                            tmem_stage_base
                                                                            + T.uint32(
                                                                                q_inner_i
                                                                                * num_heads
                                                                            ),
                                                                            "uint32",
                                                                        )
                                                                        # TMEM->register read as two 32x32b chunks (x16 for 32 heads,
                                                                        # x32 for 64), with a tcgen05.wait.ld after each issue.
                                                                        tmem_addr_hi = (
                                                                            _builder_scalar(
                                                                                "tmem_addr_hi",
                                                                                tmem_addr
                                                                                + T.uint32(
                                                                                    num_heads // 2
                                                                                ),
                                                                                "uint32",
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx[tcgen05_ld](
                                                                                *[
                                                                                    accum[head_i]
                                                                                    for head_i in range(
                                                                                        num_heads
                                                                                        // 2
                                                                                    )
                                                                                ],
                                                                                T.cuda.get_tmem_addr(
                                                                                    T.uint32(0),
                                                                                    0,
                                                                                    tmem_addr,
                                                                                ),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx.tcgen05.wait__ld.sync.aligned()
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx[tcgen05_ld](
                                                                                *[
                                                                                    accum[
                                                                                        num_heads
                                                                                        // 2
                                                                                        + head_i
                                                                                    ]
                                                                                    for head_i in range(
                                                                                        num_heads
                                                                                        // 2
                                                                                    )
                                                                                ],
                                                                                T.cuda.get_tmem_addr(
                                                                                    T.uint32(0),
                                                                                    0,
                                                                                    tmem_addr_hi,
                                                                                ),
                                                                            )
                                                                        )
                                                                        _builder_emit(
                                                                            T.ptx.tcgen05.wait__ld.sync.aligned()
                                                                        )
                                                                        reduced = _builder_scalar(
                                                                            "reduced",
                                                                            wrelu_reduce(
                                                                                accum,
                                                                                cached_weights,
                                                                                q_inner_i,
                                                                            ),
                                                                            "float32",
                                                                        )
                                                                        result = _builder_assign(
                                                                            "result",
                                                                            T.cast(
                                                                                scale_kv * reduced,
                                                                                logits_tir_dtype,
                                                                            ),
                                                                            locals().get(
                                                                                "result",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                        q_offset = _builder_scalar(
                                                                            "q_offset",
                                                                            q_row_off_base[0]
                                                                            + T.cast(
                                                                                q_inner_i, "uint64"
                                                                            )
                                                                            * T.cast(
                                                                                logits_stride,
                                                                                "uint64",
                                                                            ),
                                                                            "uint64",
                                                                        )
                                                                        if config.compressed_logits:
                                                                            # Unconditional store with the column clamped into the
                                                                            # row's stride padding (a range guard becomes a BSSY/BRA region).
                                                                            rel_kv = (
                                                                                _builder_scalar(
                                                                                    "rel_kv",
                                                                                    kv_offset
                                                                                    - seq_k_start[
                                                                                        q_inner_i
                                                                                    ],
                                                                                    "uint32",
                                                                                )
                                                                            )
                                                                            col = _builder_scalar(
                                                                                "col",
                                                                                T.min(
                                                                                    rel_kv,
                                                                                    logits_stride
                                                                                    - T.uint32(1),
                                                                                ),
                                                                                "uint32",
                                                                            )
                                                                            _builder_emit(
                                                                                store_logits(
                                                                                    q_offset
                                                                                    + T.cast(
                                                                                        col,
                                                                                        "uint64",
                                                                                    ),
                                                                                    result,
                                                                                )
                                                                            )
                                                                        else:
                                                                            _builder_emit(
                                                                                store_logits(
                                                                                    q_offset
                                                                                    + T.cast(
                                                                                        kv_offset,
                                                                                        "uint64",
                                                                                    ),
                                                                                    result,
                                                                                )
                                                                            )
                                                                    # Release this tmem stage once per kv block AFTER the token loop;
                                                                    # inside the last token ptxas fuses it with the compressed guard branch.
                                                                    _builder_emit(
                                                                        tmem_pipe.empty.arrive(
                                                                            tmem_stage_idx
                                                                        )
                                                                    )
                                                                    kv_idx = _builder_assign(
                                                                        "kv_idx",
                                                                        kv_idx + T.uint32(1),
                                                                        locals().get(
                                                                            "kv_idx",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    kv_offset = _builder_assign(
                                                                        "kv_offset",
                                                                        kv_offset
                                                                        + T.uint32(block_kv),
                                                                        locals().get(
                                                                            "kv_offset",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    kv_stage_idx = _builder_assign(
                                                                        "kv_stage_idx",
                                                                        kv_stage_idx + T.uint32(1),
                                                                        locals().get(
                                                                            "kv_stage_idx",
                                                                            _BUILDER_MISSING,
                                                                        ),
                                                                    )
                                                                    with T.If(
                                                                        kv_stage_idx
                                                                        >= T.uint32(num_kv_stages)
                                                                    ):
                                                                        with T.Then():
                                                                            kv_stage_idx = _builder_assign(
                                                                                "kv_stage_idx",
                                                                                kv_stage_idx
                                                                                - T.uint32(
                                                                                    num_kv_stages
                                                                                ),
                                                                                locals().get(
                                                                                    "kv_stage_idx",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            kv_phase = _builder_assign(
                                                                                "kv_phase",
                                                                                kv_phase
                                                                                ^ T.uint32(1),
                                                                                locals().get(
                                                                                    "kv_phase",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                    tmem_stage_idx = (
                                                                        _builder_assign(
                                                                            "tmem_stage_idx",
                                                                            tmem_stage_idx
                                                                            + T.uint32(
                                                                                num_math_warpgroups
                                                                            ),
                                                                            locals().get(
                                                                                "tmem_stage_idx",
                                                                                _BUILDER_MISSING,
                                                                            ),
                                                                        )
                                                                    )
                                                                    with T.If(
                                                                        tmem_stage_idx
                                                                        >= T.uint32(num_tmem_stages)
                                                                    ):
                                                                        with T.Then():
                                                                            tmem_stage_idx = _builder_assign(
                                                                                "tmem_stage_idx",
                                                                                tmem_stage_idx
                                                                                - T.uint32(
                                                                                    num_tmem_stages
                                                                                ),
                                                                                locals().get(
                                                                                    "tmem_stage_idx",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                                            tmem_phase = _builder_assign(
                                                                                "tmem_phase",
                                                                                tmem_phase
                                                                                ^ T.uint32(1),
                                                                                locals().get(
                                                                                    "tmem_phase",
                                                                                    _BUILDER_MISSING,
                                                                                ),
                                                                            )
                                                        _builder_emit(
                                                            q_pipe.empty.arrive(q_stage_idx)
                                                        )
                                                        q_idx = _builder_assign(
                                                            "q_idx",
                                                            q_idx + T.uint32(config.num_sms),
                                                            locals().get("q_idx", _BUILDER_MISSING),
                                                        )
                                                        q_stage_idx = _builder_assign(
                                                            "q_stage_idx",
                                                            q_stage_idx + T.uint32(1),
                                                            locals().get(
                                                                "q_stage_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        with T.If(
                                                            q_stage_idx >= T.uint32(num_q_stages)
                                                        ):
                                                            with T.Then():
                                                                q_stage_idx = _builder_assign(
                                                                    "q_stage_idx",
                                                                    q_stage_idx
                                                                    - T.uint32(num_q_stages),
                                                                    locals().get(
                                                                        "q_stage_idx",
                                                                        _BUILDER_MISSING,
                                                                    ),
                                                                )
                                                                q_phase = _builder_assign(
                                                                    "q_phase",
                                                                    q_phase ^ T.uint32(1),
                                                                    locals().get(
                                                                        "q_phase", _BUILDER_MISSING
                                                                    ),
                                                                )
                                                    _builder_emit(
                                                        named_barrier_sync_8(
                                                            T.uint32(num_math_threads)
                                                        )
                                                    )
                                                    with T.If(warp_idx == 0):
                                                        with T.Then():
                                                            _builder_emit(
                                                                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                                                                    T.uint32(0),
                                                                    T.uint32(num_tmem_cols),
                                                                )
                                                            )

    return (
        builder.get()
        .with_attr("tirx.persistent_kernel", True)
        .with_attr(
            "tirx.kernel_launch_params",
            [
                "blockIdx.x",
                "threadIdx.x",
                "tirx.use_programtic_dependent_launch",
                "tirx.use_dyn_shared_memory",
            ],
        )
    )


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
    return {"executable": executable, "logits": logits}


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsFP8Config = data["config"]
    executable = invocation["executable"]
    logits = invocation["logits"]
    # Raw GMEM buffers: the host prologue builds runtime TensorMaps, then explicit TMA
    # moves Q/KV as uint8 (fp8 e4m3 bytes); KV scales and weights stay float32.
    kv_fp8, kv_scales = data["kv_in"]
    q_gmem = (
        data["q_in"].view(torch.uint8).reshape(config.seq_len * config.num_heads, config.head_dim)
    )
    kv_gmem = kv_fp8.view(torch.uint8)
    _prepare_global_barrier(executable)
    executable.mod(
        config.seq_len,
        config.seq_len_kv,
        data["max_seqlen_k"],
        logits.stride(0),
        data["cu_seq_len_k_start"],
        data["cu_seq_len_k_end"],
        logits,
        q_gmem,
        kv_gmem,
        kv_scales,
        data["weights"],
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
